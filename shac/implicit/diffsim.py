"""Graph-captured differentiable MJWarp simulator exposed to PyTorch.

One control step applies a zero-order-hold control for ``substeps`` physics steps
of size ``timestep``. The forward pass and the analytic backward of a single
physics step are each captured once as CUDA graphs, following the checkpointing
scheme of ``contrib/diffsim/_rollout.py`` in MJWarp PR #1535:

* forward: replay the forward graph ``substeps`` times, saving the step-input
  state (qpos, qvel, qacc_warmstart) before every physics step;
* backward: for each physics step in reverse, restore the saved state, seed
  ``d_out.qpos.grad`` / ``d_out.qvel.grad`` and replay the backward graph, which
  recomputes the step under a ``wp.Tape`` and runs the analytic adjoint.

PyTorch and Warp share one CUDA stream, so no host synchronisation is needed
between the two frameworks.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import mujoco
import numpy as np
import torch
import warp as wp

import mujoco_warp as mjw
from mujoco_warp._src import forward as mjw_forward

mjw.enable_grad()

INTEGRATORS = {
    # Explicit semi-implicit Euler with joint damping treated explicitly: the
    # configuration used by the earlier SHAC study (mjDSBL_EULERDAMP set).
    "euler_explicit": (mujoco.mjtIntegrator.mjINT_EULER, True),
    # MuJoCo's default Euler: joint damping is integrated implicitly.
    "euler": (mujoco.mjtIntegrator.mjINT_EULER, False),
    # Implicit in all velocity-dependent smooth forces except Coriolis terms.
    "implicitfast": (mujoco.mjtIntegrator.mjINT_IMPLICITFAST, False),
}


@dataclasses.dataclass
class SimConfig:
    xml: str
    integrator: str = "implicitfast"
    timestep: float = 0.01
    substeps: int = 2
    cone: str = "elliptic"
    iterations: int = 100
    ls_iterations: int = 50
    tolerance: float | None = None
    nconmax: int | None = None
    njmax: int | None = None
    keyframe: int | None = None
    extra_disableflags: int = 0
    # If set, overrides the constraint time constant (s) of all contacts and joint limits.
    solref_timeconst: float | None = None

    @property
    def control_dt(self) -> float:
        return self.timestep * self.substeps


def load_mj_model(cfg: SimConfig) -> mujoco.MjModel:
    model = mujoco.MjModel.from_xml_path(str(cfg.xml))
    integrator, disable_eulerdamp = INTEGRATORS[cfg.integrator]
    model.opt.integrator = integrator
    model.opt.timestep = cfg.timestep
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.cone = {
        "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC,
        "pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL,
    }[cfg.cone]
    model.opt.iterations = cfg.iterations
    model.opt.ls_iterations = cfg.ls_iterations
    if cfg.tolerance is not None:
        model.opt.tolerance = cfg.tolerance
    model.opt.disableflags |= cfg.extra_disableflags
    if cfg.solref_timeconst is not None:
        model.geom_solref[:, 0] = cfg.solref_timeconst
        model.jnt_solref[:, 0] = cfg.solref_timeconst
        if model.npair:
            model.pair_solref[:, 0] = cfg.solref_timeconst
    flag = int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
    if disable_eulerdamp:
        model.opt.disableflags |= flag
    else:
        model.opt.disableflags &= ~flag
    return model


def _seed_array(shape, dtype=wp.float32):
    return wp.zeros(shape, dtype=dtype, requires_grad=False)


class DiffSim:
    """Batched, graph-captured MJWarp simulator with an analytic backward."""

    def __init__(self, cfg: SimConfig, nworld: int, device: str = "cuda:0", graph: bool = True,
                 backward: bool = True):
        self.has_backward = backward
        self.cfg = cfg
        self.nworld = nworld
        self.mjm = load_mj_model(cfg)
        self.mjd = mujoco.MjData(self.mjm)
        if cfg.keyframe is not None:
            mujoco.mj_resetDataKeyframe(self.mjm, self.mjd, cfg.keyframe)
        mujoco.mj_forward(self.mjm, self.mjd)
        self.nq, self.nv, self.nu = self.mjm.nq, self.mjm.nv, self.mjm.nu
        self.device = wp.get_device(device)
        self.torch_device = torch.device(str(self.device))

        # Share a single non-default stream between Warp and PyTorch.
        self.stream = wp.get_stream(self.device)
        torch.cuda.set_stream(wp.stream_to_torch(self.stream))

        with wp.ScopedDevice(self.device):
            self.m = mjw.put_model(self.mjm)
            # Overflows (buffer and solver-iteration limits) are recorded as metrics instead of printed.
            self.m.opt.warn_overflow = 0
            kwargs = dict(nworld=nworld, nconmax=cfg.nconmax, njmax=cfg.njmax)
            self.d = mjw.put_data(self.mjm, self.mjd, **kwargs)
            self.d_out = mjw.put_data(self.mjm, self.mjd, **kwargs)
            for data in (self.d, self.d_out):
                data.qpos.requires_grad = True
                data.qvel.requires_grad = True
            self.d.ctrl.requires_grad = True
            self.bc = mjw.create_backward_context(self.m, self.d) if backward else None
            self.seed_qpos = _seed_array(self.d.qpos.shape)
            self.seed_qvel = _seed_array(self.d.qvel.shape)

        # Zero-copy torch views of the Warp state.
        self.t_qpos = wp.to_torch(self.d.qpos, requires_grad=False)
        self.t_qvel = wp.to_torch(self.d.qvel, requires_grad=False)
        self.t_ctrl = wp.to_torch(self.d.ctrl, requires_grad=False)
        self.t_warm = wp.to_torch(self.d.qacc_warmstart, requires_grad=False)
        self.t_time = wp.to_torch(self.d.time, requires_grad=False)
        # Overflow bits are OR-ed into the step output and are not copied back.
        self.t_overflow = wp.to_torch(self.d_out.overflow, requires_grad=False)
        self.t_nacon = wp.to_torch(self.d.nacon, requires_grad=False)
        self.t_contact_dist = wp.to_torch(self.d.contact.dist, requires_grad=False)
        self.t_contact_world = wp.to_torch(self.d.contact.worldid, requires_grad=False)
        self.t_gqpos = wp.to_torch(self.d.qpos.grad, requires_grad=False)
        self.t_gqvel = wp.to_torch(self.d.qvel.grad, requires_grad=False)
        self.t_gctrl = wp.to_torch(self.d.ctrl.grad, requires_grad=False)
        self.t_seed_qpos = wp.to_torch(self.seed_qpos, requires_grad=False)
        self.t_seed_qvel = wp.to_torch(self.seed_qvel, requires_grad=False)
        # The PR's control VJP ignores actuator force saturation; mask it here.
        self.t_act_force = wp.to_torch(self.d_out.actuator_force, requires_grad=False)
        fl = self.mjm.actuator_forcelimited.astype(bool)
        self.force_limited = torch.tensor(fl, device=self.torch_device)
        self.force_lo = torch.tensor(self.mjm.actuator_forcerange[:, 0], dtype=torch.float32, device=self.torch_device)
        self.force_hi = torch.tensor(self.mjm.actuator_forcerange[:, 1], dtype=torch.float32, device=self.torch_device)
        self.mask_saturation = bool(fl.any())

        self.init_qpos = torch.tensor(self.mjd.qpos, dtype=torch.float32, device=self.torch_device)
        self.ctrl_low = torch.tensor(self.mjm.actuator_ctrlrange[:, 0], dtype=torch.float32, device=self.torch_device)
        self.ctrl_high = torch.tensor(self.mjm.actuator_ctrlrange[:, 1], dtype=torch.float32, device=self.torch_device)
        self.ctrl_limited = torch.tensor(self.mjm.actuator_ctrllimited.astype(bool), device=self.torch_device)

        self.forward_graph = None
        self.backward_graph = None
        self.nonfinite_grad_events = 0
        # Optional per-world bound on the norm of the state adjoint passed to the previous
        # control step, as a multiple of the median norm over worlds (gradient clipping
        # through time); None disables it.
        self.state_grad_clip: float | None = None
        self.state_grad_clip_events = 0
        if graph and self.device.is_cuda:
            self._capture()

    # ------------------------------------------------------------------ physics
    def _physics_step(self):
        mjw.step(self.m, self.d, self.d_out)
        mjw_forward._copy_state(self.d_out, self.d)

    def _backward_step(self):
        tape = wp.Tape()
        with mjw.backward_context(self.bc):
            with tape:
                mjw.step(self.m, self.d, self.d_out)
            tape.zero()
            wp.copy(self.d_out.qpos.grad, self.seed_qpos)
            wp.copy(self.d_out.qvel.grad, self.seed_qvel)
            tape.backward()
        return tape

    def _capture(self):
        with wp.ScopedDevice(self.device):
            # Warm up (compiles kernels and allocates lazily created buffers).
            self._physics_step()
            if self.has_backward:
                self._backward_step()
            wp.synchronize_device(self.device)
            self._reset_to_init()
            with wp.ScopedCapture(device=self.device) as capture:
                self._physics_step()
            self.forward_graph = capture.graph
            if self.has_backward:
                self._reset_to_init()
                with wp.ScopedCapture(device=self.device) as capture:
                    self._keep_tape = self._backward_step()
                self.backward_graph = capture.graph
            self._reset_to_init()

    def _reset_to_init(self):
        self.t_qpos.copy_(self.init_qpos.expand(self.nworld, -1))
        self.t_qvel.zero_()
        self.t_ctrl.zero_()
        self.t_warm.zero_()

    def forward_physics(self):
        if self.forward_graph is not None:
            wp.capture_launch(self.forward_graph, stream=self.stream)
        else:
            self._physics_step()

    def backward_physics(self):
        if self.backward_graph is not None:
            wp.capture_launch(self.backward_graph, stream=self.stream)
        else:
            self._backward_step()

    # ------------------------------------------------------------------ control
    OVERFLOW_BITS = {"nefc": 1 << 0, "njmax_nnz": 1 << 1, "broadphase": 1 << 2, "narrowphase": 1 << 3}

    def overflowed(self) -> bool:
        """True if any buffer overflow occurred since the last ``clear_overflow``."""
        mask = sum(self.OVERFLOW_BITS.values())
        return bool(((self.t_overflow & mask) != 0).any().item())

    def overflow_stats(self) -> dict:
        import mujoco_warp._src.types as T
        bits = dict(self.OVERFLOW_BITS, iterations=int(T.OverflowType.ITERATIONS),
                    ls_iterations=int(T.OverflowType.LS_ITERATIONS))
        return {k: float(((self.t_overflow & b) != 0).float().mean().item()) for k, b in bits.items()}

    def clear_overflow(self):
        self.t_overflow.zero_()

    def action_to_ctrl(self, action: torch.Tensor, position_control: bool, home: torch.Tensor | None = None,
                       scale: float = 0.5) -> torch.Tensor:
        """Maps normalised actions in [-1, 1] to actuator controls."""
        if position_control:
            return self.clamp_ctrl(home + scale * action)
        mid = 0.5 * (self.ctrl_low + self.ctrl_high)
        half = 0.5 * (self.ctrl_high - self.ctrl_low)
        return mid + half * action

    def clamp_ctrl(self, ctrl: torch.Tensor) -> torch.Tensor:
        return torch.where(self.ctrl_limited, ctrl.clamp(self.ctrl_low, self.ctrl_high), ctrl)

    def step_nograd(self, qpos, qvel, ctrl, warm=None):
        """Advances one control step without recording checkpoints."""
        self.t_qpos.copy_(qpos)
        self.t_qvel.copy_(qvel)
        self.t_ctrl.copy_(ctrl)
        if warm is not None:
            self.t_warm.copy_(warm)
        for _ in range(self.cfg.substeps):
            self.forward_physics()
        return self.t_qpos.clone(), self.t_qvel.clone(), self.t_warm.clone()

    def step(self, qpos, qvel, ctrl, warm):
        """Differentiable control step; returns (qpos', qvel', warm')."""
        qpos_n, qvel_n, warm_n = _ControlStep.apply(qpos, qvel, ctrl, warm, self)
        return qpos_n, qvel_n, warm_n


class _ControlStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qpos, qvel, ctrl, warm, sim: DiffSim):
        k = sim.cfg.substeps
        cq = torch.empty((k, *qpos.shape), dtype=qpos.dtype, device=qpos.device)
        cv = torch.empty((k, *qvel.shape), dtype=qvel.dtype, device=qvel.device)
        cw = torch.empty((k, *warm.shape), dtype=warm.dtype, device=warm.device)
        sim.t_qpos.copy_(qpos)
        sim.t_qvel.copy_(qvel)
        sim.t_ctrl.copy_(ctrl)
        sim.t_warm.copy_(warm)
        for i in range(k):
            cq[i].copy_(sim.t_qpos)
            cv[i].copy_(sim.t_qvel)
            cw[i].copy_(sim.t_warm)
            sim.forward_physics()
        ctx.sim = sim
        ctx.save_for_backward(cq, cv, cw, ctrl.detach().clone())
        warm_out = sim.t_warm.clone()
        ctx.mark_non_differentiable(warm_out)
        return sim.t_qpos.clone(), sim.t_qvel.clone(), warm_out

    @staticmethod
    def backward(ctx, g_qpos, g_qvel, _g_warm):
        sim: DiffSim = ctx.sim
        cq, cv, cw, ctrl = ctx.saved_tensors
        if g_qpos is None:
            g_qpos = torch.zeros_like(cq[0])
        if g_qvel is None:
            g_qvel = torch.zeros_like(cv[0])
        sim.t_seed_qpos.copy_(g_qpos)
        sim.t_seed_qvel.copy_(g_qvel)
        g_ctrl = torch.zeros_like(ctrl)
        sim.t_ctrl.copy_(ctrl)
        for i in reversed(range(sim.cfg.substeps)):
            sim.t_qpos.copy_(cq[i])
            sim.t_qvel.copy_(cv[i])
            sim.t_warm.copy_(cw[i])
            sim.backward_physics()
            if sim.mask_saturation:
                f = sim.t_act_force
                saturated = sim.force_limited & ((f <= sim.force_lo) | (f >= sim.force_hi))
                g_ctrl += torch.where(saturated, 0.0, sim.t_gctrl)
            else:
                g_ctrl += sim.t_gctrl
            sim.t_seed_qpos.copy_(sim.t_gqpos)
            sim.t_seed_qvel.copy_(sim.t_gqvel)
        gq, gv = sim.t_seed_qpos.clone(), sim.t_seed_qvel.clone()
        # Worlds that were replaced after a non-finite state may carry NaN adjoints.
        finite = torch.isfinite(gq).all() & torch.isfinite(gv).all() & torch.isfinite(g_ctrl).all()
        if not bool(finite):
            sim.nonfinite_grad_events += 1
            gq, gv, g_ctrl = (torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) for x in (gq, gv, g_ctrl))
        if sim.state_grad_clip is not None:
            norm = torch.sqrt(gq.square().sum(1) + gv.square().sum(1))
            active = norm[norm > 0]
            bound = sim.state_grad_clip * (active.median() if active.numel() else norm.new_tensor(0.0))
            scale = (bound / norm.clamp(min=1e-30)).clamp(max=1.0)
            sim.state_grad_clip_events += int((scale < 1.0).sum())
            gq, gv = gq * scale[:, None], gv * scale[:, None]
        return gq, gv, g_ctrl, None, None


def repo_path(*parts) -> Path:
    return Path(__file__).resolve().parent.joinpath(*parts)
