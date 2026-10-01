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
from mujoco_warp._src import adjoint as mjw_adjoint
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
    # Optional contact softening: solimp (d0, dmax, width, midpoint, power) and margin (m) for all geoms.
    geom_solimp: tuple | None = None
    geom_margin: float | None = None
    # Backward-only contact smoothing: when set, the backward recomputes each step with a model whose
    # geom solimp / margin are replaced by these values (forward physics unchanged; recompute mode only).
    grad_solimp: tuple | None = None
    grad_margin: float | None = None
    # Optional name of a model-editing function registered in robots.MODEL_EDITS
    # (applied to the compiled MjModel before any other override).
    model_edit: str | None = None

    @property
    def control_dt(self) -> float:
        return self.timestep * self.substeps


def load_mj_model(cfg: SimConfig) -> mujoco.MjModel:
    model = mujoco.MjModel.from_xml_path(str(cfg.xml))
    if cfg.model_edit:
        from robots import apply_edits
        apply_edits(model, cfg.model_edit)
    if cfg.geom_solimp is not None:
        model.geom_solimp[:] = np.asarray(cfg.geom_solimp, dtype=np.float64)
        if model.npair:
            model.pair_solimp[:] = np.asarray(cfg.geom_solimp, dtype=np.float64)
    if cfg.geom_margin is not None:
        model.geom_margin[:] = cfg.geom_margin
        if model.npair:
            model.pair_margin[:] = cfg.geom_margin
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
            self.m_grad = self.m
            if backward and (cfg.grad_solimp is not None or cfg.grad_margin is not None):
                gcfg = dataclasses.replace(cfg, geom_solimp=cfg.grad_solimp or cfg.geom_solimp,
                                           geom_margin=cfg.grad_margin if cfg.grad_margin is not None else cfg.geom_margin,
                                           grad_solimp=None, grad_margin=None)
                self.m_grad = mjw.put_model(load_mj_model(gcfg))
                self.m_grad.opt.warn_overflow = 0
            self.bc = mjw.create_backward_context(self.m_grad, self.d) if backward else None
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
        # Contacts of the last physics step live in d_out (step() runs forward on d_out).
        self.t_contact_geom = wp.to_torch(self.d_out.contact.geom, requires_grad=False)
        self.t_out_contact_dist = wp.to_torch(self.d_out.contact.dist, requires_grad=False)
        self.t_out_contact_world = wp.to_torch(self.d_out.contact.worldid, requires_grad=False)
        self.t_out_nacon = wp.to_torch(self.d_out.nacon, requires_grad=False)
        self.base_contact = (self.t_contact_geom, self.t_out_contact_dist, self.t_out_contact_world, self.t_out_nacon)
        self.cur_contact = self.base_contact  # contacts of the most recent physics step
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
        # When set, the backward pass never synchronises with the host; events are
        # counted in device tensors instead of Python integers.
        self.sync_free = False
        self.nonfinite_grad_worlds = torch.zeros((), dtype=torch.long, device=self.torch_device)
        self.state_grad_clip_worlds = torch.zeros((), dtype=torch.long, device=self.torch_device)
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
                mjw.step(self.m_grad, self.d, self.d_out)
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
        self.cur_contact = self.base_contact
        for _ in range(self.cfg.substeps):
            self.forward_physics()
        return self.t_qpos.clone(), self.t_qvel.clone(), self.t_warm.clone()

    def step(self, qpos, qvel, ctrl, warm):
        """Differentiable control step; returns (qpos', qvel', warm')."""
        if self.window is not None:
            return _WindowControlStep.apply(qpos, qvel, ctrl, warm, self)
        qpos_n, qvel_n, warm_n = _ControlStep.apply(qpos, qvel, ctrl, warm, self)
        return qpos_n, qvel_n, warm_n

    # ------------------------------------------------------------------ stored-forward window
    # With ``enable_window(T)``, each of the T control steps of a rollout window owns K+1 Data
    # slots; physics step k of control step t runs step(S[t][k] -> S[t][k+1]) from its own captured
    # graph, so all forward results stay resident, and the backward calls the analytic
    # step_backward(S[t][k], S[t][k+1]) directly instead of re-running the forward under a tape.
    window = None

    def enable_window(self, T: int):
        assert self.has_backward and self.device.is_cuda
        assert self.m_grad is self.m, "backward-only smoothing needs the recompute backward"

        K = self.cfg.substeps
        kwargs = dict(nworld=self.nworld, nconmax=self.cfg.nconmax, njmax=self.cfg.njmax)
        self.window = T
        self.win_t = 0
        self.slots, self.win_fwd, self.win_bwd = [], [], []
        with wp.ScopedDevice(self.device):
            for t in range(T):
                row = [mjw.put_data(self.mjm, self.mjd, **kwargs) for _ in range(K + 1)]
                for d in row:  # allocate every gradient buffer before capture (as input and output)
                    mjw_adjoint.step_backward_arrays(d, d)
                self.slots.append(row)
            # Warm up once (allocations happen outside capture), then capture every slot pair.
            s0 = self.slots[0]
            self._win_copy_in(s0[0], self.init_qpos.expand(self.nworld, -1), torch.zeros_like(self.t_qvel),
                              torch.zeros_like(self.t_ctrl), torch.zeros_like(self.t_warm))
            mjw.step(self.m, s0[0], s0[1])
            with mjw.backward_context(self.bc):
                mjw_adjoint.step_backward(self.m, s0[0], s0[1], self.bc)
            wp.synchronize_device(self.device)
            for t in range(T):
                fw, bw = [], []
                for k in range(K):
                    a, b = self.slots[t][k], self.slots[t][k + 1]
                    with wp.ScopedCapture(device=self.device) as cap:
                        mjw.step(self.m, a, b)
                    fw.append(cap.graph)
                    with wp.ScopedCapture(device=self.device) as cap:
                        with mjw.backward_context(self.bc):
                            mjw_adjoint.step_backward(self.m, a, b, self.bc)
                    bw.append(cap.graph)
                self.win_fwd.append(fw)
                self.win_bwd.append(bw)
        # Torch views of the slots.
        tv = lambda arr: wp.to_torch(arr, requires_grad=False)  # noqa: E731
        self.win_views = []
        for row in self.slots:
            self.win_views.append([dict(
                qpos=tv(d.qpos), qvel=tv(d.qvel), ctrl=tv(d.ctrl), warm=tv(d.qacc_warmstart),
                gqpos=tv(d.qpos.grad), gqvel=tv(d.qvel.grad), gctrl=tv(d.ctrl.grad),
                force=tv(d.actuator_force), overflow=tv(d.overflow),
                contact=(tv(d.contact.geom), tv(d.contact.dist), tv(d.contact.worldid), tv(d.nacon)),
                grads=[tv(x.grad) for x in mjw_adjoint.step_backward_arrays(d, d)
                       if x.grad is not None and x is not d.qpos and x is not d.qvel]) for d in row])

        # Gradient buffers to clear before each backward step (one fused launch).
        self.win_zero = [[views[k]["grads"] + views[k + 1]["grads"] + [views[k]["gqpos"], views[k]["gqvel"]]
                          for k in range(self.cfg.substeps)] for views in self.win_views]

    def begin_window(self):
        """Starts a new rollout window (the next differentiable step uses slot 0)."""
        self.win_t = 0

    @staticmethod
    def _win_copy_in(d, qpos, qvel, ctrl, warm):
        wp.to_torch(d.qpos, requires_grad=False).copy_(qpos)
        wp.to_torch(d.qvel, requires_grad=False).copy_(qvel)
        wp.to_torch(d.ctrl, requires_grad=False).copy_(ctrl)
        wp.to_torch(d.qacc_warmstart, requires_grad=False).copy_(warm)


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
        sim.cur_contact = sim.base_contact
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
        if sim.sync_free:
            # Count on the device and sanitise unconditionally (no host synchronisation).
            finite = torch.isfinite(gq).all(1) & torch.isfinite(gv).all(1) & torch.isfinite(g_ctrl).all(1)
            sim.nonfinite_grad_worlds += (~finite).sum()
            gq, gv, g_ctrl = (torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) for x in (gq, gv, g_ctrl))
        else:
            finite = torch.isfinite(gq).all() & torch.isfinite(gv).all() & torch.isfinite(g_ctrl).all()
            if not bool(finite):
                sim.nonfinite_grad_events += 1
                gq, gv, g_ctrl = (torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) for x in (gq, gv, g_ctrl))
        if sim.state_grad_clip is not None:
            norm = torch.sqrt(gq.square().sum(1) + gv.square().sum(1))
            active = norm[norm > 0]
            bound = sim.state_grad_clip * (active.median() if active.numel() else norm.new_tensor(0.0))
            scale = (bound / norm.clamp(min=1e-30)).clamp(max=1.0)
            if sim.sync_free:
                sim.state_grad_clip_worlds += (scale < 1.0).sum()
            else:
                sim.state_grad_clip_events += int((scale < 1.0).sum())
            gq, gv = gq * scale[:, None], gv * scale[:, None]
        return gq, gv, g_ctrl, None, None


class _WindowControlStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qpos, qvel, ctrl, warm, sim: DiffSim):
        t = sim.win_t
        assert t < sim.window, "more differentiable control steps than the window size; call begin_window()"
        sim.win_t += 1
        K = sim.cfg.substeps
        views = sim.win_views[t]
        v0 = views[0]
        v0["qpos"].copy_(qpos)
        v0["qvel"].copy_(qvel)
        v0["ctrl"].copy_(ctrl)
        v0["warm"].copy_(warm)
        for k in range(K):
            wp.capture_launch(sim.win_fwd[t][k], stream=sim.stream)
        vK = views[K]
        # Overflow bits of this window are OR-ed into the main overflow buffer.
        for vk in views[1:]:
            sim.t_overflow.bitwise_or_(vk["overflow"])
        sim.cur_contact = vK["contact"]
        ctx.sim, ctx.t = sim, t
        warm_out = vK["warm"].clone()
        ctx.mark_non_differentiable(warm_out)
        return vK["qpos"].clone(), vK["qvel"].clone(), warm_out

    @staticmethod
    def backward(ctx, g_qpos, g_qvel, _g_warm):
        sim, t = ctx.sim, ctx.t
        K = sim.cfg.substeps
        views = sim.win_views[t]
        vK = views[K]
        vK["gqpos"].copy_(g_qpos if g_qpos is not None else 0.0)
        vK["gqvel"].copy_(g_qvel if g_qvel is not None else 0.0)
        g_ctrl = torch.zeros_like(views[0]["ctrl"])
        for k in reversed(range(K)):
            vin, vout = views[k], views[k + 1]
            torch._foreach_zero_(sim.win_zero[t][k])
            wp.capture_launch(sim.win_bwd[t][k], stream=sim.stream)
            gc = vin["gctrl"]
            if sim.mask_saturation:
                f = vout["force"]
                saturated = sim.force_limited & ((f <= sim.force_lo) | (f >= sim.force_hi))
                g_ctrl += torch.where(saturated, 0.0, gc)
            else:
                g_ctrl += gc
        gq, gv = views[0]["gqpos"].clone(), views[0]["gqvel"].clone()
        finite = torch.isfinite(gq).all(1) & torch.isfinite(gv).all(1) & torch.isfinite(g_ctrl).all(1)
        sim.nonfinite_grad_worlds += (~finite).sum()
        gq, gv, g_ctrl = (torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) for x in (gq, gv, g_ctrl))
        if sim.state_grad_clip is not None:
            norm = torch.sqrt(gq.square().sum(1) + gv.square().sum(1))
            active = norm[norm > 0]
            bound = sim.state_grad_clip * (active.median() if active.numel() else norm.new_tensor(0.0))
            scale = (bound / norm.clamp(min=1e-30)).clamp(max=1.0)
            sim.state_grad_clip_worlds += (scale < 1.0).sum()
            gq, gv = gq * scale[:, None], gv * scale[:, None]
        return gq, gv, g_ctrl, None, None


def repo_path(*parts) -> Path:
    return Path(__file__).resolve().parent.joinpath(*parts)
