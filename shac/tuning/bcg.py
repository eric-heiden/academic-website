"""Bundled contact gradients (BCG) for the batched MJWarp simulator.

After Aditya et al., "Bundled Contact Gradients: Stabilizing Differentiable Simulation for Deployable Dynamic Tasks"
(arXiv 2609.30951). Each environment is simulated as B branches (N * B MJWarp worlds, world = env * B + branch).
When an environment's contact load exceeded a threshold in the previous control step, its branches start from the
environment state plus small joint perturbations (constants for the gradient); otherwise all branches start from the
same state. After the control step the branch states are averaged and the rollout continues from the average, so the
backward pass differentiates exactly the simulated, averaged map: the per-branch sensitivities are averaged. The
forward and the backward stay consistent, unlike backward-only contact smoothing.

Differences from the paper (see research/bcg_spec.md): perturbations are drawn in joint space on the policy's joints
(std sigma_q, sigma_qd) instead of mapping 1 cm Cartesian foot offsets through a damped pseudo-inverse of the contact
Jacobian; the trigger is the per-world total vertical contact load (in units of m g) instead of a per-contact normal
impulse; the bundle horizon is one control step (H = 1).
"""

from __future__ import annotations

import dataclasses

import torch
import warp as wp

from diffsim import DiffSim, SimConfig


@dataclasses.dataclass(frozen=True)
class BCGConfig:
    branches: int = 8           # B
    trigger: str = "force"      # force | always | never
    tau_mg: float = 1.2         # trigger when the total vertical contact load exceeds tau_mg * m * g
    sigma_q: float = 0.02       # joint-position perturbation std (rad)
    sigma_qd: float = 0.05      # joint-velocity perturbation std (rad/s)
    antithetic: bool = True     # branch pairs (delta, -delta); requires an even B
    seed: int = 0


class BundledDiffSim(DiffSim):
    """DiffSim with N environments x B branches; step() takes and returns per-environment tensors."""

    bundled = True

    def __init__(self, cfg: SimConfig, num_envs: int, bcg: BCGConfig, joints: torch.Tensor | None = None,
                 device: str = "cuda:0", graph: bool = True, backward: bool = True):
        assert bcg.branches >= 1
        assert not bcg.antithetic or bcg.branches % 2 == 0, "antithetic pairs need an even number of branches"
        self.nenv, self.B, self.bcg = num_envs, bcg.branches, bcg
        self._probe_on = False
        super().__init__(cfg, num_envs * bcg.branches, device=device, graph=graph, backward=backward)
        m, dev = self.mjm, self.torch_device
        assert m.nq == m.nv + 1, "expects a single free joint followed by hinge/slide joints"
        self.weight = float(m.body_subtreemass[1]) * 9.81
        # Perturbed joints (indices into the hinge joints, i.e. qpos[7:] / qvel[6:]); default: all hinges.
        self.pjoints = (torch.arange(m.nv - 6, device=dev) if joints is None else joints.to(dev).long())
        self.gen_b = torch.Generator(device=dev).manual_seed(bcg.seed)
        self.fz = torch.zeros(self.nworld, device=dev)            # max vertical contact load over the substeps
        self.trigger = torch.zeros(num_envs, dtype=torch.bool, device=dev)
        self.n_bundles = torch.zeros((), dtype=torch.long, device=dev)
        self.n_bad_branches = torch.zeros((), dtype=torch.long, device=dev)
        self._fz_views = {}
        self.t_out_qfrc_c = wp.to_torch(self.d_out.qfrc_constraint, requires_grad=False)
        self._probe_on = True

    # ------------------------------------------------------------------ trigger statistic
    def forward_physics(self):
        # Recompute path and step_nograd: one physics step into d_out; record its vertical contact load.
        super().forward_physics()
        if self._probe_on:
            torch.maximum(self.fz, self.t_out_qfrc_c[:, 2], out=self.fz)

    def _window_fz(self):
        t = self.win_t - 1
        for k in range(1, self.cfg.substeps + 1):
            d = self.slots[t][k]
            v = self._fz_views.get(id(d))
            if v is None:
                v = self._fz_views[id(d)] = wp.to_torch(d.qfrc_constraint, requires_grad=False)
            torch.maximum(self.fz, v[:, 2], out=self.fz)

    def _update_trigger(self):
        load = self.fz.view(self.nenv, self.B).mean(1)  # load of the averaged trajectory (= unbundled if untriggered)
        mode = self.bcg.trigger
        if mode == "always":
            self.trigger = torch.ones_like(self.trigger)
        elif mode == "never":
            self.trigger = torch.zeros_like(self.trigger)
        else:
            self.trigger = load > self.bcg.tau_mg * self.weight

    def mark_reset(self, done):
        """Clears the trigger of environments that were reset (their contact load belongs to the old episode)."""
        self.trigger = self.trigger & ~done

    # ------------------------------------------------------------------ bundled step
    def _perturbation(self, start):
        N, B, nj = self.nenv, self.B, self.pjoints.numel()
        nb = B // 2 if self.bcg.antithetic else B
        z = torch.randn(2, N, nb, nj, device=self.torch_device, generator=self.gen_b)
        if self.bcg.antithetic:
            z = torch.cat((z, -z), 2)
        scale = start.float()[:, None, None]
        dq = torch.zeros(N, B, self.nv - 6, device=self.torch_device)
        dqd = torch.zeros_like(dq)
        dq[:, :, self.pjoints] = self.bcg.sigma_q * z[0] * scale
        dqd[:, :, self.pjoints] = self.bcg.sigma_qd * z[1] * scale
        return dq.view(N * B, -1), dqd.view(N * B, -1)

    def _expand(self, x):
        return x.unsqueeze(1).expand(self.nenv, self.B, *x.shape[1:]).reshape(self.nenv * self.B, *x.shape[1:])

    def _average(self, qn, vn, wn):
        N, B = self.nenv, self.B
        q3, v3, w3 = qn.view(N, B, -1), vn.view(N, B, -1), wn.view(N, B, -1)
        ok = torch.isfinite(q3).all(2) & torch.isfinite(v3).all(2) & (v3.abs().amax(2) < 1e4)
        self.n_bad_branches += (~ok).sum()
        wgt = (ok.float() / ok.float().sum(1, keepdim=True).clamp(min=1.0)).unsqueeze(2)
        quat = q3[..., 3:7]
        sgn = torch.where((quat * quat[:, :1]).sum(-1, keepdim=True) < 0, -1.0, 1.0).detach()
        q3 = torch.cat((q3[..., :3], sgn * quat, q3[..., 7:]), -1)
        q = (wgt * torch.where(ok[..., None], q3, 0.0)).sum(1)
        v = (wgt * torch.where(ok[..., None], v3, 0.0)).sum(1)
        quat = q[:, 3:7]
        q = torch.cat((q[:, :3], quat / quat.norm(dim=-1, keepdim=True).clamp(min=1e-9), q[:, 7:]), -1)
        dead = ~ok.any(1, keepdim=True)  # every branch non-finite: return NaN so the env replaces the world
        q = torch.where(dead, float("nan"), q)
        v = torch.where(dead, float("nan"), v)
        w = (wgt * torch.where(ok[..., None], w3, 0.0)).sum(1).detach()
        return q, v, w

    def _bundled(self, q, v, ctrl, w, differentiable):
        start = self.trigger
        dq, dqd = self._perturbation(start)
        qb = self._expand(q)
        qb = torch.cat((qb[:, :7], qb[:, 7:] + dq), 1)
        vb = self._expand(v)
        vb = torch.cat((vb[:, :6], vb[:, 6:] + dqd), 1)
        self.fz.zero_()
        if differentiable:
            qn, vn, wn = DiffSim.step(self, qb, vb, self._expand(ctrl), self._expand(w))
            if self.window is not None:
                self._window_fz()
        else:
            qn, vn, wn = DiffSim.step_nograd(self, qb, vb, self._expand(ctrl), self._expand(w))
        self.n_bundles += start.sum()
        self._update_trigger()
        return self._average(qn, vn, wn)

    def step(self, qpos, qvel, ctrl, warm):
        return self._bundled(qpos, qvel, ctrl, warm, True)

    def step_nograd(self, qpos, qvel, ctrl, warm=None):
        if warm is None:
            warm = torch.zeros_like(qvel)
        with torch.no_grad():
            return self._bundled(qpos, qvel, ctrl, warm, False)
