"""Feed-forward gravity torque for an articulation.

FeatherPGS has no equivalent of MuJoCo's per-joint gravity compensation, so the
Franka's position drive has to fight the arm's own weight. This computes the
exact generalized gravity torque tau_g(q) = dU/dq by differencing the potential
energy through forward kinematics, and writes it into control.joint_f.

The gradient is taken numerically only to keep this independent of the solver's
internals; the same quantity already exists inside the solver's inverse-dynamics
pass and could be subtracted there directly.
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton


class GravityCompensator:
    def __init__(self, env, n_dofs=7, eps=1.0e-4):
        self.env = env
        self.model = env.model
        self.n = n_dofs
        self.eps = eps
        self.scratch = env.model.state()
        self.mass = self.model.body_mass.numpy().copy()
        self.com = self.model.body_com.numpy().copy()
        self.g = self.model.gravity.numpy().reshape(-1)[:3].copy()
        # Only the arm's own links carry weight the drive must hold.
        self.mask = np.zeros(len(self.mass), dtype=bool)
        self.mask[:14] = True
        self.qd_zero = wp.zeros_like(self.model.joint_qd)
        self._q = wp.zeros_like(self.model.joint_q)

    def _potential(self, q_np):
        self._q.assign(q_np.astype(np.float32))
        newton.eval_fk(self.model, self._q, self.qd_zero, self.scratch)
        bq = self.scratch.body_q.numpy()
        u = 0.0
        for b in np.nonzero(self.mask)[0]:
            x, y, z, qx, qy, qz, qw = bq[b]
            cx, cy, cz = self.com[b]
            # rotate the body-frame COM offset into the world frame
            t = 2.0 * np.cross([qx, qy, qz], [cx, cy, cz])
            world_com = np.array([x, y, z]) + np.array([cx, cy, cz]) + qw * t + np.cross([qx, qy, qz], t)
            u -= self.mass[b] * float(np.dot(self.g, world_com))
        return u

    def torque(self):
        q = self.env.state_0.joint_q.numpy().copy()
        base = self._potential(q)
        tau = np.zeros(self.n, dtype=np.float32)
        for i in range(self.n):
            qp = q.copy()
            qp[i] += self.eps
            tau[i] = (self._potential(qp) - base) / self.eps
        return tau

    def apply(self):
        tau = self.torque()
        f = self.env.control.joint_f.numpy()
        f[: self.n] = tau
        self.env.control.joint_f.assign(f)
        return tau
