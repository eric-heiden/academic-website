"""Differentiable batched forward kinematics (PyTorch) for selected points on MuJoCo bodies.

Given qpos (N, nq) of a tree with a free-joint root and hinge/slide joints, returns the world
positions of points fixed in chosen bodies. Only the chains from the root to those bodies are
evaluated. Follows MuJoCo's convention: a body frame is its parent frame composed with
(body_pos, body_quat), followed by its joints in order (a hinge rotates about its axis through
its anchor jnt_pos, a slide translates along its axis).
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch


def _quat_to_mat(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1).reshape(*q.shape[:-1], 3, 3)


def _axis_angle_mat(axis, angle):
    """Rotation matrices (N, 3, 3) about a fixed unit axis (3,) by angles (N,)."""
    x, y, z = axis
    c, s = torch.cos(angle), torch.sin(angle)
    C = 1 - c
    return torch.stack((
        c + x * x * C, x * y * C - z * s, x * z * C + y * s,
        y * x * C + z * s, c + y * y * C, y * z * C - x * s,
        z * x * C - y * s, z * y * C + x * s, c + z * z * C), -1).reshape(-1, 3, 3)


class PointFK:
    """World positions of points (body, local offset) as differentiable functions of qpos."""

    def __init__(self, m: mujoco.MjModel, points: list[tuple[str, tuple]], device):
        self.dev = device
        f32 = dict(dtype=torch.float32, device=device)
        self.points = []
        needed = set()
        for body, off in points:
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
            assert b >= 0, body
            self.points.append((b, torch.tensor(off, **f32)))
            while b > 0:
                needed.add(b)
                b = m.body_parentid[b]
        self.order = sorted(needed)  # parents have smaller ids in MuJoCo
        assert m.jnt_type[m.body_jntadr[1]] == mujoco.mjtJoint.mjJNT_FREE and m.body_parentid[1] == 0
        self.bodies = {}
        for b in self.order:
            joints = []
            for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]):
                joints.append((int(m.jnt_type[j]), int(m.jnt_qposadr[j]), torch.tensor(m.jnt_axis[j], **f32),
                               torch.tensor(m.jnt_pos[j], **f32), tuple(float(a) for a in m.jnt_axis[j])))
            self.bodies[b] = (int(m.body_parentid[b]), torch.tensor(m.body_pos[b], **f32),
                              _quat_to_mat(torch.tensor(m.body_quat[b], **f32)), joints)

        self._build_levels(m)

    def _build_levels(self, m):
        """Groups the non-root bodies by depth so that equal-depth bodies of different chains are
        evaluated together; only used when every such body has exactly one hinge joint."""
        depth = {}
        for b in self.order:
            depth[b] = 0 if m.body_parentid[b] == 0 else depth[m.body_parentid[b]] + 1
        self.levels = None
        rest = [b for b in self.order if depth[b] > 0]
        if any(len(self.bodies[b][3]) != 1 or self.bodies[b][3][0][0] != mujoco.mjtJoint.mjJNT_HINGE for b in rest):
            return
        f32 = dict(dtype=torch.float32, device=self.dev)
        self.root = [b for b in self.order if depth[b] == 0]
        assert len(self.root) == 1
        self.slot = {self.root[0]: 0}
        levels = []
        for dlev in range(1, max(depth.values()) + 1):
            bs = [b for b in rest if depth[b] == dlev]
            for b in bs:
                self.slot[b] = len(self.slot)
            levels.append(dict(
                parent=torch.tensor([self.slot[self.bodies[b][0]] for b in bs], dtype=torch.long, device=self.dev),
                bpos=torch.stack([self.bodies[b][1] for b in bs]),
                bmat=torch.stack([self.bodies[b][2] for b in bs]),
                adr=torch.tensor([self.bodies[b][3][0][1] for b in bs], dtype=torch.long, device=self.dev),
                axis=torch.stack([self.bodies[b][3][0][2] for b in bs]),
                jpos=torch.stack([self.bodies[b][3][0][3] for b in bs]),
                first=len(self.slot) - len(bs), count=len(bs)))
        self.levels = levels
        self.pt_slot = torch.tensor([self.slot[b] for b, _ in self.points], dtype=torch.long, device=self.dev)
        self.pt_off = torch.stack([off for _, off in self.points])

    def _call_levels(self, qpos):
        n = qpos.shape[0]
        adr = self.bodies[self.root[0]][3][0][1]
        quat = qpos[:, adr + 3:adr + 7]
        pos = [qpos[:, None, adr:adr + 3]]
        rot = [_quat_to_mat(quat / quat.norm(dim=-1, keepdim=True))[:, None]]
        P, R = pos[0], rot[0]
        for lv in self.levels:
            Rp = R[:, lv["parent"]]  # (N, B, 3, 3)
            pp = P[:, lv["parent"]]
            p = pp + torch.einsum("nbij,bj->nbi", Rp, lv["bpos"])
            Rb = Rp @ lv["bmat"]
            anchor = p + torch.einsum("nbij,bj->nbi", Rb, lv["jpos"])
            ang = qpos[:, lv["adr"]]  # (N, B)
            ax = lv["axis"]  # (B, 3)
            c, s_ = torch.cos(ang), torch.sin(ang)
            C = 1 - c
            x, y, z = ax[:, 0], ax[:, 1], ax[:, 2]
            Ra = torch.stack((c + x * x * C, x * y * C - z * s_, x * z * C + y * s_,
                              y * x * C + z * s_, c + y * y * C, y * z * C - x * s_,
                              z * x * C - y * s_, z * y * C + x * s_, c + z * z * C), -1).reshape(n, -1, 3, 3)
            Rb = Rb @ Ra
            p = anchor - torch.einsum("nbij,bj->nbi", Rb, lv["jpos"])
            P = torch.cat((P, p), 1)
            R = torch.cat((R, Rb), 1)
        return P[:, self.pt_slot] + torch.einsum("npij,pj->npi", R[:, self.pt_slot], self.pt_off)

    def __call__(self, qpos: torch.Tensor) -> torch.Tensor:
        """Returns (N, npoints, 3)."""
        if self.levels is not None:
            return self._call_levels(qpos)
        return self._call_generic(qpos)

    def _call_generic(self, qpos: torch.Tensor) -> torch.Tensor:
        n = qpos.shape[0]
        pos, rot = {}, {}
        for b in self.order:
            parent, bpos, bmat, joints = self.bodies[b]
            if b == 1 or parent == 0:
                jt, adr = joints[0][0], joints[0][1]
                assert jt == mujoco.mjtJoint.mjJNT_FREE
                p = qpos[:, adr:adr + 3]
                R = _quat_to_mat(qpos[:, adr + 3:adr + 7] / qpos[:, adr + 3:adr + 7].norm(dim=-1, keepdim=True))
                pos[b], rot[b] = p, R
                continue
            Rp, pp = rot[parent], pos[parent]
            p = pp + torch.einsum("nij,j->ni", Rp, bpos)
            R = Rp @ bmat
            for jt, adr, axis, jpos, axis_t in joints:
                if jt == mujoco.mjtJoint.mjJNT_HINGE:
                    anchor = p + torch.einsum("nij,j->ni", R, jpos)
                    R = R @ _axis_angle_mat(axis_t, qpos[:, adr])
                    p = anchor - torch.einsum("nij,j->ni", R, jpos)
                elif jt == mujoco.mjtJoint.mjJNT_SLIDE:
                    p = p + torch.einsum("nij,j->ni", R, axis) * qpos[:, adr:adr + 1]
                else:
                    raise NotImplementedError(jt)
            pos[b], rot[b] = p, R
        return torch.stack([pos[b] + torch.einsum("nij,j->ni", rot[b], off) for b, off in self.points], 1)


if __name__ == "__main__":
    # Validation against MuJoCo kinematics on random configurations.
    import sys

    from robots import ROBOTS, apply_edits

    for name in sys.argv[1:] or ["h1_il", "g1", "go1_il", "humanoid"]:
        r = ROBOTS[name]
        m = mujoco.MjModel.from_xml_path(r.xml)
        if r.model_edit:
            apply_edits(m, r.model_edit)
        d = mujoco.MjData(m)
        bodies = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(2, m.nbody)][-4:]
        pts = [(b, (0.03, -0.02, 0.05)) for b in bodies]
        fk = PointFK(m, pts, "cpu")
        rng = np.random.default_rng(0)
        err = 0.0
        for _ in range(20):
            q = m.qpos0.copy()
            q[:3] = rng.normal(size=3)
            quat = rng.normal(size=4)
            q[3:7] = quat / np.linalg.norm(quat)
            q[7:] += rng.uniform(-0.5, 0.5, m.nq - 7)
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            ref = np.stack([d.xpos[mujoco.mj_name2id(m, 1, b)] + d.xmat[mujoco.mj_name2id(m, 1, b)].reshape(3, 3) @ np.array(o)
                            for b, o in pts])
            out = fk(torch.tensor(q[None], dtype=torch.float32))[0].numpy()
            err = max(err, np.abs(out - ref).max())
        print(name, "max abs error", err)
