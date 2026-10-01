"""Configurable, synchronisation-free differentiable locomotion tasks on ``DiffSim``.

A task is a ``TaskCfg``: observation terms, weighted reward terms, termination
conditions, reset randomisation, and optional velocity commands. Reward and
observation terms are plain PyTorch functions of the simulator state, so every
term that depends on ``qpos``/``qvel`` is differentiable through the analytic
simulator adjoint. Terms that read contact information from the simulator
(foot contacts, air time) are not differentiable and only reach the policy
through the critic.

Unlike ``locomotion.LocomotionEnv``, ``step`` never synchronises with the host:
resets and replacements use ``torch.where`` on every step, and episode
statistics are accumulated in device tensors.

Conventions: MuJoCo free joint, ``qpos[0:3]`` world position, ``qpos[3:7]``
quaternion (w, x, y, z), ``qvel[0:3]`` world-frame linear velocity,
``qvel[3:6]`` body-frame angular velocity; z is up.
"""

from __future__ import annotations

import dataclasses
import math

import mujoco
import numpy as np
import torch
import warp as wp

from diffsim import DiffSim, SimConfig
from robots import ROBOTS, Robot

# ----------------------------------------------------------------------------- math


def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw), -1)


def quat_apply(q, v):
    """Rotates vectors v (N, 3) by unit quaternions q (N, 4)."""
    w, xyz = q[:, 0:1], q[:, 1:4]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def quat_apply_inv(q, v):
    """Rotates world vectors v into the frame of q (applies the conjugate)."""
    w, xyz = q[:, 0:1], -q[:, 1:4]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def yaw_of(q):
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# ----------------------------------------------------------------------------- config


@dataclasses.dataclass(frozen=True)
class TaskCfg:
    robot: str
    # Observation terms (name, scale); see Env.OBS.
    obs: tuple = (("height", 1.0), ("quat", 1.0), ("lin_vel_w", 1.0), ("ang_vel_b", 1.0), ("joint_pos", 1.0),
                  ("joint_vel", 0.1), ("up", 1.0), ("heading", 1.0), ("prev_action", 1.0))
    # Reward terms (name, weight); see Env.REWARDS. Parameters are fields below.
    rewards: tuple = ()
    # Actions: "torque" maps [-1, 1] to the ctrl range; "position" sets targets home + scale * a.
    action_mode: str = "torque"
    action_scale: float = 0.5
    # Termination.
    min_height: float = 0.0
    max_height: float = 100.0
    min_up: float = -1.0  # minimum body-z . world-z
    episode_length: int = 1000
    # Reset randomisation (uniform, symmetric): base xy/z offset, tilt angle (rad), yaw (rad),
    # joint offset (rad), velocity (all dofs), multiplicative joint scale range.
    reset_xy: float = 0.1
    reset_z: float = 0.1
    reset_tilt: float = math.pi / 24.0
    reset_yaw: float = 0.0
    reset_joint: float = 0.2
    reset_joint_scale: tuple = (1.0, 1.0)
    reset_vel: float = 0.25
    reset_vel_joints_only: bool = False  # velocity noise on joint dofs only (Isaac Lab style)
    spawn_dz: float = 0.0  # added to the nominal base height at reset (m)
    # If > 0, resets draw from a bank of this many initial states pre-filtered on the CPU so that
    # no floor or self contact penetrates deeper than bank_max_pen (m).
    init_bank: int = 0
    bank_max_pen: float = 0.01
    # Velocity commands (body-frame vx, vy, yaw rate), resampled every resample_s seconds.
    cmd_vx: tuple = (0.0, 0.0)
    cmd_vy: tuple = (0.0, 0.0)
    cmd_wz: tuple = (0.0, 0.0)
    cmd_resample_s: float = 10.0
    cmd_standing_frac: float = 0.0
    # Reward parameters.
    speed_cap: float | None = None  # forward-velocity reward clipped above this (m/s)
    height_pivot: float = 0.0  # SHAC humanoid height reward pivot (m)
    height_cap: float = 0.1
    target_height: float = 0.0  # base_height_l2 target (m)
    track_std: float = 0.5  # std of the exponential velocity-tracking kernels (m/s, rad/s)
    feet_air_threshold: float = 0.4  # s
    soft_limit_factor: float = 0.9  # joint_pos_limits: fraction of the range that is free
    deviation_joints: tuple = ()  # joint names for joint_deviation_l1
    foot_geoms: tuple = ()  # geom names used for contact features (one foot per geom)
    foot_bodies: tuple = ()  # alternatively: body names; all collision geoms of a body form one foot
    # Terminate when any collision geom of these bodies touches the floor (Isaac Lab illegal contact).
    term_bodies: tuple = ()
    # Separate root velocity noise (6 dofs); when None, reset_vel applies to all dofs as before.
    reset_root_vel: float | None = None
    limit_joints: tuple = ()  # joint names for joint_pos_limits (empty: all limited joints)
    deviation_joints2: tuple = ()  # second joint set for joint_deviation2_l1
    # Policy controls only the actuators of these joints (position mode); the others hold the home pose.
    act_joints: tuple = ()
    obs_act_joints_only: bool = False  # joint_pos_rel / joint_vel observations for the controlled joints only
    gait_period: float = 0.0  # s; period of the gait clock (observation "phase" and gait rewards)
    # Differentiable gait prior (foot points via torch forward kinematics):
    foot_points: tuple = ()  # ((body, (x, y, z) offset), ...) foot sole points
    gait_offsets: tuple = ()  # phase offset per foot point (fraction of the period)
    gait_duty: float = 0.6  # stance fraction of the cycle
    gait_height: float = 0.08  # swing apex above the standing foot height (m)
    raibert_reach: float = 0.1  # foot target: nominal + reach * v + raibert_gain * (v - v_cmd) (yaw frame, s)
    raibert_gain: float = 0.1


# ----------------------------------------------------------------------------- env


class Env:
    """Batched differentiable locomotion environment (no host synchronisation in ``step``)."""

    def __init__(self, task: TaskCfg, sim_cfg: SimConfig, num_envs: int, seed: int = 0, device: str = "cuda:0",
                 stochastic_init: bool = True, backward: bool = True):
        self.task = task
        self.robot: Robot = ROBOTS[task.robot]
        self.sim = DiffSim(sim_cfg, num_envs, device=device, backward=backward)
        self.sim.sync_free = True
        self.N = num_envs
        self.dev = self.sim.torch_device
        self.dt = sim_cfg.control_dt
        self.stochastic_init = stochastic_init
        self.gen = torch.Generator(device=self.dev).manual_seed(seed)
        m = self.sim.mjm
        self.mjm = m
        self.nq, self.nv, self.nu = m.nq, m.nv, m.nu
        self.q0 = self._nominal_qpos()
        self.home = self.q0[7:].clone()
        f32 = dict(dtype=torch.float32, device=self.dev)
        # Joint data (hinge/slide joints after the free joint, one dof each).
        self.jnt_lo = torch.tensor(m.jnt_range[1:, 0], **f32)
        self.jnt_hi = torch.tensor(m.jnt_range[1:, 1], **f32)
        self.jnt_limited = torch.tensor(m.jnt_limited[1:].astype(bool), device=self.dev)
        mid, half = 0.5 * (self.jnt_lo + self.jnt_hi), 0.5 * (self.jnt_hi - self.jnt_lo)
        self.soft_lo = mid - task.soft_limit_factor * half
        self.soft_hi = mid + task.soft_limit_factor * half
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, m.njnt)]
        self.joint_names = names
        self.dev_idx = torch.tensor([names.index(n) for n in task.deviation_joints], dtype=torch.long,
                                    device=self.dev)
        # Actuators: force = gain * ctrl + b0 + b1 * length + b2 * velocity, joint torque = gear * force.
        self.act_joint = torch.tensor(m.actuator_trnid[:, 0] - 1, dtype=torch.long, device=self.dev)
        self.act_gain = torch.tensor(m.actuator_gainprm[:, 0], **f32)
        self.act_b = torch.tensor(m.actuator_biasprm[:, :3], **f32)
        self.act_gear = torch.tensor(m.actuator_gear[:, 0], **f32)
        self.act_frc_lo = torch.tensor(np.where(m.actuator_forcelimited, m.actuator_forcerange[:, 0], -np.inf), **f32)
        self.act_frc_hi = torch.tensor(np.where(m.actuator_forcelimited, m.actuator_forcerange[:, 1], np.inf), **f32)
        self.home_ctrl = self.home[self.act_joint]
        if task.act_joints:
            assert task.action_mode == "position"
            act_names = [names[j - 1] for j in m.actuator_trnid[:, 0]]
            self.pol_act = torch.tensor([act_names.index(n) for n in task.act_joints], dtype=torch.long,
                                        device=self.dev)
            self.pol_joint = torch.tensor([names.index(n) for n in task.act_joints], dtype=torch.long,
                                          device=self.dev)
            self.nu = len(task.act_joints)
        else:
            self.pol_act = None
            self.pol_joint = torch.arange(m.njnt - 1, device=self.dev)
        # Map collision geoms to foot indices for contact features (-1: not a foot).
        foot_of_geom = np.full(m.ngeom + 1, -1, dtype=np.int64)
        for k, g in enumerate(task.foot_geoms):
            foot_of_geom[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g)] = k
        for k, b in enumerate(task.foot_bodies):
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
            assert bid >= 0, f"unknown body {b}"
            foot_of_geom[np.nonzero(m.geom_bodyid == bid)[0]] = k
        self.foot_of_geom = torch.tensor(foot_of_geom, device=self.dev)
        self.nfeet = len(task.foot_geoms) + len(task.foot_bodies)
        term_geom = np.zeros(m.ngeom + 1, dtype=bool)
        for b in task.term_bodies:
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
            assert bid >= 0, f"unknown body {b}"
            term_geom[np.nonzero(m.geom_bodyid == bid)[0]] = True
        self.term_geom = torch.tensor(term_geom, device=self.dev)
        self.term_contact = torch.zeros(num_envs, dtype=torch.bool, device=self.dev)
        self.dev_idx2 = torch.tensor([names.index(n) for n in task.deviation_joints2], dtype=torch.long,
                                     device=self.dev)
        lim_sel = torch.zeros(m.njnt - 1, dtype=torch.bool, device=self.dev)
        if task.limit_joints:
            lim_sel[[names.index(n) for n in task.limit_joints]] = True
        else:
            lim_sel[:] = True
        self.limit_sel = lim_sel & self.jnt_limited
        # Foot points for the gait prior (standing height and nominal yaw-frame offsets at the home pose).
        self.fk = None
        if task.foot_points:
            from fk import PointFK
            self.fk = PointFK(m, list(task.foot_points), self.dev)
            feet0 = self.fk(self.q0[None])[0]
            self.foot_z0 = feet0[:, 2] - self.q0[2] + self._standing_height()
            self.foot_nominal = feet0[:, 0:2] - self.q0[0:2]  # q0 has zero yaw
            self.gait_off = torch.tensor(task.gait_offsets, dtype=torch.float32, device=self.dev)
        # Commands.
        lo = torch.tensor([task.cmd_vx[0], task.cmd_vy[0], task.cmd_wz[0]], **f32)
        hi = torch.tensor([task.cmd_vx[1], task.cmd_vy[1], task.cmd_wz[1]], **f32)
        self.cmd_lo, self.cmd_hi = lo, hi
        self.has_cmd = bool((hi - lo).abs().sum() > 0 or hi.abs().sum() > 0)
        self.cmd_resample = max(1, round(task.cmd_resample_s / self.dt))
        self.rewards = [(n, float(w)) for n, w in task.rewards if w != 0.0]
        for n, _ in self.rewards:
            assert n in self.REWARDS, f"unknown reward term {n}"
        for n, _ in task.obs:
            assert n in self.OBS, f"unknown observation term {n}"

        # State.
        N = num_envs
        self.q = self.q0.expand(N, -1).clone()
        self.v = torch.zeros(N, self.nv, device=self.dev)
        self.w = torch.zeros_like(self.v)
        self.prev_action = torch.zeros(N, self.nu, device=self.dev)
        self.prev_qd = torch.zeros(N, self.nv - 6, device=self.dev)
        self.progress = torch.zeros(N, dtype=torch.long, device=self.dev)
        self.cmd = torch.zeros(N, 3, device=self.dev)
        self.air_time = torch.zeros(N, max(self.nfeet, 1), device=self.dev)
        self.contact_time = torch.zeros(N, max(self.nfeet, 1), device=self.dev)
        self.contact = torch.zeros(N, max(self.nfeet, 1), dtype=torch.bool, device=self.dev)
        # Episode statistics (device tensors, read once per epoch by the trainer).
        self.ep_ret = torch.zeros(N, device=self.dev)
        self.ep_len = torch.zeros(N, device=self.dev)
        self.ep_x0 = torch.zeros(N, 2, device=self.dev)
        self.ep_track = torch.zeros(N, device=self.dev)
        self.stats = torch.zeros(8, dtype=torch.float64, device=self.dev)
        self.nonfinite_resets = torch.zeros((), dtype=torch.long, device=self.dev)
        self.bank_accept = None
        self._feet_prev = None
        if task.init_bank and stochastic_init:
            self._build_bank()
        self.reset_all()
        self.num_obs = self.obs().shape[1]

    STAT_KEYS = ("episodes", "return", "length", "speed_x", "track_err", "terminated", "dist_x", "dist_y")

    # ------------------------------------------------------------------ setup
    def _nominal_qpos(self):
        m = self.sim.mjm
        nid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_NUMERIC, "init_qpos")
        if nid >= 0:
            adr = m.numeric_adr[nid]
            return torch.tensor(m.numeric_data[adr:adr + m.nq], dtype=torch.float32, device=self.dev)
        return self.sim.init_qpos.clone()

    def _standing_height(self):
        """Base height of the nominal pose with the lowest collision geom on the floor."""
        m = self.sim.mjm
        d = mujoco.MjData(m)
        d.qpos[:] = self.q0.cpu().numpy()
        mujoco.mj_forward(m, d)
        floor = [i for i in range(d.ncon) if m.geom_bodyid[d.contact.geom[i, 0]] == 0
                 or m.geom_bodyid[d.contact.geom[i, 1]] == 0]
        pen = min((d.contact.dist[i] for i in floor), default=0.0)
        return float(self.q0[2]) - float(pen) if floor else float(self.q0[2])

    def _u(self, *shape):
        return 2.0 * torch.rand(*shape, device=self.dev, generator=self.gen) - 1.0

    def _build_bank(self):
        """Samples initial states on the CPU with MuJoCo and keeps those without deep penetration."""
        t = self.task
        m = self.sim.mjm
        d = mujoco.MjData(m)
        rng = np.random.default_rng(1234)
        q0 = self.q0.cpu().numpy().astype(np.float64)
        lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]
        lim = m.jnt_limited[1:].astype(bool)
        qs, vs, tried = [], [], 0
        while len(qs) < t.init_bank and tried < 50 * t.init_bank:
            tried += 1
            q = q0.copy()
            u = lambda *sh: rng.uniform(-1.0, 1.0, sh)  # noqa: E731
            q[0:2] += t.reset_xy * u(2)
            q[2] += t.spawn_dz + t.reset_z * u(1)[0]
            ang = t.reset_tilt * u(1)[0]
            ax = u(3)
            ax /= max(np.linalg.norm(ax), 1e-9)
            dq = np.concatenate(([np.cos(0.5 * ang)], np.sin(0.5 * ang) * ax))
            quat = np.empty(4)
            mujoco.mju_mulQuat(quat, dq, q[3:7])
            q[3:7] = quat
            s_lo, s_hi = t.reset_joint_scale
            q[7:] = q0[7:] * rng.uniform(s_lo, s_hi, m.nq - 7) + t.reset_joint * u(m.nq - 7)
            q[7:] = np.where(lim, np.clip(q[7:], lo, hi), q[7:])
            v = np.zeros(m.nv)
            if t.reset_root_vel is not None:
                v[0:6] = t.reset_root_vel * u(6)
                v[6:] = t.reset_vel * u(m.nv - 6)
            elif t.reset_vel_joints_only:
                v[6:] = t.reset_vel * u(m.nv - 6)
            else:
                v[:] = t.reset_vel * u(m.nv)
            d.qpos[:] = q
            d.qvel[:] = v
            mujoco.mj_forward(m, d)
            if d.ncon and d.contact.dist[:d.ncon].min() < -t.bank_max_pen:
                continue
            qs.append(q.astype(np.float32))
            vs.append(v.astype(np.float32))
        assert len(qs) == t.init_bank, f"init bank: only {len(qs)} valid states in {tried} tries"
        self.bank_accept = len(qs) / tried
        self.bank_q = torch.tensor(np.stack(qs), device=self.dev)
        self.bank_v = torch.tensor(np.stack(vs), device=self.dev)

    def _sample_init(self, n):
        t = self.task
        if self.stochastic_init and t.init_bank:
            idx = torch.randint(0, t.init_bank, (n,), device=self.dev, generator=self.gen)
            q, v = self.bank_q[idx].clone(), self.bank_v[idx].clone()
            if t.reset_yaw:
                yaw = t.reset_yaw * self._u(n)
                qz = torch.stack((torch.cos(0.5 * yaw), 0 * yaw, 0 * yaw, torch.sin(0.5 * yaw)), -1)
                q[:, 3:7] = quat_mul(qz, q[:, 3:7])
                c, s_ = torch.cos(yaw), torch.sin(yaw)
                vx, vy = v[:, 0].clone(), v[:, 1].clone()
                v[:, 0], v[:, 1] = c * vx - s_ * vy, s_ * vx + c * vy
            return q, v
        q = self.q0.expand(n, -1).clone()
        v = torch.zeros(n, self.nv, device=self.dev)
        if self.stochastic_init:
            q[:, 0:2] += t.reset_xy * self._u(n, 2)
            q[:, 2] += t.spawn_dz + t.reset_z * self._u(n)
            angle = t.reset_tilt * self._u(n)
            axis = torch.nn.functional.normalize(self._u(n, 3), dim=-1)
            dq = torch.cat((torch.cos(0.5 * angle)[:, None], torch.sin(0.5 * angle)[:, None] * axis), -1)
            quat = quat_mul(dq, q[:, 3:7])
            if t.reset_yaw:
                yaw = t.reset_yaw * self._u(n)
                qz = torch.stack((torch.cos(0.5 * yaw), 0 * yaw, 0 * yaw, torch.sin(0.5 * yaw)), -1)
                quat = quat_mul(qz, quat)
            q[:, 3:7] = quat
            s_lo, s_hi = t.reset_joint_scale
            scale = s_lo + (s_hi - s_lo) * torch.rand(n, self.nq - 7, device=self.dev, generator=self.gen)
            q[:, 7:] = self.home * scale + t.reset_joint * self._u(n, self.nq - 7)
            q[:, 7:] = torch.where(self.jnt_limited, torch.minimum(torch.maximum(q[:, 7:], self.jnt_lo), self.jnt_hi),
                                   q[:, 7:])
            if t.reset_root_vel is not None:
                v[:, 0:6] += t.reset_root_vel * self._u(n, 6)
                v[:, 6:] += t.reset_vel * self._u(n, self.nv - 6)
            elif t.reset_vel_joints_only:
                v[:, 6:] += t.reset_vel * self._u(n, self.nv - 6)
            else:
                v += t.reset_vel * self._u(n, self.nv)
        return q, v

    def _sample_cmd(self, n):
        t = self.task
        c = self.cmd_lo + (self.cmd_hi - self.cmd_lo) * torch.rand(n, 3, device=self.dev, generator=self.gen)
        if t.cmd_standing_frac > 0:
            stand = torch.rand(n, device=self.dev, generator=self.gen) < t.cmd_standing_frac
            c = torch.where(stand[:, None], 0.0, c)
        return c

    def reset_all(self):
        self._feet_prev = None
        self.q, self.v = self._sample_init(self.N)
        self.w = torch.zeros_like(self.v)
        self.prev_action.zero_()
        self.prev_qd.zero_()
        self.progress.zero_()
        self.cmd = self._sample_cmd(self.N)
        self.air_time.zero_()
        self.contact_time.zero_()
        self.ep_ret.zero_()
        self.ep_len.zero_()
        self.ep_x0 = self.q[:, 0:2].clone()
        self.ep_track.zero_()

    def detach(self):
        self.q, self.v, self.w = self.q.detach(), self.v.detach(), self.w.detach()
        self.prev_action = self.prev_action.detach()
        self.prev_qd = self.prev_qd.detach()
        if self._feet_prev is not None:
            self._feet_prev = self._feet_prev.detach()

    # ------------------------------------------------------------------ features
    class F:
        """Lazily computed state features."""

        def __init__(self, env, q, v):
            self.env, self.q, self.v = env, q, v
            self._c = {}

        def __getattr__(self, name):
            c = self.__dict__["_c"]
            if name not in c:
                c[name] = getattr(Env, "_f_" + name)(self.env, self)
            return c[name]

    def _f_height(self, f):
        return f.q[:, 2]

    def _f_quat(self, f):
        return f.q[:, 3:7]

    def _f_lin_vel_w(self, f):
        return f.v[:, 0:3]

    def _f_lin_vel_b(self, f):
        return quat_apply_inv(f.quat, f.v[:, 0:3])

    def _f_ang_vel_b(self, f):
        return f.v[:, 3:6]

    def _f_gravity_b(self, f):
        g = torch.zeros_like(f.q[:, 0:3])
        g[:, 2] = -1.0
        return quat_apply_inv(f.quat, g)

    def _f_up(self, f):
        q = f.q
        return 1.0 - 2.0 * (q[:, 4] ** 2 + q[:, 5] ** 2)

    def _f_heading(self, f):
        q = f.q
        return 1.0 - 2.0 * (q[:, 5] ** 2 + q[:, 6] ** 2)

    def _f_yaw(self, f):
        return yaw_of(f.quat)

    def _f_lin_vel_yaw(self, f):
        """Linear velocity in the yaw-aligned (heading) frame."""
        yaw = f.yaw
        c, s = torch.cos(yaw), torch.sin(yaw)
        vw = f.v[:, 0:3]
        return torch.stack((c * vw[:, 0] + s * vw[:, 1], -s * vw[:, 0] + c * vw[:, 1], vw[:, 2]), -1)

    def _f_feet(self, f):
        return self.fk(f.q)

    def _f_swing(self, f):
        """Per-foot swing progress in [0, 1) during swing, -1 in stance, from the gait clock."""
        t = self.task
        ph = (self.progress.float()[:, None] * self.dt / t.gait_period + self.gait_off) % 1.0
        return torch.where(ph >= t.gait_duty, (ph - t.gait_duty) / (1.0 - t.gait_duty), -1.0)

    def _f_joint_pos(self, f):
        return f.q[:, 7:]

    def _f_joint_vel(self, f):
        return f.v[:, 6:]

    # ------------------------------------------------------------------ observations
    def _obs_height(self, f, a):
        return f.height[:, None]

    def _obs_quat(self, f, a):
        return f.quat

    def _obs_lin_vel_w(self, f, a):
        return f.lin_vel_w

    def _obs_lin_vel_b(self, f, a):
        return f.lin_vel_b

    def _obs_ang_vel_b(self, f, a):
        return f.ang_vel_b

    def _obs_gravity_b(self, f, a):
        return f.gravity_b

    def _obs_joint_pos(self, f, a):
        return f.joint_pos

    def _obs_joint_pos_rel(self, f, a):
        if self.task.obs_act_joints_only:
            return f.joint_pos[:, self.pol_joint] - self.home[self.pol_joint]
        return f.joint_pos - self.home

    def _obs_joint_vel(self, f, a):
        if self.task.obs_act_joints_only:
            return f.joint_vel[:, self.pol_joint]
        return f.joint_vel

    def _obs_up(self, f, a):
        return f.up[:, None]

    def _obs_heading(self, f, a):
        return f.heading[:, None]

    def _obs_prev_action(self, f, a):
        return a

    def _obs_command(self, f, a):
        return self.cmd

    def _obs_phase(self, f, a):
        ph = 2.0 * math.pi * self.progress.float() * self.dt / self.task.gait_period
        return torch.stack((torch.sin(ph), torch.cos(ph)), -1)

    def _obs_contact(self, f, a):
        return self.contact.float()

    def observe(self, q, v, action):
        f = self.F(self, q, v)
        return torch.cat([s * getattr(self, "_obs_" + n)(f, action) for n, s in self.task.obs], -1)

    def obs(self):
        return self.observe(self.q, self.v, self.prev_action)

    # ------------------------------------------------------------------ rewards
    # Each term returns an unweighted (N,) tensor; the task weight multiplies it.
    def _r_fwd_vel(self, f, a, pa):
        vx = f.v[:, 0]
        return vx if self.task.speed_cap is None else torch.clamp(vx, max=self.task.speed_cap)

    def _r_up(self, f, a, pa):
        return f.up

    def _r_heading(self, f, a, pa):
        return f.heading

    def _r_height_linear(self, f, a, pa):
        return f.height - self.task.min_height

    def _r_height_shac(self, f, a, pa):
        x = (f.height - self.task.height_pivot).clamp(-1.0, self.task.height_cap)
        return torch.where(x < 0, -200.0 * x * x, 10.0 * x)

    def _r_lateral_vel_l2(self, f, a, pa):
        return f.v[:, 1].square()

    def _r_action_l2(self, f, a, pa):
        return a.square().sum(-1)

    def _r_action_rate_l2(self, f, a, pa):
        return (a - pa).square().sum(-1)

    def _r_alive(self, f, a, pa):
        return torch.ones_like(f.height)

    def _r_termination(self, f, a, pa):
        """1 on the step that terminates (not differentiable; reaches the actor through the critic)."""
        return self._term.float()

    def _r_fwd_disp(self, f, a, pa):
        """Average forward velocity over the control step: (x_{t+1} - x_t) / dt (differentiable in both)."""
        v = (f.q[:, 0] - self._x_before) / self.dt
        return v if self.task.speed_cap is None else torch.clamp(v, max=self.task.speed_cap)

    def _r_upright_l2(self, f, a, pa):
        """(1 - up)^2: non-saturating restoring term near upright."""
        return (1.0 - f.up).square()

    def _r_height_below_l2(self, f, a, pa):
        """relu(target_height - h)^2."""
        return (self.task.target_height - f.height).clamp(min=0.0).square()

    def _r_track_lin_vel_xy_exp(self, f, a, pa):
        err = (self.cmd[:, 0:2] - f.lin_vel_b[:, 0:2]).square().sum(-1)
        return torch.exp(-err / self.task.track_std ** 2)

    def _r_track_lin_vel_xy_yaw_exp(self, f, a, pa):
        err = (self.cmd[:, 0:2] - f.lin_vel_yaw[:, 0:2]).square().sum(-1)
        return torch.exp(-err / self.task.track_std ** 2)

    def _r_track_ang_vel_z_exp(self, f, a, pa):
        err = (self.cmd[:, 2] - f.ang_vel_b[:, 2]).square()
        return torch.exp(-err / self.task.track_std ** 2)

    def _r_track_ang_vel_z_world_exp(self, f, a, pa):
        wz = quat_apply(f.quat, f.ang_vel_b)[:, 2]
        return torch.exp(-(self.cmd[:, 2] - wz).square() / self.task.track_std ** 2)

    def _r_track_lin_vel_xy_l2(self, f, a, pa):
        """Squared planar tracking error (body frame); keeps a gradient far from the target."""
        return (self.cmd[:, 0:2] - f.lin_vel_b[:, 0:2]).square().sum(-1)

    def _r_track_lin_vel_proj(self, f, a, pa):
        """Planar velocity projected on the command direction, capped at the command speed (linear pull)."""
        c = self.cmd[:, 0:2]
        n = c.norm(dim=-1)
        proj = (f.lin_vel_b[:, 0:2] * c).sum(-1) / n.clamp(min=1e-6)
        return torch.where(n > 1e-3, torch.minimum(proj, n), 0.0)

    def _r_track_ang_vel_z_l2(self, f, a, pa):
        return (self.cmd[:, 2] - f.ang_vel_b[:, 2]).square()

    def _r_lin_vel_z_l2(self, f, a, pa):
        return f.lin_vel_b[:, 2].square()

    def _r_ang_vel_xy_l2(self, f, a, pa):
        return f.ang_vel_b[:, 0:2].square().sum(-1)

    def _r_flat_orientation_l2(self, f, a, pa):
        return f.gravity_b[:, 0:2].square().sum(-1)

    def _r_base_height_l2(self, f, a, pa):
        return (f.height - self.task.target_height).square()

    def _r_joint_torques_l2(self, f, a, pa):
        return self.joint_torque(f, a).square().sum(-1)

    def _r_joint_acc_l2(self, f, a, pa):
        return ((f.joint_vel - self.prev_qd) / self.dt).square().sum(-1)

    def _r_joint_vel_l2(self, f, a, pa):
        return f.joint_vel.square().sum(-1)

    def _r_joint_pos_limits(self, f, a, pa):
        jp = f.joint_pos
        out = (self.soft_lo - jp).clamp(min=0.0) + (jp - self.soft_hi).clamp(min=0.0)
        return torch.where(self.limit_sel, out, 0.0).sum(-1)

    def _r_joint_deviation_l1(self, f, a, pa):
        return (f.joint_pos[:, self.dev_idx] - self.home[self.dev_idx]).abs().sum(-1)

    def _r_joint_deviation2_l1(self, f, a, pa):
        return (f.joint_pos[:, self.dev_idx2] - self.home[self.dev_idx2]).abs().sum(-1)

    def _r_gait_height(self, f, a, pa):
        """Squared deviation of foot heights from the clock profile (differentiable through FK)."""
        sw = f.swing
        ref = self.foot_z0 + self.task.gait_height * torch.where(sw >= 0, torch.sin(math.pi * sw.clamp(min=0.0)), 0.0)
        return (f.feet[:, :, 2] - ref).square().sum(-1)

    def _r_gait_slip(self, f, a, pa):
        """Squared horizontal foot velocity of clock-stance feet over the control step."""
        vel = (f.feet[:, :, 0:2] - self._feet_before[:, :, 0:2]) / self.dt
        # The first step after a reset has no valid previous foot position.
        keep = (f.swing < 0) & ~self._fresh_step[:, None]
        return torch.where(keep, vel.square().sum(-1), 0.0).sum(-1)

    def _r_raibert(self, f, a, pa):
        """Raibert foot placement in the yaw frame during the second half of swing."""
        t = self.task
        yaw = f.yaw
        c, s_ = torch.cos(yaw), torch.sin(yaw)
        rel = f.feet[:, :, 0:2] - f.q[:, None, 0:2]
        rel = torch.stack((c[:, None] * rel[..., 0] + s_[:, None] * rel[..., 1],
                           -s_[:, None] * rel[..., 0] + c[:, None] * rel[..., 1]), -1)
        v = f.lin_vel_yaw[:, 0:2]
        target = self.foot_nominal + (t.raibert_reach * v + t.raibert_gain * (v - self.cmd[:, 0:2]))[:, None, :]
        return torch.where(f.swing >= 0.5, (rel - target).square().sum(-1), 0.0).sum(-1)

    def _r_pose_l2(self, f, a, pa):
        return (f.joint_pos - self.home).square().sum(-1)

    def _r_energy(self, f, a, pa):
        return (self.joint_torque(f, a) * f.joint_vel[:, self.act_joint]).abs().sum(-1)

    def _r_feet_air_time(self, f, a, pa):
        """Isaac Lab feet_air_time (quadrupeds): rewards steps longer than the threshold (not differentiable)."""
        first = self.first_contact
        r = ((self.last_air_time - self.task.feet_air_threshold) * first).sum(-1)
        return r * (self.cmd[:, 0:2].norm(dim=-1) > 0.1)

    def _r_feet_air_time_biped(self, f, a, pa):
        """Isaac Lab feet_air_time_positive_biped: single-stance time, capped (not differentiable)."""
        in_contact = self.contact
        mode_time = torch.where(in_contact, self.contact_time, self.air_time)
        single = in_contact.int().sum(-1) == 1
        r = torch.where(single[:, None], mode_time, 0.0).amin(-1).clamp(max=self.task.feet_air_threshold)
        return r * (self.cmd[:, 0:2].norm(dim=-1) > 0.1)

    def joint_torque(self, f, a):
        """Differentiable actuator joint torques (force limits applied; ctrl clamping as in ``ctrl``)."""
        ctrl = self.ctrl(a)
        jp = f.joint_pos[:, self.act_joint]
        jv = f.joint_vel[:, self.act_joint]
        frc = self.act_gain * ctrl + self.act_b[:, 0] + self.act_b[:, 1] * jp + self.act_b[:, 2] * jv
        frc = torch.maximum(torch.minimum(frc, self.act_frc_hi), self.act_frc_lo)
        return self.act_gear * frc

    def reward(self, f, action, prev_action):
        r = torch.zeros_like(f.height)
        for n, wgt in self.rewards:
            r = r + wgt * getattr(self, "_r_" + n)(f, action, prev_action)
        return r

    # ------------------------------------------------------------------ contacts (non-differentiable)
    def _update_contacts(self):
        if not self.nfeet and not self.task.term_bodies:
            return
        con, dist, cworld, nacon = self.sim.cur_contact
        n = dist.shape[0]
        valid = torch.arange(n, device=self.dev) < nacon[0]
        valid = valid & (dist < 0.0)
        world = cworld.long().clamp(0, self.N - 1)
        g = con.long().clamp(min=-1)
        if self.task.term_bodies:
            tc = valid & (self.term_geom[g[:, 0]] | self.term_geom[g[:, 1]])
            hit_t = torch.zeros(self.N, device=self.dev)
            hit_t.index_add_(0, world, tc.float())
            self.term_contact = hit_t > 0
        if not self.nfeet:
            return
        foot = torch.maximum(self.foot_of_geom[g[:, 0]], self.foot_of_geom[g[:, 1]])
        valid = valid & (foot >= 0)
        hit = torch.zeros(self.N * self.nfeet, device=self.dev)
        hit.index_add_(0, world * self.nfeet + foot.clamp(min=0), valid.float())
        contact = hit.view(self.N, self.nfeet) > 0
        self.first_contact = (self.air_time > 0) & contact
        self.last_air_time = self.air_time.clone()
        self.air_time = torch.where(contact, 0.0, self.air_time + self.dt)
        self.contact_time = torch.where(contact, self.contact_time + self.dt, 0.0)
        self.contact = contact

    # ------------------------------------------------------------------ termination
    def terminated(self, f):
        t = self.task
        q, v = f.q, f.v
        bad = ~torch.isfinite(q).all(1) | ~torch.isfinite(v).all(1) | (v.abs().amax(1) > 1e4)
        fell = (f.height < t.min_height) | (f.height > t.max_height) | (f.up < t.min_up)
        if t.term_bodies:
            fell = fell | self.term_contact
        return fell | bad, bad

    def ctrl(self, action):
        if self.task.action_mode == "position":
            if self.pol_act is not None:
                base = self.home_ctrl.expand(action.shape[0], -1)
                full = base.index_copy(1, self.pol_act, base[:, self.pol_act] + self.task.action_scale * action)
                return self.sim.clamp_ctrl(full)
            return self.sim.clamp_ctrl(self.home_ctrl + self.task.action_scale * action)
        return self.sim.action_to_ctrl(action, False)

    # ------------------------------------------------------------------ step
    def step(self, action, differentiable: bool = True):
        """Returns obs, reward, done, info; resets finished envs in place without host synchronisation."""
        ctrl = self.ctrl(action)
        self._x_before = self.q[:, 0]
        if self.fk is not None:
            # Foot positions of the previous step's final state (one FK evaluation per step).
            self._fresh_step = self.progress == 0
            self._feet_before = self._feet_prev if self._feet_prev is not None else self.fk(self.q)
        if differentiable:
            q, v, w = self.sim.step(self.q, self.v, ctrl, self.w)
        else:
            q, v, w = self.sim.step_nograd(self.q, self.v, ctrl.detach(), self.w)
        self._update_contacts()
        self.progress += 1
        f = self.F(self, q, v)
        term, bad = self.terminated(f)
        self._term = term
        self.nonfinite_resets += bad.sum()
        # One reset sample per step serves both non-finite replacement and episode resets (bad implies done).
        nq, nv = self._sample_init(self.N)
        q = torch.where(bad[:, None], nq, q)
        v = torch.where(bad[:, None], nv, v)
        w = torch.where(bad[:, None], 0.0, w)
        f = self.F(self, q, v)
        # The displacement reward of replaced (non-finite) worlds is meaningless; zero the whole reward.
        self._x_before = torch.where(bad, q[:, 0].detach(), self._x_before)
        rew = torch.where(bad, 0.0, self.reward(f, action, self.prev_action))
        obs_before_reset = self.observe(q, v, action)
        trunc = (self.progress >= self.task.episode_length) & ~term
        done = term | trunc

        # Episode statistics.
        rd = rew.detach()
        self.ep_ret += rd
        self.ep_len += 1
        if self.has_cmd:
            self.ep_track += (self.cmd[:, 0:2] - f.lin_vel_b[:, 0:2].detach()).norm(dim=-1)
        d = done.double()
        dur = self.ep_len * self.dt
        dxy = q[:, 0:2].detach() - self.ep_x0
        self.stats += torch.stack((d.sum(), (self.ep_ret * d).sum(), (self.ep_len * d).sum(),
                                   ((dxy[:, 0] / dur) * d).sum(), ((self.ep_track / self.ep_len) * d).sum(),
                                   (term.double() * d).sum(), (dxy[:, 0] * d).sum(), (dxy[:, 1] * d).sum()))
        # Command resampling and resets.
        resample = done | ((self.progress % self.cmd_resample) == 0)
        self.cmd = torch.where(resample[:, None], self._sample_cmd(self.N), self.cmd)
        q = torch.where(done[:, None], nq, q)
        v = torch.where(done[:, None], nv, v)
        w = torch.where(done[:, None], 0.0, w)
        action = torch.where(done[:, None], 0.0, action)
        self.progress = torch.where(done, 0, self.progress)
        self.ep_ret = torch.where(done, 0.0, self.ep_ret)
        self.ep_len = torch.where(done, 0.0, self.ep_len)
        self.ep_track = torch.where(done, 0.0, self.ep_track)
        self.ep_x0 = torch.where(done[:, None], q[:, 0:2].detach(), self.ep_x0)
        if self.nfeet:
            self.air_time = torch.where(done[:, None], 0.0, self.air_time)
            self.contact_time = torch.where(done[:, None], 0.0, self.contact_time)
        self.prev_qd = v[:, 6:].detach()  # after resets: the reset state's joint velocity
        if self.fk is not None:
            self._feet_prev = f.feet  # post-step feet (stale for reset worlds, masked next step)
        self.q, self.v, self.w, self.prev_action = q, v, w, action
        info = {"terminated": term, "truncated": trunc, "nonfinite": bad, "obs_before_reset": obs_before_reset}
        return self.observe(q, v, action), rew, done, info

    def pop_stats(self):
        """Returns and clears the episode statistics accumulated since the last call (one host sync)."""
        s = self.stats.cpu().numpy()
        self.stats.zero_()
        n = s[0]
        out = {"episodes": int(n)}
        if n > 0:
            for i, k in enumerate(self.STAT_KEYS[1:], 1):
                out[k] = float(s[i] / n)
        return out


Env.OBS = {n[5:] for n in dir(Env) if n.startswith("_obs_")}
Env.REWARDS = {n[3:] for n in dir(Env) if n.startswith("_r_")}

# ----------------------------------------------------------------------------- tasks

SHAC_OBS = TaskCfg.obs

TASKS: dict[str, TaskCfg] = {
    # SHAC reference Ant (as in locomotion.py).
    "ant": TaskCfg(robot="ant", rewards=(("fwd_vel", 1.0), ("up", 0.1), ("heading", 1.0), ("height_linear", 1.0)),
                   min_height=0.26, max_height=1.0, min_up=0.1),
    # SHAC-style Go1 used in the published study (locomotion.py "go1").
    "go1": TaskCfg(robot="go1", rewards=(("fwd_vel", 1.0), ("up", 0.1), ("heading", 1.0), ("height_shac", 1.0),
                                         ("lateral_vel_l2", -1.0), ("action_rate_l2", -0.02)),
                   action_mode="position", action_scale=0.5, speed_cap=1.5, height_pivot=0.22, height_cap=0.05,
                   min_height=0.18, max_height=0.6, min_up=0.5),
    # SHAC reference Humanoid (as in locomotion.py).
    "humanoid": TaskCfg(robot="humanoid", rewards=(("fwd_vel", 1.0), ("up", 0.1), ("heading", 1.0),
                                                   ("height_shac", 1.0), ("action_l2", -0.002)),
                        height_pivot=0.84, height_cap=0.1, min_height=0.74, max_height=2.1, min_up=0.0),
}

# Clean Isaac-Lab-style resets: spawn 6 cm above contact, no root noise, joints +-0.2 rad,
# joint velocities +-0.1 rad/s, states with > 1 cm floor/self penetration rejected.
CLEAN_RESET = dict(init_bank=20000, spawn_dz=0.06, reset_xy=0.0, reset_z=0.0, reset_tilt=0.0, reset_vel=0.1,
                   reset_vel_joints_only=True)
TASKS["humanoid_clean"] = dataclasses.replace(TASKS["humanoid"], **CLEAN_RESET)
# Survival-oriented humanoid: SHAC terms with displacement progress capped at 1.5 m/s, alive bonus,
# termination at 0.8 m with a death penalty, and non-saturating upright/height shaping.
TASKS["humanoid_surv"] = dataclasses.replace(
    TASKS["humanoid"], **CLEAN_RESET, min_height=0.8, speed_cap=1.5, target_height=1.2,
    rewards=(("fwd_disp", 1.0), ("up", 0.1), ("heading", 1.0), ("height_shac", 1.0), ("action_l2", -0.01),
             ("alive", 2.0), ("termination", -1.0), ("upright_l2", -5.0), ("height_below_l2", -10.0)))

# Isaac Lab velocity-tracking observation set (flat terrain, no height scan).
IL_OBS = (("lin_vel_b", 1.0), ("ang_vel_b", 1.0), ("gravity_b", 1.0), ("command", 1.0), ("joint_pos_rel", 1.0),
          ("joint_vel", 1.0), ("prev_action", 1.0))
# Isaac Lab H1 flat velocity task (H1FlatEnvCfg) on the Isaac-Lab-equivalent H1 (PD servos with IL gains
# and pose, floor-only collisions). Termination: torso_link touches the floor (Isaac Lab base_contact), with a
# pelvis-height backstop. feet_slide and pushes are omitted.
H1_ARMS = ("left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "right_shoulder_pitch",
           "right_shoulder_roll", "right_shoulder_yaw", "right_elbow")
TASKS["h1_vel"] = TaskCfg(
    robot="h1_il", obs=IL_OBS, action_mode="position", action_scale=0.5,
    rewards=(("track_lin_vel_xy_yaw_exp", 1.0), ("track_ang_vel_z_world_exp", 1.0), ("termination", -200.0),
             ("feet_air_time_biped", 1.0), ("ang_vel_xy_l2", -0.05), ("joint_pos_limits", -1.0),
             ("joint_deviation_l1", -0.2), ("joint_deviation2_l1", -0.1), ("flat_orientation_l2", -1.0),
             ("action_rate_l2", -0.005), ("joint_acc_l2", -1.25e-7)),
    track_std=0.5, feet_air_threshold=0.6, foot_bodies=("left_ankle_link", "right_ankle_link"),
    term_bodies=("torso_link",), limit_joints=("left_ankle", "right_ankle"),
    deviation_joints=("left_hip_yaw", "left_hip_roll", "right_hip_yaw", "right_hip_roll") + H1_ARMS,
    deviation_joints2=("torso",), soft_limit_factor=0.9,
    min_height=0.35, max_height=2.0, min_up=-1.0, episode_length=1000,
    init_bank=20000, spawn_dz=0.02, reset_xy=0.0, reset_z=0.0, reset_tilt=0.0, reset_yaw=math.pi,
    reset_joint=0.0, reset_joint_scale=(1.0, 1.0), reset_vel=0.0, reset_root_vel=0.5,
    cmd_vx=(0.0, 1.0), cmd_vy=(0.0, 0.0), cmd_wz=(-1.0, 1.0), cmd_resample_s=10.0, cmd_standing_frac=0.02)

# Isaac Lab G1 flat velocity task (G1FlatEnvCfg) on the mjlab G1; the policy controls the 12 leg joints and
# observes them (48 observations), the waist and arms hold the home pose with their servos.
G1_LEGS = tuple(f"{s}_{j}_joint" for s in ("left", "right")
                for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"))
TASKS["g1_vel"] = TaskCfg(
    robot="g1", obs=IL_OBS, action_mode="position", action_scale=0.5, act_joints=G1_LEGS, obs_act_joints_only=True,
    rewards=(("track_lin_vel_xy_yaw_exp", 1.0), ("track_ang_vel_z_world_exp", 1.0), ("termination", -200.0),
             ("lin_vel_z_l2", -0.2), ("ang_vel_xy_l2", -0.05), ("joint_torques_l2", -2e-6), ("joint_acc_l2", -1e-7),
             ("action_rate_l2", -0.005), ("flat_orientation_l2", -1.0), ("joint_pos_limits", -1.0),
             ("joint_deviation_l1", -0.1), ("feet_air_time_biped", 0.75)),
    track_std=0.5, feet_air_threshold=0.4, foot_bodies=("left_ankle_roll_link", "right_ankle_roll_link"),
    term_bodies=("torso_link", "pelvis"),
    limit_joints=("left_ankle_pitch_joint", "left_ankle_roll_joint", "right_ankle_pitch_joint",
                  "right_ankle_roll_joint"),
    deviation_joints=("left_hip_roll_joint", "left_hip_yaw_joint", "right_hip_roll_joint", "right_hip_yaw_joint"),
    soft_limit_factor=0.9, min_height=0.3, max_height=1.5, min_up=-1.0, episode_length=1000,
    init_bank=20000, spawn_dz=0.02, reset_xy=0.0, reset_z=0.0, reset_tilt=0.0, reset_yaw=math.pi,
    reset_joint=0.0, reset_joint_scale=(1.0, 1.0), reset_vel=0.0, reset_root_vel=0.5,
    cmd_vx=(0.0, 1.0), cmd_vy=(-0.5, 0.5), cmd_wz=(-1.0, 1.0), cmd_resample_s=10.0, cmd_standing_frac=0.1)

# Gait-prior variants: clock observation plus differentiable foot-height, stance-slip and Raibert terms.
GAIT_OBS = IL_OBS + (("phase", 1.0),)
GAIT_REW = (("gait_height", -40.0), ("gait_slip", -2.0), ("raibert", -10.0))
TASKS["h1_gait"] = dataclasses.replace(
    TASKS["h1_vel"], obs=GAIT_OBS, rewards=TASKS["h1_vel"].rewards + GAIT_REW, gait_period=0.8,
    foot_points=(("left_ankle_link", (0.05, 0.0, -0.06)), ("right_ankle_link", (0.05, 0.0, -0.06))),
    gait_offsets=(0.0, 0.5), gait_duty=0.6, gait_height=0.08)
TASKS["g1_gait"] = dataclasses.replace(
    TASKS["g1_vel"], obs=GAIT_OBS, rewards=TASKS["g1_vel"].rewards + GAIT_REW, gait_period=0.8,
    foot_points=(("left_ankle_roll_link", (0.04, 0.0, -0.035)), ("right_ankle_roll_link", (0.04, 0.0, -0.035))),
    gait_offsets=(0.0, 0.5), gait_duty=0.6, gait_height=0.08)

# G1 with stronger foot-height and placement terms and H1's command ranges (no lateral commands, 2% standing),
# the configuration with which the G1 learned to walk (run g5_g1_strongprior; Isaac Lab gains via model_edit g1_il).
TASKS["g1_gait_strong"] = dataclasses.replace(
    TASKS["g1_gait"], rewards=TASKS["g1_vel"].rewards + (("gait_height", -120.0), ("gait_slip", -2.0), ("raibert", -20.0)),
    cmd_vy=(0.0, 0.0), cmd_standing_frac=0.02)

# Classic MuJoCo Humanoid (torque control) with the survival terms and the gait prior; the Raibert target uses
# zero command (forward progress is rewarded by fwd_disp).
TASKS["humanoid_gait"] = dataclasses.replace(
    TASKS["humanoid_surv"], obs=TASKS["humanoid_surv"].obs + (("phase", 1.0),),
    rewards=TASKS["humanoid_surv"].rewards + (("gait_height", -40.0), ("gait_slip", -2.0), ("raibert", -10.0)),
    gait_period=0.8, foot_points=(("foot_left", (0.0, 0.0, -0.03)), ("foot_right", (0.0, 0.0, -0.03))),
    gait_offsets=(0.0, 0.5), gait_duty=0.6, gait_height=0.08, raibert_gain=0.0)

# Isaac Lab Go1 flat velocity task (UnitreeGo1FlatEnvCfg) on the Isaac-Lab-equivalent Go1. Yaw-rate commands
# are sampled directly (no heading controller); the action is tanh-squashed with scale 0.5 rad.
TASKS["go1_vel"] = TaskCfg(
    robot="go1_il", obs=IL_OBS, action_mode="position", action_scale=0.5,
    rewards=(("track_lin_vel_xy_exp", 1.5), ("track_ang_vel_z_exp", 0.75), ("lin_vel_z_l2", -2.0),
             ("ang_vel_xy_l2", -0.05), ("joint_torques_l2", -2e-4), ("joint_acc_l2", -2.5e-7),
             ("action_rate_l2", -0.01), ("flat_orientation_l2", -2.5), ("feet_air_time", 0.25)),
    track_std=0.5, feet_air_threshold=0.5, foot_geoms=("FR", "FL", "RR", "RL"), term_bodies=("trunk",),
    min_height=0.1, max_height=1.0, min_up=-1.0, episode_length=1000,
    init_bank=20000, spawn_dz=0.02, reset_xy=0.0, reset_z=0.0, reset_tilt=0.0, reset_yaw=math.pi,
    reset_joint=0.0, reset_joint_scale=(0.5, 1.5), reset_vel=0.0, reset_root_vel=0.5,
    cmd_vx=(-1.0, 1.0), cmd_vy=(-1.0, 1.0), cmd_wz=(-1.0, 1.0), cmd_resample_s=10.0, cmd_standing_frac=0.02)
# Go1 with a trot clock (FR+RL / FL+RR), period 0.5 s, duty 0.5, swing height 0.08 m.
TASKS["go1_gait"] = dataclasses.replace(
    TASKS["go1_vel"], obs=IL_OBS + (("phase", 1.0),),
    rewards=TASKS["go1_vel"].rewards + (("gait_height", -40.0), ("gait_slip", -2.0), ("raibert", -10.0)),
    gait_period=0.5, gait_duty=0.5, gait_height=0.08, gait_offsets=(0.0, 0.5, 0.5, 0.0),
    foot_points=(("FR_calf", (0.0, 0.0, -0.213)), ("FL_calf", (0.0, 0.0, -0.213)),
                 ("RR_calf", (0.0, 0.0, -0.213)), ("RL_calf", (0.0, 0.0, -0.213))))
# Stage A: forward commands only (comparable with the forward-speed task of the earlier study).
TASKS["go1_vel_fwd"] = dataclasses.replace(TASKS["go1_vel"], cmd_vx=(0.0, 1.5), cmd_vy=(0.0, 0.0), cmd_wz=(0.0, 0.0))


def make_task(name: str, **overrides) -> TaskCfg:
    return dataclasses.replace(TASKS[name], **overrides)
