"""Differentiable locomotion environments on top of ``DiffSim``.

Observation, reward, termination, and reset randomisation follow the Ant and
Humanoid environments of the SHAC reference implementation (DiffRL), converted
from its y-up convention to MuJoCo's z-up convention. The same definitions are
used for every integrator and timestep, so only the simulator changes between
compared configurations.
"""

from __future__ import annotations

import dataclasses
import math

import mujoco
import torch

from diffsim import DiffSim, SimConfig
from robots import ROBOTS, Robot


@dataclasses.dataclass(frozen=True)
class TaskSpec:
    # Height reward: x = clip(h - pivot, -1, cap); reward = 10 x if x > 0 else -200 x^2
    # (SHAC Humanoid). ``linear_height`` instead uses h - min_height (SHAC Ant).
    linear_height: bool
    height_pivot: float
    height_cap: float
    action_penalty: float
    action_rate_penalty: float
    action_scale: float  # position-control action scale (rad)
    # Optional forward-speed cap (m/s): the velocity term becomes min(v_x, speed_target).
    speed_target: float | None = None
    lateral_penalty: float = 0.0  # weight of v_y^2
    velocity_weight: float = 1.0  # weight of the forward-velocity term
    up_weight: float = 0.1  # weight of the uprightness term (SHAC reference: 0.1)
    angvel_penalty: float = 0.0  # weight of the squared torso roll/pitch rate (body frame)
    # Optional gait clock: the policy observes (sin, cos) of a phase with period gait_period (s),
    # and the reward penalises gait_weight * squared deviation of the hip/knee pitch joints
    # (indices gait_joints = (hip_l, knee_l, hip_r, knee_r) into the joint coordinates) from an
    # alternating stepping pattern around the nominal pose.
    gait_period: float = 0.0
    gait_weight: float = 0.0
    gait_hip_amp: float = 0.2
    gait_knee_amp: float = 0.4
    gait_joints: tuple = ()


TASKS = {
    "ant": TaskSpec(linear_height=True, height_pivot=0.0, height_cap=0.0, action_penalty=0.0,
                    action_rate_penalty=0.0, action_scale=0.0),
    "humanoid": TaskSpec(linear_height=False, height_pivot=0.84, height_cap=0.1, action_penalty=0.002,
                         action_rate_penalty=0.0, action_scale=0.0),
    "go1": TaskSpec(linear_height=False, height_pivot=0.22, height_cap=0.05, action_penalty=0.0,
                    action_rate_penalty=0.02, action_scale=0.5, speed_target=1.5, lateral_penalty=1.0),
    "humanoid_ref": TaskSpec(linear_height=False, height_pivot=0.84, height_cap=0.1, action_penalty=0.002,
                             action_rate_penalty=0.0, action_scale=0.0),
    "go1_kv": TaskSpec(linear_height=False, height_pivot=0.22, height_cap=0.05, action_penalty=0.0,
                       action_rate_penalty=0.02, action_scale=0.5, speed_target=1.5, lateral_penalty=1.0),
    "h1_pd": TaskSpec(linear_height=False, height_pivot=0.85, height_cap=0.1, action_penalty=0.0,
                      action_rate_penalty=0.02, action_scale=0.5, gait_joints=(2, 3, 7, 8)),
    "h1": TaskSpec(linear_height=False, height_pivot=0.85, height_cap=0.1, action_penalty=0.002,
                   action_rate_penalty=0.0, action_scale=0.0),
}


def nominal_qpos(robot: Robot, sim: DiffSim) -> torch.Tensor:
    if robot.name == "ant":
        m = sim.mjm
        nid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_NUMERIC, "init_qpos")
        adr = m.numeric_adr[nid]
        return torch.tensor(m.numeric_data[adr:adr + m.nq], dtype=torch.float32, device=sim.torch_device)
    return sim.init_qpos.clone()


def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw), -1)


class LocomotionEnv:
    """Batched differentiable environment. State tensors live on the GPU."""

    def __init__(self, robot: str, sim_cfg: SimConfig, num_envs: int, episode_length: int = 1000,
                 stochastic_init: bool = True, seed: int = 0, device: str = "cuda:0",
                 task_overrides: dict | None = None, init_noise_scale: float = 1.0):
        self.init_noise_scale = init_noise_scale
        self.robot = ROBOTS[robot]
        self.task = dataclasses.replace(TASKS[robot], **(task_overrides or {}))
        self.sim = DiffSim(sim_cfg, num_envs, device=device)
        self.num_envs = num_envs
        self.episode_length = episode_length
        self.stochastic_init = stochastic_init
        self.dev = self.sim.torch_device
        self.gen = torch.Generator(device=self.dev).manual_seed(seed)
        self.q0 = nominal_qpos(self.robot, self.sim)
        self.home = self.q0[7:].clone()
        self.nu = self.sim.nu
        self.num_obs = 1 + 4 + 3 + 3 + 2 * (self.sim.nq - 7) + 2 + self.nu + (2 if self.task.gait_period else 0)
        self.q = self.q0.expand(num_envs, -1).clone()
        self.v = torch.zeros(num_envs, self.sim.nv, device=self.dev)
        self.w = torch.zeros_like(self.v)
        self.prev_action = torch.zeros(num_envs, self.nu, device=self.dev)
        self.progress = torch.zeros(num_envs, dtype=torch.long, device=self.dev)
        self.nonfinite_resets = 0
        self.reset_all()

    # -------------------------------------------------------------- reset
    def _sample_init(self, n):
        q = self.q0.expand(n, -1).clone()
        v = torch.zeros(n, self.sim.nv, device=self.dev)
        if self.stochastic_init:
            k = self.init_noise_scale
            u = lambda *s: k * (2 * torch.rand(*s, device=self.dev, generator=self.gen) - 1)  # noqa: E731
            q[:, 0:3] += 0.1 * u(n, 3)
            angle = (math.pi / 24.0) * u(n)
            axis = torch.nn.functional.normalize(0.5 * u(n, 3), dim=-1)
            dq = torch.cat((torch.cos(0.5 * angle)[:, None], torch.sin(0.5 * angle)[:, None] * axis), -1)
            q[:, 3:7] = quat_mul(dq, q[:, 3:7])
            q[:, 7:] += 0.2 * u(n, self.sim.nq - 7)
            lo = torch.tensor(self.sim.mjm.jnt_range[1:, 0], dtype=torch.float32, device=self.dev)
            hi = torch.tensor(self.sim.mjm.jnt_range[1:, 1], dtype=torch.float32, device=self.dev)
            lim = torch.tensor(self.sim.mjm.jnt_limited[1:].astype(bool), device=self.dev)
            q[:, 7:] = torch.where(lim, torch.minimum(torch.maximum(q[:, 7:], lo), hi), q[:, 7:])
            v += 0.25 * u(n, self.sim.nv)
        return q, v

    def reset_all(self):
        self.q, self.v = self._sample_init(self.num_envs)
        self.w = torch.zeros_like(self.v)
        self.prev_action.zero_()
        self.progress.zero_()

    def detach(self):
        self.q, self.v, self.w = self.q.detach(), self.v.detach(), self.w.detach()
        self.prev_action = self.prev_action.detach()

    # -------------------------------------------------------------- features
    @staticmethod
    def up_z(q):
        return 1.0 - 2.0 * (q[:, 4] ** 2 + q[:, 5] ** 2)

    @staticmethod
    def heading_x(q):
        return 1.0 - 2.0 * (q[:, 5] ** 2 + q[:, 6] ** 2)

    def phase(self):
        return 2.0 * math.pi * self.progress.float() * self.sim.cfg.control_dt / self.task.gait_period

    def observe(self, q, v, prev_action):
        parts = [q[:, 2:3], q[:, 3:7], v[:, 0:3], v[:, 3:6], q[:, 7:], 0.1 * v[:, 6:],
                 self.up_z(q)[:, None], self.heading_x(q)[:, None], prev_action]
        if self.task.gait_period:
            ph = self.phase()
            parts.append(torch.stack((torch.sin(ph), torch.cos(ph)), -1))
        return torch.cat(parts, -1)

    def gait_error(self, q):
        t = self.task
        hl, kl, hr, kr = t.gait_joints
        ph = self.phase()
        err = torch.zeros(q.shape[0], device=q.device)
        for hip, knee, off in ((hl, kl, 0.0), (hr, kr, math.pi)):
            s = torch.sin(ph + off)
            hip_ref = self.home[hip] - t.gait_hip_amp * s
            knee_ref = self.home[knee] + t.gait_knee_amp * s.clamp(min=0.0)
            err = err + (q[:, 7 + hip] - hip_ref).square() + (q[:, 7 + knee] - knee_ref).square()
        return err

    def reward(self, q, v, action, prev_action):
        t = self.task
        h = q[:, 2]
        vx = v[:, 0] if t.speed_target is None else torch.clamp(v[:, 0], max=t.speed_target)
        r = t.velocity_weight * vx + t.up_weight * self.up_z(q) + self.heading_x(q)
        if t.angvel_penalty:
            r = r - t.angvel_penalty * v[:, 3:5].square().sum(-1)
        if t.gait_period and t.gait_weight:
            r = r - t.gait_weight * self.gait_error(q)
        if t.lateral_penalty:
            r = r - t.lateral_penalty * v[:, 1].square()
        if t.linear_height:
            r = r + (h - self.robot.min_height)
        else:
            x = (h - t.height_pivot).clamp(-1.0, t.height_cap)
            r = r + torch.where(x < 0, -200.0 * x * x, 10.0 * x)
        if t.action_penalty:
            r = r - t.action_penalty * action.square().sum(-1)
        if t.action_rate_penalty:
            r = r - t.action_rate_penalty * (action - prev_action).square().sum(-1)
        return r

    def terminated(self, q, v):
        r = self.robot
        bad = ~torch.isfinite(q).all(1) | ~torch.isfinite(v).all(1) | (v.abs().amax(1) > 1e4)
        fell = (q[:, 2] < r.min_height) | (q[:, 2] > r.max_height) | (self.up_z(q) < r.min_up)
        return fell | bad, bad

    def ctrl(self, action):
        return self.sim.action_to_ctrl(action, self.robot.position_control, self.home, self.task.action_scale)

    def obs(self):
        return self.observe(self.q, self.v, self.prev_action)

    # -------------------------------------------------------------- step
    def step(self, action, differentiable: bool = True):
        """Returns obs, reward, done, info; resets finished envs in place."""
        ctrl = self.ctrl(action)
        if differentiable:
            q, v, w = self.sim.step(self.q, self.v, ctrl, self.w)
        else:
            q, v, w = self.sim.step_nograd(self.q, self.v, ctrl.detach(), self.w)
        self.progress += 1
        term, bad = self.terminated(q, v)
        # Replace non-finite states before they reach reward or observation.
        if bad.any():
            self.nonfinite_resets += int(bad.sum().item())
            safe_q, safe_v = self._sample_init(self.num_envs)
            q = torch.where(bad[:, None], safe_q, q)
            v = torch.where(bad[:, None], safe_v, v)
            w = torch.where(bad[:, None], 0.0, w)
        rew = self.reward(q, v, action, self.prev_action)
        rew = torch.where(bad, 0.0, rew)
        obs_before_reset = self.observe(q, v, action)
        trunc = self.progress >= self.episode_length
        done = term | trunc
        info = {"terminated": term, "truncated": trunc & ~term, "nonfinite": bad,
                "obs_before_reset": obs_before_reset, "x": q[:, 0].detach(), "y": q[:, 1].detach(), "vx": v[:, 0].detach()}
        if done.any():
            nq, nv = self._sample_init(self.num_envs)
            q = torch.where(done[:, None], nq, q)
            v = torch.where(done[:, None], nv, v)
            w = torch.where(done[:, None], 0.0, w)
            action = torch.where(done[:, None], 0.0, action)
            self.progress = torch.where(done, 0, self.progress)
        self.q, self.v, self.w, self.prev_action = q, v, w, action
        return self.observe(q, v, action), rew, done, info
