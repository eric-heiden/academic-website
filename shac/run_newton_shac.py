from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
import warp as wp

REPO_ROOT = Path(__file__).resolve().parents[2]
DIFFRL_ROOT = REPO_ROOT / "DiffRL"
if str(DIFFRL_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFRL_ROOT))

import newton
import newton.examples
from newton.solvers import SolverMuJoCo

from models.actor import ActorDeterministicMLP, ActorStochasticMLP
from models.critic import CriticMLP
from utils.running_mean_std import RunningMeanStd


@dataclass
class StepContext:
    env: "NewtonMuJoCoTorchEnv"


@dataclass
class CartpoleRewardWeights:
    pole_angle: float = 1.0
    pole_velocity: float = 0.1
    cart_position: float = 0.05
    cart_velocity: float = 0.1
    action: float = 0.0


ANT_START_JOINT_Q = (0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0)
ANT_START_ROT = (math.sin(-0.25 * math.pi), 0.0, 0.0, math.cos(-0.25 * math.pi))
ANT_TERMINATION_HEIGHT = 0.27
ANT_MAX_HEALTHY_HEIGHT = 1.5
ANT_HEIGHT_REWARD_CAP = 0.6
ANT_INVALID_PENALTY = -50.0
ANT_JOINT_VEL_OBS_SCALING = 0.1
ANT_ACTION_PENALTY = 0.0


def normalize_vec(x: torch.Tensor, eps: float = 1.0e-9) -> torch.Tensor:
    return x / x.norm(p=2, dim=-1, keepdim=True).clamp(min=eps)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)
    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return torch.stack([x, y, z, w], dim=-1).view(shape)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([-q[..., :3], q[..., 3:4]], dim=-1)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w.square() - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c


def quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    theta = (angle / 2.0).unsqueeze(-1)
    xyz = normalize_vec(axis) * theta.sin()
    return normalize_vec(torch.cat([xyz, theta.cos()], dim=-1))


class NewtonMuJoCoStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, qd: torch.Tensor, joint_f: torch.Tensor, step_ctx: StepContext):
        env = step_ctx.env
        q_in = q.detach().contiguous()
        qd_in = qd.detach().contiguous()
        joint_f_in = joint_f.detach().contiguous()
        q_out = torch.empty_like(q_in)
        qd_out = torch.empty_like(qd_in)
        env.step_warp(q_in, qd_in, joint_f_in, q_out, qd_out, requires_grad=False)
        ctx.step_ctx = step_ctx
        ctx.save_for_backward(q_in, qd_in, joint_f_in)
        return q_out, qd_out

    @staticmethod
    def backward(ctx, grad_q_out: torch.Tensor | None, grad_qd_out: torch.Tensor | None):
        q, qd, joint_f = ctx.saved_tensors
        env = ctx.step_ctx.env

        q_req = q.detach().clone().requires_grad_(True)
        qd_req = qd.detach().clone().requires_grad_(True)
        joint_f_req = joint_f.detach().clone().requires_grad_(True)
        q_out = torch.empty_like(q_req)
        qd_out = torch.empty_like(qd_req)

        env.zero_solver_buffers()
        with wp.Tape() as tape:
            arrays = env.step_warp(
                q_req,
                qd_req,
                joint_f_req,
                q_out,
                qd_out,
                requires_grad=True,
                zero_buffers=False,
            )

        grads = {}
        if grad_q_out is not None:
            grads[arrays["q_out"]] = wp.from_torch(grad_q_out.contiguous().view(-1), dtype=wp.float32)
        if grad_qd_out is not None:
            grads[arrays["qd_out"]] = wp.from_torch(grad_qd_out.contiguous().view(-1), dtype=wp.float32)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Running the tape backwards may produce incorrect gradients.*",
                category=UserWarning,
            )
            tape.backward(grads=grads)

        def torch_grad(name: str, template: torch.Tensor) -> torch.Tensor:
            grad = tape.gradients.get(arrays[name])
            if grad is None:
                return torch.zeros_like(template)
            return wp.to_torch(grad).reshape_as(template).clone()

        return torch_grad("q", q), torch_grad("qd", qd), torch_grad("joint_f", joint_f), None


class NewtonMuJoCoTorchEnv:
    def __init__(
        self,
        *,
        env_name: str,
        num_envs: int,
        device: str,
        dt: float,
        force_scale: float,
        contact_backend: str,
        cartpole_reward: CartpoleRewardWeights | None = None,
    ):
        self.env_name = env_name
        self.num_envs = num_envs
        self.torch_device = torch.device(device)
        self.wp_device = wp.device_from_torch(self.torch_device)
        self.dt = dt
        self.force_scale = force_scale
        self.contact_backend = contact_backend
        self.cartpole_reward = cartpole_reward or CartpoleRewardWeights()

        if env_name == "cartpole":
            self._build_cartpole()
        elif env_name == "ant":
            self._build_ant()
        else:
            raise ValueError(f"unknown env_name: {env_name}")

        use_contacts = contact_backend != "none"
        self.solver = SolverMuJoCo(
            self.model,
            requires_grad=True,
            disable_contacts=not use_contacts,
            use_mujoco_contacts=contact_backend != "newton",
            integrator="euler",
            solver="newton",
            iterations=8,
            ls_iterations=8,
            update_data_interval=1,
            nconmax=128 if use_contacts else None,
            njmax=512 if use_contacts else None,
        )
        self.contacts = self.model.contacts() if contact_backend == "newton" else None
        self.step_ctx = StepContext(self)
        self.q_dim = self.model.joint_coord_count // self.num_envs
        self.qd_dim = self.model.joint_dof_count // self.num_envs
        self.start_q = torch.as_tensor(
            self.model.joint_q.numpy().reshape(self.num_envs, self.q_dim),
            dtype=torch.float32,
            device=self.torch_device,
        )
        self.start_qd = torch.as_tensor(
            self.model.joint_qd.numpy().reshape(self.num_envs, self.qd_dim),
            dtype=torch.float32,
            device=self.torch_device,
        )
        self.ant_start_rotation = torch.tensor(ANT_START_ROT, dtype=torch.float32, device=self.torch_device)
        self.ant_inv_start_rotation = quat_conjugate(self.ant_start_rotation).view(1, 4).repeat(self.num_envs, 1)
        self.ant_basis_x = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.torch_device).repeat(
            self.num_envs, 1
        )
        self.ant_basis_y = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=self.torch_device).repeat(
            self.num_envs, 1
        )
        self.ant_targets = torch.tensor([10000.0, 0.0, 0.0], dtype=torch.float32, device=self.torch_device).repeat(
            self.num_envs, 1
        )
        self.num_obs = int(self.observe(self.start_q, self.start_qd).shape[-1])

    def _build_cartpole(self) -> None:
        source = newton.ModelBuilder(up_axis="Y")
        SolverMuJoCo.register_custom_attributes(source)
        source.default_joint_cfg.armature = 0.1
        source.add_urdf(
            str(DIFFRL_ROOT / "envs" / "assets" / "cartpole.urdf"),
            floating=False,
            up_axis="Y",
            xform=wp.transform(
                wp.vec3(0.0, 0.0, 0.0),
                wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -0.5 * math.pi),
            ),
        )
        source.joint_q[-1] = -math.pi

        builder = newton.ModelBuilder(up_axis="Y")
        SolverMuJoCo.register_custom_attributes(builder)
        builder.replicate(source, self.num_envs, spacing=(2.0, 0.0, 0.0))
        self.model = builder.finalize(device=self.wp_device, requires_grad=True)
        self.num_actions = 1

    def _build_ant(self) -> None:
        source = newton.ModelBuilder(up_axis="Y")
        SolverMuJoCo.register_custom_attributes(source)
        source.default_shape_cfg.ke = 4.0e4
        source.default_shape_cfg.kd = 1.0e4
        source.default_shape_cfg.kf = 3.0e3
        source.default_shape_cfg.mu = 0.75
        source.default_joint_cfg.limit_ke = 1.0e3
        source.default_joint_cfg.limit_kd = 1.0e1
        source.add_mjcf(str(DIFFRL_ROOT / "envs" / "assets" / "ant.xml"), up_axis="Z", armature_scale=50.0)
        source.joint_q[7:15] = ANT_START_JOINT_Q
        source.joint_target_q[7:15] = ANT_START_JOINT_Q
        source.shape_material_ke = [4.0e4] * len(source.shape_material_ke)
        source.shape_material_kd = [1.0e4] * len(source.shape_material_kd)
        source.shape_material_kf = [3.0e3] * len(source.shape_material_kf)
        source.shape_material_mu = [0.75] * len(source.shape_material_mu)

        builder = newton.ModelBuilder(up_axis="Y")
        SolverMuJoCo.register_custom_attributes(builder)
        builder.replicate(source, self.num_envs, spacing=(0.0, 0.0, 0.0))
        ground_cfg = newton.ModelBuilder.ShapeConfig(ke=4.0e4, kd=1.0e4, kf=3.0e3, mu=0.75)
        builder.add_ground_plane(cfg=ground_cfg)
        self.model = builder.finalize(device=self.wp_device, requires_grad=True)
        self.num_actions = 8

    def zero_solver_buffers(self) -> None:
        data = self.solver.mjw_data
        data.qacc_warmstart.zero_()
        data.qfrc_applied.zero_()
        data.ctrl.zero_()
        data.act.zero_()
        data.xfrc_applied.zero_()

    def step_warp(
        self,
        q: torch.Tensor,
        qd: torch.Tensor,
        joint_f: torch.Tensor,
        q_out: torch.Tensor,
        qd_out: torch.Tensor,
        *,
        requires_grad: bool,
        zero_buffers: bool = True,
    ) -> dict[str, wp.array]:
        if zero_buffers:
            self.zero_solver_buffers()
        state_in = self.model.state(requires_grad=requires_grad)
        state_out = self.model.state(requires_grad=requires_grad)
        control = self.model.control(requires_grad=requires_grad)

        q_wp = wp.from_torch(
            q.contiguous().view(-1), dtype=wp.float32, requires_grad=requires_grad, retain_grad=requires_grad
        )
        qd_wp = wp.from_torch(
            qd.contiguous().view(-1), dtype=wp.float32, requires_grad=requires_grad, retain_grad=requires_grad
        )
        f_wp = wp.from_torch(
            joint_f.contiguous().view(-1),
            dtype=wp.float32,
            requires_grad=requires_grad,
            retain_grad=requires_grad,
        )
        q_out_wp = wp.from_torch(
            q_out.contiguous().view(-1), dtype=wp.float32, requires_grad=requires_grad, retain_grad=requires_grad
        )
        qd_out_wp = wp.from_torch(
            qd_out.contiguous().view(-1), dtype=wp.float32, requires_grad=requires_grad, retain_grad=requires_grad
        )

        state_in.joint_q = q_wp
        state_in.joint_qd = qd_wp
        state_out.joint_q = q_out_wp
        state_out.joint_qd = qd_out_wp
        control.joint_f = f_wp
        contacts = None
        if self.contact_backend == "newton":
            newton.eval_fk(self.model, state_in.joint_q, state_in.joint_qd, state_in)
            contacts = self.model.collide(state_in, self.contacts)
        self.solver.step(state_in, state_out, control, contacts, self.dt)
        wp.synchronize()
        self.last_state = state_out
        return {"q": q_wp, "qd": qd_wp, "joint_f": f_wp, "q_out": q_out_wp, "qd_out": qd_out_wp}

    def step(self, q: torch.Tensor, qd: torch.Tensor, joint_f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return NewtonMuJoCoStep.apply(q, qd, joint_f, self.step_ctx)

    def reset(self, noise: float = 0.0, stochastic_init: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.start_q.clone()
        qd = self.start_qd.clone()
        if self.env_name == "ant" and stochastic_init:
            q, qd = self._randomize_ant_reset(q, qd)
        elif stochastic_init:
            q = q + math.pi * (torch.rand_like(q) - 0.5)
            qd = qd + 0.5 * (torch.rand_like(qd) - 0.5)
        elif noise > 0.0:
            q = q + noise * torch.randn_like(q)
            qd = qd + 0.25 * noise * torch.randn_like(qd)
            if self.env_name == "ant":
                q[:, 3:7] = normalize_vec(q[:, 3:7])
        return q, qd

    def _randomize_ant_reset(self, q: torch.Tensor, qd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q[:, 0:3] = q[:, 0:3] + 0.1 * (torch.rand((q.shape[0], 3), device=self.torch_device) - 0.5) * 2.0
        angle = (torch.rand(q.shape[0], device=self.torch_device) - 0.5) * math.pi / 12.0
        axis = normalize_vec(torch.rand((q.shape[0], 3), device=self.torch_device) - 0.5)
        q[:, 3:7] = normalize_vec(quat_mul(q[:, 3:7], quat_from_angle_axis(angle, axis)))
        q[:, 7:] = q[:, 7:] + 0.2 * (torch.rand_like(q[:, 7:]) - 0.5) * 2.0
        qd[:] = 0.5 * (torch.rand_like(qd) - 0.5)
        return q, qd

    def reset_done(
        self,
        q: torch.Tensor,
        qd: torch.Tensor,
        env_ids: torch.Tensor,
        *,
        stochastic_init: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if env_ids.numel() == 0:
            return q, qd
        reset_q = self.start_q[env_ids].clone()
        reset_qd = self.start_qd[env_ids].clone()
        if self.env_name == "ant" and stochastic_init:
            reset_q, reset_qd = self._randomize_ant_reset(reset_q, reset_qd)
        elif stochastic_init:
            reset_q = reset_q + math.pi * (torch.rand_like(reset_q) - 0.5)
            reset_qd = reset_qd + 0.5 * (torch.rand_like(reset_qd) - 0.5)

        mask_q = torch.zeros((self.num_envs, 1), dtype=torch.bool, device=self.torch_device)
        mask_q[env_ids] = True
        mask_qd = mask_q
        q_next = torch.where(mask_q, self.start_q, q)
        qd_next = torch.where(mask_qd, self.start_qd, qd)
        q_next = q_next.clone()
        qd_next = qd_next.clone()
        q_next[env_ids] = reset_q
        qd_next[env_ids] = reset_qd
        return q_next, qd_next

    def observe(self, q: torch.Tensor, qd: torch.Tensor, prev_action: torch.Tensor | None = None) -> torch.Tensor:
        if self.env_name == "cartpole":
            x = q[:, 0:1]
            theta = q[:, 1:2]
            xdot = qd[:, 0:1]
            theta_dot = qd[:, 1:2]
            return torch.cat([x, xdot, torch.sin(theta), torch.cos(theta), theta_dot], dim=-1)

        if prev_action is None:
            prev_action = torch.zeros((q.shape[0], self.num_actions), dtype=torch.float32, device=self.torch_device)

        torso_pos = q[:, 0:3]
        torso_rot = normalize_vec(q[:, 3:7])
        lin_vel = qd[:, 0:3]
        ang_vel = qd[:, 3:6]
        to_target = self.ant_targets[: q.shape[0]] + self.start_q[: q.shape[0], 0:3] - torso_pos
        to_target = to_target.clone()
        to_target[:, 1] = 0.0
        target_dirs = normalize_vec(to_target)
        torso_quat = quat_mul(torso_rot, self.ant_inv_start_rotation[: q.shape[0]])
        up_vec = quat_rotate(torso_quat, self.ant_basis_y[: q.shape[0]])
        heading_vec = quat_rotate(torso_quat, self.ant_basis_x[: q.shape[0]])
        heading_alignment = (heading_vec * target_dirs).sum(dim=-1, keepdim=True)
        return torch.cat(
            [
                torso_pos[:, 1:2],
                torso_rot,
                lin_vel,
                ang_vel,
                q[:, 7:],
                ANT_JOINT_VEL_OBS_SCALING * qd[:, 6:],
                up_vec[:, 1:2],
                heading_alignment,
                prev_action.clone(),
            ],
            dim=-1,
        )

    def action_to_joint_f(self, action: torch.Tensor) -> torch.Tensor:
        joint_f = torch.zeros((self.num_envs, self.qd_dim), dtype=torch.float32, device=self.torch_device)
        if self.env_name == "cartpole":
            joint_f[:, 0] = action[:, 0] * self.force_scale
        else:
            joint_f[:, 6 : 6 + self.num_actions] = action[:, : self.num_actions] * self.force_scale
        return joint_f

    def reward(
        self, q: torch.Tensor, qd: torch.Tensor, action: torch.Tensor, obs: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.env_name == "cartpole":
            x = q[:, 0]
            theta = torch.atan2(torch.sin(q[:, 1]), torch.cos(q[:, 1]))
            xdot = qd[:, 0]
            theta_dot = qd[:, 1]
            weights = self.cartpole_reward
            return -(
                weights.pole_angle * theta.square()
                + weights.pole_velocity * theta_dot.square()
                + weights.cart_position * x.square()
                + weights.cart_velocity * xdot.square()
                + weights.action * action[:, 0].square()
            )

        if obs is None:
            obs = self.observe(q, qd, action)
        progress_reward = obs[:, 5]
        up_reward = 0.1 * obs[:, 27]
        heading_reward = obs[:, 28]
        height_reward = torch.clamp(obs[:, 0] - ANT_TERMINATION_HEIGHT, max=ANT_HEIGHT_REWARD_CAP)
        return progress_reward + up_reward + heading_reward + height_reward + ANT_ACTION_PENALTY * action.square().sum(
            dim=-1
        )

    def done(self, q: torch.Tensor, progress: torch.Tensor, episode_length: int) -> torch.Tensor:
        done = progress >= episode_length
        if self.env_name == "ant":
            done = torch.logical_or(done, self.fallen_state(q))
        return done

    def fallen_state(self, q: torch.Tensor) -> torch.Tensor:
        if self.env_name != "ant":
            return torch.zeros(q.shape[0], dtype=torch.bool, device=q.device)
        finite = torch.isfinite(q).all(dim=-1)
        return torch.logical_and(finite, q[:, 1] < ANT_TERMINATION_HEIGHT)

    def invalid_state(self, q: torch.Tensor, qd: torch.Tensor) -> torch.Tensor:
        invalid = torch.logical_or(~torch.isfinite(q).all(dim=-1), ~torch.isfinite(qd).all(dim=-1))
        if self.env_name == "ant":
            invalid = torch.logical_or(invalid, q[:, 1] > ANT_MAX_HEALTHY_HEIGHT)
            invalid = torch.logical_or(invalid, q[:, 0].abs() > 100.0)
            invalid = torch.logical_or(invalid, q[:, 2].abs() > 100.0)
            invalid = torch.logical_or(invalid, qd.abs().amax(dim=-1) > 100.0)
        else:
            invalid = torch.logical_or(invalid, q.abs().amax(dim=-1) > 1000.0)
            invalid = torch.logical_or(invalid, qd.abs().amax(dim=-1) > 1000.0)
        return invalid

    def sanitize_state(
        self,
        q: torch.Tensor,
        qd: torch.Tensor,
        action: torch.Tensor,
        invalid: torch.Tensor,
        *,
        stochastic_init: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not invalid.any():
            return q, qd, action
        invalid_ids = invalid.nonzero(as_tuple=False).squeeze(-1)
        q, qd = self.reset_done(q, qd, invalid_ids, stochastic_init=stochastic_init)
        action = torch.where(invalid.unsqueeze(-1), torch.zeros_like(action), action)
        return q, qd, action

    def make_viewer_state(self, q: torch.Tensor, qd: torch.Tensor) -> newton.State:
        state = self.model.state(requires_grad=False)
        state.joint_q = wp.from_torch(q.detach().contiguous().view(-1), dtype=wp.float32, requires_grad=False)
        state.joint_qd = wp.from_torch(qd.detach().contiguous().view(-1), dtype=wp.float32, requires_grad=False)
        newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)
        wp.synchronize()
        return state


def make_actor(env: NewtonMuJoCoTorchEnv, stochastic: bool = False) -> torch.nn.Module:
    cfg = {
        "actor_mlp": {"units": [128, 64, 32] if env.env_name == "ant" else [64, 64], "activation": "elu"},
        "actor_logstd_init": -1.0,
    }
    if stochastic:
        actor = ActorStochasticMLP(env.num_obs, env.num_actions, cfg, device=str(env.torch_device))
    else:
        actor = ActorDeterministicMLP(env.num_obs, env.num_actions, cfg, device=str(env.torch_device))
        final = actor.actor[-1]
        if isinstance(final, torch.nn.Linear):
            torch.nn.init.zeros_(final.weight)
            torch.nn.init.zeros_(final.bias)
    return actor


def make_critic(env: NewtonMuJoCoTorchEnv) -> torch.nn.Module:
    cfg = {
        "critic_mlp": {"units": [64, 64], "activation": "elu"},
    }
    return CriticMLP(env.num_obs, cfg, device=str(env.torch_device))


def obs_rms_snapshot(obs_rms: RunningMeanStd | None) -> tuple[torch.Tensor, torch.Tensor] | None:
    if obs_rms is None:
        return None
    return obs_rms.mean.clone(), obs_rms.var.clone()


def normalize_obs(obs: torch.Tensor, stats: tuple[torch.Tensor, torch.Tensor] | RunningMeanStd | None) -> torch.Tensor:
    if stats is None:
        return obs
    if isinstance(stats, RunningMeanStd):
        return stats.normalize(obs)
    mean, var = stats
    return (obs - mean) / torch.sqrt(var + 1.0e-5)


@torch.no_grad()
def compute_critic_targets(
    rewards: torch.Tensor,
    done_mask: torch.Tensor,
    next_values: torch.Tensor,
    *,
    gamma: float,
    critic_method: str,
    td_lambda: float,
) -> torch.Tensor:
    if critic_method == "one-step":
        return rewards + gamma * next_values
    if critic_method != "td-lambda":
        raise ValueError(f"unknown critic method: {critic_method}")

    steps_num, num_envs = rewards.shape
    targets = torch.zeros_like(rewards)
    ai = torch.zeros(num_envs, dtype=torch.float32, device=rewards.device)
    bi = torch.zeros(num_envs, dtype=torch.float32, device=rewards.device)
    lam = torch.ones(num_envs, dtype=torch.float32, device=rewards.device)
    for step in reversed(range(steps_num)):
        done = done_mask[step]
        lam = lam * td_lambda * (1.0 - done) + done
        ai = (1.0 - done) * (
            td_lambda * gamma * ai
            + gamma * next_values[step]
            + (1.0 - lam) / (1.0 - td_lambda) * rewards[step]
        )
        bi = gamma * (next_values[step] * done + bi * (1.0 - done)) + rewards[step]
        targets[step] = (1.0 - td_lambda) * ai + lam * bi
    return targets


def run_training(args: argparse.Namespace) -> dict:
    if args.contact_backend is None:
        args.contact_backend = "mujoco" if args.env == "ant" else "none"
    if args.horizon is None:
        args.horizon = 32 if args.env == "ant" else 48
    if args.eval_horizon is None:
        args.eval_horizon = 480 if args.env == "ant" else 180
    if args.episode_length is None:
        args.episode_length = 1000 if args.env == "ant" else 240
    if args.force_scale is None:
        args.force_scale = 200.0 if args.env == "ant" else 1000.0
    if args.grad_clip is None:
        args.grad_clip = 1.0 if args.env == "ant" else 100.0
    if args.reset_noise is None:
        args.reset_noise = 0.0 if args.env == "ant" else 0.05
    if args.termination_penalty is None:
        args.termination_penalty = 10.0 if args.env == "ant" else 0.0
    if args.lr_schedule is None:
        args.lr_schedule = "linear" if args.env == "ant" else "constant"
    if args.adam_beta1 is None:
        args.adam_beta1 = 0.7 if args.env == "ant" else 0.9
    if args.adam_beta2 is None:
        args.adam_beta2 = 0.95 if args.env == "ant" else 0.999
    if args.critic_lr is None:
        args.critic_lr = 2.0e-3 if args.env == "ant" else 1.0e-3
    if args.critic_iterations is None:
        args.critic_iterations = 16 if args.env == "ant" else 8
    if args.critic_method is None:
        args.critic_method = "td-lambda" if args.env == "ant" else "one-step"
    if args.stochastic_actor is None:
        args.stochastic_actor = args.env == "ant"
    if args.stochastic_init is None:
        args.stochastic_init = args.env == "ant"
    if args.use_critic is None:
        args.use_critic = args.env == "ant"
    if args.obs_rms is None:
        args.obs_rms = args.env == "ant"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    wp.init()

    env = NewtonMuJoCoTorchEnv(
        env_name=args.env,
        num_envs=args.num_envs,
        device=args.device,
        dt=args.dt,
        force_scale=args.force_scale,
        contact_backend=args.contact_backend,
        cartpole_reward=CartpoleRewardWeights(
            pole_angle=args.cartpole_pole_angle_penalty,
            pole_velocity=args.cartpole_pole_velocity_penalty,
            cart_position=args.cartpole_cart_position_penalty,
            cart_velocity=args.cartpole_cart_velocity_penalty,
            action=args.cartpole_action_penalty,
        ),
    )
    actor = make_actor(env, stochastic=args.stochastic_actor)
    adam_betas = (args.adam_beta1, args.adam_beta2)
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr, betas=adam_betas)
    critic = None
    target_critic = None
    critic_optimizer = None
    if args.use_critic:
        critic = make_critic(env)
        target_critic = copy.deepcopy(critic)
        for param in target_critic.parameters():
            param.requires_grad_(False)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr, betas=adam_betas)
    obs_rms = RunningMeanStd(shape=(env.num_obs,), device=env.torch_device) if args.obs_rms else None
    history = []
    best_state = None
    best_obs_rms = None
    best_epoch = 0
    best_train_reward = -float("inf")
    best_eval_return = -float("inf")
    q, qd = env.reset(noise=args.reset_noise, stochastic_init=args.stochastic_init)
    prev_action = torch.zeros((env.num_envs, env.num_actions), dtype=torch.float32, device=env.torch_device)
    progress = torch.zeros(env.num_envs, dtype=torch.long, device=env.torch_device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(env.torch_device)
        torch.cuda.synchronize(env.torch_device)

    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        epoch_t0 = time.perf_counter()
        if args.lr_schedule == "linear":
            denom = max(1, args.epochs - 1)
            actor_lr = (args.min_lr - args.lr) * float(epoch / denom) + args.lr
            critic_lr = (args.min_lr - args.critic_lr) * float(epoch / denom) + args.critic_lr
            for param_group in optimizer.param_groups:
                param_group["lr"] = actor_lr
            if critic_optimizer is not None:
                for param_group in critic_optimizer.param_groups:
                    param_group["lr"] = critic_lr

        if args.reset_each_epoch:
            q, qd = env.reset(noise=args.reset_noise, stochastic_init=args.stochastic_init)
            prev_action.zero_()
            progress.zero_()
        else:
            q = q.detach().clone()
            qd = qd.detach().clone()
            prev_action = prev_action.detach().clone()

        optimizer.zero_grad(set_to_none=True)
        rewards = []
        critic_obs = []
        critic_rewards = []
        critic_done_mask = []
        critic_next_values = []
        gamma_vec = torch.ones(env.num_envs, dtype=torch.float32, device=env.torch_device)
        reward_acc = torch.zeros(env.num_envs, dtype=torch.float32, device=env.torch_device)
        actor_loss = torch.zeros((), dtype=torch.float32, device=env.torch_device)
        norm_stats = obs_rms_snapshot(obs_rms)
        invalid_count = 0
        fall_count = 0
        timeout_count = 0

        for step_idx in range(args.horizon):
            obs_raw = env.observe(q, qd, prev_action)
            if obs_rms is not None:
                with torch.no_grad():
                    obs_rms.update(obs_raw.detach())
            obs = normalize_obs(obs_raw, norm_stats)
            if args.use_critic:
                critic_obs.append(obs.detach())
            raw_action = actor(obs, deterministic=not args.stochastic_actor)
            action = torch.tanh(raw_action)
            q_next, qd_next = env.step(q, qd, env.action_to_joint_f(action))
            invalid = env.invalid_state(q_next, qd_next)
            fell = torch.logical_and(env.fallen_state(q_next), ~invalid)
            q_next, qd_next, action = env.sanitize_state(
                q_next, qd_next, action, invalid, stochastic_init=args.stochastic_init
            )
            next_obs_raw = env.observe(q_next, qd_next, action)
            rew = env.reward(q_next, qd_next, action, obs=next_obs_raw)
            rew = torch.where(invalid, torch.full_like(rew, ANT_INVALID_PENALTY), rew)
            if args.termination_penalty > 0.0:
                rew = torch.where(fell, rew - args.termination_penalty, rew)
            scaled_rew = rew * args.rew_scale
            rewards.append(rew.detach().mean())
            if args.use_critic:
                critic_rewards.append(scaled_rew.detach())

            progress = progress + 1
            timeout = progress >= args.episode_length
            done = torch.logical_or(torch.logical_or(timeout, fell), invalid)
            invalid_count += int(invalid.detach().sum().cpu())
            fall_count += int(fell.detach().sum().cpu())
            timeout_count += int(timeout.detach().sum().cpu())
            next_obs = normalize_obs(next_obs_raw, norm_stats)
            if args.use_critic:
                assert target_critic is not None
                next_value = target_critic(next_obs).squeeze(-1)
                critic_next_values.append(next_value.detach())
                if step_idx < args.horizon - 1:
                    critic_done_mask.append(done.detach().to(torch.float32))
                else:
                    critic_done_mask.append(torch.ones_like(scaled_rew))
            else:
                next_value = torch.zeros(env.num_envs, dtype=torch.float32, device=env.torch_device)

            reward_acc = reward_acc + gamma_vec * scaled_rew
            if step_idx < args.horizon - 1:
                loss_mask = done
            else:
                loss_mask = torch.ones_like(done)
            if args.use_critic:
                segment_return = reward_acc + args.gamma * gamma_vec * next_value
            else:
                segment_return = reward_acc
            actor_loss = actor_loss - segment_return[loss_mask].sum()

            gamma_vec = gamma_vec * args.gamma
            if done.any():
                done_ids = done.nonzero(as_tuple=False).squeeze(-1)
                q_next, qd_next = env.reset_done(q_next, qd_next, done_ids, stochastic_init=args.stochastic_init)
                action = torch.where(
                    done.unsqueeze(-1),
                    torch.zeros_like(action),
                    action,
                )
                progress = torch.where(done, torch.zeros_like(progress), progress)
                gamma_vec = torch.where(done, torch.ones_like(gamma_vec), gamma_vec)
                reward_acc = torch.where(done, torch.zeros_like(reward_acc), reward_acc)
            q, qd, prev_action = q_next, qd_next, action

        loss = actor_loss / (args.horizon * args.num_envs)
        loss.backward()
        for param in actor.parameters():
            if param.grad is not None:
                param.grad.nan_to_num_(0.0, 0.0, 0.0)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), args.grad_clip).detach().cpu())
        mean_reward = float(torch.stack(rewards).mean().detach().cpu())
        final_reward = float(rewards[-1].detach().cpu())
        optimizer.step()

        value_loss = None
        if args.use_critic:
            assert critic is not None
            assert target_critic is not None
            assert critic_optimizer is not None
            with torch.no_grad():
                obs_flat = torch.cat([obs.detach() for obs in critic_obs], dim=0)
                target_values = compute_critic_targets(
                    torch.stack(critic_rewards),
                    torch.stack(critic_done_mask),
                    torch.stack(critic_next_values),
                    gamma=args.gamma,
                    critic_method=args.critic_method,
                    td_lambda=args.td_lambda,
                )
                target_flat = target_values.reshape(-1)

            sample_count = obs_flat.shape[0]
            batch_size = min(args.critic_batch_size, sample_count)
            last_loss = None
            for _ in range(args.critic_iterations):
                order = torch.randperm(sample_count, device=env.torch_device)
                for start in range(0, sample_count, batch_size):
                    idx = order[start : start + batch_size]
                    pred = critic(obs_flat[idx]).squeeze(-1)
                    critic_loss = (pred - target_flat[idx]).square().mean()
                    critic_optimizer.zero_grad(set_to_none=True)
                    critic_loss.backward()
                    for param in critic.parameters():
                        if param.grad is not None:
                            param.grad.nan_to_num_(0.0, 0.0, 0.0)
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), args.grad_clip)
                    critic_optimizer.step()
                    last_loss = critic_loss

            with torch.no_grad():
                alpha = args.target_critic_alpha
                for param, target_param in zip(critic.parameters(), target_critic.parameters()):
                    target_param.mul_(alpha)
                    target_param.add_((1.0 - alpha) * param)
            value_loss = float(last_loss.detach().cpu()) if last_loss is not None else None

        if torch.cuda.is_available():
            torch.cuda.synchronize(env.torch_device)

        selection_horizon = args.selection_horizon or args.eval_horizon
        selection_rollout = evaluate_policy(
            env, actor, selection_horizon, obs_rms=obs_rms, termination_penalty=args.termination_penalty
        )
        if selection_rollout["return"] > best_eval_return:
            best_eval_return = selection_rollout["return"]
            best_train_reward = mean_reward
            best_epoch = epoch + 1
            best_state = {name: value.detach().clone() for name, value in actor.state_dict().items()}
            if obs_rms is not None:
                best_obs_rms = {
                    "mean": obs_rms.mean.detach().clone(),
                    "var": obs_rms.var.detach().clone(),
                    "count": obs_rms.count,
                }

        epoch_s = time.perf_counter() - epoch_t0
        history.append(
            {
                "epoch": epoch + 1,
                "mean_reward": mean_reward,
                "final_step_reward": final_reward,
                "loss": float(loss.detach().cpu()),
                "grad_norm": grad_norm,
                "value_loss": value_loss,
                "selection_return": selection_rollout["return"],
                "selection_mean_reward": selection_rollout["mean_reward"],
                "invalid_resets": invalid_count,
                "fall_resets": fall_count,
                "timeout_resets": timeout_count,
                "epoch_seconds": epoch_s,
                "fps": args.num_envs * args.horizon / epoch_s,
            }
        )
        print(
            f"{args.env} epoch {epoch + 1:03d}: reward={mean_reward: .4f} "
            f"loss={float(loss.detach().cpu()): .4f} fps={history[-1]['fps']: .1f}"
        )

    total_s = time.perf_counter() - t0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        actor.load_state_dict(best_state)
    if obs_rms is not None and best_obs_rms is not None:
        obs_rms.mean = best_obs_rms["mean"]
        obs_rms.var = best_obs_rms["var"]
        obs_rms.count = best_obs_rms["count"]
    torch.save(actor.state_dict(), out_dir / f"{args.env}_actor.pt")
    if critic is not None:
        torch.save(critic.state_dict(), out_dir / f"{args.env}_critic.pt")
    if obs_rms is not None:
        torch.save(
            {"mean": obs_rms.mean, "var": obs_rms.var, "count": obs_rms.count},
            out_dir / f"{args.env}_obs_rms.pt",
        )

    rollout = evaluate_policy(env, actor, args.eval_horizon, obs_rms=obs_rms, termination_penalty=args.termination_penalty)
    video_path = None
    poster_path = None
    if args.render_video:
        render_env = env
        if args.video_num_envs != args.num_envs:
            render_env = NewtonMuJoCoTorchEnv(
                env_name=args.env,
                num_envs=args.video_num_envs,
                device=args.device,
                dt=args.dt,
                force_scale=args.force_scale,
                contact_backend=args.contact_backend,
                cartpole_reward=env.cartpole_reward,
            )
        video_path, poster_path = render_rollout(
            render_env,
            actor,
            out_dir,
            args.eval_horizon,
            args.env,
            obs_rms=obs_rms,
        )

    result = {
        "env": args.env,
        "title": "SHAC with MuJoCo Warp",
        "timestamp_pacific": pacific_now_iso(),
        "mujoco_warp_pr": "google-deepmind/mujoco_warp#1423",
        "mujoco_warp_commit": "255d522299c39a5c6905e439f26861072a10fdf0",
        "num_envs": args.num_envs,
        "contact_backend": args.contact_backend,
        "horizon": args.horizon,
        "epochs": args.epochs,
        "dt": args.dt,
        "episode_length": args.episode_length,
        "stochastic_init": args.stochastic_init,
        "stochastic_actor": args.stochastic_actor,
        "use_critic": args.use_critic,
        "obs_rms": args.obs_rms,
        "critic_method": args.critic_method,
        "td_lambda": args.td_lambda,
        "rew_scale": args.rew_scale,
        "reset_each_epoch": args.reset_each_epoch,
        "termination_penalty": args.termination_penalty,
        "ant_max_healthy_height": ANT_MAX_HEALTHY_HEIGHT if args.env == "ant" else None,
        "ant_height_reward_cap": ANT_HEIGHT_REWARD_CAP if args.env == "ant" else None,
        "ant_invalid_penalty": ANT_INVALID_PENALTY if args.env == "ant" else None,
        "lr_schedule": args.lr_schedule,
        "cartpole_reward": env.cartpole_reward.__dict__,
        "total_seconds": total_s,
        "mean_epoch_seconds": float(np.mean([h["epoch_seconds"] for h in history])),
        "mean_fps": float(np.mean([h["fps"] for h in history])),
        "best_epoch": best_epoch,
        "best_train_reward": best_train_reward,
        "best_eval_return": best_eval_return,
        "max_cuda_memory_allocated_mb": (
            float(torch.cuda.max_memory_allocated(env.torch_device) / (1024**2)) if torch.cuda.is_available() else None
        ),
        "max_cuda_memory_reserved_mb": (
            float(torch.cuda.max_memory_reserved(env.torch_device) / (1024**2)) if torch.cuda.is_available() else None
        ),
        "history": history,
        "eval": rollout,
        "video": video_path.name if video_path else None,
        "poster": poster_path.name if poster_path else None,
        "gpu": query_gpu(),
    }
    with (out_dir / f"{args.env}_results.json").open("w") as f:
        json.dump(result, f, indent=2)
    return result


@torch.no_grad()
def evaluate_policy(
    env: NewtonMuJoCoTorchEnv,
    actor: torch.nn.Module,
    horizon: int,
    *,
    obs_rms: RunningMeanStd | None = None,
    termination_penalty: float = 0.0,
) -> dict:
    q, qd = env.reset(noise=0.0)
    prev_action = torch.zeros((env.num_envs, env.num_actions), dtype=torch.float32, device=env.torch_device)
    progress = torch.zeros(env.num_envs, dtype=torch.long, device=env.torch_device)
    rewards = []
    final_obs = None
    final_obs_normalized = None
    reset_count = 0
    invalid_count = 0
    fall_count = 0
    timeout_count = 0
    for _ in range(horizon):
        obs = normalize_obs(env.observe(q, qd, prev_action), obs_rms)
        action = torch.tanh(actor(obs, deterministic=True))
        q, qd = env.step(q, qd, env.action_to_joint_f(action))
        invalid = env.invalid_state(q, qd)
        fell = torch.logical_and(env.fallen_state(q), ~invalid)
        q, qd, action = env.sanitize_state(q, qd, action, invalid, stochastic_init=False)
        final_obs = env.observe(q, qd, action)
        rew = env.reward(q, qd, action, obs=final_obs)
        rew = torch.where(invalid, torch.full_like(rew, ANT_INVALID_PENALTY), rew)
        if termination_penalty > 0.0:
            rew = torch.where(fell, rew - termination_penalty, rew)
        rewards.append(rew.mean())
        progress = progress + 1
        timeout = progress >= horizon
        done = torch.logical_or(torch.logical_or(timeout, fell), invalid)
        final_obs_normalized = normalize_obs(final_obs, obs_rms)
        invalid_count += int(invalid.sum().cpu())
        fall_count += int(fell.sum().cpu())
        timeout_count += int(timeout.sum().cpu())
        if done.any():
            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            reset_count += int(done_ids.numel())
            q, qd = env.reset_done(q, qd, done_ids, stochastic_init=False)
            action = torch.where(done.unsqueeze(-1), torch.zeros_like(action), action)
            progress = torch.where(done, torch.zeros_like(progress), progress)
        prev_action = action
    return {
        "mean_reward": float(torch.stack(rewards).mean().cpu()),
        "return": float(torch.stack(rewards).sum().cpu()),
        "reset_count": reset_count,
        "invalid_count": invalid_count,
        "fall_count": fall_count,
        "timeout_count": timeout_count,
        "final_obs_mean": [float(x) for x in final_obs.mean(dim=0).cpu().tolist()],
        "final_obs_normalized_mean": [float(x) for x in final_obs_normalized.mean(dim=0).cpu().tolist()],
    }


def render_rollout(
    env: NewtonMuJoCoTorchEnv,
    actor: torch.nn.Module,
    out_dir: Path,
    horizon: int,
    env_name: str,
    *,
    obs_rms: RunningMeanStd | None = None,
) -> tuple[Path, Path]:
    import imageio.v2 as imageio

    viewer = newton.viewer.ViewerGL(width=960, height=544, headless=True)
    viewer.show_static = True
    viewer.show_collision = True
    viewer.set_model(env.model)
    if env_name == "cartpole":
        viewer.set_camera(pos=wp.vec3(0.0, 2.0, 7.0), pitch=0.0, yaw=-90.0)
        viewer.camera.look_at((0.0, 0.0, 0.0))
        if hasattr(viewer, "camera") and hasattr(viewer.camera, "fov"):
            viewer.camera.fov = 45.0
    else:
        viewer.set_camera(pos=wp.vec3(-3.0, 2.8, 5.0), pitch=-20.0, yaw=-140.0)
        viewer.camera.look_at((0.0, 0.7, 0.0))
        if hasattr(viewer, "camera") and hasattr(viewer.camera, "fov"):
            viewer.camera.fov = 52.0

    q, qd = env.reset(noise=0.0)
    prev_action = torch.zeros((env.num_envs, env.num_actions), dtype=torch.float32, device=env.torch_device)
    progress = torch.zeros(env.num_envs, dtype=torch.long, device=env.torch_device)
    video_path = out_dir / f"{env_name}_rollout.mp4"
    poster_path = out_dir / f"{env_name}_poster.png"
    frames = []
    with imageio.get_writer(video_path, fps=max(1, int(round(1.0 / env.dt))), codec="libx264", quality=8) as writer:
        with torch.no_grad():
            for frame_idx in range(horizon):
                if env_name == "ant":
                    target_x = float(q[0, 0].detach().cpu())
                    target_y = float(q[0, 1].detach().cpu())
                    target_z = float(q[0, 2].detach().cpu())
                    viewer.set_camera(pos=wp.vec3(target_x - 3.0, target_y + 2.2, target_z + 4.5), pitch=-20.0, yaw=-140.0)
                    viewer.camera.look_at((target_x, target_y, target_z))
                state = env.make_viewer_state(q, qd)
                viewer.begin_frame(frame_idx * env.dt)
                viewer.log_state(state)
                viewer.end_frame()
                frame = viewer.get_frame().numpy()
                frames.append(frame)
                writer.append_data(frame)
                obs = normalize_obs(env.observe(q, qd, prev_action), obs_rms)
                action = torch.tanh(actor(obs, deterministic=True))
                q, qd = env.step(q, qd, env.action_to_joint_f(action))
                invalid = env.invalid_state(q, qd)
                fell = torch.logical_and(env.fallen_state(q), ~invalid)
                q, qd, action = env.sanitize_state(q, qd, action, invalid, stochastic_init=False)
                q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
                qd = torch.nan_to_num(qd, nan=0.0, posinf=0.0, neginf=0.0)
                progress = progress + 1
                timeout = progress >= horizon
                done = torch.logical_or(torch.logical_or(timeout, fell), invalid)
                if done.any():
                    done_ids = done.nonzero(as_tuple=False).squeeze(-1)
                    q, qd = env.reset_done(q, qd, done_ids, stochastic_init=False)
                    action = torch.where(done.unsqueeze(-1), torch.zeros_like(action), action)
                    progress = torch.where(done, torch.zeros_like(progress), progress)
                prev_action = action
    viewer.close()
    imageio.imwrite(poster_path, frames[len(frames) // 2])
    return video_path, poster_path


def query_gpu() -> str | None:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return None
    return proc.stdout.strip() or None


def pacific_now_iso() -> str:
    try:
        tz = ZoneInfo("America/Los_Angeles")
    except Exception:
        tz = timezone(timedelta(hours=-7), name="PDT")
    return datetime.now(tz).isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["cartpole", "ant"], default="cartpole")
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "assets"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--eval-horizon", type=int, default=None)
    parser.add_argument("--episode-length", type=int, default=None)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--min-lr", type=float, default=1.0e-5)
    parser.add_argument("--lr-schedule", choices=["constant", "linear"], default=None)
    parser.add_argument("--adam-beta1", type=float, default=None)
    parser.add_argument("--adam-beta2", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--rew-scale", type=float, default=1.0)
    parser.add_argument("--force-scale", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--reset-noise", type=float, default=None)
    parser.add_argument("--termination-penalty", type=float, default=None)
    parser.add_argument("--reset-each-epoch", action="store_true")
    parser.add_argument("--stochastic-init", dest="stochastic_init", action="store_true", default=None)
    parser.add_argument("--deterministic-init", dest="stochastic_init", action="store_false")
    parser.add_argument("--stochastic-actor", dest="stochastic_actor", action="store_true", default=None)
    parser.add_argument("--deterministic-actor", dest="stochastic_actor", action="store_false")
    parser.add_argument("--use-critic", dest="use_critic", action="store_true", default=None)
    parser.add_argument("--no-critic", dest="use_critic", action="store_false")
    parser.add_argument("--obs-rms", dest="obs_rms", action="store_true", default=None)
    parser.add_argument("--no-obs-rms", dest="obs_rms", action="store_false")
    parser.add_argument("--critic-lr", type=float, default=None)
    parser.add_argument("--critic-iterations", type=int, default=None)
    parser.add_argument("--critic-batch-size", type=int, default=1024)
    parser.add_argument("--critic-method", choices=["one-step", "td-lambda"], default=None)
    parser.add_argument("--td-lambda", type=float, default=0.95)
    parser.add_argument("--target-critic-alpha", type=float, default=0.2)
    parser.add_argument("--cartpole-pole-angle-penalty", type=float, default=1.0)
    parser.add_argument("--cartpole-pole-velocity-penalty", type=float, default=0.1)
    parser.add_argument("--cartpole-cart-position-penalty", type=float, default=0.05)
    parser.add_argument("--cartpole-cart-velocity-penalty", type=float, default=0.1)
    parser.add_argument("--cartpole-action-penalty", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--render-video", action="store_true")
    parser.add_argument("--video-num-envs", type=int, default=1)
    parser.add_argument("--selection-horizon", type=int, default=None)
    parser.add_argument("--contact-backend", choices=["mujoco", "newton", "none"], default=None)
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("MJWARP_ENABLE_AD", "1")
    run_training(parse_args())
