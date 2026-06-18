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
from typing import Any, Callable
from zoneinfo import ZoneInfo

import numpy as np
import torch
import warp as wp
import mujoco
import mujoco_warp

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

from follow_camera import SmoothedFollowCamera


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


@dataclass
class AntRewardWeights:
    progress: float = 1.0
    heading: float = 1.0
    up: float = 0.1
    height: float = 1.0
    action: float = 0.0
    alive: float = 0.5
    actions_cost: float = 0.005
    energy_cost: float = 0.05
    dof_limit_cost: float = 1.0
    dof_vel_scale: float = 0.2
    up_margin: float = 0.0
    height_margin: float = 0.0


@dataclass
class HopperRewardWeights:
    progress: float = 1.0
    height: float = 1.0
    angle: float = 1.0
    action: float = -0.1
    alive: float = 1.0


@dataclass
class CheetahRewardWeights:
    action: float = -0.1


@dataclass
class AcrobotRewardWeights:
    target: float = 8.0
    velocity: float = 0.05
    action: float = 0.002


@dataclass
class ContactTargetRewardWeights:
    target: float = 8.0
    velocity: float = 0.05
    height: float = 1.0
    action: float = 0.002


ANT_DIFFRL_START_HEIGHT = 0.75
ANT_ISAACLAB_START_HEIGHT = 0.5
ANT_START_HEIGHT = ANT_DIFFRL_START_HEIGHT
ANT_START_JOINT_Q = (0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0)
ANT_ISAACLAB_START_JOINT_Q = (
    0.0,
    0.25 * math.pi,
    0.0,
    -0.25 * math.pi,
    0.0,
    -0.25 * math.pi,
    0.0,
    0.25 * math.pi,
)
ANT_START_ROT = (math.sin(-0.25 * math.pi), 0.0, 0.0, math.cos(-0.25 * math.pi))
ANT_TERMINATION_HEIGHT = 0.27
ANT_ISAACLAB_TERMINATION_HEIGHT = 0.31
ANT_MAX_HEALTHY_HEIGHT = 1.5
ANT_HEIGHT_REWARD_CAP = 0.6
ANT_INVALID_PENALTY = -50.0
ANT_JOINT_VEL_OBS_SCALING = 0.1
ANT_ACTION_PENALTY = 0.0
ANT_DEFAULT_TERMINATION_PENALTY = 20.0
ANT_DEFAULT_SELECTION_FALL_PENALTY = 500000.0
ANT_DEFAULT_SELECTION_INVALID_PENALTY = 500000.0
HOPPER_TERMINATION_HEIGHT = -0.45
HOPPER_TERMINATION_ANGLE = math.pi / 6.0
HOPPER_TERMINATION_HEIGHT_TOLERANCE = 0.15
HOPPER_START_HEIGHT = 0.0
HOPPER_START_JOINT_Q = (-0.05, -0.05, 0.0)
CHEETAH_START_HEIGHT = -0.2
DEFAULT_GRAD_CHECK_EPS = (1.0e-1, 3.0e-2, 1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4)


def is_contact_target_env(env_name: str) -> bool:
    return env_name in {"contact_sphere", "contact_capsule"}


def is_planar_locomotion_env(env_name: str) -> bool:
    return env_name in {"hopper", "cheetah"}


def is_locomotion_env(env_name: str) -> bool:
    return env_name == "ant" or is_planar_locomotion_env(env_name)


def ant_defaults_for_asset(ant_asset: str) -> dict[str, Any]:
    if ant_asset == "diffrl":
        return {
            "sim_substeps": 16,
            "force_scale": 200.0,
            "density_override": None,
            "contact_mu": 0.75,
            "joint_damping": 1.0,
            "armature": 0.05,
            "start_height": ANT_DIFFRL_START_HEIGHT,
            "start_joint_q": list(ANT_START_JOINT_Q),
            "termination_height": ANT_TERMINATION_HEIGHT,
            "observation_style": "diffrl",
            "reward_style": "diffrl",
            "dof_limit_mode": "abs",
            "dof_limit_cost": 1.0,
            "action_order": "joint",
            "heading_weight": 1.0,
        }
    return {
        "sim_substeps": 2,
        "force_scale": 7.5,
        "density_override": None,
        "contact_mu": 1.0,
        "joint_damping": 0.1,
        "armature": 0.05,
        "start_height": ANT_ISAACLAB_START_HEIGHT,
        "start_joint_q": list(ANT_ISAACLAB_START_JOINT_Q),
        "termination_height": ANT_ISAACLAB_TERMINATION_HEIGHT,
        "observation_style": "isaac",
        "reward_style": "isaaclab_potential",
        "dof_limit_mode": "abs",
        "dof_limit_cost": 0.1,
        "action_order": "actuator",
        "heading_weight": 0.5,
    }


def resolve_ant_defaults(args: argparse.Namespace) -> None:
    if getattr(args, "env", None) != "ant":
        return
    defaults = ant_defaults_for_asset(args.ant_asset)
    if getattr(args, "sim_substeps", None) is None:
        args.sim_substeps = defaults["sim_substeps"]
    if getattr(args, "force_scale", None) is None:
        args.force_scale = defaults["force_scale"]
    if getattr(args, "ant_density_override", None) is None:
        args.ant_density_override = defaults["density_override"]
    if getattr(args, "ant_contact_mu", None) is None:
        args.ant_contact_mu = defaults["contact_mu"]
    if getattr(args, "ant_joint_damping", None) is None:
        args.ant_joint_damping = defaults["joint_damping"]
    if getattr(args, "ant_armature", None) is None:
        args.ant_armature = defaults["armature"]
    if getattr(args, "ant_start_height", None) is None:
        args.ant_start_height = defaults["start_height"]
    if getattr(args, "ant_start_joint_q", None) is None:
        args.ant_start_joint_q = list(defaults["start_joint_q"])
    if getattr(args, "ant_termination_height", None) is None:
        args.ant_termination_height = defaults["termination_height"]
    if getattr(args, "ant_observation_style", None) is None:
        args.ant_observation_style = defaults["observation_style"]
    if getattr(args, "ant_reward_style", None) is None:
        args.ant_reward_style = defaults["reward_style"]
    if getattr(args, "ant_dof_limit_mode", None) is None:
        args.ant_dof_limit_mode = defaults["dof_limit_mode"]
    if getattr(args, "ant_dof_limit_cost", None) is None:
        args.ant_dof_limit_cost = defaults["dof_limit_cost"]
    if getattr(args, "ant_action_order", None) is None:
        # DiffRL applies Ant actions directly to joint_act[:, 6:] in joint
        # coordinate order.  The MJCF actuator block has a different ordering,
        # so using actuator order sends learned torques to the wrong hips/ankles.
        args.ant_action_order = defaults["action_order"]
    if getattr(args, "ant_heading_weight", None) is None:
        args.ant_heading_weight = defaults["heading_weight"]


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


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, int | str | bool) or value is None:
        return value
    return value


def write_json(path: Path, payload: dict) -> None:
    with path.open("w") as f:
        json.dump(json_safe(payload), f, indent=2, allow_nan=False)


def finite_float(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value) if math.isfinite(float(value)) else None


def sanitize_and_clip_grad_norm(parameters, max_norm: float, value_clip: float) -> float:
    grads = []
    for param in parameters:
        if param.grad is None:
            continue
        param.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        if value_clip > 0.0:
            param.grad.clamp_(min=-value_clip, max=value_clip)
        grads.append(param.grad)
    if not grads:
        return 0.0

    total_sq = torch.zeros((), dtype=torch.float64, device=grads[0].device)
    for grad in grads:
        grad64 = grad.detach().to(torch.float64)
        total_sq = total_sq + grad64.square().sum()
    total_norm = torch.sqrt(total_sq)
    if not torch.isfinite(total_norm):
        for grad in grads:
            grad.zero_()
        return float("inf")

    total_norm_f = float(total_norm.detach().cpu())
    if max_norm > 0.0 and total_norm_f > max_norm:
        scale = max_norm / (total_norm_f + 1.0e-12)
        for grad in grads:
            grad.mul_(scale)
    return total_norm_f


def trainable_parameters(module: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [param for param in module.parameters() if param.requires_grad]


def flatten_parameters(parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    return torch.cat([param.detach().reshape(-1) for param in parameters])


def flatten_gradients(parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    chunks = []
    for param in parameters:
        if param.grad is None:
            chunks.append(torch.zeros_like(param.detach()).reshape(-1))
        else:
            chunks.append(torch.nan_to_num(param.grad.detach(), nan=0.0, posinf=0.0, neginf=0.0).reshape(-1))
    return torch.cat(chunks)


def assign_flat_parameters(parameters: list[torch.nn.Parameter], values: torch.Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for param in parameters:
            count = param.numel()
            param.copy_(values[offset : offset + count].view_as(param))
            offset += count


def finalize_terminal_reward(
    reward: torch.Tensor,
    *,
    invalid: torch.Tensor,
    fell: torch.Tensor,
    termination_penalty: float,
) -> torch.Tensor:
    reward = torch.where(invalid, torch.full_like(reward, ANT_INVALID_PENALTY), reward)
    if termination_penalty > 0.0:
        reward = torch.where(fell, torch.full_like(reward, -termination_penalty), reward)
    return reward


def rollout_selection_score(
    rollout: dict,
    *,
    num_envs: int,
    fall_penalty: float,
    invalid_penalty: float,
    displacement_weight: float = 0.0,
    height_weight: float = 0.0,
    up_weight: float = 0.0,
    heading_weight: float = 0.0,
    min_height: float | None = None,
    min_up: float | None = None,
    min_heading: float | None = None,
    max_abs_joint: float | None = None,
    posture_penalty: float = 0.0,
) -> float:
    del num_envs
    fall_events = float(rollout.get("terminal_count", rollout.get("fall_count", 0)))
    invalid_events = float(rollout.get("invalid_count", 0))
    displacement = float(rollout.get("mean_forward_displacement") or 0.0)
    height = float(rollout.get("mean_height") or 0.0)
    up = float(rollout.get("mean_up") or 0.0)
    heading = float(rollout.get("mean_heading") or 0.0)
    shortfall = rollout_constraint_shortfalls(
        rollout,
        min_height=min_height,
        min_up=min_up,
        min_heading=min_heading,
        max_abs_joint=max_abs_joint,
    )["posture_shortfall"]
    return (
        float(rollout["return"])
        + displacement_weight * displacement
        + height_weight * height
        + up_weight * up
        + heading_weight * heading
        - fall_penalty * fall_events
        - invalid_penalty * invalid_events
        - posture_penalty * shortfall
    )


def rollout_constraint_shortfalls(
    rollout: dict,
    *,
    min_height: float | None = None,
    min_up: float | None = None,
    min_heading: float | None = None,
    max_abs_joint: float | None = None,
) -> dict[str, float | None]:
    def metric_value(mean_key: str, min_key: str) -> float:
        value = rollout.get(min_key)
        if value is None:
            value = rollout.get(mean_key)
        return float(value or 0.0)

    height_value = metric_value("mean_height", "min_height")
    up_value = metric_value("mean_up", "min_up")
    heading_value = metric_value("mean_heading", "min_heading")
    joint_value = metric_value("mean_abs_joint_pos_scaled", "max_abs_joint_pos_scaled")
    height_shortfall = None if min_height is None else max(0.0, float(min_height) - height_value)
    up_shortfall = None if min_up is None else max(0.0, float(min_up) - up_value)
    heading_shortfall = None if min_heading is None else max(0.0, float(min_heading) - heading_value)
    joint_shortfall = None if max_abs_joint is None else max(0.0, joint_value - float(max_abs_joint))
    posture_shortfall = sum(
        v for v in (height_shortfall, up_shortfall, heading_shortfall, joint_shortfall) if v is not None
    )
    return {
        "height_threshold_value": height_value,
        "up_threshold_value": up_value,
        "heading_threshold_value": heading_value,
        "joint_threshold_value": joint_value,
        "height_shortfall": height_shortfall,
        "up_shortfall": up_shortfall,
        "heading_shortfall": heading_shortfall,
        "joint_shortfall": joint_shortfall,
        "posture_shortfall": posture_shortfall,
    }


def summarize_rollout_repeats(rollouts: list[dict], scores: list[float] | None = None) -> dict:
    if not rollouts:
        return {"count": 0, "samples": []}

    def numeric_values(key: str) -> list[float]:
        return [float(item[key]) for item in rollouts if item.get(key) is not None]

    def extrema(key: str) -> dict[str, float | None]:
        values = numeric_values(key)
        if not values:
            return {"mean": None, "min": None, "max": None}
        return {
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    def count_total(key: str) -> int:
        return int(sum(int(item.get(key) or 0) for item in rollouts))

    summary = {
        "count": len(rollouts),
        "fall_count_total": count_total("fall_count"),
        "fall_count_max": max(int(item.get("fall_count") or 0) for item in rollouts),
        "invalid_count_total": count_total("invalid_count"),
        "invalid_count_max": max(int(item.get("invalid_count") or 0) for item in rollouts),
        "terminal_count_total": count_total("terminal_count"),
        "terminal_count_max": max(int(item.get("terminal_count") or 0) for item in rollouts),
        "forward_displacement": extrema("mean_forward_displacement"),
        "return": extrema("return"),
        "min_height": extrema("min_height"),
        "min_up": extrema("min_up"),
        "min_heading": extrema("min_heading"),
        "max_abs_joint_pos_scaled": extrema("max_abs_joint_pos_scaled"),
        "mean_joint_limit_fraction": extrema("mean_joint_limit_fraction"),
        "max_joint_limit_fraction": extrema("max_joint_limit_fraction"),
        "max_abs_action": extrema("max_abs_action"),
        "samples": rollouts,
    }
    if scores is not None:
        summary["scores"] = scores
        summary["score"] = {
            "mean": float(np.mean(scores)) if scores else None,
            "min": float(np.min(scores)) if scores else None,
            "max": float(np.max(scores)) if scores else None,
        }
    return summary


def summarize_rollout_chunks(chunks: list[dict], *, total_envs: int, chunk_size: int) -> dict:
    if not chunks:
        return {"chunk_count": 0, "total_num_envs": 0, "chunk_size": chunk_size, "chunks": []}

    weights = np.array([float(item.get("num_envs") or 0) for item in chunks], dtype=np.float64)
    if float(weights.sum()) <= 0.0:
        weights = np.ones(len(chunks), dtype=np.float64)

    def weighted_mean(key: str) -> float | None:
        values = []
        value_weights = []
        for item, weight in zip(chunks, weights):
            value = item.get(key)
            if value is None:
                continue
            values.append(float(value))
            value_weights.append(float(weight))
        if not values:
            return None
        return float(np.average(np.array(values, dtype=np.float64), weights=np.array(value_weights, dtype=np.float64)))

    def min_value(key: str) -> float | None:
        values = [float(item[key]) for item in chunks if item.get(key) is not None]
        return float(np.min(values)) if values else None

    def max_value(key: str) -> float | None:
        values = [float(item[key]) for item in chunks if item.get(key) is not None]
        return float(np.max(values)) if values else None

    def count_total(key: str) -> int:
        return int(sum(int(item.get(key) or 0) for item in chunks))

    terminal_env_ids: list[int] = []
    terminal_steps: list[int] = []
    env_offset = 0
    for chunk in chunks:
        ids = chunk.get("terminal_env_ids") or []
        steps = chunk.get("terminal_steps") or []
        terminal_env_ids.extend([int(item) + env_offset for item in ids])
        terminal_steps.extend([int(item) for item in steps])
        env_offset += int(chunk.get("num_envs") or 0)

    terminal_count = count_total("terminal_count")
    return {
        "chunk_count": len(chunks),
        "total_num_envs": total_envs,
        "chunk_size": chunk_size,
        "num_envs": total_envs,
        "mean_reward": weighted_mean("mean_reward"),
        "return": weighted_mean("return"),
        "alive_fraction": 1.0 - terminal_count / max(1, total_envs),
        "terminal_count": terminal_count,
        "fall_count": count_total("fall_count"),
        "invalid_count": count_total("invalid_count"),
        "reset_count": count_total("reset_count"),
        "timeout_count": count_total("timeout_count"),
        "first_terminal_step": min_value("first_terminal_step"),
        "mean_terminal_step": weighted_mean("mean_terminal_step"),
        "terminal_env_ids": terminal_env_ids[:32],
        "terminal_steps": terminal_steps[:32],
        "mean_forward_displacement": weighted_mean("mean_forward_displacement"),
        "mean_completed_return": weighted_mean("mean_completed_return"),
        "unfinished_mean_return": weighted_mean("unfinished_mean_return"),
        "unfinished_mean_length": weighted_mean("unfinished_mean_length"),
        "mean_height": weighted_mean("mean_height"),
        "min_height": min_value("min_height"),
        "mean_up": weighted_mean("mean_up"),
        "min_up": min_value("min_up"),
        "mean_heading": weighted_mean("mean_heading"),
        "min_heading": min_value("min_heading"),
        "mean_abs_joint_pos_scaled": weighted_mean("mean_abs_joint_pos_scaled"),
        "max_abs_joint_pos_scaled": max_value("max_abs_joint_pos_scaled"),
        "mean_joint_limit_fraction": weighted_mean("mean_joint_limit_fraction"),
        "max_joint_limit_fraction": max_value("max_joint_limit_fraction"),
        "mean_abs_action": weighted_mean("mean_abs_action"),
        "max_abs_action": max_value("max_abs_action"),
        "mean_abs_joint_velocity": weighted_mean("mean_abs_joint_velocity"),
        "horizon": chunks[0].get("horizon"),
        "chunks": chunks,
    }


def load_obs_rms(path: Path | None, device: torch.device) -> tuple[torch.Tensor, torch.Tensor] | None:
    if path is None:
        return None
    data = torch.load(path, map_location=device)
    return data["mean"].to(device), data["var"].to(device)


class NewtonMuJoCoStep(torch.autograd.Function):
    @staticmethod
    def _snapshot_solver_state(env) -> tuple[int | None, dict[str, object]]:
        solver = getattr(env, "solver", None)
        data = getattr(solver, "mjw_data", None)
        step = getattr(solver, "_step", None)
        if data is None:
            return step, {}
        fields = {}
        for name in ("qfrc_applied", "xfrc_applied", "ctrl", "act", "qacc_warmstart"):
            if hasattr(data, name):
                fields[name] = getattr(data, name)
        return step, fields

    @staticmethod
    def _restore_solver_state(env, step: int | None, fields: dict[str, object]) -> None:
        solver = getattr(env, "solver", None)
        data = getattr(solver, "mjw_data", None)
        if solver is not None and step is not None:
            solver._step = step
        if data is None:
            return
        for name, value in fields.items():
            setattr(data, name, value)

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

        solver_step, solver_fields = NewtonMuJoCoStep._snapshot_solver_state(env)
        try:
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

            q_grad = torch_grad("q", q)
            qd_grad = torch_grad("qd", qd)
            joint_f_grad = torch_grad("joint_f", joint_f)
        finally:
            NewtonMuJoCoStep._restore_solver_state(env, solver_step, solver_fields)

        return q_grad, qd_grad, joint_f_grad, None


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
        sim_substeps: int = 1,
        mujoco_integrator: str = "euler",
        cartpole_reward: CartpoleRewardWeights | None = None,
        ant_reward: AntRewardWeights | None = None,
        hopper_reward: HopperRewardWeights | None = None,
        cheetah_reward: CheetahRewardWeights | None = None,
        acrobot_reward: AcrobotRewardWeights | None = None,
        contact_reward: ContactTargetRewardWeights | None = None,
        acrobot_actuation: str = "elbow",
        ant_asset: str = "diffrl",
        ant_disable_joint_limits: bool = False,
        ant_density_override: float | None = None,
        ant_contact_margin: float = 0.0,
        ant_contact_gap: float | None = None,
        ant_contact_mu: float | None = None,
        ant_joint_damping: float | None = None,
        ant_armature: float | None = None,
        ant_min_up: float | None = None,
        ant_start_height: float | None = None,
        ant_start_joint_q: list[float] | None = None,
        ant_reset_position_scale: float = 0.1,
        ant_reset_angle_scale: float = math.pi / 24.0,
        ant_reset_joint_scale: float = 0.2,
        ant_reset_velocity_scale: float = 0.25,
        ant_termination_height: float | None = None,
        ant_max_healthy_height: float = ANT_MAX_HEALTHY_HEIGHT,
        ant_observation_style: str | None = None,
        ant_reward_style: str | None = None,
        ant_dof_limit_mode: str = "abs",
        ant_action_order: str = "joint",
        ant_smooth_up_reward: bool = False,
        ant_reward_min_up: float | None = None,
        ant_reward_min_height: float | None = None,
        hopper_reward_style: str = "diffrl",
        hopper_start_joint_q: list[float] | None = None,
        hopper_contact_mu: float = 0.9,
        hopper_joint_damping: float = 2.0,
        hopper_armature: float = 1.0,
        hopper_termination_height: float = HOPPER_TERMINATION_HEIGHT,
        hopper_termination_angle: float = HOPPER_TERMINATION_ANGLE,
        hopper_termination_height_tolerance: float = HOPPER_TERMINATION_HEIGHT_TOLERANCE,
        hopper_reset_position_scale: float = 0.05,
        hopper_reset_angle_scale: float = 0.1,
        hopper_reset_joint_scale: float = 0.05,
        hopper_reset_velocity_scale: float = 0.05,
        phase_observation: bool = False,
        phase_period: int = 60,
        hopper_terminate_angle: bool = False,
        locomotion_disable_joint_limits: bool = False,
        mujoco_smooth_adjoint: str = "off",
        mujoco_smooth_friction_viscosity: float = 10.0,
        mujoco_smooth_friction_scale: float = 0.01,
        mujoco_smooth_friction_bypass_kf: float = 0.0,
        mujoco_smooth_penalty_damping_alpha: float = 0.0,
        mujoco_smooth_friction_surrogate_alpha: float = 0.9,
    ):
        self.env_name = env_name
        self.num_envs = num_envs
        self.torch_device = torch.device(device)
        self.wp_device = wp.device_from_torch(self.torch_device)
        self.dt = dt
        self.sim_substeps = max(1, int(sim_substeps))
        self.force_scale = force_scale
        self.contact_backend = contact_backend
        self.mujoco_integrator = mujoco_integrator
        self.cartpole_reward = cartpole_reward or CartpoleRewardWeights()
        self.ant_reward = ant_reward or AntRewardWeights()
        self.hopper_reward = hopper_reward or HopperRewardWeights()
        self.cheetah_reward = cheetah_reward or CheetahRewardWeights()
        self.acrobot_reward = acrobot_reward or AcrobotRewardWeights()
        self.contact_reward = contact_reward or ContactTargetRewardWeights()
        self.acrobot_actuation = acrobot_actuation
        if env_name == "ant":
            ant_defaults = ant_defaults_for_asset(ant_asset)
            if ant_reward is None:
                self.ant_reward.dof_limit_cost = ant_defaults["dof_limit_cost"]
            if ant_density_override is None:
                ant_density_override = ant_defaults["density_override"]
            if ant_contact_mu is None:
                ant_contact_mu = ant_defaults["contact_mu"]
            if ant_joint_damping is None:
                ant_joint_damping = ant_defaults["joint_damping"]
            if ant_armature is None:
                ant_armature = ant_defaults["armature"]
            if ant_start_height is None:
                ant_start_height = ant_defaults["start_height"]
            if ant_start_joint_q is None:
                ant_start_joint_q = list(ant_defaults["start_joint_q"])
            if ant_termination_height is None:
                ant_termination_height = ant_defaults["termination_height"]
            if ant_observation_style is None:
                ant_observation_style = ant_defaults["observation_style"]
            if ant_reward_style is None:
                ant_reward_style = ant_defaults["reward_style"]

        self.ant_asset = ant_asset
        self.ant_disable_joint_limits = ant_disable_joint_limits
        self.ant_density_override = ant_density_override
        self.ant_contact_margin = ant_contact_margin
        self.ant_contact_gap = ant_contact_gap
        self.ant_contact_mu = ant_contact_mu
        self.ant_joint_damping = ant_joint_damping
        self.ant_armature = ant_armature
        self.ant_min_up = ant_min_up
        self.ant_start_height = ant_start_height
        self.ant_start_joint_q = ant_start_joint_q
        self.ant_reset_position_scale = ant_reset_position_scale
        self.ant_reset_angle_scale = ant_reset_angle_scale
        self.ant_reset_joint_scale = ant_reset_joint_scale
        self.ant_reset_velocity_scale = ant_reset_velocity_scale
        self.ant_termination_height = ant_termination_height
        self.ant_max_healthy_height = ant_max_healthy_height
        self.ant_observation_style = ant_observation_style
        self.ant_reward_style = ant_reward_style
        if ant_dof_limit_mode not in {"abs", "upper"}:
            raise ValueError("ant_dof_limit_mode must be 'abs' or 'upper'")
        self.ant_dof_limit_mode = ant_dof_limit_mode
        self.ant_action_order = ant_action_order
        self.ant_smooth_up_reward = ant_smooth_up_reward
        self.ant_reward_min_up = ant_reward_min_up
        self.ant_reward_min_height = ant_reward_min_height
        self.hopper_reward_style = hopper_reward_style
        self.hopper_start_joint_q = hopper_start_joint_q
        self.hopper_contact_mu = hopper_contact_mu
        self.hopper_joint_damping = hopper_joint_damping
        self.hopper_armature = hopper_armature
        self.hopper_termination_height = hopper_termination_height
        self.hopper_termination_angle = hopper_termination_angle
        self.hopper_termination_height_tolerance = hopper_termination_height_tolerance
        self.hopper_reset_position_scale = hopper_reset_position_scale
        self.hopper_reset_angle_scale = hopper_reset_angle_scale
        self.hopper_reset_joint_scale = hopper_reset_joint_scale
        self.hopper_reset_velocity_scale = hopper_reset_velocity_scale
        self.phase_observation = phase_observation
        self.phase_period = max(1, int(phase_period))
        self.hopper_terminate_angle = hopper_terminate_angle
        self.locomotion_disable_joint_limits = locomotion_disable_joint_limits
        self.mujoco_smooth_adjoint = mujoco_smooth_adjoint
        self.mujoco_smooth_friction_viscosity = mujoco_smooth_friction_viscosity
        self.mujoco_smooth_friction_scale = mujoco_smooth_friction_scale
        self.mujoco_smooth_friction_bypass_kf = mujoco_smooth_friction_bypass_kf
        self.mujoco_smooth_penalty_damping_alpha = mujoco_smooth_penalty_damping_alpha
        self.mujoco_smooth_friction_surrogate_alpha = mujoco_smooth_friction_surrogate_alpha
        self.acrobot_link_length = 1.0
        self.contact_body_radius = 0.22
        self.contact_target_offset = torch.tensor([1.5, 0.0, 0.0], dtype=torch.float32, device=self.torch_device)
        self.world_spacing: tuple[float, float, float] | None = None
        self.planar_joint_limit_lower: torch.Tensor | None = None
        self.planar_joint_limit_upper: torch.Tensor | None = None

        if env_name == "cartpole":
            self._build_cartpole()
        elif env_name == "acrobot":
            self._build_acrobot()
        elif is_contact_target_env(env_name):
            self._build_contact_target(capsule=env_name == "contact_capsule")
        elif env_name == "ant":
            self._build_ant()
        elif env_name == "hopper":
            self._build_hopper()
        elif env_name == "cheetah":
            self._build_cheetah()
        else:
            raise ValueError(f"unknown env_name: {env_name}")

        use_contacts = contact_backend != "none"
        self.nconmax = None
        self.njmax = None
        if use_contacts:
            # SolverMuJoCo treats these as per-world capacities.
            self.nconmax = 128
            self.njmax = 512
        self.solver = SolverMuJoCo(
            self.model,
            requires_grad=True,
            disable_contacts=not use_contacts,
            use_mujoco_contacts=contact_backend != "newton",
            integrator=mujoco_integrator,
            solver="newton",
            iterations=8,
            ls_iterations=8,
            update_data_interval=1,
            nconmax=self.nconmax,
            njmax=self.njmax,
        )
        # MuJoCo's implicit Euler damping path currently goes through a dense
        # no-grad solve in MJWarp. Keep physical damping forces, but integrate
        # them explicitly so SHAC receives reliable state gradients.
        self.solver.mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_EULERDAMP
        self.solver.mjw_model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
        if use_contacts and mujoco_smooth_adjoint != "off":
            mujoco_warp.enable_smooth_adjoint(
                self.solver.mjw_data,
                friction_viscosity=mujoco_smooth_friction_viscosity,
                friction_scale=mujoco_smooth_friction_scale,
                friction_bypass_kf=mujoco_smooth_friction_bypass_kf,
                free_body_adjoint=mujoco_smooth_adjoint == "free_body",
                penalty_damping_alpha=mujoco_smooth_penalty_damping_alpha,
                friction_surrogate_adjoint=mujoco_smooth_adjoint == "surrogate",
                friction_surrogate_alpha=mujoco_smooth_friction_surrogate_alpha,
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
        if env_name == "ant":
            self.ant_start_rotation = normalize_vec(self.start_q[:, 3:7])
            self.ant_inv_start_rotation = quat_conjugate(self.ant_start_rotation)
        else:
            self.ant_start_rotation = torch.tensor(ANT_START_ROT, dtype=torch.float32, device=self.torch_device).view(
                1, 4
            ).repeat(self.num_envs, 1)
            self.ant_inv_start_rotation = quat_conjugate(self.ant_start_rotation)
        self.ant_basis_x = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.torch_device).repeat(
            self.num_envs, 1
        )
        self.ant_basis_y = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=self.torch_device).repeat(
            self.num_envs, 1
        )
        self.ant_targets = torch.tensor([10000.0, 0.0, 0.0], dtype=torch.float32, device=self.torch_device).repeat(
            self.num_envs, 1
        )
        self.ant_actuator_dof_indices = None
        if env_name == "ant" and getattr(self.solver, "mj_model", None) is not None:
            mj_model = self.solver.mj_model
            dof_indices = []
            for actuator_id in range(mj_model.nu):
                joint_id = int(mj_model.actuator(actuator_id).trnid[0])
                dof_indices.append(int(mj_model.jnt_dofadr[joint_id]))
            self.ant_actuator_dof_indices = torch.tensor(dof_indices, dtype=torch.long, device=self.torch_device)
        self.num_obs = int(self.observe(self.start_q, self.start_qd).shape[-1])

    def _build_contact_target(self, *, capsule: bool) -> None:
        source = newton.ModelBuilder(up_axis="Y")
        SolverMuJoCo.register_custom_attributes(source)
        source.default_shape_cfg.ke = 4.0e4
        source.default_shape_cfg.kd = 1.0e4
        source.default_shape_cfg.kf = 3.0e3
        source.default_shape_cfg.mu = 0.75

        radius = self.contact_body_radius
        body = source.add_link(
            xform=wp.transform([0.0, radius * 0.98, 0.0], wp.quat_identity()),
            mass=1.0,
            label="contact_body",
        )
        if capsule:
            source.add_shape_capsule(
                body,
                radius=radius,
                half_height=0.34,
                color=(0.15, 0.55, 0.95),
                label="target_capsule",
            )
        else:
            source.add_shape_sphere(
                body,
                radius=radius,
                color=(0.15, 0.65, 0.45),
                label="target_sphere",
            )
        source.add_articulation([source.add_joint_free(body)], label="contact_target")

        builder = newton.ModelBuilder(up_axis="Y")
        SolverMuJoCo.register_custom_attributes(builder)
        builder.replicate(source, self.num_envs, spacing=(3.0, 0.0, 0.0))
        ground_cfg = newton.ModelBuilder.ShapeConfig(ke=4.0e4, kd=1.0e4, kf=3.0e3, mu=0.75)
        builder.add_ground_plane(cfg=ground_cfg)
        self.model = builder.finalize(device=self.wp_device, requires_grad=True)
        self.num_actions = 2

    def _build_acrobot(self) -> None:
        source = newton.ModelBuilder(up_axis="Y")
        SolverMuJoCo.register_custom_attributes(source)
        source.default_joint_cfg.armature = 0.01
        source.default_joint_cfg.damping = 0.05
        source.default_joint_cfg.limit_lower = -1.0e10
        source.default_joint_cfg.limit_upper = 1.0e10
        source.default_shape_cfg.density = 500.0

        hx = 0.5 * self.acrobot_link_length
        hy = 0.045
        hz = 0.045
        link1 = source.add_link(xform=wp.transform([hx, 0.0, 0.0], wp.quat_identity()), mass=1.0)
        source.add_shape_box(link1, hx=hx, hy=hy, hz=hz, color=(0.1, 0.55, 0.95), label="upper_link")
        link2 = source.add_link(xform=wp.transform([3.0 * hx, 0.0, 0.0], wp.quat_identity()), mass=1.0)
        source.add_shape_box(link2, hx=hx, hy=hy, hz=hz, color=(0.95, 0.35, 0.15), label="lower_link")

        j0 = source.add_joint_revolute(
            parent=-1,
            child=link1,
            parent_xform=wp.transform([0.0, 0.0, 0.0], wp.quat_identity()),
            child_xform=wp.transform([-hx, 0.0, 0.0], wp.quat_identity()),
            axis=[0.0, 0.0, 1.0],
            effort_limit=1.0e6,
        )
        j1 = source.add_joint_revolute(
            parent=link1,
            child=link2,
            parent_xform=wp.transform([hx, 0.0, 0.0], wp.quat_identity()),
            child_xform=wp.transform([-hx, 0.0, 0.0], wp.quat_identity()),
            axis=[0.0, 0.0, 1.0],
            effort_limit=1.0e6,
        )
        source.add_articulation([j0, j1], label="acrobot")
        source.joint_q[0] = -0.5 * math.pi
        source.joint_q[1] = 0.0

        builder = newton.ModelBuilder(up_axis="Y")
        SolverMuJoCo.register_custom_attributes(builder)
        builder.replicate(source, self.num_envs, spacing=(3.0, 0.0, 0.0))
        self.model = builder.finalize(device=self.wp_device, requires_grad=True)
        self.num_actions = 2 if self.acrobot_actuation == "both" else 1

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
        source.default_shape_cfg.mu = self.ant_contact_mu
        source.default_shape_cfg.margin = self.ant_contact_margin
        source.default_shape_cfg.gap = self.ant_contact_gap
        source.default_joint_cfg.limit_ke = 1.0e3
        source.default_joint_cfg.limit_kd = 1.0e1
        if self.ant_asset == "nv":
            ant_asset = Path(newton.examples.get_asset("nv_ant.xml"))
            ant_source = str(ant_asset)
            ant_up_axis = "Y"
            armature_scale = 1.0
        else:
            ant_asset = DIFFRL_ROOT / "envs" / "assets" / "ant.xml"
            ant_source = str(ant_asset)
            ant_up_axis = "Z"
            armature_scale = 50.0
        if self.ant_density_override is not None:
            import xml.etree.ElementTree as ET

            root = ET.fromstring(ant_asset.read_text())
            for elem in root.iter("geom"):
                if "density" in elem.attrib:
                    elem.set("density", f"{self.ant_density_override:g}")
            ant_source = ET.tostring(root, encoding="unicode")
        ignore_names = ("^floor$",) if self.ant_asset == "nv" else ()
        source.add_mjcf(ant_source, up_axis=ant_up_axis, armature_scale=armature_scale, ignore_names=ignore_names)
        if self.ant_armature is not None:
            for dof_id in range(6, len(source.joint_armature)):
                source.joint_armature[dof_id] = self.ant_armature
        if self.ant_joint_damping is not None:
            damping_alias = source.custom_attributes.get("mujoco:dof_passive_damping")
            for dof_id in range(6, len(source.joint_damping)):
                source.joint_damping[dof_id] = self.ant_joint_damping
                if damping_alias is not None:
                    if isinstance(damping_alias.values, dict):
                        damping_alias.values[dof_id] = self.ant_joint_damping
                    elif dof_id < len(damping_alias.values):
                        damping_alias.values[dof_id] = self.ant_joint_damping
        if self.ant_contact_margin != 0.0:
            source.shape_margin = [self.ant_contact_margin] * len(source.shape_margin)
        if self.ant_contact_gap is not None:
            source.shape_gap = [self.ant_contact_gap] * len(source.shape_gap)
        ant_joint_limit_lower = list(source.joint_limit_lower[6:14])
        ant_joint_limit_upper = list(source.joint_limit_upper[6:14])
        if self.ant_disable_joint_limits:
            for dof_id in range(6, len(source.joint_limit_lower)):
                source.joint_limit_lower[dof_id] = -1.0e10
                source.joint_limit_upper[dof_id] = 1.0e10
                source.joint_limit_ke[dof_id] = 0.0
                source.joint_limit_kd[dof_id] = 0.0
        self.ant_joint_limit_lower = torch.tensor(
            ant_joint_limit_lower, dtype=torch.float32, device=self.torch_device
        )
        self.ant_joint_limit_upper = torch.tensor(
            ant_joint_limit_upper, dtype=torch.float32, device=self.torch_device
        )
        if self.ant_start_height is not None:
            source.joint_q[1] = self.ant_start_height
            source.joint_target_q[1] = self.ant_start_height
        # Keep Ant assets in Newton's Y-up convention.  Both the DiffRL import
        # and the NV MJCF need the -90 deg X free-root quaternion here to match
        # the stable diagonal support stance encoded by ANT_START_JOINT_Q after
        # Newton exports the model to the MuJoCo/MJWarp runtime.
        source.joint_q[2] = 0.0
        source.joint_target_q[2] = 0.0
        source.joint_q[3:7] = ANT_START_ROT
        source.joint_target_q[3:7] = ANT_START_ROT
        ant_start_joint_q = self.ant_start_joint_q if self.ant_start_joint_q is not None else list(ANT_START_JOINT_Q)
        if len(ant_start_joint_q) != 8:
            raise ValueError("ant_start_joint_q must contain 8 values")
        source.joint_q[7:15] = ant_start_joint_q
        source.joint_target_q[7:15] = ant_start_joint_q
        source.shape_material_ke = [4.0e4] * len(source.shape_material_ke)
        source.shape_material_kd = [1.0e4] * len(source.shape_material_kd)
        source.shape_material_kf = [3.0e3] * len(source.shape_material_kf)
        source.shape_material_mu = [self.ant_contact_mu] * len(source.shape_material_mu)

        builder = newton.ModelBuilder(up_axis="Y")
        SolverMuJoCo.register_custom_attributes(builder)
        self.world_spacing = (0.0, 0.0, 4.0) if self.contact_backend == "newton" else (0.0, 0.0, 0.0)
        builder.replicate(source, self.num_envs, spacing=self.world_spacing)
        ground_cfg = newton.ModelBuilder.ShapeConfig(
            ke=4.0e4,
            kd=1.0e4,
            kf=3.0e3,
            mu=self.ant_contact_mu,
            margin=self.ant_contact_margin,
            gap=self.ant_contact_gap,
        )
        builder.add_ground_plane(cfg=ground_cfg)
        self.model = builder.finalize(device=self.wp_device, requires_grad=True)
        self.num_actions = 8

    def _build_planar_locomotion(
        self,
        *,
        asset_name: str,
        num_actions: int,
        start_height: float,
        contact_mu: float,
        joint_damping: float,
        armature: float,
        ignore_inertial_definitions: bool = False,
    ) -> None:
        source = newton.ModelBuilder(up_axis="Y")
        SolverMuJoCo.register_custom_attributes(source)
        source.default_shape_cfg.ke = 2.0e4
        source.default_shape_cfg.kd = 1.0e3
        source.default_shape_cfg.kf = 1.0e3
        source.default_shape_cfg.mu = contact_mu
        source.default_joint_cfg.armature = armature
        source.default_joint_cfg.damping = joint_damping
        source.default_joint_cfg.limit_ke = 1.0e3
        source.default_joint_cfg.limit_kd = 1.0e1
        source.add_mjcf(
            str(DIFFRL_ROOT / "envs" / "assets" / asset_name),
            up_axis="Z",
            ignore_inertial_definitions=ignore_inertial_definitions,
        )
        source.joint_q[1] = start_height
        source.joint_target_q[1] = start_height
        hopper_start_joint_q = self.hopper_start_joint_q
        if asset_name == "hopper.xml" and hopper_start_joint_q is None:
            hopper_start_joint_q = list(HOPPER_START_JOINT_Q)
        if asset_name == "hopper.xml" and hopper_start_joint_q is not None:
            if len(hopper_start_joint_q) != num_actions:
                raise ValueError(f"hopper_start_joint_q must have {num_actions} values")
            source.joint_q[3 : 3 + num_actions] = hopper_start_joint_q
            source.joint_target_q[3 : 3 + num_actions] = hopper_start_joint_q
            self.hopper_start_joint_q = list(hopper_start_joint_q)
        self.planar_joint_limit_lower = torch.tensor(
            source.joint_limit_lower[3 : 3 + num_actions], dtype=torch.float32, device=self.torch_device
        )
        self.planar_joint_limit_upper = torch.tensor(
            source.joint_limit_upper[3 : 3 + num_actions], dtype=torch.float32, device=self.torch_device
        )
        for dof_id in range(3, len(source.joint_limit_lower)):
            if self.locomotion_disable_joint_limits:
                source.joint_limit_lower[dof_id] = -1.0e10
                source.joint_limit_upper[dof_id] = 1.0e10
                source.joint_limit_ke[dof_id] = 0.0
                source.joint_limit_kd[dof_id] = 0.0
            else:
                source.joint_limit_ke[dof_id] = 1.0e3
                source.joint_limit_kd[dof_id] = 1.0e1
        if self.locomotion_disable_joint_limits:
            self.planar_joint_limit_lower = None
            self.planar_joint_limit_upper = None
        source.shape_material_ke = [2.0e4] * len(source.shape_material_ke)
        source.shape_material_kd = [1.0e3] * len(source.shape_material_kd)
        source.shape_material_kf = [1.0e3] * len(source.shape_material_kf)
        source.shape_material_mu = [contact_mu] * len(source.shape_material_mu)

        builder = newton.ModelBuilder(up_axis="Y")
        SolverMuJoCo.register_custom_attributes(builder)
        self.world_spacing = (0.0, 0.0, 4.0) if self.contact_backend == "newton" else (0.0, 0.0, 0.0)
        builder.replicate(source, self.num_envs, spacing=self.world_spacing)
        ground_cfg = newton.ModelBuilder.ShapeConfig(ke=2.0e4, kd=1.0e3, kf=1.0e3, mu=contact_mu)
        builder.add_ground_plane(cfg=ground_cfg)
        self.model = builder.finalize(device=self.wp_device, requires_grad=True)
        self.num_actions = num_actions

    def _build_hopper(self) -> None:
        self._build_planar_locomotion(
            asset_name="hopper.xml",
            num_actions=3,
            start_height=HOPPER_START_HEIGHT,
            contact_mu=self.hopper_contact_mu,
            joint_damping=self.hopper_joint_damping,
            armature=self.hopper_armature,
        )

    def _build_cheetah(self) -> None:
        self._build_planar_locomotion(
            asset_name="half_cheetah.xml",
            num_actions=6,
            start_height=CHEETAH_START_HEIGHT,
            contact_mu=1.0,
            joint_damping=1.0,
            armature=0.1,
            ignore_inertial_definitions=True,
        )

    def zero_solver_buffers(self) -> None:
        data = self.solver.mjw_data
        data.qacc_warmstart.zero_()
        data.qfrc_applied.zero_()
        data.ctrl.zero_()
        data.act.zero_()
        data.xfrc_applied.zero_()

    def reset_solver_data(self, env_ids: torch.Tensor | None = None) -> None:
        if self.contact_backend != "mujoco":
            return
        if env_ids is None:
            mujoco_warp.reset_data(self.solver.mjw_model, self.solver.mjw_data)
            wp.synchronize()
            return
        if env_ids.numel() == 0:
            return
        reset_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.torch_device)
        reset_mask[env_ids] = True
        reset_mask_wp = wp.from_torch(reset_mask.contiguous(), dtype=wp.bool)
        mujoco_warp.reset_data(self.solver.mjw_model, self.solver.mjw_data, reset=reset_mask_wp)
        wp.synchronize()

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

        current_q_wp = q_wp
        current_qd_wp = qd_wp
        intermediates = []
        sub_dt = self.dt / self.sim_substeps
        for substep in range(self.sim_substeps):
            state_in = self.model.state(requires_grad=requires_grad)
            state_out = self.model.state(requires_grad=requires_grad)
            control = self.model.control(requires_grad=requires_grad)
            if substep == self.sim_substeps - 1:
                next_q_wp = q_out_wp
                next_qd_wp = qd_out_wp
            else:
                next_q_wp = wp.empty_like(q_wp, requires_grad=requires_grad)
                next_qd_wp = wp.empty_like(qd_wp, requires_grad=requires_grad)
                intermediates.extend([next_q_wp, next_qd_wp])
            state_in.joint_q = current_q_wp
            state_in.joint_qd = current_qd_wp
            state_out.joint_q = next_q_wp
            state_out.joint_qd = next_qd_wp
            control.joint_f = f_wp
            contacts = None
            if self.contact_backend == "newton":
                newton.eval_fk(self.model, state_in.joint_q, state_in.joint_qd, state_in)
                contacts = self.model.collide(state_in, self.contacts)
            self.solver.step(state_in, state_out, control, contacts, sub_dt)
            current_q_wp = next_q_wp
            current_qd_wp = next_qd_wp
        wp.synchronize()
        self.last_state = state_out
        return {
            "q": q_wp,
            "qd": qd_wp,
            "joint_f": f_wp,
            "q_out": q_out_wp,
            "qd_out": qd_out_wp,
            "_intermediates": intermediates,
        }

    def step(self, q: torch.Tensor, qd: torch.Tensor, joint_f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return NewtonMuJoCoStep.apply(q, qd, joint_f, self.step_ctx)

    def reset(self, noise: float = 0.0, stochastic_init: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        self.reset_solver_data()
        q = self.start_q.clone()
        qd = self.start_qd.clone()
        if self.env_name == "ant" and stochastic_init:
            q, qd = self._randomize_ant_reset(q, qd)
        elif self.env_name == "hopper" and stochastic_init:
            q, qd = self._randomize_planar_reset(
                q,
                qd,
                position_scale=self.hopper_reset_position_scale,
                angle_scale=self.hopper_reset_angle_scale,
                joint_scale=self.hopper_reset_joint_scale,
                velocity_scale=self.hopper_reset_velocity_scale,
            )
        elif self.env_name == "cheetah" and stochastic_init:
            q, qd = self._randomize_planar_reset(q, qd, position_scale=0.1, angle_scale=0.2, joint_scale=0.1, velocity_scale=0.5)
        elif is_contact_target_env(self.env_name) and stochastic_init:
            q[:, [0, 2]] = q[:, [0, 2]] + 0.15 * (torch.rand((q.shape[0], 2), device=self.torch_device) - 0.5)
            qd[:] = 0.2 * (torch.rand_like(qd) - 0.5)
        elif stochastic_init:
            q = q + math.pi * (torch.rand_like(q) - 0.5)
            qd = qd + 0.5 * (torch.rand_like(qd) - 0.5)
        elif noise > 0.0:
            if is_contact_target_env(self.env_name):
                q[:, [0, 2]] = q[:, [0, 2]] + noise * torch.randn((q.shape[0], 2), device=self.torch_device)
            else:
                q = q + noise * torch.randn_like(q)
            qd = qd + 0.25 * noise * torch.randn_like(qd)
            if self.env_name == "ant":
                q[:, 3:7] = normalize_vec(q[:, 3:7])
            elif is_contact_target_env(self.env_name):
                q[:, 3:7] = normalize_vec(q[:, 3:7])
        return q, qd

    def _randomize_ant_reset(self, q: torch.Tensor, qd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.ant_reset_position_scale > 0.0:
            q[:, 0:3] = q[:, 0:3] + self.ant_reset_position_scale * (
                torch.rand((q.shape[0], 3), device=self.torch_device) - 0.5
            ) * 2.0
        if self.ant_reset_angle_scale > 0.0:
            angle = self.ant_reset_angle_scale * (torch.rand(q.shape[0], device=self.torch_device) - 0.5) * 2.0
            axis = normalize_vec(torch.rand((q.shape[0], 3), device=self.torch_device) - 0.5)
            q[:, 3:7] = normalize_vec(quat_mul(q[:, 3:7], quat_from_angle_axis(angle, axis)))
        if self.ant_reset_joint_scale > 0.0:
            q[:, 7:] = q[:, 7:] + self.ant_reset_joint_scale * (torch.rand_like(q[:, 7:]) - 0.5) * 2.0
        if self.ant_reset_velocity_scale > 0.0:
            qd[:] = self.ant_reset_velocity_scale * (torch.rand_like(qd) - 0.5) * 2.0
        else:
            qd.zero_()
        return q, qd

    def _randomize_planar_reset(
        self,
        q: torch.Tensor,
        qd: torch.Tensor,
        *,
        position_scale: float,
        angle_scale: float,
        joint_scale: float,
        velocity_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q[:, 0:2] = q[:, 0:2] + position_scale * (torch.rand((q.shape[0], 2), device=self.torch_device) - 0.5) * 2.0
        q[:, 2] = q[:, 2] + angle_scale * (torch.rand(q.shape[0], device=self.torch_device) - 0.5)
        q[:, 3:] = q[:, 3:] + joint_scale * (torch.rand_like(q[:, 3:]) - 0.5) * 2.0
        q = self._clamp_planar_joint_q(q)
        qd[:] = velocity_scale * (torch.rand_like(qd) - 0.5) * 2.0
        return q, qd

    def _clamp_planar_joint_q(self, q: torch.Tensor, margin: float = 0.02) -> torch.Tensor:
        if self.planar_joint_limit_lower is None or self.planar_joint_limit_upper is None:
            return q
        lower = self.planar_joint_limit_lower.view(1, -1)
        upper = self.planar_joint_limit_upper.view(1, -1)
        finite = torch.isfinite(lower) & torch.isfinite(upper) & (upper > lower + 2.0 * margin)
        if not bool(finite.any().detach().cpu()):
            return q
        q_joints = q[:, 3:]
        clamped = torch.maximum(torch.minimum(q_joints, upper - margin), lower + margin)
        q = q.clone()
        q[:, 3:] = torch.where(finite, clamped, q_joints)
        return q

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
        self.reset_solver_data(env_ids)
        reset_q = self.start_q[env_ids].clone()
        reset_qd = self.start_qd[env_ids].clone()
        if self.env_name == "ant" and stochastic_init:
            reset_q, reset_qd = self._randomize_ant_reset(reset_q, reset_qd)
        elif self.env_name == "hopper" and stochastic_init:
            reset_q, reset_qd = self._randomize_planar_reset(
                reset_q,
                reset_qd,
                position_scale=self.hopper_reset_position_scale,
                angle_scale=self.hopper_reset_angle_scale,
                joint_scale=self.hopper_reset_joint_scale,
                velocity_scale=self.hopper_reset_velocity_scale,
            )
        elif self.env_name == "cheetah" and stochastic_init:
            reset_q, reset_qd = self._randomize_planar_reset(
                reset_q,
                reset_qd,
                position_scale=0.1,
                angle_scale=0.2,
                joint_scale=0.1,
                velocity_scale=0.5,
            )
        elif is_contact_target_env(self.env_name) and stochastic_init:
            reset_q[:, [0, 2]] = reset_q[:, [0, 2]] + 0.15 * (
                torch.rand((reset_q.shape[0], 2), device=self.torch_device) - 0.5
            )
            reset_qd[:] = 0.2 * (torch.rand_like(reset_qd) - 0.5)
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

    def phase_features(self, phase: torch.Tensor | None, count: int) -> torch.Tensor:
        if phase is None:
            phase = torch.zeros(count, dtype=torch.float32, device=self.torch_device)
        phase = phase.to(dtype=torch.float32, device=self.torch_device)
        angle = 2.0 * math.pi * phase / float(self.phase_period)
        return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)

    def ant_dof_pos_scaled(self, q: torch.Tensor) -> torch.Tensor:
        lower = self.ant_joint_limit_lower[: q.shape[-1] - 7].view(1, -1)
        upper = self.ant_joint_limit_upper[: q.shape[-1] - 7].view(1, -1)
        center = 0.5 * (upper + lower)
        half_range = 0.5 * (upper - lower).clamp(min=1.0e-6)
        return ((q[:, 7:] - center) / half_range).clamp(-5.0, 5.0)

    def ant_morphology_metrics(
        self, q: torch.Tensor, qd: torch.Tensor, action: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if self.env_name != "ant":
            raise ValueError("ant_morphology_metrics is only valid for Ant")
        abs_joint = self.ant_dof_pos_scaled(q).abs()
        abs_action = action.abs()
        return {
            "mean_abs_joint_pos_scaled": abs_joint.mean(dim=-1),
            "max_abs_joint_pos_scaled": abs_joint.max(dim=-1).values,
            "mean_joint_limit_fraction": (abs_joint > 0.98).to(torch.float32).mean(dim=-1),
            "mean_abs_action": abs_action.mean(dim=-1),
            "max_abs_action": abs_action.max(dim=-1).values,
            "mean_abs_joint_velocity": qd[:, 6:].abs().mean(dim=-1),
        }

    def ant_pose_terms(
        self, q: torch.Tensor, qd: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return torso_pos, torso_quat, lin_vel, ang_vel, up_vec, heading_alignment

    def ant_isaac_observation(
        self, q: torch.Tensor, qd: torch.Tensor, prev_action: torch.Tensor, phase: torch.Tensor | None
    ) -> torch.Tensor:
        torso_pos, torso_quat, lin_vel, ang_vel, up_vec, heading_alignment = self.ant_pose_terms(q, qd)
        inv_torso_quat = quat_conjugate(torso_quat)
        vel_local = quat_rotate(inv_torso_quat, lin_vel)
        angvel_local = quat_rotate(inv_torso_quat, ang_vel)
        heading_vec = quat_rotate(torso_quat, self.ant_basis_x[: q.shape[0]])
        target_dirs = normalize_vec(self.ant_targets[: q.shape[0]] + self.start_q[: q.shape[0], 0:3] - torso_pos)
        target_dirs = target_dirs.clone()
        target_dirs[:, 1] = 0.0
        target_dirs = normalize_vec(target_dirs)
        signed_target = torch.cross(heading_vec, target_dirs, dim=-1)[:, 1:2]
        angle_to_target = torch.atan2(signed_target, heading_alignment.clamp(-1.0, 1.0))
        yaw = torch.atan2(heading_vec[:, 2:3], heading_vec[:, 0:1])
        roll = torch.atan2(up_vec[:, 0:1], up_vec[:, 1:2])
        obs = torch.cat(
            [
                torso_pos[:, 1:2],
                vel_local,
                angvel_local,
                torch.atan2(torch.sin(yaw), torch.cos(yaw)),
                torch.atan2(torch.sin(roll), torch.cos(roll)),
                torch.atan2(torch.sin(angle_to_target), torch.cos(angle_to_target)),
                up_vec[:, 1:2],
                heading_alignment,
                self.ant_dof_pos_scaled(q),
                self.ant_reward.dof_vel_scale * qd[:, 6:],
                prev_action.clone(),
            ],
            dim=-1,
        )
        if self.phase_observation:
            obs = torch.cat([obs, self.phase_features(phase, q.shape[0])], dim=-1)
        return obs

    def observe(
        self,
        q: torch.Tensor,
        qd: torch.Tensor,
        prev_action: torch.Tensor | None = None,
        *,
        phase: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.env_name == "cartpole":
            x = q[:, 0:1]
            theta = q[:, 1:2]
            xdot = qd[:, 0:1]
            theta_dot = qd[:, 1:2]
            return torch.cat([x, xdot, torch.sin(theta), torch.cos(theta), theta_dot], dim=-1)

        if self.env_name == "acrobot":
            theta1 = q[:, 0:1]
            theta2 = q[:, 1:2]
            theta12 = theta1 + theta2
            end = self.acrobot_end_effector(q)
            target = self.acrobot_target(q.shape[0])
            return torch.cat(
                [
                    torch.sin(theta1),
                    torch.cos(theta1),
                    torch.sin(theta2),
                    torch.cos(theta2),
                    torch.sin(theta12),
                    torch.cos(theta12),
                    0.2 * qd,
                    end - target,
                    prev_action if prev_action is not None else torch.zeros((q.shape[0], self.num_actions), dtype=q.dtype, device=q.device),
                ],
                dim=-1,
            )

        if is_contact_target_env(self.env_name):
            if prev_action is None:
                prev_action = torch.zeros((q.shape[0], self.num_actions), dtype=torch.float32, device=self.torch_device)
            pos = q[:, 0:3]
            rot = normalize_vec(q[:, 3:7])
            qd_scaled = 0.2 * qd
            target_error = pos - self.contact_target(q.shape[0])
            return torch.cat([pos[:, 1:2], rot, qd_scaled, target_error, prev_action.clone()], dim=-1)

        if is_planar_locomotion_env(self.env_name):
            return torch.cat([q[:, 1:], qd], dim=-1)

        if self.env_name != "ant":
            raise ValueError(f"unknown env_name: {self.env_name}")

        if prev_action is None:
            prev_action = torch.zeros((q.shape[0], self.num_actions), dtype=torch.float32, device=self.torch_device)

        if self.ant_observation_style == "isaac":
            return self.ant_isaac_observation(q, qd, prev_action, phase)

        torso_pos, _, lin_vel, ang_vel, up_vec, heading_alignment = self.ant_pose_terms(q, qd)
        obs = torch.cat(
            [
                torso_pos[:, 1:2],
                normalize_vec(q[:, 3:7]),
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
        if self.phase_observation:
            obs = torch.cat([obs, self.phase_features(phase, q.shape[0])], dim=-1)
        return obs

    def action_to_joint_f(self, action: torch.Tensor) -> torch.Tensor:
        action = torch.clamp(action, -1.0, 1.0)
        joint_f = torch.zeros((self.num_envs, self.qd_dim), dtype=torch.float32, device=self.torch_device)
        if self.env_name == "cartpole":
            joint_f[:, 0] = action[:, 0] * self.force_scale
        elif self.env_name == "acrobot":
            if self.acrobot_actuation == "both":
                joint_f[:, 0:2] = action[:, 0:2] * self.force_scale
            else:
                joint_f[:, 1] = action[:, 0] * self.force_scale
        elif is_contact_target_env(self.env_name):
            joint_f[:, 0] = action[:, 0] * self.force_scale
            joint_f[:, 2] = action[:, 1] * self.force_scale
        elif is_planar_locomotion_env(self.env_name):
            joint_f[:, 3 : 3 + self.num_actions] = action[:, : self.num_actions] * self.force_scale
        elif self.env_name == "ant":
            if self.ant_action_order == "actuator" and self.ant_actuator_dof_indices is not None:
                joint_f[:, self.ant_actuator_dof_indices] = action[:, : self.num_actions] * self.force_scale
            else:
                joint_f[:, 6 : 6 + self.num_actions] = action[:, : self.num_actions] * self.force_scale
        else:
            raise ValueError(f"unknown env_name: {self.env_name}")
        return joint_f

    def acrobot_target(self, count: int) -> torch.Tensor:
        return torch.tensor([0.0, 1.65], dtype=torch.float32, device=self.torch_device).view(1, 2).repeat(count, 1)

    def acrobot_end_effector(self, q: torch.Tensor) -> torch.Tensor:
        theta1 = q[:, 0]
        theta2 = q[:, 1]
        theta12 = theta1 + theta2
        length = self.acrobot_link_length
        x = length * torch.cos(theta1) + length * torch.cos(theta12)
        y = length * torch.sin(theta1) + length * torch.sin(theta12)
        return torch.stack([x, y], dim=-1)

    def contact_target(self, count: int) -> torch.Tensor:
        target = self.start_q[:count, 0:3] + self.contact_target_offset.view(1, 3)
        target = target.clone()
        target[:, 1] = self.contact_body_radius
        return target

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

        if self.env_name == "acrobot":
            end = self.acrobot_end_effector(q)
            target = self.acrobot_target(q.shape[0])
            target_error = (end - target).square().sum(dim=-1)
            weights = self.acrobot_reward
            return -(
                weights.target * target_error
                + weights.velocity * qd.square().sum(dim=-1)
                + weights.action * action.square().sum(dim=-1)
            )

        if is_contact_target_env(self.env_name):
            target = self.contact_target(q.shape[0])
            pos = q[:, 0:3]
            target_error = (pos[:, [0, 2]] - target[:, [0, 2]]).square().sum(dim=-1)
            height_error = (pos[:, 1] - target[:, 1]).square()
            weights = self.contact_reward
            return -(
                weights.target * target_error
                + weights.height * height_error
                + weights.velocity * qd.square().sum(dim=-1)
                + weights.action * action.square().sum(dim=-1)
            )

        if self.env_name == "hopper":
            if obs is None:
                obs = self.observe(q, qd, action)
            weights = self.hopper_reward
            if self.hopper_reward_style == "gym":
                return weights.progress * obs[:, 5] + weights.alive + weights.action * action.square().sum(dim=-1)
            height_diff = obs[:, 0] - (self.hopper_termination_height + self.hopper_termination_height_tolerance)
            height_reward = torch.clamp(height_diff, -1.0, 0.3)
            height_reward = torch.where(height_reward < 0.0, -200.0 * height_reward.square(), height_reward)
            progress_reward = weights.progress * obs[:, 5]
            angle_reward = weights.angle * (1.0 - obs[:, 1].square() / (self.hopper_termination_angle**2))
            return (
                progress_reward
                + weights.height * height_reward
                + angle_reward
                + weights.action * action.square().sum(dim=-1)
            )

        if self.env_name == "cheetah":
            if obs is None:
                obs = self.observe(q, qd, action)
            return obs[:, 8] + self.cheetah_reward.action * action.square().sum(dim=-1)

        if self.env_name != "ant":
            raise ValueError(f"unknown env_name: {self.env_name}")

        if obs is None:
            obs = self.observe(q, qd, action)

        def apply_ant_margin_penalties(reward: torch.Tensor, up_proj: torch.Tensor) -> torch.Tensor:
            if self.ant_reward.up_margin > 0.0 and self.ant_reward_min_up is not None:
                up_shortfall = torch.relu(float(self.ant_reward_min_up) - up_proj)
                reward = reward - self.ant_reward.up_margin * up_shortfall.square()
            if self.ant_reward.height_margin > 0.0 and self.ant_reward_min_height is not None:
                height_shortfall = torch.relu(float(self.ant_reward_min_height) - q[:, 1])
                reward = reward - self.ant_reward.height_margin * height_shortfall.square()
            return reward

        if self.ant_reward_style in {
            "isaac",
            "isaaclab",
            "isaaclab_potential",
            "isaaclab_potential_height",
            "isaac_heading_gated",
        }:
            if self.ant_observation_style == "isaac":
                up_proj = obs[:, 10]
                heading_proj = obs[:, 11]
                dof_pos_scaled = obs[:, 12:20]
                dof_vel = qd[:, 6:]
            else:
                up_proj = obs[:, 27]
                heading_proj = obs[:, 28]
                dof_pos_scaled = self.ant_dof_pos_scaled(q)
                dof_vel = qd[:, 6:]
            weights = self.ant_reward
            heading_reward = torch.where(
                heading_proj > 0.8,
                torch.full_like(heading_proj, weights.heading),
                weights.heading * heading_proj / 0.8,
            )
            if self.ant_smooth_up_reward:
                up_reward = weights.up * torch.clamp(up_proj, min=0.0, max=1.0)
            else:
                up_reward = torch.where(up_proj > 0.93, torch.full_like(up_proj, weights.up), torch.zeros_like(up_proj))
            action_dof_vel = dof_vel
            if self.ant_action_order == "actuator" and self.ant_actuator_dof_indices is not None:
                action_dof_vel = qd[:, self.ant_actuator_dof_indices]
            actions_cost = action.square().sum(dim=-1)
            energy_cost = torch.abs(action * action_dof_vel).sum(dim=-1)
            limit_threshold = 0.99
            if self.ant_dof_limit_mode == "upper":
                dof_limit_violation = torch.relu((dof_pos_scaled - limit_threshold) / (1.0 - limit_threshold))
            else:
                dof_limit_violation = torch.relu((dof_pos_scaled.abs() - limit_threshold) / (1.0 - limit_threshold))
            dof_limit_cost = dof_limit_violation.sum(dim=-1)
            height_term: torch.Tensor | float = 0.0
            if self.ant_reward_style in {"isaac", "isaaclab_potential_height", "isaac_heading_gated"}:
                height_reward = torch.clamp(q[:, 1] - self.ant_termination_height, min=0.0, max=ANT_HEIGHT_REWARD_CAP)
                height_term = weights.height * height_reward
            progress = qd[:, 0]
            if self.ant_reward_style == "isaac_heading_gated":
                progress = progress * torch.clamp(heading_proj, min=0.0)
            reward = (
                weights.progress * progress
                + weights.alive
                + heading_reward
                + up_reward
                + height_term
                - weights.actions_cost * actions_cost
                - weights.energy_cost * energy_cost
                - weights.dof_limit_cost * dof_limit_cost
            )
            return apply_ant_margin_penalties(reward, up_proj)
        progress_reward = obs[:, 5]
        up_proj = obs[:, 27]
        up_reward = self.ant_reward.up * up_proj
        heading_reward = self.ant_reward.heading * obs[:, 28]
        height_reward = torch.clamp(obs[:, 0] - self.ant_termination_height, max=ANT_HEIGHT_REWARD_CAP)
        reward = (
            self.ant_reward.progress * progress_reward
            + up_reward
            + heading_reward
            + self.ant_reward.height * height_reward
            + self.ant_reward.action * action.square().sum(dim=-1)
        )
        return apply_ant_margin_penalties(reward, up_proj)

    def transition_reward(
        self,
        q: torch.Tensor,
        qd: torch.Tensor,
        q_next: torch.Tensor,
        qd_next: torch.Tensor,
        action: torch.Tensor,
        obs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        reward = self.reward(q_next, qd_next, action, obs=obs)
        if self.env_name == "ant" and self.ant_reward_style in {"isaaclab_potential", "isaaclab_potential_height"}:
            control_dt = max(float(self.dt), 1.0e-8)
            potential_progress = (q_next[:, 0] - q[:, 0]) / control_dt
            reward = reward + self.ant_reward.progress * (potential_progress - qd_next[:, 0])
        return reward

    def done(self, q: torch.Tensor, progress: torch.Tensor, episode_length: int) -> torch.Tensor:
        done = progress >= episode_length
        if self.env_name in {"ant", "hopper"}:
            done = torch.logical_or(done, self.fallen_state(q))
        return done

    def fallen_state(self, q: torch.Tensor) -> torch.Tensor:
        if self.env_name == "hopper":
            finite = torch.isfinite(q).all(dim=-1)
            low_height = q[:, 1] < self.hopper_termination_height
            if self.hopper_terminate_angle:
                bad_angle = q[:, 2].abs() > self.hopper_termination_angle
                low_height = torch.logical_or(low_height, bad_angle)
            return torch.logical_and(finite, low_height)
        if self.env_name != "ant":
            return torch.zeros(q.shape[0], dtype=torch.bool, device=q.device)
        finite = torch.isfinite(q).all(dim=-1)
        fallen = torch.logical_and(finite, q[:, 1] < self.ant_termination_height)
        if self.ant_min_up is not None:
            torso_rot = normalize_vec(q[:, 3:7])
            torso_quat = quat_mul(torso_rot, self.ant_inv_start_rotation[: q.shape[0]])
            up_vec = quat_rotate(torso_quat, self.ant_basis_y[: q.shape[0]])
            fallen = torch.logical_or(fallen, torch.logical_and(finite, up_vec[:, 1] < self.ant_min_up))
        return fallen

    def invalid_state(self, q: torch.Tensor, qd: torch.Tensor) -> torch.Tensor:
        invalid = torch.logical_or(~torch.isfinite(q).all(dim=-1), ~torch.isfinite(qd).all(dim=-1))
        if self.env_name == "ant":
            root_disp = q[:, 0:3] - self.start_q[: q.shape[0], 0:3]
            invalid = torch.logical_or(invalid, q[:, 1] > self.ant_max_healthy_height)
            invalid = torch.logical_or(invalid, root_disp[:, 0].abs() > 100.0)
            invalid = torch.logical_or(invalid, root_disp[:, 2].abs() > 100.0)
            invalid = torch.logical_or(invalid, qd.abs().amax(dim=-1) > 100.0)
        elif is_planar_locomotion_env(self.env_name):
            invalid = torch.logical_or(invalid, q[:, 0].abs() > 100.0)
            invalid = torch.logical_or(invalid, q[:, 1].abs() > 10.0)
            invalid = torch.logical_or(invalid, qd.abs().amax(dim=-1) > 100.0)
        elif is_contact_target_env(self.env_name):
            pos_disp = q[:, 0:3] - self.start_q[: q.shape[0], 0:3]
            invalid = torch.logical_or(invalid, q[:, 1] < 0.02)
            invalid = torch.logical_or(invalid, q[:, 1] > 3.0)
            invalid = torch.logical_or(invalid, pos_disp[:, 0].abs() > 20.0)
            invalid = torch.logical_or(invalid, pos_disp[:, 2].abs() > 20.0)
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


class NewtonPolicyMLP(torch.nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list[int],
        *,
        stochastic: bool,
        actor_logstd_init: float,
        device: torch.device,
    ):
        super().__init__()
        layers: list[torch.nn.Module] = []
        last_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(last_dim, hidden_dim))
            layers.append(torch.nn.ELU())
            last_dim = hidden_dim
        self.backbone = torch.nn.Sequential(*layers)
        self.mean = torch.nn.Linear(last_dim, action_dim)
        self.log_std = torch.nn.Parameter(torch.full((action_dim,), actor_logstd_init))
        self.stochastic = stochastic
        self.to(device)

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mean = self.mean(self.backbone(obs))
        if deterministic or not self.stochastic:
            return mean
        std = self.log_std.exp()
        return torch.distributions.Normal(mean, std).rsample()


def make_actor(
    env: NewtonMuJoCoTorchEnv,
    stochastic: bool = False,
    hidden_dims: list[int] | None = None,
    actor_logstd_init: float = -1.0,
    actor_layer_norm: bool = True,
    action_squash: str = "tanh",
) -> torch.nn.Module:
    if hidden_dims is None:
        hidden_dims = [128, 64, 32] if is_locomotion_env(env.env_name) else [64, 64]
    if not actor_layer_norm:
        actor = NewtonPolicyMLP(
            env.num_obs,
            env.num_actions,
            hidden_dims,
            stochastic=stochastic,
            actor_logstd_init=actor_logstd_init,
            device=env.torch_device,
        )
        actor.action_squash = action_squash
        actor.outputs_raw_action = True
        return actor
    cfg = {
        "actor_mlp": {"units": hidden_dims, "activation": "elu"},
        "actor_logstd_init": actor_logstd_init,
    }
    if stochastic:
        actor = ActorStochasticMLP(env.num_obs, env.num_actions, cfg, device=str(env.torch_device))
    else:
        actor = ActorDeterministicMLP(env.num_obs, env.num_actions, cfg, device=str(env.torch_device))
        final = actor.actor[-1]
        if env.env_name == "cartpole" and isinstance(final, torch.nn.Linear):
            torch.nn.init.zeros_(final.weight)
            torch.nn.init.zeros_(final.bias)
    actor.action_squash = action_squash
    actor.outputs_raw_action = True
    return actor


def freeze_actor_backbone(actor: torch.nn.Module) -> None:
    head_prefixes = []
    for module_name in ("mean", "actor", "mu_net"):
        module = getattr(actor, module_name, None)
        if isinstance(module, torch.nn.Linear):
            head_prefixes.append(f"{module_name}.")
        elif isinstance(module, torch.nn.Sequential):
            for idx in range(len(module) - 1, -1, -1):
                if isinstance(module[idx], torch.nn.Linear):
                    head_prefixes.append(f"{module_name}.{idx}.")
                    break
    head_names = tuple(head_prefixes) + ("log_std", "logstd")
    trainable = 0
    for name, param in actor.named_parameters():
        is_head = name.startswith(head_names)
        param.requires_grad_(is_head)
        trainable += int(is_head)
    if trainable == 0:
        raise RuntimeError(f"could not identify actor head parameters in {type(actor).__name__}")


def load_actor_checkpoint(actor: torch.nn.Module, path: Path, device: torch.device) -> None:
    state = torch.load(path, map_location=device)
    try:
        actor.load_state_dict(state)
        return
    except RuntimeError:
        pass

    def linear_indices(prefix: str, weights: dict[str, torch.Tensor]) -> list[int]:
        indices = []
        for key, value in weights.items():
            if not key.startswith(prefix) or not key.endswith(".weight"):
                continue
            if getattr(value, "ndim", 0) != 2:
                continue
            suffix = key.removeprefix(prefix)
            layer_idx = suffix.split(".", 1)[0]
            if layer_idx.isdigit():
                indices.append(int(layer_idx))
        return sorted(indices)

    def try_map_linear_stack(
        *,
        source_prefix: str,
        source_has_separate_head: bool,
        target_prefix: str,
        target_has_separate_head: bool,
    ) -> bool:
        target = actor.state_dict()
        source_linear = linear_indices(source_prefix, state)
        target_linear = linear_indices(target_prefix, target)
        if not source_linear or not target_linear:
            return False

        source_head_weight = "mean.weight" if source_has_separate_head else f"{source_prefix}{source_linear[-1]}.weight"
        source_head_bias = "mean.bias" if source_has_separate_head else f"{source_prefix}{source_linear[-1]}.bias"
        target_head_weight = "mean.weight" if target_has_separate_head else f"{target_prefix}{target_linear[-1]}.weight"
        target_head_bias = "mean.bias" if target_has_separate_head else f"{target_prefix}{target_linear[-1]}.bias"
        if source_head_weight not in state or target_head_weight not in target:
            return False
        if state[source_head_weight].shape != target[target_head_weight].shape:
            return False

        source_hidden = source_linear if source_has_separate_head else source_linear[:-1]
        target_hidden = target_linear if target_has_separate_head else target_linear[:-1]
        if len(source_hidden) != len(target_hidden):
            return False

        mapped = {}
        for src_idx, dst_idx in zip(source_hidden, target_hidden):
            for param_name in ("weight", "bias"):
                src_key = f"{source_prefix}{src_idx}.{param_name}"
                dst_key = f"{target_prefix}{dst_idx}.{param_name}"
                if src_key not in state or dst_key not in target or state[src_key].shape != target[dst_key].shape:
                    return False
                mapped[dst_key] = state[src_key]
        mapped[target_head_weight] = state[source_head_weight]
        if source_head_bias in state and target_head_bias in target:
            if state[source_head_bias].shape != target[target_head_bias].shape:
                return False
            mapped[target_head_bias] = state[source_head_bias]
        for src_key, dst_key in (("logstd", "logstd"), ("log_std", "log_std"), ("logstd", "log_std"), ("log_std", "logstd")):
            if src_key in state and dst_key in target and state[src_key].shape == target[dst_key].shape:
                mapped[dst_key] = state[src_key]
                break
        missing, unexpected = actor.load_state_dict(mapped, strict=False)
        missing = [key for key in missing if key not in {"logstd", "log_std"}]
        unexpected = [key for key in unexpected if key not in target]
        return not missing and not unexpected

    source_prefix = None
    for prefix in ("actor.", "mu_net."):
        if any(key.startswith(prefix) for key in state):
            source_prefix = prefix
            break
    if source_prefix is not None:
        target = actor.state_dict()
        if any(key.startswith("mu_net.") for key in target):
            target_prefix = "mu_net."
        elif any(key.startswith("actor.") for key in target):
            target_prefix = "actor."
        else:
            target_prefix = None
        if target_prefix is not None:
            mapped = {}
            for key, value in state.items():
                if key in {"logstd", "log_std"}:
                    if "logstd" in target:
                        mapped["logstd"] = value
                    elif "log_std" in target:
                        mapped["log_std"] = value
                    continue
                if key.startswith(source_prefix):
                    mapped[f"{target_prefix}{key.removeprefix(source_prefix)}"] = value
            missing, unexpected = actor.load_state_dict(mapped, strict=False)
            unexpected = [key for key in unexpected if key not in target]
            missing = [key for key in missing if key not in {"logstd", "log_std"}]
            if not unexpected and not missing:
                return

        if any(key.startswith("backbone.") for key in actor.state_dict()):
            if try_map_linear_stack(
                source_prefix=source_prefix,
                source_has_separate_head=False,
                target_prefix="backbone.",
                target_has_separate_head=True,
            ):
                return

    if "backbone.0.weight" in state and any(key.startswith("backbone.") for key in actor.state_dict()):
        target = actor.state_dict()
        filtered = {key: value for key, value in state.items() if key in target and target[key].shape == value.shape}
        missing, unexpected = actor.load_state_dict(filtered, strict=False)
        unexpected = [key for key in unexpected if key not in target]
        missing = [key for key in missing if key != "log_std"]
        if not unexpected and not missing:
            return
        if try_map_linear_stack(
            source_prefix="backbone.",
            source_has_separate_head=True,
            target_prefix="backbone.",
            target_has_separate_head=True,
        ):
            return

    if "backbone.0.weight" not in state:
        actor.load_state_dict(state)
        return

    target = actor.state_dict()
    prefix = "mu_net" if any(key.startswith("mu_net.") for key in target) else "actor"
    mapped = {
        f"{prefix}.0.weight": state["backbone.0.weight"],
        f"{prefix}.0.bias": state["backbone.0.bias"],
        f"{prefix}.2.weight": state["backbone.2.weight"],
        f"{prefix}.2.bias": state["backbone.2.bias"],
        f"{prefix}.3.weight": state["backbone.3.weight"],
        f"{prefix}.3.bias": state["backbone.3.bias"],
        f"{prefix}.5.weight": state["backbone.5.weight"],
        f"{prefix}.5.bias": state["backbone.5.bias"],
        f"{prefix}.6.weight": state["backbone.6.weight"],
        f"{prefix}.6.bias": state["backbone.6.bias"],
        f"{prefix}.8.weight": state["backbone.8.weight"],
        f"{prefix}.8.bias": state["backbone.8.bias"],
        f"{prefix}.9.weight": state["mean.weight"],
        f"{prefix}.9.bias": state["mean.bias"],
    }
    if "logstd" in target and "log_std" in state:
        mapped["logstd"] = state["log_std"]
    missing, unexpected = actor.load_state_dict(mapped, strict=False)
    unexpected = [key for key in unexpected if key not in target]
    if unexpected or any(not key.startswith("logstd") for key in missing):
        raise RuntimeError(f"could not map PPO actor checkpoint {path}: missing={missing}, unexpected={unexpected}")


def make_critic(env: NewtonMuJoCoTorchEnv, hidden_dims: list[int] | None = None) -> torch.nn.Module:
    if hidden_dims is None:
        hidden_dims = [64, 64]
    cfg = {
        "critic_mlp": {"units": hidden_dims, "activation": "elu"},
    }
    return CriticMLP(env.num_obs, cfg, device=str(env.torch_device))


def obs_rms_snapshot(obs_rms: RunningMeanStd | None) -> tuple[torch.Tensor, torch.Tensor] | None:
    if obs_rms is None:
        return None
    return obs_rms.mean.clone(), obs_rms.var.clone()


def clone_module_state(module: torch.nn.Module | None) -> dict[str, torch.Tensor] | None:
    if module is None:
        return None
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def clone_optimizer_state(optimizer: torch.optim.Optimizer | None) -> dict[str, Any] | None:
    if optimizer is None:
        return None
    return copy.deepcopy(optimizer.state_dict())


def clone_obs_rms_state(obs_rms: RunningMeanStd | None) -> dict[str, Any] | None:
    if obs_rms is None:
        return None
    return {
        "mean": obs_rms.mean.detach().clone(),
        "var": obs_rms.var.detach().clone(),
        "count": obs_rms.count,
    }


def restore_obs_rms_state(obs_rms: RunningMeanStd | None, state: dict[str, Any] | None) -> None:
    if obs_rms is None or state is None:
        return
    obs_rms.mean = state["mean"].detach().clone()
    obs_rms.var = state["var"].detach().clone()
    obs_rms.count = state["count"]


def normalize_obs(obs: torch.Tensor, stats: tuple[torch.Tensor, torch.Tensor] | RunningMeanStd | None) -> torch.Tensor:
    if stats is None:
        return obs
    if isinstance(stats, RunningMeanStd):
        return stats.normalize(obs)
    mean, var = stats
    return (obs - mean) / torch.sqrt(var + 1.0e-5)


def deterministic_policy_action(actor: torch.nn.Module, obs: torch.Tensor) -> torch.Tensor:
    action = actor(obs, deterministic=True)
    if getattr(actor, "outputs_raw_action", False):
        return squash_policy_action(actor, action)
    if getattr(actor, "action_squash", None) in {"tanh", "none"}:
        return action
    return torch.tanh(action)


def squash_policy_action(actor: torch.nn.Module, raw_action: torch.Tensor) -> torch.Tensor:
    if getattr(actor, "action_squash", "tanh") == "none":
        return torch.clamp(raw_action, -1.0, 1.0)
    return torch.tanh(raw_action)


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


def shac_rollout_loss(
    env: NewtonMuJoCoTorchEnv,
    actor: torch.nn.Module,
    *,
    horizon: int,
    gamma: float,
    rew_scale: float,
    termination_penalty: float,
    obs_stats: tuple[torch.Tensor, torch.Tensor] | None,
    q0: torch.Tensor,
    qd0: torch.Tensor,
    prev_action0: torch.Tensor,
    stochastic_actor: bool,
    loss_objective: str = "reward",
    displacement_objective_weight: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    if loss_objective not in {"reward", "displacement"}:
        raise ValueError(f"unknown SHAC rollout loss objective: {loss_objective}")
    env.reset_solver_data()
    q = q0.clone()
    qd = qd0.clone()
    prev_action = prev_action0.clone()
    gamma_vec = torch.ones(env.num_envs, dtype=torch.float32, device=env.torch_device)
    loss = torch.zeros((), dtype=torch.float32, device=env.torch_device)
    root_x_start = q[:, 0].detach().clone()
    rewards = []
    invalid_count = 0
    fall_count = 0
    progress = torch.zeros(env.num_envs, dtype=torch.long, device=env.torch_device)
    for _ in range(horizon):
        obs = normalize_obs(env.observe(q, qd, prev_action, phase=progress), obs_stats)
        raw_action = actor(obs, deterministic=not stochastic_actor)
        action = squash_policy_action(actor, raw_action)
        q_next, qd_next = env.step(q, qd, env.action_to_joint_f(action))
        invalid = env.invalid_state(q_next, qd_next)
        fell = torch.logical_and(env.fallen_state(q_next), ~invalid)
        next_obs = env.observe(q_next, qd_next, action, phase=progress + 1)
        rew = env.transition_reward(q, qd, q_next, qd_next, action, obs=next_obs)
        rew = finalize_terminal_reward(rew, invalid=invalid, fell=fell, termination_penalty=termination_penalty)
        rewards.append(rew.detach().mean())
        if loss_objective == "reward":
            loss = loss - (gamma_vec * rew * rew_scale).sum()
        done = torch.logical_or(invalid, fell)
        invalid_count += int(invalid.detach().sum().cpu())
        fall_count += int(fell.detach().sum().cpu())
        gamma_vec = gamma_vec * gamma * (~done).to(torch.float32)
        q, qd = q_next, qd_next
        prev_action = action
        progress = torch.where(done, torch.zeros_like(progress), progress + 1)
    if loss_objective == "displacement":
        loss = -float(displacement_objective_weight) * (q[:, 0] - root_x_start).sum()
    denom = max(1, horizon * env.num_envs)
    return loss / denom, {
        "loss_objective": loss_objective,
        "mean_reward": float(torch.stack(rewards).mean().detach().cpu()) if rewards else None,
        "mean_displacement": float((q[:, 0] - root_x_start).mean().detach().cpu()),
        "invalid_count": invalid_count,
        "fall_count": fall_count,
    }


@torch.no_grad()
def warmup_policy_state(
    env: NewtonMuJoCoTorchEnv,
    actor: torch.nn.Module,
    q: torch.Tensor,
    qd: torch.Tensor,
    prev_action: torch.Tensor,
    progress: torch.Tensor,
    *,
    steps: int,
    obs_rms: RunningMeanStd | None,
    stochastic_init: bool,
    stop_height_min: float | None = None,
    stop_up_min: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    invalid_count = 0
    fall_count = 0
    stopped = torch.zeros(q.shape[0], dtype=torch.bool, device=q.device)
    with torch.no_grad():
        for _ in range(max(0, steps)):
            active = ~stopped
            if not bool(active.any().cpu()):
                break
            obs = normalize_obs(env.observe(q, qd, prev_action, phase=progress), obs_rms)
            action = deterministic_policy_action(actor, obs)
            action = torch.where(active.unsqueeze(-1), action, torch.zeros_like(action))
            q_next, qd_next = env.step(q, qd, env.action_to_joint_f(action))
            q_next = torch.where(active.unsqueeze(-1), q_next, q)
            qd_next = torch.where(active.unsqueeze(-1), qd_next, qd)
            invalid = torch.logical_and(active, env.invalid_state(q_next, qd_next))
            fell = torch.logical_and(active, torch.logical_and(env.fallen_state(q_next), ~invalid))
            done = torch.logical_or(invalid, fell)
            invalid_count += int(invalid.sum().cpu())
            fall_count += int(fell.sum().cpu())
            if done.any():
                done_ids = done.nonzero(as_tuple=False).squeeze(-1)
                q_next, qd_next = env.reset_done(q_next, qd_next, done_ids, stochastic_init=stochastic_init)
                action = torch.where(done.unsqueeze(-1), torch.zeros_like(action), action)
                progress = torch.where(done, torch.zeros_like(progress), progress)
            stop = torch.zeros_like(stopped)
            if env.env_name == "ant" and (stop_height_min is not None or stop_up_min is not None):
                torso_pos, _, _, _, up_vec, _ = env.ant_pose_terms(q_next, qd_next)
                if stop_height_min is not None:
                    stop = torch.logical_or(stop, torso_pos[:, 1] < float(stop_height_min))
                if stop_up_min is not None:
                    stop = torch.logical_or(stop, up_vec[:, 1] < float(stop_up_min))
                stop = torch.logical_and(active, torch.logical_and(stop, ~done))
                stopped = torch.logical_or(stopped, stop)
            q, qd, prev_action = q_next, qd_next, action
            progress = progress + torch.logical_and(active, ~done).to(dtype=progress.dtype)
    return q.detach(), qd.detach(), prev_action.detach(), progress.detach(), {
        "warmup_steps": int(max(0, steps)),
        "warmup_invalid_resets": invalid_count,
        "warmup_fall_resets": fall_count,
        "warmup_stop_count": int(stopped.sum().cpu()),
    }


def differentiable_survival_margin(
    env: NewtonMuJoCoTorchEnv,
    q: torch.Tensor,
    qd: torch.Tensor,
    *,
    height_min: float | None,
    height_weight: float,
    up_min: float | None,
    up_weight: float,
    heading_min: float | None,
    heading_weight: float,
    angle_max: float | None,
    angle_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    penalty = torch.zeros(q.shape[0], dtype=torch.float32, device=q.device)
    metrics: dict[str, torch.Tensor] = {}

    if height_min is not None and height_weight > 0.0:
        height = q[:, 1]
        height_shortfall = torch.relu(float(height_min) - height)
        penalty = penalty + float(height_weight) * height_shortfall.square()
        metrics["height_shortfall"] = height_shortfall.detach()

    if env.env_name == "ant":
        _, _, _, _, up_vec, heading_alignment = env.ant_pose_terms(q, qd)
        if up_min is not None and up_weight > 0.0:
            up_shortfall = torch.relu(float(up_min) - up_vec[:, 1])
            penalty = penalty + float(up_weight) * up_shortfall.square()
            metrics["up_shortfall"] = up_shortfall.detach()
        if heading_min is not None and heading_weight > 0.0:
            heading_shortfall = torch.relu(float(heading_min) - heading_alignment.squeeze(-1))
            penalty = penalty + float(heading_weight) * heading_shortfall.square()
            metrics["heading_shortfall"] = heading_shortfall.detach()
    elif env.env_name == "hopper" and angle_max is not None and angle_weight > 0.0:
        angle_excess = torch.relu(q[:, 2].abs() - float(angle_max))
        penalty = penalty + float(angle_weight) * angle_excess.square()
        metrics["angle_excess"] = angle_excess.detach()

    metrics["penalty"] = penalty.detach()
    return penalty, metrics


def one_step_action_loss(
    env: NewtonMuJoCoTorchEnv,
    *,
    q0: torch.Tensor,
    qd0: torch.Tensor,
    action: torch.Tensor,
    termination_penalty: float,
) -> tuple[torch.Tensor, dict]:
    q, qd = env.step(q0, qd0, env.action_to_joint_f(action))
    invalid = env.invalid_state(q, qd)
    fell = torch.logical_and(env.fallen_state(q), ~invalid)
    obs = env.observe(q, qd, action)
    rew = env.transition_reward(q0, qd0, q, qd, action, obs=obs)
    rew = finalize_terminal_reward(rew, invalid=invalid, fell=fell, termination_penalty=termination_penalty)
    return -rew.mean(), {
        "mean_reward": float(rew.detach().mean().cpu()),
        "invalid_count": int(invalid.detach().sum().cpu()),
        "fall_count": int(fell.detach().sum().cpu()),
    }


def central_difference_rows(
    *,
    base_values: torch.Tensor,
    analytic_grad: torch.Tensor,
    directions: torch.Tensor,
    epsilons: list[float],
    evaluate,
    assign,
) -> list[dict]:
    rows = []
    analytic_directional = torch.mv(directions, analytic_grad).detach().cpu().numpy()
    for eps in epsilons:
        fd_values = []
        losses_plus = []
        losses_minus = []
        for direction in directions:
            assign(base_values + eps * direction)
            plus = evaluate()
            assign(base_values - eps * direction)
            minus = evaluate()
            fd_values.append((plus - minus) / (2.0 * eps))
            losses_plus.append(plus)
            losses_minus.append(minus)
        assign(base_values)

        fd = np.asarray(fd_values, dtype=np.float64)
        ad = analytic_directional.astype(np.float64)
        abs_error = np.abs(ad - fd)
        denom = np.maximum(np.maximum(np.abs(fd), np.abs(ad)), 1.0e-12)
        rel_error = abs_error / denom
        rows.append(
            {
                "epsilon": eps,
                "mean_abs_error": float(abs_error.mean()),
                "median_abs_error": float(np.median(abs_error)),
                "max_abs_error": float(abs_error.max()),
                "mean_relative_error": float(rel_error.mean()),
                "median_relative_error": float(np.median(rel_error)),
                "max_relative_error": float(rel_error.max()),
                "sign_agreement": float(np.mean(np.sign(ad) == np.sign(fd))),
                "mean_fd_abs": float(np.mean(np.abs(fd))),
                "mean_ad_abs": float(np.mean(np.abs(ad))),
                "per_direction": [
                    {
                        "direction": int(i),
                        "analytic": float(ad[i]),
                        "finite_difference": float(fd[i]),
                        "abs_error": float(abs_error[i]),
                        "relative_error": float(rel_error[i]),
                        "loss_plus": float(losses_plus[i]),
                        "loss_minus": float(losses_minus[i]),
                    }
                    for i in range(len(fd_values))
                ],
            }
        )
    return rows


def masked_random_directions(
    *,
    count: int,
    width: int,
    mask: torch.Tensor,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    directions = torch.randn((count, width), generator=generator, dtype=dtype, device=device)
    directions = directions * mask.view(1, -1).to(dtype=dtype, device=device)
    return directions / directions.norm(dim=1, keepdim=True).clamp(min=1.0e-12)


def run_gradient_check(args: argparse.Namespace) -> dict:
    if args.contact_backend is None:
        args.contact_backend = "newton" if args.env == "ant" else ("mujoco" if is_planar_locomotion_env(args.env) or is_contact_target_env(args.env) else "none")
    resolve_ant_defaults(args)
    if args.sim_substeps is None:
        args.sim_substeps = 2 if args.env == "ant" else (16 if is_locomotion_env(args.env) else 1)
    if args.horizon is None:
        args.horizon = args.grad_check_horizon
    if args.eval_horizon is None:
        args.eval_horizon = args.horizon
    if args.episode_length is None:
        args.episode_length = 1000 if is_locomotion_env(args.env) else 240
    if args.force_scale is None:
        args.force_scale = 7.5 if args.env == "ant" else (200.0 if is_planar_locomotion_env(args.env) else (35.0 if is_contact_target_env(args.env) else (20.0 if args.env == "acrobot" else 1000.0)))
    if args.reset_noise is None:
        args.reset_noise = 0.0
    if args.termination_penalty is None:
        args.termination_penalty = ANT_DEFAULT_TERMINATION_PENALTY if args.env in {"ant", "hopper"} else 0.0
    if args.stochastic_actor is None:
        args.stochastic_actor = is_locomotion_env(args.env)
    if args.obs_rms is None:
        args.obs_rms = args.obs_rms_path is not None

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
        sim_substeps=args.sim_substeps,
        mujoco_integrator=args.mujoco_integrator,
        mujoco_smooth_adjoint=args.mujoco_smooth_adjoint,
        mujoco_smooth_friction_viscosity=args.mujoco_smooth_friction_viscosity,
        mujoco_smooth_friction_scale=args.mujoco_smooth_friction_scale,
        mujoco_smooth_friction_bypass_kf=args.mujoco_smooth_friction_bypass_kf,
        mujoco_smooth_penalty_damping_alpha=args.mujoco_smooth_penalty_damping_alpha,
        mujoco_smooth_friction_surrogate_alpha=args.mujoco_smooth_friction_surrogate_alpha,
        acrobot_actuation=args.acrobot_actuation,
        ant_asset=args.ant_asset,
        ant_disable_joint_limits=args.ant_disable_joint_limits,
        ant_density_override=args.ant_density_override,
        ant_contact_margin=args.ant_contact_margin,
        ant_contact_gap=args.ant_contact_gap,
        ant_contact_mu=args.ant_contact_mu,
        ant_joint_damping=args.ant_joint_damping,
        ant_armature=args.ant_armature,
        ant_min_up=args.ant_min_up,
        ant_start_height=args.ant_start_height,
        ant_start_joint_q=args.ant_start_joint_q,
        ant_reset_position_scale=args.ant_reset_position_scale,
        ant_reset_angle_scale=args.ant_reset_angle_scale,
        ant_reset_joint_scale=args.ant_reset_joint_scale,
        ant_reset_velocity_scale=args.ant_reset_velocity_scale,
        ant_termination_height=args.ant_termination_height,
        ant_max_healthy_height=args.ant_max_healthy_height,
        ant_observation_style=args.ant_observation_style,
        ant_reward_style=args.ant_reward_style,
        ant_dof_limit_mode=args.ant_dof_limit_mode,
        ant_action_order=args.ant_action_order,
        ant_smooth_up_reward=args.ant_smooth_up_reward,
        ant_reward_min_up=args.ant_reward_min_up,
        ant_reward_min_height=args.ant_reward_min_height,
        hopper_reward_style=args.hopper_reward_style,
        hopper_start_joint_q=args.hopper_start_joint_q,
        hopper_contact_mu=args.hopper_contact_mu,
        hopper_joint_damping=args.hopper_joint_damping,
        hopper_armature=args.hopper_armature,
        hopper_termination_height=args.hopper_termination_height,
        hopper_termination_angle=args.hopper_termination_angle,
        hopper_termination_height_tolerance=args.hopper_termination_height_tolerance,
        hopper_reset_position_scale=args.hopper_reset_position_scale,
        hopper_reset_angle_scale=args.hopper_reset_angle_scale,
        hopper_reset_joint_scale=args.hopper_reset_joint_scale,
        hopper_reset_velocity_scale=args.hopper_reset_velocity_scale,
        phase_observation=args.phase_observation,
        phase_period=args.phase_period,
        hopper_terminate_angle=args.hopper_terminate_angle,
        locomotion_disable_joint_limits=args.locomotion_disable_joint_limits,
        ant_reward=AntRewardWeights(
            progress=args.ant_progress_weight,
            heading=args.ant_heading_weight,
            up=args.ant_up_weight,
            height=args.ant_height_weight,
            action=args.ant_action_penalty,
            alive=args.ant_alive_reward,
            actions_cost=args.ant_actions_cost,
            energy_cost=args.ant_energy_cost,
            dof_limit_cost=args.ant_dof_limit_cost,
            dof_vel_scale=args.ant_dof_vel_scale,
            up_margin=args.ant_up_margin_penalty,
            height_margin=args.ant_height_margin_penalty,
        ),
        hopper_reward=HopperRewardWeights(
            progress=args.hopper_progress_weight,
            height=args.hopper_height_weight,
            angle=args.hopper_angle_weight,
            action=args.hopper_action_penalty,
            alive=args.hopper_alive_reward,
        ),
        cheetah_reward=CheetahRewardWeights(action=args.cheetah_action_penalty),
        acrobot_reward=AcrobotRewardWeights(
            target=args.acrobot_target_weight,
            velocity=args.acrobot_velocity_weight,
            action=args.acrobot_action_weight,
        ),
        contact_reward=ContactTargetRewardWeights(
            target=args.contact_target_weight,
            velocity=args.contact_velocity_weight,
            height=args.contact_height_weight,
            action=args.contact_action_weight,
        ),
    )
    actor = make_actor(
        env,
        stochastic=args.stochastic_actor,
        hidden_dims=args.actor_hidden_dims,
        actor_logstd_init=args.actor_logstd_init,
        actor_layer_norm=args.actor_layer_norm,
        action_squash=args.action_squash,
    )
    if args.actor_path is not None:
        load_actor_checkpoint(actor, args.actor_path, env.torch_device)
    if args.train_final_layer_only:
        freeze_actor_backbone(actor)
    obs_stats = load_obs_rms(args.obs_rms_path, env.torch_device) if args.obs_rms_path is not None else None

    q0, qd0 = env.reset(noise=0.0, stochastic_init=False)
    prev0 = torch.zeros((env.num_envs, env.num_actions), dtype=torch.float32, device=env.torch_device)
    params = trainable_parameters(actor)
    base_params = flatten_parameters(params)
    epsilons = args.grad_check_eps or list(DEFAULT_GRAD_CHECK_EPS)

    def eval_policy_loss() -> float:
        torch.manual_seed(args.seed + 1000)
        with torch.no_grad():
            loss_value, _ = shac_rollout_loss(
                env,
                actor,
                horizon=args.horizon,
                gamma=args.gamma,
                rew_scale=args.rew_scale,
                termination_penalty=args.termination_penalty,
                obs_stats=obs_stats,
                q0=q0,
                qd0=qd0,
                prev_action0=prev0,
                stochastic_actor=args.stochastic_actor,
            )
        return float(loss_value.detach().cpu())

    actor.zero_grad(set_to_none=True)
    torch.manual_seed(args.seed + 1000)
    policy_loss, policy_metrics = shac_rollout_loss(
        env,
        actor,
        horizon=args.horizon,
        gamma=args.gamma,
        rew_scale=args.rew_scale,
        termination_penalty=args.termination_penalty,
        obs_stats=obs_stats,
        q0=q0,
        qd0=qd0,
        prev_action0=prev0,
        stochastic_actor=args.stochastic_actor,
    )
    policy_loss.backward()
    policy_grad = flatten_gradients(params)
    policy_grad_norm = float(policy_grad.to(torch.float64).norm().detach().cpu())

    generator = torch.Generator(device=env.torch_device)
    generator.manual_seed(args.seed + 2000)
    policy_directions = torch.randn(
        (args.grad_check_directions, base_params.numel()),
        generator=generator,
        dtype=base_params.dtype,
        device=base_params.device,
    )
    policy_directions = policy_directions / policy_directions.norm(dim=1, keepdim=True).clamp(min=1.0e-12)
    policy_rows = central_difference_rows(
        base_values=base_params,
        analytic_grad=policy_grad,
        directions=policy_directions,
        epsilons=epsilons,
        evaluate=eval_policy_loss,
        assign=lambda values: assign_flat_parameters(params, values),
    )

    with torch.no_grad():
        obs0 = normalize_obs(env.observe(q0, qd0, prev0), obs_stats)
        action0 = deterministic_policy_action(actor, obs0).detach()
    action0.requires_grad_(True)
    action_loss, action_metrics = one_step_action_loss(
        env,
        q0=q0,
        qd0=qd0,
        action=action0,
        termination_penalty=args.termination_penalty,
    )
    action_loss.backward()
    action_grad = torch.nan_to_num(action0.grad.detach().reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    action_base = action0.detach().reshape(-1)
    action_generator = torch.Generator(device=env.torch_device)
    action_generator.manual_seed(args.seed + 3000)
    action_directions = torch.randn(
        (args.grad_check_directions, action_base.numel()),
        generator=action_generator,
        dtype=action_base.dtype,
        device=action_base.device,
    )
    action_directions = action_directions / action_directions.norm(dim=1, keepdim=True).clamp(min=1.0e-12)
    mutable_action = action_base.clone()

    def assign_action(values: torch.Tensor) -> None:
        mutable_action.copy_(values)

    def eval_action_loss() -> float:
        with torch.no_grad():
            loss_value, _ = one_step_action_loss(
                env,
                q0=q0,
                qd0=qd0,
                action=mutable_action.view_as(action0),
                termination_penalty=args.termination_penalty,
            )
        return float(loss_value.detach().cpu())

    action_rows = central_difference_rows(
        base_values=action_base,
        analytic_grad=action_grad,
        directions=action_directions,
        epsilons=epsilons,
        evaluate=eval_action_loss,
        assign=assign_action,
    )

    state_q = q0.detach().clone().requires_grad_(True)
    state_qd = qd0.detach().clone().requires_grad_(True)
    state_action = action0.detach().clone().requires_grad_(True)
    state_action_loss, state_action_metrics = one_step_action_loss(
        env,
        q0=state_q,
        qd0=state_qd,
        action=state_action,
        termination_penalty=args.termination_penalty,
    )
    state_action_loss.backward()
    state_grad = torch.cat(
        [
            torch.nan_to_num(state_q.grad.detach(), nan=0.0, posinf=0.0, neginf=0.0).reshape(-1),
            torch.nan_to_num(state_qd.grad.detach(), nan=0.0, posinf=0.0, neginf=0.0).reshape(-1),
            torch.nan_to_num(state_action.grad.detach(), nan=0.0, posinf=0.0, neginf=0.0).reshape(-1),
        ]
    )
    state_base = torch.cat([q0.detach().reshape(-1), qd0.detach().reshape(-1), action0.detach().reshape(-1)])
    state_generator = torch.Generator(device=env.torch_device)
    state_generator.manual_seed(args.seed + 4000)
    state_directions = torch.randn(
        (args.grad_check_directions, state_base.numel()),
        generator=state_generator,
        dtype=state_base.dtype,
        device=state_base.device,
    )
    state_directions = state_directions / state_directions.norm(dim=1, keepdim=True).clamp(min=1.0e-12)
    mutable_state = state_base.clone()
    q_count = q0.numel()
    qd_count = qd0.numel()

    def assign_state(values: torch.Tensor) -> None:
        mutable_state.copy_(values)

    def eval_state_action_loss() -> float:
        with torch.no_grad():
            q_eval = mutable_state[:q_count].view_as(q0)
            qd_eval = mutable_state[q_count : q_count + qd_count].view_as(qd0)
            action_eval = mutable_state[q_count + qd_count :].view_as(action0)
            loss_value, _ = one_step_action_loss(
                env,
                q0=q_eval,
                qd0=qd_eval,
                action=action_eval,
                termination_penalty=args.termination_penalty,
            )
        return float(loss_value.detach().cpu())

    state_action_rows = central_difference_rows(
        base_values=state_base,
        analytic_grad=state_grad,
        directions=state_directions,
        epsilons=epsilons,
        evaluate=eval_state_action_loss,
        assign=assign_state,
    )
    state_width = state_base.numel()
    if is_planar_locomotion_env(env.env_name):
        component_names = ["root_planar_q", "joint_q", "root_planar_qd", "joint_qd", "action"]
    elif env.q_dim == 7 and env.qd_dim == 6:
        component_names = ["root_pos_q", "root_quat_q", "root_qd", "action"]
    elif env.q_dim >= 7 and env.qd_dim >= 6:
        component_names = ["root_pos_q", "root_quat_q", "joint_q", "root_qd", "joint_qd", "action"]
    else:
        component_names = ["joint_q", "joint_qd", "action"]
    component_masks: dict[str, torch.Tensor] = {
        component: torch.zeros(state_width, dtype=torch.float32, device=env.torch_device)
        for component in component_names
    }
    for env_id in range(env.num_envs):
        q_offset = env_id * env.q_dim
        qd_offset = q_count + env_id * env.qd_dim
        action_offset = q_count + qd_count + env_id * env.num_actions
        if "root_pos_q" in component_masks:
            component_masks["root_pos_q"][q_offset : q_offset + 3] = 1.0
            component_masks["root_quat_q"][q_offset + 3 : q_offset + 7] = 1.0
            if "joint_q" in component_masks:
                component_masks["joint_q"][q_offset + 7 : q_offset + env.q_dim] = 1.0
            component_masks["root_qd"][qd_offset : qd_offset + 6] = 1.0
            if "joint_qd" in component_masks:
                component_masks["joint_qd"][qd_offset + 6 : qd_offset + env.qd_dim] = 1.0
        elif "root_planar_q" in component_masks:
            component_masks["root_planar_q"][q_offset : q_offset + 3] = 1.0
            component_masks["joint_q"][q_offset + 3 : q_offset + env.q_dim] = 1.0
            component_masks["root_planar_qd"][qd_offset : qd_offset + 3] = 1.0
            component_masks["joint_qd"][qd_offset + 3 : qd_offset + env.qd_dim] = 1.0
        else:
            component_masks["joint_q"][q_offset : q_offset + env.q_dim] = 1.0
            component_masks["joint_qd"][qd_offset : qd_offset + env.qd_dim] = 1.0
        component_masks["action"][action_offset : action_offset + env.num_actions] = 1.0

    component_rows = {}
    for idx, (component, mask) in enumerate(component_masks.items()):
        directions = masked_random_directions(
            count=args.grad_check_directions,
            width=state_width,
            mask=mask,
            seed=args.seed + 5000 + idx,
            device=env.torch_device,
            dtype=state_base.dtype,
        )
        component_rows[component] = central_difference_rows(
            base_values=state_base,
            analytic_grad=state_grad,
            directions=directions,
            epsilons=epsilons,
            evaluate=eval_state_action_loss,
            assign=assign_state,
        )

    result = {
        "env": args.env,
        "mode": "gradcheck",
        "title": "SHAC with MuJoCo Warp",
        "timestamp_pacific": pacific_now_iso(),
        "contact_backend": args.contact_backend,
        "newton_commit": git_commit_for_imported_module(newton),
        "newton_path": str(Path(newton.__path__[0]).resolve()) if hasattr(newton, "__path__") else None,
        "mujoco_warp_commit": git_commit_for_imported_module(mujoco_warp),
        "num_envs": args.num_envs,
        "horizon": args.horizon,
        "dt": args.dt,
        "sim_substeps": env.sim_substeps,
        "mujoco_integrator": env.mujoco_integrator,
        "mujoco_smooth_adjoint": args.mujoco_smooth_adjoint,
        "mujoco_smooth_friction_viscosity": args.mujoco_smooth_friction_viscosity,
        "mujoco_smooth_friction_scale": args.mujoco_smooth_friction_scale,
        "mujoco_smooth_friction_bypass_kf": args.mujoco_smooth_friction_bypass_kf,
        "mujoco_smooth_penalty_damping_alpha": args.mujoco_smooth_penalty_damping_alpha,
        "mujoco_smooth_friction_surrogate_alpha": args.mujoco_smooth_friction_surrogate_alpha,
        "nconmax": env.nconmax,
        "njmax": env.njmax,
        "world_spacing": list(env.world_spacing) if env.world_spacing is not None else None,
        "disable_eulerdamp": True,
        "force_scale": args.force_scale,
        "stochastic_actor": args.stochastic_actor,
        "actor_hidden_dims": args.actor_hidden_dims,
        "actor_logstd_init": args.actor_logstd_init,
        "actor_layer_norm": args.actor_layer_norm,
        "train_final_layer_only": args.train_final_layer_only,
        "action_squash": args.action_squash,
        "acrobot_actuation": env.acrobot_actuation if args.env == "acrobot" else None,
        "ant_asset": env.ant_asset if args.env == "ant" else None,
        "ant_disable_joint_limits": env.ant_disable_joint_limits if args.env == "ant" else None,
        "ant_density_override": env.ant_density_override if args.env == "ant" else None,
        "ant_start_height": env.ant_start_height if args.env == "ant" else None,
        "ant_start_joint_q": env.ant_start_joint_q if args.env == "ant" else None,
        "ant_termination_height": env.ant_termination_height if args.env == "ant" else None,
        "ant_observation_style": env.ant_observation_style if args.env == "ant" else None,
        "ant_reward_style": env.ant_reward_style if args.env == "ant" else None,
        "ant_dof_limit_mode": env.ant_dof_limit_mode if args.env == "ant" else None,
        "ant_action_order": env.ant_action_order if args.env == "ant" else None,
        "ant_reset_position_scale": env.ant_reset_position_scale if args.env == "ant" else None,
        "ant_reset_angle_scale": env.ant_reset_angle_scale if args.env == "ant" else None,
        "ant_reset_joint_scale": env.ant_reset_joint_scale if args.env == "ant" else None,
        "ant_reset_velocity_scale": env.ant_reset_velocity_scale if args.env == "ant" else None,
        "hopper_terminate_angle": env.hopper_terminate_angle if args.env == "hopper" else None,
        "hopper_termination_angle": env.hopper_termination_angle if args.env == "hopper" else None,
        "hopper_termination_height": env.hopper_termination_height if args.env == "hopper" else None,
        "hopper_termination_height_tolerance": env.hopper_termination_height_tolerance if args.env == "hopper" else None,
        "hopper_reward_style": env.hopper_reward_style if args.env == "hopper" else None,
        "hopper_start_joint_q": env.hopper_start_joint_q if args.env == "hopper" else None,
        "hopper_contact_mu": env.hopper_contact_mu if args.env == "hopper" else None,
        "hopper_joint_damping": env.hopper_joint_damping if args.env == "hopper" else None,
        "hopper_armature": env.hopper_armature if args.env == "hopper" else None,
        "hopper_reset_position_scale": env.hopper_reset_position_scale if args.env == "hopper" else None,
        "hopper_reset_angle_scale": env.hopper_reset_angle_scale if args.env == "hopper" else None,
        "hopper_reset_joint_scale": env.hopper_reset_joint_scale if args.env == "hopper" else None,
        "hopper_reset_velocity_scale": env.hopper_reset_velocity_scale if args.env == "hopper" else None,
        "contact_reward": env.contact_reward.__dict__ if is_contact_target_env(args.env) else None,
        "hopper_reward": env.hopper_reward.__dict__ if args.env == "hopper" else None,
        "cheetah_reward": env.cheetah_reward.__dict__ if args.env == "cheetah" else None,
        "actor_path": str(args.actor_path) if args.actor_path is not None else None,
        "obs_rms_path": str(args.obs_rms_path) if args.obs_rms_path is not None else None,
        "epsilon_values": epsilons,
        "directions": args.grad_check_directions,
        "ant_contact_margin": env.ant_contact_margin if args.env == "ant" else None,
        "ant_contact_gap": env.ant_contact_gap if args.env == "ant" else None,
        "ant_contact_mu": env.ant_contact_mu if args.env == "ant" else None,
        "ant_joint_damping": env.ant_joint_damping if args.env == "ant" else None,
        "ant_armature": env.ant_armature if args.env == "ant" else None,
        "ant_min_up": env.ant_min_up if args.env == "ant" else None,
        "locomotion_disable_joint_limits": env.locomotion_disable_joint_limits if is_planar_locomotion_env(args.env) else None,
        "policy": {
            "loss": float(policy_loss.detach().cpu()),
            "analytic_grad_norm": finite_float(policy_grad_norm),
            "metrics": policy_metrics,
            "epsilon_sweep": policy_rows,
        },
        "one_step_action": {
            "loss": float(action_loss.detach().cpu()),
            "analytic_grad_norm": finite_float(float(action_grad.to(torch.float64).norm().detach().cpu())),
            "metrics": action_metrics,
            "epsilon_sweep": action_rows,
        },
        "one_step_state_action": {
            "loss": float(state_action_loss.detach().cpu()),
            "analytic_grad_norm": finite_float(float(state_grad.to(torch.float64).norm().detach().cpu())),
            "metrics": state_action_metrics,
            "epsilon_sweep": state_action_rows,
            "components": component_rows,
        },
        "gpu": query_gpu(),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grad_check_file = args.grad_check_file or f"{args.env}_gradient_check.json"
    out_path = out_dir / grad_check_file
    write_json(out_path, result)
    print(f"wrote gradient check to {out_path}")
    return result


def make_env_from_args(args: argparse.Namespace, num_envs: int) -> NewtonMuJoCoTorchEnv:
    return NewtonMuJoCoTorchEnv(
        env_name=args.env,
        num_envs=num_envs,
        device=args.device,
        dt=args.dt,
        force_scale=args.force_scale,
        contact_backend=args.contact_backend,
        sim_substeps=args.sim_substeps,
        mujoco_integrator=args.mujoco_integrator,
        mujoco_smooth_adjoint=args.mujoco_smooth_adjoint,
        mujoco_smooth_friction_viscosity=args.mujoco_smooth_friction_viscosity,
        mujoco_smooth_friction_scale=args.mujoco_smooth_friction_scale,
        mujoco_smooth_friction_bypass_kf=args.mujoco_smooth_friction_bypass_kf,
        mujoco_smooth_penalty_damping_alpha=args.mujoco_smooth_penalty_damping_alpha,
        mujoco_smooth_friction_surrogate_alpha=args.mujoco_smooth_friction_surrogate_alpha,
        acrobot_actuation=args.acrobot_actuation,
        ant_asset=args.ant_asset,
        ant_disable_joint_limits=args.ant_disable_joint_limits,
        ant_density_override=args.ant_density_override,
        ant_contact_margin=args.ant_contact_margin,
        ant_contact_gap=args.ant_contact_gap,
        ant_contact_mu=args.ant_contact_mu,
        ant_joint_damping=args.ant_joint_damping,
        ant_armature=args.ant_armature,
        ant_min_up=args.ant_min_up,
        ant_start_height=args.ant_start_height,
        ant_start_joint_q=args.ant_start_joint_q,
        ant_reset_position_scale=args.ant_reset_position_scale,
        ant_reset_angle_scale=args.ant_reset_angle_scale,
        ant_reset_joint_scale=args.ant_reset_joint_scale,
        ant_reset_velocity_scale=args.ant_reset_velocity_scale,
        ant_termination_height=args.ant_termination_height,
        ant_max_healthy_height=args.ant_max_healthy_height,
        ant_observation_style=args.ant_observation_style,
        ant_reward_style=args.ant_reward_style,
        ant_dof_limit_mode=args.ant_dof_limit_mode,
        ant_action_order=args.ant_action_order,
        ant_smooth_up_reward=args.ant_smooth_up_reward,
        ant_reward_min_up=args.ant_reward_min_up,
        ant_reward_min_height=args.ant_reward_min_height,
        hopper_reward_style=args.hopper_reward_style,
        hopper_start_joint_q=args.hopper_start_joint_q,
        hopper_contact_mu=args.hopper_contact_mu,
        hopper_joint_damping=args.hopper_joint_damping,
        hopper_armature=args.hopper_armature,
        hopper_termination_height=args.hopper_termination_height,
        hopper_termination_angle=args.hopper_termination_angle,
        hopper_termination_height_tolerance=args.hopper_termination_height_tolerance,
        hopper_reset_position_scale=args.hopper_reset_position_scale,
        hopper_reset_angle_scale=args.hopper_reset_angle_scale,
        hopper_reset_joint_scale=args.hopper_reset_joint_scale,
        hopper_reset_velocity_scale=args.hopper_reset_velocity_scale,
        phase_observation=args.phase_observation,
        phase_period=args.phase_period,
        hopper_terminate_angle=args.hopper_terminate_angle,
        locomotion_disable_joint_limits=args.locomotion_disable_joint_limits,
        ant_reward=AntRewardWeights(
            progress=args.ant_progress_weight,
            heading=args.ant_heading_weight,
            up=args.ant_up_weight,
            height=args.ant_height_weight,
            action=args.ant_action_penalty,
            alive=args.ant_alive_reward,
            actions_cost=args.ant_actions_cost,
            energy_cost=args.ant_energy_cost,
            dof_limit_cost=args.ant_dof_limit_cost,
            dof_vel_scale=args.ant_dof_vel_scale,
            up_margin=args.ant_up_margin_penalty,
            height_margin=args.ant_height_margin_penalty,
        ),
        hopper_reward=HopperRewardWeights(
            progress=args.hopper_progress_weight,
            height=args.hopper_height_weight,
            angle=args.hopper_angle_weight,
            action=args.hopper_action_penalty,
            alive=args.hopper_alive_reward,
        ),
        cheetah_reward=CheetahRewardWeights(action=args.cheetah_action_penalty),
        acrobot_reward=AcrobotRewardWeights(
            target=args.acrobot_target_weight,
            velocity=args.acrobot_velocity_weight,
            action=args.acrobot_action_weight,
        ),
        contact_reward=ContactTargetRewardWeights(
            target=args.contact_target_weight,
            velocity=args.contact_velocity_weight,
            height=args.contact_height_weight,
            action=args.contact_action_weight,
        ),
        cartpole_reward=CartpoleRewardWeights(
            pole_angle=args.cartpole_pole_angle_penalty,
            pole_velocity=args.cartpole_pole_velocity_penalty,
            cart_position=args.cartpole_cart_position_penalty,
            cart_velocity=args.cartpole_cart_velocity_penalty,
            action=args.cartpole_action_penalty,
        ),
    )


def make_eval_args(args: argparse.Namespace) -> tuple[argparse.Namespace, bool]:
    eval_args = copy.copy(args)
    has_overrides = False
    if getattr(args, "eval_contact_backend", None) is not None:
        eval_args.contact_backend = args.eval_contact_backend
        has_overrides = True
    if getattr(args, "eval_mujoco_smooth_adjoint", None) is not None:
        eval_args.mujoco_smooth_adjoint = args.eval_mujoco_smooth_adjoint
        has_overrides = True
    if getattr(args, "eval_ant_dof_limit_mode", None) is not None:
        eval_args.ant_dof_limit_mode = args.eval_ant_dof_limit_mode
        has_overrides = True
    return eval_args, has_overrides


def run_training(args: argparse.Namespace) -> dict:
    if args.contact_backend is None:
        args.contact_backend = "mujoco" if is_locomotion_env(args.env) or is_contact_target_env(args.env) else "none"
    resolve_ant_defaults(args)
    if args.sim_substeps is None:
        args.sim_substeps = 2 if args.env == "ant" else (16 if is_locomotion_env(args.env) else 1)
    if args.horizon is None:
        args.horizon = 32 if is_locomotion_env(args.env) else (48 if is_contact_target_env(args.env) else (64 if args.env == "acrobot" else 48))
    if args.eval_horizon is None:
        args.eval_horizon = (
            480 if is_locomotion_env(args.env) else (240 if args.env == "acrobot" or is_contact_target_env(args.env) else 180)
        )
    if args.selection_horizon is None and is_locomotion_env(args.env):
        args.selection_horizon = args.eval_horizon
    if args.episode_length is None:
        args.episode_length = 1000 if is_locomotion_env(args.env) else 240
    if args.force_scale is None:
        args.force_scale = 7.5 if args.env == "ant" else (200.0 if is_locomotion_env(args.env) else (35.0 if is_contact_target_env(args.env) else (20.0 if args.env == "acrobot" else 1000.0)))
    if args.grad_clip is None:
        args.grad_clip = 1.0 if is_locomotion_env(args.env) else (10.0 if args.env == "acrobot" or is_contact_target_env(args.env) else 100.0)
    if args.reset_noise is None:
        args.reset_noise = 0.0 if is_locomotion_env(args.env) or is_contact_target_env(args.env) else 0.05
    if args.termination_penalty is None:
        args.termination_penalty = ANT_DEFAULT_TERMINATION_PENALTY if args.env in {"ant", "hopper"} else 0.0
    if args.lr_schedule is None:
        args.lr_schedule = "linear" if is_locomotion_env(args.env) else "constant"
    if args.adam_beta1 is None:
        args.adam_beta1 = 0.7 if is_locomotion_env(args.env) else 0.9
    if args.adam_beta2 is None:
        args.adam_beta2 = 0.95 if is_locomotion_env(args.env) else 0.999
    if args.critic_lr is None:
        args.critic_lr = 2.0e-4 if args.env == "hopper" else (2.0e-3 if args.env in {"ant", "cheetah"} else 1.0e-3)
    if args.critic_iterations is None:
        args.critic_iterations = 16 if is_locomotion_env(args.env) else 8
    if args.critic_method is None:
        args.critic_method = "td-lambda" if is_locomotion_env(args.env) else "one-step"
    if args.stochastic_actor is None:
        args.stochastic_actor = is_locomotion_env(args.env)
    if args.stochastic_init is None:
        args.stochastic_init = is_locomotion_env(args.env)
    if args.use_critic is None:
        args.use_critic = is_locomotion_env(args.env)
    if args.obs_rms is None:
        args.obs_rms = is_locomotion_env(args.env)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    wp.init()

    env = make_env_from_args(args, args.num_envs)
    eval_env_args, has_eval_env_overrides = make_eval_args(args)
    if args.selection_env_counts is not None:
        selection_env_counts = list(dict.fromkeys(int(item) for item in args.selection_env_counts))
    else:
        selection_env_counts = [int(args.selection_num_envs or args.num_envs)]
    if not selection_env_counts or any(item <= 0 for item in selection_env_counts):
        raise ValueError("--selection-env-counts must contain positive integers")
    selection_env_count = selection_env_counts[0]
    selection_env_cache: dict[int, NewtonMuJoCoTorchEnv] = {}

    def get_selection_env(num_envs: int) -> NewtonMuJoCoTorchEnv:
        if num_envs not in selection_env_cache:
            if has_eval_env_overrides or num_envs != args.num_envs:
                selection_env_cache[num_envs] = make_env_from_args(eval_env_args, num_envs)
            else:
                selection_env_cache[num_envs] = env
        return selection_env_cache[num_envs]

    selection_env = get_selection_env(selection_env_count)
    actor = make_actor(
        env,
        stochastic=args.stochastic_actor,
        hidden_dims=args.actor_hidden_dims,
        actor_logstd_init=args.actor_logstd_init,
        actor_layer_norm=args.actor_layer_norm,
        action_squash=args.action_squash,
    )
    if args.actor_path is not None:
        load_actor_checkpoint(actor, args.actor_path, env.torch_device)
    if args.train_final_layer_only:
        freeze_actor_backbone(actor)
    anchor_actor = None
    if args.anchor_action_penalty > 0.0:
        anchor_actor = make_actor(
            env,
            stochastic=args.stochastic_actor,
            hidden_dims=args.actor_hidden_dims,
            actor_logstd_init=args.actor_logstd_init,
            actor_layer_norm=args.actor_layer_norm,
            action_squash=args.action_squash,
        )
        if args.anchor_actor_path is not None:
            load_actor_checkpoint(anchor_actor, args.anchor_actor_path, env.torch_device)
        else:
            anchor_actor.load_state_dict(
                {name: value.detach().clone() for name, value in actor.state_dict().items()}
            )
        anchor_actor.eval()
        for param in anchor_actor.parameters():
            param.requires_grad_(False)
    adam_betas = (args.adam_beta1, args.adam_beta2)
    actor_params = trainable_parameters(actor)
    if not actor_params:
        raise RuntimeError("actor has no trainable parameters")
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(actor_params, lr=args.lr, momentum=args.sgd_momentum)
    else:
        optimizer = torch.optim.Adam(actor_params, lr=args.lr, betas=adam_betas)
    critic = None
    target_critic = None
    critic_optimizer = None
    if args.use_critic:
        critic = make_critic(env, hidden_dims=args.critic_hidden_dims)
        target_critic = copy.deepcopy(critic)
        for param in target_critic.parameters():
            param.requires_grad_(False)
        if args.optimizer == "sgd":
            critic_optimizer = torch.optim.SGD(critic.parameters(), lr=args.critic_lr, momentum=args.sgd_momentum)
        else:
            critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr, betas=adam_betas)
    obs_rms = RunningMeanStd(shape=(env.num_obs,), device=env.torch_device) if args.obs_rms else None
    if obs_rms is not None and args.obs_rms_path is not None:
        obs_data = torch.load(args.obs_rms_path, map_location=env.torch_device)
        obs_rms.mean = obs_data["mean"].to(env.torch_device)
        obs_rms.var = obs_data["var"].to(env.torch_device)
        obs_rms.count = obs_data["count"]
    history = []
    best_state = None
    best_obs_rms = None
    best_epoch = 0
    best_train_reward = None
    best_eval_return = -float("inf")
    best_eval_score = -float("inf")
    q, qd = env.reset(noise=args.reset_noise, stochastic_init=args.stochastic_init)
    prev_action = torch.zeros((env.num_envs, env.num_actions), dtype=torch.float32, device=env.torch_device)
    progress = torch.zeros(env.num_envs, dtype=torch.long, device=env.torch_device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(env.torch_device)
        torch.cuda.synchronize(env.torch_device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    live_history_path = out_dir / f"{args.env}_history_live.json"

    def compact_selection_rollout(candidate: dict) -> dict:
        return {
            "num_envs": candidate.get("num_envs"),
            "return": candidate.get("return"),
            "mean_reward": candidate.get("mean_reward"),
            "alive_fraction": candidate.get("alive_fraction"),
            "mean_forward_displacement": candidate.get("mean_forward_displacement"),
            "min_forward_displacement": candidate.get("min_forward_displacement"),
            "max_forward_displacement": candidate.get("max_forward_displacement"),
            "mean_height": candidate.get("mean_height"),
            "min_height": candidate.get("min_height"),
            "mean_up": candidate.get("mean_up"),
            "min_up": candidate.get("min_up"),
            "mean_heading": candidate.get("mean_heading"),
            "min_heading": candidate.get("min_heading"),
            "terminal_count": candidate.get("terminal_count"),
            "fall_count": candidate.get("fall_count"),
            "invalid_count": candidate.get("invalid_count"),
        }

    def selection_evaluation() -> tuple[dict, float, dict]:
        worst_rollout = None
        worst_score = float("inf")
        worst_shortfalls = None
        worst_env_count = None
        robust_entries = []
        repeat_count = max(1, int(args.selection_repeats))
        selection_horizon = args.selection_horizon or args.eval_horizon
        for env_count in selection_env_counts:
            current_selection_env = get_selection_env(env_count)
            for repeat_idx in range(repeat_count):
                if args.selection_uninterrupted:
                    candidate_rollout = evaluate_policy_uninterrupted(
                        current_selection_env,
                        actor,
                        selection_horizon,
                        obs_rms=obs_rms,
                        termination_penalty=args.termination_penalty,
                        stochastic_init=args.eval_stochastic_init,
                    )
                else:
                    candidate_rollout = evaluate_policy(
                        current_selection_env,
                        actor,
                        selection_horizon,
                        obs_rms=obs_rms,
                        termination_penalty=args.termination_penalty,
                        stochastic_init=args.eval_stochastic_init,
                    )
                candidate_score = rollout_selection_score(
                    candidate_rollout,
                    num_envs=current_selection_env.num_envs,
                    fall_penalty=args.selection_fall_penalty,
                    invalid_penalty=args.selection_invalid_penalty,
                    displacement_weight=args.selection_displacement_weight,
                    height_weight=args.selection_height_weight,
                    up_weight=args.selection_up_weight,
                    heading_weight=args.selection_heading_weight,
                    min_height=args.selection_min_height,
                    min_up=args.selection_min_up,
                    min_heading=args.selection_min_heading,
                    max_abs_joint=args.selection_max_abs_joint,
                    posture_penalty=args.selection_posture_penalty,
                )
                candidate_shortfalls = rollout_constraint_shortfalls(
                    candidate_rollout,
                    min_height=args.selection_min_height,
                    min_up=args.selection_min_up,
                    min_heading=args.selection_min_heading,
                    max_abs_joint=args.selection_max_abs_joint,
                )
                robust_entries.append(
                    {
                        "env_count": env_count,
                        "repeat": repeat_idx + 1,
                        "score": candidate_score,
                        "shortfalls": candidate_shortfalls,
                        "rollout": compact_selection_rollout(candidate_rollout),
                    }
                )
                if candidate_score < worst_score or worst_rollout is None:
                    worst_rollout = candidate_rollout
                    worst_score = candidate_score
                    worst_shortfalls = candidate_shortfalls
                    worst_env_count = env_count
        assert worst_rollout is not None
        assert worst_shortfalls is not None
        worst_rollout = copy.deepcopy(worst_rollout)
        worst_rollout["selection_env_count"] = worst_env_count
        if len(selection_env_counts) > 1 or repeat_count > 1:
            worst_rollout["robust_selection"] = {
                "mode": "worst_score",
                "env_counts": selection_env_counts,
                "repeats": repeat_count,
                "worst_env_count": worst_env_count,
                "worst_score": worst_score,
                "entries": robust_entries,
            }
        return worst_rollout, worst_score, worst_shortfalls

    run_t0 = time.perf_counter()
    initial_selection_rollout, initial_selection_score, _ = selection_evaluation()
    best_eval_return = initial_selection_rollout["return"]
    best_eval_score = initial_selection_score
    accepted_selection_score = initial_selection_score
    best_state = clone_module_state(actor)
    torch.save(best_state, out_dir / f"{args.env}_best_actor.pt")
    if critic is not None:
        torch.save(
            clone_module_state(critic),
            out_dir / f"{args.env}_best_critic.pt",
        )
    if obs_rms is not None:
        best_obs_rms = clone_obs_rms_state(obs_rms)
        torch.save(best_obs_rms, out_dir / f"{args.env}_best_obs_rms.pt")
    write_json(
        live_history_path,
        {
            "env": args.env,
            "timestamp_pacific": pacific_now_iso(),
            "history": history,
            "initial_selection": initial_selection_rollout,
            "initial_selection_score": initial_selection_score,
            "best_epoch": best_epoch,
            "best_eval_return": best_eval_return,
            "best_eval_score": best_eval_score,
        },
    )
    print(
        f"{args.env} initial: sel={initial_selection_score: .1f} "
        f"ret={initial_selection_rollout['return']: .1f} "
        f"dx={initial_selection_rollout['mean_forward_displacement']: .2f} "
        f"falls={initial_selection_rollout['fall_count']} "
        f"invalid={initial_selection_rollout['invalid_count']}",
        flush=True,
    )

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

        guard_actor_state = clone_module_state(actor) if args.selection_guard_updates else None
        guard_optimizer_state = clone_optimizer_state(optimizer) if args.selection_guard_updates else None
        guard_critic_state = clone_module_state(critic) if args.selection_guard_updates else None
        guard_critic_optimizer_state = (
            clone_optimizer_state(critic_optimizer) if args.selection_guard_updates else None
        )
        guard_obs_rms_state = clone_obs_rms_state(obs_rms) if args.selection_guard_updates else None
        guard_reference_score = accepted_selection_score
        guard_acceptance_threshold = (
            accepted_selection_score - args.selection_guard_max_score_drop
            if args.selection_guard_updates
            else None
        )
        update_accepted = True
        update_rolled_back = False

        if args.reset_each_epoch:
            q, qd = env.reset(noise=args.reset_noise, stochastic_init=args.stochastic_init)
            prev_action = torch.zeros_like(prev_action)
            progress = torch.zeros_like(progress)
        else:
            q = q.detach().clone()
            qd = qd.detach().clone()
            prev_action = prev_action.detach().clone()

        warmup_metrics = {
            "warmup_steps": 0,
            "warmup_invalid_resets": 0,
            "warmup_fall_resets": 0,
            "warmup_stop_count": 0,
        }
        if args.training_warmup_steps > 0:
            warmup_steps = int(args.training_warmup_steps)
            if args.training_warmup_jitter > 0:
                low = max(0, warmup_steps - int(args.training_warmup_jitter))
                high = warmup_steps + int(args.training_warmup_jitter)
                warmup_steps = int(np.random.randint(low, high + 1))
            q, qd, prev_action, progress, warmup_metrics = warmup_policy_state(
                env,
                actor,
                q,
                qd,
                prev_action,
                progress,
                steps=warmup_steps,
                obs_rms=obs_rms,
                stochastic_init=args.stochastic_init,
                stop_height_min=args.training_warmup_stop_height_min,
                stop_up_min=args.training_warmup_stop_up_min,
            )

        guard_q = q.detach().clone() if args.selection_guard_updates else None
        guard_qd = qd.detach().clone() if args.selection_guard_updates else None
        guard_prev_action = prev_action.detach().clone() if args.selection_guard_updates else None
        guard_progress = progress.detach().clone() if args.selection_guard_updates else None

        optimizer.zero_grad(set_to_none=True)
        rewards = []
        critic_obs = []
        critic_rewards = []
        critic_done_mask = []
        critic_next_values = []
        gamma_vec = torch.ones(env.num_envs, dtype=torch.float32, device=env.torch_device)
        reward_acc = torch.zeros(env.num_envs, dtype=torch.float32, device=env.torch_device)
        actor_loss = torch.zeros((), dtype=torch.float32, device=env.torch_device)
        anchor_loss_acc = torch.zeros((), dtype=torch.float32, device=env.torch_device)
        norm_stats = obs_rms_snapshot(obs_rms)
        invalid_count = 0
        fall_count = 0
        timeout_count = 0
        survival_penalties = []
        survival_penalty_max = []
        objective_root_x_start = q[:, 0].detach().clone()

        for step_idx in range(args.horizon):
            obs_raw = env.observe(q, qd, prev_action, phase=progress)
            if obs_rms is not None and not args.freeze_obs_rms:
                with torch.no_grad():
                    obs_rms.update(obs_raw.detach())
            obs = normalize_obs(obs_raw, norm_stats)
            if args.use_critic:
                critic_obs.append(obs.detach())
            raw_action = actor(obs, deterministic=not args.stochastic_actor)
            action = squash_policy_action(actor, raw_action)
            if anchor_actor is not None:
                with torch.no_grad():
                    anchor_raw_action = anchor_actor(obs, deterministic=True)
                    anchor_action = squash_policy_action(anchor_actor, anchor_raw_action)
                action_delta = (action - anchor_action).square().sum(dim=-1)
                actor_loss = actor_loss + args.anchor_action_penalty * action_delta.sum()
                anchor_loss_acc = anchor_loss_acc + action_delta.mean().detach()
            q_next, qd_next = env.step(q, qd, env.action_to_joint_f(action))
            invalid = env.invalid_state(q_next, qd_next)
            fell = torch.logical_and(env.fallen_state(q_next), ~invalid)
            q_next, qd_next, action = env.sanitize_state(
                q_next, qd_next, action, invalid, stochastic_init=args.stochastic_init
            )
            next_obs_raw = env.observe(q_next, qd_next, action, phase=progress + 1)
            rew = env.transition_reward(q, qd, q_next, qd_next, action, obs=next_obs_raw)
            rew = finalize_terminal_reward(
                rew,
                invalid=invalid,
                fell=fell,
                termination_penalty=args.termination_penalty,
            )
            survival_penalty, survival_metrics = differentiable_survival_margin(
                env,
                q_next,
                qd_next,
                height_min=args.survival_height_min,
                height_weight=args.survival_height_penalty,
                up_min=args.survival_up_min,
                up_weight=args.survival_up_penalty,
                heading_min=args.survival_heading_min,
                heading_weight=args.survival_heading_penalty,
                angle_max=args.survival_angle_max,
                angle_weight=args.survival_angle_penalty,
            )
            if bool((survival_penalty > 0.0).detach().any().cpu()):
                actor_loss = actor_loss + (gamma_vec * survival_penalty).sum()
            survival_penalties.append(survival_metrics["penalty"].mean())
            survival_penalty_max.append(survival_metrics["penalty"].max())
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
                early_terminal = torch.logical_or(invalid, fell)
                next_value = torch.where(early_terminal, torch.zeros_like(next_value), next_value)
                critic_next_values.append(next_value.detach())
                if step_idx < args.horizon - 1:
                    critic_done_mask.append(done.detach().to(torch.float32))
                else:
                    critic_done_mask.append(torch.ones_like(scaled_rew))
            else:
                next_value = torch.zeros(env.num_envs, dtype=torch.float32, device=env.torch_device)

            reward_acc = reward_acc + gamma_vec * scaled_rew
            if args.actor_objective == "reward":
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

        if args.actor_objective == "displacement":
            actor_loss = actor_loss - args.displacement_objective_weight * (q[:, 0] - objective_root_x_start).sum()

        loss = actor_loss / (args.horizon * args.num_envs)
        loss.backward()
        grad_norm = sanitize_and_clip_grad_norm(
            trainable_parameters(actor), args.grad_clip, args.grad_value_clip
        )
        mean_reward = float(torch.stack(rewards).mean().detach().cpu())
        final_reward = float(rewards[-1].detach().cpu())
        mean_anchor_action_mse = (
            float((anchor_loss_acc / max(1, args.horizon)).detach().cpu()) if anchor_actor is not None else None
        )
        mean_survival_penalty = (
            float(torch.stack(survival_penalties).mean().detach().cpu()) if survival_penalties else 0.0
        )
        max_survival_penalty = (
            float(torch.stack(survival_penalty_max).max().detach().cpu()) if survival_penalty_max else 0.0
        )
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
                    sanitize_and_clip_grad_norm(
                        trainable_parameters(critic), args.grad_clip, args.grad_value_clip
                    )
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

        selection_evaluated = (epoch + 1) % max(1, args.selection_interval) == 0 or epoch == args.epochs - 1
        if selection_evaluated:
            selection_rollout, selection_score, selection_shortfalls = selection_evaluation()
        else:
            selection_rollout = history[-1]["selection_rollout"] if history and "selection_rollout" in history[-1] else initial_selection_rollout
            selection_score = history[-1]["selection_score"] if history else initial_selection_score
            selection_shortfalls = rollout_constraint_shortfalls(
                selection_rollout,
                min_height=args.selection_min_height,
                min_up=args.selection_min_up,
                min_heading=args.selection_min_heading,
                max_abs_joint=args.selection_max_abs_joint,
            )
        if args.selection_guard_updates and selection_evaluated:
            assert guard_acceptance_threshold is not None
            update_accepted = selection_score >= guard_acceptance_threshold
            if update_accepted:
                accepted_selection_score = selection_score
            else:
                update_rolled_back = True
                assert guard_actor_state is not None
                assert guard_optimizer_state is not None
                actor.load_state_dict(guard_actor_state)
                optimizer.load_state_dict(guard_optimizer_state)
                if critic is not None and guard_critic_state is not None:
                    critic.load_state_dict(guard_critic_state)
                if critic_optimizer is not None and guard_critic_optimizer_state is not None:
                    critic_optimizer.load_state_dict(guard_critic_optimizer_state)
                restore_obs_rms_state(obs_rms, guard_obs_rms_state)
                if guard_q is not None and guard_qd is not None and guard_prev_action is not None and guard_progress is not None:
                    q = guard_q
                    qd = guard_qd
                    prev_action = guard_prev_action
                    progress = guard_progress

        if selection_evaluated and update_accepted and selection_score > best_eval_score:
            best_eval_return = selection_rollout["return"]
            best_eval_score = selection_score
            best_train_reward = mean_reward
            best_epoch = epoch + 1
            best_state = clone_module_state(actor)
            torch.save(best_state, out_dir / f"{args.env}_best_actor.pt")
            if critic is not None:
                torch.save(
                    clone_module_state(critic),
                    out_dir / f"{args.env}_best_critic.pt",
                )
            if obs_rms is not None:
                best_obs_rms = clone_obs_rms_state(obs_rms)
                torch.save(best_obs_rms, out_dir / f"{args.env}_best_obs_rms.pt")

        epoch_s = time.perf_counter() - epoch_t0
        history.append(
            {
                "epoch": epoch + 1,
                "selection_evaluated": selection_evaluated,
                "selection_guard_active": args.selection_guard_updates,
                "selection_guard_reference_score": guard_reference_score if args.selection_guard_updates else None,
                "selection_guard_acceptance_threshold": guard_acceptance_threshold,
                "selection_update_accepted": update_accepted,
                "selection_update_rolled_back": update_rolled_back,
                "training_warmup_steps": warmup_metrics["warmup_steps"],
                "training_warmup_invalid_resets": warmup_metrics["warmup_invalid_resets"],
                "training_warmup_fall_resets": warmup_metrics["warmup_fall_resets"],
                "training_warmup_stop_count": warmup_metrics["warmup_stop_count"],
                "mean_reward": mean_reward,
                "final_step_reward": final_reward,
                "loss": float(loss.detach().cpu()),
                "actor_objective": args.actor_objective,
                "displacement_objective_weight": (
                    args.displacement_objective_weight if args.actor_objective == "displacement" else None
                ),
                "grad_norm": grad_norm,
                "anchor_action_mse": mean_anchor_action_mse,
                "anchor_action_penalty": args.anchor_action_penalty if anchor_actor is not None else None,
                "survival_margin_penalty": mean_survival_penalty,
                "survival_margin_penalty_max": max_survival_penalty,
                "value_loss": value_loss,
                "selection_return": selection_rollout["return"],
                "selection_score": selection_score,
                "selection_mean_reward": selection_rollout["mean_reward"],
                "selection_forward_displacement": selection_rollout["mean_forward_displacement"],
                "selection_alive_fraction": selection_rollout["alive_fraction"],
                "selection_mean_height": selection_rollout["mean_height"],
                "selection_min_height": selection_rollout.get("min_height"),
                "selection_mean_up": selection_rollout["mean_up"],
                "selection_min_up": selection_rollout.get("min_up"),
                "selection_mean_heading": selection_rollout["mean_heading"],
                "selection_min_heading": selection_rollout.get("min_heading"),
                "selection_height_shortfall": selection_shortfalls["height_shortfall"],
                "selection_up_shortfall": selection_shortfalls["up_shortfall"],
                "selection_heading_shortfall": selection_shortfalls["heading_shortfall"],
                "selection_joint_shortfall": selection_shortfalls["joint_shortfall"],
                "selection_posture_shortfall": selection_shortfalls["posture_shortfall"],
                "selection_terminal_count": selection_rollout.get("terminal_count"),
                "selection_fall_count": selection_rollout["fall_count"],
                "selection_invalid_count": selection_rollout["invalid_count"],
                "selection_rollout": selection_rollout,
                "invalid_resets": invalid_count,
                "fall_resets": fall_count,
                "timeout_resets": timeout_count,
                "epoch_seconds": epoch_s,
                "fps": args.num_envs * args.horizon / epoch_s,
            }
        )
        write_json(
            live_history_path,
            {
                "env": args.env,
                "timestamp_pacific": pacific_now_iso(),
                "history": history,
                "initial_selection": initial_selection_rollout,
                "initial_selection_score": initial_selection_score,
                "best_epoch": best_epoch,
                "best_eval_return": best_eval_return,
                "best_eval_score": best_eval_score,
            },
        )
        print(
            f"{args.env} epoch {epoch + 1:03d}: reward={mean_reward: .4f} "
            f"loss={float(loss.detach().cpu()): .4f} sel={selection_score: .1f}"
            f"{'' if selection_evaluated else ' (cached)'} "
            f"{'accepted' if update_accepted else 'rolled_back'} "
            f"ret={selection_rollout['return']: .1f} dx={selection_rollout['mean_forward_displacement']: .2f} "
            f"falls={selection_rollout['fall_count']} invalid={selection_rollout['invalid_count']} "
            f"fps={history[-1]['fps']: .1f}"
        )

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

    final_eval_num_envs = args.final_eval_num_envs or selection_env.num_envs
    eval_chunk_size = args.eval_chunk_size or final_eval_num_envs
    use_chunked_final_eval = eval_chunk_size < final_eval_num_envs

    def final_eval_once(*, uninterrupted: bool) -> dict:
        if use_chunked_final_eval:
            return evaluate_policy_chunked(
                lambda n: make_env_from_args(eval_env_args, n),
                actor,
                args.eval_horizon,
                total_envs=final_eval_num_envs,
                chunk_size=eval_chunk_size,
                obs_rms=obs_rms,
                termination_penalty=args.termination_penalty,
                stochastic_init=args.eval_stochastic_init,
                uninterrupted=uninterrupted,
            )
        if final_eval_num_envs != selection_env.num_envs:
            eval_env = make_env_from_args(eval_env_args, final_eval_num_envs)
        else:
            eval_env = selection_env
        evaluator = evaluate_policy_uninterrupted if uninterrupted else evaluate_policy
        return evaluator(
            eval_env,
            actor,
            args.eval_horizon,
            obs_rms=obs_rms,
            termination_penalty=args.termination_penalty,
            stochastic_init=args.eval_stochastic_init,
        )

    rollout = final_eval_once(uninterrupted=False)
    rollout_uninterrupted = final_eval_once(uninterrupted=True)

    def score_rollout(candidate: dict) -> float:
        return rollout_selection_score(
            candidate,
            num_envs=int(candidate.get("num_envs") or selection_env.num_envs),
            fall_penalty=args.selection_fall_penalty,
            invalid_penalty=args.selection_invalid_penalty,
            displacement_weight=args.selection_displacement_weight,
            height_weight=args.selection_height_weight,
            up_weight=args.selection_up_weight,
            heading_weight=args.selection_heading_weight,
            min_height=args.selection_min_height,
            min_up=args.selection_min_up,
            min_heading=args.selection_min_heading,
            max_abs_joint=args.selection_max_abs_joint,
            posture_penalty=args.selection_posture_penalty,
        )

    eval_score = score_rollout(rollout)
    eval_repeats = None
    eval_uninterrupted_repeats = None
    if args.final_eval_repeats > 1:
        repeated_rollouts = [rollout]
        repeated_uninterrupted = [rollout_uninterrupted]
        for _ in range(args.final_eval_repeats - 1):
            repeated_rollouts.append(final_eval_once(uninterrupted=False))
            repeated_uninterrupted.append(final_eval_once(uninterrupted=True))
        eval_repeats = summarize_rollout_repeats(repeated_rollouts, [score_rollout(item) for item in repeated_rollouts])
        eval_uninterrupted_repeats = summarize_rollout_repeats(
            repeated_uninterrupted,
            [score_rollout(item) for item in repeated_uninterrupted],
        )
    video_path = None
    poster_path = None
    if args.render_video:
        render_env = make_env_from_args(eval_env_args, args.video_num_envs)
        video_horizon = args.video_horizon or args.eval_horizon
        video_path, poster_path = render_rollout(
            render_env,
            actor,
            out_dir,
            video_horizon,
            args.env,
            obs_rms=obs_rms,
        )
    total_s = time.perf_counter() - run_t0

    result = {
        "env": args.env,
        "title": "SHAC with MuJoCo Warp",
        "timestamp_pacific": pacific_now_iso(),
        "mujoco_warp_pr": "google-deepmind/mujoco_warp#1423",
        "newton_commit": git_commit_for_imported_module(newton),
        "newton_path": str(Path(newton.__path__[0]).resolve()) if hasattr(newton, "__path__") else None,
        "mujoco_warp_commit": git_commit_for_imported_module(mujoco_warp),
        "num_envs": args.num_envs,
        "selection_num_envs": selection_env.num_envs,
        "selection_env_counts": selection_env_counts,
        "seed": args.seed,
        "contact_backend": args.contact_backend,
        "eval_contact_backend": eval_env_args.contact_backend,
        "horizon": args.horizon,
        "epochs": args.epochs,
        "dt": args.dt,
        "sim_substeps": env.sim_substeps,
        "mujoco_integrator": env.mujoco_integrator,
        "mujoco_smooth_adjoint": args.mujoco_smooth_adjoint,
        "eval_mujoco_smooth_adjoint": eval_env_args.mujoco_smooth_adjoint,
        "mujoco_smooth_friction_viscosity": args.mujoco_smooth_friction_viscosity,
        "mujoco_smooth_friction_scale": args.mujoco_smooth_friction_scale,
        "mujoco_smooth_friction_bypass_kf": args.mujoco_smooth_friction_bypass_kf,
        "mujoco_smooth_penalty_damping_alpha": args.mujoco_smooth_penalty_damping_alpha,
        "mujoco_smooth_friction_surrogate_alpha": args.mujoco_smooth_friction_surrogate_alpha,
        "nconmax": env.nconmax,
        "njmax": env.njmax,
        "world_spacing": list(env.world_spacing) if env.world_spacing is not None else None,
        "force_scale": args.force_scale,
        "episode_length": args.episode_length,
        "disable_eulerdamp": True,
        "stochastic_init": args.stochastic_init,
        "eval_stochastic_init": args.eval_stochastic_init,
        "stochastic_actor": args.stochastic_actor,
        "actor_hidden_dims": args.actor_hidden_dims,
        "actor_logstd_init": args.actor_logstd_init,
        "actor_layer_norm": args.actor_layer_norm,
        "action_squash": args.action_squash,
        "train_final_layer_only": args.train_final_layer_only,
        "actor_objective": args.actor_objective,
        "displacement_objective_weight": (
            args.displacement_objective_weight if args.actor_objective == "displacement" else None
        ),
        "critic_hidden_dims": args.critic_hidden_dims,
        "use_critic": args.use_critic,
        "obs_rms": args.obs_rms,
        "freeze_obs_rms": args.freeze_obs_rms,
        "actor_path": str(args.actor_path) if args.actor_path is not None else None,
        "anchor_actor_path": str(args.anchor_actor_path) if args.anchor_actor_path is not None else None,
        "anchor_action_penalty": args.anchor_action_penalty,
        "obs_rms_path": str(args.obs_rms_path) if args.obs_rms_path is not None else None,
        "critic_method": args.critic_method,
        "td_lambda": args.td_lambda,
        "rew_scale": args.rew_scale,
        "reset_each_epoch": args.reset_each_epoch,
        "training_warmup_steps": args.training_warmup_steps,
        "training_warmup_jitter": args.training_warmup_jitter,
        "training_warmup_stop_height_min": args.training_warmup_stop_height_min,
        "training_warmup_stop_up_min": args.training_warmup_stop_up_min,
        "termination_penalty": args.termination_penalty,
        "terminal_fall_reward": -args.termination_penalty if args.termination_penalty > 0.0 else None,
        "selection_fall_penalty": args.selection_fall_penalty,
        "selection_invalid_penalty": args.selection_invalid_penalty,
        "selection_displacement_weight": args.selection_displacement_weight,
        "selection_height_weight": args.selection_height_weight,
        "selection_up_weight": args.selection_up_weight,
        "selection_heading_weight": args.selection_heading_weight,
        "selection_min_height": args.selection_min_height,
        "selection_min_up": args.selection_min_up,
        "selection_min_heading": args.selection_min_heading,
        "selection_max_abs_joint": args.selection_max_abs_joint,
        "selection_posture_penalty": args.selection_posture_penalty,
        "selection_repeats": args.selection_repeats,
        "selection_interval": args.selection_interval,
        "selection_guard_updates": args.selection_guard_updates,
        "selection_guard_max_score_drop": args.selection_guard_max_score_drop,
        "survival_height_min": args.survival_height_min,
        "survival_height_penalty": args.survival_height_penalty,
        "survival_up_min": args.survival_up_min,
        "survival_up_penalty": args.survival_up_penalty,
        "survival_heading_min": args.survival_heading_min,
        "survival_heading_penalty": args.survival_heading_penalty,
        "survival_angle_max": args.survival_angle_max,
        "survival_angle_penalty": args.survival_angle_penalty,
        "lr": args.lr,
        "min_lr": args.min_lr,
        "optimizer": args.optimizer,
        "sgd_momentum": args.sgd_momentum if args.optimizer == "sgd" else None,
        "adam_beta1": args.adam_beta1,
        "adam_beta2": args.adam_beta2,
        "grad_clip": args.grad_clip,
        "grad_value_clip": args.grad_value_clip,
        "critic_lr": args.critic_lr,
        "critic_iterations": args.critic_iterations,
        "critic_batch_size": args.critic_batch_size,
        "gamma": args.gamma,
        "target_critic_alpha": args.target_critic_alpha,
        "selection_horizon": args.selection_horizon,
        "selection_uninterrupted": args.selection_uninterrupted,
        "final_eval_repeats": args.final_eval_repeats,
        "final_eval_num_envs": final_eval_num_envs,
        "eval_chunk_size": eval_chunk_size if use_chunked_final_eval else None,
        "ant_asset": env.ant_asset if args.env == "ant" else None,
        "ant_max_healthy_height": env.ant_max_healthy_height if args.env == "ant" else None,
        "ant_termination_height": env.ant_termination_height if args.env == "ant" else None,
        "ant_start_height": env.ant_start_height if args.env == "ant" else None,
        "ant_start_joint_q": env.ant_start_joint_q if args.env == "ant" else None,
        "ant_reset_position_scale": env.ant_reset_position_scale if args.env == "ant" else None,
        "ant_reset_angle_scale": env.ant_reset_angle_scale if args.env == "ant" else None,
        "ant_reset_joint_scale": env.ant_reset_joint_scale if args.env == "ant" else None,
        "ant_reset_velocity_scale": env.ant_reset_velocity_scale if args.env == "ant" else None,
        "ant_density_override": env.ant_density_override if args.env == "ant" else None,
        "ant_height_reward_cap": ANT_HEIGHT_REWARD_CAP if args.env == "ant" else None,
        "ant_invalid_penalty": ANT_INVALID_PENALTY if args.env == "ant" else None,
        "lr_schedule": args.lr_schedule,
        "cartpole_reward": env.cartpole_reward.__dict__,
        "acrobot_reward": env.acrobot_reward.__dict__,
        "acrobot_actuation": env.acrobot_actuation if args.env == "acrobot" else None,
        "contact_reward": env.contact_reward.__dict__ if is_contact_target_env(args.env) else None,
        "ant_reward": env.ant_reward.__dict__,
        "hopper_reward": env.hopper_reward.__dict__ if args.env == "hopper" else None,
        "hopper_terminate_angle": env.hopper_terminate_angle if args.env == "hopper" else None,
        "hopper_termination_angle": env.hopper_termination_angle if args.env == "hopper" else None,
        "hopper_termination_height": env.hopper_termination_height if args.env == "hopper" else None,
        "hopper_termination_height_tolerance": env.hopper_termination_height_tolerance if args.env == "hopper" else None,
        "hopper_reward_style": env.hopper_reward_style if args.env == "hopper" else None,
        "hopper_start_joint_q": env.hopper_start_joint_q if args.env == "hopper" else None,
        "hopper_contact_mu": env.hopper_contact_mu if args.env == "hopper" else None,
        "hopper_joint_damping": env.hopper_joint_damping if args.env == "hopper" else None,
        "hopper_armature": env.hopper_armature if args.env == "hopper" else None,
        "hopper_reset_position_scale": env.hopper_reset_position_scale if args.env == "hopper" else None,
        "hopper_reset_angle_scale": env.hopper_reset_angle_scale if args.env == "hopper" else None,
        "hopper_reset_joint_scale": env.hopper_reset_joint_scale if args.env == "hopper" else None,
        "hopper_reset_velocity_scale": env.hopper_reset_velocity_scale if args.env == "hopper" else None,
        "cheetah_reward": env.cheetah_reward.__dict__ if args.env == "cheetah" else None,
        "ant_disable_joint_limits": env.ant_disable_joint_limits if args.env == "ant" else None,
        "ant_contact_margin": env.ant_contact_margin if args.env == "ant" else None,
        "ant_contact_gap": env.ant_contact_gap if args.env == "ant" else None,
        "ant_contact_mu": env.ant_contact_mu if args.env == "ant" else None,
        "ant_joint_damping": env.ant_joint_damping if args.env == "ant" else None,
        "ant_armature": env.ant_armature if args.env == "ant" else None,
        "ant_min_up": env.ant_min_up if args.env == "ant" else None,
        "ant_observation_style": env.ant_observation_style if args.env == "ant" else None,
        "ant_reward_style": env.ant_reward_style if args.env == "ant" else None,
        "ant_dof_limit_mode": env.ant_dof_limit_mode if args.env == "ant" else None,
        "eval_ant_dof_limit_mode": eval_env_args.ant_dof_limit_mode if args.env == "ant" else None,
        "ant_action_order": env.ant_action_order if args.env == "ant" else None,
        "ant_smooth_up_reward": env.ant_smooth_up_reward if args.env == "ant" else None,
        "ant_reward_min_up": env.ant_reward_min_up if args.env == "ant" else None,
        "ant_reward_min_height": env.ant_reward_min_height if args.env == "ant" else None,
        "phase_observation": env.phase_observation if args.env == "ant" else None,
        "phase_period": env.phase_period if args.env == "ant" else None,
        "locomotion_disable_joint_limits": env.locomotion_disable_joint_limits if is_planar_locomotion_env(args.env) else None,
        "total_seconds": total_s,
        "mean_epoch_seconds": float(np.mean([h["epoch_seconds"] for h in history])) if history else None,
        "mean_fps": float(np.mean([h["fps"] for h in history])) if history else None,
        "initial_selection": initial_selection_rollout,
        "initial_selection_score": initial_selection_score,
        "best_epoch": best_epoch,
        "best_train_reward": best_train_reward,
        "best_eval_return": best_eval_return,
        "best_eval_score": best_eval_score,
        "eval_score": eval_score,
        "max_cuda_memory_allocated_mb": (
            float(torch.cuda.max_memory_allocated(env.torch_device) / (1024**2)) if torch.cuda.is_available() else None
        ),
        "max_cuda_memory_reserved_mb": (
            float(torch.cuda.max_memory_reserved(env.torch_device) / (1024**2)) if torch.cuda.is_available() else None
        ),
        "history": history,
        "eval": rollout,
        "eval_uninterrupted": rollout_uninterrupted,
        "eval_repeats": eval_repeats,
        "eval_uninterrupted_repeats": eval_uninterrupted_repeats,
        "video": video_path.name if video_path else None,
        "poster": poster_path.name if poster_path else None,
        "video_horizon": args.video_horizon or args.eval_horizon if video_path else None,
        "gpu": query_gpu(),
    }
    write_json(out_dir / f"{args.env}_results.json", result)
    return result


@torch.no_grad()
def evaluate_policy(
    env: NewtonMuJoCoTorchEnv,
    actor: torch.nn.Module,
    horizon: int,
    *,
    obs_rms: RunningMeanStd | None = None,
    termination_penalty: float = 0.0,
    stochastic_init: bool = False,
) -> dict:
    q, qd = env.reset(noise=0.0, stochastic_init=stochastic_init)
    prev_action = torch.zeros((env.num_envs, env.num_actions), dtype=torch.float32, device=env.torch_device)
    progress = torch.zeros(env.num_envs, dtype=torch.long, device=env.torch_device)
    rewards = []
    final_obs = None
    final_obs_normalized = None
    reset_count = 0
    invalid_count = 0
    fall_count = 0
    timeout_count = 0
    episode_returns = torch.zeros(env.num_envs, dtype=torch.float32, device=env.torch_device)
    episode_lengths = torch.zeros(env.num_envs, dtype=torch.long, device=env.torch_device)
    forward_displacement = torch.zeros(env.num_envs, dtype=torch.float32, device=env.torch_device)
    completed_returns = []
    completed_lengths = []
    height_samples = []
    height_mins = []
    up_samples = []
    up_mins = []
    heading_samples = []
    heading_mins = []
    ant_metric_samples: dict[str, list[torch.Tensor]] = {
        "mean_abs_joint_pos_scaled": [],
        "max_abs_joint_pos_scaled": [],
        "mean_joint_limit_fraction": [],
        "mean_abs_action": [],
        "max_abs_action": [],
        "mean_abs_joint_velocity": [],
    }
    for _ in range(horizon):
        obs = normalize_obs(env.observe(q, qd, prev_action, phase=progress), obs_rms)
        action = deterministic_policy_action(actor, obs)
        q_prev = q
        qd_prev = qd
        root_x_before = q[:, 0].clone()
        q, qd = env.step(q, qd, env.action_to_joint_f(action))
        root_x_after = q[:, 0].clone()
        invalid = env.invalid_state(q, qd)
        fell = torch.logical_and(env.fallen_state(q), ~invalid)
        forward_displacement = forward_displacement + torch.where(
            invalid,
            torch.zeros_like(root_x_after),
            root_x_after - root_x_before,
        )
        q, qd, action = env.sanitize_state(q, qd, action, invalid, stochastic_init=stochastic_init)
        final_obs = env.observe(q, qd, action, phase=progress + 1)
        if env.env_name == "ant":
            torso_pos, _, _, _, up_vec, heading_alignment = env.ant_pose_terms(q, qd)
            heights = torso_pos[:, 1].detach()
            ups = up_vec[:, 1].detach()
            headings = heading_alignment.squeeze(-1).detach()
            height_samples.append(heights.mean())
            height_mins.append(heights.min())
            up_samples.append(ups.mean())
            up_mins.append(ups.min())
            heading_samples.append(headings.mean())
            heading_mins.append(headings.min())
            morphology = env.ant_morphology_metrics(q, qd, action)
            for key, value in morphology.items():
                ant_metric_samples[key].append(value.detach())
        rew = env.transition_reward(q_prev, qd_prev, q, qd, action, obs=final_obs)
        rew = finalize_terminal_reward(rew, invalid=invalid, fell=fell, termination_penalty=termination_penalty)
        rewards.append(rew.mean())
        episode_returns = episode_returns + rew
        episode_lengths = episode_lengths + 1
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
            completed_returns.extend(float(x) for x in episode_returns[done_ids].detach().cpu().tolist())
            completed_lengths.extend(int(x) for x in episode_lengths[done_ids].detach().cpu().tolist())
            episode_returns[done_ids] = 0.0
            episode_lengths[done_ids] = 0
            forward_displacement[done_ids] = torch.where(
                timeout[done_ids],
                forward_displacement[done_ids],
                torch.zeros_like(forward_displacement[done_ids]),
            )
            q, qd = env.reset_done(q, qd, done_ids, stochastic_init=stochastic_init)
            action = torch.where(done.unsqueeze(-1), torch.zeros_like(action), action)
            progress = torch.where(done, torch.zeros_like(progress), progress)
        prev_action = action
    alive_fraction = 1.0 - float(fall_count + invalid_count) / max(1, horizon * env.num_envs)
    return {
        "num_envs": env.num_envs,
        "mean_reward": float(torch.stack(rewards).mean().cpu()),
        "return": float(torch.stack(rewards).sum().cpu()),
        "alive_fraction": alive_fraction,
        "reset_count": reset_count,
        "invalid_count": invalid_count,
        "fall_count": fall_count,
        "timeout_count": timeout_count,
        "completed_episodes": len(completed_returns),
        "mean_completed_return": float(np.mean(completed_returns)) if completed_returns else None,
        "mean_completed_length": float(np.mean(completed_lengths)) if completed_lengths else None,
        "mean_forward_displacement": float(forward_displacement.mean().detach().cpu()),
        "mean_height": float(torch.stack(height_samples).mean().cpu()) if height_samples else None,
        "min_height": float(torch.stack(height_mins).min().cpu()) if height_mins else None,
        "mean_up": float(torch.stack(up_samples).mean().cpu()) if up_samples else None,
        "min_up": float(torch.stack(up_mins).min().cpu()) if up_mins else None,
        "mean_heading": float(torch.stack(heading_samples).mean().cpu()) if heading_samples else None,
        "min_heading": float(torch.stack(heading_mins).min().cpu()) if heading_mins else None,
        "mean_abs_joint_pos_scaled": (
            float(torch.cat(ant_metric_samples["mean_abs_joint_pos_scaled"]).mean().cpu())
            if ant_metric_samples["mean_abs_joint_pos_scaled"]
            else None
        ),
        "max_abs_joint_pos_scaled": (
            float(torch.cat(ant_metric_samples["max_abs_joint_pos_scaled"]).max().cpu())
            if ant_metric_samples["max_abs_joint_pos_scaled"]
            else None
        ),
        "mean_joint_limit_fraction": (
            float(torch.cat(ant_metric_samples["mean_joint_limit_fraction"]).mean().cpu())
            if ant_metric_samples["mean_joint_limit_fraction"]
            else None
        ),
        "max_joint_limit_fraction": (
            float(torch.cat(ant_metric_samples["mean_joint_limit_fraction"]).max().cpu())
            if ant_metric_samples["mean_joint_limit_fraction"]
            else None
        ),
        "mean_abs_action": (
            float(torch.cat(ant_metric_samples["mean_abs_action"]).mean().cpu())
            if ant_metric_samples["mean_abs_action"]
            else None
        ),
        "max_abs_action": (
            float(torch.cat(ant_metric_samples["max_abs_action"]).max().cpu())
            if ant_metric_samples["max_abs_action"]
            else None
        ),
        "mean_abs_joint_velocity": (
            float(torch.cat(ant_metric_samples["mean_abs_joint_velocity"]).mean().cpu())
            if ant_metric_samples["mean_abs_joint_velocity"]
            else None
        ),
        "unfinished_mean_return": float(episode_returns.mean().detach().cpu()),
        "unfinished_mean_length": float(episode_lengths.to(torch.float32).mean().detach().cpu()),
        "final_obs_mean": [float(x) for x in final_obs.mean(dim=0).cpu().tolist()],
        "final_obs_normalized_mean": [float(x) for x in final_obs_normalized.mean(dim=0).cpu().tolist()],
    }


@torch.no_grad()
def evaluate_policy_uninterrupted(
    env: NewtonMuJoCoTorchEnv,
    actor: torch.nn.Module,
    horizon: int,
    *,
    obs_rms: RunningMeanStd | None = None,
    termination_penalty: float = 0.0,
    stochastic_init: bool = False,
) -> dict:
    q, qd = env.reset(noise=0.0, stochastic_init=stochastic_init)
    prev_action = torch.zeros((env.num_envs, env.num_actions), dtype=torch.float32, device=env.torch_device)
    progress = torch.zeros(env.num_envs, dtype=torch.long, device=env.torch_device)
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.torch_device)
    terminal_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.torch_device)
    terminal_fall = torch.zeros(env.num_envs, dtype=torch.bool, device=env.torch_device)
    terminal_invalid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.torch_device)
    episode_returns = torch.zeros(env.num_envs, dtype=torch.float32, device=env.torch_device)
    forward_displacement = torch.zeros(env.num_envs, dtype=torch.float32, device=env.torch_device)
    rewards = []
    height_samples = []
    height_mins = []
    up_samples = []
    up_mins = []
    heading_samples = []
    heading_mins = []
    ant_metric_samples: dict[str, list[torch.Tensor]] = {
        "mean_abs_joint_pos_scaled": [],
        "max_abs_joint_pos_scaled": [],
        "mean_joint_limit_fraction": [],
        "mean_abs_action": [],
        "max_abs_action": [],
        "mean_abs_joint_velocity": [],
    }
    for step_idx in range(horizon):
        obs = normalize_obs(env.observe(q, qd, prev_action, phase=progress), obs_rms)
        action = deterministic_policy_action(actor, obs)
        action = torch.where(active.unsqueeze(-1), action, torch.zeros_like(action))
        root_x_before = q[:, 0].clone()
        q_next, qd_next = env.step(q, qd, env.action_to_joint_f(action))
        root_x_after = q_next[:, 0].clone()
        finite = torch.logical_and(torch.isfinite(q_next).all(dim=-1), torch.isfinite(qd_next).all(dim=-1))
        invalid = torch.logical_or(env.invalid_state(q_next, qd_next), ~finite)
        fell = torch.logical_and(env.fallen_state(q_next), ~invalid)
        new_terminal = torch.logical_and(active, torch.logical_or(fell, invalid))

        next_obs = env.observe(q_next, qd_next, action, phase=progress + 1)
        rew = env.transition_reward(q, qd, q_next, qd_next, action, obs=next_obs)
        rew = finalize_terminal_reward(rew, invalid=invalid, fell=fell, termination_penalty=termination_penalty)
        rew = torch.where(active, rew, torch.zeros_like(rew))
        rewards.append(rew.mean())
        episode_returns = episode_returns + rew
        forward_displacement = forward_displacement + torch.where(
            active & finite,
            root_x_after - root_x_before,
            torch.zeros_like(root_x_after),
        )

        if env.env_name == "ant":
            torso_pos, _, _, _, up_vec, heading_alignment = env.ant_pose_terms(q_next, qd_next)
            sample_mask = active & finite
            if sample_mask.any():
                heights = torso_pos[sample_mask, 1].detach()
                ups = up_vec[sample_mask, 1].detach()
                headings = heading_alignment[sample_mask].detach()
                height_samples.append(heights.mean())
                height_mins.append(heights.min())
                up_samples.append(ups.mean())
                up_mins.append(ups.min())
                heading_samples.append(headings.mean())
                heading_mins.append(headings.min())
                morphology = env.ant_morphology_metrics(q_next, qd_next, action)
                for key, value in morphology.items():
                    ant_metric_samples[key].append(value[sample_mask].detach())

        terminal_step = torch.where(new_terminal, torch.full_like(terminal_step, step_idx + 1), terminal_step)
        terminal_fall = torch.logical_or(terminal_fall, torch.logical_and(active, fell))
        terminal_invalid = torch.logical_or(terminal_invalid, torch.logical_and(active, invalid))
        active = torch.logical_and(active, ~torch.logical_or(fell, invalid))
        freeze = torch.logical_or(invalid, ~finite)
        q = torch.where(freeze.unsqueeze(-1), q, q_next)
        qd = torch.where(freeze.unsqueeze(-1), torch.zeros_like(qd), qd_next)
        prev_action = action
        progress = torch.where(active, progress + 1, progress)

    terminal_count = int((terminal_step >= 0).sum().cpu())
    fall_count = int(terminal_fall.sum().cpu())
    invalid_count = int(terminal_invalid.sum().cpu())
    terminal_steps = terminal_step[terminal_step >= 0]
    terminal_ids = (terminal_step >= 0).nonzero(as_tuple=False).squeeze(-1).detach().cpu().tolist()
    terminal_step_values = terminal_step[terminal_step >= 0].detach().cpu().tolist()
    return {
        "num_envs": env.num_envs,
        "mean_reward": float(torch.stack(rewards).mean().cpu()) if rewards else 0.0,
        "return": float(torch.stack(rewards).sum().cpu()) if rewards else 0.0,
        "alive_fraction": 1.0 - terminal_count / max(1, env.num_envs),
        "terminal_count": terminal_count,
        "fall_count": fall_count,
        "invalid_count": invalid_count,
        "first_terminal_step": int(terminal_steps.min().cpu()) if terminal_steps.numel() else None,
        "mean_terminal_step": float(terminal_steps.to(torch.float32).mean().cpu()) if terminal_steps.numel() else None,
        "terminal_env_ids": [int(item) for item in terminal_ids[:32]],
        "terminal_steps": [int(item) for item in terminal_step_values[:32]],
        "mean_forward_displacement": float(forward_displacement.mean().detach().cpu()),
        "mean_completed_return": float(episode_returns.mean().detach().cpu()),
        "mean_height": float(torch.stack(height_samples).mean().cpu()) if height_samples else None,
        "min_height": float(torch.stack(height_mins).min().cpu()) if height_mins else None,
        "mean_up": float(torch.stack(up_samples).mean().cpu()) if up_samples else None,
        "min_up": float(torch.stack(up_mins).min().cpu()) if up_mins else None,
        "mean_heading": float(torch.stack(heading_samples).mean().cpu()) if heading_samples else None,
        "min_heading": float(torch.stack(heading_mins).min().cpu()) if heading_mins else None,
        "mean_abs_joint_pos_scaled": (
            float(torch.cat(ant_metric_samples["mean_abs_joint_pos_scaled"]).mean().cpu())
            if ant_metric_samples["mean_abs_joint_pos_scaled"]
            else None
        ),
        "max_abs_joint_pos_scaled": (
            float(torch.cat(ant_metric_samples["max_abs_joint_pos_scaled"]).max().cpu())
            if ant_metric_samples["max_abs_joint_pos_scaled"]
            else None
        ),
        "mean_joint_limit_fraction": (
            float(torch.cat(ant_metric_samples["mean_joint_limit_fraction"]).mean().cpu())
            if ant_metric_samples["mean_joint_limit_fraction"]
            else None
        ),
        "max_joint_limit_fraction": (
            float(torch.cat(ant_metric_samples["mean_joint_limit_fraction"]).max().cpu())
            if ant_metric_samples["mean_joint_limit_fraction"]
            else None
        ),
        "mean_abs_action": (
            float(torch.cat(ant_metric_samples["mean_abs_action"]).mean().cpu())
            if ant_metric_samples["mean_abs_action"]
            else None
        ),
        "max_abs_action": (
            float(torch.cat(ant_metric_samples["max_abs_action"]).max().cpu())
            if ant_metric_samples["max_abs_action"]
            else None
        ),
        "mean_abs_joint_velocity": (
            float(torch.cat(ant_metric_samples["mean_abs_joint_velocity"]).mean().cpu())
            if ant_metric_samples["mean_abs_joint_velocity"]
            else None
        ),
        "horizon": horizon,
    }


@torch.no_grad()
def evaluate_policy_chunked(
    env_factory: Callable[[int], NewtonMuJoCoTorchEnv],
    actor: torch.nn.Module,
    horizon: int,
    *,
    total_envs: int,
    chunk_size: int,
    obs_rms: RunningMeanStd | None = None,
    termination_penalty: float = 0.0,
    stochastic_init: bool = False,
    uninterrupted: bool = False,
) -> dict:
    chunks: list[dict] = []
    remaining = int(total_envs)
    chunk_size = max(1, int(chunk_size))
    while remaining > 0:
        current = min(chunk_size, remaining)
        chunk_env = env_factory(current)
        evaluator = evaluate_policy_uninterrupted if uninterrupted else evaluate_policy
        chunks.append(
            evaluator(
                chunk_env,
                actor,
                horizon,
                obs_rms=obs_rms,
                termination_penalty=termination_penalty,
                stochastic_init=stochastic_init,
            )
        )
        del chunk_env
        remaining -= current
    return summarize_rollout_chunks(chunks, total_envs=total_envs, chunk_size=chunk_size)


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
    viewer.show_collision = False
    viewer.set_model(env.model)
    follow_camera = SmoothedFollowCamera(env_name, env.dt)

    q, qd = env.reset(noise=0.0)
    prev_action = torch.zeros((env.num_envs, env.num_actions), dtype=torch.float32, device=env.torch_device)
    progress = torch.zeros(env.num_envs, dtype=torch.long, device=env.torch_device)
    invalid_terminal = torch.zeros(env.num_envs, dtype=torch.bool, device=env.torch_device)
    video_path = out_dir / f"{env_name}_rollout.mp4"
    poster_path = out_dir / f"{env_name}_poster.png"
    frames = []
    with imageio.get_writer(video_path, fps=max(1, int(round(1.0 / env.dt))), codec="libx264", quality=8) as writer:
        with torch.no_grad():
            for frame_idx in range(horizon):
                state = env.make_viewer_state(q, qd)
                follow_camera.update(viewer, q, state=state, model=env.model)
                viewer.begin_frame(frame_idx * env.dt)
                viewer.log_state(state)
                viewer.end_frame()
                frame = viewer.get_frame().numpy()
                frames.append(frame)
                writer.append_data(frame)
                active = ~invalid_terminal
                if active.any():
                    obs = normalize_obs(env.observe(q, qd, prev_action, phase=progress), obs_rms)
                    action = deterministic_policy_action(actor, obs)
                    action = torch.where(active.unsqueeze(-1), action, torch.zeros_like(action))
                    q_next, qd_next = env.step(q, qd, env.action_to_joint_f(action))
                    invalid = env.invalid_state(q_next, qd_next)
                    finite = torch.logical_and(
                        torch.isfinite(q_next).all(dim=-1),
                        torch.isfinite(qd_next).all(dim=-1),
                    )
                    freeze = torch.logical_or(invalid, ~finite)
                    q = torch.where(freeze.unsqueeze(-1), q, q_next)
                    qd = torch.where(freeze.unsqueeze(-1), torch.zeros_like(qd), qd_next)
                    action = torch.where(freeze.unsqueeze(-1), torch.zeros_like(action), action)
                    invalid_terminal = torch.logical_or(invalid_terminal, freeze)
                    progress = progress + active.to(dtype=progress.dtype)
                    prev_action = action
                else:
                    prev_action = torch.zeros_like(prev_action)
    viewer.close()
    imageio.imwrite(poster_path, frames[len(frames) // 2])
    write_json(
        out_dir / f"{env_name}_render_metadata.json",
        {
            "horizon": horizon,
            "dt": env.dt,
            "video": video_path.name,
            "poster": poster_path.name,
            "camera": "SmoothedFollowCamera",
            "source": "ViewerGL.get_frame()",
            "overlays": False,
            "reset_splicing": False,
            "invalid_state_policy": "freeze_invalid_worlds",
            "width": 960,
            "height": 544,
        },
    )
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


def git_commit_for_imported_module(module: Any) -> str | None:
    def commit_for_repo(repo: Path) -> str | None:
        if not (repo / ".git").exists():
            return None
        try:
            return subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return None

    if getattr(module, "__name__", "") == "mujoco_warp":
        for env_name in ("MUJOCO_WARP_REPO", "MJWARP_REPO"):
            value = os.environ.get(env_name)
            if value:
                commit = commit_for_repo(Path(value).expanduser().resolve())
                if commit is not None:
                    return commit
        for repo in (Path.home() / "repos" / "mujoco_warp-differentiability",):
            commit = commit_for_repo(repo)
            if commit is not None:
                return commit

    module_path = Path(getattr(module, "__file__", "")).resolve()
    if not module_path:
        return None
    for candidate in [module_path.parent, *module_path.parents]:
        commit = commit_for_repo(candidate)
        if commit is not None:
            return commit
    return None


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "gradcheck"], default="train")
    parser.add_argument(
        "--env",
        choices=["cartpole", "acrobot", "contact_sphere", "contact_capsule", "ant", "hopper", "cheetah"],
        default="cartpole",
    )
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "assets"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--selection-num-envs", type=int, default=None)
    parser.add_argument("--selection-env-counts", type=parse_int_list, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--eval-horizon", type=int, default=None)
    parser.add_argument("--episode-length", type=int, default=None)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--sim-substeps", type=int, default=None)
    parser.add_argument("--mujoco-integrator", choices=["euler", "rk4", "implicitfast", "implicit"], default="euler")
    parser.add_argument("--mujoco-smooth-adjoint", choices=["off", "smooth", "free_body", "surrogate"], default="off")
    parser.add_argument("--mujoco-smooth-friction-viscosity", type=float, default=10.0)
    parser.add_argument("--mujoco-smooth-friction-scale", type=float, default=0.01)
    parser.add_argument("--mujoco-smooth-friction-bypass-kf", type=float, default=0.0)
    parser.add_argument("--mujoco-smooth-penalty-damping-alpha", type=float, default=0.0)
    parser.add_argument("--mujoco-smooth-friction-surrogate-alpha", type=float, default=0.9)
    parser.add_argument("--eval-contact-backend", choices=["mujoco", "newton", "none"], default=None)
    parser.add_argument("--eval-mujoco-smooth-adjoint", choices=["off", "smooth", "free_body", "surrogate"], default=None)
    parser.add_argument("--eval-ant-dof-limit-mode", choices=["abs", "upper"], default=None)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--min-lr", type=float, default=1.0e-5)
    parser.add_argument("--lr-schedule", choices=["constant", "linear"], default=None)
    parser.add_argument("--adam-beta1", type=float, default=None)
    parser.add_argument("--adam-beta2", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--rew-scale", type=float, default=1.0)
    parser.add_argument("--force-scale", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--grad-value-clip", type=float, default=1.0e6)
    parser.add_argument("--reset-noise", type=float, default=None)
    parser.add_argument("--termination-penalty", type=float, default=None)
    parser.add_argument("--reset-each-epoch", action="store_true")
    parser.add_argument("--training-warmup-steps", type=int, default=0)
    parser.add_argument("--training-warmup-jitter", type=int, default=0)
    parser.add_argument("--training-warmup-stop-height-min", type=float, default=None)
    parser.add_argument("--training-warmup-stop-up-min", type=float, default=None)
    parser.add_argument("--stochastic-init", dest="stochastic_init", action="store_true", default=None)
    parser.add_argument("--deterministic-init", dest="stochastic_init", action="store_false")
    parser.add_argument("--eval-stochastic-init", dest="eval_stochastic_init", action="store_true")
    parser.add_argument("--eval-deterministic-init", dest="eval_stochastic_init", action="store_false")
    parser.set_defaults(eval_stochastic_init=False)
    parser.add_argument("--stochastic-actor", dest="stochastic_actor", action="store_true", default=None)
    parser.add_argument("--deterministic-actor", dest="stochastic_actor", action="store_false")
    parser.add_argument("--actor-hidden-dims", type=parse_int_list, default=None)
    parser.add_argument("--critic-hidden-dims", type=parse_int_list, default=None)
    parser.add_argument("--actor-logstd-init", type=float, default=-1.0)
    parser.add_argument("--action-squash", choices=["tanh", "none"], default="tanh")
    parser.add_argument("--anchor-actor-path", type=Path, default=None)
    parser.add_argument("--anchor-action-penalty", type=float, default=0.0)
    parser.add_argument("--actor-objective", choices=["reward", "displacement"], default="reward")
    parser.add_argument("--displacement-objective-weight", type=float, default=1.0)
    parser.add_argument("--train-final-layer-only", action="store_true")
    parser.add_argument("--actor-layer-norm", dest="actor_layer_norm", action="store_true", default=True)
    parser.add_argument("--no-actor-layer-norm", dest="actor_layer_norm", action="store_false")
    parser.add_argument("--use-critic", dest="use_critic", action="store_true", default=None)
    parser.add_argument("--no-critic", dest="use_critic", action="store_false")
    parser.add_argument("--obs-rms", dest="obs_rms", action="store_true", default=None)
    parser.add_argument("--no-obs-rms", dest="obs_rms", action="store_false")
    parser.add_argument("--freeze-obs-rms", action="store_true")
    parser.add_argument("--critic-lr", type=float, default=None)
    parser.add_argument("--critic-iterations", type=int, default=None)
    parser.add_argument("--critic-batch-size", type=int, default=1024)
    parser.add_argument("--critic-method", choices=["one-step", "td-lambda"], default=None)
    parser.add_argument("--td-lambda", type=float, default=0.95)
    parser.add_argument("--target-critic-alpha", type=float, default=0.2)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--sgd-momentum", type=float, default=0.0)
    parser.add_argument("--cartpole-pole-angle-penalty", type=float, default=1.0)
    parser.add_argument("--cartpole-pole-velocity-penalty", type=float, default=0.1)
    parser.add_argument("--cartpole-cart-position-penalty", type=float, default=0.05)
    parser.add_argument("--cartpole-cart-velocity-penalty", type=float, default=0.1)
    parser.add_argument("--cartpole-action-penalty", type=float, default=0.0)
    parser.add_argument("--acrobot-target-weight", type=float, default=8.0)
    parser.add_argument("--acrobot-velocity-weight", type=float, default=0.05)
    parser.add_argument("--acrobot-action-weight", type=float, default=0.002)
    parser.add_argument("--acrobot-actuation", choices=["elbow", "both"], default="elbow")
    parser.add_argument("--contact-target-weight", type=float, default=8.0)
    parser.add_argument("--contact-velocity-weight", type=float, default=0.05)
    parser.add_argument("--contact-height-weight", type=float, default=1.0)
    parser.add_argument("--contact-action-weight", type=float, default=0.002)
    parser.add_argument("--ant-progress-weight", type=float, default=1.0)
    parser.add_argument("--ant-heading-weight", type=float, default=None)
    parser.add_argument("--ant-up-weight", type=float, default=0.1)
    parser.add_argument("--ant-height-weight", type=float, default=1.0)
    parser.add_argument("--ant-action-penalty", type=float, default=0.0)
    parser.add_argument("--ant-alive-reward", type=float, default=0.5)
    parser.add_argument("--ant-actions-cost", type=float, default=0.005)
    parser.add_argument("--ant-energy-cost", type=float, default=0.05)
    parser.add_argument("--ant-dof-limit-cost", type=float, default=None)
    parser.add_argument("--ant-dof-vel-scale", type=float, default=0.2)
    parser.add_argument("--ant-asset", choices=["diffrl", "nv"], default="diffrl")
    parser.add_argument("--ant-disable-joint-limits", action="store_true")
    parser.add_argument("--ant-density-override", type=float, default=None)
    parser.add_argument("--ant-contact-margin", type=float, default=0.0)
    parser.add_argument("--ant-contact-gap", type=float, default=None)
    parser.add_argument("--ant-contact-mu", type=float, default=None)
    parser.add_argument("--ant-joint-damping", type=float, default=None)
    parser.add_argument("--ant-armature", type=float, default=None)
    parser.add_argument("--ant-min-up", type=float, default=None)
    parser.add_argument("--ant-start-height", type=float, default=None)
    parser.add_argument("--ant-start-joint-q", type=parse_float_list, default=None)
    parser.add_argument("--ant-reset-position-scale", type=float, default=0.1)
    parser.add_argument("--ant-reset-angle-scale", type=float, default=math.pi / 24.0)
    parser.add_argument("--ant-reset-joint-scale", type=float, default=0.2)
    parser.add_argument("--ant-reset-velocity-scale", type=float, default=0.25)
    parser.add_argument("--ant-termination-height", type=float, default=None)
    parser.add_argument("--ant-max-healthy-height", type=float, default=ANT_MAX_HEALTHY_HEIGHT)
    parser.add_argument("--ant-observation-style", choices=["diffrl", "isaac"], default=None)
    parser.add_argument(
        "--ant-reward-style",
        choices=["diffrl", "isaac", "isaaclab", "isaaclab_potential", "isaaclab_potential_height", "isaac_heading_gated"],
        default=None,
    )
    parser.add_argument("--ant-dof-limit-mode", choices=["abs", "upper"], default=None)
    parser.add_argument("--ant-action-order", choices=["joint", "actuator"], default=None)
    parser.add_argument("--ant-smooth-up-reward", action="store_true")
    parser.add_argument("--ant-reward-min-up", type=float, default=None)
    parser.add_argument("--ant-reward-min-height", type=float, default=None)
    parser.add_argument("--ant-up-margin-penalty", type=float, default=0.0)
    parser.add_argument("--ant-height-margin-penalty", type=float, default=0.0)
    parser.add_argument("--phase-observation", action="store_true")
    parser.add_argument("--phase-period", type=int, default=60)
    parser.add_argument("--hopper-height-weight", type=float, default=1.0)
    parser.add_argument("--hopper-progress-weight", type=float, default=1.0)
    parser.add_argument("--hopper-angle-weight", type=float, default=1.0)
    parser.add_argument("--hopper-action-penalty", type=float, default=-0.1)
    parser.add_argument("--hopper-alive-reward", type=float, default=1.0)
    parser.add_argument("--hopper-reward-style", choices=["diffrl", "gym"], default="diffrl")
    parser.add_argument("--hopper-start-joint-q", type=parse_float_list, default=None)
    parser.add_argument("--hopper-contact-mu", type=float, default=0.9)
    parser.add_argument("--hopper-joint-damping", type=float, default=2.0)
    parser.add_argument("--hopper-armature", type=float, default=1.0)
    parser.add_argument("--hopper-termination-height", type=float, default=HOPPER_TERMINATION_HEIGHT)
    parser.add_argument("--hopper-termination-angle", type=float, default=HOPPER_TERMINATION_ANGLE)
    parser.add_argument("--hopper-termination-height-tolerance", type=float, default=HOPPER_TERMINATION_HEIGHT_TOLERANCE)
    parser.add_argument("--hopper-reset-position-scale", type=float, default=0.05)
    parser.add_argument("--hopper-reset-angle-scale", type=float, default=0.1)
    parser.add_argument("--hopper-reset-joint-scale", type=float, default=0.05)
    parser.add_argument("--hopper-reset-velocity-scale", type=float, default=0.05)
    parser.add_argument("--hopper-terminate-angle", action="store_true")
    parser.add_argument("--cheetah-action-penalty", type=float, default=-0.1)
    parser.add_argument("--locomotion-disable-joint-limits", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--render-video", action="store_true")
    parser.add_argument("--video-num-envs", type=int, default=1)
    parser.add_argument("--video-horizon", type=int, default=None)
    parser.add_argument("--final-eval-repeats", type=int, default=1)
    parser.add_argument("--final-eval-num-envs", type=int, default=None)
    parser.add_argument("--eval-chunk-size", type=int, default=None)
    parser.add_argument("--selection-horizon", type=int, default=None)
    parser.add_argument("--selection-uninterrupted", action="store_true")
    parser.add_argument("--selection-fall-penalty", type=float, default=ANT_DEFAULT_SELECTION_FALL_PENALTY)
    parser.add_argument("--selection-invalid-penalty", type=float, default=ANT_DEFAULT_SELECTION_INVALID_PENALTY)
    parser.add_argument("--selection-displacement-weight", type=float, default=0.0)
    parser.add_argument("--selection-height-weight", type=float, default=0.0)
    parser.add_argument("--selection-up-weight", type=float, default=0.0)
    parser.add_argument("--selection-heading-weight", type=float, default=0.0)
    parser.add_argument("--selection-min-height", type=float, default=None)
    parser.add_argument("--selection-min-up", type=float, default=None)
    parser.add_argument("--selection-min-heading", type=float, default=None)
    parser.add_argument("--selection-max-abs-joint", type=float, default=None)
    parser.add_argument("--selection-posture-penalty", type=float, default=0.0)
    parser.add_argument("--selection-repeats", type=int, default=1)
    parser.add_argument("--selection-interval", type=int, default=1)
    parser.add_argument("--selection-guard-updates", action="store_true")
    parser.add_argument("--selection-guard-max-score-drop", type=float, default=0.0)
    parser.add_argument("--survival-height-min", type=float, default=None)
    parser.add_argument("--survival-height-penalty", type=float, default=0.0)
    parser.add_argument("--survival-up-min", type=float, default=None)
    parser.add_argument("--survival-up-penalty", type=float, default=0.0)
    parser.add_argument("--survival-heading-min", type=float, default=None)
    parser.add_argument("--survival-heading-penalty", type=float, default=0.0)
    parser.add_argument("--survival-angle-max", type=float, default=None)
    parser.add_argument("--survival-angle-penalty", type=float, default=0.0)
    parser.add_argument("--contact-backend", choices=["mujoco", "newton", "none"], default=None)
    parser.add_argument("--actor-path", type=Path, default=None)
    parser.add_argument("--obs-rms-path", type=Path, default=None)
    parser.add_argument("--grad-check-file", default=None)
    parser.add_argument("--grad-check-horizon", type=int, default=8)
    parser.add_argument("--grad-check-directions", type=int, default=8)
    parser.add_argument("--grad-check-eps", type=parse_float_list, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("MJWARP_ENABLE_AD", "1")
    parsed_args = parse_args()
    if parsed_args.mode == "gradcheck":
        run_gradient_check(parsed_args)
    else:
        run_training(parsed_args)
