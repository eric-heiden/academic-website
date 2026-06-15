from __future__ import annotations

import argparse
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

from models.actor import ActorDeterministicMLP


@dataclass
class StepContext:
    env: "NewtonMuJoCoTorchEnv"


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
        use_contacts: bool,
    ):
        self.env_name = env_name
        self.num_envs = num_envs
        self.torch_device = torch.device(device)
        self.wp_device = wp.device_from_torch(self.torch_device)
        self.dt = dt
        self.force_scale = force_scale
        self.use_contacts = use_contacts

        if env_name == "cartpole":
            self._build_cartpole()
        elif env_name == "ant":
            self._build_ant()
        else:
            raise ValueError(f"unknown env_name: {env_name}")

        self.solver = SolverMuJoCo(
            self.model,
            requires_grad=True,
            disable_contacts=not use_contacts,
            use_mujoco_contacts=True,
            integrator="euler",
            solver="newton",
            iterations=8,
            ls_iterations=8,
            update_data_interval=1,
            nconmax=64 if use_contacts else None,
            njmax=256 if use_contacts else None,
        )
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
        self.num_obs = int(self.observe(self.start_q, self.start_qd).shape[-1])

    def _build_cartpole(self) -> None:
        source = newton.ModelBuilder(up_axis="Y")
        SolverMuJoCo.register_custom_attributes(source)
        source.default_joint_cfg.armature = 0.1
        source.add_urdf(str(DIFFRL_ROOT / "envs" / "assets" / "cartpole.urdf"), floating=False, up_axis="Y")
        source.joint_q[-1] = -math.pi

        builder = newton.ModelBuilder(up_axis="Y")
        SolverMuJoCo.register_custom_attributes(builder)
        builder.replicate(source, self.num_envs, spacing=(2.0, 0.0, 0.0))
        self.model = builder.finalize(device=self.wp_device, requires_grad=True)
        self.num_actions = 1

    def _build_ant(self) -> None:
        source = newton.ModelBuilder(up_axis="Z")
        SolverMuJoCo.register_custom_attributes(source)
        source.add_mjcf(str(DIFFRL_ROOT / "envs" / "assets" / "ant.xml"))

        builder = newton.ModelBuilder(up_axis="Z")
        SolverMuJoCo.register_custom_attributes(builder)
        builder.replicate(source, self.num_envs, spacing=(4.0, 4.0, 0.0))
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
        self.solver.step(state_in, state_out, control, None, self.dt)
        wp.synchronize()
        self.last_state = state_out
        return {"q": q_wp, "qd": qd_wp, "joint_f": f_wp, "q_out": q_out_wp, "qd_out": qd_out_wp}

    def step(self, q: torch.Tensor, qd: torch.Tensor, joint_f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return NewtonMuJoCoStep.apply(q, qd, joint_f, self.step_ctx)

    def reset(self, noise: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.start_q.clone()
        qd = self.start_qd.clone()
        if noise > 0.0:
            q = q + noise * torch.randn_like(q)
            qd = qd + 0.25 * noise * torch.randn_like(qd)
        return q, qd

    def observe(self, q: torch.Tensor, qd: torch.Tensor) -> torch.Tensor:
        if self.env_name == "cartpole":
            x = q[:, 0:1]
            theta = q[:, 1:2]
            xdot = qd[:, 0:1]
            theta_dot = qd[:, 1:2]
            return torch.cat([x, xdot, torch.sin(theta), torch.cos(theta), theta_dot], dim=-1)

        root = q[:, :7]
        hinges = q[:, 7:]
        root_vel = qd[:, :6]
        hinge_vel = qd[:, 6:]
        height = root[:, 2:3]
        return torch.cat([height, root[:, 3:7], root_vel, hinges, hinge_vel], dim=-1)

    def action_to_joint_f(self, action: torch.Tensor) -> torch.Tensor:
        joint_f = torch.zeros((self.num_envs, self.qd_dim), dtype=torch.float32, device=self.torch_device)
        if self.env_name == "cartpole":
            joint_f[:, 0] = action[:, 0] * self.force_scale
        else:
            joint_f[:, 6 : 6 + self.num_actions] = action[:, : self.num_actions] * self.force_scale
        return joint_f

    def reward(self, q: torch.Tensor, qd: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if self.env_name == "cartpole":
            x = q[:, 0]
            theta = torch.atan2(torch.sin(q[:, 1]), torch.cos(q[:, 1]))
            xdot = qd[:, 0]
            theta_dot = qd[:, 1]
            return -(
                theta.square()
                + 0.1 * theta_dot.square()
                + 0.05 * x.square()
                + 0.1 * xdot.square()
                + 0.001 * action[:, 0].square()
            )

        forward = qd[:, 0]
        height = q[:, 2]
        healthy = -10.0 * (height - 0.55).square()
        energy = -0.005 * action.square().sum(dim=-1)
        return forward + healthy + energy

    def make_viewer_state(self, q: torch.Tensor, qd: torch.Tensor) -> newton.State:
        state = self.model.state(requires_grad=False)
        state.joint_q = wp.from_torch(q.detach().contiguous().view(-1), dtype=wp.float32, requires_grad=False)
        state.joint_qd = wp.from_torch(qd.detach().contiguous().view(-1), dtype=wp.float32, requires_grad=False)
        newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)
        wp.synchronize()
        return state


def make_actor(env: NewtonMuJoCoTorchEnv) -> torch.nn.Module:
    cfg = {
        "actor_mlp": {"units": [64, 64], "activation": "elu"},
    }
    actor = ActorDeterministicMLP(env.num_obs, env.num_actions, cfg, device=str(env.torch_device))
    final = actor.actor[-1]
    if isinstance(final, torch.nn.Linear):
        torch.nn.init.zeros_(final.weight)
        torch.nn.init.zeros_(final.bias)
    return actor


def run_training(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    wp.init()

    env = NewtonMuJoCoTorchEnv(
        env_name=args.env,
        num_envs=args.num_envs,
        device=args.device,
        dt=args.dt,
        force_scale=args.force_scale,
        use_contacts=args.env == "ant",
    )
    actor = make_actor(env)
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr)
    history = []
    best_state = None
    best_epoch = 0
    best_train_reward = -float("inf")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(env.torch_device)
        torch.cuda.synchronize(env.torch_device)

    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        epoch_t0 = time.perf_counter()
        q, qd = env.reset(noise=args.reset_noise)
        optimizer.zero_grad(set_to_none=True)
        rewards = []
        discount = 1.0
        objective = torch.zeros((), dtype=torch.float32, device=env.torch_device)

        for _ in range(args.horizon):
            obs = env.observe(q, qd)
            action = torch.tanh(actor(obs, deterministic=True))
            q, qd = env.step(q, qd, env.action_to_joint_f(action))
            rew = env.reward(q, qd, action)
            rewards.append(rew.detach().mean())
            objective = objective + discount * rew.mean()
            discount *= args.gamma

        loss = -objective / args.horizon
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), args.grad_clip).detach().cpu())
        mean_reward = float(torch.stack(rewards).mean().detach().cpu())
        final_reward = float(rewards[-1].detach().cpu())
        if mean_reward > best_train_reward:
            best_train_reward = mean_reward
            best_epoch = epoch + 1
            best_state = {name: value.detach().clone() for name, value in actor.state_dict().items()}

        optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize(env.torch_device)

        epoch_s = time.perf_counter() - epoch_t0
        history.append(
            {
                "epoch": epoch + 1,
                "mean_reward": mean_reward,
                "final_step_reward": final_reward,
                "loss": float(loss.detach().cpu()),
                "grad_norm": grad_norm,
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
    torch.save(actor.state_dict(), out_dir / f"{args.env}_actor.pt")

    rollout = evaluate_policy(env, actor, args.eval_horizon)
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
                use_contacts=args.env == "ant",
            )
        video_path, poster_path = render_rollout(render_env, actor, out_dir, args.eval_horizon, args.env)

    result = {
        "env": args.env,
        "title": "SHAC with MuJoCo Warp",
        "timestamp_pacific": pacific_now_iso(),
        "mujoco_warp_pr": "google-deepmind/mujoco_warp#1423",
        "mujoco_warp_commit": "255d522299c39a5c6905e439f26861072a10fdf0",
        "num_envs": args.num_envs,
        "horizon": args.horizon,
        "epochs": args.epochs,
        "dt": args.dt,
        "total_seconds": total_s,
        "mean_epoch_seconds": float(np.mean([h["epoch_seconds"] for h in history])),
        "mean_fps": float(np.mean([h["fps"] for h in history])),
        "best_epoch": best_epoch,
        "best_train_reward": best_train_reward,
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
def evaluate_policy(env: NewtonMuJoCoTorchEnv, actor: torch.nn.Module, horizon: int) -> dict:
    q, qd = env.reset(noise=0.0)
    rewards = []
    final_obs = None
    for _ in range(horizon):
        obs = env.observe(q, qd)
        action = torch.tanh(actor(obs, deterministic=True))
        q, qd = env.step(q, qd, env.action_to_joint_f(action))
        rewards.append(env.reward(q, qd, action).mean())
        final_obs = env.observe(q, qd)
    return {
        "mean_reward": float(torch.stack(rewards).mean().cpu()),
        "return": float(torch.stack(rewards).sum().cpu()),
        "final_obs_mean": [float(x) for x in final_obs.mean(dim=0).cpu().tolist()],
    }


def render_rollout(
    env: NewtonMuJoCoTorchEnv,
    actor: torch.nn.Module,
    out_dir: Path,
    horizon: int,
    env_name: str,
) -> tuple[Path, Path]:
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    viewer = newton.viewer.ViewerGL(width=960, height=544, headless=True)
    viewer.set_model(env.model)
    if env_name == "cartpole":
        viewer.set_camera(pos=wp.vec3(7.3, -14.0, 2.3), pitch=-5.0, yaw=-225.0)
        if hasattr(viewer, "camera") and hasattr(viewer.camera, "fov"):
            viewer.camera.fov = 90.0
    else:
        viewer.set_camera(pos=wp.vec3(3.0, -5.0, 2.0), pitch=-18.0, yaw=-135.0)
        if hasattr(viewer, "camera") and hasattr(viewer.camera, "fov"):
            viewer.camera.fov = 72.0

    q, qd = env.reset(noise=0.0)
    video_path = out_dir / f"{env_name}_rollout.mp4"
    poster_path = out_dir / f"{env_name}_poster.png"
    frames = []
    trail: list[tuple[float, float]] = []
    with imageio.get_writer(video_path, fps=max(1, int(round(1.0 / env.dt))), codec="libx264", quality=8) as writer:
        with torch.no_grad():
            for frame_idx in range(horizon):
                state = env.make_viewer_state(q, qd)
                viewer.begin_frame(frame_idx * env.dt)
                viewer.log_state(state)
                viewer.end_frame()
                frame = viewer.get_frame().numpy()
                frame = decorate_rollout_frame(frame, env_name, q, state, env, trail, Image, ImageDraw)
                frames.append(frame)
                writer.append_data(frame)
                obs = env.observe(q, qd)
                action = torch.tanh(actor(obs, deterministic=True))
                q, qd = env.step(q, qd, env.action_to_joint_f(action))
                q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
                qd = torch.nan_to_num(qd, nan=0.0, posinf=0.0, neginf=0.0)
    viewer.close()
    Image.fromarray(frames[len(frames) // 2]).save(poster_path)
    return video_path, poster_path


def decorate_rollout_frame(
    frame: np.ndarray,
    env_name: str,
    q: torch.Tensor,
    state: newton.State,
    env: NewtonMuJoCoTorchEnv,
    trail: list[tuple[float, float]],
    image_cls,
    image_draw,
) -> np.ndarray:
    image = image_cls.fromarray(frame)
    draw = image_draw.Draw(image, "RGBA")
    width, height = image.size

    if env_name == "cartpole":
        q_np = np.nan_to_num(q[0].detach().cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0)
        cart_x = float(q_np[0])
        theta = float(q_np[1])
        pivot = (cart_x, 0.0)
        tip = (cart_x + math.sin(theta), math.cos(theta))
        trail.append(tip)
        if len(trail) > 120:
            del trail[: len(trail) - 120]

        margin = 56
        x_min, x_max = -2.5, 5.0
        z_min, z_max = -1.25, 1.35

        def project(point: tuple[float, float]) -> tuple[int, int]:
            x, z = point
            sx = margin + (x - x_min) / (x_max - x_min) * (width - 2 * margin)
            sy = height - margin - (z - z_min) / (z_max - z_min) * (height - 2 * margin)
            return int(round(sx)), int(round(sy))

        rail_a = project((x_min, 0.0))
        rail_b = project((x_max, 0.0))
        draw.line([rail_a, rail_b], fill=(244, 209, 96, 210), width=5)

        if len(trail) > 1:
            trail_points = [project(p) for p in trail]
            draw.line(trail_points, fill=(116, 206, 238, 130), width=3)

        cx, cy = project(pivot)
        cart_w = max(28, int(0.38 / (x_max - x_min) * (width - 2 * margin)))
        cart_h = 26
        draw.rounded_rectangle(
            [cx - cart_w, cy - cart_h // 2, cx + cart_w, cy + cart_h // 2],
            radius=4,
            fill=(48, 79, 145, 235),
            outline=(184, 225, 255, 240),
            width=2,
        )
        tip_px = project(tip)
        draw.line([project(pivot), tip_px], fill=(244, 115, 90, 245), width=7)
        draw.ellipse(
            [tip_px[0] - 9, tip_px[1] - 9, tip_px[0] + 9, tip_px[1] + 9],
            fill=(255, 230, 109, 245),
            outline=(255, 255, 255, 230),
            width=2,
        )
        return np.asarray(image)

    body_q = np.nan_to_num(state.body_q.numpy().reshape(-1, 7), nan=0.0, posinf=0.0, neginf=0.0)
    bodies_per_env = env.model.body_count // env.num_envs
    body_q = body_q[:bodies_per_env]
    root = body_q[0, :3]
    trail.append((float(root[0]), float(root[1])))
    if len(trail) > 180:
        del trail[: len(trail) - 180]

    margin = 64
    span = 2.6
    center_x = float(root[0])
    center_y = float(root[1])

    def project_xy(point: np.ndarray | tuple[float, float]) -> tuple[int, int]:
        px, py = float(point[0]), float(point[1])
        sx = margin + ((px - center_x) / span + 0.5) * (width - 2 * margin)
        sy = height - margin - ((py - center_y) / span + 0.5) * (height - 2 * margin)
        return int(round(sx)), int(round(sy))

    for offset in np.linspace(-span * 0.5, span * 0.5, 5):
        a = project_xy((center_x - span * 0.5, center_y + float(offset)))
        b = project_xy((center_x + span * 0.5, center_y + float(offset)))
        c = project_xy((center_x + float(offset), center_y - span * 0.5))
        d = project_xy((center_x + float(offset), center_y + span * 0.5))
        draw.line([a, b], fill=(111, 124, 143, 80), width=1)
        draw.line([c, d], fill=(111, 124, 143, 80), width=1)

    if len(trail) > 1:
        trail_points = [project_xy(p) for p in trail]
        draw.line(trail_points, fill=(92, 214, 178, 150), width=4)

    root_px = project_xy(root[:2])
    for point in body_q[1:, :2]:
        p = project_xy(point)
        draw.line([root_px, p], fill=(111, 179, 255, 190), width=4)
        draw.ellipse([p[0] - 6, p[1] - 6, p[0] + 6, p[1] + 6], fill=(251, 191, 36, 235))
    draw.ellipse(
        [root_px[0] - 12, root_px[1] - 12, root_px[0] + 12, root_px[1] + 12],
        fill=(239, 68, 68, 235),
        outline=(255, 255, 255, 220),
        width=2,
    )
    return np.asarray(image)


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
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--eval-horizon", type=int, default=180)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--force-scale", type=float, default=1000.0)
    parser.add_argument("--grad-clip", type=float, default=100.0)
    parser.add_argument("--reset-noise", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--render-video", action="store_true")
    parser.add_argument("--video-num-envs", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("MJWARP_ENABLE_AD", "1")
    run_training(parse_args())
