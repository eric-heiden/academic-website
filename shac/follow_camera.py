from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
import warp as wp


@dataclass(frozen=True)
class CameraPreset:
    offset: tuple[float, float, float]
    look_offset: tuple[float, float, float]
    pitch: float
    yaw: float
    fov: float
    smooth_time: float = 0.35
    reset_distance: float = 2.5


PRESETS: dict[str, CameraPreset] = {
    "cartpole": CameraPreset(offset=(0.0, 2.0, 7.0), look_offset=(0.0, 0.0, 0.0), pitch=0.0, yaw=-90.0, fov=45.0),
    "acrobot": CameraPreset(offset=(0.0, 0.0, 5.0), look_offset=(0.0, 0.0, 0.0), pitch=-90.0, yaw=-90.0, fov=42.0),
    "contact_sphere": CameraPreset(offset=(-2.0, 1.35, 3.4), look_offset=(0.0, 0.12, 0.0), pitch=-18.0, yaw=-145.0, fov=45.0),
    "contact_capsule": CameraPreset(offset=(-2.0, 1.35, 3.4), look_offset=(0.0, 0.12, 0.0), pitch=-18.0, yaw=-145.0, fov=45.0),
    "hopper": CameraPreset(offset=(-2.6, 1.2, 4.0), look_offset=(0.0, 0.25, 0.0), pitch=-15.0, yaw=-120.0, fov=45.0),
    "cheetah": CameraPreset(offset=(-3.6, 1.0, 5.5), look_offset=(0.0, 0.0, 0.0), pitch=-12.0, yaw=-115.0, fov=45.0),
    "ant": CameraPreset(offset=(-2.45, 1.35, 3.35), look_offset=(0.0, 0.32, 0.0), pitch=-18.0, yaw=-135.0, fov=40.0),
    "single_hinge_gravity": CameraPreset(offset=(0.55, 1.05, 2.65), look_offset=(0.55, 0.0, 0.0), pitch=-8.0, yaw=-90.0, fov=42.0),
    "double_hinge_gravity_static": CameraPreset(offset=(0.55, 1.05, 2.65), look_offset=(0.55, 0.0, 0.0), pitch=-8.0, yaw=-90.0, fov=42.0),
    "double_hinge_zero_g_forced": CameraPreset(offset=(0.55, 1.05, 2.65), look_offset=(0.55, 0.0, 0.0), pitch=-8.0, yaw=-90.0, fov=42.0),
    "planar_chain_zero_g": CameraPreset(offset=(0.0, 1.05, 2.8), look_offset=(0.0, 0.0, 0.0), pitch=-8.0, yaw=-90.0, fov=42.0),
    "planar_branch_zero_g": CameraPreset(offset=(0.0, 1.05, 2.8), look_offset=(0.0, 0.0, 0.0), pitch=-8.0, yaw=-90.0, fov=42.0),
    "free_body_zero_g": CameraPreset(offset=(0.0, 1.25, 2.8), look_offset=(0.0, 0.0, 0.0), pitch=-10.0, yaw=-90.0, fov=42.0),
}


class SmoothedFollowCamera:
    def __init__(self, env_name: str, dt: float, preset: CameraPreset | None = None):
        self.env_name = env_name
        self.base_env_name = env_name.removesuffix("_ppo")
        self.dt = dt
        self.preset = preset or PRESETS.get(self.base_env_name, PRESETS["ant"])
        self._target: torch.Tensor | None = None

    def target_from_state(self, q: torch.Tensor, state=None, model=None) -> torch.Tensor | None:
        if state is None or model is None or not hasattr(state, "body_q") or state.body_q is None:
            return None
        try:
            body_q = state.body_q.numpy()
        except Exception:
            return None
        if body_q is None or len(body_q) == 0:
            return None
        body_q = np.asarray(body_q)
        if body_q.ndim != 2 or body_q.shape[1] < 3:
            return None
        env_count = int(q.shape[0]) if hasattr(q, "shape") and len(q.shape) > 0 else 1
        body_count = body_q.shape[0] // max(1, env_count)
        if body_count <= 0:
            return None
        pos = body_q[:body_count, :3]
        finite = np.isfinite(pos).all(axis=1)
        if not finite.any():
            return None

        weights = None
        if hasattr(model, "body_mass") and model.body_mass is not None:
            try:
                mass = np.asarray(model.body_mass.numpy(), dtype=np.float64)
                if mass.shape[0] >= body_count:
                    weights = mass[:body_count]
                    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
            except Exception:
                weights = None
        if weights is None or float(weights[finite].sum()) <= 0.0:
            center = pos[finite].mean(axis=0)
        else:
            center = np.average(pos[finite], axis=0, weights=weights[finite])
        return torch.tensor(center, dtype=torch.float32)

    def target_from_q(self, q: torch.Tensor, state=None, model=None) -> torch.Tensor:
        state_target = self.target_from_state(q, state=state, model=model)
        if state_target is not None and self.base_env_name in {"ant", "hopper", "cheetah"}:
            return state_target
        q0 = q[0].detach().to(dtype=torch.float32, device="cpu")
        if self.base_env_name in {"ant", "contact_sphere", "contact_capsule"}:
            return q0[0:3].clone()
        if self.base_env_name in {"hopper", "cheetah"}:
            return torch.tensor([float(q0[0]), float(q0[1]), 0.0], dtype=torch.float32)
        if self.base_env_name == "cartpole":
            return torch.tensor([float(q0[0]), 0.0, 0.0], dtype=torch.float32)
        if self.base_env_name in {"planar_chain_zero_g", "planar_branch_zero_g"}:
            return torch.tensor([float(q0[0]), 0.0, float(q0[1])], dtype=torch.float32)
        if self.base_env_name == "free_body_zero_g":
            return q0[0:3].clone()
        return torch.zeros(3, dtype=torch.float32)

    def update(self, viewer, q: torch.Tensor, *, state=None, model=None) -> None:
        raw_target = self.target_from_q(q, state=state, model=model)
        if self._target is None:
            self._target = raw_target
        else:
            distance = torch.linalg.vector_norm(raw_target - self._target).item()
            if distance > self.preset.reset_distance:
                self._target = raw_target
            else:
                alpha = 1.0 - math.exp(-self.dt / max(self.preset.smooth_time, 1.0e-6))
                self._target = self._target + alpha * (raw_target - self._target)

        target = self._target
        pos = tuple(float(target[i] + self.preset.offset[i]) for i in range(3))
        look = tuple(float(target[i] + self.preset.look_offset[i]) for i in range(3))
        viewer.set_camera(pos=wp.vec3(*pos), pitch=self.preset.pitch, yaw=self.preset.yaw)
        viewer.camera.look_at(look)
        if hasattr(viewer, "camera") and hasattr(viewer.camera, "fov"):
            viewer.camera.fov = self.preset.fov
