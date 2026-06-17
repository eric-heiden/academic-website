from __future__ import annotations

from dataclasses import dataclass
import math

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
}


class SmoothedFollowCamera:
    def __init__(self, env_name: str, dt: float, preset: CameraPreset | None = None):
        self.env_name = env_name
        self.base_env_name = env_name.removesuffix("_ppo")
        self.dt = dt
        self.preset = preset or PRESETS.get(self.base_env_name, PRESETS["ant"])
        self._target: torch.Tensor | None = None

    def target_from_q(self, q: torch.Tensor) -> torch.Tensor:
        q0 = q[0].detach().to(dtype=torch.float32, device="cpu")
        if self.base_env_name in {"ant", "contact_sphere", "contact_capsule"}:
            return q0[0:3].clone()
        if self.base_env_name in {"hopper", "cheetah"}:
            return torch.tensor([float(q0[0]), float(q0[1]), 0.0], dtype=torch.float32)
        if self.base_env_name == "cartpole":
            return torch.tensor([float(q0[0]), 0.0, 0.0], dtype=torch.float32)
        return torch.zeros(3, dtype=torch.float32)

    def update(self, viewer, q: torch.Tensor) -> None:
        raw_target = self.target_from_q(q)
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
