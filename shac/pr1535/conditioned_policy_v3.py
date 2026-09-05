"""Inference-time control conditioning for calibrated v3 gait actors."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


class PreviousActionLowPass(nn.Module):
    """Causally blend desired control with the applied control in the observation."""

    def __init__(
        self,
        actor: nn.Module,
        normalizer: nn.Module,
        *,
        previous_action_offset: int,
        action_dim: int,
        alpha: float,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.normalizer = normalizer
        self.previous_action_offset = previous_action_offset
        self.action_dim = action_dim
        self.alpha = alpha

    def mean_action(self, observation: torch.Tensor) -> torch.Tensor:
        desired_action = self.actor.mean_action(observation)
        raw_observation = (
            observation * torch.sqrt(self.normalizer.variance + 1.0e-6)
            + self.normalizer.mean
        )
        start = self.previous_action_offset
        previous_action = raw_observation[:, start : start + self.action_dim].clamp(
            -1.0, 1.0
        )
        return previous_action + self.alpha * (desired_action - previous_action)


class PeriodicActionResidual(nn.Module):
    """Add an auditable gait-clock residual to a feedback policy.

    The v3 observation ends in the *raw* phase sine and cosine (after undoing
    observation normalization).  Using ``cos(phase) - 1`` makes the residual
    exactly zero during the phase-zero balance warm-up, avoiding the startup
    impulse that made several early Ant attempts fail.
    """

    def __init__(
        self,
        actor: nn.Module,
        normalizer: nn.Module,
        *,
        action_dim: int,
        base_action_scale: float,
        sine_coefficients: list[float],
        cosine_minus_one_coefficients: list[float],
        action_bias: list[float],
    ) -> None:
        super().__init__()
        if not 0.0 <= base_action_scale <= 2.0:
            raise ValueError("base-action scale must be in [0, 2]")
        for name, values in (
            ("sine coefficients", sine_coefficients),
            ("cosine-minus-one coefficients", cosine_minus_one_coefficients),
            ("action bias", action_bias),
        ):
            if len(values) != action_dim:
                raise ValueError(f"{name} must contain {action_dim} values")
        self.actor = actor
        self.normalizer = normalizer
        self.base_action_scale = base_action_scale
        actor_parameter = next(actor.parameters())
        self.register_buffer(
            "sine_coefficients",
            torch.as_tensor(
                sine_coefficients,
                dtype=actor_parameter.dtype,
                device=actor_parameter.device,
            ),
        )
        self.register_buffer(
            "cosine_minus_one_coefficients",
            torch.as_tensor(
                cosine_minus_one_coefficients,
                dtype=actor_parameter.dtype,
                device=actor_parameter.device,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.as_tensor(
                action_bias,
                dtype=actor_parameter.dtype,
                device=actor_parameter.device,
            ),
        )

    def mean_action(self, observation: torch.Tensor) -> torch.Tensor:
        raw_observation = (
            observation * torch.sqrt(self.normalizer.variance + 1.0e-6)
            + self.normalizer.mean
        )
        phase_sine = raw_observation[:, -2:-1]
        phase_cosine_minus_one = raw_observation[:, -1:] - 1.0
        residual = (
            self.action_bias
            + phase_sine * self.sine_coefficients
            + phase_cosine_minus_one * self.cosine_minus_one_coefficients
        )
        return (
            self.base_action_scale * self.actor.mean_action(observation) + residual
        ).clamp(-1.0, 1.0)


class AntPhasePD(nn.Module):
    """State-feedback diagonal trot teacher in the Ant XML joint coordinates."""

    def __init__(
        self,
        actor: nn.Module,
        normalizer: nn.Module,
        *,
        hip_amplitude: float,
        ankle_base: float,
        ankle_lift: float,
        hip_stiffness: float,
        ankle_stiffness: float,
        hip_damping: float,
        ankle_damping: float,
        hip_forward_bias: float,
        hip_side_bias: float,
        ankle_bias: float,
        base_action_scale: float,
        pd_action_scale: float,
        hip_phase_offset: float = 0.0,
        lift_phase_offset: float = 0.0,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.normalizer = normalizer
        self.hip_amplitude = hip_amplitude
        self.ankle_base = ankle_base
        self.ankle_lift = ankle_lift
        self.hip_stiffness = hip_stiffness
        self.ankle_stiffness = ankle_stiffness
        self.hip_damping = hip_damping
        self.ankle_damping = ankle_damping
        self.hip_forward_bias = hip_forward_bias
        self.hip_side_bias = hip_side_bias
        self.ankle_bias = ankle_bias
        self.base_action_scale = base_action_scale
        self.pd_action_scale = pd_action_scale
        self.hip_phase_offset = hip_phase_offset
        self.lift_phase_offset = lift_phase_offset

    def mean_action(self, observation: torch.Tensor) -> torch.Tensor:
        raw = (
            observation * torch.sqrt(self.normalizer.variance + 1.0e-6)
            + self.normalizer.mean
        )
        # raw = qpos[2:] (13), qvel (14), previous action (8), sin, cos.
        joint_qpos = raw[:, 5:13]
        joint_qvel = raw[:, 19:27]
        phase_sine = raw[:, -2]
        phase_cosine = raw[:, -1]
        hip_phase = (
            phase_cosine * math.cos(self.hip_phase_offset)
            - phase_sine * math.sin(self.hip_phase_offset)
        )
        lift_phase = (
            phase_sine * math.cos(self.lift_phase_offset)
            + phase_cosine * math.sin(self.lift_phase_offset)
        )
        pair_a_swing = lift_phase.clamp_min(0.0)
        pair_b_swing = (-lift_phase).clamp_min(0.0)
        swing = torch.stack(
            (pair_a_swing, pair_b_swing, pair_a_swing, pair_b_swing), dim=-1
        )
        hip_sign = raw.new_tensor((-1.0, 1.0, 1.0, -1.0))
        hip_side = raw.new_tensor((1.0, 1.0, -1.0, -1.0))
        ankle_sign = raw.new_tensor((1.0, -1.0, -1.0, 1.0))
        hip_target = (
            -self.hip_amplitude * hip_sign * hip_phase[:, None]
            + self.hip_forward_bias * hip_sign
            + self.hip_side_bias * hip_side
        )
        ankle_magnitude = self.ankle_base - self.ankle_lift * swing
        ankle_target = ankle_sign * (ankle_magnitude + self.ankle_bias)

        target = torch.empty_like(joint_qpos)
        target[:, (0, 2, 4, 6)] = hip_target
        target[:, (1, 3, 5, 7)] = ankle_target
        stiffness = raw.new_tensor(
            (
                self.hip_stiffness,
                self.ankle_stiffness,
                self.hip_stiffness,
                self.ankle_stiffness,
                self.hip_stiffness,
                self.ankle_stiffness,
                self.hip_stiffness,
                self.ankle_stiffness,
            )
        )
        damping = raw.new_tensor(
            (
                self.hip_damping,
                self.ankle_damping,
                self.hip_damping,
                self.ankle_damping,
                self.hip_damping,
                self.ankle_damping,
                self.hip_damping,
                self.ankle_damping,
            )
        )
        joint_order_action = (
            stiffness * (target - joint_qpos) - damping * joint_qvel
        ) / 150.0
        # XML actuator order is hip4, ankle4, hip1, ankle1, hip2, ankle2,
        # hip3, ankle3 rather than qpos joint order.
        pd_action = joint_order_action[:, (6, 7, 0, 1, 2, 3, 4, 5)]
        return (
            self.base_action_scale * self.actor.mean_action(observation)
            + self.pd_action_scale * pd_action
        ).clamp(-1.0, 1.0)


def conditioned_actor(
    actor: nn.Module,
    normalizer: nn.Module,
    checkpoint: dict[str, Any],
    loaded: Any,
) -> nn.Module:
    conditioning = checkpoint.get("control_conditioning", {})
    ant_pd = conditioning.get("ant_phase_pd")
    if ant_pd is not None:
        if getattr(loaded.spec, "name", None) != "ant":
            raise ValueError("Ant phase-PD conditioning requires the Ant model")
        actor = AntPhasePD(
            actor,
            normalizer,
            hip_amplitude=float(ant_pd["hip_amplitude"]),
            ankle_base=float(ant_pd["ankle_base"]),
            ankle_lift=float(ant_pd["ankle_lift"]),
            hip_stiffness=float(ant_pd["hip_stiffness"]),
            ankle_stiffness=float(ant_pd["ankle_stiffness"]),
            hip_damping=float(ant_pd["hip_damping"]),
            ankle_damping=float(ant_pd["ankle_damping"]),
            hip_forward_bias=float(ant_pd.get("hip_forward_bias", 0.0)),
            hip_side_bias=float(ant_pd.get("hip_side_bias", 0.0)),
            ankle_bias=float(ant_pd.get("ankle_bias", 0.0)),
            base_action_scale=float(ant_pd.get("base_action_scale", 0.0)),
            pd_action_scale=float(ant_pd.get("pd_action_scale", 1.0)),
            # Legacy PD searches used the opposite-signed sine hip target.
            # Preserve their semantics when loading a checkpoint written
            # before phase offsets became explicit
            # (-cos(phase + pi/2) == sin(phase)).
            hip_phase_offset=float(
                ant_pd.get("hip_phase_offset", 0.5 * math.pi)
            ),
            lift_phase_offset=float(ant_pd.get("lift_phase_offset", 0.0)),
        )
    periodic = conditioning.get("periodic_action_residual")
    if periodic is not None:
        actor = PeriodicActionResidual(
            actor,
            normalizer,
            action_dim=loaded.model.nu,
            base_action_scale=float(periodic.get("base_action_scale", 1.0)),
            sine_coefficients=list(periodic["sine_coefficients"]),
            cosine_minus_one_coefficients=list(
                periodic.get(
                    "cosine_minus_one_coefficients", [0.0] * loaded.model.nu
                )
            ),
            action_bias=list(periodic.get("action_bias", [0.0] * loaded.model.nu)),
        )
    alpha = float(conditioning.get("previous_action_low_pass_alpha", 1.0))
    if alpha == 1.0:
        return actor
    if not 0.0 < alpha <= 1.0:
        raise ValueError("checkpoint control-filter alpha must be in (0, 1]")
    return PreviousActionLowPass(
        actor,
        normalizer,
        previous_action_offset=(loaded.model.nq - 2) + loaded.model.nv,
        action_dim=loaded.model.nu,
        alpha=alpha,
    )
