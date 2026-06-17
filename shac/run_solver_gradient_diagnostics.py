#!/usr/bin/env python3
"""Minimal SolverMuJoCo gradient diagnostics for the SHAC report."""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
import warp as wp

import newton
from newton.solvers import SolverMuJoCo

from follow_camera import SmoothedFollowCamera


DEFAULT_EPS = [1.0e-1, 3.0e-2, 1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5, 1.0e-5]


def pacific_now_iso() -> str:
    try:
        tz = ZoneInfo("America/Los_Angeles")
    except Exception:
        tz = timezone(timedelta(hours=-7), name="PDT")
    return datetime.now(tz).isoformat(timespec="seconds")


def finite_float(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")


def clean_grad(tensor: torch.Tensor | None, template: torch.Tensor) -> torch.Tensor:
    if tensor is None:
        return torch.zeros_like(template)
    return torch.nan_to_num(tensor.detach(), nan=0.0, posinf=0.0, neginf=0.0)


@dataclass
class StepContext:
    env: "MinimalSolverEnv"


class NewtonSolverStep(torch.autograd.Function):
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
        f_req = joint_f.detach().clone().requires_grad_(True)
        q_out = torch.empty_like(q_req)
        qd_out = torch.empty_like(qd_req)

        env.zero_solver_buffers()
        with wp.Tape() as tape:
            arrays = env.step_warp(q_req, qd_req, f_req, q_out, qd_out, requires_grad=True, zero_buffers=False)

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


class MinimalSolverEnv:
    def __init__(self, scene: str, device: str, dt: float, *, solver_iterations: int = 4, ls_iterations: int = 4):
        self.scene = scene
        self.torch_device = torch.device(device)
        self.wp_device = wp.device_from_torch(self.torch_device)
        self.dt = dt

        if scene == "single_hinge_zero_g":
            self.model = self._build_single_hinge()
        elif scene == "limited_hinge_zero_g":
            self.model = self._build_limited_hinge()
        elif scene == "free_limited_hinge_zero_g":
            self.model = self._build_free_limited_hinge()
        elif scene == "single_hinge_gravity":
            self.model = self._build_single_hinge_gravity()
        elif scene == "double_hinge_gravity":
            self.model = self._build_double_hinge(gravity=-9.81)
        elif scene in {"double_hinge_gravity_static", "double_hinge_zero_g_dynamic", "double_hinge_zero_g_forced"}:
            self.model = self._build_double_hinge(gravity=-9.81 if scene == "double_hinge_gravity_static" else 0.0)
        elif scene == "double_limited_hinge_zero_g":
            self.model = self._build_double_limited_hinge()
        elif scene == "planar_chain_zero_g":
            self.model = self._build_planar_locomotion_branch(branches=1)
        elif scene == "planar_branch_zero_g":
            self.model = self._build_planar_locomotion_branch(branches=2)
        elif scene == "free_body_zero_g":
            self.model = self._build_free_body()
        else:
            raise ValueError(f"unknown scene: {scene}")

        self.solver = SolverMuJoCo(
            self.model,
            disable_contacts=True,
            requires_grad=True,
            integrator="euler",
            solver="newton",
            jacobian=os.environ.get("NEWTON_SHAC_DIAG_JACOBIAN"),
            iterations=solver_iterations,
            ls_iterations=ls_iterations,
            update_data_interval=1,
        )
        self.step_ctx = StepContext(self)
        self.q_dim = self.model.joint_coord_count
        self.qd_dim = self.model.joint_dof_count

    def _build_single_hinge(self) -> newton.Model:
        builder = newton.ModelBuilder(up_axis="Y", gravity=0.0)
        SolverMuJoCo.register_custom_attributes(builder)
        visual_cfg = self._visual_shape_cfg()
        inertia = wp.mat33(np.eye(3, dtype=np.float32))
        link = builder.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=inertia)
        builder.add_shape_box(
            body=link,
            xform=wp.transform(wp.vec3(0.45, 0.0, 0.0), wp.quat_identity()),
            hx=0.45,
            hy=0.045,
            hz=0.045,
            cfg=visual_cfg,
            color=wp.vec3(0.17, 0.45, 0.88),
        )
        joint = builder.add_joint_revolute(parent=-1, child=link, axis=wp.vec3(0.0, 0.0, 1.0), armature=0.0)
        builder.add_articulation([joint])
        return builder.finalize(device=self.wp_device, requires_grad=True)

    def _build_free_limited_hinge(self) -> newton.Model:
        builder = newton.ModelBuilder(up_axis="Y", gravity=0.0)
        SolverMuJoCo.register_custom_attributes(builder)
        visual_cfg = self._visual_shape_cfg()
        base_inertia = wp.mat33(np.diag([0.12, 0.10, 0.09]).astype(np.float32))
        base = builder.add_link(mass=2.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=base_inertia)
        builder.add_shape_box(
            body=base,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            hx=0.16,
            hy=0.10,
            hz=0.08,
            cfg=visual_cfg,
            color=wp.vec3(0.43, 0.42, 0.85),
        )
        free = builder.add_joint_free(parent=-1, child=base)
        child_inertia = wp.mat33(np.diag([0.05, 0.04, 0.03]).astype(np.float32))
        child = builder.add_link(mass=0.7, com=wp.vec3(0.28, 0.0, 0.0), inertia=child_inertia)
        builder.add_shape_box(
            body=child,
            xform=wp.transform(wp.vec3(0.28, 0.0, 0.0), wp.quat_identity()),
            hx=0.28,
            hy=0.04,
            hz=0.04,
            cfg=visual_cfg,
            color=wp.vec3(0.91, 0.52, 0.17),
        )
        hinge = builder.add_joint_revolute(
            parent=base,
            child=child,
            parent_xform=wp.transform(wp.vec3(0.18, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            axis=wp.vec3(0.0, 0.0, 1.0),
            limit_lower=-0.5,
            limit_upper=0.5,
            limit_ke=1.0e3,
            limit_kd=10.0,
            armature=0.0,
        )
        builder.add_articulation([free, hinge])
        return builder.finalize(device=self.wp_device, requires_grad=True)

    def _build_limited_hinge(self) -> newton.Model:
        builder = newton.ModelBuilder(up_axis="Y", gravity=0.0)
        SolverMuJoCo.register_custom_attributes(builder)
        visual_cfg = self._visual_shape_cfg()
        inertia = wp.mat33(np.diag([0.08, 0.08, 0.08]).astype(np.float32))
        link = builder.add_link(mass=1.0, com=wp.vec3(0.35, 0.0, 0.0), inertia=inertia)
        builder.add_shape_box(
            body=link,
            xform=wp.transform(wp.vec3(0.35, 0.0, 0.0), wp.quat_identity()),
            hx=0.35,
            hy=0.045,
            hz=0.045,
            cfg=visual_cfg,
            color=wp.vec3(0.9, 0.37, 0.16),
        )
        joint = builder.add_joint_revolute(
            parent=-1,
            child=link,
            axis=wp.vec3(0.0, 0.0, 1.0),
            limit_lower=-0.5,
            limit_upper=0.5,
            limit_ke=1.0e3,
            limit_kd=10.0,
            armature=0.0,
        )
        builder.add_articulation([joint])
        return builder.finalize(device=self.wp_device, requires_grad=True)

    def _build_single_hinge_gravity(self) -> newton.Model:
        builder = newton.ModelBuilder(up_axis="Y", gravity=-9.81)
        SolverMuJoCo.register_custom_attributes(builder)
        visual_cfg = self._visual_shape_cfg()
        inertia = wp.mat33(np.diag([0.08, 0.08, 0.08]).astype(np.float32))
        link = builder.add_link(mass=1.0, com=wp.vec3(0.45, 0.0, 0.0), inertia=inertia)
        builder.add_shape_box(
            body=link,
            xform=wp.transform(wp.vec3(0.45, 0.0, 0.0), wp.quat_identity()),
            hx=0.45,
            hy=0.045,
            hz=0.045,
            cfg=visual_cfg,
            color=wp.vec3(0.08, 0.57, 0.49),
        )
        joint = builder.add_joint_revolute(parent=-1, child=link, axis=wp.vec3(0.0, 0.0, 1.0), armature=0.0)
        builder.add_articulation([joint])
        return builder.finalize(device=self.wp_device, requires_grad=True)

    def _build_double_hinge(self, gravity: float) -> newton.Model:
        builder = newton.ModelBuilder(up_axis="Y", gravity=gravity)
        SolverMuJoCo.register_custom_attributes(builder)
        visual_cfg = self._visual_shape_cfg()
        inertia0 = wp.mat33(np.diag([0.08, 0.08, 0.08]).astype(np.float32))
        inertia1 = wp.mat33(np.diag([0.05, 0.05, 0.05]).astype(np.float32))
        link0 = builder.add_link(mass=1.0, com=wp.vec3(0.35, 0.0, 0.0), inertia=inertia0)
        builder.add_shape_box(
            body=link0,
            xform=wp.transform(wp.vec3(0.35, 0.0, 0.0), wp.quat_identity()),
            hx=0.35,
            hy=0.04,
            hz=0.04,
            cfg=visual_cfg,
            color=wp.vec3(0.86, 0.29, 0.31),
        )
        joint0 = builder.add_joint_revolute(parent=-1, child=link0, axis=wp.vec3(0.0, 0.0, 1.0), armature=0.0)
        link1 = builder.add_link(mass=0.8, com=wp.vec3(0.28, 0.0, 0.0), inertia=inertia1)
        builder.add_shape_box(
            body=link1,
            xform=wp.transform(wp.vec3(0.28, 0.0, 0.0), wp.quat_identity()),
            hx=0.28,
            hy=0.038,
            hz=0.038,
            cfg=visual_cfg,
            color=wp.vec3(0.89, 0.55, 0.14),
        )
        joint1 = builder.add_joint_revolute(
            parent=link0,
            child=link1,
            parent_xform=wp.transform(wp.vec3(0.7, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            axis=wp.vec3(0.0, 0.0, 1.0),
            armature=0.0,
        )
        builder.add_articulation([joint0, joint1])
        return builder.finalize(device=self.wp_device, requires_grad=True)

    def _build_double_limited_hinge(self) -> newton.Model:
        builder = newton.ModelBuilder(up_axis="Y", gravity=0.0)
        SolverMuJoCo.register_custom_attributes(builder)
        visual_cfg = self._visual_shape_cfg()
        inertia0 = wp.mat33(np.diag([0.08, 0.08, 0.08]).astype(np.float32))
        inertia1 = wp.mat33(np.diag([0.05, 0.05, 0.05]).astype(np.float32))
        link0 = builder.add_link(mass=1.0, com=wp.vec3(0.35, 0.0, 0.0), inertia=inertia0)
        builder.add_shape_box(
            body=link0,
            xform=wp.transform(wp.vec3(0.35, 0.0, 0.0), wp.quat_identity()),
            hx=0.35,
            hy=0.04,
            hz=0.04,
            cfg=visual_cfg,
            color=wp.vec3(0.84, 0.27, 0.32),
        )
        joint0 = builder.add_joint_revolute(
            parent=-1,
            child=link0,
            axis=wp.vec3(0.0, 0.0, 1.0),
            limit_lower=-0.5,
            limit_upper=0.5,
            limit_ke=1.0e3,
            limit_kd=10.0,
            armature=0.0,
        )
        link1 = builder.add_link(mass=0.8, com=wp.vec3(0.28, 0.0, 0.0), inertia=inertia1)
        builder.add_shape_box(
            body=link1,
            xform=wp.transform(wp.vec3(0.28, 0.0, 0.0), wp.quat_identity()),
            hx=0.28,
            hy=0.038,
            hz=0.038,
            cfg=visual_cfg,
            color=wp.vec3(0.9, 0.56, 0.15),
        )
        joint1 = builder.add_joint_revolute(
            parent=link0,
            child=link1,
            parent_xform=wp.transform(wp.vec3(0.7, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            axis=wp.vec3(0.0, 0.0, 1.0),
            limit_lower=-0.5,
            limit_upper=0.5,
            limit_ke=1.0e3,
            limit_kd=10.0,
            armature=0.0,
        )
        builder.add_articulation([joint0, joint1])
        return builder.finalize(device=self.wp_device, requires_grad=True)

    def _build_planar_locomotion_branch(self, branches: int) -> newton.Model:
        builder = newton.ModelBuilder(up_axis="Y", gravity=0.0)
        SolverMuJoCo.register_custom_attributes(builder)
        visual_cfg = self._visual_shape_cfg()

        def unlimited(axis: wp.vec3, *, armature: float = 0.0) -> newton.ModelBuilder.JointDofConfig:
            return newton.ModelBuilder.JointDofConfig.create_unlimited(axis)

        torso_inertia = wp.mat33(np.diag([0.18, 0.22, 0.12]).astype(np.float32))
        torso = builder.add_link(mass=2.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=torso_inertia, label="planar_torso")
        builder.add_shape_box(
            body=torso,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            hx=0.34,
            hy=0.075,
            hz=0.10,
            cfg=visual_cfg,
            color=wp.vec3(0.20, 0.43, 0.80),
        )
        root = builder.add_joint_d6(
            parent=-1,
            child=torso,
            linear_axes=[unlimited(wp.vec3(1.0, 0.0, 0.0)), unlimited(wp.vec3(0.0, 0.0, 1.0))],
            angular_axes=[unlimited(wp.vec3(0.0, 1.0, 0.0))],
            label="planar_root",
        )

        joints = [root]
        anchors = [0.32] if branches == 1 else [0.32, -0.32]
        colors = [
            (wp.vec3(0.91, 0.45, 0.18), wp.vec3(0.97, 0.70, 0.21), wp.vec3(0.28, 0.67, 0.54)),
            (wp.vec3(0.77, 0.30, 0.61), wp.vec3(0.46, 0.50, 0.86), wp.vec3(0.22, 0.62, 0.78)),
        ]

        for branch_id, anchor_x in enumerate(anchors):
            direction = 1.0 if anchor_x >= 0.0 else -1.0
            parent = torso
            parent_anchor = anchor_x
            for segment_id, length in enumerate((0.30, 0.26, 0.22)):
                mass = 0.65 - 0.12 * segment_id
                inertia = wp.mat33(np.diag([0.035, 0.045, 0.025]).astype(np.float32))
                link = builder.add_link(
                    mass=mass,
                    com=wp.vec3(direction * 0.5 * length, 0.0, 0.0),
                    inertia=inertia,
                    label=f"branch{branch_id}_segment{segment_id}",
                )
                builder.add_shape_box(
                    body=link,
                    xform=wp.transform(wp.vec3(direction * 0.5 * length, 0.0, 0.0), wp.quat_identity()),
                    hx=0.5 * length,
                    hy=0.045,
                    hz=0.045,
                    cfg=visual_cfg,
                    color=colors[branch_id][segment_id],
                )
                joint = builder.add_joint_revolute(
                    parent=parent,
                    child=link,
                    parent_xform=wp.transform(wp.vec3(parent_anchor, 0.0, 0.0), wp.quat_identity()),
                    child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                    axis=wp.vec3(0.0, 1.0, 0.0),
                    armature=0.0,
                    limit_ke=0.0,
                    limit_kd=0.0,
                )
                joints.append(joint)
                parent = link
                parent_anchor = direction * length

        builder.add_articulation(joints)
        return builder.finalize(device=self.wp_device, requires_grad=True)

    def _build_free_body(self) -> newton.Model:
        builder = newton.ModelBuilder(up_axis="Y", gravity=0.0)
        SolverMuJoCo.register_custom_attributes(builder)
        visual_cfg = self._visual_shape_cfg()
        inertia = wp.mat33(np.eye(3, dtype=np.float32))
        body = builder.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=inertia)
        builder.add_shape_box(
            body=body,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            hx=0.18,
            hy=0.13,
            hz=0.09,
            cfg=visual_cfg,
            color=wp.vec3(0.48, 0.32, 0.82),
        )
        joint = builder.add_joint_free(parent=-1, child=body)
        builder.add_articulation([joint])
        return builder.finalize(device=self.wp_device, requires_grad=True)

    @staticmethod
    def _visual_shape_cfg() -> newton.ModelBuilder.ShapeConfig:
        return newton.ModelBuilder.ShapeConfig(
            density=0.0,
            collision_group=0,
            has_shape_collision=False,
            has_particle_collision=False,
            is_visible=True,
        )

    def base_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.scene == "single_hinge_zero_g":
            q = torch.tensor([[0.23]], dtype=torch.float32, device=self.torch_device)
            qd = torch.tensor([[0.17]], dtype=torch.float32, device=self.torch_device)
            joint_f = torch.tensor([[0.00]], dtype=torch.float32, device=self.torch_device)
            return q, qd, joint_f
        if self.scene == "single_hinge_gravity":
            q = torch.tensor([[0.41]], dtype=torch.float32, device=self.torch_device)
            qd = torch.tensor([[0.23]], dtype=torch.float32, device=self.torch_device)
            joint_f = torch.tensor([[0.00]], dtype=torch.float32, device=self.torch_device)
            return q, qd, joint_f
        if self.scene == "limited_hinge_zero_g":
            q = torch.tensor([[0.72]], dtype=torch.float32, device=self.torch_device)
            qd = torch.tensor([[0.18]], dtype=torch.float32, device=self.torch_device)
            joint_f = torch.tensor([[0.00]], dtype=torch.float32, device=self.torch_device)
            return q, qd, joint_f
        if self.scene == "free_limited_hinge_zero_g":
            q = torch.tensor([[0.05, 0.04, -0.03, 0.02, -0.03, 0.04, 1.0, 0.72]], dtype=torch.float32, device=self.torch_device)
            q[:, 3:7] = torch.nn.functional.normalize(q[:, 3:7], dim=-1)
            qd = torch.tensor([[0.08, -0.04, 0.03, 0.02, -0.01, 0.03, 0.18]], dtype=torch.float32, device=self.torch_device)
            joint_f = torch.zeros((1, 7), dtype=torch.float32, device=self.torch_device)
            return q, qd, joint_f
        if self.scene == "double_hinge_gravity":
            q = torch.tensor([[0.37, -0.26]], dtype=torch.float32, device=self.torch_device)
            qd = torch.tensor([[0.19, -0.31]], dtype=torch.float32, device=self.torch_device)
            joint_f = torch.tensor([[0.00, 0.00]], dtype=torch.float32, device=self.torch_device)
            return q, qd, joint_f
        if self.scene == "double_limited_hinge_zero_g":
            q = torch.tensor([[0.72, -0.74]], dtype=torch.float32, device=self.torch_device)
            qd = torch.tensor([[0.18, -0.16]], dtype=torch.float32, device=self.torch_device)
            joint_f = torch.tensor([[0.00, 0.00]], dtype=torch.float32, device=self.torch_device)
            return q, qd, joint_f
        if self.scene == "double_hinge_gravity_static":
            q = torch.tensor([[0.37, -0.26]], dtype=torch.float32, device=self.torch_device)
            qd = torch.tensor([[0.0, 0.0]], dtype=torch.float32, device=self.torch_device)
            joint_f = torch.tensor([[0.00, 0.00]], dtype=torch.float32, device=self.torch_device)
            return q, qd, joint_f
        if self.scene == "double_hinge_zero_g_dynamic":
            q = torch.tensor([[0.37, -0.26]], dtype=torch.float32, device=self.torch_device)
            qd = torch.tensor([[0.19, -0.31]], dtype=torch.float32, device=self.torch_device)
            joint_f = torch.tensor([[0.00, 0.00]], dtype=torch.float32, device=self.torch_device)
            return q, qd, joint_f
        if self.scene == "double_hinge_zero_g_forced":
            q = torch.tensor([[0.37, -0.26]], dtype=torch.float32, device=self.torch_device)
            qd = torch.tensor([[0.0, 0.0]], dtype=torch.float32, device=self.torch_device)
            joint_f = torch.tensor([[0.25, -0.15]], dtype=torch.float32, device=self.torch_device)
            return q, qd, joint_f
        if self.scene == "planar_chain_zero_g":
            q = torch.tensor([[0.0, 0.0, 0.05, 0.14, -0.18, 0.11]], dtype=torch.float32, device=self.torch_device)
            qd = torch.zeros((1, 6), dtype=torch.float32, device=self.torch_device)
            joint_f = torch.tensor([[0.0, 0.0, 0.0, 0.35, -0.22, 0.17]], dtype=torch.float32, device=self.torch_device)
            return q, qd, joint_f
        if self.scene == "planar_branch_zero_g":
            q = torch.tensor(
                [[0.0, 0.0, 0.0, 0.10, -0.15, 0.08, -0.12, 0.18, -0.06]],
                dtype=torch.float32,
                device=self.torch_device,
            )
            qd = torch.zeros((1, 9), dtype=torch.float32, device=self.torch_device)
            joint_f = torch.tensor(
                [[0.0, 0.0, 0.0, 0.45, -0.30, 0.20, -0.35, 0.25, -0.18]],
                dtype=torch.float32,
                device=self.torch_device,
            )
            return q, qd, joint_f

        q = torch.tensor(
            [[0.11, 0.21, -0.17, 0.02, -0.03, 0.04, 1.0]], dtype=torch.float32, device=self.torch_device
        )
        q[:, 3:7] = torch.nn.functional.normalize(q[:, 3:7], dim=-1)
        qd = torch.tensor([[0.13, -0.19, 0.07, 0.031, -0.023, 0.017]], dtype=torch.float32, device=self.torch_device)
        joint_f = torch.zeros((1, 6), dtype=torch.float32, device=self.torch_device)
        return q, qd, joint_f

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
        return {"q": q_wp, "qd": qd_wp, "joint_f": f_wp, "q_out": q_out_wp, "qd_out": qd_out_wp}

    def step(self, q: torch.Tensor, qd: torch.Tensor, joint_f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return NewtonSolverStep.apply(q, qd, joint_f, self.step_ctx)

    def make_viewer_state(self, q: torch.Tensor, qd: torch.Tensor) -> newton.State:
        state = self.model.state(requires_grad=False)
        state.joint_q = wp.from_torch(q.detach().contiguous().view(-1), dtype=wp.float32, requires_grad=False)
        state.joint_qd = wp.from_torch(qd.detach().contiguous().view(-1), dtype=wp.float32, requires_grad=False)
        newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)
        wp.synchronize()
        return state


@dataclass
class DiagnosticCase:
    name: str
    scene: str
    q_weight: list[float]
    qd_weight: list[float]
    groups: dict[str, tuple[str, list[int]]]
    normalize_quat_for_fd: bool = False


def make_cases(dt: float) -> list[DiagnosticCase]:
    return [
        DiagnosticCase(
            name="hinge_q_out",
            scene="single_hinge_zero_g",
            q_weight=[1.0],
            qd_weight=[0.0],
            groups={
                "joint_q": ("q", [0]),
                "joint_qd": ("qd", [0]),
                "joint_force": ("joint_f", [0]),
            },
        ),
        DiagnosticCase(
            name="hinge_qd_out",
            scene="single_hinge_zero_g",
            q_weight=[0.0],
            qd_weight=[1.0],
            groups={
                "joint_q": ("q", [0]),
                "joint_qd": ("qd", [0]),
                "joint_force": ("joint_f", [0]),
            },
        ),
        DiagnosticCase(
            name="pendulum_qd_out",
            scene="single_hinge_gravity",
            q_weight=[0.0],
            qd_weight=[1.0],
            groups={
                "joint_q": ("q", [0]),
                "joint_qd": ("qd", [0]),
                "joint_force": ("joint_f", [0]),
            },
        ),
        DiagnosticCase(
            name="limited_hinge_qd_out",
            scene="limited_hinge_zero_g",
            q_weight=[0.0],
            qd_weight=[1.0],
            groups={
                "joint_q": ("q", [0]),
                "joint_qd": ("qd", [0]),
                "joint_force": ("joint_f", [0]),
            },
        ),
        DiagnosticCase(
            name="free_limited_hinge_qd_out",
            scene="free_limited_hinge_zero_g",
            q_weight=[0.0] * 8,
            qd_weight=[0.2, -0.1, 0.15, 0.05, -0.03, 0.04, 1.0],
            groups={
                "root_pos_q": ("q", [0, 1, 2]),
                "root_quat_q_raw": ("q", [3, 4, 5, 6]),
                "joint_q": ("q", [7]),
                "root_qd": ("qd", [0, 1, 2, 3, 4, 5]),
                "joint_qd": ("qd", [6]),
                "joint_force": ("joint_f", [6]),
            },
        ),
        DiagnosticCase(
            name="double_limited_hinge_qd_out",
            scene="double_limited_hinge_zero_g",
            q_weight=[0.0, 0.0],
            qd_weight=[0.8, -0.6],
            groups={
                "joint_q": ("q", [0, 1]),
                "joint_qd": ("qd", [0, 1]),
                "joint_force": ("joint_f", [0, 1]),
            },
        ),
        DiagnosticCase(
            name="two_link_qd_out",
            scene="double_hinge_gravity",
            q_weight=[0.0, 0.0],
            qd_weight=[0.8, -0.6],
            groups={
                "joint_q": ("q", [0, 1]),
                "joint_qd": ("qd", [0, 1]),
                "joint_force": ("joint_f", [0, 1]),
            },
        ),
        DiagnosticCase(
            name="two_link_gravity_static_qd_out",
            scene="double_hinge_gravity_static",
            q_weight=[0.0, 0.0],
            qd_weight=[0.8, -0.6],
            groups={
                "joint_q": ("q", [0, 1]),
                "joint_qd": ("qd", [0, 1]),
                "joint_force": ("joint_f", [0, 1]),
            },
        ),
        DiagnosticCase(
            name="two_link_zero_g_dynamic_qd_out",
            scene="double_hinge_zero_g_dynamic",
            q_weight=[0.0, 0.0],
            qd_weight=[0.8, -0.6],
            groups={
                "joint_q": ("q", [0, 1]),
                "joint_qd": ("qd", [0, 1]),
                "joint_force": ("joint_f", [0, 1]),
            },
        ),
        DiagnosticCase(
            name="two_link_zero_g_forced_qd_out",
            scene="double_hinge_zero_g_forced",
            q_weight=[0.0, 0.0],
            qd_weight=[0.8, -0.6],
            groups={
                "joint_q": ("q", [0, 1]),
                "joint_qd": ("qd", [0, 1]),
                "joint_force": ("joint_f", [0, 1]),
            },
        ),
        DiagnosticCase(
            name="planar_chain_forced_qd_out",
            scene="planar_chain_zero_g",
            q_weight=[0.0] * 6,
            qd_weight=[1.0, -0.1, 0.35, 0.5, -0.3, 0.2],
            groups={
                "root_planar_q": ("q", [0, 1, 2]),
                "joint_q": ("q", [3, 4, 5]),
                "root_planar_qd": ("qd", [0, 1, 2]),
                "joint_qd": ("qd", [3, 4, 5]),
                "joint_force": ("joint_f", [3, 4, 5]),
            },
        ),
        DiagnosticCase(
            name="planar_branch_forced_qd_out",
            scene="planar_branch_zero_g",
            q_weight=[0.0] * 9,
            qd_weight=[1.0, -0.1, 0.35, 0.5, -0.3, 0.2, -0.4, 0.25, -0.15],
            groups={
                "root_planar_q": ("q", [0, 1, 2]),
                "front_joint_q": ("q", [3, 4, 5]),
                "rear_joint_q": ("q", [6, 7, 8]),
                "root_planar_qd": ("qd", [0, 1, 2]),
                "joint_qd": ("qd", [3, 4, 5, 6, 7, 8]),
                "joint_force": ("joint_f", [3, 4, 5, 6, 7, 8]),
            },
        ),
        DiagnosticCase(
            name="free_position_out",
            scene="free_body_zero_g",
            q_weight=[1.0, -0.7, 0.4, 0.0, 0.0, 0.0, 0.0],
            qd_weight=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            groups={
                "root_pos_q": ("q", [0, 1, 2]),
                "root_quat_q_raw": ("q", [3, 4, 5, 6]),
                "root_linear_qd": ("qd", [0, 1, 2]),
                "root_angular_qd": ("qd", [3, 4, 5]),
                "root_force": ("joint_f", [0, 1, 2, 3, 4, 5]),
            },
        ),
        DiagnosticCase(
            name="free_quaternion_out_raw_fd",
            scene="free_body_zero_g",
            q_weight=[0.0, 0.0, 0.0, 0.6, -0.4, 0.2, 0.9],
            qd_weight=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            groups={
                "root_pos_q": ("q", [0, 1, 2]),
                "root_quat_q_raw": ("q", [3, 4, 5, 6]),
                "root_linear_qd": ("qd", [0, 1, 2]),
                "root_angular_qd": ("qd", [3, 4, 5]),
                "root_force": ("joint_f", [0, 1, 2, 3, 4, 5]),
            },
        ),
        DiagnosticCase(
            name="free_quaternion_out_normalized_fd",
            scene="free_body_zero_g",
            q_weight=[0.0, 0.0, 0.0, 0.6, -0.4, 0.2, 0.9],
            qd_weight=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            groups={
                "root_pos_q": ("q", [0, 1, 2]),
                "root_quat_q_normalized": ("q", [3, 4, 5, 6]),
                "root_linear_qd": ("qd", [0, 1, 2]),
                "root_angular_qd": ("qd", [3, 4, 5]),
                "root_force": ("joint_f", [0, 1, 2, 3, 4, 5]),
            },
            normalize_quat_for_fd=True,
        ),
        DiagnosticCase(
            name="free_velocity_out",
            scene="free_body_zero_g",
            q_weight=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            qd_weight=[0.3, -0.5, 0.7, -0.2, 0.4, -0.6],
            groups={
                "root_pos_q": ("q", [0, 1, 2]),
                "root_quat_q_raw": ("q", [3, 4, 5, 6]),
                "root_linear_qd": ("qd", [0, 1, 2]),
                "root_angular_qd": ("qd", [3, 4, 5]),
                "root_force": ("joint_f", [0, 1, 2, 3, 4, 5]),
            },
        ),
    ]


def normalize_if_needed(case: DiagnosticCase, q: torch.Tensor) -> torch.Tensor:
    if case.normalize_quat_for_fd and q.shape[-1] >= 7:
        return torch.cat([q[:, :3], torch.nn.functional.normalize(q[:, 3:7], dim=-1), q[:, 7:]], dim=-1)
    return q


def evaluate_loss(
    env: MinimalSolverEnv,
    case: DiagnosticCase,
    q: torch.Tensor,
    qd: torch.Tensor,
    joint_f: torch.Tensor,
) -> torch.Tensor:
    q = normalize_if_needed(case, q)
    q_out, qd_out = env.step(q, qd, joint_f)
    q_weight = torch.tensor([case.q_weight], dtype=torch.float32, device=q.device)
    qd_weight = torch.tensor([case.qd_weight], dtype=torch.float32, device=q.device)
    return (q_out * q_weight).sum() + (qd_out * qd_weight).sum()


def finite_difference_group(
    env: MinimalSolverEnv,
    case: DiagnosticCase,
    base_q: torch.Tensor,
    base_qd: torch.Tensor,
    base_f: torch.Tensor,
    variable: str,
    indices: list[int],
    eps: float,
) -> torch.Tensor:
    values = []
    for index in indices:
        q_pos = base_q.detach().clone()
        q_neg = base_q.detach().clone()
        qd_pos = base_qd.detach().clone()
        qd_neg = base_qd.detach().clone()
        f_pos = base_f.detach().clone()
        f_neg = base_f.detach().clone()
        if variable == "q":
            q_pos[0, index] += eps
            q_neg[0, index] -= eps
        elif variable == "qd":
            qd_pos[0, index] += eps
            qd_neg[0, index] -= eps
        elif variable == "joint_f":
            f_pos[0, index] += eps
            f_neg[0, index] -= eps
        else:
            raise ValueError(f"unknown variable: {variable}")

        with torch.no_grad():
            plus = evaluate_loss(env, case, q_pos, qd_pos, f_pos)
            minus = evaluate_loss(env, case, q_neg, qd_neg, f_neg)
        values.append(float(((plus - minus) / (2.0 * eps)).detach().cpu()))
    return torch.tensor(values, dtype=torch.float64)


def expected_group_grad(case: DiagnosticCase, group: str, dt: float) -> list[float] | None:
    if case.name == "hinge_q_out":
        if group == "joint_q":
            return [1.0]
        if group == "joint_qd":
            return [dt]
        if group == "joint_force":
            return [dt * dt]
    if case.name == "hinge_qd_out":
        if group == "joint_q":
            return [0.0]
        if group == "joint_qd":
            return [1.0]
    if case.name == "free_position_out":
        weights = [1.0, -0.7, 0.4]
        if group == "root_pos_q":
            return weights
        if group == "root_linear_qd":
            return [dt * v for v in weights]
        if group in {"root_quat_q_raw", "root_angular_qd"}:
            return [0.0] * (4 if "quat" in group else 3)
    if case.name == "free_velocity_out":
        weights = [0.3, -0.5, 0.7, -0.2, 0.4, -0.6]
        if group == "root_linear_qd":
            return weights[:3]
        if group == "root_angular_qd":
            return weights[3:]
        if group == "root_pos_q":
            return [0.0, 0.0, 0.0]
    return None


def run_case(env: MinimalSolverEnv, case: DiagnosticCase, epsilons: list[float]) -> dict:
    base_q, base_qd, base_f = env.base_state()
    q_req = base_q.detach().clone().requires_grad_(True)
    qd_req = base_qd.detach().clone().requires_grad_(True)
    f_req = base_f.detach().clone().requires_grad_(True)
    loss = evaluate_loss(env, case, q_req, qd_req, f_req)
    loss.backward()

    grads = {
        "q": clean_grad(q_req.grad, q_req).reshape(-1).to(torch.float64),
        "qd": clean_grad(qd_req.grad, qd_req).reshape(-1).to(torch.float64),
        "joint_f": clean_grad(f_req.grad, f_req).reshape(-1).to(torch.float64),
    }

    groups = {}
    for group_name, (variable, indices) in case.groups.items():
        analytic = grads[variable][indices].detach().cpu()
        rows = []
        best = None
        for eps in epsilons:
            fd = finite_difference_group(env, case, base_q, base_qd, base_f, variable, indices, eps)
            diff = analytic - fd
            denom = max(float(analytic.norm()), float(fd.norm()), 1.0e-12)
            rel = float(diff.norm()) / denom
            row = {
                "epsilon": eps,
                "analytic": [finite_float(v) for v in analytic.tolist()],
                "finite_difference": [finite_float(v) for v in fd.tolist()],
                "abs_error_norm": finite_float(float(diff.norm())),
                "relative_error": finite_float(rel),
            }
            rows.append(row)
            if best is None or rel < best["relative_error"]:
                best = row
        expected = expected_group_grad(case, group_name, env.dt)
        groups[group_name] = {
            "variable": variable,
            "indices": indices,
            "expected_simple_dynamics": expected,
            "analytic_norm": finite_float(float(analytic.norm())),
            "best_relative_error": finite_float(best["relative_error"] if best else None),
            "best_epsilon": finite_float(best["epsilon"] if best else None),
            "epsilon_sweep": rows,
        }

    with torch.no_grad():
        q_out, qd_out = env.step(base_q, base_qd, base_f)

    return {
        "case": case.name,
        "scene": case.scene,
        "loss": finite_float(float(loss.detach().cpu())),
        "normalize_quat_for_fd": case.normalize_quat_for_fd,
        "base_q": [finite_float(v) for v in base_q.reshape(-1).detach().cpu().tolist()],
        "base_qd": [finite_float(v) for v in base_qd.reshape(-1).detach().cpu().tolist()],
        "base_joint_f": [finite_float(v) for v in base_f.reshape(-1).detach().cpu().tolist()],
        "q_out": [finite_float(v) for v in q_out.reshape(-1).detach().cpu().tolist()],
        "qd_out": [finite_float(v) for v in qd_out.reshape(-1).detach().cpu().tolist()],
        "groups": groups,
    }


def render_scene_video(
    env: MinimalSolverEnv,
    viewer: newton.viewer.ViewerGL,
    out_dir: Path,
    *,
    seconds: float,
    fps: int,
    width: int,
    height: int,
) -> dict:
    import imageio.v2 as imageio

    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / f"{env.scene}.mp4"
    poster_path = out_dir / f"{env.scene}_poster.png"
    frame_count = max(1, int(round(seconds * fps)))

    viewer.set_model(env.model)
    follow_camera = SmoothedFollowCamera(env.scene, env.dt)

    q, qd, joint_f = env.base_state()
    frames = []
    with imageio.get_writer(video_path, fps=fps, codec="libx264", quality=8) as writer:
        with torch.no_grad():
            for frame_idx in range(frame_count):
                follow_camera.update(viewer, q)
                state = env.make_viewer_state(q, qd)
                viewer.begin_frame(frame_idx / float(fps))
                viewer.log_state(state)
                viewer.end_frame()
                frame = viewer.get_frame().numpy()
                frames.append(frame)
                writer.append_data(frame)

                q_next = torch.empty_like(q)
                qd_next = torch.empty_like(qd)
                env.step_warp(q, qd, joint_f, q_next, qd_next, requires_grad=False)
                q, qd = q_next, qd_next
    imageio.imwrite(poster_path, frames[len(frames) // 2])
    return {
        "scene": env.scene,
        "video": str(video_path.relative_to(Path(__file__).resolve().parent)),
        "poster": str(poster_path.relative_to(Path(__file__).resolve().parent)),
        "seconds": seconds,
        "fps": fps,
        "frames": frame_count,
        "source": "ViewerGL.get_frame()",
        "overlays": False,
        "camera": "SmoothedFollowCamera",
    }


def render_diagnostic_videos(args: argparse.Namespace) -> list[dict]:
    scenes = [
        "single_hinge_gravity",
        "double_hinge_gravity_static",
        "double_hinge_zero_g_forced",
        "planar_chain_zero_g",
        "planar_branch_zero_g",
        "free_body_zero_g",
    ]
    out_dir = Path(args.video_dir)
    videos = []
    viewer = newton.viewer.ViewerGL(width=args.video_width, height=args.video_height, headless=True)
    viewer.show_static = True
    viewer.show_collision = True
    try:
        for scene in scenes:
            env = MinimalSolverEnv(
                scene,
                args.device,
                args.dt,
                solver_iterations=args.solver_iterations,
                ls_iterations=args.ls_iterations,
            )
            videos.append(
                render_scene_video(
                    env,
                    viewer,
                    out_dir,
                    seconds=args.video_seconds,
                    fps=args.video_fps,
                    width=args.video_width,
                    height=args.video_height,
                )
            )
    finally:
        viewer.close()
    return videos


def run(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    wp.init()

    epsilons = args.eps or list(DEFAULT_EPS)
    envs: dict[str, MinimalSolverEnv] = {}
    cases = make_cases(args.dt)
    if args.cases:
        wanted = set(args.cases)
        cases = [case for case in cases if case.name in wanted]
        missing = wanted.difference(case.name for case in cases)
        if missing:
            raise ValueError(f"unknown diagnostic case(s): {sorted(missing)}")
    results = []
    for case in cases:
        if case.scene not in envs:
            envs[case.scene] = MinimalSolverEnv(
                case.scene,
                args.device,
                args.dt,
                solver_iterations=args.solver_iterations,
                ls_iterations=args.ls_iterations,
            )
        results.append(run_case(envs[case.scene], case, epsilons))

    result = {
        "mode": "solver_gradient_diagnostics",
        "title": "SHAC with MuJoCo Warp",
        "timestamp_pacific": pacific_now_iso(),
        "device": args.device,
        "dt": args.dt,
        "solver_iterations": args.solver_iterations,
        "ls_iterations": args.ls_iterations,
        "epsilon_values": epsilons,
        "cases": results,
        "notes": [
            "All scenes run with SolverMuJoCo, MJWarp backend, contacts disabled, and Euler integration.",
            "Finite differences perturb each input component independently; the normalized quaternion case re-normalizes q[3:7] before each evaluation.",
        ],
    }
    if args.render_videos:
        result["videos"] = render_diagnostic_videos(args)

    out_path = Path(args.out)
    write_json(out_path, result)
    print(f"wrote solver diagnostics to {out_path}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "assets" / "solver_gradient_diagnostics" / "solver_gradient_diagnostics.json"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--solver-iterations", type=int, default=4)
    parser.add_argument("--ls-iterations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eps", type=float, nargs="*")
    parser.add_argument("--cases", nargs="*", help="Optional subset of diagnostic case names to run.")
    parser.add_argument("--render-videos", action="store_true")
    parser.add_argument(
        "--video-dir",
        default=str(Path(__file__).resolve().parent / "assets" / "solver_gradient_diagnostics" / "videos"),
    )
    parser.add_argument("--video-seconds", type=float, default=4.0)
    parser.add_argument("--video-fps", type=int, default=60)
    parser.add_argument("--video-width", type=int, default=960)
    parser.add_argument("--video-height", type=int, default=544)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
