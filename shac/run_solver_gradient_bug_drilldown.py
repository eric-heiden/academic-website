#!/usr/bin/env python3
"""Intermediate MJWarp gradient drilldown for the SHAC report."""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import warp as wp

from run_solver_gradient_diagnostics import DEFAULT_EPS, MinimalSolverEnv, finite_float, pacific_now_iso, write_json
from mujoco_warp._src.types import vec10


@wp.kernel
def _weighted_vector_sum(
    values: wp.array2d[float],
    weights: wp.array[float],
    out: wp.array[float],
):
    i = wp.tid()
    wp.atomic_add(out, 0, values[0, i] * weights[i])


@wp.kernel
def _weighted_dense_matrix_sum(
    values: wp.array3d[float],
    weights: wp.array2d[float],
    out: wp.array[float],
):
    row, col = wp.tid()
    wp.atomic_add(out, 0, values[0, row, col] * weights[row, col])


@wp.kernel
def _weighted_spatial_sum(
    values: wp.array2d[wp.spatial_vector],
    weights: wp.array2d[float],
    out: wp.array[float],
):
    i = wp.tid()
    value = values[0, i]
    angular = wp.spatial_top(value)
    linear = wp.spatial_bottom(value)
    angular_weight = wp.vec3(weights[i, 0], weights[i, 1], weights[i, 2])
    linear_weight = wp.vec3(weights[i, 3], weights[i, 4], weights[i, 5])
    wp.atomic_add(out, 0, wp.dot(angular, angular_weight) + wp.dot(linear, linear_weight))


@wp.kernel
def _weighted_vec10_sum(
    values: wp.array2d[vec10],
    weights: wp.array2d[float],
    out: wp.array[float],
):
    row = wp.tid()
    value = values[0, row]
    total = (
        value[0] * weights[row, 0]
        + value[1] * weights[row, 1]
        + value[2] * weights[row, 2]
        + value[3] * weights[row, 3]
        + value[4] * weights[row, 4]
        + value[5] * weights[row, 5]
        + value[6] * weights[row, 6]
        + value[7] * weights[row, 7]
        + value[8] * weights[row, 8]
        + value[9] * weights[row, 9]
    )
    wp.atomic_add(out, 0, total)


@dataclass
class IntermediateCheck:
    name: str
    scene: str
    target: str
    weights: list[float] | list[list[float]] | None
    note: str
    weight_mode: str = "literal"


def make_checks() -> list[IntermediateCheck]:
    return [
        IntermediateCheck(
            name="dense_mass_offdiag",
            scene="double_hinge_zero_g_forced",
            target="M",
            weights=[[0.0, 1.0], [1.0, 0.0]],
            note="Direct <M(q), G> probe for the two dense off-diagonal mass entries.",
        ),
        IntermediateCheck(
            name="gravity_bias_force",
            scene="double_hinge_gravity_static",
            target="qfrc_bias",
            weights=[0.8, -0.6],
            note="Direct qfrc_bias(q) probe for the gravity-only two-link scene.",
        ),
        IntermediateCheck(
            name="gravity_smooth_force",
            scene="double_hinge_gravity_static",
            target="qfrc_smooth",
            weights=[0.8, -0.6],
            note="Direct qfrc_smooth(q) probe after qfrc_passive - qfrc_bias assembly.",
        ),
        IntermediateCheck(
            name="gravity_cdof_fixed_cfrc",
            scene="double_hinge_gravity_static",
            target="cdof",
            weights=None,
            note="Product-rule probe: differentiate cdof(q) while holding base cfrc_int fixed.",
            weight_mode="cdof_from_bias_weights",
        ),
        IntermediateCheck(
            name="gravity_cfrc_fixed_cdof",
            scene="double_hinge_gravity_static",
            target="cfrc_int",
            weights=None,
            note="Product-rule probe: differentiate accumulated cfrc_int(q) while holding base cdof fixed.",
            weight_mode="cfrc_from_bias_weights",
        ),
        IntermediateCheck(
            name="gravity_cfrc_body2_fixed_cdof",
            scene="double_hinge_gravity_static",
            target="cfrc_int",
            weights=None,
            note="Body-2-only cfrc_int probe; this bypasses parent accumulation in the reported loss.",
            weight_mode="cfrc_body2_from_bias_weights",
        ),
        IntermediateCheck(
            name="gravity_cinert_probe",
            scene="double_hinge_gravity_static",
            target="cinert",
            weights=[
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.2, -0.3, 0.5, 0.7, -0.4, 0.6, 0.1, -0.2, 0.3, 0.0],
                [-0.5, 0.4, -0.1, 0.2, 0.3, -0.7, 0.6, -0.1, 0.2, 0.0],
            ],
            note="Direct fixed-weight probe for cinert(q), upstream of RNE force construction.",
        ),
        IntermediateCheck(
            name="single_pendulum_bias_force",
            scene="single_hinge_gravity",
            target="qfrc_bias",
            weights=[1.0],
            note="Single-DOF gravity bias-force control case that should remain accurate.",
        ),
    ]


def _base_intermediates(
    env: MinimalSolverEnv,
    q: torch.Tensor,
    qd: torch.Tensor,
    joint_f: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q_out = torch.empty_like(q)
    qd_out = torch.empty_like(qd)
    env.step_warp(q, qd, joint_f, q_out, qd_out, requires_grad=False)
    cdof = np.asarray(env.solver.mjw_data.cdof.numpy()[0], dtype=np.float32)
    cfrc = np.asarray(env.solver.mjw_data.cfrc_int.numpy()[0], dtype=np.float32)
    dof_bodyid = np.asarray(env.solver.mjw_model.dof_bodyid.numpy(), dtype=np.int32)
    return cdof, cfrc, dof_bodyid


def _resolved_check(
    env: MinimalSolverEnv,
    check: IntermediateCheck,
    base_q: torch.Tensor,
    base_qd: torch.Tensor,
    base_f: torch.Tensor,
) -> IntermediateCheck:
    if check.weight_mode == "literal":
        return check

    bias_weights = np.asarray([0.8, -0.6], dtype=np.float32)
    cdof, cfrc, dof_bodyid = _base_intermediates(env, base_q, base_qd, base_f)

    if check.weight_mode == "cdof_from_bias_weights":
        weights = np.zeros((env.qd_dim, 6), dtype=np.float32)
        for dofid in range(env.qd_dim):
            weights[dofid] = bias_weights[dofid] * cfrc[dof_bodyid[dofid]]
        return IntermediateCheck(check.name, check.scene, check.target, weights.tolist(), check.note)

    if check.weight_mode == "cfrc_from_bias_weights":
        weights = np.zeros((cfrc.shape[0], 6), dtype=np.float32)
        for dofid in range(env.qd_dim):
            weights[dof_bodyid[dofid]] += bias_weights[dofid] * cdof[dofid]
        return IntermediateCheck(check.name, check.scene, check.target, weights.tolist(), check.note)

    if check.weight_mode == "cfrc_body2_from_bias_weights":
        weights = np.zeros((cfrc.shape[0], 6), dtype=np.float32)
        child_dof = env.qd_dim - 1
        weights[dof_bodyid[child_dof]] = bias_weights[child_dof] * cdof[child_dof]
        return IntermediateCheck(check.name, check.scene, check.target, weights.tolist(), check.note)

    raise ValueError(f"unknown weight mode: {check.weight_mode}")


def _run_step_loss(
    env: MinimalSolverEnv,
    q: torch.Tensor,
    qd: torch.Tensor,
    joint_f: torch.Tensor,
    check: IntermediateCheck,
    *,
    requires_grad: bool,
) -> tuple[float, wp.array | None, wp.array]:
    q_out = torch.empty_like(q)
    qd_out = torch.empty_like(qd)

    env.zero_solver_buffers()
    state_in = env.model.state(requires_grad=requires_grad)
    state_out = env.model.state(requires_grad=requires_grad)
    control = env.model.control(requires_grad=requires_grad)

    q_wp = wp.from_torch(
        q.contiguous().view(-1), dtype=wp.float32, requires_grad=requires_grad, retain_grad=requires_grad
    )
    qd_wp = wp.from_torch(
        qd.contiguous().view(-1), dtype=wp.float32, requires_grad=requires_grad, retain_grad=requires_grad
    )
    f_wp = wp.from_torch(
        joint_f.contiguous().view(-1), dtype=wp.float32, requires_grad=requires_grad, retain_grad=requires_grad
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
    env.solver.step(state_in, state_out, control, None, env.dt)

    loss = wp.zeros(1, dtype=float, requires_grad=requires_grad, device=env.wp_device)
    if check.target == "M":
        if check.weights is None:
            raise ValueError(f"{check.name} has no weights")
        weights = wp.array(check.weights, dtype=float, device=env.wp_device)
        wp.launch(
            _weighted_dense_matrix_sum,
            dim=(env.qd_dim, env.qd_dim),
            inputs=[env.solver.mjw_data.M, weights],
            outputs=[loss],
        )
    elif check.target in {"cdof", "cfrc_int", "cacc"}:
        if check.weights is None:
            raise ValueError(f"{check.name} has no weights")
        weights = wp.array(check.weights, dtype=float, device=env.wp_device)
        values = getattr(env.solver.mjw_data, check.target)
        wp.launch(_weighted_spatial_sum, dim=len(check.weights), inputs=[values, weights], outputs=[loss])
    elif check.target == "cinert":
        if check.weights is None:
            raise ValueError(f"{check.name} has no weights")
        weights = wp.array(check.weights, dtype=float, device=env.wp_device)
        wp.launch(
            _weighted_vec10_sum,
            dim=len(check.weights),
            inputs=[env.solver.mjw_data.cinert, weights],
            outputs=[loss],
        )
    else:
        if check.weights is None:
            raise ValueError(f"{check.name} has no weights")
        weights = wp.array(check.weights, dtype=float, device=env.wp_device)
        values = getattr(env.solver.mjw_data, check.target)
        wp.launch(_weighted_vector_sum, dim=env.qd_dim, inputs=[values, weights], outputs=[loss])

    wp.synchronize()
    return float(loss.numpy()[0]), q_wp, loss


def _finite_difference(
    env: MinimalSolverEnv,
    check: IntermediateCheck,
    base_q: torch.Tensor,
    base_qd: torch.Tensor,
    base_f: torch.Tensor,
    eps: float,
) -> list[float]:
    values = []
    for index in range(env.q_dim):
        q_pos = base_q.detach().clone()
        q_neg = base_q.detach().clone()
        q_pos[0, index] += eps
        q_neg[0, index] -= eps
        plus, _, _ = _run_step_loss(env, q_pos, base_qd, base_f, check, requires_grad=False)
        minus, _, _ = _run_step_loss(env, q_neg, base_qd, base_f, check, requires_grad=False)
        values.append((plus - minus) / (2.0 * eps))
    return values


def run_check(env: MinimalSolverEnv, check: IntermediateCheck, epsilons: list[float]) -> dict:
    base_q, base_qd, base_f = env.base_state()
    check = _resolved_check(env, check, base_q, base_qd, base_f)
    with wp.Tape() as tape:
        loss_value, q_wp, loss = _run_step_loss(
            env,
            base_q.detach().clone(),
            base_qd.detach().clone(),
            base_f.detach().clone(),
            check,
            requires_grad=True,
        )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Running the tape backwards may produce incorrect gradients.*",
            category=UserWarning,
        )
        tape.backward(grads={loss: wp.ones_like(loss)})

    grad = tape.gradients.get(q_wp)
    analytic = [0.0 for _ in range(env.q_dim)] if grad is None else wp.to_torch(grad).detach().cpu().reshape(-1).tolist()
    analytic_tensor = torch.tensor(analytic, dtype=torch.float64)

    rows = []
    best = None
    for eps in epsilons:
        fd = _finite_difference(env, check, base_q, base_qd, base_f, eps)
        fd_tensor = torch.tensor(fd, dtype=torch.float64)
        diff = analytic_tensor - fd_tensor
        denom = max(float(analytic_tensor.norm()), float(fd_tensor.norm()), 1.0e-12)
        rel = float(diff.norm()) / denom
        row = {
            "epsilon": eps,
            "analytic": [finite_float(v) for v in analytic],
            "finite_difference": [finite_float(v) for v in fd],
            "abs_error_norm": finite_float(float(diff.norm())),
            "relative_error": finite_float(rel),
        }
        rows.append(row)
        if best is None or rel < best["relative_error"]:
            best = row

    return {
        "check": check.name,
        "scene": check.scene,
        "target": check.target,
        "note": check.note,
        "loss": finite_float(loss_value),
        "base_q": [finite_float(v) for v in base_q.reshape(-1).detach().cpu().tolist()],
        "base_qd": [finite_float(v) for v in base_qd.reshape(-1).detach().cpu().tolist()],
        "best_relative_error": finite_float(best["relative_error"]),
        "best_epsilon": finite_float(best["epsilon"]),
        "epsilon_sweep": rows,
    }


def run(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    wp.init()

    epsilons = args.eps or list(DEFAULT_EPS)
    envs: dict[str, MinimalSolverEnv] = {}
    results = []
    for check in make_checks():
        if check.scene not in envs:
            envs[check.scene] = MinimalSolverEnv(check.scene, args.device, args.dt)
        results.append(run_check(envs[check.scene], check, epsilons))

    result = {
        "mode": "solver_gradient_bug_drilldown",
        "timestamp_pacific": pacific_now_iso(),
        "device": args.device,
        "dt": args.dt,
        "epsilon_values": epsilons,
        "checks": results,
        "notes": [
            "Targets are intermediate MJWarp arrays captured inside one SolverMuJoCo step.",
            "The scalar loss is a fixed weighted sum of the target array; finite differences perturb q only.",
        ],
    }
    out_path = Path(args.out)
    write_json(out_path, result)
    print(f"wrote solver bug drilldown to {out_path}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(
            Path(__file__).resolve().parent
            / "assets"
            / "solver_gradient_diagnostics"
            / "solver_gradient_bug_drilldown.json"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--eps", type=float, nargs="*")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
