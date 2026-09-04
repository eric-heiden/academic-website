#!/usr/bin/env python3
"""Finite-difference gate for MJWarp PR #1535's analytic step adjoint.

This probe deliberately uses MJWarp directly.  The PR records its analytic
backward only for ``step(model, data_in, data_out)``; Newton's current adapter
uses the in-place overload and therefore cannot exercise this code path.

The default Ant model is vendored next to this script.  The Humanoid model is
the one bundled with the exact PR checkout.  Run this script once in the PR's
frozen uv environment and, optionally, once in the current Newton environment.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

HERE = Path(__file__).resolve().parent
DEFAULT_PR_ROOT = Path("/home/horde/repos/mujoco_warp-pr1535")
EXPECTED_PR_HEAD = "02d09b139fdf091e1e859d7f41c47a8f71d30574"


def git_value(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bind_state(
    data: Any, qpos: np.ndarray, qvel: np.ndarray, ctrl: np.ndarray, *, grad: bool
) -> None:
    data.qpos = wp.array(qpos[None], dtype=float, requires_grad=grad)
    data.qvel = wp.array(qvel[None], dtype=float, requires_grad=grad)
    data.ctrl = wp.array(ctrl[None], dtype=float, requires_grad=grad)
    # Warm-start acceleration is intentionally outside the differentiable state.
    data.qacc_warmstart.zero_()


def configure_model(
    path: Path, name: str
) -> tuple[mujoco.MjModel, mujoco.MjData, np.ndarray, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(str(path))
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_EULERDAMP
    model.opt.iterations = 20
    data = mujoco.MjData(model)

    if name == "ant":
        numeric_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, "init_qpos")
        if numeric_id < 0:
            raise RuntimeError("Ant model has no init_qpos numeric")
        start = model.numeric_adr[numeric_id]
        stop = start + model.numeric_size[numeric_id]
        data.qpos[:] = model.numeric_data[start:stop]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
    elif name == "humanoid":
        # A few CPU steps establish the nominal eight-contact standing state.
        for _ in range(10):
            mujoco.mj_step(model, data)
    else:
        raise ValueError(name)
    return (
        model,
        data,
        data.qpos.astype(np.float32).copy(),
        data.qvel.astype(np.float32).copy(),
    )


def compare_objective(
    *,
    name: str,
    device_model: Any,
    model: mujoco.MjModel,
    data_in: Any,
    data_out: Any,
    backward_workspace: Any,
    qpos: np.ndarray,
    qvel: np.ndarray,
    ctrl: np.ndarray,
    seed: np.ndarray,
    directions: np.ndarray,
    epsilon: float,
) -> dict[str, Any]:
    nq, nv = model.nq, model.nv

    def forward(test_ctrl: np.ndarray) -> tuple[np.ndarray, int]:
        bind_state(data_in, qpos, qvel, test_ctrl.astype(np.float32), grad=False)
        mjw.step(device_model, data_in, data_out)
        wp.synchronize()
        state = np.concatenate((data_out.qpos.numpy()[0], data_out.qvel.numpy()[0]))
        return state, int(data_out.nacon.numpy()[0])

    bind_state(data_in, qpos, qvel, ctrl, grad=True)
    data_out.qpos.requires_grad = True
    data_out.qvel.requires_grad = True
    for array in (
        data_in.qpos,
        data_in.qvel,
        data_in.ctrl,
        data_out.qpos,
        data_out.qvel,
    ):
        if array.grad is not None:
            array.grad.zero_()

    started = time.perf_counter()
    with wp.Tape() as tape:
        mjw.step(device_model, data_in, data_out)
    seeds = {
        data_out.qpos: wp.array(seed[:nq][None], dtype=float),
        data_out.qvel: wp.array(seed[nq : nq + nv][None], dtype=float),
    }
    with mjw.backward_context(backward_workspace):
        tape.backward(grads=seeds)
    wp.synchronize()
    analytic_seconds = time.perf_counter() - started
    analytic_gradient = data_in.ctrl.grad.numpy()[0].copy()

    comparisons = []
    fd_started = time.perf_counter()
    for direction in directions:
        plus, plus_contacts = forward(ctrl + epsilon * direction)
        minus, minus_contacts = forward(ctrl - epsilon * direction)
        finite_difference = float(seed @ (plus - minus) / (2.0 * epsilon))
        analytic = float(analytic_gradient @ direction)
        absolute_error = abs(analytic - finite_difference)
        relative_error = absolute_error / max(
            abs(analytic), abs(finite_difference), 1.0e-10
        )
        comparisons.append(
            {
                "analytic": analytic,
                "finite_difference": finite_difference,
                "absolute_error": absolute_error,
                "relative_error": relative_error,
                "plus_contacts": plus_contacts,
                "minus_contacts": minus_contacts,
            }
        )
    wp.synchronize()
    finite_difference_seconds = time.perf_counter() - fd_started
    errors = np.asarray([item["relative_error"] for item in comparisons])
    return {
        "name": name,
        "epsilon": epsilon,
        "directions": len(comparisons),
        "all_finite": bool(
            np.isfinite(analytic_gradient).all()
            and all(math.isfinite(item["finite_difference"]) for item in comparisons)
        ),
        "control_gradient_norm": float(np.linalg.norm(analytic_gradient)),
        "median_relative_error": float(np.median(errors)),
        "max_relative_error": float(np.max(errors)),
        "comparisons": comparisons,
        "timing_seconds": {
            "analytic_forward_and_backward": analytic_seconds,
            "all_central_differences": finite_difference_seconds,
        },
    }


def probe_model(name: str, path: Path, args: argparse.Namespace) -> dict[str, Any]:
    model, host_data, qpos, qvel = configure_model(path, name)
    control = np.zeros(model.nu, dtype=np.float32)
    rng = np.random.default_rng(args.seed + (0 if name == "ant" else 1000))
    directions = rng.normal(size=(args.directions, model.nu)).astype(np.float32)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    started = time.perf_counter()
    device_model = mjw.put_model(model)
    kwargs = {"nworld": 1, "nconmax": args.nconmax, "njmax": args.njmax}
    data_in = mjw.put_data(model, host_data, **kwargs)
    data_out = mjw.put_data(model, host_data, **kwargs)
    backward_workspace = mjw.create_backward_context(device_model, data_in)
    wp.synchronize()
    setup_seconds = time.perf_counter() - started

    # Compile and record the nominal contact count before timing the objectives.
    bind_state(data_in, qpos, qvel, control, grad=False)
    started = time.perf_counter()
    mjw.step(device_model, data_in, data_out)
    wp.synchronize()
    first_forward_seconds = time.perf_counter() - started
    nominal_contacts = int(data_out.nacon.numpy()[0])

    mixed_seed = rng.normal(size=model.nq + model.nv).astype(np.float32)
    mixed_seed /= np.linalg.norm(mixed_seed)
    velocity_seed = np.zeros(model.nq + model.nv, dtype=np.float32)
    velocity_seed[model.nq] = 1.0
    objectives = [
        compare_objective(
            name="random mixed next-state projection",
            device_model=device_model,
            model=model,
            data_in=data_in,
            data_out=data_out,
            backward_workspace=backward_workspace,
            qpos=qpos,
            qvel=qvel,
            ctrl=control,
            seed=mixed_seed,
            directions=directions,
            epsilon=args.epsilon,
        ),
        compare_objective(
            name="next root x velocity",
            device_model=device_model,
            model=model,
            data_in=data_in,
            data_out=data_out,
            backward_workspace=backward_workspace,
            qpos=qpos,
            qvel=qvel,
            ctrl=control,
            seed=velocity_seed,
            directions=directions,
            epsilon=args.epsilon,
        ),
    ]
    passes = all(
        objective["all_finite"]
        and objective["max_relative_error"] <= args.max_relative_error
        for objective in objectives
    )
    return {
        "result": "pass" if passes else "fail",
        "criterion": {"maximum_directional_relative_error": args.max_relative_error},
        "model": {
            "name": name,
            "xml": str(path.resolve()),
            "xml_sha256": sha256(path),
            "initialization": (
                "model init_qpos numeric, then mj_forward"
                if name == "ant"
                else "model qpos0 after 10 CPU mj_step settling steps"
            ),
            "nq": model.nq,
            "nv": model.nv,
            "nu": model.nu,
            "nbody": model.nbody,
            "ngeom": model.ngeom,
            "njnt": model.njnt,
            "ntendon": model.ntendon,
            "neq": model.neq,
            "nflex": model.nflex,
            "nominal_contacts": nominal_contacts,
            "timestep": float(model.opt.timestep),
            "integrator": mujoco.mjtIntegrator(model.opt.integrator).name,
            "cone": mujoco.mjtCone(model.opt.cone).name,
            "solver": mujoco.mjtSolver(model.opt.solver).name,
            "solver_iterations": int(model.opt.iterations),
            "euler_damping_disabled": bool(
                model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_EULERDAMP
            ),
        },
        "objectives": objectives,
        "timing_seconds": {
            "model_data_and_backward_workspace": setup_seconds,
            "first_forward_in_process": first_forward_seconds,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr-root", type=Path, default=DEFAULT_PR_ROOT)
    parser.add_argument("--ant-xml", type=Path, default=HERE / "models" / "ant.xml")
    parser.add_argument("--humanoid-xml", type=Path)
    parser.add_argument(
        "--models", nargs="+", choices=("ant", "humanoid"), default=("ant", "humanoid")
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--directions", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=1.0e-2)
    parser.add_argument("--max-relative-error", type=float, default=5.0e-3)
    parser.add_argument("--nconmax", type=int, default=128)
    parser.add_argument("--njmax", type=int, default=512)
    parser.add_argument("--allow-other-head", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.humanoid_xml is None:
        args.humanoid_xml = args.pr_root / "benchmarks" / "humanoid" / "humanoid.xml"
    if args.directions < 1 or args.epsilon <= 0.0:
        parser.error("directions and epsilon must be positive")
    return args


def main() -> int:
    args = parse_args()
    script_path = Path(__file__).resolve()
    pr_head = git_value(args.pr_root, "rev-parse", "HEAD")
    if pr_head != EXPECTED_PR_HEAD and not args.allow_other_head:
        raise RuntimeError(
            f"Expected PR #1535 head {EXPECTED_PR_HEAD}, found {pr_head}. "
            "Use --allow-other-head only for an intentional later revision."
        )
    imported_root = Path(mjw.__file__).resolve().parent.parent
    if imported_root != args.pr_root.resolve():
        raise RuntimeError(
            f"Imported MJWarp from {imported_root}, expected {args.pr_root.resolve()}"
        )

    wp.init()
    wp.set_device(args.device)
    wp.config.max_unroll = max(wp.config.max_unroll, 64)
    # This opt-in must precede put_model and differentiated compilation.
    mjw.enable_grad()

    paths = {"ant": args.ant_xml, "humanoid": args.humanoid_xml}
    for name in args.models:
        if not paths[name].is_file():
            raise FileNotFoundError(paths[name])
    results = [probe_model(name, paths[name], args) for name in args.models]
    payload = {
        "schema_version": 1,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "result": "pass"
        if all(item["result"] == "pass" for item in results)
        else "fail",
        "command": [sys.executable, *sys.argv],
        "script": {
            "path": str(script_path),
            "sha256": sha256(script_path),
        },
        "pr": {
            "url": "https://github.com/google-deepmind/mujoco_warp/pull/1535",
            "head": pr_head,
            "checkout": str(args.pr_root.resolve()),
            "working_tree": git_value(args.pr_root, "status", "--short"),
            "import_path": str(Path(mjw.__file__).resolve()),
        },
        "versions": {
            "python": platform.python_version(),
            "mujoco_warp": getattr(mjw, "__version__", "unknown"),
            "mujoco": mujoco.__version__,
            "warp": wp.__version__,
            "numpy": np.__version__,
        },
        "device": {
            "requested": args.device,
            "warp": str(wp.get_device()),
            "name": wp.get_device().name,
            "architecture": wp.get_device().arch,
            "memory_bytes": wp.get_device().total_memory,
        },
        "config": {
            "seed": args.seed,
            "directions": args.directions,
            "epsilon": args.epsilon,
            "maximum_relative_error": args.max_relative_error,
        },
        "models": results,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    print(rendered)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n")
    return 0 if payload["result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
