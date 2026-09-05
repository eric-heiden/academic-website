#!/usr/bin/env python3
"""Run reproducible, uninterrupted multi-seed audits of v3 gait policies."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
from conditioned_policy_v3 import conditioned_actor
import train_gaits_v3 as gait
import warp as wp


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def aggregate(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    excluded = {
        "seed",
        "worlds",
        "steps",
        "control_dt",
        "simulated_seconds",
        "noise",
        "gate",
        "component_means",
    }
    scalar_names = sorted(
        name
        for name, value in evaluations[0].items()
        if name not in excluded and isinstance(value, (int, float))
    )
    result = {}
    for name in scalar_names:
        values = np.asarray([item[name] for item in evaluations], dtype=np.float64)
        result[name] = {
            "mean": float(values.mean()),
            "population_std": float(values.std()),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "by_seed": {str(item["seed"]): float(item[name]) for item in evaluations},
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-policy", choices=("best", "final"), default="best"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worlds", type=int, default=1024)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seeds", type=int, nargs="+", default=[9001, 9011, 9029])
    parser.add_argument("--nconmax", type=int, default=128)
    parser.add_argument("--njmax", type=int, default=512)
    args = parser.parse_args()
    if args.worlds <= 0 or (args.steps is not None and args.steps <= 0):
        parser.error("--worlds and --steps must be positive")
    if not args.seeds:
        parser.error("--seeds must not be empty")
    return args


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    checkpoint = torch.load(
        checkpoint_path, map_location=args.device, weights_only=False
    )
    if checkpoint.get("format") not in {
        "mjwarp-pr1535-full-gait-v3",
        "mjwarp-pr1535-shac-gait-v3",
    }:
        raise ValueError("unsupported gait checkpoint format")
    config_dict = dict(checkpoint["config"])
    config = SimpleNamespace(**config_dict)
    task = str(checkpoint["task"])
    pr_root = Path(config.pr_root).resolve()
    if Path(mjw.__file__).resolve().parent.parent != pr_root:
        raise RuntimeError("the imported MuJoCo Warp is not the checkpoint PR tree")
    if gait.git_head(pr_root) != checkpoint.get("pr_head"):
        raise RuntimeError("checkpoint and checked-out MuJoCo Warp revisions differ")
    xml_path = (
        Path(config.xml).resolve()
        if config_dict.get("xml")
        else (
            pr_root / gait.v1.TASKS[task].default_xml
            if task == "humanoid"
            else gait.v1.TASKS[task].default_xml
        ).resolve()
    )

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    wp.config.max_unroll = max(wp.config.max_unroll, 64)
    wp.set_device(args.device)
    mjw.enable_grad()
    loaded = gait.v1.load_model(gait.v1.TASKS[task], xml_path)
    bridge = gait.v1.make_bridge(loaded, args)
    actor, critic, normalizer = gait.make_networks(loaded, config, bridge.torch_device)
    del critic
    policy = args.checkpoint_policy
    actor.load_state_dict(checkpoint[f"{policy}_actor"])
    normalizer.load_state_dict(checkpoint[f"{policy}_normalizer"])
    actor.eval()
    normalizer.eval()
    evaluation_actor = conditioned_actor(actor, normalizer, checkpoint, loaded)
    steps = args.steps or (600 if task == "humanoid" else 400)

    evaluations = [
        gait.evaluate_policy(
            bridge,
            loaded,
            evaluation_actor,
            normalizer,
            config,
            seed=seed,
            steps=steps,
            noise=True,
        )
        for seed in args.seeds
    ]
    nominal = gait.evaluate_policy(
        bridge,
        loaded,
        evaluation_actor,
        normalizer,
        config,
        seed=0,
        steps=steps,
        noise=False,
    )
    result = {
        "schema": "mjwarp-pr1535-gait-audit-v3",
        "status": "completed",
        "task": task,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": gait.sha256_file(checkpoint_path),
        "checkpoint_format": checkpoint["format"],
        "checkpoint_policy": policy,
        "worlds_per_seed": args.worlds,
        "seeds": args.seeds,
        "steps": steps,
        "simulated_seconds": steps
        * float(loaded.model.opt.timestep)
        * int(config.action_repeat),
        "evaluations": evaluations,
        "aggregate": aggregate(evaluations),
        "nominal_evaluation": nominal,
        "all_seed_gates_pass": all(item["gate"]["pass"] for item in evaluations),
        "nominal_gate_pass": nominal["gate"]["pass"],
        "support_definition": (
            {
                "point": "outer endpoint of each distal capsule: 2 * geom_xpos - body_xpos",
                "support_height_m": 0.100,
                "basis": "0.08 m capsule radius + 0.02 m combined geom contact margin",
            }
            if task == "ant"
            else {
                "point": "left/right foot body origin",
                "support_height_m": 0.060,
            }
        ),
        "provenance": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "script": str(Path(__file__).resolve()),
            "script_sha256": gait.sha256_file(Path(__file__).resolve()),
            "gait_harness_sha256": gait.sha256_file(Path(gait.__file__).resolve()),
            "conditioning_script_sha256": gait.sha256_file(
                Path(__file__).with_name("conditioned_policy_v3.py")
            ),
            "bridge_sha256": gait.sha256_file(
                gait.SCRIPT_DIR / "mjwarp_torch_bridge.py"
            ),
            "xml": str(xml_path),
            "xml_sha256": gait.sha256_file(xml_path),
            "mjwarp_pr_head": gait.git_head(pr_root),
            "newton_head": gait.git_head(Path(config.newton_root)),
            "versions": {
                "python": platform.python_version(),
                "mujoco": mujoco.__version__,
                "mujoco_warp": getattr(mjw, "__version__", None),
                "warp": wp.__version__,
                "torch": torch.__version__,
                "numpy": np.__version__,
            },
            "method_notes": [
                "Each noisy lane begins from an independently sampled perturbation.",
                "Rollouts are uninterrupted: terminal lanes freeze and are never reset.",
                "Speed is displacement divided by the complete requested horizon, so early falls cannot inflate it.",
                "All reported foot kinematics are recomputed at returned qpos with MJWarp kinematics.",
                "Support is limited to the model contact envelope; near-ground swing outside that envelope is not counted as stance.",
                "Checkpoint-declared causal control conditioning is included in every rollout.",
            ],
        },
    }
    write_json(output_path, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
