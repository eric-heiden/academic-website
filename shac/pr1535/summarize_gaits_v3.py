#!/usr/bin/env python3
"""Validate and summarize the publishable v3 Ant and Humanoid gait artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = SCRIPT_DIR / "results" / "gaits_v3"
TASKS = ("ant", "humanoid")
METRICS = (
    "final_alive_fraction",
    "mean_survival_fraction",
    "mean_forward_speed_over_horizon",
    "mean_abs_lateral_displacement",
    "mean_action_rate_rms",
    "mean_support_foot_slip_rms",
    "mean_up_while_alive",
    "mean_heading_while_alive",
    "mean_single_support_fraction",
    "mean_two_or_more_support_fraction",
    "mean_diagonal_support_fraction",
    "mean_alternating_support_switches_per_second",
    "mean_alternating_diagonal_support_switches_per_second",
    "mean_flight_fraction",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def metric_ranges(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in METRICS:
        values = [float(row[name]) for row in evaluations if name in row]
        if values:
            result[name] = {
                "minimum": min(values),
                "maximum": max(values),
                "mean": sum(values) / len(values),
            }
    return result


def summarized_metrics(row: dict[str, Any]) -> dict[str, float]:
    return {name: float(row[name]) for name in METRICS if name in row}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    output = (args.output or results_dir / "summary.json").expanduser().resolve()
    summaries: dict[str, Any] = {}
    revision_pairs: set[tuple[str, str]] = set()

    for task in TASKS:
        audit_path = results_dir / f"{task}_audit.json"
        shac_path = results_dir / f"{task}_shac.json"
        render_path = results_dir / f"{task}_viewergl.json"
        checkpoint_path = results_dir / f"{task}_checkpoint.pt"
        source_checkpoint_path = results_dir / f"{task}_shac_source_checkpoint.pt"
        audit = read_json(audit_path)
        shac = read_json(shac_path)
        render = read_json(render_path)

        expected_source_hashes = {
            "audit": sha256_file(SCRIPT_DIR / "audit_gaits_v3.py"),
            "shac": sha256_file(SCRIPT_DIR / "fine_tune_gaits_shac_v3.py"),
            "render": sha256_file(SCRIPT_DIR / "render_gaits_v3.py"),
            "gait": sha256_file(SCRIPT_DIR / "train_gaits_v3.py"),
            "conditioning": sha256_file(SCRIPT_DIR / "conditioned_policy_v3.py"),
            "bridge": sha256_file(SCRIPT_DIR / "mjwarp_torch_bridge.py"),
        }

        require(audit.get("status") == "completed", f"{task} audit incomplete")
        require(shac.get("status") == "completed", f"{task} SHAC run incomplete")
        require(audit.get("task") == task, f"{task} audit task mismatch")
        require(shac.get("task") == task, f"{task} SHAC task mismatch")
        require(render.get("task") == task, f"{task} render task mismatch")
        require(audit.get("all_seed_gates_pass") is True, f"{task} noisy audit failed")
        require(audit.get("nominal_gate_pass") is True, f"{task} nominal audit failed")
        require(
            shac.get("holdout_all_gates_pass") is True,
            f"{task} SHAC holdout failed",
        )
        require(
            render["independent_nominal_evaluation"]["gate"]["pass"] is True,
            f"{task} render evaluation failed",
        )
        require(
            render["recorded_gait_evaluation"]["gate"]["pass"] is True,
            f"{task} exact recorded trajectory failed",
        )
        require(render["behavior"]["final_alive"] is True, f"{task} render fell")
        require(
            audit["provenance"]["script_sha256"] == expected_source_hashes["audit"],
            f"{task} audit source hash mismatch",
        )
        require(
            shac["provenance"]["script_sha256"] == expected_source_hashes["shac"],
            f"{task} SHAC source hash mismatch",
        )
        require(
            render["software"]["renderer_script_sha256"]
            == expected_source_hashes["render"],
            f"{task} renderer source hash mismatch",
        )
        for manifest_name, manifest in (("audit", audit), ("SHAC", shac)):
            require(
                manifest["provenance"]["gait_harness_sha256"]
                == expected_source_hashes["gait"],
                f"{task} {manifest_name} gait-harness hash mismatch",
            )
            require(
                manifest["provenance"]["bridge_sha256"]
                == expected_source_hashes["bridge"],
                f"{task} {manifest_name} bridge hash mismatch",
            )
        require(
            audit["provenance"]["conditioning_script_sha256"]
            == expected_source_hashes["conditioning"],
            f"{task} audit conditioning hash mismatch",
        )
        require(
            render["software"]["conditioning_script_sha256"]
            == expected_source_hashes["conditioning"],
            f"{task} render conditioning hash mismatch",
        )

        checkpoint_sha256 = sha256_file(checkpoint_path)
        source_checkpoint_sha256 = sha256_file(source_checkpoint_path)
        require(
            audit["checkpoint_sha256"] == checkpoint_sha256,
            f"{task} audit/checkpoint hash mismatch",
        )
        require(
            shac["checkpoint_sha256"] == checkpoint_sha256,
            f"{task} SHAC/checkpoint hash mismatch",
        )
        require(
            render["checkpoint_sha256"] == checkpoint_sha256,
            f"{task} render/checkpoint hash mismatch",
        )
        require(
            shac["source_checkpoint_sha256"] == source_checkpoint_sha256,
            f"{task} SHAC source-checkpoint hash mismatch",
        )
        for artifact_name in ("video", "poster"):
            artifact = results_dir / render["artifacts"][artifact_name]
            require(artifact.is_file(), f"missing {task} {artifact_name}")
            require(
                sha256_file(artifact)
                == render["artifacts"][f"{artifact_name}_sha256"],
                f"{task} {artifact_name} hash mismatch",
            )

        audit_revision = (
            audit["provenance"]["mjwarp_pr_head"],
            audit["provenance"]["newton_head"],
        )
        shac_revision = (
            shac["provenance"]["mjwarp_pr_head"],
            shac["provenance"]["newton_head"],
        )
        render_revision = (
            render["physics"]["pr_head"],
            render["physics"]["newton_head"],
        )
        require(
            audit_revision == shac_revision == render_revision,
            f"{task} revision mismatch",
        )
        revision_pairs.add(audit_revision)

        evaluations = audit["evaluations"]
        direction_checks = [row["direction_check"] for row in shac["history"]]
        summaries[task] = {
            "checkpoint": checkpoint_path.name,
            "checkpoint_sha256": checkpoint_sha256,
            "audit": {
                "seeds": audit["seeds"],
                "worlds_per_seed": audit["worlds_per_seed"],
                "total_noisy_worlds": len(evaluations) * audit["worlds_per_seed"],
                "control_steps": audit["steps"],
                "simulated_seconds": audit["simulated_seconds"],
                "all_noisy_gates_pass": True,
                "nominal_gate_pass": True,
                "gate_thresholds": evaluations[0]["gate"]["thresholds"],
                "noisy_metric_ranges": metric_ranges(evaluations),
                "nominal_metrics": summarized_metrics(audit["nominal_evaluation"]),
                "support_definition": audit["support_definition"],
            },
            "shac": {
                "source_checkpoint": source_checkpoint_path.name,
                "source_checkpoint_sha256": source_checkpoint_sha256,
                "best_update": shac.get("best_update"),
                "accepted_updates": sum(
                    bool(row.get("accepted")) for row in shac["history"]
                ),
                "holdout_count": len(shac["holdouts"]),
                "holdout_all_gates_pass": True,
                "nominal_holdout_gate_pass": shac.get("nominal_holdout", {})
                .get("gate", {})
                .get("pass"),
                "direction_checks": direction_checks,
                "maximum_direction_relative_error": max(
                    float(row["relative_error"]) for row in direction_checks
                ),
                "all_direction_signs_match": all(
                    row["same_sign"] for row in direction_checks
                ),
            },
            "viewergl": {
                "video": render["artifacts"]["video"],
                "poster": render["artifacts"]["poster"],
                "final_alive": True,
                "simulated_seconds": render["behavior"]["simulated_seconds"],
                "forward_speed_over_horizon": render["behavior"][
                    "forward_speed_over_horizon"
                ],
                "displacement_x": render["behavior"]["displacement_x"],
                "displacement_y": render["behavior"]["displacement_y"],
                "recorded_gait_gate_pass": True,
                "recorded_gait_metrics": summarized_metrics(
                    render["recorded_gait_evaluation"]
                ),
                "independent_nominal_gate_pass": True,
            },
        }

    require(len(revision_pairs) == 1, "Ant and Humanoid revisions differ")
    mjwarp_head, newton_head = revision_pairs.pop()
    result = {
        "schema": "mjwarp-pr1535-final-gaits-v3-summary",
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mjwarp_pr_head": mjwarp_head,
        "newton_head": newton_head,
        "all_final_gates_pass": True,
        "tasks": summaries,
        "method_notes": [
            "Noisy audits contain three independent 1024-lane, uninterrupted full-horizon rollouts.",
            "Terminal lanes freeze; displacement-derived speed always uses the complete requested horizon.",
            "Nominal full-horizon evaluation is a mandatory gate in SHAC selection and final audit.",
            "ViewerGL records a fresh single-lane MJWarp rollout and uses Newton only to render recorded qpos states.",
        ],
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(f"Validated final gait bundle and wrote {output}")


if __name__ == "__main__":
    main()
