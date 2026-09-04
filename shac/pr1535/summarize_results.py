#!/usr/bin/env python3
"""Validate and summarize the evidence used by the MJWarp PR #1535 report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PR_HEAD = "02d09b139fdf091e1e859d7f41c47a8f71d30574"
PR_URL = "https://github.com/google-deepmind/mujoco_warp/pull/1535"
SEEDS = (17, 29, 41)
TASKS = ("ant", "humanoid")
PRIMARY_METRICS = (
    "mean_return",
    "mean_displacement",
    "mean_alive_fraction",
    "final_alive_fraction",
)
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def fail(message: str) -> None:
    raise SystemExit(f"summary validation failed: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject_constant(value: str) -> None:
    fail(f"non-finite JSON constant {value!r}")


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing input {path}")
    try:
        value = json.loads(path.read_text(), parse_constant=reject_constant)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(f"cannot read {path}: {exc}")
    require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def at(value: Any, path: str, source: Path) -> Any:
    current = value
    for component in path.split("."):
        if not isinstance(current, dict) or component not in current:
            fail(f"{source}: missing {path}")
        current = current[component]
    return current


def number(value: Any, label: str) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be numeric",
    )
    result = float(value)
    require(math.isfinite(result), f"{label} must be finite")
    return result


def timestamp(value: Any, label: str) -> str:
    require(isinstance(value, str), f"{label} must be an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        fail(f"{label} is not an ISO timestamp: {exc}")
    require(parsed.tzinfo is not None, f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat()


def stats(values: list[float]) -> dict[str, Any]:
    require(bool(values), "cannot summarize an empty sample")
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "population_std": statistics.pstdev(values),
    }


def aggregate_pair(
    runs: list[dict[str, Any]],
    before: Callable[[dict[str, Any]], dict[str, Any]],
    after: Callable[[dict[str, Any]], dict[str, Any]],
    before_name: str,
    after_name: str,
) -> dict[str, Any]:
    before_summary: dict[str, Any] = {}
    after_summary: dict[str, Any] = {}
    delta_summary: dict[str, Any] = {}
    for metric in PRIMARY_METRICS:
        before_values = [number(before(run)[metric], metric) for run in runs]
        after_values = [number(after(run)[metric], metric) for run in runs]
        deltas = [b - a for a, b in zip(before_values, after_values, strict=True)]
        before_summary[metric] = stats(before_values)
        after_summary[metric] = stats(after_values)
        delta = stats(deltas)
        baseline = before_summary[metric]["mean"]
        if abs(baseline) > 1.0e-12:
            delta["percent_change_of_initial_mean"] = 100.0 * delta["mean"] / baseline
        delta_summary[metric] = delta
    return {
        before_name: before_summary,
        after_name: after_summary,
        "paired_delta": delta_summary,
    }


def normalized_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in config.items() if key not in {"seed", "output"}
    }


def compact_config(config: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "worlds",
        "horizon",
        "epochs",
        "eval_steps",
        "eval_every",
        "hidden",
        "gamma",
        "td_lambda",
        "actor_lr",
        "critic_lr",
        "critic_iterations",
        "target_polyak",
        "adam_beta1",
        "adam_beta2",
        "max_grad_norm",
        "reset_interval",
    )
    return {key: config[key] for key in keys}


def validate_metrics(metrics: Any, label: str) -> dict[str, Any]:
    require(isinstance(metrics, dict), f"{label} must be an object")
    for metric in PRIMARY_METRICS:
        require(metric in metrics, f"{label} missing {metric}")
        number(metrics[metric], f"{label}.{metric}")
    return metrics


def validate_training(
    task: str,
    train_sha: str,
    bridge_sha: str,
    artifacts: dict[str, dict[str, Any]],
    input_times: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    runs: list[dict[str, Any]] = []
    model_hashes: set[str] = set()
    newton_heads: set[str] = set()
    versions: set[str] = set()
    reference_config: dict[str, Any] | None = None

    for seed in SEEDS:
        json_path = RESULTS / f"{task}_seed{seed}.json"
        checkpoint_path = RESULTS / f"{task}_seed{seed}.pt"
        run = load_json(json_path)
        label = json_path.name

        require(run.get("schema") == "mjwarp-pr1535-shac-style-v1", f"{label}: schema")
        require(run.get("mode") == "train", f"{label}: mode")
        config = at(run, "config", json_path)
        require(isinstance(config, dict), f"{label}: config must be an object")
        require(config.get("task") == task, f"{label}: task mismatch")
        require(config.get("seed") == seed, f"{label}: seed mismatch")
        require(config.get("mode") == "train", f"{label}: config mode mismatch")
        comparable_config = normalized_config(config)
        if reference_config is None:
            reference_config = comparable_config
        else:
            require(
                comparable_config == reference_config,
                f"{task}: configs differ across seeds",
            )

        require(
            at(run, "provenance.pr.head", json_path) == PR_HEAD, f"{label}: PR head"
        )
        require(
            at(run, "provenance.pr.url", json_path) == PR_URL,
            f"{label}: PR URL",
        )
        require(
            at(run, "provenance.pr.exact_worktree_import", json_path) is True,
            f"{label}: exact PR import not recorded",
        )
        require(
            at(run, "provenance.script_sha256", json_path) == train_sha,
            f"{label}: train_shac.py hash mismatch",
        )
        require(
            at(run, "provenance.bridge_sha256", json_path) == bridge_sha,
            f"{label}: bridge hash mismatch",
        )
        model_hashes.add(at(run, "provenance.model.xml_sha256", json_path))
        newton_heads.add(at(run, "provenance.newton_head", json_path))
        versions.add(
            json.dumps(at(run, "provenance.versions", json_path), sort_keys=True)
        )
        input_times.append(
            timestamp(at(run, "provenance.timestamp_utc", json_path), label)
        )

        run_data = at(run, "run", json_path)
        require(run_data.get("status") == "completed", f"{label}: run incomplete")
        require(
            run_data.get("epochs_completed") == config.get("epochs"), f"{label}: epochs"
        )
        require(run_data.get("cold_start") is True, f"{label}: not a cold start")
        require(
            run_data.get("canonical_shac") is False, f"{label}: method label changed"
        )
        evaluations = run_data.get("evaluations")
        require(
            isinstance(evaluations, list) and len(evaluations) >= 2,
            f"{label}: evaluations",
        )
        require(evaluations[0].get("epoch") == 0, f"{label}: initial evaluation epoch")
        require(
            evaluations[-1].get("epoch") == config.get("epochs"),
            f"{label}: final evaluation epoch",
        )

        metric_block = run_data.get("metrics")
        require(isinstance(metric_block, dict), f"{label}: metrics")
        initial = validate_metrics(
            metric_block.get("initial"), f"{label}.metrics.initial"
        )
        selected = validate_metrics(metric_block.get("best"), f"{label}.metrics.best")
        final = validate_metrics(metric_block.get("final"), f"{label}.metrics.final")
        initial_eval = validate_metrics(
            evaluations[0].get("metrics"), f"{label}.evaluations[0]"
        )
        final_eval = validate_metrics(
            evaluations[-1].get("metrics"), f"{label}.evaluations[-1]"
        )
        for metric in PRIMARY_METRICS:
            require(
                initial[metric] == initial_eval[metric],
                f"{label}: initial metric mismatch",
            )
            require(
                final[metric] == final_eval[metric], f"{label}: final metric mismatch"
            )

        holdout = run_data.get("holdout")
        require(isinstance(holdout, dict), f"{label}: holdout")
        require(
            holdout.get("selection_independent") is True,
            f"{label}: holdout independence",
        )
        validate_metrics(holdout.get("initial"), f"{label}.holdout.initial")
        validate_metrics(holdout.get("selected_best"), f"{label}.holdout.selected_best")
        selected_epoch = holdout.get("selected_epoch")
        require(
            isinstance(selected_epoch, int) and selected_epoch == selected.get("epoch"),
            f"{label}: selected epoch mismatch",
        )

        require(checkpoint_path.is_file(), f"missing checkpoint {checkpoint_path}")
        checkpoint = run_data.get("checkpoint")
        require(
            isinstance(checkpoint, str)
            and Path(checkpoint).name == checkpoint_path.name,
            f"{label}: checkpoint path mismatch",
        )
        for artifact in (json_path, checkpoint_path):
            relative = artifact.relative_to(ROOT).as_posix()
            artifacts[relative] = {
                "sha256": sha256(artifact),
                "bytes": artifact.stat().st_size,
            }
        runs.append(run)

    require(reference_config is not None, f"{task}: missing config")
    require(len(model_hashes) == 1, f"{task}: model hashes differ across seeds")
    require(len(newton_heads) == 1, f"{task}: Newton heads differ across seeds")
    require(len(versions) == 1, f"{task}: package versions differ across seeds")
    return (
        runs,
        reference_config,
        {
            "model_sha256": next(iter(model_hashes)),
            "newton_head": next(iter(newton_heads)),
            "versions": next(iter(versions)),
        },
    )


def summarize_training(
    runs: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    selected_epochs = [
        {
            "seed": run["config"]["seed"],
            "epoch": run["run"]["holdout"]["selected_epoch"],
        }
        for run in runs
    ]
    return {
        "seeds": [run["config"]["seed"] for run in runs],
        "algorithm": runs[0]["run"]["algorithm"],
        "canonical_shac": runs[0]["run"]["canonical_shac"],
        "cold_start": runs[0]["run"]["cold_start"],
        "config": compact_config(config),
        "selected_epochs": selected_epochs,
        "holdout": {
            "selection_independent": True,
            **aggregate_pair(
                runs,
                lambda run: run["run"]["holdout"]["initial"],
                lambda run: run["run"]["holdout"]["selected_best"],
                "initial",
                "selected_best",
            ),
        },
        "selection_set": aggregate_pair(
            runs,
            lambda run: run["run"]["metrics"]["initial"],
            lambda run: run["run"]["metrics"]["final"],
            "initial",
            "final",
        ),
    }


def contact_mismatch_one_step(comparison: dict[str, Any], nominal: int) -> bool:
    plus = comparison.get("plus_contacts")
    minus = comparison.get("minus_contacts")
    require(
        isinstance(plus, int) and isinstance(minus, int),
        "invalid one-step contact count",
    )
    return plus != nominal or minus != nominal or plus != minus


def summarize_one_step(
    probe_sha: str,
    artifacts: dict[str, dict[str, Any]],
    input_times: list[str],
) -> tuple[dict[str, Any], dict[str, str]]:
    stacks: dict[str, Any] = {}
    model_hashes: dict[str, set[str]] = {task: set() for task in TASKS}
    reference_config: dict[str, Any] | None = None
    reference_tasks: dict[str, Any] | None = None
    files = {
        "frozen_pr_lock": RESULTS / "gradient_frozen_lock.json",
        "newton_integration": RESULTS / "gradient_newton_stack.json",
    }
    for stack, path in files.items():
        data = load_json(path)
        label = path.name
        require(data.get("schema_version") == 1, f"{label}: schema")
        require(data.get("result") == "pass", f"{label}: overall probe did not pass")
        require(at(data, "pr.head", path) == PR_HEAD, f"{label}: PR head")
        require(at(data, "pr.url", path) == PR_URL, f"{label}: PR URL")
        require(at(data, "script.sha256", path) == probe_sha, f"{label}: probe hash")
        input_times.append(timestamp(data.get("timestamp_utc"), label))
        models = data.get("models")
        require(
            isinstance(models, list) and len(models) == len(TASKS), f"{label}: models"
        )
        require(
            {model.get("model", {}).get("name") for model in models} == set(TASKS),
            f"{label}: tasks",
        )

        task_summary: dict[str, Any] = {}
        for entry in models:
            model = entry.get("model")
            require(isinstance(model, dict), f"{label}: model entry")
            task = model.get("name")
            require(task in TASKS, f"{label}: unexpected model {task}")
            require(entry.get("result") == "pass", f"{label}: {task} did not pass")
            nominal = model.get("nominal_contacts")
            require(isinstance(nominal, int), f"{label}: {task} nominal contacts")
            model_hash = model.get("xml_sha256")
            require(
                isinstance(model_hash, str) and len(model_hash) == 64,
                f"{label}: model hash",
            )
            model_hashes[task].add(model_hash)
            objectives = entry.get("objectives")
            require(
                isinstance(objectives, list) and objectives,
                f"{label}: {task} objectives",
            )
            objective_summary: list[dict[str, Any]] = []
            for objective in objectives:
                comparisons = objective.get("comparisons")
                require(
                    isinstance(comparisons, list) and comparisons,
                    f"{label}: comparisons",
                )
                mismatch_count = sum(
                    contact_mismatch_one_step(comparison, nominal)
                    for comparison in comparisons
                )
                objective_summary.append(
                    {
                        "name": objective.get("name"),
                        "directions": objective.get("directions"),
                        "median_relative_error": number(
                            objective.get("median_relative_error"),
                            "median relative error",
                        ),
                        "max_relative_error": number(
                            objective.get("max_relative_error"),
                            "maximum relative error",
                        ),
                        "control_gradient_l2_norm": number(
                            objective.get("control_gradient_norm"), "gradient norm"
                        ),
                        "contact_mismatch_direction_count": mismatch_count,
                    }
                )
            task_summary[task] = {
                "result": entry["result"],
                "nominal_contacts": nominal,
                "criterion_max_relative_error": entry["criterion"][
                    "maximum_directional_relative_error"
                ],
                "objectives": objective_summary,
            }
        if reference_config is None:
            reference_config = data["config"]
            reference_tasks = task_summary
        else:
            require(data["config"] == reference_config, "one-step configs differ")
            require(
                task_summary == reference_tasks,
                "one-step numerical results differ between package stacks",
            )
        stacks[stack] = {
            "result": data["result"],
            "versions": data["versions"],
            "config": data["config"],
            "tasks": task_summary,
        }
        relative = path.relative_to(ROOT).as_posix()
        artifacts[relative] = {"sha256": sha256(path), "bytes": path.stat().st_size}

    for task, hashes in model_hashes.items():
        require(
            len(hashes) == 1, f"one-step {task}: model hashes differ between stacks"
        )
    return {"stacks": stacks}, {
        task: next(iter(hashes)) for task, hashes in model_hashes.items()
    }


def summarize_multistep(
    task: str,
    train_sha: str,
    bridge_sha: str,
    expected_model_sha: str,
    artifacts: dict[str, dict[str, Any]],
    input_times: list[str],
) -> dict[str, Any]:
    path = RESULTS / f"{task}_multistep_gradient.json"
    data = load_json(path)
    label = path.name
    require(data.get("schema") == "mjwarp-pr1535-shac-style-v1", f"{label}: schema")
    require(data.get("mode") == "gradcheck", f"{label}: mode")
    require(at(data, "config.task", path) == task, f"{label}: task")
    require(at(data, "provenance.pr.head", path) == PR_HEAD, f"{label}: PR head")
    require(at(data, "provenance.pr.url", path) == PR_URL, f"{label}: PR URL")
    require(
        at(data, "provenance.script_sha256", path) == train_sha, f"{label}: script hash"
    )
    require(
        at(data, "provenance.bridge_sha256", path) == bridge_sha,
        f"{label}: bridge hash",
    )
    require(
        at(data, "provenance.model.xml_sha256", path) == expected_model_sha,
        f"{label}: model hash differs from training and one-step probes",
    )
    input_times.append(timestamp(at(data, "provenance.timestamp_utc", path), label))

    config = data["config"]
    checks = at(data, "run.checks", path)
    require(isinstance(checks, list) and checks, f"{label}: checks")
    horizons = config.get("gradcheck_horizons")
    require(
        isinstance(horizons, list)
        and [check.get("horizon") for check in checks] == horizons,
        f"{label}: horizon list mismatch",
    )
    compact_checks: list[dict[str, Any]] = []
    for check in checks:
        comparisons = check.get("comparisons")
        require(isinstance(comparisons, list) and comparisons, f"{label}: comparisons")
        for comparison in comparisons:
            number(
                comparison.get("contact_count_mismatch_steps"), "contact mismatch steps"
            )
            require(
                isinstance(comparison.get("same_sign"), bool), f"{label}: same_sign"
            )
        mismatch_directions = sum(
            comparison["contact_count_mismatch_steps"] > 0 for comparison in comparisons
        )
        mismatch_steps = sum(
            comparison["contact_count_mismatch_steps"] for comparison in comparisons
        )
        sign_mismatches = sum(not comparison["same_sign"] for comparison in comparisons)
        computed_all_same_sign = sign_mismatches == 0
        require(
            check.get("all_same_sign") is computed_all_same_sign,
            f"{label}: sign summary",
        )
        gradient_norm = number(check.get("gradient_l2_norm"), "gradient norm")
        max_relative_error = number(
            check.get("max_relative_error"), "maximum relative error"
        )
        expected_pass = (
            gradient_norm > 0.0
            and max_relative_error <= config["gradcheck_max_relative"]
            and computed_all_same_sign
        )
        require(check.get("pass") is expected_pass, f"{label}: pass criterion")
        compact_checks.append(
            {
                "horizon": check["horizon"],
                "pass": check["pass"],
                "objective": number(check.get("objective"), "objective"),
                "gradient_l2_norm": gradient_norm,
                "max_relative_error": max_relative_error,
                "all_same_sign": check["all_same_sign"],
                "sign_mismatch_direction_count": sign_mismatches,
                "contact_mismatch_direction_count": mismatch_directions,
                "contact_mismatch_step_count": mismatch_steps,
            }
        )
    expected_status = "pass" if all(check["pass"] for check in checks) else "fail"
    require(at(data, "run.status", path) == expected_status, f"{label}: overall status")
    relative = path.relative_to(ROOT).as_posix()
    artifacts[relative] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    return {
        "status": expected_status,
        "seed": config["seed"],
        "worlds": config["worlds"],
        "directions": config["gradcheck_directions"],
        "epsilon": config["gradcheck_eps"],
        "criterion_max_relative_error": config["gradcheck_max_relative"],
        "checks": compact_checks,
    }


def validate_local_model(path: Path, expected_sha: str, label: str) -> None:
    require(path.is_file(), f"missing {label} model source {path}")
    require(sha256(path) == expected_sha, f"{label} model source hash mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS / "summary.json")
    args = parser.parse_args()

    source_paths = {
        "train_shac.py": ROOT / "train_shac.py",
        "mjwarp_torch_bridge.py": ROOT / "mjwarp_torch_bridge.py",
        "gradient_probe.py": ROOT / "gradient_probe.py",
        "models/ant.xml": ROOT / "models" / "ant.xml",
        "summarize_results.py": Path(__file__).resolve(),
    }
    for label, path in source_paths.items():
        require(path.is_file(), f"missing source {label}")
    source_hashes = {label: sha256(path) for label, path in source_paths.items()}
    artifacts: dict[str, dict[str, Any]] = {}
    input_times: list[str] = []

    training_runs: dict[str, list[dict[str, Any]]] = {}
    training_configs: dict[str, dict[str, Any]] = {}
    training_provenance: dict[str, dict[str, str]] = {}
    for task in TASKS:
        runs, config, provenance = validate_training(
            task,
            source_hashes["train_shac.py"],
            source_hashes["mjwarp_torch_bridge.py"],
            artifacts,
            input_times,
        )
        training_runs[task] = runs
        training_configs[task] = config
        training_provenance[task] = provenance

    one_step, one_step_model_hashes = summarize_one_step(
        source_hashes["gradient_probe.py"], artifacts, input_times
    )
    for task in TASKS:
        require(
            training_provenance[task]["model_sha256"] == one_step_model_hashes[task],
            f"{task}: training and one-step model hashes differ",
        )

    validate_local_model(
        source_paths["models/ant.xml"], one_step_model_hashes["ant"], "Ant"
    )
    humanoid_path = Path(training_runs["humanoid"][0]["provenance"]["model"]["xml"])
    validate_local_model(humanoid_path, one_step_model_hashes["humanoid"], "Humanoid")
    source_hashes["external/humanoid.xml"] = sha256(humanoid_path)

    multistep = {
        task: summarize_multistep(
            task,
            source_hashes["train_shac.py"],
            source_hashes["mjwarp_torch_bridge.py"],
            one_step_model_hashes[task],
            artifacts,
            input_times,
        )
        for task in TASKS
    }

    versions = {
        task: json.loads(training_provenance[task]["versions"]) for task in TASKS
    }
    require(
        versions["ant"] == versions["humanoid"],
        "training package versions differ by task",
    )
    require(
        training_provenance["ant"]["newton_head"]
        == training_provenance["humanoid"]["newton_head"],
        "Newton head differs by task",
    )
    require(input_times, "no input timestamps found")

    summary = {
        "schema": "mjwarp-pr1535-shac-report-summary-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_timestamp_range_utc": {
            "earliest": min(input_times),
            "latest": max(input_times),
        },
        "provenance": {
            "pr": {"url": PR_URL, "head": PR_HEAD},
            "newton_head": training_provenance["ant"]["newton_head"],
            "training_versions": versions["ant"],
            "sources": source_hashes,
            "artifacts": dict(sorted(artifacts.items())),
        },
        "training": {
            task: summarize_training(training_runs[task], training_configs[task])
            for task in TASKS
        },
        "gradients": {"one_step": one_step, "multi_horizon": multistep},
    }

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
