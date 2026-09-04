#!/usr/bin/env python3
"""Validate and aggregate the six fixed SHAC-style v2 training runs.

The summarizer is deliberately strict: it accepts exactly three Ant and three
Humanoid training JSON files, verifies their run status, seeds, per-task
configuration equality, pinned repository heads, source hashes, evaluation
history, and checkpoint existence, then emits deterministic JSON.  It does not
import or execute either training harness.

By default it expects ``results/fixed/{ant,humanoid}_seed{17,29,41}.json``::

    python shac/pr1535/summarize_fixed_results.py \
      --output shac/pr1535/results/fixed/summary.json

Explicit inputs can instead be supplied with three paths for each task::

    python shac/pr1535/summarize_fixed_results.py \
      --ant-json ANT17.json ANT29.json ANT41.json \
      --humanoid-json HUMAN17.json HUMAN29.json HUMAN41.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results" / "fixed"
EXPECTED_SCHEMA = "mjwarp-pr1535-shac-style-v2"
DEFAULT_PR_HEAD = "02d09b139fdf091e1e859d7f41c47a8f71d30574"
DEFAULT_NEWTON_HEAD = "d37f4d3d341ccce1e06a1dff21e9a054759b4855"
TASKS = ("ant", "humanoid")
POLICIES = ("initial", "selected_best")
METRICS = (
    "mean_return",
    "mean_displacement",
    "final_alive_fraction",
    "mean_alive_fraction",
    "mean_minimum_height",
    "mean_action_rms",
)
CONFIG_RUN_SPECIFIC_KEYS = frozenset({"seed", "output"})


class ValidationError(ValueError):
    """Raised when an input is not part of the requested six-run matrix."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _object(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _array(value: Any, label: str) -> list[Any]:
    _require(isinstance(value, list), f"{label} must be a JSON array")
    return value


def _integer(value: Any, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{label} must be an integer",
    )
    return value


def _finite_number(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{label} must be finite")
    return result


def _resolve_checkpoint(source_json: Path, recorded_path: Any) -> Path:
    _require(
        isinstance(recorded_path, str) and recorded_path,
        f"{source_json}: run.checkpoint must be a non-empty path",
    )
    path = Path(recorded_path).expanduser()
    if not path.is_absolute():
        path = source_json.parent / path
    return path.resolve()


def _normalized_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in config.items()
        if key not in CONFIG_RUN_SPECIFIC_KEYS
    }


def _validate_metrics(metrics: Any, label: str) -> dict[str, Any]:
    result = _object(metrics, label)
    for field in METRICS:
        _finite_number(result.get(field), f"{label}.{field}")
    _integer(result.get("seed"), f"{label}.seed")
    _integer(result.get("steps"), f"{label}.steps")
    _integer(result.get("worlds"), f"{label}.worlds")
    return result


def _validate_evaluations(
    evaluations_value: Any,
    *,
    label: str,
    epochs_completed: int,
    selected_epoch: int,
) -> list[dict[str, Any]]:
    evaluations_raw = _array(evaluations_value, label)
    _require(evaluations_raw, f"{label} must not be empty")
    evaluations: list[dict[str, Any]] = []
    previous_epoch = -1
    for index, raw in enumerate(evaluations_raw):
        entry = _object(raw, f"{label}[{index}]")
        epoch = _integer(entry.get("epoch"), f"{label}[{index}].epoch")
        _require(epoch > previous_epoch, f"{label} epochs must be strictly increasing")
        _validate_metrics(entry.get("metrics"), f"{label}[{index}].metrics")
        evaluations.append(entry)
        previous_epoch = epoch
    epochs = [entry["epoch"] for entry in evaluations]
    _require(epochs[0] == 0, f"{label} must begin at epoch 0")
    _require(
        epochs[-1] == epochs_completed,
        f"{label} must end at completed epoch {epochs_completed}",
    )
    _require(
        selected_epoch in epochs,
        f"{label} does not contain selected epoch {selected_epoch}",
    )
    return evaluations


def _validate_source_hashes(
    provenance: dict[str, Any], *, source_json: Path
) -> dict[str, dict[str, str]]:
    expected_sources = {
        "v2_trainer": (
            SCRIPT_DIR / "train_shac_v2.py",
            provenance.get("script_sha256"),
        ),
        "v1_base_harness": (
            SCRIPT_DIR / "train_shac.py",
            _object(
                provenance.get("base_harness"),
                f"{source_json}: provenance.base_harness",
            ).get("sha256"),
        ),
        "torch_bridge": (
            SCRIPT_DIR / "mjwarp_torch_bridge.py",
            provenance.get("bridge_sha256"),
        ),
    }
    result: dict[str, dict[str, str]] = {}
    for name, (path, recorded_hash) in expected_sources.items():
        _require(path.is_file(), f"Required source does not exist: {path}")
        _require(
            isinstance(recorded_hash, str) and len(recorded_hash) == 64,
            f"{source_json}: invalid recorded {name} SHA256",
        )
        actual_hash = _sha256(path)
        _require(
            actual_hash == recorded_hash,
            f"{source_json}: {name} hash is {recorded_hash}, current source is {actual_hash}",
        )
        result[name] = {"path": str(path), "sha256": actual_hash}
    return result


def _load_run(
    source_json: Path,
    *,
    expected_task: str,
    expected_pr_head: str,
    expected_newton_head: str,
) -> dict[str, Any]:
    source_json = source_json.expanduser().resolve()
    _require(source_json.is_file(), f"Missing input JSON: {source_json}")

    def reject_nonfinite_constant(value: str) -> None:
        raise ValidationError(f"{source_json}: non-finite JSON constant {value!r}")

    try:
        payload = json.loads(
            source_json.read_text(), parse_constant=reject_nonfinite_constant
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"Could not read {source_json}: {error}") from error
    payload = _object(payload, str(source_json))
    _require(
        payload.get("schema") == EXPECTED_SCHEMA,
        f"{source_json}: expected schema {EXPECTED_SCHEMA!r}",
    )
    _require(payload.get("mode") == "train", f"{source_json}: mode is not train")

    config = _object(payload.get("config"), f"{source_json}: config")
    _require(config.get("mode") == "train", f"{source_json}: config.mode is not train")
    _require(
        config.get("task") == expected_task,
        f"{source_json}: config.task is not {expected_task}",
    )
    seed = _integer(config.get("seed"), f"{source_json}: config.seed")

    provenance = _object(payload.get("provenance"), f"{source_json}: provenance")
    pr = _object(provenance.get("pr"), f"{source_json}: provenance.pr")
    _require(
        pr.get("head") == expected_pr_head,
        f"{source_json}: PR head {pr.get('head')!r} != {expected_pr_head}",
    )
    _require(
        pr.get("exact_worktree_import") is True,
        f"{source_json}: provenance does not confirm the exact PR worktree import",
    )
    _require(
        provenance.get("newton_head") == expected_newton_head,
        f"{source_json}: Newton head {provenance.get('newton_head')!r} != {expected_newton_head}",
    )
    model = _object(provenance.get("model"), f"{source_json}: provenance.model")
    _require(
        model.get("task") == expected_task,
        f"{source_json}: provenance.model.task is not {expected_task}",
    )
    source_hashes = _validate_source_hashes(provenance, source_json=source_json)

    run = _object(payload.get("run"), f"{source_json}: run")
    _require(
        run.get("status") == "completed",
        f"{source_json}: run.status is not completed",
    )
    _require(run.get("cold_start") is True, f"{source_json}: run is not cold-start")
    _require(
        run.get("canonical_shac") is False,
        f"{source_json}: canonical_shac label changed",
    )
    epochs_completed = _integer(
        run.get("epochs_completed"), f"{source_json}: run.epochs_completed"
    )
    _require(
        epochs_completed
        == _integer(config.get("epochs"), f"{source_json}: config.epochs"),
        f"{source_json}: completed epoch count differs from config",
    )

    holdout = _object(run.get("holdout"), f"{source_json}: run.holdout")
    _require(
        holdout.get("selection_independent") is True,
        f"{source_json}: holdout is not selection-independent",
    )
    selected_epoch = _integer(
        holdout.get("selected_epoch"), f"{source_json}: holdout.selected_epoch"
    )
    _require(
        0 <= selected_epoch <= epochs_completed,
        f"{source_json}: selected epoch is outside the completed range",
    )
    holdout_initial = _validate_metrics(
        holdout.get("initial"), f"{source_json}: run.holdout.initial"
    )
    holdout_selected = _validate_metrics(
        holdout.get("selected_best"),
        f"{source_json}: run.holdout.selected_best",
    )
    _require(
        holdout_initial["seed"] == holdout_selected["seed"] == seed + 20_000,
        f"{source_json}: holdout seeds are not the paired seed {seed + 20_000}",
    )
    for field in (
        "steps",
        "worlds",
        "noise_profile",
        "control_dt",
        "simulated_seconds",
    ):
        _require(
            holdout_initial.get(field) == holdout_selected.get(field),
            f"{source_json}: holdout {field} differs between paired policies",
        )

    evaluations = _validate_evaluations(
        run.get("evaluations"),
        label=f"{source_json}: run.evaluations",
        epochs_completed=epochs_completed,
        selected_epoch=selected_epoch,
    )
    metrics_summary = _object(run.get("metrics"), f"{source_json}: run.metrics")
    best_summary = _object(
        metrics_summary.get("best"), f"{source_json}: run.metrics.best"
    )
    _require(
        best_summary.get("epoch") == selected_epoch,
        f"{source_json}: metrics.best epoch differs from holdout selected epoch",
    )

    checkpoint = _resolve_checkpoint(source_json, run.get("checkpoint"))
    _require(
        checkpoint.is_file(), f"Missing checkpoint for {source_json}: {checkpoint}"
    )
    _require(checkpoint != source_json, f"{source_json}: checkpoint aliases its JSON")

    return {
        "task": expected_task,
        "seed": seed,
        "source_json": source_json,
        "source_json_sha256": _sha256(source_json),
        "checkpoint": checkpoint,
        "checkpoint_sha256": _sha256(checkpoint),
        "config": config,
        "normalized_config": _normalized_config(config),
        "provenance": provenance,
        "source_hashes": source_hashes,
        "holdout": {
            "selected_epoch": selected_epoch,
            "initial": holdout_initial,
            "selected_best": holdout_selected,
        },
        "evaluations": evaluations,
    }


def _population_summary(values_by_seed: dict[int, float]) -> dict[str, Any]:
    _require(values_by_seed, "Cannot summarize an empty value set")
    ordered = {str(seed): values_by_seed[seed] for seed in sorted(values_by_seed)}
    values = list(ordered.values())
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
    return {
        "by_seed": ordered,
        "mean": mean,
        "population_std": math.sqrt(variance),
    }


def _aggregate_holdout(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in METRICS:
        initial = {
            record["seed"]: _finite_number(
                record["holdout"]["initial"][metric],
                f"{record['source_json']}: holdout.initial.{metric}",
            )
            for record in records
        }
        selected = {
            record["seed"]: _finite_number(
                record["holdout"]["selected_best"][metric],
                f"{record['source_json']}: holdout.selected_best.{metric}",
            )
            for record in records
        }
        paired_delta = {
            seed: selected[seed] - initial[seed] for seed in sorted(initial)
        }
        initial_summary = _population_summary(initial)
        selected_summary = _population_summary(selected)
        delta_summary = _population_summary(paired_delta)
        if abs(initial_summary["mean"]) > 1.0e-12:
            delta_summary["percent_change_of_initial_mean"] = (
                100.0 * delta_summary["mean"] / initial_summary["mean"]
            )
        result[metric] = {
            "initial": initial_summary,
            "selected_best": selected_summary,
            "paired_delta_selected_minus_initial": delta_summary,
        }
    return result


def _aggregate_evaluation_history(records: list[dict[str, Any]]) -> dict[str, Any]:
    epoch_sequences = [
        [entry["epoch"] for entry in record["evaluations"]] for record in records
    ]
    reference_epochs = epoch_sequences[0]
    for record, epochs in zip(records[1:], epoch_sequences[1:], strict=True):
        _require(
            epochs == reference_epochs,
            f"{record['source_json']}: evaluation epoch schedule is inconsistent",
        )

    aggregate: list[dict[str, Any]] = []
    for index, epoch in enumerate(reference_epochs):
        metric_summaries: dict[str, Any] = {}
        for metric in METRICS:
            values = {
                record["seed"]: _finite_number(
                    record["evaluations"][index]["metrics"][metric],
                    f"{record['source_json']}: evaluation {epoch} {metric}",
                )
                for record in records
            }
            metric_summaries[metric] = _population_summary(values)
        aggregate.append({"epoch": epoch, "metrics": metric_summaries})

    per_seed = {
        str(record["seed"]): record["evaluations"]
        for record in sorted(records, key=lambda item: item["seed"])
    }
    return {
        "epochs": reference_epochs,
        "aggregate_by_epoch": aggregate,
        "per_seed": per_seed,
    }


def _validate_task_matrix(
    records: list[dict[str, Any]], *, task: str, expected_seeds: tuple[int, ...]
) -> None:
    _require(len(records) == 3, f"{task}: exactly three runs are required")
    actual_seeds = tuple(sorted(record["seed"] for record in records))
    _require(
        len(set(actual_seeds)) == 3,
        f"{task}: run seeds must be unique, got {actual_seeds}",
    )
    _require(
        actual_seeds == expected_seeds,
        f"{task}: expected seeds {expected_seeds}, got {actual_seeds}",
    )
    reference_config = records[0]["normalized_config"]
    for record in records[1:]:
        _require(
            record["normalized_config"] == reference_config,
            f"{task}: config for seed {record['seed']} differs beyond seed/output",
        )

    reference_sources = records[0]["source_hashes"]
    for record in records[1:]:
        _require(
            record["source_hashes"] == reference_sources,
            f"{task}: recorded source hashes differ across seeds",
        )


def _aggregate_task(records: list[dict[str, Any]]) -> dict[str, Any]:
    records = sorted(records, key=lambda item: item["seed"])
    selected_epochs = {
        record["seed"]: record["holdout"]["selected_epoch"] for record in records
    }
    return {
        "run_count": len(records),
        "seeds": [record["seed"] for record in records],
        "consistent_config": records[0]["normalized_config"],
        "consistent_config_sha256": _canonical_sha256(records[0]["normalized_config"]),
        "selected_epochs": _population_summary(selected_epochs),
        "holdout": {
            "comparison": "selected_best minus initial, paired by holdout seed",
            "metrics": _aggregate_holdout(records),
        },
        "evaluation_history": _aggregate_evaluation_history(records),
    }


def _default_paths(results_dir: Path, task: str, seeds: Iterable[int]) -> list[Path]:
    return [results_dir / f"{task}_seed{seed}.json" for seed in seeds]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--ant-json", type=Path, nargs=3)
    parser.add_argument("--humanoid-json", type=Path, nargs=3)
    parser.add_argument("--expected-seeds", type=int, nargs=3, default=[17, 29, 41])
    parser.add_argument("--expected-pr-head", default=DEFAULT_PR_HEAD)
    parser.add_argument("--expected-newton-head", default=DEFAULT_NEWTON_HEAD)
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_RESULTS_DIR / "summary.json"
    )
    args = parser.parse_args()
    _require(
        len(set(args.expected_seeds)) == 3,
        "--expected-seeds must contain three unique integers",
    )
    return args


def main() -> None:
    try:
        args = parse_args()
        results_dir = args.results_dir.expanduser().resolve()
        expected_seeds = tuple(sorted(args.expected_seeds))
        paths_by_task = {
            "ant": args.ant_json
            if args.ant_json is not None
            else _default_paths(results_dir, "ant", expected_seeds),
            "humanoid": args.humanoid_json
            if args.humanoid_json is not None
            else _default_paths(results_dir, "humanoid", expected_seeds),
        }
        flat_paths = [
            path.expanduser().resolve()
            for task in TASKS
            for path in paths_by_task[task]
        ]
        _require(len(flat_paths) == 6, "Exactly six source JSON paths are required")
        _require(len(set(flat_paths)) == 6, "Source JSON paths must be unique")

        records_by_task: dict[str, list[dict[str, Any]]] = {}
        for task in TASKS:
            records = [
                _load_run(
                    path,
                    expected_task=task,
                    expected_pr_head=args.expected_pr_head,
                    expected_newton_head=args.expected_newton_head,
                )
                for path in paths_by_task[task]
            ]
            records.sort(key=lambda item: item["seed"])
            _validate_task_matrix(records, task=task, expected_seeds=expected_seeds)
            records_by_task[task] = records

        all_records = [record for task in TASKS for record in records_by_task[task]]
        reference_sources = all_records[0]["source_hashes"]
        for record in all_records[1:]:
            _require(
                record["source_hashes"] == reference_sources,
                "Recorded training source hashes differ across the six-run matrix",
            )

        output = args.output.expanduser().resolve()
        _require(
            output not in set(flat_paths),
            "Output path must not overwrite an input JSON",
        )
        checkpoints = {record["checkpoint"] for record in all_records}
        _require(
            output not in checkpoints, "Output path must not overwrite a checkpoint"
        )

        payload = {
            "schema": "mjwarp-pr1535-shac-style-v2-fixed-summary-v1",
            "deterministic": True,
            "validation": {
                "all_six_runs_completed": True,
                "exactly_three_runs_per_task": True,
                "expected_seeds": list(expected_seeds),
                "per_task_config_consistent_ignoring": sorted(CONFIG_RUN_SPECIFIC_KEYS),
                "expected_pr_head": args.expected_pr_head,
                "all_exact_pr_worktree_imports": True,
                "expected_newton_head": args.expected_newton_head,
                "all_checkpoints_exist": True,
                "all_recorded_source_hashes_match_current_files": True,
            },
            "sources": reference_sources,
            "inputs": [
                {
                    "task": record["task"],
                    "seed": record["seed"],
                    "source_json": str(record["source_json"]),
                    "source_json_sha256": record["source_json_sha256"],
                    "checkpoint": str(record["checkpoint"]),
                    "checkpoint_sha256": record["checkpoint_sha256"],
                }
                for record in all_records
            ],
            "tasks": {task: _aggregate_task(records_by_task[task]) for task in TASKS},
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        print(json.dumps(payload["tasks"], indent=2, sort_keys=True))
        print(f"Wrote {output}")
    except ValidationError as error:
        raise SystemExit(f"validation failed: {error}") from error


if __name__ == "__main__":
    main()
