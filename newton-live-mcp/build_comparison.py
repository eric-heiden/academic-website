"""Audit and export the registered three-condition Newton comparison without running trials.

Only saved JSON/text evidence is read. Private reference archives, connection
files and server logs are never opened. Statistics are paired by registered case,
conditioned on both trials passing all eligibility and physical-quality gates.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime
import hashlib
import itertools
import json
import math
import re
from pathlib import Path

import numpy as np

USAGE_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
)
COST_FIELDS = (
    "startup_inclusive_seconds",
    "input_output_tokens",
    "uncached_input_output_tokens",
    "cached_input_tokens",
    "candidate_rollouts",
    "simulation_process_starts",
    "verification_process_starts",
    "failed_candidates",
    "failed_commands",
    "mcp_errors",
)
PAIR_FIELDS = (
    "startup_inclusive_seconds",
    "input_output_tokens",
    "uncached_input_output_tokens",
)
PRIMARY_CONDITIONS = ("live", "restart", "ipython")
LOG_NAMES = {"rollouts.jsonl", "live_rollouts.jsonl", "process_events.jsonl"}
SAFE_ROOT_FILES = {
    "summary.json",
    "task.json",
    "TASK.md",
    "config.json",
    "config.py",
    "agent.jsonl",
    "agent.stderr",
    "integrity-manifest.json",
    "integrity_audit.json",
    "ipython_environment.json",
    "provenance.json",
    "verification.log",
}
SECRET_KEYS = {
    "key",
    "token",
    "access_token",
    "refresh_token",
    "api_key",
    "apikey",
    "password",
    "secret",
    "authorization",
    "connection_file",
    "kernel_file",
    "verification_reference_file",
    "private_input_hashes",
    "truth_parameters",
    "authkey",
    "auth_key",
    "hmac_key",
    "signing_key",
    "signature_key",
}
REDACTION = "[redacted]"
MAX_EXPORT_BYTES = 95 * 1024 * 1024
MAX_EXPORT_LINE_BYTES = 16 * 1024 * 1024


def sha256(path: Path) -> str:
    """Hash an existing evidence file without executing its contents."""
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(data)
    return hasher.hexdigest()


def load_json(path: Path):
    """Read one JSON evidence object."""
    return json.loads(path.read_text())


def write_json(path: Path, value) -> None:
    """Write strict JSON atomically after recursive sanitization."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(sanitize(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sanitize(value):
    """Redact credentials in mappings, nested JSON and textual IPython connection replies."""
    if isinstance(value, dict):
        if value.get("type") in {"image", "image_url", "input_image"}:
            return {"type": value["type"], "omitted": "Embedded image payload"}
        return {
            key: REDACTION if str(key).lower() in SECRET_KEYS else sanitize(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if not isinstance(value, str):
        return value
    try:
        nested = json.loads(value)
    except (ValueError, TypeError):
        nested = None
    if isinstance(nested, (dict, list)):
        return json.dumps(sanitize(nested), ensure_ascii=False)
    value = re.sub(
        r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", "[image payload omitted]", value
    )
    value = re.sub(
        r"(?i)([\"']?\b(?:key|authkey|auth_key|hmac_key|signing_key|signature_key|token|access_token|refresh_token|api_key|apikey|password|secret|authorization)[\"']?\s*[:=]\s*)(?:b)?(?:'[^'\n]*'|\"[^\"\n]*\"|[^\s,;}]+)",
        r"\1[redacted]",
        value,
    )
    value = re.sub(r"(?i)Bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [redacted]", value)
    value = re.sub(
        r"\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b",
        REDACTION,
        value,
    )
    value = re.sub(
        r"(?:/[^\s\"'<>]*)/(?:private|calibration-private)/[^\s\"'<>]*",
        "[private artifact path]",
        value,
    )
    if len(value) > 4096 and re.fullmatch(r"[A-Za-z0-9+/=\s]+", value):
        return "[base64 payload omitted]"
    return value


def json_lines(path: Path) -> tuple[list[dict], list[str]]:
    """Preserve line counts and report malformed records instead of dropping them silently."""
    rows, issues = [], []
    if not path.is_file():
        return rows, [f"Missing {path.name}"]
    with path.open() as stream:
        for index, line in enumerate(stream, 1):
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise TypeError("record is not an object")
                rows.append(record)
            except (ValueError, TypeError):
                issues.append(f"Malformed JSON record {path.name}:{index}")
    return rows, issues


def audit_events(workspace: Path) -> dict:
    """Recompute token subsets, tool calls and error indications from raw agent events."""
    events, issues = json_lines(workspace / "agent.jsonl")
    usage = dict.fromkeys(USAGE_FIELDS, 0)
    counts, failed_commands, mcp_errors = collections.Counter(), [], []
    completed_turns, turn_errors = 0, 0
    for event in events:
        if event.get("type") == "turn.completed":
            completed_turns += 1
            for field in USAGE_FIELDS:
                value = event.get("usage", {}).get(field, 0)
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    issues.append(f"Invalid event usage {field}")
                else:
                    usage[field] += value
        if event.get("type") in {"turn.failed", "error"}:
            turn_errors += 1
        if event.get("type") != "item.completed":
            continue
        item = event.get("item", {})
        kind = item.get("type", "unknown")
        counts[kind] += 1
        if kind == "command_execution" and item.get("exit_code") not in (None, 0):
            failed_commands.append(item.get("id"))
        if kind == "mcp_tool_call":
            result = item.get("result") or {}
            texts = (
                [
                    part.get("text", "")
                    for part in result.get("content", [])
                    if part.get("type") == "text"
                ]
                if isinstance(result, dict)
                else []
            )
            if (
                item.get("error")
                or item.get("status") == "failed"
                or (isinstance(result, dict) and result.get("isError"))
                or any(text.lstrip().startswith("❌") for text in texts)
            ):
                mcp_errors.append(item.get("id"))
    if (
        usage["cached_input_tokens"] > usage["input_tokens"]
        or usage["reasoning_output_tokens"] > usage["output_tokens"]
    ):
        issues.append("Token subset exceeds its containing total")
    usage["uncached_input_tokens"] = (
        usage["input_tokens"] - usage["cached_input_tokens"]
    )
    usage["input_output_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    usage["uncached_input_output_tokens"] = (
        usage["uncached_input_tokens"] + usage["output_tokens"]
    )
    return {
        "usage": usage,
        "usage_observed": completed_turns > 0,
        "completed_turns": completed_turns,
        "items_by_type": dict(counts),
        "tool_items": sum(
            counts[key] for key in ("command_execution", "mcp_tool_call", "tool_call")
        ),
        "mcp_tool_items": counts["mcp_tool_call"],
        "failed_command_ids": failed_commands,
        "mcp_error_indication_ids": mcp_errors,
        "turn_error_events": turn_errors,
        "issues": issues,
    }


def audit_candidates(workspace: Path) -> dict:
    """Recount all retained candidate rows and distinguish trial and verifier processes."""
    candidates, process_rows, verifier_rows, issues, seen = [], [], [], [], set()
    for path in sorted(workspace.rglob("*.jsonl")):
        if path.name not in LOG_NAMES or path.is_symlink():
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(workspace.resolve()) or resolved in seen:
            continue
        seen.add(resolved)
        records, errors = json_lines(path)
        issues.extend(errors)
        relative = path.relative_to(workspace)
        verifier = relative.parts[0] == "verification"
        if path.name == "process_events.jsonl":
            for index, record in enumerate(records):
                if record.get("event") != "simulation_process_start":
                    issues.append(f"Unexpected process event in {relative}")
                (verifier_rows if verifier else process_rows).append(
                    {"path": str(relative), "index": index, "record": record}
                )
        elif not verifier:
            for index, record in enumerate(records):
                candidates.append(
                    {"path": str(relative), "index": index, "measurement": record}
                )
    return {
        "candidates": candidates,
        "candidate_rollouts": len(candidates),
        "failed_candidates": sum(
            not row["measurement"].get("success", False) for row in candidates
        ),
        "simulation_process_starts": len(process_rows),
        "verification_process_starts": len(verifier_rows),
        "processes": process_rows,
        "verifier_processes": verifier_rows,
        "issues": issues,
    }


def finite_number(value) -> bool:
    """Recognize finite, nonnegative numeric measurement fields."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def physical_success(
    quality: dict, thresholds: dict, expected_frames: int, *, real: bool = False
) -> bool:
    """Independently check full finite physics and all registered numerical quality gates."""
    if (
        not quality.get("success")
        or quality.get("finite") is not True
        or quality.get("frames") != expected_frames
    ):
        return False
    if quality.get("sample_count") != expected_frames:
        return False
    if any(
        not finite_number(quality.get(key)) or quality[key] > limit
        for key, limit in thresholds.items()
    ):
        return False
    if quality.get("thresholds") != thresholds:
        return False
    episodes = quality.get("per_episode", [])
    if episodes and [item.get("episode") for item in episodes] != quality.get(
        "episodes"
    ):
        return False
    if real and (
        len(episodes) != expected_frames // 600
        or any(item.get("sample_count") != 600 for item in episodes)
    ):
        return False
    for item in episodes:
        required = (
            thresholds
            if real
            else {key: limit for key, limit in thresholds.items() if key in item}
        )
        if any(
            not finite_number(item.get(key)) or item[key] > limit
            for key, limit in required.items()
        ):
            return False
    return True


def trial_record(entry: dict, registration: dict) -> dict:
    """Audit one registered workspace without interpreting submitted executable code."""
    row = {
        key: entry[key] for key in ("id", "scenario", "variant", "condition", "cohort")
    }
    row.update(
        status="missing",
        eligible=False,
        success=False,
        eligible_success=False,
        issues=[],
        failures=[],
    )
    workspace = Path(entry["workspace"]).resolve()
    if not (workspace / "summary.json").is_file():
        row["failures"].append("No completed summary")
        return row
    try:
        summary, task = (
            load_json(workspace / "summary.json"),
            load_json(workspace / "task.json"),
        )
    except (OSError, ValueError) as error:
        row.update(status="unreadable", issues=[f"Unreadable summary/task: {error}"])
        return row
    row["status"] = "completed"
    issues, failures = row["issues"], row["failures"]
    for key in ("scenario", "variant", "condition"):
        if summary.get(key) != entry[key] or task.get(key) != entry[key]:
            issues.append(f"Registered identity mismatch: {key}")
    for key in ("model", "reasoning_effort"):
        if summary.get(key) != registration.get(key):
            issues.append(f"Registered model mismatch: {key}")
    if summary.get("phase") != registration.get("phase", "confirmation"):
        failures.append("Non-confirmation phase")
    for name, key in (
        ("task.json", "task_sha256"),
        ("TASK.md", "prompt_sha256"),
        ("integrity-manifest.json", "integrity_manifest_sha256"),
    ):
        path = workspace / name
        if not entry.get(key) or not path.is_file() or sha256(path) != entry[key]:
            issues.append(f"Registered file hash mismatch: {name}")
    try:
        manifest = load_json(workspace / "integrity-manifest.json")
    except (OSError, ValueError):
        manifest = {}
    sources = manifest.get("source_hashes", {})
    if not sources or summary.get("task_source_hashes") != sources:
        issues.append("Summary source identity differs from frozen integrity manifest")
    expected_sources = entry.get("source_hashes", registration.get("source_hashes"))
    if expected_sources is not None and sources != expected_sources:
        issues.append("Executable source hashes differ from registration")
    if task.get("integrity_manifest_sha256") != entry.get("integrity_manifest_sha256"):
        issues.append("Task integrity-manifest commitment differs from registration")
    row["source_commit"] = entry.get("source_commit", registration.get("source_commit"))
    row["source_identity_sha256"] = hashlib.sha256(
        json.dumps(sources, sort_keys=True).encode()
    ).hexdigest()
    row["reference_identity"] = {
        key: task[key]
        for key in (
            "reference_sha256",
            "verification_reference_sha256",
            "input_hashes",
            "geometry_hashes",
        )
        if key in task
    }
    for flag in (
        "shared_sources_unchanged",
        "task_unchanged",
        "external_sources_unchanged",
    ):
        if summary.get(flag) is not True:
            failures.append(flag)
    for flag in (
        "references_unchanged",
        "geometry_unchanged",
        "integrity_manifest_unchanged",
    ):
        required = (
            (
                entry["scenario"] in {"panda_real", "panda_calibration"}
                and flag == "references_unchanged"
            )
            or (entry["scenario"] == "panda_real" and flag == "geometry_unchanged")
            or flag in summary
        )
        if required and summary.get(flag) is not True:
            failures.append(flag)
    events, measurements = audit_events(workspace), audit_candidates(workspace)
    issues.extend(events["issues"] + measurements["issues"])
    for key in USAGE_FIELDS:
        if summary.get("usage", {}).get(key, 0) != events["usage"][key]:
            issues.append(f"Raw token accounting mismatch: {key}")
    for key in (
        "tool_items",
        "mcp_tool_items",
        "failed_command_ids",
        "mcp_error_indication_ids",
    ):
        if summary.get(key) != events[key]:
            issues.append(f"Raw action accounting mismatch: {key}")
    for summary_key, recomputed_key in (
        ("candidate_rollouts", "candidate_rollouts"),
        ("simulation_process_starts_during_trial", "simulation_process_starts"),
        ("verification_process_starts", "verification_process_starts"),
    ):
        if summary.get(summary_key) != measurements[recomputed_key]:
            issues.append(f"Raw process/candidate accounting mismatch: {summary_key}")
    row.update(events["usage"])
    row.update(
        {
            key: measurements[key]
            for key in (
                "candidate_rollouts",
                "failed_candidates",
                "simulation_process_starts",
                "verification_process_starts",
            )
        }
    )
    row.update(
        usage_observed=events["usage_observed"],
        failed_commands=len(events["failed_command_ids"]),
        mcp_errors=len(events["mcp_error_indication_ids"]),
        turn_error_events=events["turn_error_events"],
        tool_items=events["tool_items"],
        mcp_tool_items=events["mcp_tool_items"],
    )
    row["usage_complete"] = bool(
        events["usage_observed"]
        and summary.get("timed_out") is False
        and summary.get("exit_code") == 0
        and events["turn_error_events"] == 0
    )
    for key in ("agent_elapsed_seconds", "startup_inclusive_seconds"):
        row[key] = summary.get(key)
    startup = summary.get(
        "application_startup_seconds", summary.get("live_startup_seconds")
    )
    row["application_startup_seconds"] = startup
    if not all(
        finite_number(row.get(key))
        for key in (
            "agent_elapsed_seconds",
            "startup_inclusive_seconds",
            "application_startup_seconds",
        )
    ) or not math.isclose(
        row["startup_inclusive_seconds"],
        row["agent_elapsed_seconds"] + startup,
        abs_tol=1e-6,
    ):
        issues.append("Inconsistent startup-inclusive timing")
    if summary.get("timed_out") is not False or summary.get("exit_code") != 0:
        failures.append("Agent timeout or nonzero exit")
    if finite_number(row["agent_elapsed_seconds"]) and row[
        "agent_elapsed_seconds"
    ] > task.get("budget_seconds", -1):
        failures.append("Agent time budget exceeded")
    if (
        row["candidate_rollouts"] > task.get("candidate_budget", 12)
        or summary.get("within_candidate_budget") is not True
    ):
        failures.append("Candidate budget exceeded")
    thresholds = task.get("thresholds", {})
    frames = 9600 if entry["scenario"] == "panda_real" else 1500
    try:
        verification = load_json(workspace / "verification/metrics.json")
    except (OSError, ValueError):
        verification = {}
    physics = physical_success(
        verification, thresholds, frames, real=entry["scenario"] == "panda_real"
    )
    if entry["scenario"] == "panda_real":
        physics &= verification.get("episodes") == [
            21,
            22,
            23,
            25,
            26,
            27,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
        ]
    elif entry["scenario"] == "panda_calibration":
        physics &= (
            verification.get("episodes") == [2]
            and len(verification.get("per_episode", [])) == 1
        )
    final_config = verification.get("config")
    if entry["scenario"] == "panda_real":
        try:
            if load_json(workspace / "config.json") != final_config:
                issues.append(
                    "Submitted numeric configuration differs from independent verifier"
                )
        except (OSError, ValueError):
            issues.append("Submitted numeric configuration is missing or unreadable")
    if summary.get("quality", {}).get("config") != final_config:
        issues.append("Summary and independent verifier configuration mismatch")
    if entry["scenario"] in {"panda_real", "panda_calibration"}:
        training_frames = 1800 if entry["scenario"] == "panda_real" else 3000
        matched = [
            item["measurement"]
            for item in measurements["candidates"]
            if item["measurement"].get("config") == final_config
            and physical_success(
                item["measurement"],
                thresholds,
                training_frames,
                real=entry["scenario"] == "panda_real",
            )
        ]
        expected_episodes = [2, 3, 4] if entry["scenario"] == "panda_real" else [0, 1]
        matched = [
            item
            for item in matched
            if item.get("episodes") == expected_episodes
            and len(item.get("per_episode", [])) == len(expected_episodes)
        ]
        physics &= bool(matched)
        if entry["scenario"] == "panda_real":
            try:
                training = load_json(workspace / "verification/training/metrics.json")
            except (OSError, ValueError):
                training = {}
            physics &= (
                training.get("config") == final_config
                and training.get("episodes") == expected_episodes
                and physical_success(training, thresholds, 1800, real=True)
            )
    row["physics_success"] = bool(physics)
    if not physics:
        failures.append("Independent physical quality or matching training failed")
    reported_success = bool(
        summary.get("study_success", summary.get("quality", {}).get("success"))
    )
    row["reported_success"] = reported_success
    row["eligible"] = not issues and not [
        failure
        for failure in failures
        if failure != "Independent physical quality or matching training failed"
    ]
    row["success"] = bool(physics and reported_success)
    row["eligible_success"] = bool(row["eligible"] and row["success"])
    row["candidate_history"] = measurements["candidates"]
    row["quality"] = verification
    row["summary_sha256"] = sha256(workspace / "summary.json")
    return row


def paired_statistics(
    numerator, denominator, *, bootstrap_samples: int = 20000, seed: int = 20260920
) -> dict:
    """Calculate B/A geometric ratios with exact paired tests and paired bootstrap intervals."""
    numerator, denominator = (
        np.asarray(numerator, float),
        np.asarray(denominator, float),
    )
    if (
        numerator.shape != denominator.shape
        or numerator.ndim != 1
        or not len(numerator)
    ):
        return {"n": 0, "geometric_ratio": None, "bootstrap_95_interval": None}
    if (
        not np.isfinite(numerator).all()
        or not np.isfinite(denominator).all()
        or np.any(numerator <= 0)
        or np.any(denominator <= 0)
    ):
        raise ValueError("Paired ratios require finite positive costs")
    logs = np.log(numerator / denominator)
    positive, negative = int(np.sum(logs > 1e-12)), int(np.sum(logs < -1e-12))
    non_ties = positive + negative
    sign_p = (
        min(
            1.0,
            2
            * sum(math.comb(non_ties, k) for k in range(min(positive, negative) + 1))
            / 2**non_ties,
        )
        if non_ties
        else 1.0
    )
    if len(logs) > 20:
        raise ValueError(
            "Exact paired permutation is bounded to at most20 pairs; report smaller registered cohorts"
        )
    observed = abs(float(logs.mean()))
    extreme = sum(
        abs(float(np.dot(signs, logs) / len(logs))) >= observed - 1e-12
        for signs in itertools.product((-1.0, 1.0), repeat=len(logs))
    )
    rng = np.random.default_rng(seed)
    bootstrap = np.exp(
        logs[rng.integers(0, len(logs), size=(bootstrap_samples, len(logs)))].mean(
            axis=1
        )
    )
    return {
        "n": len(logs),
        "geometric_ratio": float(np.exp(logs.mean())),
        "aggregate_ratio": float(numerator.sum() / denominator.sum()),
        "paired_ratios": (numerator / denominator).tolist(),
        "denominator_wins": positive,
        "numerator_wins": negative,
        "ties": len(logs) - non_ties,
        "exact_two_sided_sign_p": sign_p,
        "exact_two_sided_log_ratio_permutation_p": extreme / 2 ** len(logs),
        "bootstrap_95_interval": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "interpretation": "Ratio >1 favors denominator condition. Exact tests assume paired exchangeability; bootstrap resamples whole registered pairs. Small-n intervals can be unstable/degenerate. These are descriptive unadjusted multiple comparisons, not universal tool claims.",
    }


def pairwise(rows: list[dict]) -> list[dict]:
    """Match exact registered cases and keep exclusions visible for each comparison."""
    comparisons = []
    for cohort in ("existing_primary", "real_primary", "sensitivity"):
        cohort_rows = [row for row in rows if row["cohort"] == cohort]
        scopes = ["all"] + sorted({row["scenario"] for row in cohort_rows})
        for scope in scopes:
            selected = [
                row for row in cohort_rows if scope == "all" or row["scenario"] == scope
            ]
            conditions = (
                list(itertools.combinations(PRIMARY_CONDITIONS, 2))
                if cohort != "sensitivity"
                else [(condition, "ipython_fixed") for condition in PRIMARY_CONDITIONS]
            )
            for denominator, numerator in conditions:
                left = {
                    (row["scenario"], row["variant"]): row
                    for row in (rows if cohort == "sensitivity" else selected)
                    if row["condition"] == denominator
                    and (
                        cohort != "sensitivity"
                        or row["cohort"] in {"real_primary", "existing_primary"}
                    )
                }
                right = {
                    (row["scenario"], row["variant"]): row
                    for row in selected
                    if row["condition"] == numerator
                }
                case_keys = sorted(
                    set(right) if cohort == "sensitivity" else set(left) | set(right)
                )
                paired, excluded = [], []
                for key in case_keys:
                    first, second = left.get(key), right.get(key)
                    reason = None
                    if first is None or second is None:
                        reason = "missing paired condition"
                    elif (
                        not first["eligible_success"] or not second["eligible_success"]
                    ):
                        reason = "one or both trials failed quality or eligibility"
                    elif first.get("source_identity_sha256") != second.get(
                        "source_identity_sha256"
                    ) or first.get("reference_identity") != second.get(
                        "reference_identity"
                    ):
                        reason = "source/reference identities differ"
                    if reason:
                        excluded.append(
                            {"scenario": key[0], "variant": key[1], "reason": reason}
                        )
                    else:
                        paired.append((first, second))
                if not case_keys:
                    continue
                metrics = {}
                for metric in PAIR_FIELDS:
                    usable = [
                        (first, second)
                        for first, second in paired
                        if (
                            metric == "startup_inclusive_seconds"
                            or first.get("usage_complete")
                            and second.get("usage_complete")
                        )
                        and finite_number(first.get(metric))
                        and finite_number(second.get(metric))
                        and first[metric] > 0
                        and second[metric] > 0
                    ]
                    metrics[metric] = paired_statistics(
                        [second[metric] for first, second in usable],
                        [first[metric] for first, second in usable],
                    )
                    metrics[metric]["pair_ids"] = [
                        [first["id"], second["id"]] for first, second in usable
                    ]
                comparisons.append(
                    {
                        "cohort": cohort,
                        "scope": scope,
                        "numerator_condition": numerator,
                        "denominator_condition": denominator,
                        "registered_pairs": len(case_keys),
                        "successful_eligible_pairs": len(paired),
                        "excluded_pairs": excluded,
                        "metrics": metrics,
                        "selection_note": "Costs are conditioned on both paired runs succeeding. See all-registered success rates and all-completed costs to assess survivor selection.",
                    }
                )
    return comparisons


def grouped_totals(rows: list[dict]) -> list[dict]:
    """Report success denominators and costs including completed failures."""
    result = []
    for cohort, condition in sorted(
        {(row["cohort"], row["condition"]) for row in rows}
    ):
        group = [
            row
            for row in rows
            if row["cohort"] == cohort and row["condition"] == condition
        ]
        for scope in ["all"] + sorted({row["scenario"] for row in group}):
            selected = [
                row for row in group if scope == "all" or row["scenario"] == scope
            ]
            completed = [row for row in selected if row["status"] == "completed"]
            totals = {
                metric: sum(
                    row[metric] for row in completed if finite_number(row.get(metric))
                )
                for metric in COST_FIELDS
            }
            coverage = {
                metric: sum(
                    finite_number(row.get(metric))
                    and ("tokens" not in metric or row.get("usage_complete", False))
                    for row in completed
                )
                for metric in COST_FIELDS
            }
            result.append(
                {
                    "cohort": cohort,
                    "condition": condition,
                    "scope": scope,
                    "registered_trials": len(selected),
                    "completed_trials": len(completed),
                    "eligible_trials": sum(row["eligible"] for row in selected),
                    "successful_eligible_trials": sum(
                        row["eligible_success"] for row in selected
                    ),
                    "success_rate_all_registered": sum(
                        row["eligible_success"] for row in selected
                    )
                    / len(selected),
                    "all_completed_costs": totals,
                    "cost_coverage_completed_trials": coverage,
                    "cost_note": "All completed rows contribute observed costs, including failures/ineligible trials. Missing final usage is not evidence of zero token expenditure.",
                }
            )
    return result


def export_evidence(workspace: Path, target: Path) -> list[dict]:
    """Export only allowlisted sanitized text, retaining hashes and explicit omissions."""
    records = []
    for path in sorted(workspace.rglob("*")):
        if (
            not path.is_file()
            or path.is_symlink()
            or not path.resolve().is_relative_to(workspace.resolve())
        ):
            continue
        relative = path.relative_to(workspace)
        forbidden = (
            any(
                word in path.name.lower() for word in ("connection", "kernel", "server")
            )
            or path.suffix in {".npz", ".npy", ".pkl"}
            or any(part in {"private", "__pycache__"} for part in relative.parts)
        )
        allowed = (
            path.suffix in {".py", ".json"}
            or (
                len(relative.parts) == 1
                and (
                    path.name in SAFE_ROOT_FILES
                    or path.suffix in {".py", ".json", ".md", ".txt", ".log"}
                )
            )
            or path.name in LOG_NAMES
            or (
                relative.parts[0] == "verification"
                and path.name in {"metrics.json", "metrics.log"}
            )
        )
        if forbidden or not allowed:
            if forbidden:
                records.append(
                    {
                        "path": str(relative),
                        "omitted": "Private connection/server/binary artifact; not opened or hashed",
                    }
                )
            continue
        record = {
            "path": str(relative),
            "raw_sha256": sha256(path),
            "raw_bytes": path.stat().st_size,
        }
        if path.stat().st_size > MAX_EXPORT_BYTES:
            records.append(record | {"omitted": "Exceeds declared export size cap"})
            continue
        output = target / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".jsonl":
            malformed, oversized, lines = 0, 0, []
            with path.open() as stream:
                for index, line in enumerate(stream, 1):
                    if len(line.encode()) > MAX_EXPORT_LINE_BYTES:
                        oversized += 1
                        item = {
                            "omitted_record": index,
                            "reason": "Oversized record",
                            "raw_line_sha256": hashlib.sha256(
                                line.encode()
                            ).hexdigest(),
                        }
                    else:
                        try:
                            item = json.loads(line)
                        except ValueError:
                            malformed += 1
                            item = {
                                "malformed_record": index,
                                "sanitized_text": line.rstrip(),
                            }
                    lines.append(json.dumps(sanitize(item), allow_nan=False))
            output.write_text("\n".join(lines) + ("\n" if lines else ""))
            record.update(
                records=len(lines),
                malformed_records=malformed,
                oversized_records=oversized,
            )
        elif path.suffix == ".json":
            try:
                write_json(output, load_json(path))
            except ValueError:
                output.write_text(sanitize(path.read_text()))
                record["malformed_json"] = True
        else:
            output.write_text(sanitize(path.read_text(errors="replace")))
        record.update(
            export_sha256=sha256(output),
            export_bytes=output.stat().st_size,
            sanitized=True,
        )
        records.append(record)
    return records


def build(
    registration_path: Path,
    output: Path,
    *,
    require_complete: bool = False,
    export: bool = False,
    strict_audit: bool = False,
) -> dict:
    """Build comparison tables only from an explicit fixed registration."""
    registration = load_json(registration_path)
    entries = registration["trials"]
    ids, identities, workspaces = set(), set(), set()
    for entry in entries:
        identity = tuple(
            entry[key] for key in ("cohort", "scenario", "variant", "condition")
        )
        workspace = str(Path(entry["workspace"]).resolve())
        if (
            entry["id"] in ids
            or identity in identities
            or workspace in workspaces
            or not re.fullmatch(r"[A-Za-z0-9_.-]+", entry["id"])
        ):
            raise ValueError(
                "Registration contains duplicate/reused/unsafe trial identities or workspaces"
            )
        ids.add(entry["id"])
        identities.add(identity)
        workspaces.add(workspace)
    rows = [trial_record(entry, registration) for entry in entries]
    if require_complete and any(row["status"] != "completed" for row in rows):
        raise ValueError(
            "Registered trials remain missing or unreadable; no final output written"
        )
    if strict_audit and any(row["issues"] for row in rows):
        raise ValueError(
            "Independent accounting/registration audit has unresolved mismatches"
        )
    for row in rows:
        competitors = [
            item
            for item in rows
            if all(item[key] == row[key] for key in ("cohort", "scenario", "variant"))
            and item["eligible_success"]
        ]
        row["best_eligible_metrics"] = [
            metric
            for metric in PAIR_FIELDS
            if row["eligible_success"]
            and len(competitors) >= 2
            and finite_number(row.get(metric))
            and (metric == "startup_inclusive_seconds" or row.get("usage_complete"))
            and round(row[metric], 2 if metric.endswith("seconds") else 0)
            == min(
                round(item[metric], 2 if metric.endswith("seconds") else 0)
                for item in competitors
                if finite_number(item.get(metric))
                and (
                    metric == "startup_inclusive_seconds" or item.get("usage_complete")
                )
            )
        ]
    result = {
        "schema_version": 1,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "registration_raw_sha256": sha256(registration_path),
        "registered_trials": len(rows),
        "completed_trials": sum(row["status"] == "completed" for row in rows),
        "successful_eligible_trials": sum(row["eligible_success"] for row in rows),
        "trials": rows,
        "groups": grouped_totals(rows),
        "pairwise": pairwise(rows),
        "definitions": {
            "startup_inclusive_seconds": "Application/kernel startup plus full agent elapsed time; fresh post-agent verification excluded",
            "input_output_tokens": "Reported input plus output; cached/reasoning subsets not added again",
            "uncached_input_output_tokens": "Input minus cached input plus output; token counts, not a monetary price estimate",
            "candidate_rollouts": "Retained completed candidate records; interrupted/partial attempts remain in raw command/MCP logs",
            "statistics": "Ratios require successful eligible same-case pairs; bootstrap resamples pairs with a fixed seed; exact two-sided sign and sign-flip tests are unadjusted descriptive comparisons",
        },
    }
    write_json(output / "comparison.json", result)
    write_json(output / "registration.json", registration)
    fields = [
        "id",
        "cohort",
        "scenario",
        "variant",
        "condition",
        "status",
        "eligible",
        "physics_success",
        "success",
        "eligible_success",
        "usage_observed",
        "usage_complete",
        "agent_elapsed_seconds",
        "application_startup_seconds",
        *COST_FIELDS,
        "input_tokens",
        "output_tokens",
        "uncached_input_tokens",
        "reasoning_output_tokens",
        "tool_items",
        "mcp_tool_items",
    ]
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    group_rows = []
    for group in result["groups"]:
        flat = {
            key: value for key, value in group.items() if not isinstance(value, dict)
        }
        flat.update(group["all_completed_costs"])
        flat.update(
            {
                key + "_coverage": value
                for key, value in group["cost_coverage_completed_trials"].items()
            }
        )
        group_rows.append(flat)
    with (output / "comparison-groups.csv").open("w", newline="") as stream:
        if group_rows:
            writer = csv.DictWriter(stream, fieldnames=list(group_rows[0]))
            writer.writeheader()
            writer.writerows(group_rows)
    if export:
        manifest = {
            entry["id"]: export_evidence(
                Path(entry["workspace"]), output / "trials" / entry["id"]
            )
            for entry in entries
            if Path(entry["workspace"]).is_dir()
        }
        write_json(output / "export-manifest.json", manifest)
    return result


def main() -> None:
    """Audit a registration and optionally create sanitized public evidence copies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--strict-audit", action="store_true")
    parser.add_argument("--export-evidence", action="store_true")
    args = parser.parse_args()
    result = build(
        args.registration,
        args.output,
        require_complete=args.require_complete,
        export=args.export_evidence,
        strict_audit=args.strict_audit,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "registered_trials",
                    "completed_trials",
                    "successful_eligible_trials",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
