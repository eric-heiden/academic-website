"""Build auditable report data from completed, registered Newton agent trials.

This script reads saved text/JSON evidence only. It never imports Newton, runs
submitted configuration code, reads connection files, or opens private reference
or truth files. Development records are outside its input and output trees.
"""

import argparse
import collections
import datetime
import hashlib
import json
import math
import re
from pathlib import Path

SECRET_KEYS = {
    "token",
    "access_token",
    "refresh_token",
    "api_key",
    "apikey",
    "password",
    "secret",
    "authorization",
}
PRIVATE_KEYS = {
    "seed",
    "truth",
    "truth_parameters",
    "connection_file",
    "verification_reference_file",
}
ROOT_FILES = {
    "summary.json",
    "task.json",
    "TASK.md",
    "config.py",
    "agent.jsonl",
    "agent.stderr",
    "verification.log",
}
LOG_NAMES = {"rollouts.jsonl", "live_rollouts.jsonl", "process_events.jsonl"}
METRICS = (
    "startup_inclusive_seconds",
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "uncached_input_tokens",
    "input_output_tokens",
)
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_LINE_BYTES = 8 * 1024 * 1024


def number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sanitize(value):
    """Remove credentials, private-content records, and embedded image payloads."""
    if isinstance(value, dict):
        if value.get("type") in {"image", "input_image", "image_url"}:
            return {
                "type": value["type"],
                "omitted": "Embedded image payload; no image data exported",
            }
        return {
            key: "[redacted]"
            if key.lower() in SECRET_KEYS | PRIVATE_KEYS
            else sanitize(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if not isinstance(value, str):
        return value
    if re.search(
        r"calibration-private|truth\.json|heldout-\d+\.npz", value, re.IGNORECASE
    ):
        return "[record containing a private calibration artifact reference omitted]"
    if len(value) > 4096 and re.fullmatch(r"[A-Za-z0-9+/=\s]+", value):
        return f"[base64 payload omitted; sha256={digest(value.encode())}]"
    try:
        nested = json.loads(value)
    except (ValueError, TypeError):
        nested = None
    if isinstance(nested, (dict, list)):
        return json.dumps(sanitize(nested), ensure_ascii=False)
    value = re.sub(
        r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", "[image data URI omitted]", value
    )
    value = re.sub(
        r"(?i)([\"']?(?:token|access_token|refresh_token|api_key|password|authorization)[\"']?\s*[:=]\s*)[\"']?[^\s,}\"']+[\"']?",
        r"\1[redacted]",
        value,
    )
    value = re.sub(r"(?i)Bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [redacted]", value)
    value = re.sub(
        r"\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})",
        "[credential redacted]",
        value,
    )
    value = re.sub(
        r"(?m)^([A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|API_KEY)[A-Z0-9_]*\s*=).+$",
        r"\1[redacted]",
        value,
    )
    return value.replace("/home/horde/apps/newton-live-mcp", "<newton>").replace(
        "/home/horde/", "<home>/"
    )


def plan_entries(registration, trial_root):
    entries = registration.get("trials", registration.get("trial_order"))
    if not isinstance(entries, list) or not entries:
        raise ValueError(
            "Registration must contain a nonempty trials or trial_order list"
        )
    normalized, seen = [], set()
    for index, item in enumerate(entries, 1):
        scenario, condition, variant = (
            item["scenario"],
            item["condition"],
            item["variant"],
        )
        run = item.get(
            "run_id", item.get("trial_id", f"{scenario}-{condition}-{variant}")
        )
        if (
            not re.fullmatch(r"[a-z0-9_-]+", run)
            or run in seen
            or condition not in {"live", "restart"}
        ):
            raise ValueError(f"Invalid or duplicate registered run: {run}")
        seen.add(run)
        workspace = Path(item.get("workspace", run))
        workspace = (
            workspace if workspace.is_absolute() else trial_root / workspace
        ).resolve()
        if not workspace.is_relative_to(trial_root):
            raise ValueError(f"Workspace escapes confirmation directory: {run}")
        normalized.append(
            {
                "run_id": run,
                "trial_id": item.get("trial_id", run),
                "scenario": scenario,
                "condition": condition,
                "variant": variant,
                "index": item.get("index", index),
                "pair_id": item.get("pair_id", f"{scenario}-{variant}"),
                "workspace": workspace,
                "source_commit": item.get(
                    "source_commit", registration.get("source_commit")
                ),
                "source_revision": item.get("source_revision"),
                "attempt": item.get("attempt", 1),
                "task_source_sha256": item.get(
                    "task_source_sha256", registration.get("task_source_sha256", {})
                ),
            }
        )
    if registration.get("trial_count", len(normalized)) != len(normalized):
        raise ValueError("Registration trial_count disagrees with trial list")
    return normalized


def trial_record(entry, registration):
    """Use explicit eligibility gates, retaining every failed or pending trial."""
    folder = entry["workspace"]
    row = {
        key: value
        for key, value in entry.items()
        if key not in {"workspace", "task_source_sha256"}
    }
    row.setdefault("source_commit", registration.get("source_commit"))
    row["study"] = (
        "calibration" if entry["scenario"] == "panda_calibration" else "original"
    )
    if not (folder / "summary.json").exists():
        return row | {"status": "pending", "eligible_success": False}, None, None
    try:
        summary, task = (
            read_json(folder / "summary.json"),
            read_json(folder / "task.json"),
        )
    except (OSError, ValueError):
        return (
            row | {"status": "incomplete_evidence", "eligible_success": False},
            None,
            None,
        )
    elapsed = summary.get("agent_elapsed_seconds")
    startup = summary.get("live_startup_seconds")
    inclusive = summary.get("startup_inclusive_seconds")
    budget = task.get("budget_seconds", registration.get("agent_budget_seconds"))
    ceiling = registration.get(
        "completed_candidate_ceiling", registration.get("candidate_ceiling", 12)
    )
    count = summary.get("candidate_rollouts")
    hashes = summary.get("task_source_hashes")
    registered_hashes = entry.get(
        "task_source_sha256", registration.get("task_source_sha256", {})
    )
    quality = summary.get("quality", {})
    gates = {
        "registered_identity": all(
            summary.get(key) == entry[key] == task.get(key)
            for key in ("scenario", "condition", "variant")
        ),
        "confirmation_phase": summary.get("phase")
        == task.get("phase")
        == "confirmation",
        "requested_model": summary.get("model")
        == registration.get("model", "gpt-6-astra")
        and summary.get("reasoning_effort")
        == registration.get("reasoning_effort", "xhigh"),
        "verified_quality": quality.get("success") is True,
        "clean_exit": summary.get("exit_code") == 0,
        "no_timeout": summary.get("timed_out") is False,
        "time_budget": number(elapsed) and number(budget) and elapsed <= budget,
        "registered_time_budget": budget == registration.get("agent_budget_seconds"),
        "candidate_budget": summary.get("within_candidate_budget") is True
        and isinstance(count, int)
        and not isinstance(count, bool)
        and 0 <= count <= ceiling,
        "shared_sources_unchanged": summary.get("shared_sources_unchanged") is True,
        "task_hashes_consistent": isinstance(hashes, dict)
        and bool(hashes)
        and hashes == task.get("source_hashes"),
        "registered_source_hashes": bool(registered_hashes)
        and isinstance(hashes, dict)
        and hashes == registered_hashes,
        "timing_accounting": number(elapsed)
        and number(startup)
        and number(inclusive)
        and math.isclose(elapsed + startup, inclusive, abs_tol=1e-6),
    }
    spec_hashes = registration.get("task_spec_sha256", {})
    if entry["scenario"] in spec_hashes:
        metadata = {
            "scenario",
            "condition",
            "variant",
            "phase",
            "budget_seconds",
            "camera",
            "source_hashes",
            "reference_sha256",
            "verification_reference_sha256",
        }
        spec = {key: value for key, value in task.items() if key not in metadata}
        gates["registered_task_spec"] = (
            digest(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode())
            == spec_hashes[entry["scenario"]]
        )
    if row["study"] == "calibration":
        commitment = next(
            (
                item
                for item in registration.get("calibration_reference_commitments", [])
                if item.get("variant") == entry["variant"]
            ),
            {},
        )
        gates.update(
            {
                "training_quality": quality.get("training_success") is True
                and summary.get("training_quality", {}).get("success") is True,
                "held_out_quality": quality.get("held_out_success") is True,
                "references_unchanged": summary.get("references_unchanged") is True,
                "registered_references": bool(commitment)
                and task.get("reference_sha256") == commitment.get("training_sha256")
                and task.get("verification_reference_sha256")
                == commitment.get("heldout_sha256"),
            }
        )
    usage = summary.get("usage", {})
    tokens = {
        key: usage.get(key)
        if isinstance(usage.get(key), int)
        and not isinstance(usage.get(key), bool)
        and usage[key] >= 0
        else None
        for key in (
            "input_tokens",
            "cached_input_tokens",
            "cache_write_input_tokens",
            "output_tokens",
            "reasoning_output_tokens",
        )
    }
    valid_subsets = all(
        tokens[subset] is not None
        and tokens[total] is not None
        and tokens[subset] <= tokens[total]
        for subset, total in (
            ("cached_input_tokens", "input_tokens"),
            ("reasoning_output_tokens", "output_tokens"),
        )
    )
    tokens["uncached_input_tokens"] = (
        tokens["input_tokens"] - tokens["cached_input_tokens"]
        if valid_subsets
        else None
    )
    tokens["input_output_tokens"] = (
        tokens["input_tokens"] + tokens["output_tokens"]
        if tokens["input_tokens"] is not None and tokens["output_tokens"] is not None
        else None
    )
    row.update(
        {
            "status": "completed",
            "eligible_success": all(gates.values()),
            "gates": gates,
            "failure_reasons": [key for key, passed in gates.items() if not passed],
            "budget_seconds": budget,
            "candidate_ceiling": ceiling,
            "agent_elapsed_seconds": elapsed,
            "live_startup_seconds": startup,
            "startup_inclusive_seconds": inclusive,
            **tokens,
            "token_subsets_valid": valid_subsets,
            "candidate_rollouts": count,
            "simulation_process_starts": summary.get(
                "simulation_process_starts_during_trial"
            ),
            "verification_process_starts": summary.get("verification_process_starts"),
            "tool_calls": summary.get("tool_items"),
            "mcp_tool_calls": summary.get("mcp_tool_items"),
            "timed_out": summary.get("timed_out"),
            "exit_code": summary.get("exit_code"),
            "quality": sanitize(quality),
            "training_quality": sanitize(summary.get("training_quality")),
            "summary_sha256": digest((folder / "summary.json").read_bytes()),
            "task_sha256": digest((folder / "task.json").read_bytes()),
            "source_hashes": hashes,
            "references": {
                key: task.get(key)
                for key in ("reference_sha256", "verification_reference_sha256")
            },
        }
    )
    return row, summary, task


def pairs_for(rows):
    grouped = collections.defaultdict(dict)
    for row in rows:
        if row["condition"] in grouped[row["pair_id"]]:
            raise ValueError(f"Duplicate method in pair {row['pair_id']}")
        grouped[row["pair_id"]][row["condition"]] = row
    result = []
    for pair_id, methods in grouped.items():
        live, restart = methods.get("live"), methods.get("restart")
        complete = bool(
            live and restart and live["status"] == restart["status"] == "completed"
        )
        eligible = complete and live["eligible_success"] and restart["eligible_success"]
        matched = (
            complete
            and live["source_hashes"] == restart["source_hashes"]
            and live["references"] == restart["references"]
        )
        ratios = {}
        for metric in METRICS:
            numerator, denominator = (
                (restart.get(metric), live.get(metric)) if complete else (None, None)
            )
            ratios[metric] = (
                numerator / denominator
                if eligible
                and matched
                and number(numerator)
                and number(denominator)
                and denominator > 0
                else None
            )
        result.append(
            {
                "pair_id": pair_id,
                "complete": complete,
                "both_eligible_successes": bool(eligible),
                "matched_sources_and_references": bool(matched),
                "ratios_restart_over_live": ratios,
                "live": live["run_id"] if live else None,
                "restart": restart["run_id"] if restart else None,
                "failure_reasons": {
                    condition: row.get("failure_reasons", [row["status"]])
                    for condition, row in methods.items()
                    if not row["eligible_success"]
                },
            }
        )
    return result


def safe_files(folder):
    """Allowlist exports; never traverse symlinks or copy references/credentials."""
    files = []
    for path in folder.rglob("*"):
        if (
            not path.is_file()
            or path.is_symlink()
            or not path.resolve().is_relative_to(folder)
        ):
            continue
        relative = path.relative_to(folder)
        if any("private" in part.lower() for part in relative.parts):
            continue
        if (
            (len(relative.parts) == 1 and path.name in ROOT_FILES)
            or path.name in LOG_NAMES
            or str(relative) == "verification/metrics.json"
        ):
            files.append(path)
    return sorted(files)


def export_file(source, target):
    info = {"source_bytes": source.stat().st_size}
    if info["source_bytes"] > MAX_FILE_BYTES:
        return info | {
            "exported": False,
            "reason": "File exceeds 64 MiB export ceiling",
        }
    data = source.read_bytes()
    info["original_sha256"] = digest(data)
    text = data.decode("utf-8", errors="replace")
    if source.suffix == ".jsonl":
        lines, malformed, oversized = [], 0, 0
        for index, line in enumerate(text.splitlines(), 1):
            if len(line.encode()) > MAX_LINE_BYTES:
                lines.append(
                    json.dumps(
                        {
                            "line": index,
                            "omitted": "Record exceeds 8 MiB; retained in private original",
                            "sha256": digest(line.encode()),
                        }
                    )
                )
                oversized += 1
                continue
            try:
                value = json.loads(line)
            except ValueError:
                malformed += 1
                value = {"malformed_source_line": index, "text": line}
            lines.append(
                json.dumps(sanitize(value), ensure_ascii=False, allow_nan=False)
            )
        output = "\n".join(lines) + "\n"
        info.update(
            {
                "records": len(lines),
                "malformed_lines": malformed,
                "oversized_lines_omitted": oversized,
            }
        )
    elif source.suffix == ".json":
        output = (
            json.dumps(sanitize(json.loads(text)), indent=2, allow_nan=False) + "\n"
        )
    else:
        output = sanitize(text)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(output)
    return info | {
        "exported": True,
        "export_sha256": digest(output.encode()),
        "export_bytes": len(output.encode()),
        "sanitized": output.encode() != data,
    }


def candidate_history(folder):
    files = [
        path
        for path in safe_files(folder)
        if path.name in {"rollouts.jsonl", "live_rollouts.jsonl"}
        and path.relative_to(folder).parts[0] != "verification"
    ]
    records = []
    for path in sorted(files, key=lambda path: (path.stat().st_mtime_ns, str(path))):
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if not isinstance(value, dict):
                continue
            records.append(
                {
                    "candidate_index": len(records) + 1,
                    "source": str(path.relative_to(folder)),
                    "line": line_number,
                    "complete": value.get("frames")
                    == value.get("sample_count")
                    == value.get("expected_frames")
                    and isinstance(value.get("frames"), int)
                    and value["frames"] > 0,
                    "metrics": sanitize(value),
                }
            )
    return {
        "order_basis": "Within-file line order; files ordered by saved modification time, then path. Across-file completion order may be approximate when files contain multiple candidates.",
        "multiple_log_files": len(files) > 1,
        "records": records,
    }


def event_accounting(folder):
    """Describe recorded tool use without assuming that an image guided a decision."""
    path = folder / "agent.jsonl"
    counts, tools = collections.Counter(), collections.Counter()
    image_blocks, execute_observe_text, malformed = 0, 0, 0
    if not path.exists() or path.stat().st_size > MAX_FILE_BYTES:
        return {"available": False}

    def images(value):
        if isinstance(value, dict):
            return int(value.get("type") == "image") + sum(
                images(item) for key, item in value.items() if key != "data"
            )
        if isinstance(value, list):
            return sum(images(item) for item in value)
        return 0

    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            malformed += 1
            continue
        if event.get("type") != "item.completed":
            continue
        item = event.get("item", {})
        kind = item.get("type")
        if kind in {"command_execution", "mcp_tool_call", "tool_call", "file_change"}:
            counts[kind] += 1
        if kind == "mcp_tool_call":
            tool = item.get("tool", item.get("name", "unknown"))
            tools[tool] += 1
            image_blocks += images(item.get("result"))
            if tool == "newton_execute":
                code = str(item.get("arguments", {}))
                execute_observe_text += bool(
                    re.search(r"dispatch\([^\n]{0,10}observe|\.observe\(", code)
                )
    return {
        "available": True,
        "completed_items_by_type": dict(counts),
        "mcp_tools_by_name": dict(tools),
        "command_and_mcp_calls": sum(
            counts[kind] for kind in ("command_execution", "mcp_tool_call", "tool_call")
        ),
        "actions_including_file_changes": sum(counts.values()),
        "direct_mcp_observe_calls": tools["newton_observe"],
        "mcp_image_result_blocks": image_blocks,
        "execute_calls_containing_observe_text": execute_observe_text,
        "malformed_lines": malformed,
        "interpretation": "Counts document calls and returned image blocks. Code mentioning observe is not proof that it executed; image availability does not establish how the agent used it.",
    }


def build(
    registration_path,
    trial_root,
    output,
    *,
    export_evidence=False,
    require_complete=False,
    registration_history=(),
):
    registration = read_json(registration_path)
    plan = plan_entries(registration, trial_root.resolve())
    rows, histories, completed = [], {}, []
    for entry in plan:
        row, summary, _ = trial_record(entry, registration)
        rows.append(row)
        if summary is not None:
            completed.append(entry)
            histories[entry["run_id"]] = candidate_history(entry["workspace"])
            history = histories[entry["run_id"]]
            row["completed_candidate_log_records"] = sum(
                record["complete"] for record in history["records"]
            )
            row["gates"]["candidate_log_count_matches"] = (
                row["candidate_rollouts"] == row["completed_candidate_log_records"]
            )
            row["failure_reasons"] = [
                name for name, passed in row["gates"].items() if not passed
            ]
            row["eligible_success"] = all(row["gates"].values())
            row["event_accounting"] = event_accounting(entry["workspace"])
            row["command_and_mcp_calls"] = row.pop("tool_calls")
            history["first_passing_candidate"] = next(
                (
                    record["candidate_index"]
                    for record in history["records"]
                    if record["complete"] and record["metrics"].get("success") is True
                ),
                None,
            )
    pending = [row["run_id"] for row in rows if row["status"] != "completed"]
    if require_complete and pending:
        raise ValueError(
            f"Registration is incomplete: {len(pending)} of {len(rows)} trial summaries remain unavailable"
        )
    infrastructure = registration.get("infrastructure_attempts", [])
    if any(item.get("agent_launched") is not False for item in infrastructure):
        raise ValueError(
            "Infrastructure attempts must explicitly record agent_launched: false"
        )
    result = {
        "built_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "registration_sha256": digest(registration_path.read_bytes()),
        "source_commit": registration.get("source_commit"),
        "source_commits": sorted(
            {row["source_commit"] for row in rows if row.get("source_commit")}
        ),
        "infrastructure_attempts": sanitize(infrastructure),
        "pre_agent_startup_failures": len(infrastructure),
        "registered_trials": len(rows),
        "completed_trials": len(completed),
        "all_registered_trials_complete": not pending,
        "pending_trials": pending,
        "eligible_successes": sum(row["eligible_success"] for row in rows),
        "completed_unsuccessful_trials": [
            row["run_id"]
            for row in rows
            if row["status"] == "completed" and not row["eligible_success"]
        ],
        "notes": [
            "Development results are retained separately and are never pooled here.",
            "Success requires every recorded gate, not quality alone.",
            "Time budget applies to agent elapsed time; reported comparative elapsed time also includes live startup.",
            "Ratios are restart/live, and are suppressed unless both methods pass all gates with matched sources/references.",
            "Cached input and reasoning output are subsets; uncached input is input minus cached input. Token volumes are not monetary costs.",
            "Missing token measurements remain null. Failures and pending trials are retained.",
            "Per-trial registered source commits/hashes take precedence over the original global version. Source versions must match within each reported pair.",
            "Pre-agent infrastructure attempts are retained separately; they are not agent trials and do not enter agent totals or pair ratios.",
        ],
        "trials": rows,
        "pairs": pairs_for(rows),
    }
    totals = []
    for study in ("original", "calibration"):
        for condition in ("live", "restart"):
            group = [
                row
                for row in rows
                if row["study"] == study
                and row["condition"] == condition
                and row["status"] == "completed"
            ]
            totals.append(
                {
                    "study": study,
                    "condition": condition,
                    "completed": len(group),
                    "eligible_successes": sum(row["eligible_success"] for row in group),
                    "scope": "All completed trials, including failures; not a success-conditioned comparison",
                    "sums": {
                        metric: sum(row[metric] for row in group)
                        if group and all(number(row.get(metric)) for row in group)
                        else None
                        for metric in METRICS
                    },
                }
            )
    result["condition_totals"] = totals
    exports = {}
    infrastructure_exports = {}
    if export_evidence:
        for entry in completed:
            files = {}
            for source in safe_files(entry["workspace"]):
                relative = source.relative_to(entry["workspace"])
                files[str(relative)] = export_file(
                    source, output / "trials" / entry["run_id"] / relative
                )
            exports[entry["run_id"]] = files
        for index, attempt in enumerate(infrastructure, 1):
            run = attempt.get("run_id", f"startup-failure-{index}")
            if not re.fullmatch(r"[a-z0-9_-]+", run):
                raise ValueError(f"Invalid infrastructure run_id: {run}")
            folder = Path(attempt["workspace"])
            folder = (folder if folder.is_absolute() else trial_root / folder).resolve()
            if not folder.is_relative_to(trial_root.resolve()):
                raise ValueError(
                    "Infrastructure workspace escapes confirmation directory"
                )
            log_base = attempt.get("logs_base", "registration_directory")
            if log_base not in {"workspace", "registration_directory"}:
                raise ValueError(f"Unknown infrastructure logs_base: {log_base}")
            log_root = (
                registration_path.parent.resolve()
                if log_base == "registration_directory"
                else folder
            )
            files = {}
            for relative in attempt.get("logs", []):
                source = log_root / relative
                if (
                    Path(relative).is_absolute()
                    or ".." in Path(relative).parts
                    or source.is_symlink()
                    or not source.resolve().is_relative_to(log_root)
                    or source.suffix not in {".log", ".stderr", ".jsonl"}
                    or any("private" in part.lower() for part in Path(relative).parts)
                ):
                    raise ValueError(f"Unsafe infrastructure log: {relative}")
                files[relative] = export_file(
                    source, output / "infrastructure" / run / relative
                )
            infrastructure_exports[run] = files
        write_json(
            output / "export-manifest.json",
            {
                "policy": "Allowlisted text/JSON only. Credentials, private-content records and base64 images redacted. Connection descriptors, reference NPZs and truth/seed files never read or exported. Original-file hashes preserve provenance; exported copies are sanitized.",
                "trials": exports,
                "infrastructure_attempts": infrastructure_exports,
            },
        )
    history_manifest = []
    history_names = set()
    for path in registration_history:
        if path.name in history_names or path.suffix != ".json":
            raise ValueError("Registration history requires distinct JSON filenames")
        history_names.add(path.name)
        info = export_file(path, output / "registration-history" / path.name)
        history_manifest.append({"file": path.name, **info})
    result["registration_history"] = history_manifest
    write_json(output / "registration.json", sanitize(registration))
    write_json(output / "candidate-history.json", histories)
    write_json(output / "results.json", sanitize(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "data" / "confirmation"
    )
    parser.add_argument("--export-evidence", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument(
        "--registration-history",
        type=Path,
        action="append",
        default=[],
        help="Retain a prior registration version with original and sanitized hashes; repeat for multiple versions",
    )
    args = parser.parse_args()
    result = build(
        args.registration,
        args.trials,
        args.output,
        export_evidence=args.export_evidence,
        require_complete=args.require_complete,
        registration_history=args.registration_history,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "registered_trials",
                    "completed_trials",
                    "eligible_successes",
                    "pre_agent_startup_failures",
                    "all_registered_trials_complete",
                    "completed_unsuccessful_trials",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
