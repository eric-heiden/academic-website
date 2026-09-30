"""Export harness-improvement-loop trials (v4) into data/v4 with credential redaction."""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
import zipfile
from pathlib import Path

from export_v3 import redact

HERE = Path(__file__).resolve().parent
LOOP = Path("/home/horde/artifacts/newton-live-mcp-v4/loop")
DATA = HERE / "data" / "v4"
PUBLIC_FILES = ("summary.json", "verification.json", "TASK.md", "spec.json", "agent.times.jsonl")
sys.path.insert(0, "/home/horde/apps/newton-live-mcp")
from tools.mcp_evaluation.visual.analyze import timing  # noqa: E402

MODEL_KEYS = {"claude-opus-5-5": "opus", "gpt-6-astra": "astra"}


def trial_rows() -> list[dict]:
    rows = []
    for iteration in sorted(p for p in LOOP.iterdir() if p.is_dir() and p.name.startswith("i")):
        for workspace in sorted(p for p in iteration.iterdir() if p.is_dir() and (p / "summary.json").exists()):
            if ".infra-failure" in workspace.name or workspace.parent.name.startswith("i2") and not (workspace / "summary.json").exists():
                continue
            s = json.loads((workspace / "summary.json").read_text())
            v = s["verification"]
            reverified = workspace / "verification_v2.json"
            if reverified.exists():
                # G1 submissions were re-verified after the time-scaled playback was fixed.
                v2 = json.loads(reverified.read_text())
                v = {k: v2.get(k) for k in ("success", "integrity", "failed_checks", "metrics", "normalized_worst")}
                v["reverified_from"] = s["verification"].get("success")
                s["success"] = bool(v["success"]) and not s["timed_out"]
            split = timing(workspace, s["cli"])
            usage = s.get("usage") or {}
            rows.append(
                {
                    "iteration": iteration.name,
                    "harness": s.get("harness_version"),
                    "trial": workspace.name,
                    "task": s["task"],
                    "model": MODEL_KEYS.get(s["model"], s["model"]),
                    "condition": s["condition"],
                    "replicate": workspace.name.rsplit("-", 1)[-1],
                    "success": bool(s["success"]),
                    "timed_out": bool(s["timed_out"]),
                    "seconds": s["total_seconds"],
                    "input_tokens": usage.get("input_tokens"),
                    "uncached_tokens": usage.get("uncached_input_plus_output"),
                    "output_tokens": usage.get("output_tokens"),
                    "cost_usd": s.get("cost_usd"),
                    "tool_calls": s["tool_call_total"],
                    "mcp_calls": sum(n for k, n in (s.get("tool_calls") or {}).items() if "newton" in k),
                    "tool_errors": s.get("tool_errors"),
                    "images": s.get("mcp_images_returned"),
                    "metrics": v.get("metrics"),
                    "normalized_worst": v.get("normalized_worst"),
                    "failed_checks": v.get("failed_checks"),
                    "tool_seconds": split.get("tool_union_seconds"),
                    "model_seconds": split.get("model_seconds"),
                    "infrastructure_retries": s.get("infrastructure_retries", 0),
                    "reverified_from": v.get("reverified_from"),
                }
            )
    return rows


def paired_ratio(pairs: list[tuple[float, float]], seed: int = 7) -> dict | None:
    """Geometric mean of MCP/restart with a percentile bootstrap interval."""
    logs = [math.log(a / b) for a, b in pairs if a and b]
    if not logs:
        return None
    rng = random.Random(seed)
    boot = sorted(sum(rng.choice(logs) for _ in logs) / len(logs) for _ in range(4000))
    return {
        "ratio": math.exp(sum(logs) / len(logs)),
        "ci95": [math.exp(boot[int(0.025 * len(boot))]), math.exp(boot[int(0.975 * len(boot)) - 1])],
        "n": len(logs),
        "mcp_better": sum(1 for x in logs if x < 0),
    }


def pairs(rows: list[dict]) -> list[dict]:
    index = {}
    for row in rows:
        key = (row["iteration"], row["task"], row["model"], row["replicate"])
        entry = index.setdefault(key, dict(zip(("iteration", "task", "model", "replicate"), key, strict=True)))
        entry[row["condition"]] = row
    return sorted(index.values(), key=lambda p: (p["iteration"], p["task"], p["model"], p["replicate"]))


def summarize(rows: list[dict]) -> dict:
    groups = {}
    for pair in pairs(rows):
        if "mcp" not in pair or "restart" not in pair:
            continue
        for scope in (pair["iteration"], f'{pair["iteration"]}:{pair["model"]}', "all", f'all:{pair["model"]}'):
            group = groups.setdefault(scope, {"pairs": 0, "mcp_pass": 0, "restart_pass": 0, "values": {}})
            group["pairs"] += 1
            group["mcp_pass"] += pair["mcp"]["success"]
            group["restart_pass"] += pair["restart"]["success"]
            for metric in ("seconds", "input_tokens", "uncached_tokens", "output_tokens", "tool_calls", "cost_usd"):
                group["values"].setdefault(metric, []).append((pair["mcp"][metric], pair["restart"][metric]))
    return {
        scope: {
            **{k: v for k, v in group.items() if k != "values"},
            "ratios": {metric: paired_ratio(values) for metric, values in group["values"].items()},
        }
        for scope, group in sorted(groups.items(), key=lambda item: (item[0].startswith("all"), item[0]))
    }


def export() -> dict:
    DATA.mkdir(parents=True, exist_ok=True)
    rows = trial_rows()
    manifest = {}
    with zipfile.ZipFile(DATA / "loop-transcripts.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for row in rows:
            workspace = LOOP / row["iteration"] / row["trial"]
            for filename in PUBLIC_FILES:
                path = workspace / filename
                if path.exists():
                    archive.writestr(f"{row['iteration']}/{row['trial']}/{filename}", redact(path.read_text(errors="replace")))
            for filename in ("agent.jsonl", "agent.stderr", "host.log", "verification.log"):
                path = workspace / filename
                if path.exists():
                    archive.writestr(f"{row['iteration']}/{row['trial']}/{filename}", redact(path.read_text(errors="replace")))
            script = json.loads((workspace / "spec.json").read_text())
            for name in script.get("starter_sha256", {}):
                if name.endswith(".py") and (workspace / name).exists():
                    archive.writestr(f"{row['iteration']}/{row['trial']}/submitted_{name}", (workspace / name).read_text())
    manifest["loop-transcripts.zip"] = hashlib.sha256((DATA / "loop-transcripts.zip").read_bytes()).hexdigest()
    result = {"trials": rows, "summary": summarize(rows), "manifest": manifest}
    (DATA / "loop.json").write_text(json.dumps(result, indent=1, default=float) + "\n")
    return result


if __name__ == "__main__":
    result = export()
    for scope, group in result["summary"].items():
        ratios = {k: round(v["ratio"], 2) for k, v in group["ratios"].items() if v}
        print(scope, group["pairs"], "pairs; pass mcp", group["mcp_pass"], "restart", group["restart_pass"], ratios)
