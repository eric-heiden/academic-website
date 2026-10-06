"""Export harness-improvement-loop trials (v4) into data/v4 with credential redaction."""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
import zipfile
from pathlib import Path

from export_v3 import redact as _redact_credentials

# Held-out episode ids of the physical replay tasks stay private (agents have network access and could fetch the
# public recordings); published records name them by task and position.
_PRIVATE = Path.home() / ".newton-visual-private"


def _heldout_names() -> dict[str, str]:
    names = {}
    for task, prefix in (("abc_replay", "heldout"), ("abc_bin", "bin_heldout")):
        scenes = _PRIVATE / task / "heldout" / "scenes"
        stems = sorted(path.stem for path in scenes.glob("*.json")) if scenes.exists() else []
        spec = _PRIVATE / task / "heldout_spec.json"
        if spec.exists():
            data = json.loads(spec.read_text())
            # Spares and the rest of the pool are unseen too.
            for key in ("heldout", "spare", "pool"):
                for entry in data.get(key) or []:
                    stem = entry if isinstance(entry, str) else str(entry.get("id") or entry.get("uuid") or "")
                    if stem and stem not in stems:
                        stems.append(stem[:8])
        for k, stem in enumerate(dict.fromkeys(stems)):
            names[stem] = f"{prefix}_{k + 1}"
    return names


HELDOUT_NAMES = _heldout_names()


def redact(text: str) -> str:
    text = _redact_credentials(text)
    for episode, name in HELDOUT_NAMES.items():
        text = text.replace(episode, name)
    return text

HERE = Path(__file__).resolve().parent
LOOP = Path("/home/horde/artifacts/newton-live-mcp-v4/loop")
DATA = HERE / "data" / "v4"
PUBLIC_FILES = ("summary.json", "verification.json", "TASK.md", "spec.json", "agent.times.jsonl")
sys.path.insert(0, "/home/horde/apps/newton-live-mcp")
from tools.mcp_evaluation.visual.analyze import timing  # noqa: E402

MODEL_KEYS = {"claude-opus-5-5": "opus", "gpt-6-astra": "astra"}
# Runs that the runner wrongly classified as infrastructure failures (the agent printed TASK.md,
# whose "tools unavailable" token the old check matched). The original run is the pair's trial;
# the automatic rerun is kept as an unpaired "-rerun" row.
REINSTATED = {("i11", "abc_look-astra-mcp-p0.infra-failure-1"): "abc_look-astra-mcp-p0"}


def _record(run_dir: Path, name: str) -> Path:
    """A trial file: harness records sit in the run directory; from h12 on, the agent's files in its workspace/."""
    path = run_dir / name
    return path if path.exists() else run_dir / "workspace" / name


def trial_rows() -> list[dict]:
    rows = []
    for iteration in sorted(p for p in LOOP.iterdir() if p.is_dir() and p.name.startswith("i")):
        for workspace in sorted(p for p in iteration.iterdir() if p.is_dir() and (p / "summary.json").exists()):
            trial = REINSTATED.get((iteration.name, workspace.name), workspace.name)
            replicate = trial.rsplit("-", 1)[-1]
            if (iteration.name, workspace.name) not in REINSTATED and workspace.name in {
                name for (it, _), name in REINSTATED.items() if it == iteration.name
            }:
                trial, replicate = f"{workspace.name}-rerun", f"{replicate}-rerun"
            if ".infra-failure" in trial or workspace.parent.name.startswith("i2") and not (workspace / "summary.json").exists():
                continue
            s = json.loads((workspace / "summary.json").read_text())
            v = s["verification"]
            reverified = workspace / "verification_v2.json"
            if not reverified.exists():
                # From i15: verifier fixes made during an iteration are applied to every trial of the task
                # (tools/mcp_evaluation/v4/reverify.py writes reverify/result.json).
                reverified = workspace / "reverify" / "result.json"
            if reverified.exists():
                # G1 submissions were re-verified after the time-scaled playback was fixed (verification_v2.json).
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
                    "trial": trial,
                    "dir": workspace.name,
                    "task": s["task"],
                    "model": MODEL_KEYS.get(s["model"], s["model"]),
                    "condition": s["condition"],
                    "replicate": replicate,
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
                    # From h12: review flags (introspection in the submission, downloads, hidden-data paths).
                    "flags": [k for k in ("introspection", "downloads", "private_reference") if s.get(k)],
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


def _sequential(row: dict) -> bool:
    """Whether the trial ran under a harness that runs trials one at a time (h12 on)."""
    try:
        return int(str(row.get("harness") or "h0").lstrip("h").split("-")[0]) >= 12
    except ValueError:
        return False


def summarize(rows: list[dict]) -> dict:
    groups = {}
    for pair in pairs(rows):
        if "mcp" not in pair or "restart" not in pair:
            continue
        if any((pair[c].get("infrastructure_retries") or 0) > 0 and not _sequential(pair[c]) for c in ("mcp", "restart")):
            # Through h11 a retried trial ran without its concurrent partner, so the pair is not a paired comparison.
            # From h12 every trial runs alone, so a retry runs under the same conditions as its partner.
            continue
        for scope in (pair["iteration"], f'{pair["iteration"]}:{pair["model"]}', "all", f'all:{pair["model"]}'):
            group = groups.setdefault(scope, {"pairs": 0, "mcp_pass": 0, "restart_pass": 0, "values": {}})
            group["pairs"] += 1
            group["mcp_pass"] += pair["mcp"]["success"]
            group["restart_pass"] += pair["restart"]["success"]
            timed_out = pair["mcp"]["timed_out"] or pair["restart"]["timed_out"]
            group["timed_out_pairs"] = group.get("timed_out_pairs", 0) + int(timed_out)
            for metric in ("seconds", "input_tokens", "uncached_tokens", "output_tokens", "tool_calls", "cost_usd"):
                if metric == "seconds" and timed_out:
                    continue  # a timeout censors the time at the budget, so it is not a measured duration
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
    # One archive per iteration keeps every file well below GitHub's size limits.
    (DATA / "loop-transcripts.zip").unlink(missing_ok=True)
    for iteration in sorted({row["iteration"] for row in rows}):
        name = f"loop-transcripts-{iteration}.zip"
        with zipfile.ZipFile(DATA / name, "w", zipfile.ZIP_DEFLATED) as archive:
            for row in (r for r in rows if r["iteration"] == iteration):
                workspace = LOOP / row["iteration"] / row.get("dir", row["trial"])
                for filename in (*PUBLIC_FILES, "agent.jsonl", "agent.stderr", "host.log", "verification.log"):
                    path = _record(workspace, filename)
                    if path.exists():
                        archive.writestr(f"{row['iteration']}/{row['trial']}/{filename}", redact(path.read_text(errors="replace")))
                script = json.loads((workspace / "spec.json").read_text())
                for script_name in script.get("starter_sha256", {}):
                    submitted = _record(workspace, script_name)
                    if script_name.endswith(".py") and submitted.exists():
                        archive.writestr(f"{row['iteration']}/{row['trial']}/submitted_{script_name}", redact(submitted.read_text()))
        manifest[name] = hashlib.sha256((DATA / name).read_bytes()).hexdigest()
    result = {"trials": rows, "summary": summarize(rows), "manifest": manifest}
    (DATA / "loop.json").write_text(json.dumps(result, indent=1, default=float) + "\n")
    return result


if __name__ == "__main__":
    result = export()
    for scope, group in result["summary"].items():
        ratios = {k: round(v["ratio"], 2) for k, v in group["ratios"].items() if v}
        print(scope, group["pairs"], "pairs; pass mcp", group["mcp_pass"], "restart", group["restart_pass"], ratios)
