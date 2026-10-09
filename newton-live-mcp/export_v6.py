"""Export the v6 trials (cheap models with a Debug-Card harness) to data/v6/trials.json.

Reads every run directory of the v6 loop directories (loop/v6-*, loop/v6.*) with the study's analysis script
(research/v6/analyze.py), which skips infrastructure failures and takes tokens from the usage timeline.
"""

from __future__ import annotations

import glob
import importlib.util
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOOP = Path("/home/horde/artifacts/newton-live-mcp-v4/loop")
ANALYZE = Path("/home/horde/artifacts/newton-live-mcp-v5/research/v6/analyze.py")
KEEP = (
    "iteration", "trial", "task", "model", "condition", "harness", "arm", "success", "failed_checks", "seconds",
    "budget_seconds", "first_pass_seconds", "cost_usd", "first_pass_cost_usd", "input_tokens", "output_tokens", "turns",
    "tool_calls", "mcp_calls", "shell_python", "source_reads", "features", "card_usage",
)


def main() -> None:
    spec = importlib.util.spec_from_file_location("v6_analyze", ANALYZE)
    analyze = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyze)
    rows = []
    for loop in sorted(glob.glob(str(LOOP / "v6-*")) + glob.glob(str(LOOP / "v6.*"))):
        if not os.path.isdir(loop):
            continue
        for run_dir in sorted(glob.glob(os.path.join(loop, "*/"))):
            row = analyze.load(run_dir)
            if row is None:
                continue
            row["iteration"] = os.path.basename(loop)
            cards = row.get("card_usage") or {}
            row["card_usage"] = {key: value for key, value in cards.items() if key != "events"}
            rows.append({key: row.get(key) for key in KEEP})
    out = HERE / "data" / "v6"
    out.mkdir(parents=True, exist_ok=True)
    (out / "trials.json").write_text(json.dumps(rows, indent=1, default=float) + "\n")
    print(f"wrote {len(rows)} v6 trials to {out / 'trials.json'}")


if __name__ == "__main__":
    main()
