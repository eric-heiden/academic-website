"""Isolated epoch timing of shac2 configurations (run as one exclusive queue job).

Each configuration trains for --epochs epochs; the median epoch time over the epochs after
--skip is recorded, along with the rollout/backward/rest split. Configurations are given as
JSON objects {"name", "task", "set": {...}, "tset": {...}} in a file or inline.

Usage: python time_suite.py --suite suites/x.json --out results/loop/timing_x.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--suite", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--skip", type=int, default=8)
a = ap.parse_args()
suite = json.loads(Path(a.suite).read_text())
out = Path(a.out)
rows = json.loads(out.read_text()) if out.exists() else []
done = {r["name"] for r in rows}
for c in suite:
    if c["name"] in done:
        continue
    tmp = ROOT / "runs" / "timing" / c["name"]
    tmp.parent.mkdir(parents=True, exist_ok=True)
    algo = c.get("algo", "shac2")
    sets = {"epochs" if algo == "shac2" else "iterations": a.epochs, **c.get("set", {})}
    if algo == "shac2":
        sets.setdefault("log_every", 1000)
        sets.setdefault("save_every", 100000)
    else:
        sets.setdefault("save_every", 100000)
    cmd = [sys.executable, f"{algo}.py", "--task", c["task"], "--out", str(tmp),
           "--set", *[f"{k}={json.dumps(v)}" for k, v in sets.items()]]
    if c.get("tset"):
        cmd += ["--tset", *[f"{k}={json.dumps(v)}" for k, v in c["tset"].items()]]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(c["name"], "FAILED", r.stdout[-2000:], r.stderr[-2000:], flush=True)
        rows.append({"name": c["name"], "failed": True})
        continue
    h = json.loads(tmp.with_suffix(".json").read_text())["history"][a.skip:]
    med = lambda k: statistics.median(x[k] for x in h if k in x)  # noqa: E731
    row = {"name": c["name"], "task": c["task"], "set": c.get("set", {}), "tset": c.get("tset", {}),
           "epoch_ms": 1e3 * med("epoch_s"), "rollout_ms": 1e3 * med("rollout_s"),
           "samples_per_epoch": h[0]["env_steps"] // (h[0]["epoch"] + 1)}
    if algo == "shac2":
        row.update(backward_ms=1e3 * med("backward_s"), rest_ms=1e3 * med("rest_s"))
    rows.append(row)
    print(json.dumps(row), flush=True)
    out.write_text(json.dumps(rows, indent=1))
