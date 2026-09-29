"""Aggregates SHAC training runs and evaluations into compact report data."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent

p = argparse.ArgumentParser()
p.add_argument("--tag", default="e4")
p.add_argument("--curve-points", type=int, default=50)
args = p.parse_args()

runs = defaultdict(list)
for f in sorted((ROOT / "runs" / args.tag).glob("*.json")):
    robot_cfg, seed = f.stem.rsplit("_s", 1)
    h = json.loads(f.read_text())
    if h["history"][-1]["epoch"] != h["args"]["epochs"] - 1:
        continue
    ev = {}
    for which in ("train", "ref"):
        e = ROOT / "results" / f"{args.tag}_eval" / f"{f.stem}_{which}.json"
        if e.exists():
            ev[which] = json.loads(e.read_text())
    runs[robot_cfg].append({"seed": int(seed), "history": h["history"], "eval": ev, "args": h["args"]})


def mean_sd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"mean": statistics.fmean(xs), "sd": statistics.pstdev(xs) if len(xs) > 1 else 0.0, "n": len(xs),
            "values": xs}


summary = {}
for key, rs in sorted(runs.items()):
    cut = max(key.find("_explicit_"), key.find("_implicit_"))
    robot, cfg = key[:cut], key[cut + 1:]
    E = len(rs[0]["history"])
    idx = sorted({round(i * (E - 1) / (args.curve_points - 1)) for i in range(args.curve_points)})
    curve = []
    for i in idx:
        curve.append({
            "epoch": i,
            "env_steps": rs[0]["history"][i]["env_steps"],
            "return": mean_sd([r["history"][i]["recent_return"] for r in rs]),
            "speed": mean_sd([r["history"][i]["recent_speed"] for r in rs]),
            "length": mean_sd([r["history"][i]["recent_length"] for r in rs]),
        })
    row = {"robot": robot, "config": cfg, "seeds": [r["seed"] for r in rs], "curve": curve,
           "train_wall_s": mean_sd([r["history"][-1]["wall_s"] for r in rs]),
           "skipped_updates": sum(r["history"][-1]["skipped_updates"] for r in rs),
           "nonfinite_grad_events": sum(r["history"][-1]["nonfinite_grad_events"] for r in rs)}
    for which in ("train", "ref"):
        evs = [r["eval"].get(which) for r in rs]
        if all(evs):
            row[which] = {k: mean_sd([e[k] for e in evs]) for k in (
                "survival_fraction", "mean_forward_speed", "mean_abs_lateral_m", "mean_return_per_step",
                "action_rate_rms", "mean_survived_seconds")}
    summary[key] = row
    ref = row.get("ref", {})
    tr = row.get("train", {})
    fmt = lambda d, k: f"{d[k]['mean']:.2f}±{d[k]['sd']:.2f}" if d and d.get(k) else "  -  "  # noqa: E731
    print(f"{key:28s} n={len(rs)} train: v={fmt(tr, 'mean_forward_speed')} surv={fmt(tr, 'survival_fraction')} | "
          f"ref: v={fmt(ref, 'mean_forward_speed')} surv={fmt(ref, 'survival_fraction')} "
          f"lat={fmt(ref, 'mean_abs_lateral_m')} skip={row['skipped_updates']}")
(ROOT / "results" / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=1))
