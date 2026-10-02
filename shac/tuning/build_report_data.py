"""Builds data/locomotion.json for the report's locomotion figures.

- go1: training tracking error on the full Isaac Lab Go1 command set (mean over seeds of the training-episode
  tracking error, averaged in bins of 25 epochs) for SHAC, SAPO, and the SAPO ablation.
- sat: share of saturated action components and mean squash derivative along training (results/loop/sat_curve.json).
- gait: foot-contact timelines and foot heights of the first environment under a fixed forward command
  (results/loop/gait/<run>.json, from gait_eval.py).
"""

import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = Path("/home/horde/repos/reports-shac-wt/shac/data/locomotion.json")

GO1 = [
    ("shac", "SHAC", [f"s95_go1vel_s{i}" for i in range(4)]),
    ("noent", "SAPO without entropy terms", ["abl_noent_s0", "abl_noent_s1"]),
    ("sapo", "SAPO", ["ilfull_sapo_T8_s0", "ilfull_sapo_T8_s1"]),
    ("shacent", "SHAC + entropy terms", ["abl_shacent_s0", "abl_shacent_s1"]),
    ("squash", "SHAC + squash penalty", ["abl_squash03_s0", "abl_squash03_s1"]),
]
SAT = [("shac", "s95_go1vel_s0"), ("noent", "abl_noent_s0"), ("sapo", "ilfull_sapo_T8_s0"),
       ("shacent", "abl_shacent_s0"), ("squash", "abl_squash03_s0")]
GAIT = [
    ("g1_noprior", "G1, SAPO without gait prior", "g1vel_sapo_s0"),
    ("g1_prior", "G1, SHAC with gait prior", "g5_g1_strongprior"),
    ("g1_walk", "G1, SAPO with gait prior and colliding feet", None),
    ("h1", "H1, SAPO", "h1vel_sapo_s0"),
]
BIN = 25


def hist(name):
    return json.loads((ROOT / "runs" / "loop" / f"{name}.json").read_text())["history"]


def curve(runs):
    per = []
    for r in runs:
        h = [x for x in hist(r) if x.get("ep_episodes", 0) > 0]
        bins = {}
        for x in h:
            bins.setdefault(x["epoch"] // BIN, []).append(x["ep_track_err"])
        per.append({b: st.mean(v) for b, v in bins.items()})
    common = sorted(set.intersection(*[set(p) for p in per]))
    return [b * BIN + BIN // 2 for b in common], [round(st.mean(p[b] for p in per), 4) for b in common]


def main(gait_walk=None):
    data = {"go1": [], "sat": [], "gait": []}
    for key, label, runs in GO1:
        x, y = curve(runs)
        data["go1"].append({"key": key, "label": label, "seeds": len(runs), "epoch": x, "track": y})
    sat = json.loads((ROOT / "results" / "loop" / "sat_curve.json").read_text())
    for key, run in SAT:
        rows = sorted((r for r in sat if r["run"] == run), key=lambda r: r["epoch"])
        data["sat"].append({"key": key, "epoch": [r["epoch"] for r in rows],
                            "sat": [round(r["sat_frac"], 4) for r in rows],
                            "deriv": [round(r["squash_deriv"], 4) for r in rows]})
    for key, label, run in GAIT:
        run = run or gait_walk
        f = ROOT / "results" / "loop" / "gait" / f"{run}.json"
        if not run or not f.exists():
            continue
        row = next(r for r in json.loads(f.read_text()) if r["cmd"][:3] == [0.5, 0.0, 0.0])
        data["gait"].append({"key": key, "label": label, "dt": row["timeline"]["dt"],
                             "contact": row["timeline"]["contact"], "height": row["timeline"]["height"]})
    OUT.write_text(json.dumps(data))
    print("wrote", OUT, {k: len(v) for k, v in data.items()})


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else None)
