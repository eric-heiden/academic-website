"""Builds data/tuning.json for the retuning section of the report from runs/loop histories.

Each series is a training run (or a set of seeds) with the episode length (s) and the planar
tracking error (m/s) of training episodes, averaged over 10-epoch bins. Runs are grouped by task.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = Path("/home/horde/repos/reports-shac-wt/shac/data/tuning.json")

# Isolated epoch time (s) of the final trainer per configuration (results/loop/timing1-3.json, GPU otherwise idle).
EPOCH_S = {"h1_vel": 0.1862, "h1_gait": 0.2320, "go1_vel": 0.1031, "g1_gait": 0.3305, "ppo_go1": 0.4444,
           "ppo_h1": 0.3513, "ppo_h1gait": 0.3827, "hum_d10": 0.3703, "go1_gait": 0.1061}  # timing2 (current trainer) where available
SERIES_TIMING = {"h1_vel": "h1_vel", "h1_gait": "h1_gait", "h1_gait_long": "h1_gait", "h1_gait_ppo": "ppo_h1gait", "h1_vel_ppo": "ppo_h1",
                 "h1_gait_ta09": "h1_gait", "hum_d10_995": "hum_d10", "hum_d10_09": "hum_d10",
                 "go1_vel": "go1_vel", "go1_vel_s10": "go1_vel", "go1_gait": "go1_gait", "go1_vel_kp100": "go1_vel",
                 "go1_ppo": "ppo_go1", "go1_ppo_s025": "ppo_go1", "h1_gait_rpo": "h1_gait",
                 "g1_il_gait": "g1_gait", "go1_fwd": "go1_vel", "go1_fwd_clip": "go1_vel", "g1_il_gait_ta09": "g1_gait", "g1_strong": "g1_gait"}
NO_COMMAND = {"hum"}  # groups whose task has no velocity command (no tracking error)
# (group, key, label, run names, style)
SERIES = [
    ("h1", "h1_vel", "Isaac Lab task, α = 0.995", ["v1_h1vel_5ms_T16"], "base"),
    ("h1", "h1_gait", "Gait prior, α = 0.995", ["v2_h1gait_5ms_T16"], "alt"),
    ("h1", "h1_gait_long", "Gait prior, α = 0.995, 1500 epochs", ["v3_h1gait_long", "v3_h1gait_long_s1",
                                                          "v3_h1gait_long_s2"], "long"),
    ("h1", "h1_gait_ppo", "Gait prior, PPO", ["ppo_h1gait_5ms"], "ppo"),
    ("h1", "h1_vel_ppo", "Isaac Lab task, PPO", ["ppo_h1vel_5ms"], "ppo2"),
    ("h1", "h1_gait_ta09", "Gait prior, α = 0.9", ["a11_h1gait_ta09", "a11_h1gait_ta09_s1", "a11_h1gait_ta09_s2"], "prior"),
    ("h1", "h1_gait_rpo", "Gait prior, α = 0.9, action-gradient reuse", ["a12_h1gait_rpo4", "a12_h1gait_rpo4_s1",
                                                                             "a12_h1gait_rpo4_s2"], "extra"),
    ("g1", "g1_gait", "Gait prior, mjlab gains, α = 0.995", ["v2_g1gait_5ms_T16"], "base"),
    ("g1", "g1_il_gait", "Gait prior, Isaac Lab gains, α = 0.995", ["g2_g1ilgait"], "alt"),
    ("g1", "g1_il_gait_ta09", "Gait prior, Isaac Lab gains, α = 0.9", ["g2_g1ilgait_ta09", "g3_g1ilgait_ta09_s1", "g3_g1ilgait_ta09_s2"], "long"),
    ("g1", "g1_strong", "Stronger gait prior, Isaac Lab gains, α = 0.9", ["g5_g1_strongprior", "g5_g1_strongprior_s1",
                                                                        "g5_g1_strongprior_s2"], "extra"),
    ("hum", "hum_nod_995", "Model damping, α = 0.995", ["hd0_humref_nod_2ms_T16"], "base"),
    ("hum", "hum_d10_995", "Damping 10 N·m·s/rad, α = 0.995", ["hd1_humref_d10_2ms_T16"], "alt"),
    ("hum", "hum_nod_09", "Model damping, α = 0.9", ["hd4_humref_nod_2ms_T16_ta09"], "prior"),
    ("hum", "hum_d10_09", "Damping 10 N·m·s/rad, α = 0.9", ["hd3_humref_d10_2ms_T16_ta09", "hd3_humref_d10_ta09_s1",
                                                     "hd3_humref_d10_ta09_s2"], "long"),
    ("go1", "go1_vel", "Isaac Lab task, SHAC", ["v1_go1vel_5ms_T8"], "prior"),
    ("go1", "go1_gait", "Isaac Lab task with trot clock, SHAC", ["v4_go1gait_5ms_T8"], "alt"),
    ("go1", "go1_fwd", "Forward commands, SHAC", ["v1_go1velfwd_5ms_T8", "y_go1_il_fwd_s1", "y_go1_il_fwd_s2",
                                                 "y_go1_il_fwd_s3"], "long"),
    ("go1", "go1_fwd_clip", "Forward commands, per-world clipping, SHAC", [f"z2_go1fwd_wclip_s{i}" for i in range(4)], "extra"),
    ("go1", "go1_ppo", "PPO, action scale 0.5", ["ppo_go1vel_5ms"], "base"),
    ("go1", "go1_ppo_s025", "PPO, Isaac Lab action scale 0.25", ["ppo_go1vel_5ms_s025"], "ppo"),
]
BIN = 10


def load(name):
    p = ROOT / "runs" / "loop" / f"{name}.json"
    if not p.exists() or not (ROOT / "jobs" / "done" / f"{name}.json").exists():  # completed runs only
        return None
    d = json.loads(p.read_text())
    return d, d["history"]


def binned(h, key, dt):
    out = {}
    for r in h:
        v = r.get("ep_" + key)
        if v is None:
            continue
        if key == "length":
            v = v * dt
        out.setdefault(r["epoch"] // BIN, []).append(v)
    return {b: statistics.mean(vs) for b, vs in out.items()}


data = {"series": []}
for group, key, label, runs, style in SERIES:
    per_run = []
    for name in runs:
        x = load(name)
        if x is None:
            continue
        d, h = x
        dt = d["config"]["control_dt"]
        samples = h[0]["env_steps"] // (h[0]["epoch"] + 1)
        per_run.append((binned(h, "length", dt), binned(h, "track_err", dt), samples, d.get("algo", "shac")))
    if not per_run:
        continue
    bins = sorted(set.intersection(*[set(p[0]) for p in per_run]))
    samples = per_run[0][2]
    data["series"].append({
        "group": group, "key": key, "label": label, "style": style, "seeds": len(per_run),
        "algo": per_run[0][3],
        "epoch": [b * BIN + BIN - 1 for b in bins],
        "samples_m": [(b * BIN + BIN) * samples / 1e6 for b in bins],
        "minutes": ([(b * BIN + BIN) * EPOCH_S[SERIES_TIMING[key]] / 60 for b in bins]
                    if key in SERIES_TIMING else None),
        "length": [statistics.mean(p[0][b] for p in per_run) for b in bins],
        "track": (None if group in NO_COMMAND else
                  [statistics.mean(p[1][b] for p in per_run) if all(b in p[1] for p in per_run) else None
                   for b in bins]),
    })
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(data))
print(OUT, [(s["key"], s["seeds"], len(s["epoch"])) for s in data["series"]])
