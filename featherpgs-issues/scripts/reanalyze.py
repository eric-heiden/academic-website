"""Re-score every saved run on whether the cube was actually picked up.

Reaching the SHAKE phase is not the same as holding the cube: the arm follows
its scripted trajectory regardless, so a run that drops the object still logs
shake frames. A grasp counts only if the cube is off the table when the shake
starts and still in the hand when it ends.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATA = Path("/home/horde/repos/fpgs-study/data")
LIFTED_Z = 0.10       # the lift phase raises the cube to about 0.22 m
HELD_SEP = 0.08       # cube/TCP separation that still counts as held


def score(name):
    p = DATA / f"{name}.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    if "phase" not in d or len(d["phase"]) == 0:
        return None
    ph, z = d["phase"], d["cube_pos"][:, 2]
    sep = np.linalg.norm(d["rel_pos"], axis=1)
    m = ph == 4
    out = {"reached_shake": bool(m.any()), "frames": int(len(ph))}
    if not m.any():
        out["outcome"] = "never reaches the shake"
        return out
    out["z_at_shake"] = float(z[m][0])
    out["sep_max"] = float(sep[m].max())
    lifted = z[m][0] > LIFTED_Z
    held = sep[m].max() < HELD_SEP
    if not lifted:
        out["outcome"] = "cube never leaves the table"
    elif not held:
        out["outcome"] = "cube dropped during the shake"
    else:
        rl = d["rel_pos_local"][m]
        drift = (rl - rl[0]) * 1000.0
        out["outcome"] = "held"
        out["drift_max_mm"] = float(np.abs(drift).max())
        out["drift_rms_mm"] = float(np.sqrt((drift ** 2).sum(axis=1).mean()))
    return out


NAMES = (
    ["mujoco_base", "fpgs_default", "fpgs_100x", "repeat_traces"]
    + [f"fpgs_gc_{k}" for k in ("default", "sharedanchor", "velit4",
                                "anchor_velit4", "it32", "best")]
    + [f"cap{c}" for c in (32, 48, 64, 96, 128, 256)]
    + [f"fpgs_ss{s}" for s in (16, 8, 4, 2, 1)]
    + [f"mujoco_ss{s}" for s in (16, 8, 4, 2, 1)]
    + [f"def_{k}" for k in ("library_defaults", "defaults_no_double_buffer",
                            "split_rows150", "split_rows201", "split_rows256",
                            "matrixfree_rows256")]
    + [f"spec_{k}" for k in ("base_gap1", "velit4_gap1", "velit4_gap0.02",
                             "base_gap0.02", "velit8_gap0.02")]
)

if __name__ == "__main__":
    out = {}
    for n in NAMES:
        s = score(n)
        if s:
            out[n] = s
    Path("/home/horde/repos/fpgs-study/rescored.json").write_text(
        json.dumps(out, indent=2))
    for k, v in out.items():
        extra = f"{v['drift_max_mm']:8.2f} mm" if "drift_max_mm" in v else ""
        print(f"{k:28s} {v['outcome']:32s} {extra}")
