# SPDX-License-Identifier: Apache-2.0
"""Decompose the observed grasp drift by reference frame.

If the cube is genuinely sliding on the pads, its pose relative to the *finger
bodies* must change. If instead the measured drift lives between the hand and
the TCP frame, the contact is fine and something in the articulation is moving.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

import newton
from newton.viewer import ViewerNull

from newtontests.franka_cube_shake import Phase, create_parser
from newtontests.experiments import Variant, VARIANTS
from newtontests.measure import rel_pose


def run(variant="baseline", num_frames=900, amplitude=0.03, frequency=1.0):
    argv_backup = sys.argv
    sys.argv = ["frames"]
    args = newton.examples.default_args(create_parser())
    sys.argv = argv_backup
    args.shake_amplitude = amplitude
    args.shake_frequency = frequency

    viewer = ViewerNull(num_frames=num_frames)
    ex = Variant(viewer, args, VARIANTS[variant])

    labels = list(ex.model.body_label)
    print("BODIES:")
    for i, l in enumerate(labels):
        print(f"  {i:3d} {l}")

    def find(short):
        for i, l in enumerate(labels):
            if l.rsplit("/", 1)[-1] == short:
                return i
        return None

    idx = {
        "hand": find("fr3_hand"),
        "tcp": find("fr3_hand_tcp"),
        "lfinger": find("fr3_leftfinger"),
        "rfinger": find("fr3_rightfinger"),
        "link7": find("fr3_link7"),
        "cube": ex.cube_index,
    }
    print("INDICES:", idx, flush=True)

    refs = {}
    rows = []
    for frame in range(num_frames):
        ex.step()
        bq = ex.state_0.body_q.numpy()
        phase = int(ex.phase_index.numpy()[0])
        pairs = {
            "cube_in_tcp": (idx["tcp"], idx["cube"]),
            "cube_in_hand": (idx["hand"], idx["cube"]),
            "cube_in_lfinger": (idx["lfinger"], idx["cube"]),
            "cube_in_rfinger": (idx["rfinger"], idx["cube"]),
            "tcp_in_hand": (idx["hand"], idx["tcp"]),
            "lfinger_in_hand": (idx["hand"], idx["lfinger"]),
            "hand_in_link7": (idx["link7"], idx["hand"]),
        }
        rec = {"frame": frame, "t": frame / 60.0, "phase": phase}
        for name, (p, c) in pairs.items():
            if p is None or c is None:
                continue
            rp, _ = rel_pose(bq[p], bq[c])
            if phase == Phase.SHAKE.value and name not in refs:
                refs[name] = rp.copy()
            rec[name] = [float(v) for v in rp]
            if name in refs:
                rec[name + "_d"] = [float(v) for v in (rp - refs[name])]
        rows.append(rec)
        if frame % 120 == 0 and phase == Phase.SHAKE.value:
            msg = f"[frames] f={frame:4d} "
            for name in pairs:
                if name + "_d" in rec:
                    d = np.array(rec[name + "_d"]) * 1000.0
                    msg += f"{name}=({d[0]:+.2f},{d[1]:+.2f},{d[2]:+.2f})mm "
            print(msg, flush=True)

    sh = [r for r in rows if r["phase"] == Phase.SHAKE.value]
    t = np.array([r["t"] for r in sh])
    out = {}
    for name in ["cube_in_tcp", "cube_in_hand", "cube_in_lfinger", "cube_in_rfinger",
                 "tcp_in_hand", "lfinger_in_hand", "hand_in_link7"]:
        if name + "_d" not in sh[0]:
            continue
        arr = np.array([r[name + "_d"] for r in sh])
        rates = [float(np.polyfit(t, arr[:, k], 1)[0]) * 1000.0 for k in range(3)]
        out[name] = {
            "drift_rate_mm_per_s": rates,
            "total_mm": [float(v) * 1000.0 for v in arr[-1]],
            "norm_total_mm": float(np.linalg.norm(arr[-1])) * 1000.0,
        }
    print(json.dumps(out, indent=2))
    return out, rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="baseline")
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out, rows = run(a.variant, a.frames)
    if a.out:
        json.dump({"summary": out, "rows": rows}, open(a.out, "w"))
