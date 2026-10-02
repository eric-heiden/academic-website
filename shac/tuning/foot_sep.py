"""Lateral separation of the two feet in the heading frame during a recorded rollout (record2.py npz).

Applies the robot's model edits, computes the foot-body positions by forward kinematics, and reports, per episode,
the median and minimum of the lateral distance between the left and right foot in the base heading frame, the share
of steps in which the left foot is not to the left of the right foot (crossing), and the minimum distance between
any left-foot and right-foot collision geom (negative: interpenetration).
Usage: python foot_sep.py --npz media/x.npz --task g1_walk --model-edit floor_only+g1_feet3+g1_il
"""
import argparse
import json

import mujoco
import numpy as np

import robots

ap = argparse.ArgumentParser()
ap.add_argument("--npz", required=True)
ap.add_argument("--left", default="left_ankle_roll_link")
ap.add_argument("--right", default="right_ankle_roll_link")
ap.add_argument("--model-edit", default="floor_only+g1_feet3+g1_il")
ap.add_argument("--label", default="")
a = ap.parse_args()
d0 = np.load(a.npz, allow_pickle=True)
m = mujoco.MjModel.from_xml_path(str(d0["xml"]))
for e in a.model_edit.split("+"):
    robots.MODEL_EDITS[e](m)
d = mujoco.MjData(m)
L = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, a.left)
R = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, a.right)
lg = [g for g in range(m.ngeom) if m.geom_bodyid[g] == L and m.geom_contype[g] + m.geom_conaffinity[g] > 0]
rg = [g for g in range(m.ngeom) if m.geom_bodyid[g] == R and m.geom_contype[g] + m.geom_conaffinity[g] > 0]
q = d0["qpos"]
out = []
for ep in range(q.shape[0]):
    lat, gap = [], []
    for t in range(0, q.shape[1]):
        d.qpos[:] = q[ep, t]
        mujoco.mj_kinematics(m, d)
        w, x, y, z = q[ep, t, 3:7]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        dv = d.xpos[L, :2] - d.xpos[R, :2]
        lat.append(-np.sin(yaw) * dv[0] + np.cos(yaw) * dv[1])
        gmin = min(mujoco.mj_geomDistance(m, d, g1, g2, 0.2, None) for g1 in lg for g2 in rg)
        gap.append(gmin)
    lat, gap = np.array(lat), np.array(gap)
    out.append(dict(ep=ep, lat_median=float(np.median(lat)), lat_min=float(lat.min()), cross=float((lat <= 0).mean()),
                    geom_gap_min=float(gap.min()), penetrating=float((gap < 0).mean())))
res = {k: round(float(np.mean([o[k] for o in out])), 4) for k in out[0] if k != "ep"}
res["geom_gap_min"] = round(float(min(o["geom_gap_min"] for o in out)), 4)
res["label"] = a.label
print(json.dumps(res))
