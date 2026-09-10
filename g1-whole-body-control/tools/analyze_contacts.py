"""Geometric ground-contact and inversion audit of saved reference and plant poses."""

import argparse, json
from pathlib import Path
import mujoco
import numpy as np
import newton.viewer
from newton.examples.robot.example_robot_g1_wbc import Example
from newton.examples.robot.wbc_controller import MotionReference

p = argparse.ArgumentParser()
p.add_argument("trajectory")
p.add_argument("output")
a = p.parse_args()
ex = Example(
    newton.viewer.ViewerNull(), Example.create_parser().parse_args(["--viewer", "null"])
)
m = ex.mj
d = mujoco.MjData(m)
raw = np.load(a.trajectory)
ref = MotionReference(m, raw["reference"])
result = {}
for name, poses in [
    ("reference", [ref.sample(t)[0] for t in raw["rows"][:, 0]]),
    ("simulation", raw["qpos"]),
]:
    events = []
    for t, q in zip(raw["rows"][:, 0], poses, strict=True):
        d.qpos[:] = q
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        bodies = []
        for c in d.contact:
            if c.dist > 0.001:
                continue
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if b1 == 0 or b2 == 0:
                bodies.append(m.body(max(b1, b2)).name)
        nonfoot = [
            b
            for b in bodies
            if not b.endswith(("left_ankle_roll_link", "right_ankle_roll_link"))
        ]
        events.append(
            dict(
                t=float(t),
                up=float(d.xmat[1].reshape(3, 3)[2, 2]),
                ground_bodies=sorted(set(bodies)),
                nonfoot=sorted(set(nonfoot)),
            )
        )
    result[name] = dict(
        inverted_frames=sum(e["up"] < 0 for e in events),
        airborne_frames=sum(not e["ground_bodies"] for e in events),
        nonfoot_frames=sum(bool(e["nonfoot"]) for e in events),
        events=events,
    )
Path(a.output).write_text(json.dumps(result, indent=2))
print({k: {x: y for x, y in v.items() if x != "events"} for k, v in result.items()})
