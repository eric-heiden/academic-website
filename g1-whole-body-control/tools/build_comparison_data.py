"""Build compact report JSON from local MPC results; never publish trajectory archives."""

import argparse
import json
from pathlib import Path

import mujoco
import newton.utils
import numpy as np
from newton.examples.robot.wbc_controller import MotionReference

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("manifest", type=Path)
parser.add_argument("--runs", type=Path, required=True)
parser.add_argument("--assets", type=Path, required=True)
a = parser.parse_args()
cases = json.loads(a.manifest.read_text())
a.assets.mkdir(parents=True, exist_ok=True)
model = mujoco.MjModel.from_xml_path(
    str(newton.utils.download_asset("unitree_g1") / "mjcf/g1_29dof_rev_1_0.xml")
)
data = mujoco.MjData(model)
feet = [model.body(side + "_ankle_roll_link").id for side in ("left", "right")]
hands = [model.body(side + "_wrist_yaw_link").id for side in ("left", "right")]
head = model.body("torso_link").id
corners = np.array([[x, y, -0.035] for x in (-0.05, 0.12) for y in (-0.025, 0.025)])
labels = {
    "fd": "Short GN",
    "dial": "DIAL-MPC",
    "analytic": "Analytic GN",
    "hybrid": "Hybrid",
    "previous": "Original GN",
    "fast": "GN at 25 Hz",
}
motions = {
    "walk": "Walking",
    "dance": "Dancing",
    "jumpjack": "Jumping jacks",
    "backflip": "Backflip",
}
curves, media, results = {}, {}, []


def kinematics(q):
    data.qpos[:] = q
    mujoco.mj_kinematics(model, data)
    clearance = [
        float((data.xpos[b] + corners @ data.xmat[b].reshape(3, 3).T)[:, 2].min())
        for b in feet
    ]
    return data.xmat[head].reshape(3, 3).copy(), data.xpos[hands].copy(), clearance


for case in cases:
    result = json.loads((a.runs / (case["tag"] + ".json")).read_text())
    result.update(
        tag=case["tag"],
        method=case["method"],
        repeat=case["repeat"],
        clip=case["motion"],
    )
    result["motion"] = "assets/motions/" + case["motion"] + ".csv"
    result["config"]["motion"] = result["motion"]
    result["config"]["output"] = "local/" + case["tag"]
    result["eligible"] = bool(
        result["recovered"]
        and result["nonfoot_contact_seconds"] == 0
        and not result["test_failure"]
        and result["mpc_failures"] == 0
    )
    result["status"] = (
        "fell"
        if not result["recovered"]
        else "assisted"
        if result["nonfoot_contact_seconds"]
        else "search failure"
        if result["mpc_failures"]
        else "upright"
    )
    if result["test_failure"]:
        result["status"] = "stopped"
    if case["repeat"] != 1:
        results.append(result)
        continue
    raw = np.load(a.runs / (case["tag"] + ".npz"))
    reference = MotionReference(model, raw["reference"])
    indices = np.arange(
        0, len(raw["rows"]), max(1, round(0.04 / np.median(np.diff(raw["rows"][:, 0]))))
    )
    values = {"time": [], "head": [], "hand": [], "foot": [], "reference_foot": []}
    for i in indices:
        t = float(raw["rows"][i, 0])
        rotation, wrist, foot = kinematics(raw["qpos"][i])
        rotation_ref, wrist_ref, foot_ref = kinematics(reference.sample(t)[0])
        angle = np.rad2deg(
            np.arccos(np.clip((np.trace(rotation_ref.T @ rotation) - 1) / 2, -1, 1))
        )
        values["time"].append(round(t, 5))
        values["head"].append(round(float(angle), 5))
        values["hand"].append(
            np.linalg.norm(wrist - wrist_ref, axis=1).round(6).tolist()
        )
        values["foot"].append(np.round(foot, 6).tolist())
        values["reference_foot"].append(np.round(foot_ref, 6).tolist())
    if case["motion"] == "backflip":
        rotations = []
        for configurations in (
            raw["qpos"],
            [reference.sample(t)[0] for t in raw["rows"][:, 0]],
        ):
            matrices = []
            for q in configurations:
                data.qpos[:] = q
                mujoco.mj_kinematics(model, data)
                matrices.append(data.xmat[1].reshape(3, 3).copy())
            matrices = np.asarray(matrices)
            relative = np.einsum("ij,tjk->tik", matrices[0].T, matrices)
            angle = np.unwrap(np.arctan2(-relative[:, 2, 0], relative[:, 0, 0]))
            rotations.append(float((angle[-1] - angle[0]) / (2 * np.pi)))
        result["projected_pitch_turns"], result["reference_projected_pitch_turns"] = (
            rotations
        )
        completed = abs(rotations[1]) > 0.8 and 0.8 < rotations[0] / rotations[1] < 1.2
        result["flip_completed"] = bool(completed)
        result["eligible"] = result["eligible"] and completed
        if not completed:
            result["status"] = "incomplete" if result["recovered"] else "fell"
    results.append(result)
    clip, method = case["motion"], case["method"]
    curves.setdefault(clip, {})[method] = values
    prefix = "assets/comparison_" + clip + "_" + method
    media.setdefault(clip, {})[method] = {
        "file": prefix + ".mp4",
        "poster": prefix + ".jpg",
        "caption": motions[clip]
        + " · "
        + labels[method]
        + " · "
        + result["status"]
        + ".",
    }
(a.assets / "mpc-comparison-plots.json").write_text(
    json.dumps({"curves": curves, "media": media}, separators=(",", ":")) + "\n"
)
(a.assets / "mpc-comparison-results.json").write_text(
    json.dumps(results, indent=2) + "\n"
)
(a.assets / "mpc-comparison-runs.json").write_text(json.dumps(cases, indent=2) + "\n")
