"""Evaluates a checkpoint in the reference simulator and renders a surviving episode."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from run_matrix import PY, ROOT

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--out", required=True, help="output mp4 path")
p.add_argument("--seconds", type=float, default=10.0)
p.add_argument("--distance", type=float, default=3.0)
p.add_argument("--azimuth", type=float, default=120.0)
p.add_argument("--elevation", type=float, default=-15.0)
args = p.parse_args()

out = Path(args.out)
ev = out.with_suffix(".eval.json")
steps = round(args.seconds / 0.02)
subprocess.run([PY, str(ROOT / "evaluate.py"), "--checkpoint", args.checkpoint, "--integrator", "implicitfast", "--dt", "0.001",
                "--num-envs", "32", "--steps", str(steps), "--record", "8", "--seed", "777", "--out", str(ev)],
               cwd=ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
q = np.load(ev.with_suffix(".npz"))["qpos"]
# First recorded episode without a reset jump (a fall triggers an automatic reset).
episode = next(i for i in range(q.shape[0]) if np.all(np.abs(np.diff(q[i, :, 2])) < 0.2))
subprocess.run([PY, str(ROOT / "render.py"), "--npz", str(ev.with_suffix(".npz")), "--episode", str(episode), "--out", str(out),
                "--seconds", str(args.seconds), "--distance", str(args.distance), "--azimuth", str(args.azimuth),
                "--elevation", str(args.elevation), "--poster-time", "3"], cwd=ROOT, check=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
meta = json.loads(ev.read_text())
meta["rendered_episode"] = episode
ev.write_text(json.dumps(meta, indent=1))
print(out, "episode", episode, {k: round(meta[k], 3) for k in ("survival_fraction", "mean_forward_speed")})
