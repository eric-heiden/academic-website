"""Isolated SHAC epoch timing per robot and simulator configuration.

Runs ``shac.py`` for a few epochs with the GPU otherwise idle and reports the
median epoch time over epochs 5.. and its split into rollout (forward
simulation and policy), backward (analytic simulator adjoint and actor), and
critic (TD(lambda) targets and 64 minibatch updates).
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path

from run_matrix import CONFIGS, PY, ROOT

p = argparse.ArgumentParser()
p.add_argument("--robots", nargs="+", required=True)
p.add_argument("--configs", nargs="+", default=list(CONFIGS))
p.add_argument("--horizon", type=int, default=8)
p.add_argument("--num-envs", type=int, nargs="+", default=[256])
p.add_argument("--epochs", type=int, default=25)
p.add_argument("--out", required=True)
args = p.parse_args()

rows = []
for robot in args.robots:
    for n in args.num_envs:
        for cfg in args.configs:
            integ, dt = CONFIGS[cfg]
            tmp = ROOT / "runs" / "timing" / f"{robot}_{cfg}_n{n}_T{args.horizon}"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            cmd = [PY, str(ROOT / "shac.py"), "--robot", robot, "--integrator", integ, "--dt", str(dt),
                   "--horizon", str(args.horizon), "--epochs", str(args.epochs), "--num-envs", str(n),
                   "--out", str(tmp)]
            subprocess.run(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            h = json.loads(tmp.with_suffix(".json").read_text())["history"][5:]
            med = {k: statistics.median(r[k] for r in h) for k in ("epoch_s", "rollout_s", "backward_s", "critic_s")}
            row = {"robot": robot, "config": cfg, "integrator": integ, "timestep": dt, "num_envs": n,
                   "horizon": args.horizon, **{f"median_{k}": v for k, v in med.items()},
                   "env_steps_per_s": n * args.horizon / med["epoch_s"]}
            rows.append(row)
            print(json.dumps(row), flush=True)
            Path(args.out).write_text(json.dumps(rows, indent=1))
