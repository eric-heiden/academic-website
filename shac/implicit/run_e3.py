"""Gradient quality vs simulator configuration for a fixed trained policy.

For each robot, one trained SHAC checkpoint is evaluated with policy_gradcheck.py in
every simulator configuration of the training matrix, over horizons of 1-32
control steps. Holding the policy fixed isolates the effect of integrator and
timestep on the analytic policy gradient.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from run_matrix import CONFIGS, PY, ROOT

p = argparse.ArgumentParser()
p.add_argument("--robots", nargs="+", required=True)
p.add_argument("--checkpoint-config", default="implicit_10ms")
p.add_argument("--checkpoint", default=None, help="explicit checkpoint path (overrides --checkpoint-config)")
p.add_argument("--tag", default="e4")
p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
p.add_argument("--warmup", type=int, default=50)
p.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
args = p.parse_args()

for robot in args.robots:
    ck = Path(args.checkpoint) if args.checkpoint else ROOT / "runs" / args.tag / f"{robot}_{args.checkpoint_config}_s0.pt"
    for cfg, (integ, dt) in CONFIGS.items():
        for seed in args.seeds:
            out = ROOT / "results" / "e3" / f"{robot}_{cfg}_s{seed}.json"
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            cmd = [PY, str(ROOT / "policy_gradcheck.py"), "--robot", robot, "--integrator", integ, "--dt", str(dt),
                   "--checkpoint", str(ck), "--warmup", str(args.warmup), "--seed", str(100 + seed),
                   "--horizons", *map(str, args.horizons), "--out", str(out)]
            print(" ".join(cmd), flush=True)
            subprocess.run(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
