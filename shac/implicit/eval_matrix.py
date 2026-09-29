"""Evaluates every trained checkpoint in its training simulator and a common reference.

The reference simulator (implicitfast, 1 ms physics step, same 20 ms control
interval) is finer than every training configuration, so a policy that relies on
large-step integration artifacts loses performance there.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from run_matrix import PY, ROOT

REF = ("implicitfast", 0.001)

p = argparse.ArgumentParser()
p.add_argument("--tag", default="e4")
p.add_argument("--robots", nargs="*", default=None)
p.add_argument("--steps", type=int, default=1000)
p.add_argument("--num-envs", type=int, default=256)
args = p.parse_args()

out_dir = ROOT / "results" / f"{args.tag}_eval"
out_dir.mkdir(parents=True, exist_ok=True)
for ck in sorted((ROOT / "runs" / args.tag).glob("*.pt")):
    name = ck.stem
    if args.robots and not any(name.startswith(r + "_") for r in args.robots):
        continue
    hist = json.loads(ck.with_suffix(".json").read_text())
    if hist["history"][-1]["epoch"] != hist["args"]["epochs"] - 1:
        continue  # still training
    for which, extra in (("train", []), ("ref", ["--integrator", REF[0], "--dt", str(REF[1])])):
        out = out_dir / f"{name}_{which}.json"
        if out.exists():
            continue
        cmd = [PY, str(ROOT / "evaluate.py"), "--checkpoint", str(ck), "--steps", str(args.steps),
               "--num-envs", str(args.num_envs), "--out", str(out), *extra]
        subprocess.run(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        print(out.name, "ok" if out.exists() else "FAILED", flush=True)
