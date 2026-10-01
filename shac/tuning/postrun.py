"""Queues reference-simulator evaluations for finished training runs that have none yet.

For run runs/loop/<name>.pt it evaluates the final checkpoint and the periodic checkpoints
<name>_epNNNNN.pt (every ``--stride``-th), in the reference simulator, writing
results/loop/eval/<name>.json. Usage: python3 postrun.py [--stride 2] [--dry]
"""
import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--stride", type=int, default=0, help="0: final checkpoint only")
ap.add_argument("--dry", action="store_true")
ap.add_argument("--steps", type=int, default=500)
a = ap.parse_args()
out_dir = ROOT / "results" / "loop" / "eval"
out_dir.mkdir(parents=True, exist_ok=True)
for jf in sorted((ROOT / "jobs" / "done").glob("*.json")):
    job = json.loads(jf.read_text())
    cmd = job["cmd"]
    if not isinstance(cmd, list) or not any(c in ("shac2.py", "ppo.py") for c in cmd):
        continue
    run = Path(cmd[cmd.index("--out") + 1])
    name = run.name
    res = out_dir / f"{name}.json"
    if res.exists() or (ROOT / "jobs" / "pending" / f"eval_{name}.json").exists() or \
            (ROOT / "jobs" / "running" / f"eval_{name}.json").exists() or (ROOT / "jobs" / "done" / f"eval_{name}.json").exists():
        continue
    ck = sorted((ROOT / run.parent).glob(f"{name}_ep*.pt"))
    ck = (ck[::a.stride] if a.stride else []) + [ROOT / run.with_suffix(".pt")]
    args = ["python3", "jobqueue.py", "submit", "--name", f"eval_{name}", "--priority", "10", "--", "PY", "eval2.py",
            "--sim", "ref", "--steps", str(a.steps), "--out", str(res.relative_to(ROOT)), "--ckpt",
            *[str(c.relative_to(ROOT)) for c in ck]]
    print(" ".join(args[:6]), len(ck), "checkpoints")
    if not a.dry:
        subprocess.run(args, cwd=ROOT, check=True)
