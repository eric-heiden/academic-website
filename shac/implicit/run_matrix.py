"""Runs the SHAC training matrix as a small concurrent job queue.

Learning outcomes depend only on the number of epochs, not on wall-clock, so jobs
may share the GPU. Wall-clock per epoch is measured separately in isolation
(``time_epochs.py``) and used for any time-to-performance statement.
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import time
from pathlib import Path

PY = "/home/horde/repos/mujoco_warp-pr1535-357a75d/.venv/bin/python"
ROOT = Path(__file__).resolve().parent

CONFIGS = {
    "explicit_2ms": ("euler_explicit", 0.002),
    "implicit_2ms": ("implicitfast", 0.002),
    "implicit_5ms": ("implicitfast", 0.005),
    "implicit_10ms": ("implicitfast", 0.010),
    "implicit_20ms": ("implicitfast", 0.020),
}


def done(out: Path, epochs: int) -> bool:
    import json
    f = out.with_suffix(".json")
    if not f.exists():
        return False
    h = json.loads(f.read_text())
    return bool(h["history"]) and h["history"][-1]["epoch"] == epochs - 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--robots", nargs="+", required=True)
    p.add_argument("--configs", nargs="+", default=list(CONFIGS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--parallel", type=int, default=3)
    p.add_argument("--tag", default="e4")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = p.parse_args()

    jobs = []
    for robot, cfg, seed in itertools.product(args.robots, args.configs, args.seeds):
        integ, dt = CONFIGS[cfg]
        out = ROOT / "runs" / args.tag / f"{robot}_{cfg}_s{seed}"
        if out.with_suffix(".json").exists():
            import json
            h = json.loads(out.with_suffix(".json").read_text())
            if h["history"] and h["history"][-1]["epoch"] == args.epochs - 1:
                continue
        log = ROOT / "logs" / args.tag / f"{robot}_{cfg}_s{seed}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        cmd = [PY, str(ROOT / "shac.py"), "--robot", robot, "--integrator", integ, "--dt", str(dt),
               "--horizon", str(args.horizon), "--epochs", str(args.epochs), "--num-envs", str(args.num_envs),
               "--seed", str(seed), "--out", str(out), *args.extra]
        jobs.append((cmd, log))
    print(f"{len(jobs)} jobs", flush=True)
    running = []
    while jobs or running:
        running = [(p_, log) for p_, log in running if p_.poll() is None]
        while jobs and len(running) < args.parallel:
            cmd, log = jobs.pop(0)
            out = Path(cmd[cmd.index("--out") + 1])
            lock = out.with_suffix(".lock")
            if done(out, args.epochs) or lock.exists():
                continue  # finished or claimed by another runner
            lock.parent.mkdir(parents=True, exist_ok=True)
            lock.write_text(str(time.time()))
            f = open(log, "w")
            running.append((subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT), log))
            print(f"started {log.name}", flush=True)
        time.sleep(5)
    print("all done", flush=True)


if __name__ == "__main__":
    main()
