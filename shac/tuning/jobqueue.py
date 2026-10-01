"""File-based GPU job queue for the research loop.

A job is a JSON file in jobs/pending/ with fields:
  name (str), cmd (list[str] or str), priority (int, lower runs first; default 50),
  exclusive (bool: run with the GPU otherwise idle, for timing), log (optional path).
Lanes (``python jobqueue.py lane --id N``) claim jobs by atomic rename into
jobs/running/, run them, and move them to jobs/done/ or jobs/failed/ with the
exit code and timings. ``python jobqueue.py submit ...`` adds a job.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
Q = ROOT / "jobs"
PY = "/home/horde/repos/mujoco_warp-pr1535-357a75d/.venv/bin/python"
EXCL = Q / "exclusive.flag"


def _jobs(state):
    out = []
    for p in (Q / state).glob("*.json"):
        try:
            out.append((json.loads(p.read_text()), p))
        except Exception:
            pass
    return out


def submit(name, cmd, priority=50, exclusive=False, log=None):
    job = {"name": name, "cmd": cmd, "priority": priority, "exclusive": exclusive,
           "log": log or str(ROOT / "logs" / "loop" / f"{name}.log"), "submitted": time.time()}
    path = Q / "pending" / f"{name}.json"
    if any((Q / s / f"{name}.json").exists() for s in ("running", "done")):
        print(f"skip {name}: already running/done")
        return
    path.write_text(json.dumps(job, indent=1))
    print(f"submitted {name}")


def lane(lane_id, poll=5.0):
    stop = Q / f"stop_lane{lane_id}"
    while not stop.exists() and not (Q / "stop_all").exists():
        pending = sorted(_jobs("pending"), key=lambda jp: (jp[0].get("priority", 50), jp[0].get("submitted", 0)))
        if not pending or (EXCL.exists() and EXCL.read_text().strip() != str(lane_id)):
            time.sleep(poll)
            continue
        job, path = pending[0]
        dst = Q / "running" / path.name
        try:
            os.rename(path, dst)
        except OSError:
            continue  # another lane claimed it
        if job.get("exclusive"):
            EXCL.write_text(str(lane_id))
            # Wait for other lanes' jobs to finish.
            while any(p.name != dst.name for _, p in _jobs("running")):
                time.sleep(poll)
        log = Path(job["log"])
        log.parent.mkdir(parents=True, exist_ok=True)
        cmd = job["cmd"] if isinstance(job["cmd"], list) else shlex.split(job["cmd"])
        cmd = [PY if c == "PY" else c for c in cmd]
        job.update(lane=lane_id, started=time.time(), pid=None)
        dst.write_text(json.dumps(job, indent=1))
        with open(log, "a") as f:
            f.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S')} lane {lane_id}: {shlex.join(cmd)}\n")
            f.flush()
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
            job["pid"] = proc.pid
            dst.write_text(json.dumps(job, indent=1))
            rc = proc.wait()
        job.update(finished=time.time(), returncode=rc, wall_s=time.time() - job["started"])
        (Q / ("done" if rc == 0 else "failed") / path.name).write_text(json.dumps(job, indent=1))
        dst.unlink(missing_ok=True)
        if job.get("exclusive") and EXCL.exists():
            EXCL.unlink()
    print(f"lane {lane_id} stopped")


def status():
    for state in ("running", "pending", "failed"):
        jobs = sorted(_jobs(state), key=lambda jp: jp[0].get("priority", 50))
        print(f"{state}: {len(jobs)}")
        for j, _ in jobs[:40]:
            extra = f" lane={j.get('lane')} {time.time() - j.get('started', time.time()):.0f}s" if state == "running" else ""
            print(f"  [{j.get('priority', 50)}] {j['name']}{' (excl)' if j.get('exclusive') else ''}{extra}")
    print(f"done: {len(_jobs('done'))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="op", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--name", required=True)
    s.add_argument("--priority", type=int, default=50)
    s.add_argument("--exclusive", action="store_true")
    s.add_argument("cmd", nargs=argparse.REMAINDER)
    l = sub.add_parser("lane")
    l.add_argument("--id", type=int, required=True)
    sub.add_parser("status")
    a = ap.parse_args()
    if a.op == "submit":
        cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
        submit(a.name, cmd, a.priority, a.exclusive)
    elif a.op == "lane":
        lane(a.id)
    else:
        status()
