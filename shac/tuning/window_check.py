"""Runs test_window.py for Go1 and H1 (stored-forward vs recompute, and recompute vs recompute) and saves the output."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
out = []
for task in ("go1_vel", "h1_vel"):
    for modes in ("recompute,window", "recompute,recompute", "window,window"):
        r = subprocess.run([sys.executable, "test_window.py", task, modes], cwd=ROOT, capture_output=True, text=True)
        out.append(f"### {task} {modes}\n{r.stdout}{r.stderr[-2000:] if r.returncode else ''}")
        print(out[-1], flush=True)
(ROOT / "results" / "loop" / "window_check.txt").write_text("\n".join(out))
