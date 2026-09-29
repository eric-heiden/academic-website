"""Renders one policy in two simulators from the same start state and stacks the videos.

Left: the training simulator; right: the reference simulator (implicitfast, 1 ms).
"""

import argparse
import subprocess
from pathlib import Path

import imageio_ffmpeg

from run_matrix import PY, ROOT

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--train-dt", type=float, required=True)
p.add_argument("--out", required=True)
p.add_argument("--seconds", type=float, default=6.0)
p.add_argument("--distance", type=float, default=1.8)
p.add_argument("--azimuth", type=float, default=110.0)
p.add_argument("--elevation", type=float, default=-10.0)
args = p.parse_args()

out = Path(args.out)
parts = []
for tag, dt in (("train", args.train_dt), ("ref", 0.001)):
    ev = out.with_name(out.stem + f"_{tag}.eval.json")
    subprocess.run([PY, str(ROOT / "evaluate.py"), "--checkpoint", args.checkpoint, "--integrator", "implicitfast",
                    "--dt", str(dt), "--num-envs", "16", "--steps", str(round(args.seconds / 0.02)), "--record", "1",
                    "--seed", "777", "--out", str(ev)], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    mp4 = out.with_name(out.stem + f"_{tag}.mp4")
    subprocess.run([PY, str(ROOT / "render.py"), "--npz", str(ev.with_suffix(".npz")), "--out", str(mp4),
                    "--width", "640", "--height", "480", "--seconds", str(args.seconds), "--distance", str(args.distance),
                    "--azimuth", str(args.azimuth), "--elevation", str(args.elevation), "--poster-time", "3"],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    parts.append(mp4)
ff = imageio_ffmpeg.get_ffmpeg_exe()
subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(parts[0]), "-i", str(parts[1]), "-filter_complex",
                "[0:v][1:v]hstack=inputs=2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24",
                "-movflags", "+faststart", str(out)], check=True)
poster = out.with_name(out.stem + "_poster.jpg")
subprocess.run([ff, "-y", "-loglevel", "error", "-ss", "3", "-i", str(out), "-frames:v", "1", "-q:v", "3", str(poster)],
               check=True)
print("wrote", out, poster)
