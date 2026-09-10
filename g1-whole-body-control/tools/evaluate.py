"""Sequential evaluation of fixed Kimodo clips; run inside the Newton environment."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("motions", type=Path)
p.add_argument("output", type=Path)
p.add_argument("--controllers", nargs="+", default=["pd", "qp", "mpc"])
p.add_argument(
    "--clips",
    nargs="+",
    default=["stand", "wave", "high5", "walk", "dance", "jumpjack", "backflip"],
)
p.add_argument("--seed", type=int, default=123)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
for controller in a.controllers:
    for clip in a.clips:
        stem = a.output / f"{clip}_{controller}_{a.seed}"
        command = [
            sys.executable,
            "-m",
            "newton.examples",
            "robot_g1_wbc",
            "--viewer",
            "null",
            "--num-frames",
            str(500 if clip == "stand" else 300 if clip == "backflip" else 250),
            "--controller",
            controller,
            "--seed",
            str(a.seed),
            "--output",
            str(stem),
        ]
        if clip != "stand":
            command += ["--motion", str(a.motions / f"{clip}.csv")]
        print(" ".join(command), flush=True)
        with stem.with_suffix(".log").open("w") as f:
            result = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{stem} failed: see log")
        print(stem.with_suffix(".json").read_text(), flush=True)
manifest = {
    p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in a.motions.glob("*.csv")
}
(a.output / "motion-sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
