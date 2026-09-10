"""Collect experiment summaries without rounding raw measurements."""

import argparse, json, hashlib, shutil, platform, subprocess
from pathlib import Path
from importlib.metadata import version

p = argparse.ArgumentParser()
p.add_argument("experiments", type=Path)
p.add_argument("assets", type=Path)
a = p.parse_args()
rows = []
for f in sorted((a.experiments / "final").glob("*_123.json")):
    r = json.loads(f.read_text())
    r["clip"] = f.stem.rsplit("_", 2)[0]
    r["trajectory"] = f.stem + ".npz"
    rows.append(r)
    shutil.copy2(f.with_suffix(".npz"), a.assets / r["trajectory"])
(a.assets / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
meta = dict(
    python=platform.python_version(),
    packages={
        n: version(n)
        for n in [
            "newton",
            "mujoco",
            "mujoco-warp",
            "warp-lang",
            "numpy",
            "scipy",
            "osqp",
        ]
    },
    base_commit="31f58571",
    cpu_quota=Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
    platform=platform.platform(),
)
meta["sources"] = json.loads((a.experiments / "sources.json").read_text())
meta["backflip"] = dict(
    source="https://huggingface.co/spaces/nvidia/kimodo",
    model="Kimodo-G1-RP-v1",
    date="2026-09-10",
    prompt="A person performs a standing backflip, rotating backwards once in the air, lands on both feet and stands still.",
    seed=42,
    steps=100,
    duration_seconds=6,
)
meta["motions"] = {}
(a.assets / "motions").mkdir(exist_ok=True)
for f in (a.experiments / "motions").glob("*.csv"):
    meta["motions"][f.name] = dict(
        sha256=hashlib.sha256(f.read_bytes()).hexdigest(), fps=30
    )
    shutil.copy2(f, a.assets / "motions" / f.name)
(a.assets / "provenance.json").write_text(json.dumps(meta, indent=2) + "\n")
print("Collected", len(rows), "runs")
