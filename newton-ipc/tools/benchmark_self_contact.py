"""Rerun the folding protocol with IPC surface contact and explicit settings."""

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import benchmark_folding as folding
import newton
import warp as wp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--hz", type=int, nargs="+", default=[60, 120, 240])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--methods", nargs="+", default=["ipc", "vbd", "vbd_no_self"])
    parser.add_argument("--projective", action="store_true")
    args = parser.parse_args()
    wp.init()
    device = wp.get_device("cuda:0")
    root = Path(newton.__file__).resolve().parents[1]
    config = {
        "enable_self_contact": True,
        "self_contact_thickness": 0.008,
        "self_contact_distance": 0.01,
        "self_contact_stiffness": 0.001,
        "use_projective_hessian": args.projective,
        "energy_tolerance": 1.0e-8,
    }
    sources = [
        root / "newton/_src/solvers/ipc" / name
        for name in ("solver_ipc.py", "kernels.py", "self_contact.py")
    ]
    result = {
        "environment": {
            "newton_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "source_sha256": {
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sources
            },
            "warp": wp.__version__,
            "python": platform.python_version(),
            "device": device.name,
        },
        "ipc_overrides": config,
        "protocol": "Same rest mesh, folded start, mass, material tangent, and per-substep oracle as benchmark_folding.py. 128 Newton / 32 PCG / 24 line-search iterations; VBD 20 iterations. Solver wall times exclude host diagnostics. A shared GPU may affect timing.",
        "runs": [],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for nx in args.resolution:
        for hz in args.hz:
            for repeat in range(args.repeats):
                for method in args.methods:
                    result["runs"].append(
                        folding.run_case(
                            method,
                            nx,
                            500.0,
                            1.0 / hz,
                            args.duration,
                            repeat,
                            device,
                            args.output,
                            ipc_config=config,
                        )
                    )
                    (args.output / "results.json").write_text(
                        json.dumps(result, indent=2, allow_nan=False) + "\n"
                    )


if __name__ == "__main__":
    main()
