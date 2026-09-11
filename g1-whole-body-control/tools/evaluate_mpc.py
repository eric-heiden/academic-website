"""Evaluate a published MPC manifest from a Newton checkout; archives stay local.

Run with the WBC environment and optional pinned MuJoCo Warp on PYTHONPATH.
Each run uses a fresh process to release its CUDA graphs and trajectory buffers.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path


def git_revision(path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest", type=Path, help="JSON list of tagged runs and CLI arguments"
    )
    parser.add_argument("--motions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", help="Run one tag instead of the complete manifest")
    parser.add_argument("--case-index", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    runs = json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    if args.tag and not any(run["tag"] == args.tag for run in runs):
        parser.error(f"Unknown run tag: {args.tag}")
    if args.case_index is None:
        failed = []
        for index, run in enumerate(runs):
            if args.tag and run["tag"] != args.tag:
                continue
            print("START", run["tag"], flush=True)
            command = [
                sys.executable,
                __file__,
                str(args.manifest),
                "--motions",
                str(args.motions),
                "--output",
                str(args.output),
                "--case-index",
                str(index),
            ]
            with (args.output / (run["tag"] + ".log")).open("w") as log:
                process = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=os.environ.copy(),
                    check=False,
                )
            print("FINISHED", run["tag"], "exit", process.returncode, flush=True)
            if process.returncode:
                failed.append(run["tag"])
        if failed:
            raise RuntimeError(f"Failed runs: {failed}")
        return

    import mujoco_warp as mjw
    import newton.viewer
    from newton.examples.robot import example_robot_g1_wbc as module

    run = runs[args.case_index]
    path = args.output / run["tag"]
    argv = [
        "--viewer",
        "null",
        "--test",
        "--show-rollouts",
        "--num-frames",
        str(run["frames"]),
        "--output",
        str(path),
        *run["arguments"],
    ]
    if run["motion"] != "stand":
        argv += ["--motion", str(args.motions / (run["motion"] + ".csv"))]
    source = Path(module.__file__).parent
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [source / "example_robot_g1_wbc.py", *source.glob("wbc_*.py")]
    }
    start = time.perf_counter()
    example = module.Example(
        newton.viewer.ViewerNull(), module.Example.create_parser().parse_args(argv)
    )
    setup = time.perf_counter() - start
    failure = None
    try:
        for _ in range(run["frames"]):
            example.step()
        example.test_final()
    except Exception as error:  # noqa: BLE001 — save the failed trajectory before re-raising
        failure = repr(error)
        traceback.print_exc()
    if not example.rows:
        raise RuntimeError(failure or "No recorded simulation steps")
    example.save()
    metadata = path.with_suffix(".json")
    result = json.loads(metadata.read_text())
    result.update(
        source_sha256=hashes,
        code_commit=git_revision(source),
        dependency_commit=git_revision(Path(mjw.__file__).parent),
        setup_seconds=setup,
        requested_frames=run["frames"],
        test_failure=failure,
    )
    metadata.write_text(json.dumps(result, indent=2) + "\n")
    if failure:
        raise RuntimeError(failure)


if __name__ == "__main__":
    main()
