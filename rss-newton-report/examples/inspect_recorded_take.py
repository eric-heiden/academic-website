"""Inspect the report's bundled take without running a simulation.

Run with Python and NumPy installed:
    python /path/to/report/examples/inspect_recorded_take.py
An optional positional argument selects another take directory.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "take", nargs="?", type=Path,
        default=Path(__file__).resolve().parents[1] / "evidence/rss-recorded-take/0000",
    )
    take = parser.parse_args().take
    with np.load(take / "steps.npz", allow_pickle=False) as steps:
        action = steps["action"]
        times = steps["sim_time"]
        cube_z = steps["obj_cube"][:, 2]
        result = {
            "samples": int(len(times)),
            "action_shape": list(action.shape),
            "monotonic_time": bool(np.all(np.diff(times) > 0)),
            "finite_step_arrays": all(bool(np.isfinite(steps[k]).all()) for k in steps.files),
            "max_action_component_range": float(np.ptp(action, axis=0).max()),
            "cube_height_range_m": float(np.ptp(cube_z)),
        }
    with np.load(take / "privileged.npz", allow_pickle=False) as privileged:
        result["privileged_shapes"] = {key: list(privileged[key].shape) for key in privileged.files}
    result["interpretation"] = (
        "The recorded viewer drag is an external intervention; robot actions alone "
        "do not explain the cube motion. Position arrays are not a complete solver checkpoint."
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
