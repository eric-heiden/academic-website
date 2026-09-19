"""Compare the archived plane-only trajectory with the surface-contact rerun."""

import argparse
import json
from pathlib import Path

import plot_folding as plots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--before-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg")
    args = parser.parse_args()
    after = json.loads((args.input_dir / "results.json").read_text())
    before = json.loads((args.before_dir / "results.json").read_text())
    (args.input_dir / "summary.json").write_text(
        json.dumps(plots.summary(after), indent=2) + "\n"
    )
    plots.LABELS = {
        "ipc_before": "IPC before (plane only)",
        "ipc": "IPC + self-contact",
        "vbd": "VBD + self-contact",
    }
    plots.COLORS["ipc_before"] = "#818c99"
    combined = {
        "runs": after["runs"]
        + [
            dict(row, method="ipc_before")
            for row in before["runs"]
            if row["method"] == "ipc"
        ]
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots.plot(combined, args.output_dir)
    if args.ffmpeg:
        selections = []
        for label, method, root, data in (
            ("ipc_before", "ipc", args.before_dir, before),
            ("ipc", "ipc", args.input_dir, after),
            ("vbd", "vbd", args.input_dir, after),
        ):
            row = next(
                r for r in data["runs"] if r["id"] == f"{method}-n16-k500-hz240-r0"
            )
            selections.append((label, root, row))
        plots.video(
            after, args.input_dir, args.output_dir, args.ffmpeg, selections=selections
        )


if __name__ == "__main__":
    main()
