"""Plot folding measurements and animate the recorded, unmodified trajectories."""

from __future__ import annotations

import argparse
import json
import subprocess
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

COLORS = {"ipc": "#cf534a", "vbd": "#278b75", "vbd_no_self": "#577cbd"}
LABELS = {
    "ipc": "IPC (plane only)",
    "vbd": "VBD + self-contact",
    "vbd_no_self": "VBD, self-contact off",
}


def summary(data):
    rows = []
    for nx in sorted({run["resolution"][0] for run in data["runs"]}):
        for hz in sorted({round(1 / run["dt_s"]) for run in data["runs"]}):
            row = {"resolution": [nx, nx // 2], "hz": hz}
            for method in LABELS:
                runs = [
                    r
                    for r in data["runs"]
                    if r["resolution"][0] == nx
                    and round(1 / r["dt_s"]) == hz
                    and r["method"] == method
                ]
                row[method] = {
                    "runs": len(runs),
                    "completed": sum(
                        r["completed_steps"] == r["requested_steps"] for r in runs
                    ),
                    "crossing_runs": sum(r["crossing_events"] > 0 for r in runs),
                    "surface_intersection_runs": sum(
                        r["max_triangles_cutting_panel"] > 0 for r in runs
                    ),
                    "maximum_edge_strain": max(r["maximum_edge_strain"] for r in runs),
                    "minimum_ground_gap_m": min(
                        r["minimum_ground_gap_m"] for r in runs
                    ),
                    "median_step_ms": float(
                        np.median([r["median_step_ms"] for r in runs])
                    ),
                    "median_p95_step_ms": float(
                        np.median([r["p95_step_ms"] for r in runs])
                    ),
                    "first_crossing_s": next(
                        (
                            r["first_crossing_s"]
                            for r in runs
                            if r["first_crossing_s"] is not None
                        ),
                        None,
                    ),
                    "contact_overflow_rows": sum(
                        r["contact_overflow_rows"] for r in runs
                    ),
                }
            rows.append(row)
    return rows


def plot(data, output):
    for theme, mobile in product(("light", "dark"), (False, True)):
        fg = "#25303c" if theme == "light" else "#e7edf3"
        grid = "#d9dfe6" if theme == "light" else "#455565"
        with plt.rc_context(
            {
                "text.color": fg,
                "axes.labelcolor": fg,
                "xtick.color": fg,
                "ytick.color": fg,
                "axes.edgecolor": fg,
                "font.size": 10,
                "svg.fonttype": "none",
            }
        ):
            fig, axes = plt.subplots(
                2 if mobile else 1,
                1 if mobile else 2,
                figsize=(4.2, 6.5) if mobile else (10, 3.5),
                layout="constrained",
            )
            for method, label in LABELS.items():
                row = next(
                    r
                    for r in data["runs"]
                    if r["method"] == method
                    and r["resolution"][0] == 16
                    and round(1 / r["dt_s"]) == 240
                    and r["repeat"] == 0
                )
                time = [s["time_s"] for s in row["samples"]]
                axes[0].plot(
                    time,
                    [s["triangles_cutting_panel"] for s in row["samples"]],
                    color=COLORS[method],
                    label=label,
                    lw=1.7,
                )
                axes[1].plot(
                    time,
                    [100 * s["edge_strain_max"] for s in row["samples"]],
                    color=COLORS[method],
                    label=label,
                    lw=1.7,
                )
            axes[0].set(
                ylabel="Triangles cutting lower panel [count]",
                title="Fold intersections",
                xlim=(0, 0.5),
            )
            axes[1].set(
                ylabel="Maximum edge length change [%]",
                title="Deformation during closure",
                xlim=(0, 2),
            )
            for ax in axes:
                ax.set_xlabel("Simulated time [s]")
                ax.grid(color=grid, alpha=0.65)
                ax.spines[["top", "right"]].set_visible(False)
            axes[1].legend(frameon=False, fontsize=8, labelcolor=fg, loc="upper right")
            suffix = "-mobile" if mobile else ""
            fig.savefig(output / f"folding-{theme}{suffix}.svg", transparent=True)
            plt.close(fig)


def video(data, input_dir, output, ffmpeg, *, selections=None):
    if selections is None:
        selections = [
            (
                method,
                input_dir,
                next(
                    r for r in data["runs"] if r["id"] == f"{method}-n16-k500-hz240-r0"
                ),
            )
            for method in LABELS
        ]
    methods = [selection[0] for selection in selections]
    archives = [np.load(root / f"{row['id']}.npz") for _, root, row in selections]
    rows = [selection[2] for selection in selections]
    lower = np.min([a["positions"].min(axis=(0, 1)) for a in archives], axis=0) - 0.015
    upper = np.max([a["positions"].max(axis=(0, 1)) for a in archives], axis=0) + 0.015
    xlim = (min(-0.06, lower[0]), max(0.70, upper[0]))
    ylim = (min(-0.04, lower[1]), max(0.44, upper[1]))
    zlim = (min(-0.01, lower[2]), max(0.25, upper[2]))
    fig = plt.figure(figsize=(12.8, 6.8), dpi=100, facecolor="white")
    axes3d, axes2d, surfaces, lines, texts = [], [], [], [], []
    for col, (method, archive) in enumerate(zip(methods, archives, strict=True)):
        ax = fig.add_subplot(2, 3, col + 1, projection="3d")
        ax.set_title(LABELS[method], color=COLORS[method], fontsize=13, pad=4)
        ax.set(xlim=xlim, ylim=ylim, zlim=zlim)
        ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]))
        ax.view_init(elev=18, azim=-70)
        ax.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
        ax.tick_params(labelsize=7, pad=0)
        tri = archive["triangles"]
        pinned_tri = archive["pinned"][tri].all(axis=1)
        poly = Poly3DCollection(
            archive["positions"][0][tri],
            facecolors=["#9aadb0" if fixed else COLORS[method] for fixed in pinned_tri],
            edgecolors="#434c53",
            linewidths=0.15,
            alpha=0.96,
        )
        ax.add_collection3d(poly)
        bx = fig.add_subplot(2, 3, col + 4)
        bx.axhline(0, color="#aaa", lw=1)
        bx.plot(
            [0, 0.35], [0.12, 0.12], color="#34454c", lw=4, label="Fixed lower panel"
        )
        midrow = np.isclose(archive["rest"][:, 1], 0.2)
        (line,) = bx.plot([], [], color=COLORS[method], lw=2, marker=".", ms=3)
        bx.set(xlim=xlim, ylim=zlim, xlabel="x [m]", ylabel="z [m]")
        bx.set_title("Midline cross-section (y ≈ 0.20 m)", fontsize=10)
        bx.grid(color="#ddd", linewidth=0.5)
        bx.spines[["top", "right"]].set_visible(False)
        text = bx.text(
            0.02,
            0.04,
            "",
            transform=bx.transAxes,
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
        )
        axes3d.append(ax)
        axes2d.append(bx)
        surfaces.append(poly)
        lines.append((line, midrow))
        texts.append(text)
    fig.subplots_adjust(
        left=0.055, right=0.97, top=0.88, bottom=0.08, wspace=0.30, hspace=0.38
    )
    title = fig.suptitle("", fontsize=14)
    fps, frames = 30, 240
    pipe = subprocess.Popen(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "1280x680",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output / "folding-comparison.mp4"),
        ],
        stdin=subprocess.PIPE,
    )
    try:
        for frame in range(frames):
            sim_time = round(frame / (frames - 1) * 120) / 60
            title.set_text(
                f"Cloth fold closure · t = {sim_time:.3f} s · quarter-speed playback"
            )
            for index, archive in enumerate(archives):
                fi = min(
                    round(sim_time / float(archive["frame_dt"])),
                    len(archive["positions"]) - 1,
                )
                q = archive["positions"][fi]
                surfaces[index].set_verts(q[archive["triangles"]])
                line, midrow = lines[index]
                line.set_data(q[midrow, 0], q[midrow, 2])
                samples = rows[index]["samples"]
                sample = samples[
                    min(
                        max(round(sim_time / rows[index]["dt_s"]) - 1, 0),
                        len(samples) - 1,
                    )
                ]
                status = (
                    "IPC converged"
                    if rows[index]["method"] == "ipc"
                    else "VBD: fixed 20 iterations"
                )
                cuts = sample["triangles_cutting_panel"] if sim_time else 0
                first_crossing = rows[index]["first_crossing_s"]
                crossed = first_crossing is not None and sim_time >= first_crossing
                texts[index].set_text(
                    f"Panel crossing: {'DETECTED' if crossed else 'none yet'}\n"
                    f"Current surface cuts: {cuts}\n{status}"
                )
            fig.canvas.draw()
            rgb = np.ascontiguousarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
            pipe.stdin.write(rgb.tobytes())
            if frame == 18:
                Image.fromarray(rgb).save(
                    output / "folding-comparison-poster.jpg", quality=93
                )
            if frame % 60 == 0:
                print(f"Rendered folding video frame {frame}/{frames}", flush=True)
    finally:
        pipe.stdin.close()
        code = pipe.wait()
        plt.close(fig)
    if code:
        raise RuntimeError(f"video encoder exited {code}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg")
    args = parser.parse_args()
    data = json.loads((args.input_dir / "results.json").read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.input_dir / "summary.json").write_text(
        json.dumps(summary(data), indent=2) + "\n"
    )
    plot(data, args.output_dir)
    if args.ffmpeg:
        video(data, args.input_dir, args.output_dir, args.ffmpeg)


if __name__ == "__main__":
    main()
