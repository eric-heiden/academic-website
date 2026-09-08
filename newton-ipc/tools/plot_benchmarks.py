"""Generate report figures from the checked-in SolverIPC benchmark JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

INK = "#19212b"
BLUE = "#2563a5"
GOLD = "#c7861a"
GREEN = "#2f855a"
RED = "#b94747"
GRID = "#d9dee5"


def annotate_bars(axis, bars, *, suffix="", scale=1.0) -> None:
    for bar in bars:
        value = bar.get_height()
        axis.annotate(
            f"{value * scale:.2f}{suffix}",
            (bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def performance_figure(data: dict, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.7), constrained_layout=True)
    fig.suptitle(
        "SolverIPC CUDA microbenchmarks", color=INK, fontsize=15, fontweight="bold"
    )

    graph = data["graph_modes"]
    bars = axes[0, 0].bar(
        [row["mode"] for row in graph],
        [row["median_us"] / 1000.0 for row in graph],
        color=[BLUE, GOLD],
    )
    axes[0, 0].set_title("Early-exit conditional graph")
    axes[0, 0].set_ylabel("Replay time [ms]")
    annotate_bars(axes[0, 0], bars, suffix=" ms")

    reductions = data["reductions"]
    bars = axes[0, 1].bar(
        ["Per-element\natomic", "Tile reduction"],
        [row["median_us"] for row in reductions],
        color=[RED, GREEN],
    )
    axes[0, 1].set_title("1M-vector dot reduction")
    axes[0, 1].set_ylabel("Replay time [µs]")
    axes[0, 1].set_yscale("log")
    annotate_bars(axes[0, 1], bars, suffix=" µs")

    factors = data["factorizations"]
    bars = axes[1, 0].bar(
        ["Rank-one", "Dense inverse"],
        [row["median_us"] for row in factors],
        color=[GREEN, GOLD],
    )
    axes[1, 0].set_title("1M block-Jacobi factors")
    axes[1, 0].set_ylabel("Replay time [µs]")
    axes[1, 0].set_ylim(175.0, 200.0)
    annotate_bars(axes[1, 0], bars, suffix=" µs")

    scaling = data["cloth_scaling"]
    counts = [row["particle_count"] for row in scaling]
    timings = [row["median_us"] / 1000.0 for row in scaling]
    axes[1, 1].plot(
        counts, timings, marker="o", linewidth=2.2, markersize=7, color=BLUE
    )
    for count, timing in zip(counts, timings, strict=True):
        axes[1, 1].annotate(
            f"{timing:.2f} ms",
            (count, timing),
            xytext=(5, 5),
            textcoords="offset points",
        )
    axes[1, 1].set_title("Contact-active cloth step")
    axes[1, 1].set_xlabel("Particles")
    axes[1, 1].set_ylabel("Replay time [ms]")

    for axis in axes.flat:
        axis.grid(axis="y", color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg", bbox_inches="tight")
    plt.close(fig)


def tuning_figure(data: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.7), constrained_layout=True)
    fig.suptitle(
        "Search tuning and long-horizon validation",
        color=INK,
        fontsize=15,
        fontweight="bold",
    )

    sweep = data["step_size_sweep"]
    step_sizes = np.asarray([row["initial_step_size"] for row in sweep])
    wall = np.asarray([row["wall_ms"] for row in sweep])
    newton = np.asarray([row["newton_iterations"] for row in sweep])
    line = np.asarray([row["line_search_iterations"] for row in sweep])
    axes[0].plot(
        step_sizes, wall, marker="o", linewidth=2.2, color=BLUE, label="Wall time [ms]"
    )
    axes[0].plot(
        step_sizes,
        newton,
        marker="s",
        linewidth=1.8,
        color=GREEN,
        label="Newton updates",
    )
    axes[0].plot(
        step_sizes,
        line,
        marker="^",
        linewidth=1.8,
        color=GOLD,
        label="Line-search trials",
    )
    axes[0].set_title("One transient 8×8 cloth step")
    axes[0].set_xlabel("Initial trial multiplier")
    axes[0].legend(frameon=False, fontsize=8)

    tuning = data["stability_tuning"]
    labels = ["Strict\nfloat32", "Scale-aware\nfloat32", "Selected\nlong run"]
    completed = [
        tuning[0]["simulated_steps"],
        tuning[1]["simulated_steps"],
        data["stability"]["simulated_steps"],
    ]
    requested = [
        2 * tuning[0]["requested_frames"],
        2 * tuning[1]["requested_frames"],
        2 * data["stability"]["requested_frames"],
    ]
    colors = [RED if completed[0] < requested[0] else GREEN, GREEN, GREEN]
    bars = axes[1].bar(labels, completed, color=colors)
    axes[1].scatter(
        range(3),
        requested,
        marker="_",
        s=500,
        linewidths=2,
        color=INK,
        label="Requested",
    )
    axes[1].set_title("Captured substeps before failure")
    axes[1].set_ylabel("Substeps")
    axes[1].legend(frameon=False, fontsize=8)
    for bar, value, target in zip(bars, completed, requested, strict=True):
        label = f"{value}/{target}"
        axes[1].annotate(
            label,
            (bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    for axis in axes:
        axis.grid(axis="y", color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    performance_figure(data, args.output_dir / "performance.svg")
    tuning_figure(data, args.output_dir / "tuning-and-stability.svg")


if __name__ == "__main__":
    main()
