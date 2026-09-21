"""Render standalone comparison figures from audited trials and fixed recorded windows."""

from __future__ import annotations

import argparse
import json
import math
import textwrap
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

COLORS = {
    "live": "#2463a0",
    "ipython": "#b46120",
    "restart": "#8461a8",
    "initial": "#777777",
}
LABELS = {
    "live": "Newton MCP",
    "ipython": "IPython MCP",
    "restart": "Restart",
    "initial": "Uniform initial model",
}


def save(figure, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / f"{name}.svg", bbox_inches="tight", transparent=True)
    figure.savefig(
        output / f"{name}.png", bbox_inches="tight", facecolor="white", dpi=180
    )
    svg = (output / f"{name}.svg").read_text()
    palette = {
        "#000000": "#e9f8f4",
        "#222222": "#b8c7c2",
        "#333333": "#b8c7c2",
        "#555555": "#9db9b3",
        "#777777": "#b8c7c2",
        "#dddddd": "#27433d",
        "#dedede": "#27433d",
        "#ffffff": "#091713",
        "#2463a0": "#70b5f0",
        "#b46120": "#f1a565",
        "#8461a8": "#be9bea",
    }
    for light, dark in palette.items():
        svg = svg.replace(light, dark)
    svg = svg.replace(
        "</defs>", '<style type="text/css">text { fill: #e9f8f4; }</style></defs>', 1
    )
    (output / f"{name}.dark.svg").write_text(svg)
    plt.close(figure)


def performance(comparison: dict, output: Path, *, mobile: bool = False) -> dict:
    """Plot only jointly eligible successful pairs; retain omissions explicitly."""
    rows = comparison["trials"]
    if mobile:
        figure, axes = plt.subplots(4, 1, figsize=(3.9, 10.0), constrained_layout=True)
        axes = axes.reshape(2, 2)
    else:
        figure, axes = plt.subplots(2, 2, figsize=(8.1, 6.2), constrained_layout=True)
    retained = []
    for row_index, cohort in enumerate(("existing_primary", "real_primary")):
        selected = [row for row in rows if row["cohort"] == cohort]
        cases = sorted({(row["scenario"], row["variant"]) for row in selected})
        indexed = {
            (row["scenario"], row["variant"], row["condition"]): row for row in selected
        }
        for column, metric in enumerate(
            ("startup_inclusive_seconds", "input_output_tokens")
        ):
            axis = axes[row_index, column]
            for comparator, offset, marker in (
                ("restart", -0.09, "o"),
                ("ipython", 0.09, "D"),
            ):
                audited = next(
                    item
                    for item in comparison["pairwise"]
                    if item["cohort"] == cohort
                    and item["scope"] == "all"
                    and {item["numerator_condition"], item["denominator_condition"]}
                    == {"live", comparator}
                )
                statistics = audited["metrics"][metric]
                allowed = {
                    frozenset(pair): value
                    if audited["numerator_condition"] == "live"
                    else 1 / value
                    for pair, value in zip(
                        statistics["pair_ids"],
                        statistics.get("paired_ratios", []),
                        strict=True,
                    )
                }
                xs, ratios = [], []
                for index, (scenario, variant) in enumerate(cases):
                    live = indexed.get((scenario, variant, "live"), {})
                    other = indexed.get((scenario, variant, comparator), {})
                    pair = frozenset((live.get("id"), other.get("id")))
                    eligible = pair in allowed
                    record = {
                        "cohort": cohort,
                        "scenario": scenario,
                        "variant": variant,
                        "metric": metric,
                        "comparator": comparator,
                        "included": eligible,
                    }
                    if eligible:
                        ratio = allowed[pair]
                        if not math.isclose(
                            ratio, live[metric] / other[metric], rel_tol=1e-12
                        ):
                            raise ValueError(
                                "Audited ratio disagrees with individual recorded costs"
                            )
                        record["newton_over_comparator"] = ratio
                        xs.append(index + offset)
                        ratios.append(ratio)
                    else:
                        reasons = [
                            item["reason"]
                            for item in audited["excluded_pairs"]
                            if item["scenario"] == scenario
                            and item["variant"] == variant
                        ]
                        record["omitted_reason"] = (
                            reasons[0]
                            if reasons
                            else "Missing complete usage or finite positive paired costs"
                        )
                    retained.append(record)
                axis.scatter(
                    xs,
                    ratios,
                    color=COLORS[comparator],
                    marker=marker,
                    s=26,
                    label=f"Newton / {LABELS[comparator]}",
                    zorder=3,
                )
            axis.axhline(1, color="#555555", lw=0.9, ls="--")
            axis.set_yscale("log")
            lower, upper = axis.get_ylim()
            axis.set_yticks([0.3, 0.4, 0.6, 0.8, 1, 1.25, 1.5, 2])
            axis.set_ylim(min(lower, 0.3), max(upper, 2))
            axis.yaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda value, _: f"{value:g}")
            )
            axis.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            axis.set_ylabel(
                "Time ratio" if column == 0 else "Input + output token ratio"
            )
            axis.set_title(
                ("Existing tasks" if row_index == 0 else "Real robot identification")
                + ("\nLower favors Newton" if mobile else " · lower favors Newton")
            )
            names = {
                "panda": "P",
                "allegro": "A",
                "hug": "H",
                "panda_calibration": "C",
                "panda_real": "R",
            }
            axis.set_xticks(
                range(len(cases)),
                [f"{names[name]}{variant}" for name, variant in cases],
            )
            axis.grid(axis="y", color="#dedede", lw=0.5)
            axis.set_axisbelow(True)
            if row_index == 0 and column == 0:
                axis.legend(fontsize=9, loc="best")
    save(figure, output, "comparison-performance" + (".mobile" if mobile else ""))
    return {
        "rows": retained,
        "labels": "P Panda, A Allegro, H HUG, C synthetic calibration, R independent real-data replicate; only jointly eligible successful pairs plotted",
    }


def quality(
    rows: list[dict], initial: dict, output: Path, *, mobile: bool = False
) -> dict:
    """Show pooled and per-recording held-out gates for every real submission."""
    keys = [
        "max_joint_torque_rmse_nm",
        "max_joint_torque_normalized_rmse",
        "max_joint_position_rmse_rad",
        "max_joint_velocity_rmse_rad_s",
        "position_p95_rad",
        "max_joint_speed_rad_s",
    ]
    labels = [
        "Torque\nRMSE",
        "Normalized\ntorque",
        "Position\nRMSE",
        "Velocity\nRMSE",
        "Position\np95",
        "Peak\nspeed",
    ]
    figure, axis = plt.subplots(
        figsize=(3.9, 4.8) if mobile else (8.1, 4.1), constrained_layout=True
    )
    retained = []
    for condition, offset in (("live", -0.22), ("ipython", 0), ("restart", 0.22)):
        selected = sorted(
            [
                row
                for row in rows
                if row["cohort"] == "real_primary" and row["condition"] == condition
            ],
            key=lambda row: row["variant"],
        )
        labeled = False
        for row in selected:
            measured = row.get("quality") or {}
            episodes = measured.get("per_episode", [])
            if not episodes or any(
                key not in measured.get("thresholds", {}) for key in keys
            ):
                retained.append(
                    {
                        "id": row["id"],
                        "omitted": "No complete numerical verifier measurements",
                    }
                )
                continue
            try:
                values = [
                    max(float(item[key]) for item in [measured, *episodes])
                    / measured["thresholds"][key]
                    for key in keys
                ]
            except (KeyError, ValueError, TypeError):
                values = [float("nan")] * len(keys)
            if not np.isfinite(values).all():
                retained.append(
                    {
                        "id": row["id"],
                        "omitted": "Nonfinite or missing verifier measurements",
                    }
                )
                continue
            jitter = (row["variant"] - 4) * 0.012
            axis.scatter(
                np.arange(len(keys)) + offset + jitter,
                values,
                color=COLORS[condition],
                s=19,
                alpha=0.75,
                marker="o" if row.get("eligible_success") else "x",
                label=LABELS[condition] if not labeled else None,
                zorder=3,
            )
            labeled = True
            retained.append(
                {
                    "id": row["id"],
                    "normalized_worst_pooled_or_recording_values": values,
                    "eligible_success": row.get("eligible_success"),
                }
            )
    initial_values = []
    for key in keys:
        measured = [
            item.get(key) for item in [initial, *initial.get("per_episode", [])]
        ]
        complete = bool(measured) and all(
            isinstance(value, (int, float)) and np.isfinite(value) for value in measured
        )
        initial_values.append(
            max(measured) / initial["thresholds"][key] if complete else None
        )
    present = [index for index, value in enumerate(initial_values) if value is not None]
    axis.scatter(
        present,
        [initial_values[index] for index in present],
        marker="x",
        s=55,
        color=COLORS["initial"],
        label=LABELS["initial"],
        zorder=4,
    )
    axis.axhline(1, ls="--", lw=1, color="#333333")
    axis.set_yscale("log")
    axis.set_xticks(range(len(keys)), labels)
    if mobile:
        plt.setp(
            axis.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor"
        )
    axis.set_ylabel("Worst pooled or recording metric / limit")
    axis.set_title(
        "All 27 primary real-data fits\nHeld-out numerical gates"
        if mobile
        else "All 27 primary real-data fits · pooled and per-recording held-out gates"
    )
    axis.grid(axis="y", lw=0.5, color="#dedede")
    axis.legend(ncol=2, fontsize=9)
    save(figure, output, "real-robot-quality" + (".mobile" if mobile else ""))
    return {
        "metrics": keys,
        "initial_values": initial_values,
        "initial_omissions": [
            keys[index] for index, value in enumerate(initial_values) if value is None
        ],
        "rows": retained,
    }


def window(path: Path, episode: int) -> dict:
    with np.load(path, allow_pickle=False) as trace:
        selected = (trace["episode_ids"] == episode) & (trace["window_index"] == 5)
        if np.count_nonzero(selected) != 50:
            raise ValueError(
                f"Expected exactly 50 samples in the fixed sixth window: {path}"
            )
        result = {
            "q": trace["q"][selected].copy(),
            "reference_q": trace["reference_q"][selected].copy(),
            "qd": trace["qd"][selected].copy(),
            "recorded_time": trace["recorded_time"][selected].copy(),
        }
        if any(result[key].shape != (50, 7) for key in ("q", "qd", "reference_q")):
            raise ValueError(
                "Fixed trace does not contain seven joints at all 50 steps"
            )
        if (
            not np.isfinite(result["reference_q"]).all()
            or not np.isfinite(result["recorded_time"]).all()
        ):
            raise ValueError("Measured references contain nonfinite samples")
        if not np.allclose(
            np.diff(result["recorded_time"]), 0.002, rtol=1e-9, atol=1e-12
        ):
            raise ValueError(
                "Fixed trace does not have the registered 2 ms step spacing"
            )
        result["nonfinite_simulated_values"] = int(
            np.count_nonzero(~np.isfinite(result["q"]))
            + np.count_nonzero(~np.isfinite(result["qd"]))
        )
        return result


def serializable(value):
    """Retain nonfinite trace gaps as JSON nulls rather than inventing measurements."""
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serializable(item) for item in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def traces(
    trial_root: Path,
    initial_training: Path,
    initial_heldout: Path,
    output: Path,
    *,
    mobile: bool = False,
) -> dict:
    """Compare prespecified replicate0/window5 for all seven joints without selection by outcome."""
    if mobile:
        figure, axes = plt.subplots(
            14, 1, figsize=(3.9, 19.6), sharex=True, constrained_layout=True
        )
        axes = axes.reshape(2, 7).T
    else:
        figure, axes = plt.subplots(
            7, 2, figsize=(8.1, 10.3), sharex=True, constrained_layout=True
        )
    retained = {}
    styles = {"initial": "-", "live": "-", "ipython": "--", "restart": ":"}
    for column, (split, episode, initial_path) in enumerate(
        (("training", 2, initial_training), ("heldout", 21, initial_heldout))
    ):
        paths = {"initial": initial_path}
        for condition in ("live", "ipython", "restart"):
            base = trial_root / f"panda_real-{condition}-0" / "verification"
            paths[condition] = base / (
                "training/metrics.npz" if split == "training" else "metrics.npz"
            )
        observed = {}
        retained[split] = {}
        missing = []
        for condition, path in paths.items():
            try:
                observed[condition] = window(path, episode)
            except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
                missing.append(LABELS[condition])
                retained[split][condition] = {"omitted": str(error), "path": str(path)}
                continue
            retained[split][condition] = serializable(observed[condition])
        if observed:
            reference = next(iter(observed.values()))["reference_q"]
            for row in observed.values():
                np.testing.assert_array_equal(row["reference_q"], reference)
        for joint in range(7):
            axis = axes[joint, column]
            axis.axhline(
                0, color="#222222", lw=0.7, ls="--", label="Measured reference"
            )
            for condition, row in observed.items():
                error = row["q"][:, joint] - row["reference_q"][:, joint]
                axis.plot(
                    np.arange(1, 51) * 0.002,
                    np.where(np.isfinite(error), error, np.nan),
                    color=COLORS[condition],
                    linestyle=styles[condition],
                    lw=1.25,
                    label=LABELS[condition],
                )
            axis.grid(lw=0.4, color="#dddddd")
            if column == 0 or mobile:
                axis.set_ylabel(f"Joint {joint + 1}\nerror [rad]")
            if joint == 0:
                title = f"{split.title()} recording {episode}"
                if missing:
                    title += "\n" + textwrap.fill(
                        "Unavailable: " + ", ".join(missing), width=42
                    )
                axis.set_title(title)
            if joint == 3 and not observed:
                axis.text(
                    0.5,
                    0.5,
                    "No saved verifier trace\nfor this fixed window",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
            if joint == 6:
                axis.set_xlabel("Time since measured reset [s]")
                axis.tick_params(labelbottom=True)
    for joint in range(7):
        limits = [axis.get_ylim() for axis in axes[joint]]
        low, high = min(x[0] for x in limits), max(x[1] for x in limits)
        for axis in axes[joint]:
            axis.set_ylim(low, high)
    handles = [
        Line2D([], [], color="#222222", lw=0.7, ls="--", label="Measured reference")
    ]
    handles += [
        Line2D(
            [],
            [],
            color=COLORS[condition],
            linestyle=styles[condition],
            label=LABELS[condition],
        )
        for condition in ("initial", "live", "ipython", "restart")
    ]
    figure.legend(
        handles=handles, loc="outside upper center", ncol=2 if mobile else 3, fontsize=9
    )
    save(figure, output, "real-robot-fixed-window" + (".mobile" if mobile else ""))
    return retained


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--trial-root", type=Path, required=True)
    parser.add_argument("--initial-training", type=Path, required=True)
    parser.add_argument("--initial-heldout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "svg.fonttype": "none"})
    result = json.loads(args.comparison.read_text())
    rows = result["trials"]
    evidence = {
        "performance": performance(result, args.output),
        "quality": quality(
            rows,
            json.loads(args.initial_heldout.with_suffix(".json").read_text()),
            args.output,
        ),
        "fixed_windows": traces(
            args.trial_root, args.initial_training, args.initial_heldout, args.output
        ),
    }
    mobile = {
        "performance": performance(result, args.output, mobile=True),
        "quality": quality(
            rows,
            json.loads(args.initial_heldout.with_suffix(".json").read_text()),
            args.output,
            mobile=True,
        ),
        "fixed_windows": traces(
            args.trial_root,
            args.initial_training,
            args.initial_heldout,
            args.output,
            mobile=True,
        ),
    }
    if mobile != evidence:
        raise ValueError(
            "Mobile and desktop figures must show identical numerical evidence"
        )
    (args.output / "comparison-figure-data.json").write_text(
        json.dumps(evidence, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
