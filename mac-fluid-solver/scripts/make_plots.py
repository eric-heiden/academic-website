"""Generate report plots from the example metrics and benchmark JSON files.

Run from the newton worktree:
    uv run --extra examples python <this script> --report-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.autolayout": True,
    }
)

COLORS = {"wet": "#1f77b4", "dry": "#7f7f7f", "rise": "#2ca02c", "rev": "#d62728", "extra": "#ff7f0e"}


def load(data_dir, name):
    path = os.path.join(data_dir, f"{name}.json")
    with open(path) as f:
        return json.load(f)


def series(frames, key, idx=None):
    if idx is None:
        return np.array([f[key] for f in frames])
    return np.array([f[key][idx] for f in frames])


def plot_sphere(data_dir, media_dir):
    settle = load(data_dir, "sphere_settling")["frames"]
    rise = load(data_dir, "sphere_rising")["frames"]
    dry = load(data_dir, "sphere_dry")["frames"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    t_s, t_r, t_d = (series(f, "time") for f in (settle, rise, dry))

    ax = axes[0, 0]
    ax.plot(t_s, series(settle, "position", 2), color=COLORS["wet"], label="settling (ρ=1500)")
    ax.plot(t_r, series(rise, "position", 2), color=COLORS["rise"], label="rising (ρ=500)")
    ax.plot(t_d, series(dry, "position", 2), color=COLORS["dry"], ls="--", label="dry (no fluid)")
    ax.set_xlabel("time [s]"), ax.set_ylabel("sphere height z [m]"), ax.set_title("Position")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(t_s, series(settle, "velocity", 2), color=COLORS["wet"], label="settling")
    ax.plot(t_r, series(rise, "velocity", 2), color=COLORS["rise"], label="rising")
    ax.plot(t_d, series(dry, "velocity", 2), color=COLORS["dry"], ls="--", label="dry")
    ax.set_xlabel("time [s]"), ax.set_ylabel("vertical velocity [m/s]"), ax.set_title("Velocity")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(t_s, series(settle, "fluid_force", 2), color=COLORS["wet"], label="settling")
    ax.plot(t_r, series(rise, "fluid_force", 2), color=COLORS["rise"], label="rising")
    buoyancy = 1000.0 * 9.81 * 4.0 / 3.0 * np.pi * 0.12**3
    ax.axhline(buoyancy, color="k", ls=":", lw=1, label=f"analytic buoyancy {buoyancy:.0f} N")
    ax.set_xlabel("time [s]"), ax.set_ylabel("fluid force Fz [N]"), ax.set_title("Hydrodynamic force")
    ax.set_ylim(-150, 300)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.semilogy(t_s, series(settle, "div_l2_pre"), color=COLORS["extra"], label="div RMS before projection")
    ax.semilogy(t_s, series(settle, "div_l2_post"), color=COLORS["wet"], label="div RMS after projection")
    mbe = np.abs(np.array([f["momentum_balance_error"] for f in settle])).max(axis=1)
    ax.semilogy(t_s, np.maximum(mbe, 1e-12), color=COLORS["rev"], label="action–reaction error [N·s]")
    ax.set_xlabel("time [s]"), ax.set_title("Divergence and coupling error (settling)")
    ax.legend(fontsize=8)

    fig.suptitle("Settling / rising sphere in a closed water tank", fontsize=12)
    fig.savefig(os.path.join(media_dir, "plot_sphere.png"))
    plt.close(fig)


def plot_paddle(data_dir, media_dir):
    wet = load(data_dir, "paddle_wet")
    dry = load(data_dir, "paddle_dry")
    target = wet["meta"]["omega_target"]
    wf, df = wet["frames"], dry["frames"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    ax = axes[0]
    ax.plot(series(wf, "time"), series(wf, "joint_velocity"), color=COLORS["wet"], label="wet (coupled)")
    ax.plot(series(df, "time"), series(df, "joint_velocity"), color=COLORS["dry"], ls="--", label="dry")
    ax.axhline(target, color="k", ls=":", lw=1, label=f"target {target:g} rad/s")
    ax.set_xlabel("time [s]"), ax.set_ylabel("paddle speed [rad/s]"), ax.set_title("Actuator speed under load")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(series(wf, "time"), series(wf, "fluid_torque_z"), color=COLORS["rev"], lw=0.8)
    ax.set_xlabel("time [s]"), ax.set_ylabel("fluid torque τz [N·m]"), ax.set_title("Reaction torque")

    ax = axes[2]
    ax.semilogy(series(wf, "time"), series(wf, "div_linf_post"), color=COLORS["wet"], label="div L∞ after projection")
    mbe = np.abs(np.array([f["momentum_balance_error"] for f in wf])).max(axis=1)
    ax.semilogy(series(wf, "time"), np.maximum(mbe, 1e-12), color=COLORS["rev"], label="action–reaction error [N·s]")
    ax.set_xlabel("time [s]"), ax.set_title("Fluid diagnostics")
    ax.legend(fontsize=8)

    fig.suptitle("Motor-driven paddle: dry vs. coupled", fontsize=12)
    fig.savefig(os.path.join(media_dir, "plot_paddle.png"))
    plt.close(fig)


def plot_swimmer(data_dir, media_dir):
    fwd = load(data_dir, "swimmer_forward")["frames"]
    rev = load(data_dir, "swimmer_reverse")["frames"]
    dry = load(data_dir, "swimmer_dry")["frames"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    ax = axes[0]
    for frames, key, label in ((fwd, "wet", "forward wave"), (rev, "rev", "reversed wave"), (dry, "dry", "dry")):
        x = series(frames, "com", 0) - frames[0]["com"][0]
        ax.plot(series(frames, "time"), x, color=COLORS[key], ls="--" if key == "dry" else "-", label=label)
    ax.set_xlabel("time [s]"), ax.set_ylabel("COM displacement x [m]"), ax.set_title("Propulsion")
    ax.legend(fontsize=8)

    ax = axes[1]
    for frames, key, label in ((fwd, "wet", "forward"), (rev, "rev", "reversed")):
        v = series(frames, "com_velocity", 0)
        ax.plot(series(frames, "time"), v, color=COLORS[key], lw=0.8, label=label)
    ax.set_xlabel("time [s]"), ax.set_ylabel("COM velocity x [m/s]"), ax.set_title("Swimming speed")
    ax.legend(fontsize=8)

    ax = axes[2]
    lf = np.array([f["link_forces"] for f in fwd])  # (T, links, 3)
    t = series(fwd, "time")
    for link in range(lf.shape[1]):
        ax.plot(t, np.linalg.norm(lf[:, link], axis=1), lw=0.7, label=f"link {link}")
    ax.set_xlabel("time [s]"), ax.set_ylabel("|force| [N]"), ax.set_title("Per-link hydrodynamic force (forward)")
    ax.legend(fontsize=7, ncol=2)

    fig.suptitle("Five-link swimmer: forward, reversed, and dry gaits", fontsize=12)
    fig.savefig(os.path.join(media_dir, "plot_swimmer.png"))
    plt.close(fig)


def plot_benchmarks(data_dir, media_dir):
    with open(os.path.join(data_dir, "benchmarks.json")) as f:
        results = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    for capture, label, color in ((False, "GPU (eager)", COLORS["extra"]), (True, "GPU (CUDA graph)", COLORS["wet"])):
        pts = [(r["cells"], r["ms_per_step"]) for r in results if r["device"].startswith("cuda") and r["capture"] == capture and "stage_ms" not in r]
        pts.sort()
        ax.loglog([p[0] for p in pts], [p[1] for p in pts], "o-", color=color, label=label)
    cpu = [r for r in results if r["device"] == "cpu"]
    if cpu:
        ax.loglog([cpu[0]["cells"]], [cpu[0]["ms_per_step"]], "s", color=COLORS["dry"], label="CPU (32³)")
    for r in results:
        if r["device"].startswith("cuda") and r["capture"] and "stage_ms" not in r:
            ax.annotate(f"{r['resolution']}³", (r["cells"], r["ms_per_step"]), textcoords="offset points", xytext=(6, -10), fontsize=8)
    ax.set_xlabel("grid cells"), ax.set_ylabel("ms / step"), ax.set_title("Step time vs. resolution (NVIDIA L40)")
    ax.legend(fontsize=8)

    ax = axes[1]
    staged = [r for r in results if "stage_ms" in r]
    if staged:
        stages = staged[0]["stage_ms"]
        names = list(stages.keys())
        vals = [stages[n] for n in names]
        ax.barh(names[::-1], vals[::-1], color=COLORS["wet"])
        ax.set_xlabel("ms / step"), ax.set_title(f"Stage timings at {staged[0]['resolution']}³ (synchronizing timers)")
    fig.savefig(os.path.join(media_dir, "plot_benchmarks.png"))
    plt.close(fig)


def summarize(data_dir):
    """Emit a summary JSON with headline numbers for the report text."""
    out = {}
    settle = load(data_dir, "sphere_settling")["frames"]
    rise = load(data_dir, "sphere_rising")["frames"]
    out["sphere"] = {
        "settle_final_z": settle[-1]["position"][2],
        "settle_peak_speed": float(np.abs(series(settle, "velocity", 2)).max()),
        "rise_final_z": rise[-1]["position"][2],
        "steady_buoyancy_N": float(np.mean(series(settle, "fluid_force", 2)[-60:])),
        "analytic_buoyancy_N": 1000.0 * 9.81 * 4.0 / 3.0 * np.pi * 0.12**3,
        "max_action_reaction_error": float(
            np.abs(np.array([f["momentum_balance_error"] for f in settle])).max()
        ),
        "div_l2_post_final": settle[-1]["div_l2_post"],
    }
    wet = load(data_dir, "paddle_wet")
    dry = load(data_dir, "paddle_dry")
    out["paddle"] = {
        "omega_target": wet["meta"]["omega_target"],
        "omega_wet": float(np.mean(series(wet["frames"], "joint_velocity")[-60:])),
        "omega_dry": float(np.mean(series(dry["frames"], "joint_velocity")[-60:])),
        "tau_fluid": float(np.mean(series(wet["frames"], "fluid_torque_z")[-60:])),
    }
    fwd = load(data_dir, "swimmer_forward")["frames"]
    rev = load(data_dir, "swimmer_reverse")["frames"]
    sdry = load(data_dir, "swimmer_dry")["frames"]
    out["swimmer"] = {
        "duration": fwd[-1]["time"],
        "dx_forward": fwd[-1]["com"][0] - fwd[0]["com"][0],
        "dx_reverse": rev[-1]["com"][0] - rev[0]["com"][0],
        "dx_dry": sdry[-1]["com"][0] - sdry[0]["com"][0],
        "cruise_speed": float(np.mean(series(fwd, "com_velocity", 0)[-120:])),
    }
    with open(os.path.join(data_dir, "summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=str, required=True)
    args = parser.parse_args()
    data_dir = os.path.join(args.report_dir, "data")
    media_dir = os.path.join(args.report_dir, "media")

    plot_sphere(data_dir, media_dir)
    plot_paddle(data_dir, media_dir)
    plot_swimmer(data_dir, media_dir)
    plot_benchmarks(data_dir, media_dir)
    summarize(data_dir)
    print("plots written to", media_dir)


if __name__ == "__main__":
    main()
