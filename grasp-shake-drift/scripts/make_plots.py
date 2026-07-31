# SPDX-License-Identifier: Apache-2.0
"""Build the figures for the grasp-shake drift report.

Run with the repro venv:
    uv run python make_plots.py --data-dir <report>/data --out-dir <report>/media
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK = "#172033"
MUTED = "#5f6b7d"
LINE = "#d8deea"
BLUE = "#2456a6"
TEAL = "#0f766e"
RUST = "#9a3412"
AMBER = "#b45309"


def style(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=LINE, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(INK)


def fig_timeseries(data_dir, out_dir):
    base = json.load(open(os.path.join(data_dir, "baseline_rows.json")))
    rows = [r for r in base if r["phase"] == 4]
    t = np.array([r["t"] for r in rows]) - rows[0]["t"]
    rx = (np.array([r["rel_x"] for r in rows]) - rows[0]["rel_x"]) * 1000
    rz = (np.array([r["rel_z"] for r in rows]) - rows[0]["rel_z"]) * 1000
    slip = np.array([r["slip"] for r in rows]) * 1000

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 3.9), dpi=150)
    style(a1)
    style(a2)

    a1.plot(t, slip, color=BLUE, lw=1.8, label="total slip")
    a1.plot(t, rx, color=TEAL, lw=1.3, ls="--", label="along shake axis (x)")
    a1.plot(t, rz, color=RUST, lw=1.3, ls=":", label="along approach axis (z)")
    a1.set_xlabel("time in SHAKE phase [s]")
    a1.set_ylabel("cube displacement in gripper frame [mm]")
    a1.set_title("Grasp drift accumulates linearly, one step per shake cycle", fontsize=10.5)
    a1.legend(frameon=False, fontsize=8.5, labelcolor=MUTED)

    # per-cycle increments
    n = 60
    cyc = [(rx[i + n] - rx[i]) for i in range(0, len(rx) - n, n)]
    a2.bar(np.arange(len(cyc)) + 1, cyc, color=BLUE, width=0.62)
    a2.set_xlabel("shake cycle (1 Hz)")
    a2.set_ylabel("drift added this cycle [mm]")
    a2.set_title("Every cycle adds the same increment — a ratchet, not a slide", fontsize=10.5)

    fig.tight_layout()
    p = os.path.join(out_dir, "plot_timeseries.png")
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    return p


def fig_variants(data_dir, out_dir):
    v = json.load(open(os.path.join(data_dir, "variants.json")))

    groups = [
        ("Contact source / refresh", [
            ("baseline (Newton contacts @60 Hz)", "baseline"),
            ("Newton contacts @960 Hz", "collide_every_substep"),
            ("Newton contacts @240 Hz", "collide_every_4"),
            ("MuJoCo native contacts", "mujoco_contacts"),
        ]),
        ("Friction", [
            ("finger mu 0.2", "finger_mu_0p2"),
            ("finger mu 0.5", "finger_mu_0p5"),
            ("finger mu 1.0 (baseline)", "baseline"),
            ("finger mu 3", "finger_mu_3"),
            ("finger mu 10", "finger_mu_10"),
            ("cube mu 0.3 (ignored)", "mu_0p3"),
            ("cube mu 20 (ignored)", "mu_20"),
        ]),
        ("Solver effort", [
            ("iterations 15 (baseline)", "baseline"),
            ("iterations 200", "iters_200"),
            ("tolerance 1e-10", "tol_1e10"),
            ("impratio 1000", "impratio_1000"),
        ]),
        ("Contact compliance", [
            ("finger ke 2.5e3 (default)", "baseline"),
            ("finger ke 2.5e4", "finger_ke_2p5e4"),
            ("finger ke 2.5e5", "finger_ke_2p5e5"),
            ("finger ke 2.5e6", "finger_ke_2p5e6"),
            ("no geom priority", "no_priority"),
            ("solimp dmax 0.9999", "solimp_09999"),
        ]),
        ("Load / timestep", [
            ("no shake (amplitude 0)", "amplitude_0"),
            ("substeps 4", "substeps_4"),
            ("substeps 16 (baseline)", "baseline"),
            ("substeps 64", "substeps_64"),
        ]),
    ]

    labels, vals, colors, seps = [], [], [], []
    for gi, (gname, items) in enumerate(groups):
        for lbl, key in items:
            if key not in v:
                continue
            labels.append(lbl)
            vals.append(v[key]["final_slip_mm"])
            colors.append(TEAL if v[key]["final_slip_mm"] < 2.0 else (AMBER if v[key]["final_slip_mm"] < 5.0 else BLUE))
        seps.append((gname, len(labels)))

    fig, ax = plt.subplots(figsize=(9.6, 0.30 * len(labels) + 1.6), dpi=150)
    style(ax)
    y = np.arange(len(labels))
    ax.barh(y, vals, color=colors, height=0.66)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.6)
    ax.invert_yaxis()
    ax.axvline(json.load(open(os.path.join(data_dir, "variants.json")))["baseline"]["final_slip_mm"],
               color=MUTED, lw=1.0, ls="--")
    ax.set_xlabel("cube slip after 11.5 s of shaking [mm]")
    ax.set_title("Only contact compliance moves the needle", fontsize=11, pad=10)
    for yi, val in zip(y, vals):
        ax.text(val + 0.08, yi, f"{val:.2f}", va="center", fontsize=8, color=MUTED)
    ax.set_xlim(0, max(vals) * 1.30)
    prev = 0
    for gname, end in seps:
        if end > prev:
            ax.text(max(vals) * 1.29, (prev + end - 1) / 2.0, gname, ha="right", va="center",
                    fontsize=8.4, color=INK, fontweight="bold", alpha=0.55)
        if end < len(labels):
            ax.axhline(end - 0.5, color=MUTED, lw=0.9, alpha=0.5)
        prev = end
    fig.tight_layout()
    p = os.path.join(out_dir, "plot_variants.png")
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    return p


def fig_stiffness(data_dir, out_dir):
    v = json.load(open(os.path.join(data_dir, "variants.json")))
    ke = [2.5e3, 2.5e4, 2.5e5, 2.5e6]
    keys = ["baseline", "finger_ke_2p5e4", "finger_ke_2p5e5", "finger_ke_2p5e6"]
    slip = [v[k]["final_slip_mm"] for k in keys]

    fig, ax = plt.subplots(figsize=(6.0, 3.8), dpi=150)
    style(ax)
    ax.plot(ke, slip, "o-", color=BLUE, lw=1.9, ms=6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("finger geom contact stiffness ke [N/m]")
    ax.set_ylabel("slip after 11.5 s of shaking [mm]")
    ax.set_title("Drift falls ~1/ke, then saturates once the penetration is gone", fontsize=10.5)
    for x, y, k in zip(ke, slip, keys):
        ax.annotate(f"{y:.2f} mm", (x, y), textcoords="offset points", xytext=(6, 7),
                    fontsize=8.4, color=MUTED)
    ax.axhline(v["amplitude_0"]["final_slip_mm"], color=TEAL, ls="--", lw=1.2)
    ax.text(3.2e3, v["amplitude_0"]["final_slip_mm"] * 1.12,
            "stationary-hold noise floor", fontsize=8.2, color=TEAL)
    fig.tight_layout()
    p = os.path.join(out_dir, "plot_stiffness.png")
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    return p


def fig_severity(data_dir, out_dir):
    b = json.load(open(os.path.join(data_dir, "sweep_baseline.json")))
    f = json.load(open(os.path.join(data_dir, "sweep_fixed.json")))

    def split(rows):
        fr = sorted([r for r in rows if abs(r["amplitude"] - 0.03) < 1e-9], key=lambda r: r["frequency"])
        am = sorted([r for r in rows if abs(r["frequency"] - 1.0) < 1e-9], key=lambda r: r["amplitude"])
        return fr, am

    bf, ba = split(b)
    ff, fa = split(f)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 3.9), dpi=150)
    style(a1)
    style(a2)

    a1.plot([r["frequency"] for r in bf], [r["final_slip_mm"] for r in bf], "o-",
            color=BLUE, lw=1.9, ms=6, label="as shipped (ke 2.5e3)")
    a1.plot([r["frequency"] for r in ff], [r["final_slip_mm"] for r in ff], "s-",
            color=TEAL, lw=1.9, ms=6, label="stiffened pads (ke 2.5e5)")
    a1.set_xlabel("shake frequency [Hz]   (amplitude 3 cm)")
    a1.set_ylabel("slip after ~8 s of shaking [mm]")
    a1.set_title("Drift vs shake frequency", fontsize=10.5)
    a1.legend(frameon=False, fontsize=8.6, labelcolor=MUTED)

    a2.plot([r["amplitude"] * 100 for r in ba], [r["final_slip_mm"] for r in ba], "o-",
            color=BLUE, lw=1.9, ms=6, label="as shipped (ke 2.5e3)")
    a2.plot([r["amplitude"] * 100 for r in fa], [r["final_slip_mm"] for r in fa], "s-",
            color=TEAL, lw=1.9, ms=6, label="stiffened pads (ke 2.5e5)")
    a2.set_yscale("log")
    a2.set_xlabel("shake amplitude [cm]   (1 Hz)")
    a2.set_ylabel("slip after ~8 s of shaking [mm], log scale")
    a2.set_title("Drift vs shake amplitude", fontsize=10.5)
    a2.legend(frameon=False, fontsize=8.6, labelcolor=MUTED, loc="upper left")
    drop = [r for r in ba if abs(r["amplitude"] - 0.10) < 1e-9]
    if drop:
        a2.annotate("cube dropped", (10, drop[0]["final_slip_mm"]),
                    textcoords="offset points", xytext=(-14, -22), fontsize=8.6,
                    color=RUST, fontweight="bold", ha="right")

    fig.tight_layout()
    p = os.path.join(out_dir, "plot_severity.png")
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    for fn in (fig_timeseries, fig_variants, fig_stiffness, fig_severity):
        try:
            print("wrote", fn(a.data_dir, a.out_dir))
        except FileNotFoundError as e:
            print("skipped", fn.__name__, e)
