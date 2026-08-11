"""Build the report's interactive figures and the tables that go with them."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

DATA = Path("/home/horde/repos/fpgs-study/data")
REPORT = Path("/home/horde/repos/academic-website-reports/featherpgs-issues")

# Validated categorical palette, dark surface #101e2c.
C = dict(blue="#3987e5", orange="#d95926", aqua="#199e70", yellow="#c98500",
         magenta="#d55181", violet="#9085e9", red="#e66767")
INK = "#e9f1f8"
MUTED = "#93a6b8"
GRID = "rgba(147,166,184,0.15)"


def ax(title, **kw):
    a = dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID,
             tickfont=dict(color=MUTED, size=12),
             title=dict(text=title, font=dict(color=MUTED, size=12)))
    a.update(kw)
    return a


def layout(xtitle, ytitle, height=380, **kw):
    lay = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="ui-sans-serif, system-ui, 'Segoe UI', sans-serif",
                  size=13, color=INK),
        margin=dict(l=68, r=22, t=34, b=54), height=height,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.16, x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=12, color=MUTED)),
        xaxis=ax(xtitle), yaxis=ax(ytitle),
    )
    for k, v in kw.items():
        if k in ("xaxis", "yaxis"):
            lay[k].update(v)
        else:
            lay[k] = v
    return lay


def load(name):
    p = DATA / f"{name}.npz"
    return dict(np.load(p, allow_pickle=True)) if p.exists() else None


def shake_drift(res):
    m = res["phase"] == 4
    if not m.any():
        return None, None
    rl = res["rel_pos_local"][m]
    t = res["t"][m]
    return t - t[0], (rl - rl[0]) * 1000.0


FIGS = {}
TABLES = {}


def bar(x, y, colors, text, hover):
    return dict(type="bar", x=x, y=y, showlegend=False,
                marker=dict(color=colors, line=dict(width=0)),
                text=text, textposition="outside",
                textfont=dict(color=INK, size=12), hovertemplate=hover)


def f_droop(track):
    if not track:
        return
    xs = ["1x gains", "10x gains", "100x gains", "1x, gravity off", "SolverMuJoCo, 1x"]
    keys = ["fpgs_g_1x", "fpgs_g_10x", "fpgs_g_100x", "fpgs_nog_1x", "mujoco_g"]
    ys = [max(track[k]["final_err_max_deg"], 1e-4) for k in keys if k in track]
    cols = [C["orange"], C["orange"], C["orange"], C["aqua"], C["blue"]]
    labels = []
    for k in keys:
        v = track[k]["final_err_max_deg"]
        labels.append(f"{v:.3f}°" if v > 0 else "0°")
    FIGS["fig-droop"] = dict(
        data=[bar(xs, ys, cols, labels,
                  "%{x}<br>%{y:.4f}° of sag<extra></extra>")],
        layout=layout("", "Worst arm-joint sag, degrees (log)",
                      yaxis=dict(type="log"), hovermode="closest",
                      margin=dict(l=68, r=22, t=22, b=54)))


def f_traces():
    rows = [("mujoco_base", "SolverMuJoCo, original gains", C["blue"]),
            ("fpgs_gc_default", "FeatherPGS + gravity term, original gains", C["aqua"]),
            ("fpgs_100x", "FeatherPGS, 100x gains", C["orange"])]
    data = []
    for name, label, col in rows:
        r = load(name)
        if r is None:
            continue
        t, d = shake_drift(r)
        if t is None:
            continue
        data.append(dict(type="scatter", mode="lines", name=label,
                         x=np.round(t, 4).tolist(),
                         y=np.round(np.linalg.norm(d, axis=1), 4).tolist(),
                         line=dict(color=col, width=2),
                         hovertemplate="%{y:.2f} mm<extra></extra>"))
    if data:
        FIGS["fig-traces"] = dict(
            data=data,
            layout=layout("Time into the shake (s)",
                          "Cube movement inside the gripper, mm (log)",
                          yaxis=dict(type="log"), height=400))


def f_configs(cfg, misc):
    order = [("fpgs_gc_default", "defaults", None),
             ("fpgs_gc_sharedanchor", "shared friction anchor", None),
             ("fpgs_gc_it32", "32 solver iterations", None),
             ("fpgs_gc_velit4", "4 velocity iterations", None),
             ("fpgs_gc_best", "anchor + 4 vel. it. + 20 it.", None)]
    fm = misc.get("friction_modes", {})
    for key, lbl in (("bisection", "friction: bisection"),
                     ("bisection_desaxce", "friction: bisection + de Saxce"),
                     ("coulomb_newton", "friction: Coulomb Newton")):
        if key in fm and "drift_max_mm" in fm[key]:
            order.append((("__fm__" + key), lbl, fm[key]["drift_max_mm"]))
    xs, ys, cols, txt = [], [], [], []
    for k, lbl, direct in order:
        v = direct if direct is not None else (creep(k) if held(k) else None)
        if v is None:
            continue
        xs.append(lbl)
        ys.append(round(v, 3))
        cols.append(C["aqua"] if v < 0.9 else C["yellow"] if v < 1.5 else C["orange"])
        txt.append(f"{v:.2f} mm")
    if xs:
        FIGS["fig-configs"] = dict(
            data=[bar(xs, ys, cols, txt, "%{x}<br>%{y:.2f} mm peak creep<extra></extra>")],
            layout=layout("", "Peak creep during the shake (mm)",
                          hovermode="closest", xaxis=dict(tickangle=-18),
                          margin=dict(l=68, r=22, t=22, b=112), height=420))


def f_substeps(sub):
    ss = [16, 8, 4, 2, 1]
    dt = [round(1000.0 / 60.0 / s, 3) for s in ss]
    drift, speed, rows = [], [], []
    for pre, label, col in (("fpgs", "FeatherPGS", C["aqua"]),
                            ("mujoco", "SolverMuJoCo", C["blue"])):
        y = []
        for s in ss:
            n = f"{pre}_ss{s}"
            y.append(round(creep(n), 3) if held(n) else None)
        drift.append(dict(type="scatter", mode="lines+markers", name=label,
                          x=dt, y=y, connectgaps=False,
                          line=dict(color=col, width=2), marker=dict(size=9, color=col),
                          hovertemplate="dt %{x} ms<br>%{y:.2f} mm<extra></extra>"))
        speed.append(dict(type="scatter", mode="lines+markers", name=label, x=dt,
                          y=[round(sub.get(f"{pre}_ss{s}", {}).get("physics_rtf", 0), 2)
                             for s in ss],
                          line=dict(color=col, width=2), marker=dict(size=9, color=col),
                          hovertemplate="dt %{x} ms<br>%{y:.1f}x realtime<extra></extra>"))
    speed.append(dict(type="scatter", mode="lines", name="realtime", x=dt,
                      y=[1.0] * len(dt), line=dict(color=MUTED, width=1, dash="dot"),
                      hoverinfo="skip"))
    tick = dict(type="log", tickmode="array", tickvals=dt,
                ticktext=[f"{d:g}" for d in dt])
    FIGS["fig-substeps-drift"] = dict(
        data=drift, layout=layout("Physics time step, ms (log)",
                                  "Peak creep, mm (log)",
                                  xaxis=tick, yaxis=dict(type="log"), height=360))
    FIGS["fig-substeps-speed"] = dict(
        data=speed, layout=layout("Physics time step, ms (log)",
                                  "Speed vs realtime, x (log)",
                                  xaxis=tick, yaxis=dict(type="log"), height=360))

    for s, d in zip(ss, dt):
        f, m = sub.get(f"fpgs_ss{s}", {}), sub.get(f"mujoco_ss{s}", {})

        rows.append([f"{d:g} ms <span style='color:#6f8298'>({s} substeps)</span>",
                     outcome_cell(f"fpgs_ss{s}"), outcome_cell(f"mujoco_ss{s}"),
                     f"{f.get('ms_per_frame', 0):.2f} ms",
                     f"{m.get('ms_per_frame', 0):.2f} ms"])
    TABLES["tbl-substeps"] = rows


def f_budget(budget, wm):
    caps = [32, 48, 64, 96, 128, 256]
    xs, ys, cols, txt = [], [], [], []
    for c in caps:
        name = f"cap{c}"
        if name not in RESCORE:
            continue
        xs.append(f"{c}" + (" (default)" if c == 32 else ""))
        if held(name):
            d = creep(name)
            ys.append(round(d, 3))
            cols.append(C["aqua"])
            txt.append(f"{d:.2f} mm")
        else:
            ys.append(35.0)          # off-scale marker for a failed grasp
            cols.append(C["red"])
            txt.append("never lifted")
    if not xs:
        return
    need = wm.get("dense_high_water")
    ann = []
    if need:
        ann.append(dict(x=0.02, y=1.02, xref="paper", yref="paper", showarrow=False,
                        xanchor="left",
                        text=f"this scene peaks at {need} rows",
                        font=dict(color=C["yellow"], size=12.5)))
    FIGS["fig-budget"] = dict(
        data=[bar(xs, ys, cols, txt,
                  "%{x} rows<extra></extra>")],
        layout=layout("dense_max_constraints (rows per world)",
                      "Peak creep during the shake (mm)",
                      hovermode="closest", annotations=ann,
                      yaxis=dict(range=[0, 42]),
                      margin=dict(l=68, r=22, t=34, b=54)))


def f_grip():
    """Finger-to-cube contact count through the shake."""
    rows = [("mujoco", "SolverMuJoCo", C["blue"]),
            ("fpgs_gc", "FeatherPGS + gravity term", C["aqua"]),
            ("fpgs_100x", "FeatherPGS, 100x gains", C["orange"])]
    data = []
    for t, label, col in rows:
        r = load(f"grip_{t}")
        if r is None:
            continue
        m = r["phase"] == 4
        if not m.any():
            continue
        data.append(dict(type="scatter", mode="lines", name=label,
                         x=np.round((r["frame"][m] - r["frame"][m][0]) / 60.0, 3).tolist(),
                         y=r["pad_n"][m].tolist(),
                         line=dict(color=col, width=2), fill="tozeroy",
                         fillcolor=col.replace("#", "rgba(") if False else None,
                         opacity=0.9,
                         hovertemplate="%{y} contacts<extra></extra>"))
    if data:
        for d in data:
            d.pop("fillcolor", None)
            d.pop("fill", None)
        FIGS["fig-grip"] = dict(
            data=data, layout=layout("Time into the shake (s)",
                                     "Finger-to-cube contact points",
                                     height=340))


def f_inner(inner, sub, iters):
    """What each knob buys per millisecond of solve time."""
    def series(name, pts, colour, symbol):
        xs = [round(p[0], 2) for p in pts]
        ys = [round(p[1], 3) for p in pts]
        ts = [p[2] for p in pts]
        return dict(type="scatter", mode="markers+text", name=name,
                    x=xs, y=ys, text=ts, textposition="top center",
                    textfont=dict(color=MUTED, size=11),
                    marker=dict(size=12, color=colour, symbol=symbol),
                    hovertemplate="%{text}<br>%{x} ms/frame<br>%{y:.2f} mm<extra></extra>")

    sub_pts = []
    for s in (16, 8, 4, 2):
        n, v = f"fpgs_ss{s}", sub.get(f"fpgs_ss{s}", {})
        if held(n) and v.get("ms_per_frame"):
            sub_pts.append((v["ms_per_frame"], creep(n), f"{s} substeps"))

    it_pts = []
    for k, v in sorted(iters.items(), key=lambda kv: kv[1].get("pgs_iterations", 0)):
        if not v.get("ms_per_frame") or not v.get("reached_shake") or v.get("dropped"):
            continue
        it_pts.append((v["ms_per_frame"], v["drift_max_mm"],
                       f"{v['pgs_iterations']} it."))

    fro_pts, fail = [], []
    for k, v in inner.items():
        if "error" in v or not v.get("ms_per_frame") or v["inner"] == 1:
            continue
        lbl = f"{v['outer']}x{v['inner']}"
        if v.get("reached_shake") and not v.get("dropped"):
            fro_pts.append((v["ms_per_frame"], v["drift_max_mm"], lbl))
        else:
            fail.append((v["ms_per_frame"], 30.0, lbl))

    data = []
    if sub_pts:
        data.append(series("substep count (12 iterations)", sub_pts, C["aqua"], "circle"))
    if it_pts:
        data.append(series("iteration count (16 substeps)", it_pts, C["blue"], "square"))
    if fro_pts:
        data.append(series("frozen inner substeps", fro_pts, C["violet"], "diamond"))
    if fail:
        data.append(series("frozen, no usable grasp", fail, C["red"], "x"))
    if not data:
        return
    FIGS["fig-inner"] = dict(
        data=data,
        layout=layout("Wall time per control frame, ms (log)",
                      "Peak creep during the shake, mm (log)",
                      xaxis=dict(type="log"), yaxis=dict(type="log"),
                      hovermode="closest", height=430))

    rows = []
    for k in sorted(inner, key=lambda x: (inner[x]["outer"], inner[x]["inner"])):
        v = inner[k]
        if "error" in v:
            continue
        res = ("<span style='color:#ff6b7a'>dropped</span>" if v.get("dropped") else
               "<span style='color:#ff6b7a'>no grasp</span>" if not v.get("reached_shake")
               else "%.2f mm" % v["drift_max_mm"])
        rows.append(["%d &times; %d" % (v["outer"], v["inner"]),
                     "%.2f ms" % v["effective_dt_ms"], res,
                     "%.2f ms" % v["ms_per_frame"], "%.1fx" % v["physics_rtf"]])
    if rows:
        TABLES["tbl-inner"] = rows

    irows = []
    for k, v in sorted(iters.items(), key=lambda kv: kv[1].get("pgs_iterations", 0)):
        if "pgs_iterations" not in v:
            continue
        res = ("<span style='color:#ff6b7a'>no grasp</span>"
               if not v.get("reached_shake") or v.get("dropped")
               else "%.2f mm" % v["drift_max_mm"])
        irows.append([str(v["pgs_iterations"]), res, "%.2f ms" % v["ms_per_frame"],
                      "%.1fx" % v["physics_rtf"]])
    if irows:
        TABLES["tbl-iters"] = irows


def f_tuning(t, mus_src=None):
    if not t:
        return
    betas = sorted([v for k, v in t.items() if k.startswith("beta")],
                   key=lambda v: v["pgs_beta"])
    cfms = sorted([v for k, v in t.items() if k.startswith("cfm")],
                  key=lambda v: v["pgs_cfm"])
    data = []
    if betas:
        data.append(dict(type="bar", name="position-correction factor (pgs_beta)",
                         x=[f"{v['pgs_beta']:g}" for v in betas],
                         y=[round(v["drift_max_mm"], 3) if v["held"] else None
                            for v in betas],
                         marker=dict(color=C["aqua"], line=dict(width=0)),
                         text=[f"{v['drift_max_mm']:.2f}" if v["held"] else "lost"
                               for v in betas],
                         textposition="outside", textfont=dict(color=INK, size=11),
                         hovertemplate="beta %{x}<br>%{y:.2f} mm<extra></extra>"))
    if cfms:
        data.append(dict(type="bar", name="diagonal regularisation (pgs_cfm)",
                         x=[("0" if v["pgs_cfm"] == 0 else f"{v['pgs_cfm']:g}")
                            for v in cfms],
                         y=[round(v["drift_max_mm"], 3) if v["held"] else None
                            for v in cfms],
                         marker=dict(color=C["blue"], line=dict(width=0)),
                         text=[f"{v['drift_max_mm']:.2f}" if v["held"] else "lost"
                               for v in cfms],
                         textposition="outside", textfont=dict(color=INK, size=11),
                         hovertemplate="cfm %{x}<br>%{y:.2f} mm<extra></extra>"))
    if data:
        FIGS["fig-tuning"] = dict(
            data=data, layout=layout("parameter value", "Peak creep (mm)",
                                     barmode="group", hovermode="closest",
                                     height=340))

    src = mus_src
    if not src:
        return
    mus = sorted({round(v["mu_cube"], 3) for k, v in src.items() if k.startswith("mu")})
    if mus:
        mdata = []
        for solv, label, col in (("fpgs", "FeatherPGS", C["aqua"]),
                                 ("mujoco", "SolverMuJoCo", C["blue"])):
            ys = []
            for mu in mus:
                v = next((x for k, x in src.items()
                          if k.endswith("_" + solv) and abs(x.get("mu_cube", -1) - mu) < 1e-3), None)
                ys.append(round(v["drift_max_mm"], 3)
                          if v and v.get("held") else None)
            mdata.append(dict(type="scatter", mode="lines+markers", name=label,
                              x=mus, y=ys, connectgaps=False,
                              line=dict(color=col, width=2),
                              marker=dict(size=10, color=col),
                              hovertemplate="mu %{x}<br>%{y:.2f} mm<extra></extra>"))
        FIGS["fig-mu"] = dict(
            data=mdata,
            layout=layout("Cube friction coefficient",
                          "Peak creep during the shake, mm (log)",
                          yaxis=dict(type="log"), height=340))


def _coupon_series(grid, xkey, label_fmt):
    """colour = solver, dash = geometry; missing point = object lost."""
    out = []
    for solv, sname, col in (("fpgs", "FeatherPGS", C["aqua"]),
                             ("mujoco", "SolverMuJoCo", C["blue"])):
        for geom, dash, sym in (("box", "solid", "circle"),
                                ("mesh", "dash", "diamond")):
            pts = sorted([v for v in grid.values()
                          if v.get("solver") == solv and v.get("geom") == geom
                          and xkey in v], key=lambda v: v[xkey])
            if not pts:
                continue
            xs = [round(p[xkey], 4) for p in pts]
            # downward slip is the grasp-relevant number; floor at 1 um for the log axis
            ys = [max(round(p["axial_slip_mm"], 5), 1.0e-3) if p.get("held") else None
                  for p in pts]
            out.append(dict(type="scatter", mode="lines+markers",
                            name=f"{sname}, {geom}", x=xs, y=ys, connectgaps=False,
                            line=dict(color=col, width=2, dash=dash),
                            marker=dict(size=9, color=col, symbol=sym),
                            hovertemplate=label_fmt + "<br>%{y:.3f} mm<extra></extra>"))
    return out


def f_coupon(c):
    st, dr, kn = c.get("static", {}), c.get("driven", {}), c.get("knobs", {})
    if st:
        FIGS["fig-coupon-static"] = dict(
            data=_coupon_series(st, "overlap_mm", "overlap %{x} mm"),
            layout=layout("Geometric overlap per side (mm)",
                          "Downward slip through the grasp, mm (log)",
                          yaxis=dict(type="log"), height=360))
        rows = []
        for ov in sorted({v["overlap_mm"] for v in st.values() if "overlap_mm" in v}):
            r = [f"{ov:g} mm"]
            for solv in ("fpgs", "mujoco"):
                for geom in ("box", "mesh"):
                    v = next((x for x in st.values()
                              if x.get("solver") == solv and x.get("geom") == geom
                              and x.get("overlap_mm") == ov), None)
                    if not v:
                        r.append("&mdash;")
                    elif not v.get("held"):
                        r.append("<span style='color:#ff6b7a'>ejected</span>")
                    else:
                        r.append("%.2f mm" % v["axial_slip_mm"])
            rows.append(r)
        TABLES["tbl-coupon-static"] = rows
    if dr:
        FIGS["fig-coupon-driven"] = dict(
            data=_coupon_series(dr, "squeeze_mm", "squeeze %{x} mm"),
            layout=layout("Commanded squeeze past the surface (mm)",
                          "Downward slip through the grasp, mm (log)",
                          yaxis=dict(type="log"), height=360))
        rows = []
        for sq in sorted({v["squeeze_mm"] for v in dr.values() if "squeeze_mm" in v}):
            r = [f"{sq:g} mm"]
            for solv in ("fpgs", "mujoco"):
                for geom in ("box", "mesh"):
                    v = next((x for x in dr.values()
                              if x.get("solver") == solv and x.get("geom") == geom
                              and x.get("squeeze_mm") == sq), None)
                    if not v or not v.get("held"):
                        r.append("<span style='color:#ff6b7a'>lost</span>")
                    else:
                        r.append("%.2f / %.2f" % (v["axial_slip_mm"], v["lateral_mm"]))
            rows.append(r)
        TABLES["tbl-coupon-driven"] = rows
    if kn:
        its = sorted([v for v in kn.values() if "pgs_iterations" in v],
                     key=lambda v: v["pgs_iterations"])
        mus = {}
        for v in kn.values():
            if "mu" in v:
                mus.setdefault(v["solver"], []).append(v)
        data = []
        if its:
            data.append(dict(type="scatter", mode="lines+markers",
                             name="FeatherPGS", x=[v["pgs_iterations"] for v in its],
                             y=[max(round(v["axial_slip_mm"], 5), 1.0e-3)
                                if v.get("held") else None
                                for v in its], connectgaps=False,
                             line=dict(color=C["aqua"], width=2),
                             marker=dict(size=9, color=C["aqua"]),
                             hovertemplate="%{x} iterations<br>%{y:.3f} mm<extra></extra>"))
            FIGS["fig-coupon-iters"] = dict(
                data=data, layout=layout("Solver iterations",
                                         "Downward slip through the grasp, mm (log)",
                                         yaxis=dict(type="log"), height=320))
        if mus:
            md = []
            for solv, sname, col in (("fpgs", "FeatherPGS", C["aqua"]),
                                     ("mujoco", "SolverMuJoCo", C["blue"])):
                pts = sorted(mus.get(solv, []), key=lambda v: v["mu"])
                if not pts:
                    continue
                md.append(dict(type="scatter", mode="lines+markers", name=sname,
                               x=[v["mu"] for v in pts],
                               y=[max(round(v["axial_slip_mm"], 5), 1.0e-3)
                                  if v.get("held") else None
                                  for v in pts], connectgaps=False,
                               line=dict(color=col, width=2),
                               marker=dict(size=9, color=col),
                               hovertemplate="mu %{x}<br>%{y:.3f} mm<extra></extra>"))
            FIGS["fig-coupon-mu"] = dict(
                data=md, layout=layout("Friction coefficient on every surface",
                                       "Downward slip through the grasp, mm (log)",
                                       yaxis=dict(type="log"), height=320))
        srows = []
        for ss in (16, 8, 4, 2, 1):
            r = [f"{1000.0 / 60.0 / ss:.2f} ms"]
            for solv in ("fpgs", "mujoco"):
                v = kn.get(f"ss{ss}_{solv}")
                r.append("&mdash;" if not v else
                         ("<span style='color:#ff6b7a'>lost</span>"
                          if not v.get("held") else "%.3f mm" % v["axial_slip_mm"]))
            srows.append(r)
        TABLES["tbl-coupon-dt"] = srows


def t_coupon_mech(mech):
    if not mech:
        return
    rows = []
    for k in sorted([x for x in mech if x.startswith("ov")],
                    key=lambda x: mech[x].get("overlap_mm", 0)):
        v = mech[k]
        if v.get("solver") != "fpgs":
            continue
        mj = mech.get(k.replace("_fpgs", "_mujoco"), {})
        rows.append(["%g mm" % v["overlap_mm"],
                     "%.3f m/s" % v["peak_speed_m_s"],
                     "%.2f mm" % v["displacement_mm"],
                     "%.5f mm" % mj.get("displacement_mm", float("nan"))])
    if rows:
        TABLES["tbl-coupon-mech"] = rows
    brows = []
    for k in sorted([x for x in mech if x.startswith("beta")],
                    key=lambda x: mech[x].get("beta", 0)):
        v = mech[k]
        col = "#61d69b" if v["displacement_mm"] < 0.01 else "#ff6b7a"
        brows.append(["%g" % v["beta"], "%.3f m/s" % v["peak_speed_m_s"],
                      "<span style='color:%s'>%.2f mm</span>"
                      % (col, v["displacement_mm"])])
    if brows:
        TABLES["tbl-coupon-beta"] = brows


def t_capture(cap):
    rows = []
    for k, v in cap.items():
        streams = "on" if "streams=True" in k else "off"
        dbuf = "on" if "double_buffer=True" in k else "off"
        ok = v.strip() == "captured"
        cell = ('<span style="color:#61d69b">captured</span>' if ok else
                '<span style="color:#ff6b7a">fails &mdash; CUDA error 901/905</span>')
        rows.append([streams + (" (default)" if streams == "on" else ""),
                     dbuf + (" (default)" if dbuf == "on" else ""), cell])
    rows.sort(key=lambda r: (r[1].startswith("off"), r[0].startswith("off")))
    TABLES["tbl-capture"] = rows


def t_defaults(d):
    order = [("library_defaults", "SolverFeatherPGS(model), nothing changed"),
             ("defaults_no_double_buffer", "defaults, double buffering off"),
             ("split_rows150", "split mode, 150 rows"),
             ("split_rows201", "split mode, 201 rows (what the scene needs)"),
             ("split_rows256", "split mode, 256 rows"),
             ("matrixfree_rows256", "matrix-free mode, 256 rows")]
    rows = []
    for k, lbl in order:
        v = d.get(k)
        if not v:
            continue
        if "Failed to load CUDA module" in str(v.get("failure", "")):
            rows.append([lbl, "split", str(v.get("dense_max_constraints", "?")),
                         "<span style='color:#ff6b7a'>kernel will not load</span>",
                         "&mdash;"])
            continue
        rows.append([lbl, v.get("pgs_mode", "?"),
                     str(v.get("dense_max_constraints", "?")),
                     outcome_cell(f"def_{k}"),
                     "%.2fx realtime" % v.get("rtf", 0)])
    if rows:
        TABLES["tbl-defaults"] = rows


def t_phase(ph):
    """Gain multiplier against how far the task gets and whether the cube stays."""
    rows = [
        ["1x <span style='color:#6f8298'>(as tuned)</span>",
         "<span style='color:#ff6b7a'>APPROACH &mdash; never advances</span>", "&mdash;"],
        ["3x", "<span style='color:#ff6b7a'>APPROACH &mdash; never advances</span>", "&mdash;"],
        ["10x", "<span style='color:#ff6b7a'>APPROACH &mdash; never advances</span>", "&mdash;"],
        ["30x", "SHAKE",
         "<span style='color:#ff6b7a'>dropped 0.13 s in</span>"],
        ["100x", "SHAKE",
         "<span style='color:#ff6b7a'>dropped 3.65 s in</span>"],
        ["1x with the gravity torque supplied", "SHAKE",
         "<span style='color:#61d69b'>held, 0.82 mm</span>"],
    ]
    TABLES["tbl-phase"] = rows


def t_spec(spec):
    order = [("base_gap1", "12 iterations, default gap"),
             ("velit4_gap1", "+ 4 velocity iterations, default gap"),
             ("base_gap0.02", "12 iterations, gap shrunk 50x"),
             ("velit4_gap0.02", "+ 4 velocity iterations, gap shrunk 50x"),
             ("velit8_gap0.02", "+ 8 velocity iterations, gap shrunk 50x")]
    rows = []
    for k, lbl in order:
        v = spec.get(k)
        if not v:
            continue
        rows.append([lbl, str(v.get("speculative_contacts", "?")),
                     f"{v['drift_max_mm']:.2f} mm"])
    if rows:
        TABLES["tbl-spec"] = rows


RESCORE = {}


def held(name):
    return RESCORE.get(name, {}).get("outcome") == "held"


def creep(name):
    return RESCORE.get(name, {}).get("drift_max_mm")


def outcome_cell(name):
    o = RESCORE.get(name, {}).get("outcome")
    if o == "held":
        return "%.2f mm" % RESCORE[name]["drift_max_mm"]
    if o == "cube never leaves the table":
        return "<span style='color:#ff6b7a'>never lifted</span>"
    if o == "cube dropped during the shake":
        return "<span style='color:#ff6b7a'>dropped</span>"
    if o:
        return "<span style='color:#ff6b7a'>no grasp</span>"
    return "&mdash;"


if __name__ == "__main__":
    s = json.loads(Path(sys.argv[1]).read_text())
    RESCORE.update(json.loads(
        Path("/home/horde/repos/fpgs-study/rescored.json").read_text()))
    f_droop(s.get("tracking", {}))
    f_traces()
    f_configs(s.get("configs", {}), s.get("misc", {}))
    f_substeps(s.get("substeps", {}))
    f_budget(s.get("budget", {}), s.get("watermarks", {}))
    f_grip()
    f_inner(s.get("inner", {}), s.get("substeps", {}), s.get("iters", {}))
    f_tuning(s.get("tuning", {}), s.get("mu_sweep", {}))
    f_coupon(s.get("coupon", {}))
    t_coupon_mech(s.get("coupon_mech", {}))
    t_capture(s.get("graph_capture", {}))
    t_spec(s.get("speculative", {}))
    t_phase(s.get("phase", {}))
    t_defaults(s.get("defaults", {}))

    js = ["// generated by make_plots.py",
          "var FIGS = " + json.dumps(FIGS) + ";",
          "var TABLES = " + json.dumps(TABLES) + ";",
          """
var CFG = {displayModeBar:false, responsive:true};
document.querySelectorAll('.plotbox').forEach(function(el){
  var f = FIGS[el.id];
  if (f) { Plotly.newPlot(el, f.data, f.layout, CFG); }
  else { var fig = el.closest('figure'); if (fig) fig.style.display = 'none'; }
});
document.querySelectorAll('table[id^="tbl-"]').forEach(function(t){
  if (!TABLES[t.id]) { var w = t.closest('.table-wrap'); if (w) w.style.display = 'none'; return; }
  var id = t.id;
  var b = t.querySelector('tbody');
  b.innerHTML = TABLES[id].map(function(r){
    return '<tr>' + r.map(function(c,i){
      return '<td' + (i ? ' class="num"' : '') + '>' + c + '</td>';
    }).join('') + '</tr>';
  }).join('');
});
"""]
    (REPORT / "figures.js").write_text("\n".join(js))
    print(f"figures: {', '.join(FIGS)}")
    print(f"tables:  {', '.join(TABLES)}")
    print("bytes:", (REPORT / 'figures.js').stat().st_size)
