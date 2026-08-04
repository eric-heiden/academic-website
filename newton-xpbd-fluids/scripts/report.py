"""Build the standalone HTML report from the benchmark artefacts."""

from __future__ import annotations

import argparse
import base64
import json
import shutil
from pathlib import Path

HERE = Path(__file__).parent
RES = HERE / "results"

NEWTON_SHA = "615e148d"
OMNI_SHA = "112ae73"
GPU = "NVIDIA RTX PRO 6000 Blackwell Server Edition — MIG 1g.24gb slice"
DRIVER = "580.126.20"

# ---------------------------------------------------------------- data load

rows = [json.loads(line) for line in (RES / "sweep.jsonl").read_text().splitlines() if line.strip()]
rows = [r for r in rows if "error" not in r]


def pick(group, **f):
    out = [r for r in rows if r.get("group") == group and all(r.get(k) == v for k, v in f.items())]
    return sorted(out, key=lambda r: (r["particle_count"], r["iterations"], r["h_over_s"]))


def one(group, **f):
    m = pick(group, **f)
    return m[0] if m else None


LBL = {
    "newton": "Newton SolverXPBD",
    "omnisurg:baseline": "OmniSurg PBF (baseline)",
    "omnisurg:all": "OmniSurg PBF (optimized)",
}
COL = {"newton": "var(--series-1)", "omnisurg:baseline": "var(--series-2)", "omnisurg:all": "var(--series-3)"}

# ---------------------------------------------------------------- kernel categories

# Categorical slots 1-5 in the documented fixed order, plus muted gray for the
# residual "other" bucket. Values are CSS custom properties, resolved at draw
# time so dark mode uses its own validated steps rather than a flip.
CATS = [
    ("neighbor", "Neighbor search / list build", "--series-1"),
    ("density", "Density constraint solve", "--series-2"),
    ("apply", "Integrate, apply, bounds, reorder", "--series-3"),
    ("rigid", "Rigid coupling &amp; contact pipeline", "--series-4"),
    ("render", "Render-surface pass", "--series-5"),
    ("mem", "memset / memcpy", "--muted"),
]


def categorize(name: str) -> str:
    n = name.lower()
    if "memset" in n or "memcpy" in n:
        return "mem"
    if "render_surface" in n:
        return "render"
    if "compute_cell_indices" in n or "compute_cell_offsets" in n or "build_pbf_neighbor" in n:
        return "neighbor"
    if "build_pbf_neighbors_lambdas" in n:  # fused build + first lambda
        return "neighbor"
    if "sorted_fluid" in n:
        return "apply"
    if "fluid_lambdas" in n or "fluid_deltas" in n or "pbf_lambdas" in n or "position_deltas" in n:
        return "density"
    if any(k in n for k in ("integrate", "project_fluid_bounds", "apply_pbf_deltas", "velocities_from_positions")):
        return "apply"
    if "apply_particle_deltas" in n:
        return "apply"
    return "rigid"


def breakdown(runner):
    r = one("breakdown", runner=runner)
    agg = {c[0]: 0.0 for c in CATS}
    for k in r["kernel_breakdown"]:
        agg[categorize(k["kernel"])] += k["total_ms"]
    top = [k for k in r["kernel_breakdown"] if k["total_ms"] >= 0.02]
    return r, agg, top


# ---------------------------------------------------------------- assemble payload

scaling = {
    k: [{"x": r["particle_count"], "y": r["ms_median"], "note": f"p95 {r['ms_p95']:.1f} ms"} for r in pick("scaling", runner=k)]
    for k in LBL
}
iters = {
    k: [{"x": r["iterations"], "y": r["ms_median"]} for r in pick("iterations", runner=k)]
    for k in ("newton", "omnisurg:all")
}
neigh = {
    k: [{"x": r["h_over_s"], "y": r["ms_median"]} for r in pick("neighborhood", runner=k)]
    for k in ("newton", "omnisurg:all")
}

MODE_ORDER = [
    "baseline", "uniform-grid", "fused", "specialized", "skip-render",
    "sorted", "fused+specialized", "all", "all+uniform", "all+flex",
]
MODE_DESC = {
    "baseline": "HashGrid + materialized neighbor list, no opt flags",
    "uniform-grid": "dense uniform cell grid instead of wp.HashGrid",
    "fused": "fuse neighbor build with the first lambda pass",
    "specialized": "one kernel per SPH kernel, host-precomputed coefficients",
    "skip-render": "skip the smoothing / anisotropy render pass",
    "sorted": "spatially sorted scratch copy of the particles",
    "fused+specialized": "fused build + specialized kernels",
    "all": "fused + specialized + skip-render + sorted",
    "all+uniform": "all, on the uniform grid backend",
    "all+flex": "all, with the FleX approximate density constraint",
}


def ablation(group, iterations):
    out = []
    for m in MODE_ORDER:
        r = one(group, runner=f"omnisurg:{m}")
        if r:
            out.append({"mode": m, "ms": r["ms_median"], "desc": MODE_DESC[m]})
    base = next(x["ms"] for x in out if x["mode"] == "baseline")
    for x in out:
        x["rel"] = base / x["ms"]
    nw = one(group, runner="newton")
    return {"modes": out, "newton_ms": nw["ms_median"] if nw else None, "iterations": iterations}


abl3 = ablation("ablation", 3)
abl8 = ablation("ablation8", 8)

bd = {}
for k in LBL:
    r, agg, top = breakdown(k)
    bd[k] = {"label": LBL[k], "total": r["ms_median"], "values": agg, "top": top[:8], "n": r["particle_count"]}

vstats = json.loads((RES / "visual_stats.json").read_text())

# --out-dir writes a hostable directory (index.html + assets/); with no argument
# the report is a single self-contained file with the filmstrip inlined.
ap = argparse.ArgumentParser()
ap.add_argument("--out-dir", type=Path, default=None)
cli = ap.parse_args()

FILMSTRIP = RES / "visual_filmstrip.png"
if cli.out_dir:
    (cli.out_dir / "assets").mkdir(parents=True, exist_ok=True)
    shutil.copy2(FILMSTRIP, cli.out_dir / "assets" / "visual_filmstrip.png")
    img_src = "assets/visual_filmstrip.png"
else:
    img_src = "data:image/png;base64," + base64.b64encode(FILMSTRIP.read_bytes()).decode()

# headline numbers (cube_dims lands near, not exactly on, the requested count)
n131 = {k: pick("scaling", runner=k)[-1]["ms_median"] for k in LBL}
n32 = {k: one("scaling", runner=k, particle_count=32768)["ms_median"] for k in LBL}


def slope(pts):
    """ms added per extra solver iteration (least-squares)."""
    xs = [p["x"] for p in pts]
    ys = [p["y"] for p in pts]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


slope_newton = slope(iters["newton"])
slope_omni = slope(iters["omnisurg:all"])

DATA = {
    "scaling": scaling,
    "iters": iters,
    "neigh": neigh,
    "abl3": abl3,
    "abl8": abl8,
    "breakdown": bd,
    "cats": CATS,
    "vstats": vstats,
    "labels": LBL,
}

payload = json.dumps(DATA)
charts_js = (HERE / "charts.js").read_text()

# ---------------------------------------------------------------- html

HTML = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>XPBD fluids in Newton vs OmniSurg PBF — comparison &amp; benchmark</title>
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface-1: #fcfcfb;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --muted: #898781;
  --gridline: #e1e0d9; --baseline: #c3c2b7; --border: rgba(11,11,11,.10);
  --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a;
  --series-4: #eda100; --series-5: #e87ba4;
  --good: #0ca30c; --critical: #d03b3b; --warning: #fab219;
  --code-bg: #f2f1ed;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --page: #0d0d0d; --surface-1: #1a1a19;
    --text-primary: #fff; --text-secondary: #c3c2b7; --muted: #898781;
    --gridline: #2c2c2a; --baseline: #383835; --border: rgba(255,255,255,.10);
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70;
    --series-4: #c98500; --series-5: #d55181;
    --code-bg: #232321;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface-1: #1a1a19;
  --text-primary: #fff; --text-secondary: #c3c2b7; --muted: #898781;
  --gridline: #2c2c2a; --baseline: #383835; --border: rgba(255,255,255,.10);
  --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70;
  --series-4: #c98500; --series-5: #d55181;
  --code-bg: #232321;
}
* { box-sizing: border-box; }
body { margin:0; background: var(--page); color: var(--text-primary);
  font: 15px/1.62 system-ui, -apple-system, "Segoe UI", sans-serif; }
.wrap { max-width: 1080px; margin: 0 auto; padding: 40px 24px 96px; }
header.top { display:flex; justify-content:space-between; align-items:flex-start; gap:24px;
  padding-bottom: 22px; border-bottom: 1px solid var(--border); margin-bottom: 34px; }
h1 { font-size: 27px; line-height:1.25; margin: 0 0 8px; letter-spacing:-.012em; }
h2 { font-size: 19px; margin: 52px 0 6px; letter-spacing:-.008em; }
h2 .eyebrow { display:block; font-size:11.5px; letter-spacing:.10em; text-transform:uppercase;
  color: var(--muted); font-weight:600; margin-bottom:5px; }
h3 { font-size: 15.5px; margin: 28px 0 6px; }
p { margin: 10px 0; color: var(--text-secondary); max-width: 74ch; }
p.lede { font-size: 16.5px; color: var(--text-primary); max-width: 72ch; }
a { color: var(--series-1); }
ul, ol { color: var(--text-secondary); max-width: 74ch; padding-left: 20px; }
li { margin: 5px 0; }
code, kbd { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .875em;
  background: var(--code-bg); padding: 1.5px 5px; border-radius: 4px; }
pre { background: var(--code-bg); padding: 13px 15px; border-radius: 8px; overflow-x: auto;
  font-size: 12.5px; line-height:1.55; border: 1px solid var(--border); }
pre code { background: none; padding: 0; }
.sub { color: var(--muted); font-size: 13px; }
.meta { font-size: 12.2px; color: var(--muted); text-align: right; white-space: nowrap; }
.meta b { color: var(--text-secondary); font-weight: 600; }
button.theme { background: var(--surface-1); color: var(--text-secondary); border: 1px solid var(--border);
  border-radius: 7px; padding: 5px 11px; font: inherit; font-size: 12.5px; cursor: pointer; margin-top:8px; }

.tiles { display:grid; grid-template-columns: repeat(auto-fit, minmax(198px,1fr)); gap: 14px; margin: 26px 0 8px; }
.tile { background: var(--surface-1); border: 1px solid var(--border); border-radius: 11px; padding: 17px 18px; }
.tile .k { font-size: 11.5px; letter-spacing:.055em; text-transform: uppercase; color: var(--muted); font-weight:600; }
.tile .v { font-size: 34px; line-height:1.06; margin: 9px 0 5px; letter-spacing:-.022em; }
.tile .d { font-size: 12.7px; color: var(--text-secondary); }
.tile .v small { font-size: 17px; color: var(--text-secondary); letter-spacing:0; }

.fig { background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px;
  padding: 18px 20px 12px; margin: 20px 0 0; }
figcaption { margin-bottom: 4px; }
.figtitle { display:block; font-size: 14.5px; font-weight: 600; color: var(--text-primary); }
.fignote { display:block; font-size: 12.7px; color: var(--muted); margin-top: 3px; max-width: 82ch; }
.plot { margin-top: 6px; }
.legend { display:flex; flex-wrap:wrap; gap: 15px; margin: 9px 0 2px; font-size: 12.7px; color: var(--text-secondary); }
.legend i { display:inline-block; width:11px; height:11px; border-radius:3px; margin-right:6px; vertical-align:-1px; }
.tick { font-size: 11.5px; }
.axlab { font-size: 12px; }
.dlab { font-size: 12px; }
.rlab { font-size: 12.5px; }
.rlab.em { font-weight: 600; }
details.tv { margin-top: 6px; border-top: 1px solid var(--border); padding-top: 7px; }
details.tv summary { cursor: pointer; font-size: 12.5px; color: var(--muted); list-style-position: outside; }
.tablewrap { overflow-x: auto; margin-top: 9px; }
table { border-collapse: collapse; width: 100%; font-size: 12.9px; }
th, td { text-align: left; padding: 6px 11px 6px 0; border-bottom: 1px solid var(--gridline); }
th { color: var(--muted); font-weight: 600; font-size: 11.7px; letter-spacing:.03em; text-transform: uppercase; }
td { color: var(--text-secondary); }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
tbody tr:last-child td { border-bottom: none; }
.tt { position: fixed; z-index: 50; pointer-events: none; opacity: 0; transition: opacity .09s;
  background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px; padding: 8px 11px;
  font-size: 12.5px; color: var(--text-primary); box-shadow: 0 6px 22px rgba(0,0,0,.16); max-width: 300px; }
.tt .dim { color: var(--muted); }
.cols { display:grid; grid-template-columns: 1fr 1fr; gap: 22px; }
@media (max-width: 760px) { .cols { grid-template-columns: 1fr; } .wrap { padding: 26px 15px 64px; } header.top { flex-direction: column; } .meta { text-align:left; } }
.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px; padding: 17px 19px; }
.card h3 { margin-top: 0; }
.callout { border-left: 3px solid var(--series-1); background: var(--surface-1); border-radius: 0 10px 10px 0;
  padding: 13px 17px; margin: 18px 0; }
.callout.warn { border-left-color: var(--warning); }
.callout.crit { border-left-color: var(--critical); }
.callout p:first-child { margin-top: 0; } .callout p:last-child { margin-bottom: 0; }
.callout strong { color: var(--text-primary); }
figure.shot { margin: 18px 0 0; background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 12px; padding: 14px; }
figure.shot img { width: 100%; display:block; border-radius: 7px; }
figure.shot figcaption { margin-top: 10px; font-size: 12.7px; color: var(--muted); }
.pill { display:inline-block; font-size: 11.4px; padding: 2px 8px; border-radius: 999px;
  border: 1px solid var(--border); color: var(--text-secondary); margin-right: 5px; }
.rec { display:grid; grid-template-columns: 34px 1fr; gap: 3px 14px; align-items:start; margin: 16px 0; }
.rec .n { font-variant-numeric: tabular-nums; font-weight: 600; color: var(--series-1); font-size: 15px;
  background: var(--surface-1); border:1px solid var(--border); border-radius: 8px; text-align:center; padding: 3px 0; }
.rec .b { }
.rec .b b { color: var(--text-primary); }
.rec .b p { margin: 3px 0 16px; }
hr.sep { border:none; border-top: 1px solid var(--border); margin: 42px 0 0; }
</style>
</head><body>
<div class="wrap">

<header class="top">
  <div>
    <h1>XPBD fluids in Newton vs. the OmniSurg PBF solver</h1>
    <p class="sub">Algorithmic comparison, head-to-head benchmark, and a list of what to fix.</p>
  </div>
  <div class="meta">
    <b>Newton</b> eric-heiden/flex-fluid @ __NEWTON_SHA__<br>
    <b>OmniSurg</b> feature/fluids @ __OMNI_SHA__<br>
    <b>GPU</b> RTX PRO 6000 Blackwell<br>
    <span style="color:var(--muted)">MIG 1g.24gb · driver __DRIVER__</span><br>
    <b>Warp</b> 1.16.0 · fp32<br>
    <button class="theme" id="themeBtn" type="button">Toggle theme</button>
  </div>
</header>

<p class="lede">Both implementations solve the same equations — Macklin &amp; M&uuml;ller position-based fluids, fp32, in Warp, on Newton <code>Model</code>/<code>State</code> arrays. They differ in one structural decision, and that decision costs Newton roughly an order of magnitude. This report measures the gap on an identical scene in an identical process, attributes it to specific kernels, and lists the changes that would close it.</p>

<div class="tiles">
  <div class="tile"><div class="k">At 131k particles</div><div class="v" id="t1"></div>
    <div class="d">Newton is this much slower per frame than the optimized OmniSurg solver, same scene, 8 substeps &times; 3 iterations.</div></div>
  <div class="tile"><div class="k">Cost per solver iteration</div><div class="v" id="t2"></div>
    <div class="d">Marginal ms added by one extra PBF iteration at 32k particles — Newton __SLOPE_N__ ms vs OmniSurg __SLOPE_O__ ms.</div></div>
  <div class="tile"><div class="k">Architectural vs. tuned</div><div class="v" id="t3"></div>
    <div class="d">Of the gap, this much is present before <em>any</em> of OmniSurg's optional optimizations are switched on.</div></div>
  <div class="tile"><div class="k">Density solve share</div><div class="v" id="t4"></div>
    <div class="d">Fraction of Newton's frame spent in its two PBF kernels at 65k particles. Everything else is noise.</div></div>
</div>

<h2><span class="eyebrow">Section 1</span>What the Newton fluid examples are for</h2>
<p>The branch adds eight examples under <code>newton/examples/fluid/</code>, a screen-space fluid renderer ported from the FleX demos, SDF particle contacts, and a 31-test solver suite. Read together, they describe a specific product intent.</p>

<div class="cols">
<div class="card">
<h3>Robotics manipulation of liquids — the headline</h3>
<p>Three of the eight examples are the same scenario at three levels of abstraction: <code>cup</code> (a cup of water you grab, explicitly documented as the arm-free ablation "easy to profile and tune"), <code>cup_transfer</code> (an IK-driven Franka FR3 carrying and spilling a cup, with adaptive substeps tied to carry speed), and <code>multiworld_cup</code> (the same scene replicated across isolated worlds). The design details — kinematically posed robot bodies, per-substep container-pose interpolation, wall-crossing velocity caps — are all about <em>containers moving under actuation without leaking</em>.</p>
</div>
<div class="card">
<h3>Two-way coupling as a property, not a script</h3>
<p><code>interactive_tank</code>'s docstring is the thesis: "No hand-tuned buoyancy, drag, or coupling forces are needed." Boxes float or sink from the unified XPBD solve. <code>wave_pool</code> (kinematic paddle, breaking waves, 6 bobbing primitives), <code>dam_break</code> (pillar + floating box) and <code>cereal_bowl</code> (19 torus cereal pieces in a dynamic bowl of opaque milk, ~172 shapes) repeat it with different geometry.</p>
</div>
<div class="card">
<h3>Vectorized environments</h3>
<p><code>multiworld_cup</code> exists to prove fluids replicate across <code>begin_world()</code>/<code>end_world()</code> and stay isolated; its <code>test_final()</code> asserts equal per-world particle counts and no cross-world water. Currently a 2-world proof of concept (the count is hardcoded), but the grouped hash grid and world-filtered contacts are in place.</p>
</div>
<div class="card">
<h3>FleX-grade visuals and materials</h3>
<p>A large fraction of the diff is renderer: anisotropic ellipsoid splatting, bilateral depth smoothing, refraction/Fresnel, translucent shadows, velocity-stretched diffuse foam. <code>multi_fluid_tank</code> runs three phases with per-phase absorption/IOR/specular; <code>cereal_bowl</code> renders opaque scattering milk. These exist to show arbitrary liquids, not just clear water.</p>
</div>
</div>

<p style="margin-top:18px"><strong style="color:var(--text-primary)">Scale of intent.</strong> Every example defaults to 60&ndash;120k particles at 60&nbsp;fps with 4&ndash;8 substeps and 2&ndash;4 iterations. That target is what makes the performance gap matter: it is exactly the regime where Newton currently lands at __N131_FPS__ fps and OmniSurg at __O131_FPS__ fps on this GPU slice.</p>

<h2><span class="eyebrow">Section 2</span>The one structural difference</h2>
<p>Both solvers build a spatial acceleration structure once per substep. What happens next is the whole story.</p>

<div class="cols">
<div class="card">
<h3><span class="pill" style="border-color:var(--series-1);color:var(--series-1)">Newton</span> Query the grid, every time</h3>
<p><code>compute_fluid_lambdas</code> and <code>solve_fluid_deltas</code> each open a fresh <code>wp.hash_grid_query</code> and re-walk the 27 neighbouring cells. With <em>k</em> solver iterations that is <b>2k full grid traversals per particle per substep</b>, plus one more for viscosity, one for vorticity, and one for foam spawning — 5 to 7 traversals per substep at default settings.</p>
<p>Because the grid cell width equals the query radius, each traversal visits 27 cells holding roughly 157 candidates to find the ~24 that are actually inside <em>h</em> — a <b>6.5&times; rejection rate</b> paid on every single traversal.</p>
</div>
<div class="card">
<h3><span class="pill" style="border-color:var(--series-3);color:var(--series-3)">OmniSurg</span> Materialize once, read k times</h3>
<p><code>build_pbf_neighbor_list_range</code> runs <b>once per substep</b> and writes a flat <code>int32</code> array of neighbor indices. Every constraint iteration then reads that list — no grid query, no cell walk, no rejection.</p>
<p>The list is stored <b>slot-major</b>: <code>neighbor_indices[slot * N + i]</code>. Consecutive threads read consecutive addresses, so each slot access is a fully coalesced load. The cost is memory (<code>N &times; max_neighbors &times; 4</code>&nbsp;B) and a fixed <code>max_neighbors</code> cap with overflow flags.</p>
</div>
</div>

<div class="callout">
<p><strong>This is not a micro-optimization.</strong> It changes the complexity of the inner loop from "traverse a spatial structure" to "read a contiguous array." Everything measured below follows from it. OmniSurg's five <em>optional</em> optimization flags — fused build, specialized kernels, sorted scratch, uniform grid, FleX-approximate constraint — together account for only <b>__ABL_REL__&times;</b>; the materialized list accounts for the rest.</p>
</div>

<div id="fig-iters"></div>

<h2><span class="eyebrow">Section 3</span>Benchmark method</h2>
<p>The two solvers are both Newton <code>SolverBase</code> subclasses, so they were driven by <b>one harness, in one process, against one <code>Model</code></b>. Nothing differs between runs except the solver object.</p>
<ul>
  <li><b>Scene</b> — a cube of fluid particles dropped into a tank and allowed to settle. Newton confines it with five static box shapes (it has no analytic bounds); OmniSurg uses its <code>bounds_min/max</code> projection. Both run the same particle count, spacing, jitter, gravity, and CFL velocity clamp.</li>
  <li><b>Matched physics</b> — <code>h = 1.8&nbsp;&times;</code> rest distance for both, poly6 density kernel for both, cohesion and viscosity off, relaxation 1.0. Rest density is calibrated with each solver's own lattice sum, since Newton accumulates a mass-weighted density in kg/m&sup3; and OmniSurg a massless kernel-weight sum; matching the calibrations makes the normalized constraint <code>&rho;/&rho;&#8320;&nbsp;&minus;&nbsp;1</code> identical. Verified: after 120 frames both settle to the same centre of mass, height and mean speed to within a few percent.</li>
  <li><b>Timing</b> — the whole frame (all 8 substeps) is captured into one CUDA graph and replayed; 20 warm-up frames, then 100 measured frames with a device sync per frame. Median reported. OmniSurg normally captures a smaller solver-internal graph and launches it from <code>step()</code>; its own <code>_engine_managed_stages</code> escape hatch disables that so the whole frame can be captured externally, exactly as the Newton examples do.</li>
  <li><b>Fairness</b> — OmniSurg's render-surface pass (smoothing + anisotropy) is disabled in the optimized configuration because Newton's equivalent, <code>update_render_particles()</code>, is never called headless. It is left <em>on</em> in the baseline configuration, and shown separately in the kernel breakdown.</li>
</ul>

<div class="callout warn">
<p><strong>Read the absolute numbers with care.</strong> This machine exposes a <b>MIG 1g.24gb slice</b> — roughly an eighth of an RTX PRO 6000 Blackwell — and 4 CPU cores. Absolute milliseconds are therefore several times higher than a full GPU would give, and the reduced SM count slightly <em>flatters</em> whichever solver is more launch-bound. The ratios below are the load-bearing result; the absolute figures are not a statement about Newton's real-world frame rate on a full GPU.</p>
</div>

<h2><span class="eyebrow">Section 4</span>How cost scales</h2>
<div id="fig-scaling"></div>
<div id="fig-breakdown"></div>

<h2><span class="eyebrow">Section 5</span>What OmniSurg's optimization flags actually buy</h2>
<p>All five flags default to <em>off</em> in OmniSurg. Enabling them individually and together, at both a shallow (3) and a deep (8) iteration count, separates "architecture" from "tuning".</p>
<div id="fig-abl3"></div>
<div id="fig-abl8"></div>
<p><strong style="color:var(--text-primary)">Reading.</strong> Kernel specialization (one compiled kernel per SPH kernel choice, with all coefficients precomputed on the host and divisions turned into multiplications) and skipping the render-surface pass are the reliable individual wins. Spatially sorted scratch helps here but is configuration-sensitive — it forces a canonical round-trip through <code>project_fluid_bounds</code> every iteration, and OmniSurg's own harness reports it as a regression under different settings. Fusing the neighbor build with the first lambda pass is roughly neutral: it saves a launch and a full re-read of the neighbor array, but raises register pressure in the query kernel. The FleX-approximate density constraint adds nothing on top of the specialized kernels — the specialized variants still compute the gradient sum in the loop and only overwrite it afterwards, so the compiler cannot eliminate the work.</p>

<h2><span class="eyebrow">Section 6</span>Neighborhood size</h2>
<p>The ratio <code>h / rest_distance</code> sets how many particles fall inside the smoothing kernel — cubically. Newton's default is 1.8; OmniSurg's shipped surgical config uses 2.5. Because Newton pays the traversal cost <em>k</em> times per substep and OmniSurg once, the same physics choice costs them very differently.</p>
<div id="fig-neigh"></div>

<h2><span class="eyebrow">Section 7</span>Visual comparison</h2>
<p>Identical dam-break scene, identical parameters, both solvers stepped for 90 frames. The point is that the gap is a cost gap, not a quality gap: the two produce the same flow.</p>
<figure class="shot">
  <img src="__IMGSRC__" alt="Filmstrip comparing Newton and OmniSurg dam-break simulations at seven time points; both collapse, surge along the floor, run up the far wall and settle almost identically.">
  <figcaption>A column of __VN__ particles collapses and surges down a tank. Top row Newton, bottom row OmniSurg. Colour is particle speed on a shared scale. The collapse, the surge front, the run-up at the far wall and the returning wave land at the same times in both; Newton's sheet disperses slightly more at the leading edge, OmniSurg's stays marginally more compact.</figcaption>
</figure>
<div id="fig-front"></div>

<h2><span class="eyebrow">Section 8</span>Capability, not just speed</h2>
<p>OmniSurg is faster partly because it does less. An honest comparison has to say what each side gives up.</p>
<div class="cols">
<div class="card">
<h3>Only Newton has it</h3>
<ul>
<li><b>Two-way rigid and articulation coupling</b> through the standard particle soft-contact pipeline — the entire point of the examples. OmniSurg's solver explicitly declares <code>requires_newton_contacts() &rarr; False</code> and ignores its <code>contacts</code> argument.</li>
<li><b>SDF shape collision</b> — one texture SDF sample per particle-shape pair, which is what makes cups and bowls tractable at 100k particles.</li>
<li><b>Multi-world</b> replication with a grouped hash grid and world-filtered contacts.</li>
<li><b>Per-particle mass</b>, so multi-phase fluids come for free; diffuse foam; vorticity confinement; a full screen-space renderer.</li>
<li>Fluid coexisting with cloth, soft bodies, springs and tets in the same solver.</li>
</ul>
</div>
<div class="card">
<h3>Only OmniSurg has it</h3>
<ul>
<li><b>The materialized neighbor list</b> — the subject of this report.</li>
<li><b>Compile-time kernel specialization</b> with host-precomputed coefficients.</li>
<li>An alternative <b>dense uniform-grid backend</b> (atomic-exchange linked lists) for bounded domains.</li>
<li>A <b>multi-GPU fluid track</b>: the fluid runs on a second device against a replicated model view, with a P2P probe that falls back to pinned host staging, and events overlapping it with the main track.</li>
<li>Explicit <b>neighbor overflow diagnostics</b> and a <code>max_neighbors</code> cap that bounds worst-case cost.</li>
<li>Wall friction / wall viscosity against a static triangle mesh.</li>
</ul>
</div>
</div>
<p>So the comparison is not "replace one with the other". OmniSurg's solver is a fluid-only, boundary-driven solver for a surgical irrigation scene; Newton's is a general coupled solver. But the neighbor-list architecture is <em>orthogonal</em> to all of Newton's extra capability — nothing about two-way coupling, SDF contacts or multi-world prevents materializing a neighbor list.</p>

<h2><span class="eyebrow">Section 9</span>What to do</h2>
<p>Ordered by expected value. The first item is worth more than all the others combined.</p>

<div class="rec"><div class="n">1</div><div class="b">
<b>Materialize the neighbor list once per substep.</b>
<p>Build a flat <code>int32</code> array of neighbor indices (slot-major, <code>[slot * N + i]</code>, so warp lanes read consecutive addresses) immediately after the hash-grid build, then have <code>compute_fluid_lambdas</code>, <code>solve_fluid_deltas</code>, <code>solve_fluid_velocities</code> and <code>compute_fluid_vorticity</code> read it instead of re-querying. This is the change that produces the measured gap. Add a <code>fluid_max_neighbors</code>-sized cap with overflow flags — the parameter already exists and already truncates, so the semantics are unchanged; it would simply become a real allocation bound. Memory cost at 100k particles and 64 slots is 25&nbsp;MB.</p>
</div></div>

<div class="rec"><div class="n">2</div><div class="b">
<b>If the list is not acceptable, at least halve the traversals.</b>
<p><code>compute_fluid_lambdas</code> and <code>solve_fluid_deltas</code> traverse the same neighborhood back to back within one iteration. They cannot be fully fused (the second needs every neighbor's &lambda;), but the first iteration's traversal can produce the list the second consumes — which is exactly OmniSurg's <code>fuse_neighbor_build_first_lambda</code>. Failing that, caching per-particle neighbor <em>counts</em> and cell ranges from the first traversal removes most of the rejection work from the second.</p>
</div></div>

<div class="rec"><div class="n">3</div><div class="b">
<b>Cut the 6.5&times; candidate rejection rate.</b>
<p>The grid is built with cell width equal to the query radius, so every query visits 27 cells and examines ~157 candidates to accept ~24. A cell width of <code>h/2</code> visits more cells but examines far fewer candidates; on typical PBF workloads this is a clear win and costs nothing but the <code>build()</code> radius argument. Worth measuring both ways before committing.</p>
</div></div>

<div class="rec"><div class="n">4</div><div class="b">
<b>Stop rebuilding the hash grid twice per substep.</b>
<p>The diffuse-foam layer builds a <em>second</em> full hash grid every substep at a different radius, for a render-only feature that only needs frame-rate updates. Move the whole diffuse step out of the substep loop and reuse the simulation grid. Two related hazards go away with it: the grid rebuild memsets <code>cell_starts</code>/<code>cell_ends</code> over <em>cells</em>, not particles — 16.8&nbsp;MB per build at the default 128&sup3;, and 134&nbsp;MB at the 256&sup3; the cup examples request, which at 8 substeps &times; 2 builds is on the order of a gigabyte per frame of pure memset traffic independent of particle count; and both grids share a <code>static</code> host descriptor, so two builds at different radii inside one captured graph is a latent correctness bug at replay time.</p>
</div></div>

<div class="rec"><div class="n">5</div><div class="b">
<b>Cap <code>soft_contact_max</code> by default.</b>
<p>It defaults to <code>shape_count &times; particle_count</code>, and <code>solve_particle_shape_contacts</code> is launched at that dimension <em>three times per iteration</em> for fluid scenes. Most threads early-out, but the launch is still enormous — only <code>cereal_bowl</code> caps it, at <code>6 &times; particle_count</code>. A sensible default bound (or a compaction pass) removes a large launch from every fluid scene.</p>
</div></div>

<div class="rec"><div class="n">6</div><div class="b">
<b>Address <code>body_delta</code> atomic contention for container scenes.</b>
<p>Every fluid particle in contact with a container body does <code>wp.atomic_sub</code> on the same six floats, in two of the three contact passes, every iteration. For the flagship cup scenes that is on the order of a million serialized atomics per substep onto one address. A per-body block reduction, or accumulating into a small per-shape scratch buffer before a single reduction, would remove it.</p>
</div></div>

<div class="rec"><div class="n">7</div><div class="b">
<b>Consider compile-time specialization for the SPH kernel choice.</b>
<p>Cheap, mechanical, and measured here at 5&ndash;10% on its own. Newton has fewer runtime branches than OmniSurg did, but the coefficient recomputation per neighbor (<code>315/(64&pi;h&#8313;)</code> and friends) is the same pattern and can be hoisted to host-computed scalars, with divisions turned into multiplications by a precomputed reciprocal.</p>
</div></div>

<div class="callout crit">
<p><strong>Two defects found while setting this up, unrelated to performance.</strong></p>
<p><b>The branch does not import under its own declared minimum Warp version.</b> <code>pyproject.toml</code> pins <code>warp-lang&gt;=1.14.0</code>, but <code>solver_xpbd.py</code> annotates a property as <code>wp.array[wp.int32] | None</code> and the module has no <code>from __future__ import annotations</code>, so the annotation is evaluated at class-definition time and raises <code>TypeError: unsupported operand type(s) for |</code> on Warp 1.14.0. Everything in this report ran on 1.16.0. Either raise the floor or add the future import.</p>
<p><b><code>diffuse_spawn_counter</code> is never reset in the step path.</b> It is only ever incremented; <code>clear_diffuse_particles()</code> zeroes it but is not called during stepping. On int32 overflow the derived slot index goes negative and is used unchecked in <code>wp.atomic_cas</code> — an out-of-bounds write. Reachable after ~2&sup3;&sup1; spawns.</p>
</div>

<h2><span class="eyebrow">Section 10</span>Reproducing this</h2>
<p>The harness is published alongside this page:
<a href="scripts/scene.py"><code>scene.py</code></a> (the shared scene),
<a href="scripts/runners.py"><code>runners.py</code></a> (the two solvers behind one interface),
<a href="scripts/sweep.py"><code>sweep.py</code></a> (the matrix),
<a href="scripts/bench.py"><code>bench.py</code></a> (a single configuration),
<a href="scripts/visual.py"><code>visual.py</code></a> + <a href="scripts/render_visual.py"><code>render_visual.py</code></a> (the filmstrip), and
<a href="scripts/report.py"><code>report.py</code></a> + <a href="scripts/charts.js"><code>charts.js</code></a> (this page).
Both packages are installed into a single virtualenv so they share one Warp and one Newton core.</p>
<pre><code># one venv, both implementations, same warp
cd newton-flex-fluid
uv sync --extra examples --extra dev
uv pip install 'warp-lang==1.16.0'          # 1.14 does not import this branch
uv pip install --no-deps -e ../omnisurg-fluids pyyaml

# the full matrix (49 configurations, appends to results/sweep.jsonl)
uv run --no-sync python fluidbench/sweep.py --frames 100

# one configuration with a per-kernel CUDA breakdown
uv run --no-sync python fluidbench/bench.py --runner newton \
    --particle-count 65536 --iterations 3 --kernel-breakdown

# the visual comparison
uv run --no-sync python fluidbench/visual.py --particle-count 32768 --frames 90
uv run --no-sync python fluidbench/render_visual.py</code></pre>
<p class="sub">Every measurement in this report is in <a href="data/sweep.jsonl">data/sweep.jsonl</a> &mdash; one JSON object per configuration, including the raw per-frame samples, the resolved solver parameters, and the final particle-state statistics used to verify the two solvers agree physically. The dam-break trajectory statistics behind the filmstrip are in <a href="data/visual_stats.json">data/visual_stats.json</a>.</p>

<hr class="sep">
<p class="sub" style="margin-top:20px">Newton <code>eric-heiden/flex-fluid</code> @ __NEWTON_SHA__ &middot; OmniSurg <code>feature/fluids</code> @ __OMNI_SHA__ &middot; Warp 1.16.0 &middot; __GPU__ &middot; driver __DRIVER__ &middot; 100 measured frames per configuration after 20 warm-up frames, whole frame CUDA-graph captured.</p>

</div>
<script>const DATA = __PAYLOAD__;</script>
<script>__CHARTS__</script>
<script>__BUILD__</script>
</body></html>
"""

BUILD = """
const S1 = () => getComputedStyle(document.documentElement).getPropertyValue('--series-1').trim();
const S2 = () => getComputedStyle(document.documentElement).getPropertyValue('--series-2').trim();
const S3 = () => getComputedStyle(document.documentElement).getPropertyValue('--series-3').trim();
const kfmt = (v) => v >= 1000 ? (v/1000).toFixed(v % 1000 === 0 ? 0 : 1) + 'k' : String(v);
const L = DATA.labels;

function buildAll() {
  document.querySelectorAll('.fig, figure.fig').forEach(n => { if (n.parentElement.id.startsWith('fig-')) n.remove(); });

  /* ---- 1. cost per solver iteration (the money chart) ---- */
  figure('fig-iters', {
    title: 'Cost of one extra PBF solver iteration',
    note: '32,768 particles, 8 substeps, h/rest = 1.8. Newton re-traverses the hash grid twice per iteration; OmniSurg reads a list built once per substep. The slopes are the whole argument.',
    legendItems: [{name: L['newton'], color: S1()}, {name: L['omnisurg:all'], color: S3()}],
    build: (p) => lineChart(p, {
      height: 340, unit: 'ms/frame', xLabel: 'PBF solver iterations', yLabel: 'ms / frame (median)',
      xTicks: [1,2,3,4,6,8], xFmt: (v)=>String(v), yFmt: (v)=>String(v),
      ariaLabel: 'Frame time versus solver iteration count for both implementations',
      series: [
        {name: L['newton'], short: 'Newton', color: S1(), points: DATA.iters['newton']},
        {name: L['omnisurg:all'], short: 'OmniSurg', color: S3(), points: DATA.iters['omnisurg:all']},
      ],
    }),
    tableCols: [{name:'Iterations', key:'it', num:true}, {name:'Newton (ms)', key:'n', num:true},
                {name:'OmniSurg (ms)', key:'o', num:true}, {name:'Ratio', key:'r', num:true}],
    tableRows: DATA.iters['newton'].map((p,i) => {
      const o = DATA.iters['omnisurg:all'][i];
      return {it: p.x, n: p.y.toFixed(2), o: o.y.toFixed(2), r: (p.y/o.y).toFixed(1) + '\\u00d7'};
    }),
  });

  /* ---- 2. scaling ---- */
  figure('fig-scaling', {
    title: 'Frame time versus particle count',
    note: 'Log\\u2013log. 8 substeps \\u00d7 3 iterations, whole frame CUDA-graph captured. Both axes log, so a straight line means a power law; all three are close to linear in N.',
    legendItems: [{name: L['newton'], color: S1()}, {name: L['omnisurg:baseline'], color: S2()}, {name: L['omnisurg:all'], color: S3()}],
    build: (p) => lineChart(p, {
      height: 360, unit: 'ms/frame', xLog: true, yLog: true, padRight: 118,
      xLabel: 'fluid particles', yLabel: 'ms / frame (median, log)',
      xFmt: kfmt, yFmt: (v)=>String(v),
      ariaLabel: 'Log-log frame time versus particle count for three solver configurations',
      rules: [{y: 16.67, label: '16.7 ms = 60 fps budget'}],
      series: [
        {name: L['newton'], short: 'Newton', color: S1(), points: DATA.scaling['newton']},
        {name: L['omnisurg:baseline'], short: 'OmniSurg base', color: S2(), points: DATA.scaling['omnisurg:baseline']},
        {name: L['omnisurg:all'], short: 'OmniSurg opt', color: S3(), points: DATA.scaling['omnisurg:all']},
      ],
    }),
    tableCols: [{name:'Particles', key:'n', num:true}, {name:'Newton (ms)', key:'a', num:true},
                {name:'OmniSurg baseline (ms)', key:'b', num:true}, {name:'OmniSurg optimized (ms)', key:'c', num:true},
                {name:'Newton / optimized', key:'r', num:true}],
    tableRows: DATA.scaling['newton'].map((p,i) => {
      const b = DATA.scaling['omnisurg:baseline'][i], c = DATA.scaling['omnisurg:all'][i];
      return {n: p.x.toLocaleString(), a: p.y.toFixed(1), b: b.y.toFixed(1), c: c.y.toFixed(1),
              r: (p.y/c.y).toFixed(1) + '\\u00d7'};
    }),
  });

  /* ---- 3. kernel breakdown ---- */
  const keys = DATA.cats.map(c => ({key: c[0], name: c[1].replace('&amp;','&'),
    color: getComputedStyle(document.documentElement).getPropertyValue(c[2]).trim()}));
  const order = ['newton','omnisurg:baseline','omnisurg:all'];
  figure('fig-breakdown', {
    title: 'Where the frame goes \\u2014 GPU time by kernel category, 65,536 particles',
    note: 'CUDA event timing over 4 uncaptured frames. Newton\\u2019s neighbor search is not a separate bar because it happens <em>inside</em> the two density kernels \\u2014 that is precisely the problem: the orange segment for Newton is density solve <em>plus</em> 6 full grid traversals per substep.',
    legendItems: keys.map(k => ({name: k.name, color: k.color})),
    build: (p) => stackChart(p, {
      rowH: 58, labelW: 214, xLabel: 'ms / frame', keys,
      ariaLabel: 'Stacked GPU time by kernel category for three solver configurations',
      rows: order.map(k => ({label: L[k], sub: DATA.breakdown[k].total.toFixed(1) + ' ms/frame wall clock',
                             values: DATA.breakdown[k].values})),
    }),
    tableCols: [{name:'Implementation', key:'impl'}, {name:'Kernel', key:'kern'},
                {name:'ms/frame', key:'ms', num:true}, {name:'Launches/frame', key:'lc', num:true}],
    tableRows: order.flatMap(k => DATA.breakdown[k].top.map(t => ({
      impl: L[k], kern: t.kernel.replace(/^forward kernel /,'').replace(/^builtin kernel /,'').replace(/_[0-9a-f]{8}$/,''),
      ms: t.total_ms.toFixed(3), lc: t.launches}))),
  });

  /* ---- 4 & 5. ablations ---- */
  for (const [id, key, it] of [['fig-abl3','abl3',3], ['fig-abl8','abl8',8]]) {
    const a = DATA[key];
    figure(id, {
      title: `OmniSurg optimization flags at ${it} solver iterations \\u2014 32,768 particles`,
      note: `Lower is better. Newton at this configuration: <b>${a.newton_ms.toFixed(1)} ms</b> (off the scale of this chart). Baseline here already includes the materialized neighbor list, which is not a flag \\u2014 it is how the solver is built.`,
      build: (p) => barChart(p, {
        rowH: 30, labelW: 168, unit: 'ms/frame', xLabel: 'ms / frame (median)',
        ariaLabel: 'Frame time for each OmniSurg optimization configuration',
        vFmt: (v)=>v.toFixed(2), xFmt: (v)=>v.toFixed(0),
        rows: a.modes.map(m => ({label: m.mode, value: m.ms, color: S3(),
                                 emphasis: m.mode==='baseline'||m.mode==='all', note: m.desc})),
      }),
      tableCols: [{name:'Configuration', key:'m'}, {name:'What it changes', key:'d'},
                  {name:'ms/frame', key:'ms', num:true}, {name:'vs baseline', key:'r', num:true}],
      tableRows: a.modes.map(m => ({m: m.mode, d: m.desc, ms: m.ms.toFixed(2), r: m.rel.toFixed(2)+'\\u00d7'})),
    });
  }

  /* ---- 6. neighborhood ---- */
  figure('fig-neigh', {
    title: 'Sensitivity to smoothing-length ratio h / rest_distance',
    note: '32,768 particles, 3 iterations. Neighbor count grows as the cube of this ratio. Newton pays it on every traversal; OmniSurg pays it once per substep and then reads a list.',
    legendItems: [{name: L['newton'], color: S1()}, {name: L['omnisurg:all'], color: S3()}],
    build: (p) => lineChart(p, {
      height: 320, unit: 'ms/frame', xLabel: 'h / rest_distance', yLabel: 'ms / frame (median)',
      xTicks: [1.5,1.8,2.2,2.5], xFmt: (v)=>v.toFixed(1), yFmt: (v)=>String(v),
      ariaLabel: 'Frame time versus smoothing length ratio',
      series: [
        {name: L['newton'], short: 'Newton', color: S1(), points: DATA.neigh['newton']},
        {name: L['omnisurg:all'], short: 'OmniSurg', color: S3(), points: DATA.neigh['omnisurg:all']},
      ],
    }),
    tableCols: [{name:'h / rest', key:'h', num:true}, {name:'Newton (ms)', key:'n', num:true},
                {name:'OmniSurg (ms)', key:'o', num:true}, {name:'Ratio', key:'r', num:true}],
    tableRows: DATA.neigh['newton'].map((p,i) => {
      const o = DATA.neigh['omnisurg:all'][i];
      return {h: p.x.toFixed(1), n: p.y.toFixed(1), o: o.y.toFixed(2), r: (p.y/o.y).toFixed(1)+'\\u00d7'};
    }),
  });

  /* ---- 7. surge front divergence ---- */
  const st = DATA.vstats.stats;
  const sk = Object.keys(st);
  figure('fig-front', {
    title: 'Surge-front position over time \\u2014 do the two solvers agree?',
    note: '99.5th percentile of particle x, i.e. the leading edge of the wave, in the dam-break scene above. The curves track each other to within a few millimetres until the front reaches the far wall.',
    legendItems: [{name: 'Newton', color: S1()}, {name: 'OmniSurg', color: S3()}],
    build: (p) => lineChart(p, {
      height: 300, unit: 'm', xLabel: 'frame', yLabel: 'surge front x (m)',
      xFmt: (v)=>String(v), yFmt: (v)=>v.toFixed(2),
      ariaLabel: 'Surge front position versus frame for both implementations',
      series: [
        {name: 'Newton', short: 'Newton', color: S1(), points: st[sk[0]].map(r => ({x: r.frame, y: r.front_x}))},
        {name: 'OmniSurg', short: 'OmniSurg', color: S3(), points: st[sk[1]].map(r => ({x: r.frame, y: r.front_x}))},
      ],
    }),
    tableCols: [{name:'Frame', key:'f', num:true}, {name:'Newton front x (m)', key:'a', num:true},
                {name:'OmniSurg front x (m)', key:'b', num:true}, {name:'Newton mean speed (m/s)', key:'c', num:true},
                {name:'OmniSurg mean speed (m/s)', key:'d', num:true}],
    tableRows: st[sk[0]].map((r,i) => ({f: r.frame, a: r.front_x.toFixed(4), b: st[sk[1]][i].front_x.toFixed(4),
                                        c: r.speed_mean.toFixed(3), d: st[sk[1]][i].speed_mean.toFixed(3)})),
  });
}

/* headline tiles */
const sc = DATA.scaling;
const last = (a) => a[a.length-1].y;
document.getElementById('t1').innerHTML =
  (last(sc['newton'])/last(sc['omnisurg:all'])).toFixed(1) + '<small>\\u00d7 slower</small>';
const sl = (pts) => { const xs=pts.map(p=>p.x), ys=pts.map(p=>p.y), n=xs.length,
  mx=xs.reduce((a,b)=>a+b)/n, my=ys.reduce((a,b)=>a+b)/n;
  return xs.reduce((a,x,i)=>a+(x-mx)*(ys[i]-my),0)/xs.reduce((a,x)=>a+(x-mx)**2,0); };
document.getElementById('t2').innerHTML =
  (sl(DATA.iters['newton'])/sl(DATA.iters['omnisurg:all'])).toFixed(0) + '<small>\\u00d7 steeper</small>';
document.getElementById('t3').innerHTML =
  (DATA.abl3.newton_ms / DATA.abl3.modes[0].ms).toFixed(1) + '<small>\\u00d7 of the ' +
  (DATA.abl3.newton_ms / DATA.abl3.modes.find(m=>m.mode==='all').ms).toFixed(1) + '\\u00d7 gap</small>';
const nbd = DATA.breakdown['newton'].values;
const ntot = Object.values(nbd).reduce((a,b)=>a+b,0);
document.getElementById('t4').innerHTML = (100*nbd.density/ntot).toFixed(0) + '<small>%</small>';

buildAll();
let rt; addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(buildAll, 180); });

const btn = document.getElementById('themeBtn');
btn.addEventListener('click', () => {
  const dark = document.documentElement.getAttribute('data-theme') === 'dark'
    || (!document.documentElement.getAttribute('data-theme') && matchMedia('(prefers-color-scheme: dark)').matches);
  document.documentElement.setAttribute('data-theme', dark ? 'light' : 'dark');
  buildAll();
});
"""

abl_rel = abl3["modes"][0]["ms"] / next(m["ms"] for m in abl3["modes"] if m["mode"] == "all")

html = (
    HTML.replace("__PAYLOAD__", payload)
    .replace("__CHARTS__", charts_js)
    .replace("__BUILD__", BUILD)
    .replace("__NEWTON_SHA__", NEWTON_SHA)
    .replace("__OMNI_SHA__", OMNI_SHA)
    .replace("__DRIVER__", DRIVER)
    .replace("__GPU__", GPU)
    .replace("__IMGSRC__", img_src)
    .replace("__SLOPE_N__", f"{slope_newton:.1f}")
    .replace("__SLOPE_O__", f"{slope_omni:.2f}")
    .replace("__ABL_REL__", f"{abl_rel:.2f}")
    .replace("__N131_FPS__", f"{1000.0 / n131['newton']:.1f}")
    .replace("__O131_FPS__", f"{1000.0 / n131['omnisurg:all']:.0f}")
    .replace("__VN__", f"{vstats['meta'][next(iter(vstats['meta']))]['n']:,}")
)

out = (cli.out_dir / "index.html") if cli.out_dir else (HERE / "fluid_report.html")
out.write_text(html)
print(f"wrote {out}  ({len(html) / 1024:.0f} KB)")
print(f"  newton@131k {n131['newton']:.1f} ms | omnisurg-all {n131['omnisurg:all']:.1f} ms"
      f" | ratio {n131['newton'] / n131['omnisurg:all']:.1f}x")
print(f"  slopes: newton {slope_newton:.2f} ms/iter, omnisurg {slope_omni:.3f} ms/iter"
      f" -> {slope_newton / slope_omni:.0f}x")
