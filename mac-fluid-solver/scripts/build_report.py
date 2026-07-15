"""Build the self-contained HTML report from the collected data.

Run:  python3 build_report.py --report-dir <dir>
"""

from __future__ import annotations

import argparse
import datetime
import json
import os


def load(path):
    with open(path) as f:
        return json.load(f)


ARCH_SVG = """
<svg viewBox="0 0 900 360" xmlns="http://www.w3.org/2000/svg" font-family="Inter, Arial, sans-serif" font-size="14">
  <defs>
    <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#5e6b76"/>
    </marker>
  </defs>
  <rect x="20" y="20" width="860" height="70" rx="10" fill="#f1f5f8" stroke="#d8e0e7"/>
  <text x="450" y="45" text-anchor="middle" font-weight="700">newton.Model + State (shared)</text>
  <text x="450" y="68" text-anchor="middle" fill="#5e6b76" font-size="13">bodies, joints, shapes, controls — one model, per-entry ModelViews</text>

  <rect x="60" y="140" width="280" height="120" rx="10" fill="#e8f3ee" stroke="#178a64" stroke-width="1.5"/>
  <text x="200" y="168" text-anchor="middle" font-weight="700">SolverMuJoCo</text>
  <text x="200" y="190" text-anchor="middle" font-size="13" fill="#3c4a55">owns bodies, joints, actuators</text>
  <text x="200" y="210" text-anchor="middle" font-size="13" fill="#3c4a55">integrates articulation dynamics</text>
  <text x="200" y="230" text-anchor="middle" font-size="13" fill="#3c4a55">(source entry, substepped)</text>

  <rect x="560" y="140" width="280" height="120" rx="10" fill="#e9f0fa" stroke="#2c6bc4" stroke-width="1.5"/>
  <text x="700" y="168" text-anchor="middle" font-weight="700">SolverMACFluid</text>
  <text x="700" y="190" text-anchor="middle" font-size="13" fill="#3c4a55">owns the fluid grid only</text>
  <text x="700" y="210" text-anchor="middle" font-size="13" fill="#3c4a55">bodies = moving immersed boundaries</text>
  <text x="700" y="230" text-anchor="middle" font-size="13" fill="#3c4a55">(destination entry, in-place)</text>

  <rect x="330" y="290" width="240" height="52" rx="10" fill="#fdf6ec" stroke="#c8871f" stroke-width="1.5"/>
  <text x="450" y="312" text-anchor="middle" font-weight="700" font-size="13">SolverCoupledProxy</text>
  <text x="450" y="330" text-anchor="middle" font-size="12" fill="#5e6b76">staggered proxy coupling, N passes/step</text>

  <line x1="200" y1="140" x2="200" y2="90" stroke="#5e6b76" stroke-width="1.4" marker-end="url(#arr)" marker-start="url(#arr)"/>
  <line x1="700" y1="140" x2="700" y2="90" stroke="#5e6b76" stroke-width="1.4" marker-end="url(#arr)" marker-start="url(#arr)"/>

  <path d="M 340 175 L 560 175" stroke="#178a64" stroke-width="2" marker-end="url(#arr)"/>
  <text x="450" y="166" text-anchor="middle" fill="#178a64" font-size="13" font-weight="600">body poses q, twists u (proxy sync)</text>
  <path d="M 560 235 L 340 235" stroke="#2c6bc4" stroke-width="2" marker-end="url(#arr)"/>
  <text x="450" y="256" text-anchor="middle" fill="#2c6bc4" font-size="13" font-weight="600">hydrodynamic wrenches (impulse / dt)</text>

  <line x1="285" y1="260" x2="360" y2="292" stroke="#c8871f" stroke-width="1.2" stroke-dasharray="4 3"/>
  <line x1="615" y1="260" x2="540" y2="292" stroke="#c8871f" stroke-width="1.2" stroke-dasharray="4 3"/>
</svg>
"""

MAC_SVG = """
<svg viewBox="0 0 460 300" xmlns="http://www.w3.org/2000/svg" font-family="Inter, Arial, sans-serif" font-size="13">
  <g stroke="#c6d2da" stroke-width="1">
    <rect x="60" y="40" width="160" height="160" fill="#f7f9fb"/>
    <rect x="220" y="40" width="160" height="160" fill="#f7f9fb"/>
    <rect x="60" y="200" width="160" height="60" fill="#eceff2"/>
    <rect x="220" y="200" width="160" height="60" fill="#eceff2"/>
  </g>
  <circle cx="140" cy="120" r="5" fill="#c8871f"/>
  <circle cx="300" cy="120" r="5" fill="#c8871f"/>
  <text x="140" y="105" text-anchor="middle" fill="#c8871f" font-weight="600">p(i,j,k)</text>
  <text x="300" y="105" text-anchor="middle" fill="#c8871f" font-weight="600">p(i+1,j,k)</text>
  <g fill="#2c6bc4">
    <polygon points="220,112 236,120 220,128"/>
    <polygon points="60,112 76,120 60,128"/>
    <polygon points="380,112 396,120 380,128"/>
  </g>
  <text x="228" y="145" text-anchor="middle" fill="#2c6bc4" font-weight="600">u(i+1,j,k)</text>
  <text x="68" y="145" text-anchor="middle" fill="#2c6bc4" font-weight="600">u(i,j,k)</text>
  <g fill="#178a64">
    <polygon points="132,40 140,24 148,40"/>
    <polygon points="292,40 300,24 308,40"/>
  </g>
  <text x="140" y="18" text-anchor="middle" fill="#178a64" font-weight="600">w(i,j,k+1)</text>
  <text x="326" y="18" text-anchor="middle" fill="#178a64" font-weight="600">w(i+1,j,k+1)</text>
  <text x="140" y="235" text-anchor="middle" fill="#5e6b76" font-size="12">solid cell (label = body id)</text>
  <text x="300" y="235" text-anchor="middle" fill="#5e6b76" font-size="12">constrained faces = body velocity</text>
  <text x="230" y="285" text-anchor="middle" fill="#5e6b76" font-size="12">x–z slice: pressure at centers, velocity components on faces</text>
</svg>
"""


def fmt(x, digits=3):
    return f"{x:.{digits}g}"


def build(report_dir):
    data = os.path.join(report_dir, "data")
    summary = load(os.path.join(data, "summary.json"))
    val = load(os.path.join(data, "validation.json"))
    bench = load(os.path.join(data, "benchmarks.json"))

    sp, pa, sw = summary["sphere"], summary["paddle"], summary["swimmer"]
    sp_arr = sp.get("max_action_reaction_error", 0.0)
    sp_gross = sp.get("gross_boundary_impulse_Ns", 1.0)
    sp_rel = sp.get("action_reaction_rel", 0.0)
    stamp = datetime.date.today().isoformat()

    bench_rows = ""
    for r in bench:
        if "stage_ms" in r:
            continue
        mode = "CUDA graph" if r["capture"] else ("eager" if r["device"].startswith("cuda") else "CPU")
        dev = "NVIDIA L40" if r["device"].startswith("cuda") else "CPU (single core)"
        bench_rows += (
            f"<tr><td>{r['resolution']}³</td><td>{r['cells']:,}</td><td>{dev}</td>"
            f"<td>{mode}</td><td>{r['ms_per_step']:.1f}</td>"
            f"<td>{1000.0 / r['ms_per_step']:.0f}</td></tr>\n"
        )

    stage = next(r for r in bench if "stage_ms" in r)
    stage_rows = "".join(
        f"<tr><td>{k}</td><td>{v:.2f}</td></tr>\n" for k, v in sorted(stage["stage_ms"].items(), key=lambda kv: -kv[1])
    )

    buoy_rows = ""
    for res, b in val["buoyancy_convergence"].items():
        buoy_rows += (
            f"<tr><td>{res}³</td><td>{b['wrench_z']:.1f}</td><td>{b['voxel_expected']:.1f}</td>"
            f"<td>{b['analytic']:.1f}</td><td>{100 * b['rel_error_analytic']:+.1f}%</td></tr>\n"
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SolverMACFluid — Two-Way Coupled Incompressible Fluid for Newton</title>
  <link rel="icon" href="/favicon.svg" type="image/svg+xml">
  <style>
    :root {{ color-scheme: light; --ink:#18212a; --muted:#5e6b76; --line:#d8e0e7; --panel:#f7f9fb; --accent:#2c6bc4; --warn:#b5483a; --ok:#178a64; }}
    body {{ margin:0; font-family: Inter, "Segoe UI", Arial, sans-serif; color: var(--ink); background:#fff; line-height:1.5; }}
    main {{ max-width: 1080px; margin: 0 auto; padding: 42px 32px 72px; }}
    h1 {{ font-size: 33px; line-height: 1.12; margin: 0 0 10px; }}
    h2 {{ margin: 40px 0 12px; font-size: 22px; border-bottom: 1px solid var(--line); padding-bottom: 6px; }}
    h3 {{ margin: 24px 0 8px; font-size: 17px; }}
    p {{ margin: 10px 0; }}
    code {{ background:#eef3f6; padding:1px 5px; border-radius:4px; font-size: 0.92em; }}
    pre {{ background:#eef3f6; padding:12px 14px; border-radius:8px; overflow-x:auto; font-size:13px; }}
    .lede {{ color: var(--muted); font-size: 17px; max-width: 880px; }}
    .stamp {{ color: var(--muted); font-size: 13px; margin-bottom: 28px; }}
    .grid-cards {{ display:grid; grid-template-columns: repeat(4, 1fr); gap:12px; margin: 18px 0 8px; }}
    .card {{ border:1px solid var(--line); border-radius:8px; padding:14px; background:#fff; }}
    .metric {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}
    .label {{ color: var(--muted); font-size: 13px; }}
    table {{ width:100%; border-collapse: collapse; margin: 16px 0 22px; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background:#f1f5f8; font-weight: 650; }}
    .callout {{ border:1px solid var(--line); border-left:5px solid var(--accent); background:var(--panel); padding:14px 16px; border-radius:8px; margin:20px 0; }}
    .callout.warn {{ border-left-color: var(--warn); }}
    .video-pair {{ display:grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 18px 0; }}
    video {{ width:100%; border:1px solid var(--line); border-radius:8px; background:#0f1720; }}
    img.plot {{ width:100%; max-width: 1000px; border:1px solid var(--line); border-radius:8px; margin: 10px 0; }}
    .caption {{ color: var(--muted); font-size: 13px; margin: 4px 0 14px; }}
    .ok {{ color: var(--ok); font-weight: 600; }}
    .no {{ color: var(--warn); font-weight: 600; }}
    figure {{ margin: 18px 0; }}
    ul {{ margin: 8px 0 8px 22px; padding: 0; }}
    li {{ margin: 4px 0; }}
  </style>
</head>
<body>
<main>
  <h1>SolverMACFluid: Two-Way Coupled Incompressible Fluid for Newton</h1>
  <p class="lede">
    A 3D incompressible Newtonian fluid solver on a dense staggered MAC grid, built for two-way
    coupling with articulated rigid-body solvers through Newton's experimental coupled-solver
    framework. MuJoCo owns the bodies, joints, and actuators; the fluid solver treats them as moving
    immersed boundaries and feeds hydrodynamic wrenches back — buoyancy, drag, reaction torque, and
    swimming propulsion all emerge from the coupling.
  </p>
  <p class="stamp">Eric Heiden · {stamp} · branch <code>eric-heiden/mac-fluid-solver</code> of
    <a href="https://github.com/eric-heiden/newton">eric-heiden/newton</a> · NVIDIA L40, Warp 1.15.0.dev</p>

  <div class="grid-cards">
    <div class="card"><div class="label">Divergence reduction (projection)</div>
      <div class="metric">{val["projection"]["reduction_factor"]:.1e}×</div></div>
    <div class="card"><div class="label">Action–reaction error (relative)</div>
      <div class="metric">{val["momentum_balance"]["relative"]:.1e}</div></div>
    <div class="card"><div class="label">Coupled-restart state restore</div>
      <div class="metric">{"bit-exact" if val["coupled_restart_restore"]["bit_exact"] else "FAILED"}</div></div>
    <div class="card"><div class="label">Step time, 48³ (CUDA graph)</div>
      <div class="metric">{next(r["ms_per_step"] for r in bench if r["resolution"] == 48 and r["capture"]):.1f} ms</div></div>
  </div>

  <h2>Feature overview</h2>
  <table>
    <tr><th>Implemented and tested</th><th>Out of scope / future work</th></tr>
    <tr><td>
      <ul>
        <li>3D incompressible Newtonian fluid, closed (sealed) domains</li>
        <li>Staggered MAC grid: pressure at cell centers, velocity on faces</li>
        <li>Semi-Lagrangian RK2 advection, trilinear MAC interpolation</li>
        <li>Explicit viscosity with stability check</li>
        <li>Gravity + uniform external acceleration</li>
        <li>Matrix-free Jacobi-preconditioned CG pressure projection, fixed iteration count, zero host sync</li>
        <li>Closed-domain null-space (compatibility) handling</li>
        <li>Moving rigid boundaries from analytic shape SDFs (sphere, box, capsule, cylinder, cone, ellipsoid) and triangle meshes</li>
        <li>Approximate no-slip (binary voxelized boundary, O(dx))</li>
        <li>Solver-native hydrodynamic wrench collection (pressure + viscous impulses per body)</li>
        <li>Two-way coupling via <code>SolverCoupledProxy</code> (staggered mode) with MuJoCo</li>
        <li>Fluid-state restoration on coupling-iteration restarts (bit-exact)</li>
        <li>GPU + CPU execution, CUDA-graph capture, deterministic reductions</li>
        <li>Diagnostics: divergence, residual, no-slip error, per-body wrench, momentum balance, stage timings</li>
      </ul>
    </td><td>
      <ul>
        <li>Free surfaces, multiphase flow, inflow/outflow</li>
        <li>Sparse or adaptive grids</li>
        <li>Turbulence models</li>
        <li>Differentiability</li>
        <li>Cut-cell / variational boundary fractions (forces converge first order)</li>
        <li>Implicit viscosity</li>
        <li>Strong (added-mass-stable) coupling — light thin bodies need dense bodies or feedback relaxation</li>
        <li>MacCormack / FLIP advection</li>
        <li>Multigrid pressure preconditioning</li>
      </ul>
    </td></tr>
  </table>

  <h2>MAC grid layout</h2>
  <p>
    The fluid occupies a closed axis-aligned box of uniform cells. Pressure (and cell labels:
    fluid / static solid / rigid-body index) is stored at cell centers; the x, y, z velocity
    components are stored on the corresponding cell faces. The staggering makes the discrete
    divergence and pressure gradient adjoint, so the projection removes divergence without
    checkerboard modes. Rigid bodies are voxelized each step by sampling the union signed-distance
    field of their collision shapes; every face adjacent to a solid cell is constrained to the
    rigid-body velocity at that face position.
  </p>
  <figure>{MAC_SVG}
  <figcaption class="caption">MAC staggering in a x–z slice. Faces bordering solid cells (dark) are constrained to the body's velocity; pressure sees them as Neumann boundaries.</figcaption></figure>

  <h2>Architecture: coupling with MuJoCo</h2>
  <figure>{ARCH_SVG}
  <figcaption class="caption">One shared <code>newton.Model</code>; per-entry <code>ModelView</code>s scope each solver to what it owns. The fluid never integrates rigid state.</figcaption></figure>

  <h3>Pose transfer and wrench feedback</h3>
  <p>Each coupled step, <code>SolverCoupledProxy</code> runs one or more coupling passes:</p>
  <ol>
    <li>MuJoCo steps the articulation (substepped), including last pass's fluid wrenches as external body forces.</li>
    <li>The coupler syncs the resulting body poses <code>body_q</code> and spatial velocities <code>body_qd</code> onto the proxy bodies in the fluid's model view.</li>
    <li>The fluid rasterizes the proxies into its grid, advects, applies forces and viscosity, enforces the boundary velocities, and projects pressure.</li>
    <li><code>coupling_harvest_proxy_wrenches</code> converts the fluid's accumulated per-body boundary impulses to wrenches (impulse/dt) — <em>solver-native</em> collection, not rigid-momentum differencing.</li>
    <li>On multi-pass steps, the coupler redistributes the beginning-of-step state; the fluid restores its checkpointed velocity grid (<code>iteration_restart</code>), so coupling iterations never advance the fluid extra physical time (verified bit-exact).</li>
  </ol>
  <p>
    The per-body wrench is assembled from two discrete momentum exchanges that are applied
    equal-and-opposite to the fluid interior: the <b>pressure surface impulse</b>
    ρ·q·A·n on every fluid/solid interface face (buoyancy, form drag, added-mass reaction), and
    the <b>viscous exchange impulse</b> where the diffusion stencil couples fluid faces to
    constrained faces (skin friction). Discrete action–reaction therefore holds to float32
    roundoff: measured max error {val["momentum_balance"]["error_Ns"]:.1e} N·s against boundary
    impulses of {val["momentum_balance"]["impulse_scale_Ns"]:.1f} N·s
    (relative {val["momentum_balance"]["relative"]:.1e}).
  </p>
  <div class="callout">
    <b>Coupling mode.</b> The examples use the framework's <em>staggered</em> proxy mode. The
    generic free-body velocity rewind of <em>lagged</em> mode subtracts a fictitious center-of-mass
    velocity change that a joint-constrained body never had, which injects spurious boundary
    velocity and destabilizes the loop; staggered mode simply enforces the post-step body state.
    The usual weak-coupling limit applies either way: a body is stable when its inertia exceeds the
    hydrodynamic added inertia of its immersed surface (the paddle and swimmer use dense links for
    this reason; a rigid lid case additionally uses Aitken feedback relaxation).
  </div>

  <h2>Validation</h2>
  <table>
    <tr><th>Test</th><th>Metric</th><th>Result</th></tr>
    <tr><td>MAC interpolation</td><td>trilinear reproduction of a linear velocity field</td><td class="ok">exact to 1e-5 (float32)</td></tr>
    <tr><td>Pressure projection</td><td>RMS divergence of a random field, before → after</td>
        <td class="ok">{val["projection"]["div_l2_pre"]:.1f} → {val["projection"]["div_l2_post"]:.1e} s⁻¹ ({val["projection"]["reduction_factor"]:.1e}×)</td></tr>
    <tr><td>Closed-domain null space</td><td>hydrostatic tank: max residual velocity; ∂p/∂z vs ρg</td>
        <td class="ok">{val["hydrostatics"]["max_velocity"]:.1e} m/s; gradient error {val["hydrostatics"]["gradient_rel_error"]:.1e}</td></tr>
    <tr><td>Viscosity</td><td>diffusion operator vs 7-point reference; sinusoid decay rate</td><td class="ok">matches to 1e-6</td></tr>
    <tr><td>Momentum balance</td><td>fluid ΔP vs external + boundary impulses</td>
        <td class="ok">{val["momentum_balance"]["relative"]:.1e} relative</td></tr>
    <tr><td>Coupled-iteration restore</td><td>repeat of the same interval after <code>iteration_restart</code></td>
        <td class="ok">bit-exact (deterministic reductions)</td></tr>
    <tr><td>CPU / CUDA consistency</td><td>max velocity-field difference after 3 steps</td>
        <td class="ok">{val["cpu_gpu_consistency"]["max_velocity_diff"]:.1e} m/s</td></tr>
    <tr><td>CUDA graph capture</td><td>capture one step, replay 5×</td><td class="ok">passes (finite fields, buoyancy preserved)</td></tr>
  </table>

  <h3>Buoyancy convergence (stationary sphere, r = 0.2 m)</h3>
  <p>In discrete hydrostatic equilibrium the pressure surface force on the voxelized body is
     compared to the analytic Archimedes force ρ V g. The error is dominated by the O(dx) binary
     voxelization and converges first order:</p>
  <table>
    <tr><th>Grid</th><th>Measured F<sub>z</sub> [N]</th><th>Voxel-volume ρVg [N]</th><th>Analytic ρVg [N]</th><th>Error vs analytic</th></tr>
    {buoy_rows}
  </table>

  <h2>Example 1 — Settling and rising sphere</h2>
  <p>
    A sphere (r = 0.12 m) in a 1 m³ sealed water tank. At ρ = 1500 kg/m³ it settles: it accelerates,
    approaches terminal velocity (peak {sp["settle_peak_speed"]:.2f} m/s), and lands on the tank
    floor, where the steady fluid force is {sp["steady_buoyancy_N"]:.0f} N (a floor-seated sphere
    carries less than the free-buoyancy {sp["analytic_buoyancy_N"]:.0f} N since no fluid pushes from
    below). At ρ = 500 kg/m³ it rises and comes to rest against a rigid lid at z = {sp["rise_final_z"]:.2f} m,
    with the fluid supporting its weight. The dry (no fluid) run free-falls.
  </p>
  <div class="video-pair">
    <div><video src="media/sphere_settling.mp4" controls loop muted playsinline></video>
      <p class="caption">Settling sphere (ρ = 1500). The slice shows velocity magnitude; red = fast.</p></div>
    <div><video src="media/sphere_rising.mp4" controls loop muted playsinline></video>
      <p class="caption">Rising sphere (ρ = 500) coming to rest under the lid.</p></div>
  </div>
  <img class="plot" src="media/plot_sphere.png" alt="sphere plots">
  <p class="caption">Sphere height, velocity, hydrodynamic force, and fluid diagnostics. The
    divergence drops ~3 orders of magnitude through each projection. The action–reaction error
    ({sp_arr:.2f} N·s) is float32 reduction noise relative to the ~{sp_gross:.0f} N·s gross
    hydrostatic boundary impulse exchanged per step (relative {sp_rel:.0e}).</p>

  <h2>Example 2 — Motor-driven paddle</h2>
  <p>
    A dense two-blade paddle (0.64 × 0.05 × 0.2 m) on a revolute joint with a MuJoCo velocity servo
    (target {pa["omega_target"]:g} rad/s, gain 8 N·m·s). Dry, the servo reaches
    {pa["omega_dry"]:.2f} rad/s. Coupled, the fluid reaction torque (mean
    {pa["tau_fluid"]:.1f} N·m) loads the motor down to {pa["omega_wet"]:.2f} rad/s —
    a {100 * (1 - pa["omega_wet"] / pa["omega_dry"]):.0f}% speed reduction under fluid load — while
    the blade sheds a rotating wake.
  </p>
  <div class="video-pair">
    <div><video src="media/paddle_wet.mp4" controls loop muted playsinline></video>
      <p class="caption">Coupled paddle stirring the tank (horizontal velocity slice at blade height).</p></div>
    <div><img class="plot" src="media/plot_paddle.png" alt="paddle plots" style="margin:0">
      <p class="caption">Actuator speed dry vs. wet, reaction torque, and fluid diagnostics.</p></div>
  </div>

  <h2>Example 3 — Articulated underwater swimmer</h2>
  <p>
    A five-link swimmer driven by sinusoidal joint-position targets with a phase lag per joint
    (traveling wave). Gravity is off, so all net motion must come from fluid interaction. Over
    {sw["duration"]:.0f} s the forward wave produces {sw["dx_forward"]:+.2f} m of travel
    (cruise ≈ {sw["cruise_speed"]:.2f} m/s), the reversed wave {sw["dx_reverse"]:+.2f} m — the
    swimming direction follows the wave direction — while the dry run drifts only
    {sw["dx_dry"]:+.2f} m (rigid-solver numerical drift, ~{abs(sw["dx_forward"] / max(abs(sw["dx_dry"]), 1e-9)):.0f}×
    smaller than the propulsion). The tank walls are fluid boundaries only: near the end of the run
    the head links pass out of the fluid domain, where their hydrodynamic force is exactly zero and
    they coast at constant momentum (gravity is off) while the still-submerged links keep thrusting.
  </p>
  <div class="video-pair">
    <div><video src="media/swimmer_forward.mp4" controls loop muted playsinline></video>
      <p class="caption">Forward gait: the traveling wave sheds a coherent wake and drives the swimmer forward.</p></div>
    <div><video src="media/swimmer_reverse.mp4" controls loop muted playsinline></video>
      <p class="caption">Reversed wave: the same articulation swims the other way.</p></div>
  </div>
  <img class="plot" src="media/plot_swimmer.png" alt="swimmer plots">
  <p class="caption">COM displacement for forward / reversed / dry gaits, swimming speed, and
    per-link hydrodynamic force magnitudes.</p>

  <h2>Performance</h2>
  <p>
    Cubic tank with one rigid sphere, 120 pressure CG iterations per step, viscosity on, full
    diagnostics on. The pressure solve dominates; at small grids the step is launch-overhead-bound,
    so CUDA graph capture gives up to 7× (the whole coupled step, including MuJoCo, is captured in
    the examples).
  </p>
  <table>
    <tr><th>Grid</th><th>Cells</th><th>Device</th><th>Mode</th><th>ms / step</th><th>steps / s</th></tr>
    {bench_rows}
  </table>
  <img class="plot" src="media/plot_benchmarks.png" alt="benchmark plots">
  <table style="max-width:460px">
    <tr><th>Stage (64³, eager)</th><th>ms / step</th></tr>
    {stage_rows}
  </table>

  <h2>API example</h2>
  <pre><code>import newton
from newton.solvers import SolverMACFluid, SolverMuJoCo
from newton.solvers.experimental.coupled import SolverCoupledProxy

fluid_cfg = SolverMACFluid.Config(
    resolution=(48, 48, 48), cell_size=1.0 / 48.0, origin=(-0.5, -0.5, 0.0),
    density=1000.0, kinematic_viscosity=1.0e-4, pressure_iterations=120,
)

solver = SolverCoupledProxy(
    model=model,
    entries=[
        SolverCoupledProxy.Entry(name="mjc", solver=lambda v: SolverMuJoCo(model=v),
                                 bodies=rigid_bodies, joints=joints, substeps=4),
        SolverCoupledProxy.Entry(name="fluid", solver=lambda v: SolverMACFluid(v, fluid_cfg),
                                 in_place=True),
    ],
    coupling=SolverCoupledProxy.Config(
        proxies=[SolverCoupledProxy.Proxy(source="mjc", destination="fluid",
                                          bodies=rigid_bodies, mode="staggered",
                                          collision_pipeline=lambda m: None)],
        iterations=1,
    ),
)

solver.step(state, state, control, contacts, dt)          # one coupled frame
fluid = solver.solver("fluid")
print(fluid.read_diagnostics()["body_wrench"])            # hydrodynamic wrench per body</code></pre>

  <h2>Known limitations and next steps</h2>
  <ul>
    <li><b>Binary voxelized boundaries</b> resolve forces to O(dx) (buoyancy +22% at 16³ → +6% at 48³).
        Cut-cell face fractions (Batty et al.) would make forces second-order and no-slip sharper.</li>
    <li><b>Weak coupling stability</b> requires body inertia ≳ hydrodynamic added inertia. Light thin
        plates need denser bodies, Aitken relaxation, or a future strongly-coupled (monolithic
        added-mass) solve.</li>
    <li><b>Sealed domains only</b>: a body breaching the domain boundary violates incompressibility
        (the compatibility projection spreads the error as pressure spikes). Free surfaces would
        require a liquid/air interface model.</li>
    <li><b>Explicit viscosity</b> limits ν·dt/dx² ≤ 1/6; an implicit viscous solve can reuse the CG
        infrastructure.</li>
    <li><b>Pressure solve dominates</b> (68% of the step at 64³); a geometric multigrid
        preconditioner is the natural next optimization.</li>
    <li>The residual hydrostatic solver noise (~3e-4 m/s per step at the CG tolerance floor) causes a
        slow random-walk velocity accumulation in long quiescent runs; tighter tolerance or a warm
        start removes it.</li>
    <li>Disconnected fluid regions share a single null-space correction; per-component handling is
        future work.</li>
  </ul>

  <h2>Deliverables</h2>
  <ul>
    <li>Solver: <code>newton/_src/solvers/mac_fluid/</code> (grid, boundary rasterization, kernels, pressure solver, solver + coupling hooks), exported as <code>newton.solvers.SolverMACFluid</code>.</li>
    <li>Tests: <code>newton/tests/test_solver_mac_fluid.py</code> — 14 test functions × CPU/CUDA, all passing; plus 3 example smoke tests.</li>
    <li>Examples: <code>macfluid_settling_sphere</code>, <code>macfluid_paddle</code>, <code>macfluid_swimmer</code> under <code>newton/examples/multiphysics/</code>.</li>
    <li>Docs: <code>docs/solvers/mac_fluid.rst</code> + solver-index integration.</li>
    <li>This report with raw metrics under <code>data/</code> and generation scripts under <code>scripts/</code>.</li>
  </ul>
</main>
</body>
</html>
"""
    out = os.path.join(report_dir, "index.html")
    with open(out, "w") as f:
        f.write(html)
    print("wrote", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=str, required=True)
    args = parser.parse_args()
    build(args.report_dir)
