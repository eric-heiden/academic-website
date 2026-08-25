/* Figures for "SuperDex vs Newton" — Plotly 2.x */
(function () {
  const INK = '#f2f7fb', MUTED = '#9eb0c2', LINE = '#26384a';
  const C = {
    cyan: '#42d6c6', blue: '#62a8ff', amber: '#ffbd59',
    red: '#ff6b7a', purple: '#b68cff', green: '#61d69b', grey: '#7d90a4',
  };

  const LAYOUT = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: MUTED, family: 'ui-sans-serif, system-ui, sans-serif', size: 12 },
    margin: { l: 66, r: 24, t: 18, b: 52 },
    xaxis: { gridcolor: LINE, zerolinecolor: LINE, linecolor: LINE, tickfont: { color: MUTED } },
    yaxis: { gridcolor: LINE, zerolinecolor: LINE, linecolor: LINE, tickfont: { color: MUTED } },
    legend: { orientation: 'h', y: 1.14, x: 0, font: { color: MUTED, size: 11.5 } },
    hovermode: 'closest',
    hoverlabel: { bgcolor: '#0b1723', bordercolor: LINE, font: { color: INK } },
  };
  const CFG = { displayModeBar: false, responsive: true };

  function L(over) {
    const o = JSON.parse(JSON.stringify(LAYOUT));
    return Object.assign(o, over || {},
      { xaxis: Object.assign({}, o.xaxis, (over || {}).xaxis),
        yaxis: Object.assign({}, o.yaxis, (over || {}).yaxis) });
  }
  function plot(id, data, layout) {
    const el = document.getElementById(id);
    if (el) Plotly.newPlot(el, data, layout, CFG);
  }

  /* ------------------------------------------------ 1. single-scene crossover */
  const N = [1, 8, 27, 64, 125, 216];
  const SDX = [0.0065, 0.0498, 0.1236, 0.4017, 1.0535, 2.3916];
  const XPBD = [0.2005, 0.2308, 0.2804, 0.2799, 0.2825, 0.2863];
  const MJC = [0.2503, 0.5201, 1.0261, 12.1348, null, null];
  const KAM = [0.2545, 2.3268, null, null, null, null];

  plot('fig-crossover', [
    { x: N, y: SDX, name: 'SuperDex Physics (CPU, 1 thread)', type: 'scatter',
      mode: 'lines+markers', line: { color: C.cyan, width: 2.6 }, marker: { size: 8 } },
    { x: N, y: XPBD, name: 'Newton SolverXPBD (GPU)', type: 'scatter',
      mode: 'lines+markers', line: { color: C.blue, width: 2.6 }, marker: { size: 8 } },
    { x: N, y: MJC, name: 'Newton SolverMuJoCo (GPU)', type: 'scatter',
      mode: 'lines+markers', line: { color: C.amber, width: 2.2 }, marker: { size: 7 },
      connectgaps: false },
    { x: N, y: KAM, name: 'Newton SolverKamino (GPU)', type: 'scatter',
      mode: 'lines+markers', line: { color: C.purple, width: 2.2, dash: 'dot' },
      marker: { size: 7 } },
  ], L({
    xaxis: { title: 'rigid bodies in the scene', type: 'log',
      tickmode: 'array', tickvals: N, ticktext: N.map(String) },
    yaxis: { title: 'wall-clock ms per 16.7 ms step', type: 'log' },
    shapes: [{ type: 'line', x0: 48, x1: 48, y0: 0.004, y1: 14, yref: 'y',
      line: { color: C.grey, width: 1.4, dash: 'dash' } }],
    annotations: [{ x: Math.log10(48), y: Math.log10(0.012), text: 'crossover ≈ 48 bodies',
      showarrow: false, font: { color: C.grey, size: 11 }, xanchor: 'left', xshift: 6 }],
  }));

  /* ------------------------------------------------ 2. batch scaling */
  const ENVS = [1, 16, 64, 256, 1024, 4096];
  const NEWTON_TPUT = [4218, 57168, 208149, 626596, 679429, 298295];
  const MJC_TPUT = [1841, 863, null, null, null, null];
  // SuperDex: no batch axis. One scene per process; 8 bodies costs 0.0498 ms.
  const SDX_ONE = 1000 / 0.0498;                       // scenes-steps/s, one core
  const SDX_192 = SDX_ONE * 192;                       // perfect-scaling upper bound, 192 cores
  plot('fig-batch', [
    { x: ENVS, y: NEWTON_TPUT, name: 'Newton SolverXPBD — one GPU (MIG 1g slice)',
      type: 'scatter', mode: 'lines+markers', line: { color: C.blue, width: 2.8 },
      marker: { size: 9 }, connectgaps: false },
    { x: ENVS, y: ENVS.map(() => SDX_ONE), name: 'SuperDex — one scene, one core (measured)',
      type: 'scatter', mode: 'lines', line: { color: C.cyan, width: 2.4 } },
    { x: ENVS, y: ENVS.map(() => SDX_192),
      name: 'SuperDex — 192 cores, perfect scaling (upper bound, not measured)',
      type: 'scatter', mode: 'lines', line: { color: C.cyan, width: 1.8, dash: 'dot' } },
    { x: ENVS, y: MJC_TPUT, name: 'Newton SolverMuJoCo (crashes past 16 worlds here)',
      type: 'scatter', mode: 'lines+markers', line: { color: C.amber, width: 1.8 },
      marker: { size: 7 }, connectgaps: false },
  ], L({
    xaxis: { title: 'parallel environments (8 rigid bodies each)', type: 'log',
      tickmode: 'array', tickvals: ENVS, ticktext: ENVS.map(String) },
    yaxis: { title: 'environment-steps per second', type: 'log',
      tickmode: 'array', tickvals: [1e3, 1e4, 1e5, 1e6, 4e6],
      ticktext: ['1 k', '10 k', '100 k', '1 M', '4 M'] },
    annotations: [{ x: Math.log10(1024), y: Math.log10(679429),
      text: 'peak 679 k — MIG slice saturates', showarrow: true, arrowhead: 0,
      ax: -8, ay: -30, font: { color: MUTED, size: 11 }, arrowcolor: C.grey,
      xanchor: 'right' }],
  }));

  /* ------------------------------------------------ 3. penetration vs load/stiffness */
  const PEN_K = [1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12];
  const PEN_V = [5.0004, 2.4871, 1.1170, 0.3080, -0.1849, -0.4894, -0.6793];
  const RHO = [10, 100, 1000, 5000, 20000];
  const RHO_V = [-0.4894, -0.1849, 0.3080, 0.8290, 1.4544];
  plot('fig-penetration', [
    { x: PEN_K, y: PEN_V, name: 'sweep contact stiffness k  (ρ = 1000 kg/m³)',
      type: 'scatter', mode: 'lines+markers', line: { color: C.cyan, width: 2.6 },
      marker: { size: 8 }, xaxis: 'x', yaxis: 'y' },
    { x: RHO.map(r => 1e9 * 1000 / r), y: RHO_V,
      name: 'sweep density ρ, replotted as equivalent k/load  (k = 1e9)',
      type: 'scatter', mode: 'markers', marker: { size: 13, color: C.amber, symbol: 'x',
        line: { width: 2 } } },
  ], L({
    xaxis: { title: 'contact stiffness ÷ load  (Pa/m per unit density)', type: 'log' },
    yaxis: { title: 'resting penetration (mm)   — negative = floating', zeroline: true,
      zerolinecolor: C.grey, zerolinewidth: 1.5 },
  }));

  /* ------------------------------------------------ 4. dt invariance */
  const DT = [1, 2, 4.167, 8.333, 16.667, 33.333, 50, 100, 200];
  const DTPEN = [0.3081, 0.3078, 0.3080, 0.3080, 0.3080, 0.3081, 0.3082, 0.3079, 0.3081];
  plot('fig-dt', [
    { x: DT, y: DTPEN, name: 'SuperDex resting penetration', type: 'scatter',
      mode: 'lines+markers', line: { color: C.green, width: 2.8 }, marker: { size: 9 } },
  ], L({
    xaxis: { title: 'time step (ms)', type: 'log' },
    yaxis: { title: 'resting penetration (mm)', range: [0.28, 0.34] },
    annotations: [{ x: Math.log10(16.667), y: 0.325,
      text: 'the shipped examples run here (16.7 ms)', showarrow: true, arrowhead: 0,
      ax: 0, ay: -26, font: { color: MUTED, size: 11 }, arrowcolor: C.grey }],
  }));

  /* ------------------------------------------------ 5. thread scaling */
  const T = [1, 3, 5, 9, 17];  // calling thread + N workers
  const TL = ['0', '2', '4', '8', '16'];
  const S1 = [8.06, 8.07, 6.63, 12.71, 21.75];
  const S4 = [34.54, 29.14, 19.77, 27.11, 44.99];
  const S8 = [68.93, 66.58, 43.28, 54.46, 75.16];
  const sp = a => a.map(v => a[0] / v);
  plot('fig-threads', [
    { x: TL, y: sp(S1), name: '1 duck (1 899 tets)', type: 'scatter',
      mode: 'lines+markers', line: { color: C.cyan, width: 2.6 }, marker: { size: 8 } },
    { x: TL, y: sp(S4), name: '4 ducks (7 596 tets)', type: 'scatter',
      mode: 'lines+markers', line: { color: C.blue, width: 2.6 }, marker: { size: 8 } },
    { x: TL, y: sp(S8), name: '8 ducks (15 192 tets)', type: 'scatter',
      mode: 'lines+markers', line: { color: C.purple, width: 2.6 }, marker: { size: 8 } },
    { x: TL, y: [1, 1, 1, 1, 1], name: 'no speed-up', type: 'scatter', mode: 'lines',
      line: { color: C.grey, width: 1.2, dash: 'dash' }, hoverinfo: 'skip' },
  ], L({
    xaxis: { title: 'num_worker_threads  (machine has 192 logical cores)', type: 'category' },
    yaxis: { title: 'speed-up vs single-threaded' },
  }));

  /* ------------------------------------------------ 6. soft-body cost */
  const TETS = [228, 730, 1899, 7596, 15192];
  const MS = [0.567, 2.337, 7.678, 33.767, 68.583];
  plot('fig-soft', [
    { x: TETS, y: MS, name: 'SuperDex FEM soft body (single thread)', type: 'scatter',
      mode: 'lines+markers', line: { color: C.cyan, width: 2.8 }, marker: { size: 9 } },
    { x: [200, 20000], y: [16.667, 16.667], name: 'real time at 60 Hz', type: 'scatter',
      mode: 'lines', line: { color: C.amber, width: 1.6, dash: 'dash' } },
  ], L({
    xaxis: { title: 'tetrahedra in the scene', type: 'log' },
    yaxis: { title: 'wall-clock ms per 16.7 ms step', type: 'log' },
  }));

  /* ------------------------------------------------ 7. architecture radar-ish bars */
  const AXES = ['Rigid bodies', 'Articulations', 'Cloth / shells', 'Rods / cables',
    'FEM soft bodies', 'Granular / fluid', 'Tendons + muscles',
    'Differentiable', 'Batched envs', 'Hard constraints'];
  const SDXS = [3, 3, 3, 3, 3, 0, 3, 2, 0, 0];
  const NEWS = [3, 3, 3, 3, 3, 3, 1, 1, 3, 3];
  const LBL = ['absent', 'partial', 'solid', 'first-class'];
  const bar = (name, vals, col) => ({
    y: AXES, x: vals.map(v => Math.max(v, 0.035)), name, type: 'bar', orientation: 'h',
    marker: { color: vals.map(v => (v === 0 ? C.red : col)) },
    customdata: vals.map(v => LBL[v]),
    hovertemplate: '%{y}: %{customdata}<extra>' + name + '</extra>',
  });
  plot('fig-coverage', [
    bar('SuperDex Physics 1.0.0', SDXS, C.cyan),
    bar('Newton 1.5.0.dev0 (all solvers)', NEWS, C.blue),
  ], L({
    barmode: 'group', bargap: 0.34, bargroupgap: 0.14,
    margin: { l: 148, r: 24, t: 18, b: 52 },
    xaxis: { title: '0 = absent (red) · 1 = partial/experimental · 2 = solid · 3 = first-class',
      range: [0, 3.25], tickmode: 'array', tickvals: [0, 1, 2, 3], ticktext: LBL },
    yaxis: { automargin: true },
    height: 520,
  }));

  /* ------------------------------------------------ copy buttons */
  document.querySelectorAll('.copy').forEach(b => {
    b.addEventListener('click', () => {
      const pre = b.closest('.codeblock').querySelector('pre');
      navigator.clipboard.writeText(pre.innerText);
      const t = b.textContent; b.textContent = 'Copied';
      setTimeout(() => { b.textContent = t; }, 1200);
    });
  });
})();
