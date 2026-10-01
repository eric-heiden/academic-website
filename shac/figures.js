(function () {
  "use strict";
  const root = document.documentElement;
  const css = (token) => getComputedStyle(root).getPropertyValue(token).trim();
  const figures = [];
  let data = null;
  let tuning = null;

  // Series colours are defined in shac.css so they follow the bright/dark theme.
  const CONFIG_STYLE = {
    explicit_2ms: { label: "Explicit Euler, 2 ms", color: "--series-explicit", dash: "dot" },
    implicit_2ms: { label: "Implicit, 2 ms", color: "--series-dt2", dash: "solid" },
    implicit_5ms: { label: "Implicit, 5 ms", color: "--series-dt5", dash: "solid" },
    implicit_10ms: { label: "Implicit, 10 ms", color: "--series-dt10", dash: "solid" },
    implicit_20ms: { label: "Implicit, 20 ms", color: "--series-dt20", dash: "solid" },
    implicit_10ms_long: { label: "Implicit, 10 ms, 4000 epochs", color: "--series-dt10", dash: "dash" }
  };
  const INTEGRATOR_STYLE = {
    euler_explicit: { label: "Euler, explicit damping", color: "--series-explicit", dash: "dot", symbol: "circle-open" },
    euler: { label: "Euler, implicit damping", color: "--series-dt5", dash: "dash", symbol: "square-open" },
    implicitfast: { label: "Implicitfast", color: "--series-dt20", dash: "solid", symbol: "circle" }
  };

  function baseLayout(width, xTitle, yTitle) {
    return {
      autosize: true,
      height: width < 520 ? 260 : 300,
      margin: { l: 62, r: 12, t: 6, b: 48, pad: 0 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: css("--sans"), size: 11, color: css("--muted") },
      showlegend: false,
      hovermode: "closest",
      hoverlabel: { bgcolor: css("--surface"), bordercolor: css("--line-strong"), font: { color: css("--text"), size: 11 } },
      xaxis: { title: { text: xTitle, font: { color: css("--text"), size: 11 } }, gridcolor: css("--line"),
               linecolor: css("--line-strong"), zeroline: false, fixedrange: true },
      yaxis: { title: { text: yTitle, font: { color: css("--text"), size: 11 } }, gridcolor: css("--line"),
               linecolor: css("--line-strong"), zeroline: false, fixedrange: true }
    };
  }

  function withAlpha(color, alpha) {
    return "color-mix(in srgb, " + color + " " + Math.round(alpha * 100) + "%, transparent)";
  }

  function hexToRgba(hex, alpha) {
    const h = hex.replace("#", "");
    if (h.length !== 6) return withAlpha(hex, alpha);
    const n = parseInt(h, 16);
    return "rgba(" + (n >> 16 & 255) + "," + (n >> 8 & 255) + "," + (n & 255) + "," + alpha + ")";
  }

  function draw(fig) {
    if (!data || typeof window.Plotly === "undefined") return;
    const el = fig.element;
    const width = el.clientWidth || 320;
    const spec = fig.build(width);
    window.Plotly.react(el, spec.traces, spec.layout, { responsive: true, displayModeBar: false, displaylogo: false, scrollZoom: false });
  }

  function register(id, build, options) {
    const element = document.getElementById(id);
    if (!element) return;
    const fig = { element: element, build: build, state: Object.assign({}, options || {}) };
    figures.push(fig);
    const panel = element.closest("figure");
    if (panel) {
      panel.querySelectorAll("[data-choice]").forEach(function (button) {
        button.addEventListener("click", function () {
          const key = button.dataset.choiceGroup;
          fig.state[key] = button.dataset.choice;
          panel.querySelectorAll('[data-choice-group="' + key + '"]').forEach(function (b) {
            b.setAttribute("aria-pressed", String(b === button));
          });
          draw(fig);
        });
      });
    }
    new ResizeObserver(function () { if (element.data) window.Plotly.Plots.resize(element); }).observe(element);
    return fig;
  }

  // Figure: one-control-step error vs physics timestep, per integrator.
  const integratorFig = register("fig-integrators", function (width) {
    const robot = integratorFig.state.robot;
    const rows = data.integrator.filter(function (r) { return r.robot === robot; });
    const traces = [];
    Object.keys(INTEGRATOR_STYLE).forEach(function (name) {
      const style = INTEGRATOR_STYLE[name];
      const rs = rows.filter(function (r) { return r.integrator === name; });
      const color = css(style.color);
      traces.push({
        x: rs.map(function (r) { return r.timestep_ms; }),
        y: rs.map(function (r) { return r.joint_deg_median; }),
        type: "scatter", mode: "lines+markers", name: style.label,
        line: { color: color, width: 2, dash: style.dash },
        marker: {
          color: color, size: rs.map(function (r) { return r.diverged > 0.01 ? 12 : 7; }),
          symbol: rs.map(function (r) { return r.diverged > 0.01 ? "x-thin-open" : style.symbol; }),
          line: { color: color, width: 2 }
        },
        text: rs.map(function (r) {
          return r.diverged > 0.01 ? "diverged in " + (100 * r.diverged).toFixed(1) + "% of worlds" : "stable";
        }),
        hovertemplate: style.label + "<br>%{x} ms: %{y:.3g}° (%{text})<extra></extra>"
      });
    });
    const layout = baseLayout(width, "Physics timestep (ms, log scale)", "Median joint-angle error after 20 ms (°)");
    layout.xaxis.type = "log";
    layout.xaxis.tickvals = [1, 2, 4, 5, 10, 20];
    layout.xaxis.ticktext = ["1", "2", "4", "5", "10", "20"];
    layout.yaxis.type = "log";
    layout.yaxis.tickvals = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20];
    layout.yaxis.ticktext = ["0.02", "0.05", "0.1", "0.2", "0.5", "1", "2", "5", "10", "20"];
    return { traces: traces, layout: layout };
  }, { robot: "go1" });

  // Figure: half-batch cosine of the policy gradient vs horizon.
  const gradientFig = register("fig-gradients", function (width) {
    const robot = gradientFig.state.robot;
    const metric = gradientFig.state.metric || "cosine";
    const byCfg = (data.gradient[robot] || {});
    const traces = [];
    Object.keys(CONFIG_STYLE).forEach(function (cfg) {
      const rows = byCfg[cfg];
      if (!rows) return;
      const style = CONFIG_STYLE[cfg];
      const color = css(style.color);
      const y = rows.map(function (r) { return metric === "cosine" ? r.cosine_mean : r.log10_ratio_mean; });
      traces.push({
        x: rows.map(function (r) { return r.horizon; }), y: y,
        type: "scatter", mode: "lines+markers", name: style.label,
        line: { color: color, width: 2, dash: style.dash }, marker: { color: color, size: 6 },
        hovertemplate: style.label + "<br>T = %{x}: %{y:.2f}<extra></extra>"
      });
    });
    const layout = baseLayout(width, "Horizon T (control steps of 20 ms, log scale)",
      metric === "cosine" ? "Cosine between half-batch gradients" : "log₁₀(analytic / finite-difference slope)");
    layout.xaxis.type = "log";
    layout.xaxis.tickvals = [1, 2, 4, 8, 16, 32];
    layout.xaxis.ticktext = ["1", "2", "4", "8", "16", "32"];
    if (metric === "cosine") layout.yaxis.range = [-0.35, 1.05];
    return { traces: traces, layout: layout };
  }, { robot: "ant", metric: "cosine" });

  // Figure: training curves (mean episode forward speed in the training simulator).
  const learningFig = register("fig-learning", function (width) {
    const robot = learningFig.state.robot;
    const axis = learningFig.state.axis || "epochs";
    const metric = learningFig.state.metric || "speed";
    const byCfg = data.learning[robot] || {};
    const traces = [];
    Object.keys(CONFIG_STYLE).forEach(function (cfg) {
      const c = byCfg[cfg];
      if (!c) return;
      if (cfg.endsWith("_long") && axis === "epochs") return;  // only comparable on the wall-clock axis
      if (axis !== "epochs" && !c.epoch_seconds) return;
      const style = CONFIG_STYLE[cfg];
      const color = css(style.color);
      const x = axis === "epochs" ? c.epoch : c.epoch.map(function (e) { return (e + 1) * c.epoch_seconds / 60; });
      const m = c[metric + "_mean"], s = c[metric + "_sd"];
      const upper = m.map(function (v, i) { return v === null ? null : v + s[i]; });
      const lower = m.map(function (v, i) { return v === null ? null : v - s[i]; });
      traces.push({ x: x, y: upper, type: "scatter", mode: "lines", line: { width: 0 }, hoverinfo: "skip", showlegend: false });
      traces.push({ x: x, y: lower, type: "scatter", mode: "lines", line: { width: 0 }, fill: "tonexty",
                    fillcolor: hexToRgba(color, 0.14), hoverinfo: "skip", showlegend: false });
      traces.push({ x: x, y: m, type: "scatter", mode: "lines", name: style.label,
                    line: { color: color, width: 2, dash: style.dash },
                    hovertemplate: style.label + "<br>%{x:.0f}: %{y:.2f}<extra></extra>" });
    });
    const yTitle = metric === "speed" ? "Mean episode forward speed (m/s)" : "Mean episode length (s)";
    const layout = baseLayout(width, axis === "epochs" ? "SHAC epoch" : "Isolated wall-clock time (min)", yTitle);
    return { traces: traces, layout: layout };
  }, { robot: "ant", axis: "epochs", metric: "speed" });

  // Figure: training curves of the retuned tasks (data/tuning.json).
  const TUNE_STYLE = {
    base: { color: "--tune-base", dash: "dot" },
    prior: { color: "--tune-prior", dash: "solid" },
    long: { color: "--tune-long", dash: "solid" },
    alt: { color: "--tune-alt", dash: "dash" },
    ppo: { color: "--tune-ppo", dash: "dashdot" },
    extra: { color: "--tune-extra", dash: "longdash" },
    ppo2: { color: "--tune-ppo", dash: "dot" }
  };
  const tuningFig = register("fig-tuning", function (width) {
    const group = tuningFig.state.group;
    const metric = tuningFig.state.metric || "length";
    const series = (tuning ? tuning.series : []).filter(function (s) { return s.group === group; });
    // Fall back to the environment-step axis when no series of this robot has an isolated epoch time.
    const timed = series.some(function (s) { return s.minutes; });
    const axis = (tuningFig.state.axis || "samples") === "minutes" && timed ? "minutes" : "samples";
    const noTrack = metric === "track" && !series.some(function (s) { return s.track; });
    const traces = [];
    series.forEach(function (s) {
      const st = TUNE_STYLE[s.style];
      const color = css(st.color);
      const x = axis === "samples" ? s.samples_m : s.minutes;
      if (!x || !s[metric]) return;
      const label = s.label + (s.seeds > 1 ? " (" + s.seeds + " seeds)" : "");
      traces.push({ x: x, y: s[metric], type: "scatter", mode: "lines", name: label,
                    line: { color: color, width: 2, dash: st.dash },
                    hovertemplate: label + "<br>" + (axis === "samples" ? "%{x:.2f} M steps" : "%{x:.1f} min") + ": " +
                                   (metric === "length" ? "%{y:.1f} s" : "%{y:.2f} m/s") + "<extra></extra>" });
    });
    const layout = baseLayout(width, axis === "samples" ? "Environment steps (millions, log scale)" : "Training time (min, log scale)",
      metric === "length" ? "Mean training episode length (s)" : "Mean planar tracking error (m/s)");
    layout.xaxis.type = "log";
    layout.xaxis.tickvals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300];
    layout.xaxis.ticktext = ["0.01", "0.03", "0.1", "0.3", "1", "3", "10", "30", "100", "300"];
    if (metric === "length") layout.yaxis.range = [0, 21];
    layout.showlegend = true;
    layout.legend = { orientation: "h", x: 0, xanchor: "left", y: -0.28, yanchor: "top",
                      font: { size: 11, color: css("--text") }, bgcolor: "rgba(0,0,0,0)" };
    layout.height = width < 520 ? 380 : 360;
    layout.margin.b = width < 520 ? 130 : 96;
    const notes = [];
    if (noTrack) notes.push("This task has no velocity command, so there is no tracking error.");
    else if ((tuningFig.state.axis || "samples") === "minutes" && !timed) notes.push("No isolated epoch time for these runs; shown against environment steps.");
    if (notes.length) {
      layout.annotations = [{ text: notes.join(" "), xref: "paper", yref: "paper", x: 0.5, y: 0.5, showarrow: false,
                              font: { size: 12, color: css("--muted") } }];
      if (noTrack) { layout.xaxis.visible = false; layout.yaxis.visible = false; layout.showlegend = false; }
    }
    return { traces: traces, layout: layout };
  }, { group: "h1", metric: "length", axis: "minutes" });

  function drawAll() { figures.forEach(draw); }

  function start() {
    Promise.all([fetch("data/figures.json").then(function (r) { return r.json(); }),
                 fetch("data/tuning.json").then(function (r) { return r.json(); }).catch(function () { return null; })])
      .then(function (both) {
      const json = both[0];
      tuning = both[1];
      data = json;
      if (typeof window.Plotly === "undefined") {
        figures.forEach(function (f) { f.element.textContent = "The interactive plot could not be loaded."; });
        return;
      }
      drawAll();
    }).catch(function () {
      figures.forEach(function (f) { f.element.textContent = "The figure data could not be loaded."; });
    });
  }

  window.addEventListener("report-theme-change", drawAll);
  if (document.readyState === "complete") start();
  else window.addEventListener("load", start);
})();
