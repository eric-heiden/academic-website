(async function () {
  const ratioPlot = document.getElementById("confirmation-ratios");
  const convergencePlot = document.getElementById("calibration-convergence");
  const instance = document.getElementById("calibration-instance");
  const cache = document.getElementById("ratio-uncached");
  if (!ratioPlot || !convergencePlot || !instance || !cache) return;
  try {
    const responses = await Promise.all([
      fetch("data/confirmation/results.json"),
      fetch("data/confirmation/candidate-history.json")
    ]);
    if (responses.some(response => !response.ok)) throw new Error("Confirmation data unavailable");
    const [results, histories] = await Promise.all(responses.map(response => response.json()));
    if (!window.Plotly) throw new Error("Plot library unavailable");
    const title = id => id.replace("panda_calibration-", "Calibration ").replace("panda-", "Panda ").replace("allegro-", "Allegro ").replace("hug-", "HUG ");
    function settings(height, left = 60) {
      const styles = getComputedStyle(document.documentElement);
      const css = name => styles.getPropertyValue(name).trim();
      return {
        css,
        layout: {
          autosize: true, height, margin: {l: left, r: 16, t: 48, b: 55},
          paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
          font: {family: css("--sans"), size: 11, color: css("--muted")},
          legend: {orientation: "h", x: 0, y: 1.16, font: {size: 10, color: css("--text")}},
          hoverlabel: {bgcolor: css("--surface"), bordercolor: css("--line-strong"), font: {color: css("--text")}}
        },
        axis: {gridcolor: css("--line"), linecolor: css("--line-strong"), zeroline: false, fixedrange: true}
      };
    }
    function drawRatios() {
      const pairs = results.pairs.filter(pair => pair.both_eligible_successes && pair.matched_sources_and_references);
      const {css, layout, axis} = settings(440, 94);
      const metrics = [
        ["startup_inclusive_seconds", "Elapsed time", css("--accent"), "circle", -0.12],
        ["input_output_tokens", "Input + output", css("--accent-2"), "square", 0.12]
      ];
      if (cache.checked) metrics.push(["uncached_input_tokens", "Uncached input", css("--text"), "diamond-open", 0]);
      const traces = metrics.map(([key, name, color, symbol, offset]) => ({
        x: pairs.map(pair => pair.ratios_restart_over_live[key]),
        y: pairs.map((_, index) => index + offset),
        text: pairs.map(pair => title(pair.pair_id)),
        name, type: "scatter", mode: "markers", marker: {color, symbol, size: 7},
        hovertemplate: "%{text}: %{x:.3f}×<extra>" + name + "</extra>"
      }));
      Plotly.react(ratioPlot, traces, {
        ...layout,
        xaxis: {...axis, title: {text: "Restart / live (ratio)"}, rangemode: "tozero"},
        yaxis: {...axis, tickmode: "array", tickvals: pairs.map((_, index) => index), ticktext: pairs.map(pair => title(pair.pair_id)), autorange: "reversed", showgrid: false},
        shapes: [{type: "line", x0: 1, x1: 1, yref: "paper", y0: 0, y1: 1, line: {color: css("--muted"), width: 1, dash: "dot"}}]
      }, {responsive: true, displayModeBar: false, displaylogo: false});
      ratioPlot.setAttribute("aria-label", `${pairs.length} eligible matched comparisons. Ratios greater than one favor live; ratios below one favor restart. Elapsed time and input plus output are separate series${cache.checked ? ", with uncached input also shown" : ""}. Exact values are in the result tables.`);
    }
    function drawConvergence() {
      const {css, layout, axis} = settings(320);
      const pair = results.pairs.find(pair => pair.pair_id === "panda_calibration-" + instance.value);
      if (!pair) return;
      const traces = ["live", "restart"].map((condition, index) => {
        const records = histories[pair[condition]].records.filter(record => record.complete);
        return {
          x: records.map(record => record.candidate_index),
          y: records.map(record => record.metrics.trajectory_rmse_rad * 1000),
          text: records.map(record => record.metrics.success ? "All training criteria passed" : "Training criteria not all passed"),
          name: condition === "live" ? "Live" : "Restart", type: "scatter", mode: "lines+markers",
          line: {color: css(index ? "--accent-2" : "--accent"), width: 2, dash: index ? "dash" : "solid"},
          marker: {size: 6, symbol: index ? "square" : "circle"},
          hovertemplate: "Candidate %{x}: %{y:.4f} mrad<br>%{text}<extra>" + condition + "</extra>"
        };
      });
      Plotly.react(convergencePlot, traces, {
        ...layout,
        xaxis: {...axis, title: {text: "Completed candidate (saved order)"}, tickmode: "linear", dtick: 1, range: [0.8, 9.2]},
        yaxis: {...axis, title: {text: "Training RMSE (mrad, log scale)"}, type: "log", range: [-0.8, 1.7]},
        shapes: [{type: "line", xref: "paper", x0: 0, x1: 1, y0: 0.35, y1: 0.35, line: {color: css("--muted"), width: 1, dash: "dot"}}],
        annotations: [{xref: "paper", x: 1, y: Math.log10(0.35), text: "0.35 mrad", showarrow: false, xanchor: "right", yshift: 10, font: {size: 10, color: css("--muted")}}]
      }, {responsive: true, displayModeBar: false, displaylogo: false});
      convergencePlot.setAttribute("aria-label", `Synthetic calibration instance ${instance.value}, actual training RMSE for all nine candidates per condition. Logarithmic vertical axis; the dotted line is the 0.35 mrad pooled RMSE limit. Other criteria also determine success.`);
    }
    cache.addEventListener("change", drawRatios);
    instance.addEventListener("change", drawConvergence);
    window.addEventListener("report-theme-change", () => {drawRatios(); drawConvergence();});
    drawRatios(); drawConvergence();
  } catch (error) {
    ratioPlot.textContent = "Comparison plot unavailable; the tables and downloadable results retain exact measurements.";
    convergencePlot.textContent = "Candidate plot unavailable; the downloadable candidate history retains every measurement.";
    console.error(error);
  }
})();
