(async function () {
  const plot = document.getElementById("systems-plot");
  const select = document.getElementById("systems-task");
  const startup = document.getElementById("systems-startup");
  if (!plot || !select || !startup) return;
  try {
    const response = await fetch("data/systems-timing.json");
    if (!response.ok) throw new Error("Timing data could not be loaded.");
    const measurements = await response.json();
    if (!window.Plotly) throw new Error("Plot library could not be loaded.");
    const cumulative = (values, initial) => values.reduce((result, value) => [...result, result[result.length - 1] + value], [initial]);
    function draw() {
      const value = measurements.cases.find(value => value.scenario === select.value);
      if (!value) return;
      const style = getComputedStyle(document.documentElement);
      const css = name => style.getPropertyValue(name).trim();
      const live = cumulative(value.live_seconds, startup.checked ? value.startup_seconds : 0);
      const restart = cumulative(value.restart_seconds, 0);
      const x = [0, 1, 2, 3, 4, 5];
      window.Plotly.react(plot, [
        { x, y: live, name: startup.checked ? "Live + startup" : "Live requests only", type: "scatter", mode: "lines+markers", line: { color: css("--accent"), width: 2 }, marker: { size: 5 }, hovertemplate: "%{x} evaluations: %{y:.3f} s<extra>Live</extra>" },
        { x, y: restart, name: "Restart", type: "scatter", mode: "lines+markers", line: { color: css("--accent-2"), width: 2, dash: "dash" }, marker: { size: 5, symbol: "square" }, hovertemplate: "%{x} evaluations: %{y:.3f} s<extra>Restart</extra>" }
      ], {
        autosize: true,
        height: 300,
        margin: { l: 57, r: 14, t: 34, b: 52 },
        paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
        font: { family: css("--sans"), size: 11, color: css("--muted") },
        legend: { orientation: "h", x: 0, y: 1.2, font: { size: 11, color: css("--text") } },
        hovermode: "x unified",
        hoverlabel: { bgcolor: css("--surface"), bordercolor: css("--line-strong"), font: { color: css("--text") } },
        xaxis: { title: { text: "Completed scripted evaluations" }, tickmode: "array", tickvals: x, range: [-0.1, 5.1], gridcolor: css("--line"), linecolor: css("--line-strong"), zeroline: false, fixedrange: true },
        yaxis: { title: { text: "Cumulative elapsed time (s)" }, rangemode: "tozero", gridcolor: css("--line"), linecolor: css("--line-strong"), zeroline: false, fixedrange: true },
        uirevision: select.value
      }, { responsive: true, displayModeBar: false, displaylogo: false });
      plot.setAttribute("aria-label", `${select.options[select.selectedIndex].text}: five evaluations take ${live[5].toFixed(3)} seconds with live access ${startup.checked ? "including startup" : "excluding startup"}, and ${restart[5].toFixed(3)} seconds with process restarts. Scripted cost comparison, not successful tuning.`);
    }
    select.addEventListener("change", draw);
    startup.addEventListener("change", draw);
    window.addEventListener("report-theme-change", draw);
    draw();
  } catch (error) {
    plot.textContent = "The interactive timing plot is unavailable. Table 8 contains the measured totals.";
    console.error(error);
  }
})();
