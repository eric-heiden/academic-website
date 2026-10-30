(function () {
  const root = document.documentElement;
  const validTemplates = new Set(["prism", "signal", "aperture"]);
  const picker = document.getElementById("template-picker");
  const themeButton = document.getElementById("theme-toggle");
  const themeIcon = themeButton?.querySelector(".theme-icon");
  const themeLabel = themeButton?.querySelector(".theme-label");

  function updateThemeControl() {
    const dark = root.dataset.theme === "dark";
    if (!themeButton) return;
    themeButton.setAttribute("aria-pressed", String(dark));
    themeButton.setAttribute("aria-label", dark ? "Switch to bright theme" : "Switch to dark theme");
    if (themeIcon) themeIcon.textContent = dark ? "☾" : "☀";
    if (themeLabel) themeLabel.textContent = dark ? "Dark" : "Bright";
  }

  updateThemeControl();
  themeButton?.addEventListener("click", function () {
    root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
    try { localStorage.setItem("report-theme", root.dataset.theme); } catch (_) {}
    updateThemeControl();
    window.dispatchEvent(new CustomEvent("report-theme-change"));
  });

  if (picker) {
    picker.value = validTemplates.has(root.dataset.template) ? root.dataset.template : "aperture";
    picker.addEventListener("change", function () {
      if (!validTemplates.has(picker.value)) return;
      root.dataset.template = picker.value;
      const url = new URL(window.location.href);
      url.searchParams.set("template", picker.value);
      history.replaceState({}, "", url);
      window.dispatchEvent(new CustomEvent("report-theme-change"));
    });
  }

  document.querySelectorAll(".copy-button").forEach(function (button) {
    button.addEventListener("click", async function () {
      const target = document.getElementById(button.dataset.copyTarget);
      if (!target) return;
      const text = target.innerText;
      try {
        await navigator.clipboard.writeText(text);
      } catch (_) {
        const textarea = document.createElement("textarea");
        textarea.value = text;
        textarea.style.position = "fixed";
        textarea.style.opacity = "0";
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand("copy");
        textarea.remove();
      }
      const original = button.textContent;
      button.textContent = "Copied";
      setTimeout(function () { button.textContent = original; }, 1400);
    });
  });

  const sectionLinks = Array.from(document.querySelectorAll("[data-section-link]"));
  const sections = sectionLinks
    .map(function (link) { return document.querySelector(link.getAttribute("href")); })
    .filter(Boolean);

  if ("IntersectionObserver" in window && sections.length) {
    const observer = new IntersectionObserver(function (entries) {
      const visible = entries
        .filter(function (entry) { return entry.isIntersecting; })
        .sort(function (a, b) { return b.intersectionRatio - a.intersectionRatio; });
      if (!visible.length) return;
      sectionLinks.forEach(function (link) {
        const active = link.getAttribute("href") === "#" + visible[0].target.id;
        if (active) link.setAttribute("aria-current", "true");
        else link.removeAttribute("aria-current");
      });
    }, { rootMargin: "-25% 0px -62%", threshold: [0.05, 0.25, 0.6] });
    sections.forEach(function (section) { observer.observe(section); });
  }

  const chart = document.getElementById("delta-chart");
  const thresholdInput = document.getElementById("threshold-range");
  const thresholdValue = document.getElementById("threshold-value");
  const thresholdCount = document.getElementById("threshold-count");
  if (!chart || !thresholdInput || !thresholdValue || !thresholdCount) return;

  if (typeof window.Plotly === "undefined") {
    chart.textContent = "The interactive plot could not be loaded.";
    return;
  }

  const series = [
    { id: "xpbd", label: "XPBD", color: "--accent", visible: true, phase: 0.2, base: -9.15, drift: 0.9 },
    { id: "featherstone", label: "Featherstone", color: "--accent-2", visible: true, phase: 1.7, base: -10.05, drift: 0.68 },
    { id: "mujoco", label: "MuJoCo", color: "--accent-3", visible: true, phase: 3.2, base: -8.25, drift: 1.0 }
  ];
  const scenarioMaxima = [-10.9, -10.2, -9.8, -9.4, -9.05, -8.7, -8.35, -8.1, -7.8, -7.5, -6.8, -5.9];

  function css(token) {
    return getComputedStyle(root).getPropertyValue(token).trim();
  }

  function dataFor(item) {
    const x = [];
    const y = [];
    for (let step = 0; step <= 180; step += 3) {
      const contact = Math.exp(-Math.pow((step - 92) / 26, 2)) * item.drift;
      const wave = 0.19 * Math.sin(step / 13 + item.phase) + 0.08 * Math.cos(step / 5.5 + item.phase);
      const logDifference = Math.max(-11.7, Math.min(-5.2, item.base + contact + wave));
      x.push(step);
      y.push(Math.pow(10, logDifference));
    }
    return { x: x, y: y };
  }

  function exponentLabel(exponent) {
    const superscripts = { "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹" };
    return "10" + String(exponent).split("").map(function (character) { return superscripts[character]; }).join("");
  }

  function renderThresholdValue(exponent) {
    const expression = "10^{" + exponent + "}";
    thresholdValue.setAttribute("aria-label", "10 to the power of " + exponent);
    if (window.katex) {
      window.katex.render(expression, thresholdValue, { throwOnError: false });
    } else {
      thresholdValue.textContent = exponentLabel(exponent);
    }
  }

  function drawPlot() {
    const width = chart.clientWidth || 320;
    const thresholdExponent = Number(thresholdInput.value);
    const traces = series.map(function (item) {
      const values = dataFor(item);
      return {
        x: values.x,
        y: values.y,
        type: "scatter",
        mode: "lines",
        name: item.label,
        visible: item.visible,
        line: { color: css(item.color), width: 2 },
        hovertemplate: item.label + ": %{y:.2e}<extra></extra>"
      };
    });

    const tickExponents = [-12, -11, -10, -9, -8, -7, -6, -5];
    const layout = {
      autosize: true,
      height: width < 520 ? 245 : 270,
      margin: { l: 64, r: 12, t: 4, b: 48, pad: 0 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: css("--sans"), size: 11, color: css("--muted") },
      showlegend: false,
      hovermode: "x unified",
      hoverlabel: { bgcolor: css("--surface"), bordercolor: css("--line-strong"), font: { color: css("--text"), size: 11 } },
      xaxis: {
        title: { text: "Simulation step", font: { color: css("--text"), size: 11 } },
        range: [0, 180],
        tick0: 0,
        dtick: width < 520 ? 60 : 30,
        gridcolor: css("--line"),
        linecolor: css("--line-strong"),
        zeroline: false,
        fixedrange: true
      },
      yaxis: {
        title: { text: "Maximum state difference, Δ<sub>t</sub>", font: { color: css("--text"), size: 11 } },
        type: "log",
        range: [-12, -5],
        tickvals: tickExponents.map(function (value) { return Math.pow(10, value); }),
        ticktext: tickExponents.map(exponentLabel),
        gridcolor: css("--line"),
        linecolor: css("--line-strong"),
        zeroline: false,
        fixedrange: true
      },
      shapes: [{
        type: "line",
        x0: 0,
        x1: 180,
        y0: Math.pow(10, thresholdExponent),
        y1: Math.pow(10, thresholdExponent),
        line: { color: css("--warning"), width: 1.25, dash: "dash" }
      }],
      annotations: [{
        x: 180,
        y: Math.pow(10, thresholdExponent),
        xanchor: "right",
        yanchor: "bottom",
        text: "threshold",
        showarrow: false,
        font: { color: css("--warning"), size: 10 }
      }],
      uirevision: "newton-determinism"
    };

    window.Plotly.react(chart, traces, layout, {
      responsive: true,
      displayModeBar: false,
      displaylogo: false,
      scrollZoom: false
    });
  }

  function updateThreshold() {
    const exponent = Number(thresholdInput.value);
    renderThresholdValue(exponent);
    const count = scenarioMaxima.filter(function (value) { return value <= exponent; }).length;
    thresholdCount.textContent = count + " of " + scenarioMaxima.length + " scenarios below threshold";
    drawPlot();
  }

  document.querySelectorAll(".legend-button").forEach(function (button) {
    button.addEventListener("click", function () {
      const item = series.find(function (candidate) { return candidate.id === button.dataset.series; });
      if (!item) return;
      item.visible = !item.visible;
      button.setAttribute("aria-pressed", String(item.visible));
      drawPlot();
    });
  });

  thresholdInput.addEventListener("input", updateThreshold);
  window.addEventListener("report-theme-change", drawPlot);
  new ResizeObserver(function () { window.Plotly.Plots.resize(chart); }).observe(chart);
  updateThreshold();
})();
