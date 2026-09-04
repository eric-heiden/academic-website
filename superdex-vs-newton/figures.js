(function () {
  "use strict";

  const NS = "http://www.w3.org/2000/svg";
  const COLORS = {
    blue: "#315bd7",
    teal: "#087f78",
    orange: "#b94a25",
    violet: "#7357c6",
    green: "#197a50",
    muted: "#7a8498",
  };

  function svgNode(name, attrs, text) {
    const node = document.createElementNS(NS, name);
    Object.entries(attrs || {}).forEach(([key, value]) => node.setAttribute(key, value));
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function setText(name, value) {
    document.querySelectorAll(`[data-stat="${name}"]`).forEach((node) => {
      node.textContent = value;
    });
  }

  function sampleEvery(values, maxPoints) {
    const stride = Math.max(1, Math.ceil(values.length / maxPoints));
    const result = values.filter((_, index) => index % stride === 0);
    if (result[result.length - 1] !== values[values.length - 1]) result.push(values[values.length - 1]);
    return result;
  }

  function range(values, paddingFraction) {
    let lo = Math.min(...values);
    let hi = Math.max(...values);
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [0, 1];
    if (lo === hi) {
      const pad = Math.abs(lo || 1) * 0.1;
      return [lo - pad, hi + pad];
    }
    const pad = (hi - lo) * paddingFraction;
    return [lo - pad, hi + pad];
  }

  function pathFor(data, x, y, xScale, yScale) {
    return data.map((point, index) => `${index ? "L" : "M"}${xScale(x(point)).toFixed(2)},${yScale(y(point)).toFixed(2)}`).join(" ");
  }

  function lineChart(id, raw, series, options) {
    const host = document.getElementById(id);
    if (!host || !raw.length) return;
    const data = sampleEvery(raw, 420);
    const width = 800;
    const height = 330;
    const margin = { left: 60, right: 20, top: 42, bottom: 48 };
    const plotW = width - margin.left - margin.right;
    const plotH = height - margin.top - margin.bottom;
    const xValues = data.map((d) => d[options.xKey || "t"]);
    const yValues = series.flatMap((item) => data.map((d) => d[item.key])).filter(Number.isFinite);
    if (options.band) {
      yValues.push(...data.map((d) => d[options.band.low]), ...data.map((d) => d[options.band.high]));
    }
    const [xMin, xMax] = range(xValues, 0);
    let [yMin, yMax] = range(yValues, 0.08);
    if (options.zeroFloor && yMin > 0) yMin = 0;
    const xScale = (v) => margin.left + ((v - xMin) / (xMax - xMin || 1)) * plotW;
    const yScale = (v) => margin.top + plotH - ((v - yMin) / (yMax - yMin || 1)) * plotH;

    const svg = svgNode("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": options.ariaLabel });
    const grid = svgNode("g");
    for (let i = 0; i <= 4; i += 1) {
      const fraction = i / 4;
      const y = margin.top + plotH * fraction;
      const value = yMax - (yMax - yMin) * fraction;
      grid.append(svgNode("line", { class: "grid", x1: margin.left, x2: width - margin.right, y1: y, y2: y }));
      grid.append(svgNode("text", { class: "tick", x: margin.left - 9, y: y + 4, "text-anchor": "end" }, options.yFormat ? options.yFormat(value) : value.toFixed(2)));
    }
    for (let i = 0; i <= 4; i += 1) {
      const fraction = i / 4;
      const x = margin.left + plotW * fraction;
      const value = xMin + (xMax - xMin) * fraction;
      grid.append(svgNode("line", { class: "grid", x1: x, x2: x, y1: margin.top, y2: margin.top + plotH }));
      grid.append(svgNode("text", { class: "tick", x, y: height - 25, "text-anchor": "middle" }, options.xFormat ? options.xFormat(value) : value.toFixed(1)));
    }
    svg.append(grid);

    if (options.band) {
      const upper = data.map((d) => `${xScale(d[options.xKey || "t"]).toFixed(2)},${yScale(d[options.band.high]).toFixed(2)}`);
      const lower = [...data].reverse().map((d) => `${xScale(d[options.xKey || "t"]).toFixed(2)},${yScale(d[options.band.low]).toFixed(2)}`);
      svg.append(svgNode("path", { class: "band", d: `M${upper.join(" L")} L${lower.join(" L")} Z`, fill: options.band.color }));
    }

    series.forEach((item, index) => {
      svg.append(svgNode("path", {
        class: "series",
        d: pathFor(data, (d) => d[options.xKey || "t"], (d) => d[item.key], xScale, yScale),
        stroke: item.color,
      }));
      const legendX = margin.left + index * 196;
      svg.append(svgNode("line", { x1: legendX, x2: legendX + 22, y1: 18, y2: 18, stroke: item.color, "stroke-width": 4, "stroke-linecap": "round" }));
      svg.append(svgNode("text", { class: "legend", x: legendX + 30, y: 22 }, item.label));
    });

    svg.append(svgNode("text", { class: "axis-label", x: margin.left + plotW / 2, y: height - 4, "text-anchor": "middle" }, options.xLabel));
    const yLabel = svgNode("text", { class: "axis-label", x: 14, y: margin.top + plotH / 2, "text-anchor": "middle", transform: `rotate(-90 14 ${margin.top + plotH / 2})` }, options.yLabel);
    svg.append(yLabel);
    host.replaceChildren(svg);
  }

  function pathChart(id, raw) {
    const host = document.getElementById(id);
    if (!host || !raw.length) return;
    const data = sampleEvery(raw, 420);
    const width = 800;
    const height = 330;
    const margin = { left: 60, right: 24, top: 34, bottom: 48 };
    const plotW = width - margin.left - margin.right;
    const plotH = height - margin.top - margin.bottom;
    const [xMin, xMax] = range(data.map((d) => d.root_x), 0.08);
    const [yMin, yMax] = range(data.map((d) => d.root_y), 0.08);
    const xScale = (v) => margin.left + ((v - xMin) / (xMax - xMin || 1)) * plotW;
    const yScale = (v) => margin.top + plotH - ((v - yMin) / (yMax - yMin || 1)) * plotH;
    const svg = svgNode("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "Top-down G1 root trajectory" });

    for (let i = 0; i <= 4; i += 1) {
      const fx = i / 4;
      const x = margin.left + plotW * fx;
      const y = margin.top + plotH * fx;
      svg.append(svgNode("line", { class: "grid", x1: x, x2: x, y1: margin.top, y2: margin.top + plotH }));
      svg.append(svgNode("line", { class: "grid", x1: margin.left, x2: margin.left + plotW, y1: y, y2: y }));
      svg.append(svgNode("text", { class: "tick", x, y: height - 25, "text-anchor": "middle" }, (xMin + (xMax - xMin) * fx).toFixed(1)));
      svg.append(svgNode("text", { class: "tick", x: margin.left - 9, y: y + 4, "text-anchor": "end" }, (yMax - (yMax - yMin) * fx).toFixed(1)));
    }

    svg.append(svgNode("path", { class: "series", d: pathFor(data, (d) => d.root_x, (d) => d.root_y, xScale, yScale), stroke: COLORS.blue }));
    const start = data[0];
    const end = data[data.length - 1];
    svg.append(svgNode("circle", { class: "dot", cx: xScale(start.root_x), cy: yScale(start.root_y), r: 6, fill: COLORS.teal }));
    svg.append(svgNode("circle", { class: "dot", cx: xScale(end.root_x), cy: yScale(end.root_y), r: 7, fill: COLORS.orange }));
    svg.append(svgNode("text", { class: "legend", x: xScale(start.root_x) + 11, y: yScale(start.root_y) - 8 }, "start"));
    svg.append(svgNode("text", { class: "legend", x: xScale(end.root_x) - 11, y: yScale(end.root_y) - 10, "text-anchor": "end" }, "finish"));
    svg.append(svgNode("text", { class: "axis-label", x: margin.left + plotW / 2, y: height - 4, "text-anchor": "middle" }, "root x (m)"));
    svg.append(svgNode("text", { class: "axis-label", x: 14, y: margin.top + plotH / 2, "text-anchor": "middle", transform: `rotate(-90 14 ${margin.top + plotH / 2})` }, "root y (m)"));
    host.replaceChildren(svg);
  }

  function showChartError(id) {
    const host = document.getElementById(id);
    if (host) host.innerHTML = '<div class="chart-loading">Trace unavailable. Download the data bundle below.</div>';
  }

  async function loadJson(path) {
    const response = await fetch(path);
    if (!response.ok) throw new Error(`${response.status} ${path}`);
    return response.json();
  }

  async function loadG1() {
    try {
      const data = await loadJson("data/g1-locomotion.json");
      setText("g1-distance", `${data.summary.horizontal_displacement_m.toFixed(2)} m`);
      setText("g1-speed", `${data.summary.peak_horizontal_speed_m_s.toFixed(2)} m/s`);
      setText("g1-height", `${data.summary.minimum_root_height_m.toFixed(2)} m`);
      pathChart("chart-g1-path", data.samples);
      lineChart("chart-g1-speed", data.samples, [
        { key: "speed_xy", label: "horizontal speed", color: COLORS.blue },
        { key: "command_forward", label: "forward command", color: COLORS.orange },
      ], {
        xLabel: "sequence time (s)", yLabel: "speed / command", zeroFloor: true,
        xFormat: (v) => v.toFixed(0), yFormat: (v) => v.toFixed(1),
        ariaLabel: "G1 horizontal speed and forward command over time",
      });
    } catch (error) {
      console.warn(error);
      showChartError("chart-g1-path");
      showChartError("chart-g1-speed");
    }
  }

  async function loadPanda() {
    try {
      const data = await loadJson("data/panda-pick-place.json");
      setText("panda-lift", `${(data.summary.object_lift_m * 100).toFixed(1)} cm`);
      setText("panda-peak", `${data.summary.peak_object_height_m.toFixed(2)} m`);
      setText("panda-result", "Lift + place passed");
      lineChart("chart-panda-height", data.samples, [
        { key: "object_z", label: "pen height", color: COLORS.orange },
        { key: "end_effector_z", label: "gripper height", color: COLORS.blue },
      ], {
        xLabel: "sequence time (s)", yLabel: "height (m)", zeroFloor: true,
        xFormat: (v) => v.toFixed(0), yFormat: (v) => v.toFixed(2),
        ariaLabel: "Panda gripper and pen height over the pick-and-place sequence",
      });
    } catch (error) {
      console.warn(error);
      showChartError("chart-panda-height");
    }
  }

  async function loadCloth() {
    try {
      const data = await loadJson("data/h1-jacket.json");
      setText("cloth-particles", data.summary.particle_count.toLocaleString());
      setText("cloth-span", `${data.summary.cloth_vertical_span_max_m.toFixed(2)} m`);
      setText("cloth-hand", `${data.summary.peak_left_hand_height_m.toFixed(2)} m`);
      lineChart("chart-cloth", data.samples, [
        { key: "cloth_z_median", label: "jacket median", color: COLORS.orange },
        { key: "left_hand_z", label: "left hand", color: COLORS.blue },
      ], {
        band: { low: "cloth_z_min", high: "cloth_z_max", color: COLORS.orange },
        xLabel: "sequence time (s)", yLabel: "height (m)", zeroFloor: true,
        xFormat: (v) => v.toFixed(0), yFormat: (v) => v.toFixed(1),
        ariaLabel: "H1 hand height and jacket particle height envelope over time",
      });
    } catch (error) {
      console.warn(error);
      showChartError("chart-cloth");
    }
  }

  function wireCopyButtons() {
    document.querySelectorAll(".copy").forEach((button) => {
      button.addEventListener("click", async () => {
        const code = button.closest(".code-card").querySelector("code").innerText;
        await navigator.clipboard.writeText(code);
        const label = button.textContent;
        button.textContent = "Copied";
        window.setTimeout(() => { button.textContent = label; }, 1200);
      });
    });
  }

  function wireVideos() {
    const videos = document.querySelectorAll("video[data-autoplay]");
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    if (!("IntersectionObserver" in window)) return;
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) entry.target.play().catch(() => {});
        else entry.target.pause();
      });
    }, { threshold: 0.42 });
    videos.forEach((video) => observer.observe(video));
  }

  document.addEventListener("DOMContentLoaded", () => {
    loadG1();
    loadPanda();
    loadCloth();
    wireCopyButtons();
    wireVideos();
  });
})();
