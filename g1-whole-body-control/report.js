'use strict';
const root = document.documentElement;
const themeButton = document.getElementById('theme-toggle');
const video = document.getElementById('comparison-video');
const clipSelect = document.getElementById('clip-select');
const methodSelect = document.getElementById('method-select');
let measurements;
const methods = {
  previous: ['Original GN', '#647580', '#b0bcc6'],
  fd: ['Short GN', '#147b66', '#71d2b5'],
  dial: ['DIAL', '#a55d17', '#ffbe7a'],
  analytic: ['Analytic GN', '#356bc0', '#85b7ff'],
  hybrid: ['Hybrid', '#7850b8', '#c3a1ff']
};
function accessibleLegend() {
  const graph = document.getElementById('tracking-chart');
  graph.querySelectorAll('.legendtoggle').forEach((item, i) => {
    item.setAttribute('tabindex', '0');
    item.setAttribute('role', 'button');
    item.setAttribute('aria-label', 'Toggle ' + graph.data[i].name);
    item.setAttribute('aria-pressed', String(graph.data[i].visible !== 'legendonly'));
    item.onkeydown = event => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        Plotly.restyle(graph, {visible: graph.data[i].visible === 'legendonly' ? true : 'legendonly'}, [i])
          .then(() => graph.querySelectorAll('.legendtoggle')[i].focus());
      }
    };
  });
}
function updateThemeLabel() {
  const dark = root.dataset.theme === 'dark';
  themeButton.setAttribute('aria-pressed', String(dark));
  themeButton.querySelector('.theme-label').textContent = dark ? 'Dark' : 'Bright';
  themeButton.querySelector('.theme-icon').textContent = dark ? '☾' : '☀';
  themeButton.setAttribute('aria-label', dark ? 'Switch to bright mode' : 'Switch to dark mode');
}
function updateVideo(keepTime = false) {
  if (!measurements) return;
  const available = measurements.media[clipSelect.value];
  for (const option of methodSelect.options) option.disabled = !available[option.value];
  if (!available[methodSelect.value]) methodSelect.value = 'fd';
  const media = available[methodSelect.value];
  document.getElementById('video-description').textContent = media.caption;
  video.setAttribute('aria-label', media.caption);
  video.poster = media.poster;
  if (video.getAttribute('src') === media.file) return;
  const time = keepTime ? video.currentTime : 0;
  const playing = !video.paused;
  video.pause();
  video.src = media.file;
  video.onloadedmetadata = () => {
    video.currentTime = Math.min(time, Math.max(0, video.duration - 0.02));
    if (playing) video.play().catch(() => {});
  };
  video.load();
}
function draw() {
  if (!measurements || !window.Plotly) return;
  const clip = clipSelect.value;
  const quantity = document.getElementById('quantity-select').value;
  const sideSelect = document.getElementById('foot-select');
  const side = Number(sideSelect.value);
  sideSelect.disabled = quantity === 'head';
  const dark = root.dataset.theme === 'dark';
  const style = getComputedStyle(root);
  const color = name => style.getPropertyValue(name).trim();
  const traces = [];
  const data = measurements.curves[clip];
  if (quantity === 'foot') {
    traces.push({x: data.fd.time, y: data.fd.reference_foot.map(pair => 100 * pair[side]),
      name: 'Reference', type: 'scatter', mode: 'lines', line: {color: color('--text'), width: 2, dash: 'dash'}});
  }
  for (const [key, settings] of Object.entries(methods)) {
    const d = data[key];
    traces.push({x: d.time, y: quantity === 'head' ? d.head : d[quantity].map(pair => 100 * pair[side]),
      name: settings[0], type: 'scatter', mode: 'lines', line: {color: settings[dark ? 2 : 1], width: 2}});
  }
  const unit = quantity === 'head' ? 'deg' : 'cm';
  for (const trace of traces) trace.hovertemplate = '%{x:.2f} s<br>%{y:.2f} ' + unit + '<extra>%{fullData.name}</extra>';
  Plotly.react('tracking-chart', traces, {
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: {family: 'system-ui, sans-serif', size: 11, color: color('--text')},
    margin: {l: 57, r: 12, t: 67, b: 52},
    legend: {orientation: 'h', x: 0, y: 1.28, font: {size: 10}},
    xaxis: {title: {text: 'Simulation time (s)'}, gridcolor: color('--line'), zeroline: false},
    yaxis: {title: {text: quantity === 'head' ? 'Head orientation error (deg)' :
      quantity === 'hand' ? 'Wrist position error (cm)' : 'Sole clearance (cm)'},
      gridcolor: color('--line'), rangemode: 'tozero', zerolinecolor: color('--muted')},
    hovermode: 'x unified', uirevision: clip + quantity + side
  }, {responsive: true, displaylogo: false, displayModeBar: false}).then(() => {
    const graph = document.getElementById('tracking-chart');
    accessibleLegend();
    if (!graph.keyboardLegend) {
      graph.on('plotly_afterplot', accessibleLegend);
      graph.keyboardLegend = true;
    }
  });
}
themeButton.addEventListener('click', () => {
  root.dataset.theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
  try { localStorage.setItem('report-theme', root.dataset.theme); } catch (e) {}
  updateThemeLabel(); draw();
});
clipSelect.addEventListener('change', () => {updateVideo(); draw();});
methodSelect.addEventListener('change', () => updateVideo(true));
document.getElementById('foot-select').addEventListener('change', draw);
document.getElementById('quantity-select').addEventListener('change', draw);
updateThemeLabel();
fetch('assets/mpc-comparison-plots.json').then(response => {
  if (!response.ok) throw new Error('Comparison data unavailable');
  return response.json();
}).then(data => {measurements = data; updateVideo(); draw();}).catch(() => {
  document.getElementById('tracking-chart').textContent = 'Interactive data could not load. Exact measurements remain available in the tables and results manifest.';
});
