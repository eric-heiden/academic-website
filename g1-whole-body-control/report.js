'use strict';
const root = document.documentElement;
const themeButton = document.getElementById('theme-toggle');
let measurements;
function updateThemeLabel() {
  const dark = root.dataset.theme === 'dark';
  themeButton.setAttribute('aria-pressed', String(dark));
  themeButton.querySelector('.theme-label').textContent = dark ? 'Dark' : 'Bright';
  themeButton.querySelector('.theme-icon').textContent = dark ? '☾' : '☀';
  themeButton.setAttribute('aria-label', dark ? 'Switch to bright mode' : 'Switch to dark mode');
}
function draw() {
  if (!measurements || !window.Plotly) return;
  const clip = document.getElementById('clip-select').value;
  const style = getComputedStyle(root);
  const color = name => style.getPropertyValue(name).trim();
  const traces = ['pd','qp','mpc'].map((controller,i) => {
    const d = measurements[clip][controller];
    return {x:d.time,y:d.error,type:'scatter',mode:'lines',name:controller.toUpperCase(),
      line:{color:[color('--accent-2'),color('--danger'),color('--accent')][i],width:2},
      hovertemplate:'%{x:.2f} s<br>%{y:.3f} m<extra>%{fullData.name}</extra>'};
  });
  Plotly.react('tracking-chart',traces,{paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',
    font:{family:'system-ui, sans-serif',size:12,color:color('--text')},
    margin:{l:59,r:14,t:34,b:56},legend:{orientation:'h',x:0,y:1.13},
    xaxis:{title:{text:'Simulation time (s)'},gridcolor:color('--line'),zeroline:false},
    yaxis:{title:{text:'Pelvis error (m)'},gridcolor:color('--line'),rangemode:'tozero',zeroline:false},
    hovermode:'x unified'}, {responsive:true,displaylogo:false,modeBarButtonsToRemove:['select2d','lasso2d'],displayModeBar:false});
}
themeButton.addEventListener('click',()=>{root.dataset.theme=root.dataset.theme==='dark'?'light':'dark';try{localStorage.setItem('report-theme',root.dataset.theme)}catch(e){}updateThemeLabel();draw()});
document.getElementById('clip-select').addEventListener('change',draw);
updateThemeLabel();
fetch('assets/plot-data.json').then(r=>{if(!r.ok)throw new Error('Plot data unavailable');return r.json()}).then(d=>{measurements=d;draw()}).catch(()=>{document.getElementById('tracking-chart').textContent='The interactive plot could not load. Exact measurements remain available in Table 1 and the downloadable results.'});
