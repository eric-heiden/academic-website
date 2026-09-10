'use strict';
const root = document.documentElement;
const themeButton = document.getElementById('theme-toggle');
let measurements;
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
        const visible = graph.data[i].visible === 'legendonly' ? true : 'legendonly';
        Plotly.restyle(graph, {visible}, [i]).then(() => {
          graph.querySelectorAll('.legendtoggle')[i].focus();
        });
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
function draw() {
  if (!measurements || !window.Plotly) return;
  const clip = document.getElementById('clip-select').value;
  const foot = Number(document.getElementById('foot-select').value);
  const d = measurements[clip];
  const style = getComputedStyle(root);
  const color = name => style.getPropertyValue(name).trim();
  const traces = ['reference','previous','actual'].map((key,i) => ({
    x:d.time,y:d[key].map(pair=>100*pair[foot]),type:'scatter',mode:'lines',
    name:['Reference','Previous sampling',clip === 'backflip' ? 'Tuned sampling' : 'Gauss–Newton'][i],
    line:{color:[color('--text'),color('--danger'),color('--accent')][i],width:2,dash:i===0?'dash':'solid'},
    hovertemplate:'%{x:.2f} s<br>%{y:.2f} cm<extra>%{fullData.name}</extra>'
  }));
  Plotly.react('tracking-chart',traces,{paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',
    font:{family:'system-ui, sans-serif',size:11,color:color('--text')},
    margin:{l:57,r:12,t:55,b:52},legend:{orientation:'h',x:0,y:1.22,font:{size:10}},
    xaxis:{title:{text:'Simulation time (s)'},gridcolor:color('--line'),zeroline:false},
    yaxis:{title:{text:'Sole clearance (cm)'},gridcolor:color('--line'),rangemode:'tozero',zeroline:true,zerolinecolor:color('--muted')},
    hovermode:'x unified'}, {responsive:true,displaylogo:false,displayModeBar:false}).then(() => {
      const graph = document.getElementById('tracking-chart');
      accessibleLegend();
      if (!graph.keyboardLegend) {
        graph.on('plotly_afterplot', accessibleLegend);
        graph.keyboardLegend = true;
      }
    });
}
themeButton.addEventListener('click',()=>{root.dataset.theme=root.dataset.theme==='dark'?'light':'dark';try{localStorage.setItem('report-theme',root.dataset.theme)}catch(e){}updateThemeLabel();draw()});
document.getElementById('clip-select').addEventListener('change',draw);
document.getElementById('foot-select').addEventListener('change',draw);
updateThemeLabel();
fetch('assets/gn-plot-data.json').then(r=>{if(!r.ok)throw new Error('Plot data unavailable');return r.json()}).then(d=>{measurements=d;draw()}).catch(()=>{document.getElementById('tracking-chart').textContent='The interactive plot could not load. Exact measurements remain available in the tables and downloadable results.'});
