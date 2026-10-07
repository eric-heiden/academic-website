from pathlib import Path
from bs4 import BeautifulSoup
import hashlib
import html
import json
import re
from pygments import lex
from pygments.lexers import PythonLexer
from pygments.token import Token

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(__file__).resolve().parent
VIDEO_EDITS = json.loads((ROOT / 'evidence/video-edit-map.json').read_text())

def video_duration(name):
    seconds = round(VIDEO_EDITS[name]['duration'])
    return f'{seconds // 60}:{seconds % 60:02}'

def chapter_time(name, chapter):
    return VIDEO_EDITS[name]['chapter_starts'][chapter]

def chapter_label(name, chapter):
    seconds = int(chapter_time(name, chapter))
    return f'{seconds // 60}:{seconds % 60:02}'
backup = WORK / 'report-before-four-part.html'
if not backup.exists():
    backup.write_bytes((ROOT / 'index.html').read_bytes())
    (WORK / 'report-before-four-part-manifest.json').write_bytes((ROOT / 'manifest.json').read_bytes())
    (WORK / 'report-before-four-part-reproduce.txt').write_bytes((ROOT / 'REPRODUCE.txt').read_bytes())
old = BeautifulSoup(backup.read_text(), 'html.parser')
RSS = 'https://github.com/KeplerC/RoboSimStudio/blob/9bb21c85f79240eb16cc5e610114edd74f1eb2b2/'
NEWTON = 'https://github.com/newton-physics/newton/blob/8bfd16eb8941feeaf33ed544464a25630839cff0/'
NMAIN = 'https://github.com/newton-physics/newton/blob/7dcc81d39feb6e83155f615a1c1e0402c1e62517/'
LAB = 'https://github.com/isaac-sim/IsaacLab/blob/f5383e7feb4c433372caf0d4d537e1e6c03ef0f2/'
SIM = 'https://github.com/isaac-sim/IsaacSim/blob/2469084bc328710207c6bc4ede32209082a9286c/'
ARENA = 'https://github.com/isaac-sim/IsaacLab-Arena/blob/d46405ae1d3a08e7f7aee6294f0a95f2ff6ef540/'

def link(url, label):
    return f'<a href="{html.escape(url, quote=True)}" target="_blank" rel="noreferrer">{label} ↗</a>'

def r(path, label): return link(RSS + path, label)
def n(path, label): return link(NEWTON + path, label)
def nm(path, label): return link(NMAIN + path, label)
def ev(path, label): return f'<a href="evidence/{path}">{label}</a>'
def ex(path, label=None): return f'<a href="examples/{path}">{label or path}</a>'

def highlight_snippet(match):
    source = html.unescape(match.group(1))
    if re.fullmatch(r'[0-9a-f]{12,40}', source):
        return f'<code class="revision">{html.escape(source)}</code>'
    fragments = []
    recovered = []
    for token, value in lex(source, PythonLexer(stripnl=False, ensurenl=False)):
        recovered.append(value)
        kind = next((name for group, name in (
            (Token.Comment, 'comment'), (Token.Keyword, 'keyword'),
            (Token.Literal.String, 'string'), (Token.Literal.Number, 'number'),
            (Token.Name.Function, 'function'), (Token.Name.Class, 'function'),
            (Token.Name, 'name'), (Token.Operator, 'operator'),
            (Token.Punctuation, 'punctuation'),
        ) if token in group), None)
        escaped = html.escape(value)
        fragments.append(f'<span class="tok-{kind}">{escaped}</span>' if kind else escaped)
    assert ''.join(recovered) == source, 'Syntax highlighting changed the snippet text'
    return '<code class="language-python syntax-python">' + ''.join(fragments) + '</code>'

# Keep all 17 capabilities and their evidence links, with the requested Newton baseline.
table = old.find(id='feature-table')
table.find_all('th')[2].string = 'Newton baseline'
comparisons = [
    'Interactive ViewerViser provides pause/step/reset, example switching, GUI controls, scalar plots and images.',
    'Transform gizmos and existing IK examples cover the underlying interaction; Newton also has an experimental controllers package.',
    'Viewer force picking primarily provides transient mouse manipulation with per-layer interaction state.',
    'The evaluated Newton picking path targets rigid bodies; it does not provide this persistent cloth-patch workflow.',
    'Model arrays and solver notifications provide the foundations; there is no comparable general tuning application in the evaluated viewer.',
    'Overlaps ModelBuilder / Model / State / Control, importers, examples, and Isaac Lab’s scene API.',
    'The evaluated viewer does not supply this bundled persistent development session; Python provides individual operations.',
    'Newton controllers and Isaac teleoperation integrations cover adjacent responsibilities.',
    'Viewer video, physical checkpoints and training datasets have different contracts. Lab and Arena have their own data workflows.',
    'Viewer and sensor/rendering facilities already exist. Image display does not define a training dataset.',
    'RSS specializes Newton’s viewer; the baseline also provides batching and material handling.',
    'Newton already supplies multiphysics primitives, deformables, cables, controllers and coupled-solver work.',
    'Optional cross-engine adapters are outside the viewer’s responsibility. Coupling needs separate numerical validation.',
    'Arena supplies scene/embodiment/task composition, completion semantics, variations and evaluation.',
    'A public gallery cannot establish validated executable task coverage.',
    'Lab and Arena own training, evaluation and data-conversion workflows.',
    'Newton has USD import/schema primitives; Isaac Sim supplies a broader USD authoring application.',
]
for row, comparison in zip(table.select('tbody tr'), comparisons, strict=True):
    row.find_all('td')[2].string = comparison
    # Replace obsolete in-page walkthrough links with plain capability names.
    for anchor in row.find_all('a', href=re.compile('^#')):
        anchor.unwrap()
for node in list(table.find_all(string=True)):
    changed = str(node).replace('Use the PR’s primitives', 'Use Newton’s viewer primitives').replace('PR viewer', 'Newton viewer')
    if changed != str(node): node.replace_with(changed)
matrix = str(table)

architecture = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 580" role="img" aria-labelledby="architecture-title architecture-desc">
<title id="architecture-title">How RoboSimStudio is built</title><desc id="architecture-desc">A Python scene is built by the RSS runtime into Newton. The browser connects through the studio package. The agent package accepts local commands and watches source files. Simulation changes converge on the owning simulation thread. Optional Genesis and SuperDex adapters are separate.</desc>
<defs><marker id="arrow-a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#426759"/></marker></defs>
<style>text{{font-family:Arial,sans-serif;fill:#20372e}}.name{{font-size:22px;font-weight:700}}.sub{{font-size:17px}}.pkg{{font-size:15px;fill:#567264}}.box{{fill:#fff;stroke:#acbdb1;stroke-width:1.5}}.flow{{fill:none;stroke:#426759;stroke-width:2;marker-end:url(#arrow-a)}}a:hover rect{{stroke:#163f2e;stroke-width:3}}</style>
<rect width="1000" height="580" rx="18" fill="#f4f6f0"/>
<a href="{RSS}robosimstudio/sim/scene.py" target="_blank"><rect class="box" x="25" y="28" width="290" height="104" rx="12"/><text class="name" x="45" y="62">Python scene</text><text class="sub" x="45" y="91">Named robots, objects, cloth</text><text class="pkg" x="45" y="114">Scene → build()</text></a>
<rect class="box" x="355" y="28" width="290" height="104" rx="12"/><text class="name" x="375" y="62">Browser · Viser</text><text class="sub" x="375" y="91">Select, drag, inspect, tune</text><text class="pkg" x="375" y="114">Human’s shared view</text>
<rect class="box" x="685" y="28" width="290" height="104" rx="12"/><text class="name" x="705" y="62">Coding agent / CLI</text><text class="sub" x="705" y="91">Edit source, run, inspect</text><text class="pkg" x="705" y="114">Local socket commands</text>
<path class="flow" d="M170 132V182"/><path class="flow" d="M500 132V182"/><path class="flow" d="M830 132V182"/>
<a href="{RSS}robosimstudio/sim/sim.py" target="_blank"><rect class="box" x="25" y="190" width="290" height="116" rx="12"/><text class="pkg" x="45" y="216">robosimstudio/</text><text class="name" x="45" y="246">Scene &amp; runtime</text><text class="sub" x="45" y="274">Build, control, step, record</text></a>
<a href="{RSS}robosimstudio_studio/session.py" target="_blank"><rect class="box" x="355" y="190" width="290" height="116" rx="12"/><text class="pkg" x="375" y="216">robosimstudio_studio/</text><text class="name" x="375" y="246">Interactive session</text><text class="sub" x="375" y="274">Panels, targets, viewer</text></a>
<a href="{RSS}robosimstudio_agent/server.py" target="_blank"><rect class="box" x="685" y="190" width="290" height="116" rx="12"/><text class="pkg" x="705" y="216">robosimstudio_agent/</text><text class="name" x="705" y="246">Live development</text><text class="sub" x="705" y="274">Watch file, rebuild, recover</text></a>
<path class="flow" d="M170 306V363"/><path class="flow" d="M500 306V363"/><path class="flow" d="M830 306V363"/>
<rect x="25" y="370" width="950" height="62" rx="12" fill="#dce9dd"/><text class="name" x="500" y="398" text-anchor="middle">One simulation thread</text><text class="sub" x="500" y="421" text-anchor="middle">Session tick · queued agent requests · build before swapping the scene</text>
<path class="flow" d="M340 432V466"/><path d="M825 432V468" stroke="#9b7955" stroke-width="2" stroke-dasharray="6 5"/>
<a href="{NMAIN}newton/__init__.py" target="_blank"><rect x="25" y="475" width="630" height="82" rx="12" fill="#1f513b"/><text x="45" y="507" style="fill:#fff;font-size:23px;font-weight:700">Newton physics</text><text x="45" y="536" style="fill:#e5efe7;font-size:18px">Model + State + Control · collision · solvers</text></a>
<a href="{RSS.replace('/blob/', '/tree/')}robosimstudio/engines" target="_blank"><rect x="685" y="475" width="290" height="82" rx="12" fill="#f3e8d8"/><text class="sub" x="705" y="507">Optional engine bridges</text><text class="pkg" x="705" y="536">SuperDex / Genesis · experimental</text></a>
</svg>'''

ecosystem = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 630" role="img" aria-labelledby="ecosystem-title ecosystem-desc">
<title id="ecosystem-title">Approximate ecosystem map</title><desc id="ecosystem-desc">Three overlapping areas represent physics and simulation, interactive authoring and inspection, and training, tasks and evaluation. Newton is an engine and library. RSS is a lightweight workbench above physics. Genesis and SuperDex span several areas. Isaac Sim provides scene and sensor authoring, Isaac Lab training environments, and Isaac Lab Arena task composition and evaluation. Placement is illustrative, not feature coverage or compatibility.</desc>
<style>text{font-family:Arial,sans-serif;fill:#20372e}.area{font-size:22px;font-weight:700}.project{font-size:24px;font-weight:700}.role{font-size:16px}.pill{fill:#fff;fill-opacity:.91;stroke:#fff;stroke-width:2}</style>
<rect width="1000" height="630" rx="18" fill="#faf9f4"/>
<path d="M58 170C60 30 330 12 521 102C670 173 641 350 510 410C365 480 55 424 58 170Z" fill="#b8d6c1" fill-opacity=".73" stroke="#8caf97" stroke-width="1.5"/>
<path d="M420 96C565 13 901 38 949 188C1000 349 843 443 636 408C450 376 321 206 420 96Z" fill="#e6d0ac" fill-opacity=".65" stroke="#c5a579" stroke-width="1.5"/>
<path d="M180 360C182 230 375 229 582 263C790 297 876 422 789 536C690 655 172 608 180 360Z" fill="#c9cfec" fill-opacity=".66" stroke="#a6afcf" stroke-width="1.5"/>
<text class="area" x="93" y="100">Physics &amp; simulation</text>
<text class="area" x="574" y="100">Authoring &amp; inspection</text>
<text class="area" x="355" y="576">Training, tasks &amp; evaluation</text>
<rect class="pill" x="100" y="134" width="245" height="81" rx="15"/><text class="project" x="120" y="168">Newton</text><text class="role" x="120" y="194">Physics library + viewers</text>
<rect x="421" y="144" width="221" height="91" rx="15" fill="#1f513b"/><text x="441" y="180" style="fill:#fff;font-size:25px;font-weight:700">RoboSimStudio</text><text x="441" y="209" style="fill:#fff;font-size:16px">Interactive workbench</text>
<rect class="pill" x="703" y="166" width="215" height="86" rx="15"/><text class="project" x="723" y="201">Isaac Sim</text><text class="role" x="723" y="229">USD, sensors, rendering</text>
<rect class="pill" x="110" y="272" width="232" height="90" rx="15"/><text class="project" x="130" y="307">Genesis</text><text class="role" x="130" y="334">Simulation platform</text>
<rect class="pill" x="410" y="287" width="250" height="92" rx="15"/><text class="project" x="430" y="322">SuperDex</text><text class="role" x="430" y="350">Physics, Studio, Lab</text>
<rect class="pill" x="264" y="423" width="240" height="91" rx="15"/><text class="project" x="284" y="459">Isaac Lab</text><text class="role" x="284" y="488">Learning environments</text>
<rect class="pill" x="566" y="433" width="250" height="91" rx="15"/><text class="project" x="586" y="469">Isaac Lab Arena</text><text class="role" x="586" y="498">Task composition &amp; evaluation</text>
</svg>'''
(ROOT / 'media/rss-architecture.svg').write_text(architecture)
(ROOT / 'media/ecosystem-map.svg').write_text(ecosystem)

css = '''
:root{--paper:#f6f5ef;--ink:#22372e;--green:#21533d;--muted:#606f65;--line:#d9dfd5;--warm:#e7b76b;--white:#fffefb}*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:32px}body{margin:0;background:var(--paper);color:var(--ink);font:17px/1.66 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}a{color:var(--green);text-decoration-thickness:1px;text-underline-offset:3px}a:hover{color:#986621}button,input,select{font:inherit}button,a,input,select,summary{-webkit-tap-highlight-color:transparent}button:focus-visible,a:focus-visible,input:focus-visible,select:focus-visible,summary:focus-visible{outline:3px solid #c88d34;outline-offset:4px}.rail{position:fixed;inset:0 auto 0 0;width:222px;padding:38px 24px;background:#193e2e;color:#f2f5ed;display:flex;flex-direction:column;z-index:4}.brand{font-size:25px;line-height:1.3;font-weight:650;color:#fff;letter-spacing:-.5px}.rail small{display:block;color:#bad0bf;margin-top:12px;font-size:12px;letter-spacing:1px;text-transform:uppercase}.rail nav{display:grid;gap:10px;margin-top:44px}.rail nav a{display:block;padding:11px 12px;color:#cddbcf;text-decoration:none;font-size:14px;line-height:1.55;border-left:2px solid transparent}.rail nav a.active{color:#fff;background:#ffffff10;border-color:#e6bd78}.rail nav span{display:block;font-size:11px;letter-spacing:1px;color:#d7b579}.rail-bottom{margin-top:auto;font-size:12px;color:#c4d2c5}.rail-bottom a{color:#e6ebdf}.progress{height:2px;background:#496a52;margin:16px 0}.progress i{display:block;height:100%;width:0;background:#e6bd78}main{margin-left:222px;max-width:1400px;padding:58px 62px 80px}.eyebrow{font-size:12px;letter-spacing:2px;text-transform:uppercase;color:var(--muted);font-weight:650}h1,h2{font-family:Georgia,"Times New Roman",serif;font-weight:400;line-height:1.12;letter-spacing:-1.5px}h1{font-size:clamp(42px,5.5vw,74px);margin:21px 0}h2{font-size:clamp(33px,3.9vw,49px);margin:12px 0 25px}h3{font-size:24px;line-height:1.3;margin:0 0 15px;letter-spacing:-.4px}p{margin:0 0 17px}p:last-child{margin-bottom:0}strong{font-weight:650}.lede{font-size:23px;line-height:1.48;max-width:890px;color:#405746}.meta{font-size:13px;color:var(--muted);margin-top:24px}.jump{display:inline-block;margin:24px 8px 0 0;padding:10px 17px;background:var(--green);color:white;text-decoration:none;border-radius:7px;font-size:14px}.jump:hover{background:#153b29;color:#fff}.jump.secondary{background:transparent;color:var(--green);border:1px solid #bdc8b8}.part{padding-top:64px;margin-top:56px;border-top:1px solid var(--line)}.part-lede{max-width:940px;font-size:19px;line-height:1.65}.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;margin:26px 0}.card,.feature,.video-card{border:1px solid var(--line);background:var(--white);border-radius:14px;padding:25px;min-width:0}.card p{font-size:16px}.card .label{font-size:11px;text-transform:uppercase;letter-spacing:1.2px;color:var(--muted);margin-bottom:9px}.card b.title{font-size:21px;display:block;line-height:1.35;margin-bottom:12px}.callout{padding:21px 24px;border-left:4px solid #638b6b;background:#eaf0e5;margin:25px 0;border-radius:0 9px 9px 0}.callout.warm{border-color:#b8853b;background:#f4eddf}.callout p{font-size:16px}.small,.sources{font-size:13px;color:var(--muted);line-height:1.6}.sources{margin-top:17px}.sources a{margin-right:12px}.figure{background:var(--white);padding:18px;border:1px solid var(--line);border-radius:15px;margin:28px 0}.figure svg{display:block;width:100%;height:auto}.figure figcaption{font-size:13px;color:var(--muted);margin:12px 6px 2px}.diagram-scroll{overflow-x:auto}.diagram-scroll svg{min-width:660px}.feature{margin:24px 0;padding:29px}.feature .tag{font-size:11px;text-transform:uppercase;letter-spacing:1.5px;color:#8b6b3e;margin-bottom:9px}.split{display:grid;grid-template-columns:minmax(0,1.18fr) minmax(0,1fr);gap:26px;align-items:start}.feature figure{margin:0}.feature img{display:block;width:100%;border-radius:8px}.feature figcaption{margin-top:9px;font-size:12px;color:var(--muted);line-height:1.5}.feature p{font-size:16px}.mechanism{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:18px 0;color:#415a48;font-size:13px}.mechanism span{border:1px solid #cedac9;padding:6px 10px;border-radius:6px;background:#f4f7f0}.mechanism i{font-style:normal}.chapters{display:flex;flex-wrap:wrap;gap:8px;margin:16px 0 0}.chapters button{font-size:12px;border:1px solid #b9cbb7;background:#f0f5ea;color:#254d34;padding:7px 11px;border-radius:6px;cursor:pointer}.chapters button:hover{background:#dce9d4}.video-card{margin:24px 0;padding:22px}.video-card video{display:block;width:100%;max-height:660px;background:#111c16;border-radius:8px}.video-title{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:0 0 13px;font-size:19px;font-weight:650}.video-title small{font-size:12px;font-weight:400;color:var(--muted)}.video-card>p{font-size:14px;color:var(--muted);margin-top:13px}.video-status{font-size:12px;color:var(--muted)}details{border:1px solid var(--line);border-radius:10px;background:#fffefb;margin:24px 0;padding:0 20px}summary{padding:18px 0;cursor:pointer;font-weight:650;font-size:16px}details[open]>summary{border-bottom:1px solid var(--line);margin-bottom:20px}details>*:last-child{margin-bottom:20px}details p{font-size:15px}.table-wrap{overflow-x:auto;max-width:100%;margin:20px 0}table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.6;text-align:left}th{font-size:11px;letter-spacing:.7px;text-transform:uppercase;color:#617363;background:#eef2e8}td,th{padding:15px 14px;vertical-align:top;border-bottom:1px solid var(--line)}td:first-child{font-weight:550}td a{overflow-wrap:anywhere}tr:last-child td{border-bottom:0}.table-wrap table{min-width:720px}.compact td:first-child{width:18%}.filters{display:flex;gap:12px;flex-wrap:wrap;align-items:center;font-size:13px}.filters label{display:flex;gap:7px;align-items:center}.filters input,.filters select{padding:9px 12px;max-width:100%;border:1px solid #bccab5;border-radius:5px;background:#fff}#feature-count{color:var(--muted)}code{font: .86em/1.5 ui-monospace,SFMono-Regular,Consolas,monospace;background:#edf0e8;border-radius:4px;padding:2px 5px;overflow-wrap:anywhere}pre{padding:22px;background:#19362a;color:#edf3e7;border-radius:9px;overflow-x:auto;font:13px/1.7 ui-monospace,SFMono-Regular,Consolas,monospace;white-space:pre}pre code{padding:0;background:none;font:inherit;color:inherit;overflow-wrap:normal}.three{grid-template-columns:repeat(3,minmax(0,1fr))}.choice{border-top:4px solid #b9c4b3}.choice.recommended{border-top-color:#286442;background:#f3f8ef}.choice .verdict{display:inline-block;font-size:11px;letter-spacing:1px;text-transform:uppercase;margin-bottom:13px;color:#466b4a}.flow-diagram{border:1px solid var(--line);border-radius:14px;padding:26px;background:#fcfcf7;margin:27px 0}.flow-inputs{display:grid;grid-template-columns:1fr 1fr;gap:16px}.flow-box{text-align:center;border:1px solid #c5d2bd;background:#f3f7ef;border-radius:9px;padding:16px;font-size:16px}.flow-box small{display:block;font-size:12px;color:#657561;margin-top:3px}.flow-box.owner{background:#21533d;color:white;border:0;font-weight:650}.flow-box.owner small{color:#e1eadb;font-weight:400}.flow-arrow{text-align:center;color:#6a7b63;font-size:13px;padding:12px}.flow-arrow b{font-size:20px;margin-right:7px}.flow-caption{font-size:13px;color:var(--muted);margin-top:16px}.roadmap{display:grid;gap:18px;margin:30px 0;counter-reset:phase}.phase{display:grid;grid-template-columns:64px 1fr;gap:20px;padding:27px;border:1px solid var(--line);border-radius:14px;background:#fffefb}.phase-num{width:49px;height:49px;border-radius:50%;background:#dce9d6;color:#31583a;display:grid;place-items:center;font-size:23px;font-family:Georgia,serif}.phase h3{font-size:22px}.phase p{font-size:16px}.phase .gate{padding:14px 17px;background:#f1f4e9;border-radius:7px;font-size:14px;margin-top:17px}.phase .owner{font-size:12px;color:var(--muted);margin:0 0 7px}.evidence-stats{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:20px 0}.stat{border:1px solid var(--line);border-radius:10px;padding:18px}.stat b{display:block;font:33px/1.2 Georgia,serif;margin-bottom:9px}.stat span{display:block;font-size:13px}.source-list{columns:2;column-gap:30px;padding-left:20px;font-size:13px}.source-list li{break-inside:avoid;padding:5px 0}footer{border-top:1px solid var(--line);margin-top:45px;padding-top:22px;font-size:13px;color:var(--muted)}[hidden]{display:none!important}@media(min-width:1700px){main{margin-right:auto;margin-left:calc(222px + (100vw - 1622px)/2)}}@media(max-width:1150px){main{padding:44px 32px}.rail{width:190px;padding:30px 17px}main{margin-left:190px}.three{grid-template-columns:1fr}.split{grid-template-columns:1fr 1fr}.feature{padding:23px}h3{font-size:22px}}@media(max-width:780px){body{font-size:16px}.rail{position:relative;width:auto;padding:19px 20px}.brand{font-size:22px}.rail small,.rail-bottom{display:none}.rail nav{grid-template-columns:repeat(4,1fr);gap:5px;margin-top:17px}.rail nav a{font-size:11px;padding:7px 5px;line-height:1.4}.rail nav span{font-size:10px}.rail nav a.active{border-left-color:transparent;border-bottom:2px solid #e6bd78}main{margin:0;padding:33px 19px 50px}h1{font-size:48px;letter-spacing:-1.5px}.lede{font-size:21px}.part{padding-top:38px;margin-top:39px}h2{font-size:36px}.cards,.split,.evidence-stats{grid-template-columns:1fr}.figure{padding:10px;margin:23px 0}.feature{padding:20px}.feature h3{font-size:23px}.feature p{font-size:16px}.video-card{padding:13px}.video-title{font-size:16px}.video-title small{font-size:11px}.flow-diagram{padding:16px}.flow-inputs{grid-template-columns:1fr}.flow-box{padding:13px}.phase{grid-template-columns:34px 1fr;gap:12px;padding:19px 14px}.phase-num{width:31px;height:31px;font-size:19px}.phase h3{font-size:21px}.phase .gate{padding:12px}.source-list{columns:1}.filters{align-items:stretch;flex-direction:column}.filters label{flex-wrap:wrap}.filters input{width:100%}details{padding:0 14px}pre{font-size:12px;padding:15px}.callout{padding:18px}.meta{font-size:12px}}@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}@media print{.rail,.jump,.chapters,.filters{display:none}main{margin:0;max-width:none;padding:0}body{font-size:11pt;background:white}.part{margin-top:25px;padding-top:25px;break-before:page}h1{font-size:42pt}h2{font-size:29pt}.feature,.phase,.figure{break-inside:avoid}.diagram-scroll svg{min-width:0}video{max-height:200px}.source-list{columns:1}}
'''

body = f'''
<aside class="rail"><div class="brand">RoboSimStudio<br>&amp; Newton</div><small>Engineering assessment</small><nav aria-label="Report sections">
<a href="#overview" class="active"><span>01</span>What RSS offers</a><a href="#ecosystem"><span>02</span>Where it fits</a><a href="#integration"><span>03</span>How to integrate</a><a href="#recommendations"><span>04</span>Recommendations</a></nav>
<div class="rail-bottom">Source review + working examples<div class="progress"><i id="progress"></i></div><a href="REPRODUCE.txt">Reproduce the study</a><br><a href="../rss-newton-report.zip">Download the bundle</a></div></aside>
<main><header><div class="eyebrow">Created <time datetime="2026-10-07">2026-10-07</time> · Engineering &amp; integration</div><h1>RoboSimStudio<br>&amp; Newton</h1><p class="lede">What the studio adds, how it works, and which parts belong in a Newton integration.</p>
<p class="meta">Source-linked implementation review · 3 recorded sessions · 5 runnable examples<br>Original source audit and experiments: 7 October 2026 UTC. The recordings and simulation results below are retained from that audit.</p>
<a class="jump" href="#overview">Start with RSS ↓</a><a class="jump secondary" href="#recommendations">Go to the roadmap</a>
</header>

<section class="part" id="overview"><div class="eyebrow">01 / What RSS offers</div><h2>A Python workbench for<br>interactive robot simulation.</h2>
<p class="part-lede">RoboSimStudio (RSS) combines a compact scene API, a browser-based studio, and a persistent development session. A researcher can build a robot-and-cloth scene, move things directly, inspect or change parameters, and record the result. <strong>Its primary physics backend is already Newton.</strong> The integration opportunity is to make those workflows reusable on Newton’s public interfaces.</p>
<p class="sources">{r('README.md', 'RSS repository and release scope')} {r('robosimstudio/sim/scene.py', 'Scene construction')} {nm('newton/__init__.py', 'Newton public API')}</p>
<figure class="figure"><div class="diagram-scroll">{architecture}</div><figcaption>Current implementation, simplified. Click a package to open its source. Arrows show construction and control flow; rendering updates also travel back to the browser. <a href="media/rss-architecture.svg">Open full-size diagram</a>.</figcaption></figure>
<details><summary>Navigate the code: six useful starting points</summary><div class="table-wrap"><table class="compact"><thead><tr><th>Start here</th><th>What to follow</th><th>Why it matters for Newton</th></tr></thead><tbody>
<tr><td>{r('robosimstudio/sim/scene.py', 'Scene')} → {r('robosimstudio/scene/compiler.py', 'compiler')} → {r('robosimstudio/sim/sim.py', 'Sim')}</td><td>Descriptions become a built model, solver pipeline, control and evolving state.</td><td>The convenience layer should remain optional; existing Newton applications already own these objects.</td></tr>
<tr><td>{r('robosimstudio_studio/session.py', 'StudioSession')} · {r('robosimstudio_studio/panels.py', 'panels')}</td><td>Creates the viewer, GUI and robot controls; coordinates the session tick and scene replacement.</td><td>This is where a standalone application must be separated from reusable controls.</td></tr>
<tr><td>{r('robosimstudio_studio/handles.py', 'HandleSet')} · {r('robosimstudio/core/draggers.py', 'draggers')}</td><td>Persistent targets in the studio connect to rigid-body forces or deformable-particle manipulation.</td><td>The strongest small contribution: reusable interaction semantics below the panel layout.</td></tr>
<tr><td>{r('robosimstudio/core/tuning.py', 'parameter writers')} · {r('robosimstudio_studio/tuning/apply.py', 'material updates')}</td><td>Field metadata drives controls, array writes, solver notifications and rebuild decisions.</td><td>Correct model changes are more important than the sliders themselves.</td></tr>
<tr><td>{r('robosimstudio_agent/server.py', 'LiveServer')} · {r('robosimstudio/core/bus.py', 'SimBus')}</td><td>Local commands reach the simulation thread; edited source is rebuilt before adoption.</td><td>Useful to both a browser studio and an agent host, including a future MCP adapter.</td></tr>
<tr><td>{r('robosimstudio/record/recorder.py', 'recorder')} · {r('robosimstudio_studio/garments.py', 'viewer specialization')}</td><td>Structured takes are distinct from browser appearance and offline video.</td><td>Recording semantics and rendering hooks need separate contracts.</td></tr>
</tbody></table></div></details>

<article class="feature" id="feature-scene"><div class="tag">Important feature 1</div><h3>Build a scene and immediately work with the robot.</h3><div class="split"><div>
<p><strong>For the user:</strong> name a robot, a work surface and a few objects in Python; open the studio with ready-made end-effector and gripper controls. This removes the repeated setup code normally required to turn a simulation example into an interactive experiment.</p>
<p><strong>Underneath:</strong> <code>Scene</code> collects specifications; <code>build()</code> creates the Newton-backed <code>Sim</code>. Robot metadata identifies control joints and tool frames so the studio can construct IK and gripper panels. These are application defaults built around existing simulation and control capabilities.</p>
<p><strong>The boundary to preserve:</strong> RSS’s authoring coordinates can be relative to its work surface, while state readbacks are in world coordinates. Its current scene API has one robot specification slot; adding another replaces it. A registered bimanual robot can still occupy that slot. Newton integration must not inherit this as a general scene restriction.</p>
</div><figure><img src="media/rss-agent-edited.png" loading="lazy" alt="RSS browser showing the robot, cloth and objects after a source edit"><figcaption>The supplied example builds a Panda, rigid objects, a 169-particle towel and a camera. Robot and scene setup remain editable Python.</figcaption></figure></div>
<p class="sources">{r('robosimstudio/sim/scene.py#L156', 'Scene API')} {r('robosimstudio_studio/panels.py', 'Robot panels')} {nm('newton/controllers.py', 'Newton controllers')} {ex('rss_scene.py', 'Complete scene example')}</p>
<details><summary>Read the supplied scene example</summary><p>This is the complete bundled example, not a proposed Newton API.</p><pre><code>{html.escape((ROOT/'examples/rss_scene.py').read_text())}</code></pre></details></article>

<article class="feature" id="feature-handles"><div class="tag">Important feature 2 · strongest focused contribution</div><h3>Keep hold of objects and cloth while arranging a scene.</h3><div class="split"><div>
<p><strong>For the user:</strong> select a cloth patch, create a handle, move its target and leave it holding the patch while inspecting the scene. RSS also provides rigid-body handles, multiple slots, linked motion and explicit release. This helps prepare difficult initial states or reproduce a contact problem without continually holding the mouse down.</p>
<p><strong>Underneath:</strong> the studio’s handle set manages targets and gizmos. A rigid-body dragger applies a damped force and torque toward its target. A particle dragger selects nearby particles and stores their offsets in the handle frame. Its <em>spring</em> mode adds forces before solving; its <em>pin</em> mode writes particle positions and velocities after solving. Fixed-size device buffers let the target change without changing the kernel launch shape.</p>
<p><strong>What Newton gains:</strong> a reusable persistent-target and particle-selection workflow beyond its existing rigid-body picking. The filmed cloth lift is an external pin; it does not demonstrate that a robot can grasp and support the cloth after the pin is released.</p>
</div><figure><img src="media/rss-lift.png" loading="lazy" alt="A persistent RSS manipulation handle lifting a patch of simulated cloth"><figcaption>The target remains active between mouse actions. A kinematic pin and a force-based drag have different physical effects and must be labeled clearly.</figcaption></figure></div>
<div class="mechanism" aria-label="Handle implementation"><span>Select body / particle patch</span><i>→</i><span>Store a target</span><i>→</i><span>Apply force or pin each step</span><i>→</i><span>Release explicitly</span></div>
<p class="small">For scripted particle selection, disable the nearest-particle fallback and check the returned selection count and distance. A convenient mouse interaction should not silently turn an agent’s missed grasp into a successful attachment.</p>
<p class="sources">{r('robosimstudio_studio/handles.py', 'Handle lifecycle')} {r('robosimstudio/core/draggers.py#L206', 'Rigid dragger')} {r('robosimstudio/core/draggers.py#L316', 'Particle dragger')} {n('newton/_src/viewer/picking.py', 'Newton picking')}</p>
<div class="chapters"><button data-video="rss-video" data-time="{chapter_time('rss-interaction', 'selection')}">Watch selection · {chapter_label('rss-interaction', 'selection')}</button><button data-video="rss-video" data-time="{chapter_time('rss-interaction', 'lift')}">Watch cloth lift · {chapter_label('rss-interaction', 'lift')}</button></div></article>

<article class="feature" id="feature-tuning"><div class="tag">Important feature 3</div><h3>Inspect parameters, change them, and record the experiment.</h3>
<p><strong>For the user:</strong> select something in the scene, inspect its settings, change a material or solver parameter and observe the response. RSS distinguishes physical settings from appearance controls such as cloth thickness, color or fuzz. It can record structured simulation takes as well as make images and videos.</p>
<p><strong>Underneath:</strong> parameter metadata supplies units, ranges and update rules. Some edits write into arrays read by the next simulation step. Others change values baked into a CUDA graph and require recapture. Changes to geometry, density or allocation sizes require a fresh model. In the cloth material implementation, some contact values are model-wide even though the panel is contextual; they cannot vary independently per cloth.</p>
<div class="cards three"><div class="card"><div class="label">Array value</div><b class="title">Apply on the next step</b><p>For example, a per-element cloth material value. Keep the existing model and buffers.</p></div><div class="card"><div class="label">Captured scalar</div><b class="title">Recapture the graph</b><p>A Python value passed into a captured kernel needs the application to refresh the captured work.</p></div><div class="card"><div class="label">Construction input</div><b class="title">Rebuild the model</b><p>Allocate a replacement model, rebind dependent resources, and invalidate old selections.</p></div></div>
<p><strong>What Newton gains:</strong> a usable tuning application and concrete examples of correct parameter updates. The generally useful contribution is a dependable change-notification contract; material presets and panel design can remain in RSS. A script-authored scene still needs source edits for unsupported rebuild operations. This is not yet a complete editor with undo and general USD round-trip persistence.</p>
<p class="sources">{r('robosimstudio_studio/tuning/binder.py', 'Metadata to controls')} {r('robosimstudio_studio/tuning/apply.py', 'Three update tiers')} {r('robosimstudio/core/tuning.py', 'Model writers')} {r('robosimstudio_studio/tuning/panels.py#L182', 'Rebuild limitations')} {r('robosimstudio/record/recorder.py', 'Recording format')}</p>
<div class="callout warm"><p><strong>A recorded take is not automatically robot training data.</strong> Our take contains 106 samples. The cube rises about 18 cm while the recorded robot action barely changes: the mouse supplied the intervention. A training export must record the external forces or pins, label assisted steps, or exclude those transitions. {ev('rss-take-audit.json', 'Inspect the audit')} · {ex('inspect_recorded_take.py', 'Recompute it from the bundled data')}.</p></div>
<div class="chapters"><button data-video="rss-video" data-time="{chapter_time('rss-interaction', 'tuning')}">Watch tuning panel · {chapter_label('rss-interaction', 'tuning')}</button><button data-video="rss-video" data-time="{chapter_time('rss-interaction', 'recording')}">Watch take recording · {chapter_label('rss-interaction', 'recording')}</button></div></article>

<div class="video-card"><div class="video-title">RSS: direct manipulation, tuning and recording <small>{video_duration('rss-interaction')} · edited highlights</small></div><video id="rss-video" controls playsinline preload="metadata" poster="{VIDEO_EDITS['rss-interaction']['poster']}"><source src="{VIDEO_EDITS['rss-interaction']['output']}" type="video/mp4"></video><p>Cloth selection, persistent lifting, material controls, release and take recording on Newton. Title cards and idle time are removed; retained footage plays at 1–1.4×, as marked. Only one persistent patch was demonstrated; multiple-handle and unassisted-grasp behavior still need acceptance tests.</p></div>

<article class="feature" id="feature-agent"><div class="tag">Important feature 4</div><h3>Edit the source while keeping the simulation session open.</h3>
<p><strong>For the user:</strong> a human or coding agent changes the scene file and keeps the same browser session. Commands can pause, run a bounded number of steps, inspect status, evaluate Python and capture an image. An invalid edit leaves the previous scene available so the next correction can be made in context.</p>
<p><strong>Underneath:</strong> <code>LiveServer</code> watches the file, waits for edits to settle, executes and builds a candidate scene, then adopts it only after construction succeeds. Local socket requests are handed to <code>SimBus</code>, which the simulation thread pumps between steps. This preserves a single owner for simulation mutation. Keeping the process alive retains browser and camera context; rebuilding still has a measurable cost.</p>
<p><strong>Verified here:</strong> a run command advanced exactly 12 steps; a syntax error preserved the old scene; a valid edit loaded in the same process with the camera unchanged; a tiled snapshot was produced. The warm rebuild took about 22.6 seconds on the study machine. These are functional results, not a measured productivity gain over direct Python.</p>
<p class="sources">{r('robosimstudio_agent/server.py#L61', 'Live server')} {r('robosimstudio_agent/scenefile.py', 'Source loading')} {r('robosimstudio/core/bus.py', 'Thread ownership')} {r('robosimstudio_agent/wire.py', 'Local command transport')} {ev('live-probe.json', 'Probe results')}</p>
</article>
<div class="video-card"><div class="video-title">RSS: recover from an invalid edit, then rebuild <small>{video_duration('rss-agent-loop')} · edited highlights</small></div><video id="agent-video" controls playsinline preload="metadata" poster="{VIDEO_EDITS['rss-agent-loop']['poster']}"><source src="{VIDEO_EDITS['rss-agent-loop']['output']}" type="video/mp4"></video><p>Bounded stepping, failed-edit recovery and the successful source change. Rebuild waiting time and title cards are removed; the measured warm build still took about 22.6 seconds. The probe then checks the snapshot and restores the source file. {ex('probe_live_session.py', 'Live-session example')} · {ev('agent-tiled-snapshot.png', 'Sensor snapshot')}.</p></div>

<div class="callout"><p><strong>Why Viser?</strong> It gives Python researchers a browser scene, GUI controls, transform gizmos, click callbacks and camera access without writing a separate web application. That is useful on a remote GPU machine and lets a person and an agent observe the same session. Newton already uses it, so shared interaction APIs are practical. Viser supplies the interface; the application still owns stepping, physical edits, persistence and the distinction between a browser image and a sensor observation. This explanation is an engineering inference from the implementation and {link('https://viser.studio/main/', 'Viser’s API')}.</p></div>

<details id="matrix"><summary>Full feature matrix · all 17 capabilities retained</summary><p>The four walkthroughs above cover the most useful integration candidates. The remaining rows distinguish existing overlap, companion features and unreleased or unvalidated scope.</p>
<div class="filters"><label>Search <input id="feature-search" type="search" placeholder="cloth, camera, recording…"></label><label>Show <select id="feature-filter"><option value="all">All capabilities</option><option value="upstream">Potential Newton contributions</option><option value="partner">Companion / adapters</option><option value="overlap">Existing overlap</option><option value="future">Unreleased / unvalidated</option></select></label><span id="feature-count" aria-live="polite">17 capabilities</span></div><div class="table-wrap">{matrix}</div></details>
</section>

<section class="part" id="ecosystem"><div class="eyebrow">02 / Where it fits</div><h2>A lightweight studio between<br>the physics library and the task application.</h2>
<p class="part-lede">These projects overlap, but solve different-sized problems. RSS is most useful as an interactive workbench above Newton. It can complement a learning framework or a full scene-authoring application without making Newton depend on either.</p>
<figure class="figure"><div class="diagram-scroll">{ecosystem}</div><figcaption>An intentionally approximate map of product emphasis. Blob size and overlap do not measure capability, maturity or compatibility. Projects can span all three areas. <a href="media/ecosystem-map.svg">Open full-size diagram</a>.</figcaption></figure>
<div class="table-wrap"><table class="compact"><thead><tr><th>Project</th><th>Main role</th><th>What this means for RSS + Newton</th></tr></thead><tbody>
<tr><td>Newton</td><td>Physics library, solver interfaces, model/state/control data, viewers and sensors.</td><td>Keep this as the common engine boundary. RSS adds application workflows. {nm('newton/__init__.py', 'Public API')} · {n('newton/_src/viewer/viewer_viser.py', 'Viewer')}</td></tr>
<tr><td>Genesis</td><td>A broader simulation platform with its own scene interface, physics and rendering.</td><td>RSS already has an optional Genesis bridge in the reviewed source. That does not prove equivalent behavior across engines. {r('robosimstudio/engines/genesis/bridge.py', 'RSS bridge')} · {link('https://github.com/Genesis-Embodied-AI/genesis-world/blob/main/README.md', 'Genesis overview')}</td></tr>
<tr><td>SuperDex</td><td>A platform spanning a physics engine, robotics SDK, Studio authoring tools and Lab; Lab is described as an early preview.</td><td>RSS’s optional bridge is already present. SuperDex Studio also overlaps with the authoring role. Preserve an optional boundary and validate coupling separately. {r('robosimstudio/engines/superdex/bridge.py', 'RSS bridge')} · {link('https://github.com/facebookresearch/project_superdex', 'SuperDex components')}</td></tr>
<tr><td>Isaac Sim</td><td>A larger USD-based simulation, asset and sensor-authoring application.</td><td>The reviewed develop source includes Newton integration. RSS should exchange declared scene fields or attach through supported application hooks; it should not introduce a competing stage or simulation owner. {link(SIM+'source/extensions/isaacsim.physics.newton/python/impl/newton_stage.py', 'Newton stage integration')}</td></tr>
<tr><td>Isaac Lab</td><td>Robot-learning environments and an application-owned simulation lifecycle.</td><td>The reviewed develop architecture separates physics, rendering and visualization and exposes its owned Newton data. RSS controls should borrow those resources and use Lab’s lifecycle/notification rules. {link(LAB+'docs/source/concepts/backend_architecture.rst', 'Backend architecture')} · {link(LAB+'docs/source/concepts/native-physics-api/newton.rst', 'Native Newton access')}</td></tr>
<tr><td>Isaac Lab Arena</td><td>Composition of scenes, embodiments and tasks, plus evaluation and environment generation.</td><td>Keep goals, rewards, task predicates and episode semantics in Arena or another task layer. Its agentic generation pipeline is adjacent to RSS’s live editing loop. {link(ARENA+'isaaclab_arena/environments/arena_env_builder.py', 'Environment builder')} · {link(ARENA+'docs/pages/concepts/agentic_environment_generation/system_overview.rst', 'Agent generation')}</td></tr>
</tbody></table></div>
<p class="small">Isaac integration statements are source-based: Lab and Sim develop, Arena main, at the recorded audit revisions. None of these applications or the optional Genesis/SuperDex solvers was run here. No cross-engine numerical equivalence or application compatibility is implied.</p>

<div class="cards"><article class="card"><div class="label">Robotics practitioner</div><b class="title">Direct manipulation makes setup tangible.</b><p>A researcher can arrange a cloth, perturb a grasp, inspect the result, and tune a parameter while keeping the relevant scene in view. Persistent handles are especially useful for producing a starting state that is tedious to script.</p><p>The natural first user is someone developing or debugging a small manipulation experiment. Large-scale training, full asset production and dataset governance remain responsibilities of the surrounding application.</p></article>
<article class="card"><div class="label">Coding agent</div><b class="title">The persistent runtime matters more than the GUI.</b><p>An agent benefits from named entities, numerical state, bounded stepping, recoverable rebuilds and snapshots when appearance matters. It can change Python directly instead of steering a 3D canvas through mouse clicks.</p><p>RSS already exposes many of these operations. Structured results, explicit frames, stale-object detection and repeatable batch evaluations would make them easier to use reliably. The same session can support a human inspecting an agent’s work.</p></article></div>

<div class="callout"><p><strong>Does RSS serve as an MCP server? Not in the audited release.</strong> Its agent package implements a local Unix-socket command protocol and Python evaluation, rather than MCP tool discovery and calls. An MCP adapter could expose the useful live operations, but wrapping the socket alone would not solve simulation ownership, reset or error semantics. {r('robosimstudio_agent/wire.py', 'Transport source')} · {r('robosimstudio_agent/server.py', 'Available commands')} · {link('https://modelcontextprotocol.io/specification/2026-07-28/server/tools', 'MCP tool protocol')}.</p></div>
<p>The ongoing {link('https://reports.eric-heiden.com/newton-live-mcp/', 'Newton live MCP study')} explores a closely related approach: host a Newton example persistently, execute against it, and rebuild edited source. Its results suggest that benefits depend on the workload and that agents often prefer numerical evaluation to rendered images. That study is experimental and does not benchmark RSS. The engineering opportunity is to share lifecycle and execution machinery while allowing both a Viser front end and an MCP client.</p>
<p><strong>Practical implication:</strong> support both human and agent access to the same application-owned Newton session. Keep task semantics and exports in adapters so BEHAVIOR-style tasks, Lab, Arena and other consumers can retain their own models. RSS’s public task tier is incomplete; a gallery and planned integrations should not determine Newton’s core API. {r('robosimstudio_studio/seams.py', 'Public task boundary')} · {r('docs/source/notes/roadmap.md', 'RSS roadmap')}.</p>
</section>

<section class="part" id="integration"><div class="eyebrow">03 / How to integrate</div><h2>Keep the studio separate.<br>Share the useful Newton interfaces.</h2>
<p class="part-lede">The choice is how much software and responsibility Newton should absorb. A companion package can deliver the complete RSS workflow, while focused contributions improve Newton for every application that needs similar interactions.</p>
<p class="small"><strong>Baseline used throughout:</strong> “Newton” includes the interactive ViewerViser work in {link('https://github.com/newton-physics/newton/pull/3850', 'PR #3850')}, expected to merge shortly after this review. The tested source is pinned at <code>8bfd16eb8941</code>. Original filenames and on-screen labels retain that provenance; they do not define a separate product.</p>
<div class="cards three"><article class="card choice"><div class="verdict">Do not choose</div><b class="title">Merge RSS wholesale into Newton</b><p>This imports a second scene facade, application loop, presets, recording conventions, device integrations and optional engine bridges alongside the physics library.</p><p>Users gain an included studio, but the core inherits unrelated dependencies, support expectations and overlapping APIs. The useful workflow does not require this coupling.</p></article>
<article class="card choice recommended"><div class="verdict">Preferred home for the application</div><b class="title">A separate companion repository</b><p>RSS can remain independently versioned, initially in its current home and potentially under the Newton organization once ownership and support are agreed.</p><p>This preserves the authors’ complete workflow and release cadence. Organization membership should follow a demonstrated maintenance contract, not substitute for one.</p></article>
<article class="card choice recommended"><div class="verdict">Preferred route into Newton</div><b class="title">Integrate selected components</b><p>Contribute small interaction or lifecycle capabilities that both RSS and a native Newton example need. Start with persistent particle targets and missing public viewer hooks.</p><p>This complements the companion repository. Avoid copying panels piecemeal until a working integration shows which lower-level gaps actually remain.</p></article></div>

<div class="video-card"><div class="video-title">Newton: the interactive viewer baseline <small>{video_duration('newton-pr3850-gpu')} · edited highlights</small></div><video id="newton-video" controls playsinline preload="metadata" poster="{VIDEO_EDITS['newton-pr3850-gpu']['poster']}"><source src="{VIDEO_EDITS['newton-pr3850-gpu']['output']}" type="video/mp4"></video><p>IK target movement, pause/step controls and collision inspection, with idle time and title cards removed and footage at 1.2–1.4×. Newton already supplies the viewer primitives; RSS adds the persistent manipulation and authoring workflow demonstrated above.</p></div>

<div class="flow-diagram" role="img" aria-label="Proposed integration: RSS human controls and an agent adapter send commands to a single application-owned Newton runtime. Viewer and snapshots return observations. Task and dataset adapters remain outside the runtime.">
<div class="eyebrow" style="margin-bottom:17px">Proposed boundary · not yet implemented end to end</div>
<div class="flow-inputs"><div class="flow-box">RSS studio controls<small>Viser panels · selection · persistent targets</small></div><div class="flow-box">Agent interface<small>CLI or MCP adapter · source edits · evaluations</small></div></div>
<div class="flow-arrow"><b>↓</b> Commands identify the model generation, world and object</div>
<div class="flow-box owner">Application-owned Newton session<small>One step loop · validates changes · owns reset, rebuild and shutdown</small></div>
<div class="flow-arrow"><b>↓</b> Public interfaces and explicit notifications</div>
<div class="flow-inputs"><div class="flow-box">Newton model, state and solvers<small>Forces · constraints · parameter changes</small></div><div class="flow-box">Viewer, sensors and observations<small>Human feedback · numerical state · images</small></div></div>
<p class="flow-caption">Task definitions, dataset conversion and optional foreign engines attach through separate adapters. RSS may own the application in standalone mode; Isaac Lab or another host keeps ownership when embedding controls.</p></div>

<h3 id="component-integration">Component integration details</h3>
<p class="small">These are proposed integration boundaries. “Newton” below means reusable engine, viewer or application-lifecycle interfaces; “companion” means an optional RSS application/package, potentially maintained under the Newton organization. Click a screenshot to enlarge it.</p>
<div class="table-wrap component-table-wrap"><table class="compact component-table"><thead><tr><th>RSS component and current behavior</th><th>What Newton would need</th><th>Recommended home / concrete check</th></tr></thead><tbody>
<tr><td><strong>Persistent rigid and cloth handles</strong><p>A gizmo keeps holding a body or particle patch after the mouse is released. Rigid handles apply a damped force/torque; cloth handles can apply a spring or a kinematic pin.</p>
<a class="component-shot" href="media/rss-camera-hd-both.png" data-caption="The raised gizmo holds a cloth patch while both camera images update. This is an external pin, not evidence that the robot is grasping the cloth." aria-label="Enlarge cloth-handle screenshot"><img src="media/rss-camera-hd-both.png" loading="lazy" alt="Cloth patch lifted by a persistent gizmo, with two live camera images"><span>External pin; the robot is not holding the towel. ↗</span></a>
<p class="sources">{r('robosimstudio/core/draggers.py', 'Dragger physics')} · {r('robosimstudio_studio/handles.py', 'Handle lifetime')}</p></td>
<td>Extend existing picking with reusable particle-patch selection and persistent targets. Define coordinate frame, world, selected indices, force/pin mode and release behavior. Apply targets at the correct point before or after solving; widgets should submit commands rather than write state from callbacks.</td>
<td><strong>Newton interaction primitives; RSS controls.</strong><p>Implement only behavior also useful in a native Example. Check two independent patches, release/reset/disconnect, and rejection of targets from a replaced model. Keep robot and task assumptions out of the primitive.</p></td></tr>
<tr><td><strong>Viewer extension</strong><p>RSS subclasses the Viser viewer to render its cloth appearance and adds selection, gizmos and panels. It also reaches into private viewer state.</p><p class="sources">{r('robosimstudio_studio/garments.py', 'RSS viewer subclass')} · {n('newton/_src/viewer/viewer_viser.py#L791', 'Public Viser access')}</p></td>
<td>Use the existing <code>ViewerViser.server</code> for controls. Inventory the remaining overrides and private caches, then add narrowly scoped hooks for any missing model-replacement, rendering or cleanup operations. Establish when a control may submit work to the application’s step loop.</td>
<td><strong>Small hooks in Newton’s viewer; studio layout and styling in RSS.</strong><p>The first check is a working port through public interfaces, including repeated model replacement with no stale meshes or callbacks. A new general plugin framework is not a prerequisite.</p></td></tr>
<tr><td><strong>Parameter tuning</strong><p>Metadata generates the inspector’s sliders and units. Edits either update arrays, require graph recapture, or wait for a model rebuild; material presets add save/revert controls.</p>
<a class="component-shot" href="media/rss-material-cutout.png" data-caption="The Material panel marks density as a rebuild edit and lists areal_density in Pending rebuild. In the tested hand-written Scene, Apply & rebuild directs the user back to the source script." aria-label="Enlarge pending-rebuild screenshot"><img src="media/rss-material-cutout.png" class="show-top" loading="lazy" alt="Material panel showing areal_density pending rebuild"><span>Pending rebuild is different from an applied edit. ↗</span></a>
<p class="sources">{r('robosimstudio/core/tuning.py', 'Model writers')} · {r('robosimstudio_studio/tuning/apply.py', 'Update tiers')}</p></td>
<td>Make edit scope and invalidation explicit: selected object versus global field, array write versus solver notification, recapture versus reconstruction. Return an applied/pending/rejected result. Rebuilds must rebind states, controls, solvers, renderers and selections; UI readbacks must agree with the active model.</td>
<td><strong>Newton change contracts and examples; RSS inspector/presets.</strong><p>Compare an edited model with a fresh build at the same settings, including captured execution. Test the observed inspector/Material-panel disagreement. Hand-written scenes need an explicit rebuild recipe before the GUI can promise structural edits.</p></td></tr>
<tr><td><strong>Live scene and agent host</strong><p>A file watcher builds candidate scenes, retains the previous scene after an invalid edit, and accepts local commands for stepping, inspection and snapshots.</p><p class="sources">{r('robosimstudio_agent/server.py', 'LiveServer')} · {r('robosimstudio/core/bus.py', 'Simulation-thread queue')}</p></td>
<td>Let tools attach to an existing Example or application-owned runtime. Define bounded stepping, reset versus rebuild, current-state access and model-generation checks. Use one command queue and one simulation owner. Typed results should expose failures and pending edits to a CLI or MCP adapter.</td>
<td><strong>Companion host first; share lifecycle hooks only after reuse is demonstrated.</strong><p>Repeat failed-edit recovery and successful rebuild on a native Newton application without requiring RSS Scene. Coordinate with the MCP study; the audited RSS command server itself is not an MCP implementation.</p></td></tr>
<tr><td><strong>Scene recipes, robot presets and appearance</strong><p><code>Scene</code> combines named assets, workcell poses, robot tool/gripper metadata and solver arrangement; <code>build()</code> returns a runnable Sim. Visual presets add browser appearance.</p><p class="sources">{r('robosimstudio/sim/scene.py', 'Scene recipe')} · {r('robosimstudio/scene/request.py', 'Construction inputs')}</p></td>
<td>Reuse ModelBuilder/importers and the Example lifecycle. Coordinate solver configuration with ovnewton’s construction path, keeping authored settings separate from runtime buffers. Define which identities, units and fields survive save/load; avoid a second mandatory scene graph or task vocabulary.</td>
<td><strong>RSS/asset packages for authoring convenience.</strong><p>A small shared application structure may belong in Newton once Python and USD consumers need it. First compare the same scene’s model, solver settings and reset through both paths. Browser appearance is not automatically a physical or sensor material.</p></td></tr>
<tr><td><strong>Camera views and observations</strong><p>The studio manages camera poses and shows tiled or optional RTX renders inside Viser. “Look through” moves the browser view; pose edits are session-local.</p>
<a class="component-shot" href="media/rss-cameras-cutout.png" data-caption="The Cameras panel displays front and overhead images from the tiled renderer. These are declared camera views, distinct from a screenshot of the interactive browser viewport." aria-label="Enlarge camera-panel screenshot"><img src="media/rss-cameras-cutout.png" class="show-bottom" loading="lazy" alt="RSS camera panel with front and overhead rendered images"><span>Rendered camera images, separate from the browser view. ↗</span></a>
<p class="sources">{r('robosimstudio_studio/cameras.py', 'Camera panel')} · {r('robosimstudio_studio/render_view.py', 'Optional RTX viewport')}</p></td>
<td>Connect the workflow to existing viewer and sensor facilities. Pass camera identity, pose/frame, resolution and simulation timestamp with each image. Distinguish an edited preview rig from task observations, and recreate or rebind camera resources when their model changes.</td>
<td><strong>Camera editor in RSS; public image/camera hooks in Newton only where missing.</strong><p>Check that a pose edit changes the intended view, reset restores declared poses, and snapshots identify the state they depict. Validate inside an application that already owns its renderer before adding another renderer abstraction.</p></td></tr>
<tr><td><strong>Recording and task data</strong><p>RSS records actions and simulation state in structured takes; image/video export is a separate operation. A take can include motion caused by human pins or forces.</p><p class="sources">{r('robosimstudio/record/recorder.py', 'Recorder')} · {ex('inspect_recorded_take.py', 'Recorded-take inspection')}</p></td>
<td>Expose consistent step/state access and intervention events. Specify whether actions, observations and images refer to before or after each step; identify resets and model changes. Record external assistance so it cannot be mistaken for behavior produced by the robot action alone.</td>
<td><strong>Companion recorder and dataset adapters.</strong><p>Keep rewards, goals, BEHAVIOR predicates and Lab/Arena episode semantics above Newton. First replay or inspect a take containing a pin and a reset; verify time alignment and assistance labels before calling an export policy-training data.</p></td></tr>
<tr><td><strong>Teleoperation and foreign-engine bridges</strong><p>Optional adapters turn device inputs into robot commands or connect cloth/soft-body simulation to Genesis and SuperDex. These have different maintenance and validation needs.</p><p class="sources">{r('robosimstudio/engines/feedback.py', 'Coupling feedback')}</p></td>
<td>Reuse explicit controller commands and ownership rules for devices. Treat foreign-engine coupling as numerical integration work: specify frames, timesteps, contact exchange and reaction shaping. Keep these dependencies optional; a capped feedback force does not establish accurate coupling.</td>
<td><strong>Separate device/engine packages.</strong><p>Device disconnect must release its controls. Engine bridges need impulse/contact and timestep-refinement checks. Neither should be required to install Newton or use the RSS tools on a Newton-only scene.</p></td></tr>
</tbody></table></div>
<div class="callout"><p><strong>A small part of this boundary already works.</strong> The included {ex('newton_public_ui.py', 'native Newton panel example')} obtains the public Viser server, queues a button command for the simulation thread, and rejects commands from an old model generation. CPU and CUDA checks passed, and the browser controls were exercised. It is a feasibility example, not a completed RSS port or a proposed stable plugin API. {ev('public-ui-cpu.log', 'CPU log')} · {ev('public-ui-cuda.log', 'CUDA log')} · {ev('public-ui-browser.log', 'Browser log')}.</p></div>
<p><strong>Design for a second consumer from the start.</strong> First validate the shared primitives with RSS and a native Newton example. Then attach them to an Isaac Lab-owned Newton model without creating a second model, loop or reset policy. Lab already exposes borrowed model/state/control and change notifications; those existing contracts should shape the adapter. {link(LAB+'docs/source/concepts/native-physics-api/newton.rst', 'Lab ownership and mutation rules')}.</p>
</section>

<section class="part" id="recommendations"><div class="eyebrow">04 / Recommendations &amp; next steps</div><h2>Build one shared workflow.<br>Let that decide what moves upstream.</h2>
<p class="part-lede"><strong>Integrate the reusable interaction and editing features into Newton, starting with persistent handles and safe parameter tuning.</strong> These should work in ordinary Newton examples and through Python, with RSS reusing the same implementation. Treat updating RSS’s Newton dependency as routine maintenance; it is not a roadmap milestone.</p>
<p>The following is a proposed Newton implementation sequence. The first two contributions provide the immediate user benefit: arrange a scene and tune its physics without writing a custom tool for every example. They need a small hook for applying commands on the simulation thread. Broader application conventions and camera/recording integration follow; a new application framework is not a prerequisite.</p>
<div class="roadmap" id="roadmap">
<article class="phase" id="roadmap-handles"><div class="phase-num">1</div><div><div class="owner">First contribution · Newton interaction utilities and viewer controls</div><h3>Persistent rigid-body and cloth manipulation</h3>
<p><strong>User outcome:</strong> select two corners of a cloth, move them independently, leave them held while inspecting the scene, then release them. A Python script can perform the same operations without a browser.</p>
<p><strong>Implement in Newton:</strong> extend existing rigid-body picking with particle-patch selection, multiple persistent targets, and explicit create/update/release operations. Reuse Newton’s existing force-dragging behavior; extract the useful particle selection and spring/pin mechanics from RSS. Keep target data and physics application independent of Viser. Add thin viewer controls for selecting, moving and deleting handles.</p>
<p>Define the target frame, world, selected indices and model generation. Separate force application before solving from kinematic pin enforcement after solving. Allocate bounded target buffers so moving a handle does not require graph recapture. Commands must be consumed by the application’s step loop; reset or model replacement invalidates old targets, and disconnect releases controls owned by that client.</p>
<p class="gate"><strong>Done when:</strong> a native Newton example and a headless test both manipulate two patches and a rigid body; release leaves no residual pin or force; stale commands cannot affect a rebuilt model. Verify CPU/CUDA and captured execution, plus isolation of the selected world wherever batching is supported.</p>
<p class="sources">{r('robosimstudio/core/draggers.py#L316', 'RSS particle mechanics')} · {n('newton/_src/viewer/picking.py#L17', 'Newton picking to extend')} · {ex('newton_public_ui.py', 'Existing queued-command prototype')}</p>
</div></article>
<article class="phase" id="roadmap-editing"><div class="phase-num">2</div><div><div class="owner">Next contribution · Newton parameter updates and a small inspector example</div><h3>Safe inspection and editing of simulation parameters</h3>
<p><strong>User outcome:</strong> change a cloth material or joint gain and know which object changed, whether the value is active, and whether reconstruction is required. An agent receives the same result as the GUI.</p>
<p><strong>Implement in Newton:</strong> a documented editing path for a small supported set of fields, starting with VBD cloth material values and joint gains for a declared solver. Reuse <code>ModelFlags</code> and <code>SolverBase.notify_model_changed()</code>; the missing part is field-level validation, scope, readback and coordination of solver refresh, graph recapture and rebuild. Those costs depend on the field and solver, so RSS’s three edit tiers are a useful starting point rather than a universal classification.</p>
<p>Validate before writing. Return structured applied/pending/rejected results with the effective value and required action. A structural edit needs an application-provided rebuild callback; without one, report it as unsupported. Keep model arrays authoritative so two panels and an agent cannot silently disagree. Ship a minimal Newton inspector using existing UI callbacks; keep RSS’s richer panels and preset library optional.</p>
<p class="gate"><strong>Done when:</strong> each supported edit matches a fresh build at the same settings within stated tolerances, including captured execution. Invalid edits leave the active model unchanged; global fields are labeled as global; UI and Python readbacks agree. This would directly address the panel disagreement observed in the study.</p>
<p class="sources">{r('robosimstudio_studio/tuning/apply.py', 'RSS edit tiers and scope')} · {n('newton/_src/solvers/solver.py#L582', 'Existing solver change notification')} · {ev('camera-gui-after.json', 'Observed edit and readback behavior')}</p>
</div></article>
<article class="phase" id="roadmap-application"><div class="phase-num">3</div><div><div class="owner">Shared design work · Newton examples, ovnewton and application owners</div><h3>Application lifecycle and solver configuration</h3>
<p><strong>User outcome:</strong> attach the same editing tools to a Python-built Newton example or a USD-loaded scene, without adopting <code>rss.Scene</code> or giving the studio ownership of the simulation loop.</p>
<p><strong>Implement in Newton:</strong> consolidate the hooks exercised by steps 1–2 into an optional application convention: access to the current model/state/control/solver, a safe point for commands, reset/rebuild callbacks, and notification when resources are replaced. State access must follow buffer swaps. Model generations identify stale selections, queued commands and camera bindings. Start with small helpers around the existing <code>Example</code> convention; require a second consumer before committing to a new public application class.</p>
<p>Define solver configuration separately from running resources, in coordination with ovnewton: solver choice and supported options, timestep/substeps and collision policy. Python and USD construction should use the same interpretation and explicit override precedence. Start with one solver per simulation; add coupled solver schedules only when their ownership and stepping semantics are specified. Reuse existing USD fields and define schemas/import coverage for new authored physics settings. Keep task goals, rewards, controllers and benchmark vocabulary outside this contract.</p>
<p class="gate"><strong>Done when:</strong> equivalent Python and USD rigid scenes agree on model properties, solver settings, initial state and reset; both accept the same tools. Failed rebuilds retain a usable application, successful rebuilds rebind every consumer, and one Isaac Lab adapter borrows its runtime without a second loop. Then extend the comparison to the supported cloth subset.</p>
<p class="sources">{n('newton/examples/__init__.py#L532', 'Newton example runner')} · <a href="#application-structure">RSS, USD and ovnewton API comparison</a> · {link(LAB+'docs/source/concepts/native-physics-api/newton.rst', 'Lab ownership rules')}</p>
</div></article>
<article class="phase" id="roadmap-observations"><div class="phase-num">4</div><div><div class="owner">Follow-on · Newton sensor/viewer hooks; optional recorder and studio panels</div><h3>Camera previews and recording of interventions</h3>
<p><strong>Implement in Newton:</strong> a small example connecting existing camera sensors to ViewerViser images, plus only the missing public hooks for camera updates and model rebinding. Associate observations with camera/world identity, frame, simulation step and time. Reuse <code>SensorTiledCamera</code> or an application-owned renderer; keep the browser view separate from sensor observations.</p>
<p>Expose handle create/move/release, parameter edits and reset/model-change events alongside step/state access. An optional recorder can then identify externally assisted motion and align it with actions and images. RSS can contribute its camera panel and take recorder; dataset formats and task semantics remain adapters above Newton. This work can proceed once the lifecycle and event boundaries are agreed.</p>
<p class="gate"><strong>Done when:</strong> the cloth example shows two synchronized camera images, a scripted pose edit changes the intended camera, and reset/rebuild restores or explicitly replaces the rig. A recorded take identifies every pin and reset and states whether each observation is before or after its action.</p>
<p class="sources">{n('newton/_src/sensors/sensor_tiled_camera.py#L49', 'Existing Newton tiled sensor')} · {r('robosimstudio_studio/cameras.py', 'RSS camera panel')} · {ex('inspect_recorded_take.py', 'Recorded-take inspection')}</p>
</div></article>
</div>
<p><strong>Immediate next steps:</strong> split the first implementation work into two focused Newton contributions: persistent manipulation with a native cloth example, and validated parameter edits with a minimal inspector. Use RSS as the reference implementation and first external consumer. Review the shared application hooks with ovnewton and Isaac Lab in parallel, using those examples to settle the minimum API. The milestones are working Newton features with regression tests, not compatibility patches or repository moves.</p>
<p><strong>Keep the remaining studio optional.</strong> RSS’s scene authoring conveniences, presets, file watcher, richer GUI, task definitions, dataset exporters and Genesis/SuperDex bridges can stay in a companion repository, including under the Newton umbrella if ownership is useful. Hosting is a separate decision and should not delay the feature contributions. Newton gains reusable capabilities for both practitioners and coding agents without requiring one scene or task ecosystem.</p>

<details id="acceptance"><summary>Implementation checks and current evidence</summary><div class="table-wrap"><table><thead><tr><th>Check</th><th>Required experiment</th><th>Existing evidence / remaining gap</th></tr></thead><tbody>
<tr><td>Interaction lifecycle</td><td>Drag, disconnect, reset/rebuild, reconnect, then send an old callback.</td><td>Native prototype rejects stale commands. Reusable Newton handle and rebuild APIs remain proposed work.</td></tr>
<tr><td>Cloth manipulation</td><td>Two independent patches, linked motion, release order; test a robot grasp after releasing every pin.</td><td>One persistent patch was filmed. Two-patch and unassisted-grasp validation is open.</td></tr>
<tr><td>Parameter correctness</td><td>Compare an edited model with an equivalent fresh build; check solver refresh, recapture and all affected state buffers.</td><td>Live material writes and pending rebuild behavior were probed. Fresh-build equivalence and captured execution were not measured.</td></tr>
<tr><td>Agent authoring</td><td>Bounded steps, invalid edit, valid reload, stable camera, typed status and snapshot.</td><td>Live probe passed the existing mechanics. A shared typed schema is proposed work.</td></tr>
<tr><td>Multiple worlds</td><td>Select one of N worlds, reset selected worlds, and verify all others remain unchanged.</td><td>Not covered. Required before claiming support for batched applications.</td></tr>
<tr><td>Cameras and recording</td><td>Edit/reset the camera rig; identify image step/time, pins, model changes and observation/action timing.</td><td>Two tiled camera views and a raw take were captured. Synchronized intervention-aware recording was not established.</td></tr>
<tr><td>Performance</td><td>Cold/warm startup, command latency, rebuild latency, frame latency, throughput with UI on/off.</td><td>Observed timings only on a shared GPU; no controlled comparative benchmark.</td></tr>
<tr><td>Application and solver configuration</td><td>Compare Python and USD construction, reset and failed/successful rebuild; attach to a Lab-owned runtime.</td><td>Source-level overlap was analyzed. Equivalent runtime behavior and the Lab adapter were not executed.</td></tr>
</tbody></table></div></details>

<details id="evidence"><summary>Evidence, runnable examples and limitations</summary>
<p>These are the original study’s results. The report revision preserves the media, examples and simulation logs; it does not claim a new simulation run or a completed integration.</p>
<div class="evidence-stats"><div class="stat"><b>908</b><span>RSS fast-tier tests · OK, 19 skipped<br>{ev('rss-fast.log', 'Original log')}</span></div><div class="stat"><b>165</b><span>RSS medium-tier tests · OK, 14 skipped<br>{ev('rss-medium.log', 'Original log')}</span></div><div class="stat"><b>25</b><span>Newton Viser tests · passed<br>{ev('newton-viser-tests.log', 'Original log')}</span></div></div>
<p class="small">Suite totals are those reported by the runners, including skipped tests. RSS’s fast suite includes structural checks and stubs; these numbers are not counts of validated physical phenomena.</p>
<div class="table-wrap"><table><thead><tr><th>Example</th><th>What it establishes</th><th>Evidence</th></tr></thead><tbody>
<tr><td>{ex('rss_scene.py')}</td><td>Panda, rigid objects, VBD cloth and camera; the scene used for the RSS recordings.</td><td>{ev('rss-live.log', 'Session log')} · <a href="media/rss-interaction.mp4">Original interaction recording</a></td></tr>
<tr><td>{ex('test_study_contracts.py')}</td><td>Six characterization probes: coordinate frames, settling/reset, duplicate names, robot slot, unavailable task tier and capped feedback.</td><td>{ev('study-contracts.log', 'Newton 1.6.0 run')} · {ev('rss-stable161.log', '1.6.1 rerun')}</td></tr>
<tr><td>{ex('probe_live_session.py')}</td><td>Exactly 815→827 steps; failed-edit recovery; same-process reload; camera preservation; tiled image; restored source.</td><td>{ev('live-probe.json', 'Results')} · <a href="media/rss-agent-loop.mp4">Original agent recording</a></td></tr>
<tr><td>{ex('newton_public_ui.py')}</td><td>Public viewer access, queued mutation and stale-command rejection, exercised on CPU and CUDA and through the browser.</td><td>{ev('public-ui-cpu.log', 'CPU')} · {ev('public-ui-cuda.log', 'CUDA')} · {ev('public-ui-browser.log', 'Browser')}</td></tr>
<tr><td>{ex('inspect_recorded_take.py')}</td><td>Read-only analysis of the 106-sample take: timestamps, action range, cube motion and retained array shapes.</td><td>{ev('capability-take-inspection.json', 'Inspection')} · {ev('rss-take-schema.json', 'Schema')} · {ev('rss-take-audit.json', 'Audit')}</td></tr>
</tbody></table></div>
<p><strong>What the recordings do not establish.</strong> The first mixed-scene build took about 280 seconds, and first-use IK compilation blocked simulation-thread commands for about 75 seconds. These are shared-host observations, not a portable latency benchmark. A Newton joint-overlay toggle produced NaN bounding-sphere errors; the issue was not isolated against the base and is a follow-up candidate, not an established regression. {ev('rss-first-ik-stack.txt', 'IK trace')} · {ev('pr3850-joint-overlay-probe.txt', 'Overlay probe')}.</p>
<p><strong>Runtime limits.</strong> No Isaac application, real teleoperation device, optional RTX renderer or foreign solver was exercised. The RSS studio ran with its resolved Newton stack; six probes were repeated on stable 1.6.1. The native viewer prototype ran on the interactive baseline. This is not full RSS certification on that development stack. The public task benchmark, a shipped 48-task suite and policy export could not be validated.</p>
<p><strong>Raw data:</strong> {ev('rss-recorded-take/0000/steps.npz', 'Recorded steps')} · {ev('rss-recorded-take/0000/privileged.npz', 'Body and particle state')} · {ev('runtime-environments.json', 'Exact runtime environments')} · {ev('checked-heads.json', 'Audited source revisions')} · {ev('branch-audit.json', 'Branch audit')} · {ev('freshness-check.json', 'Original freshness check')}.</p>
<p><strong>Video provenance:</strong> the embedded videos are shortened presentation edits. The original <a href="media/rss-interaction.mp4">RSS interaction</a>, <a href="media/rss-agent-loop.mp4">agent session</a> and <a href="media/newton-pr3850-gpu.mp4">Newton viewer</a> captures remain unchanged. The {ev('video-edit-map.json', 'edit map')} records every retained source interval, playback speed and updated chapter time. These edits are not performance measurements.</p>
<p><a href="REPRODUCE.txt">Reproduction commands and artifact notes</a> · <a href="manifest.json">File hashes</a> · <a href="../rss-newton-report.zip">Complete report bundle</a>.</p>
</details>
<details id="sources"><summary>Source index and report verification</summary><p>Implementation links use the exact reviewed revisions. External product overviews and the ongoing MCP study were consulted for the ecosystem discussion. Older report screenshots remain historical evidence; the four-part revision has its own rendering checks.</p><p>{ev('four-part-source-index.json', 'Sources linked by this report')} · {ev('source-index.json', 'Original audit source index')} · {ev('capability-source-index.json', 'Additional implementation sources')} · {ev('four-part-static-qa.json', 'Current structural / link checks')} · {ev('four-part-browser-qa.json', 'Current browser checks')}.</p><div id="source-list-placeholder"></div></details>
<footer>RoboSimStudio × Newton · Engineering assessment<br>Evidence is separated from proposed integration work. All three recordings, five examples and the full feature matrix are included in the bundle.</footer>
</section></main>
<div id="video-status" class="video-status" aria-live="polite" style="position:fixed;bottom:12px;right:16px;background:#fff;padding:5px 10px;border-radius:5px;display:none"></div>
'''

js = '''
const rows=[...document.querySelectorAll('#feature-table tbody tr')];
function filterFeatures(){const query=document.getElementById('feature-search').value.toLowerCase();const kind=document.getElementById('feature-filter').value;let count=0;for(const row of rows){row.hidden=!(row.textContent.toLowerCase().includes(query)&&(kind==='all'||row.dataset.kind===kind));if(!row.hidden)count++;}document.getElementById('feature-count').textContent=count+' of '+rows.length+' capabilities';}
document.getElementById('feature-search').addEventListener('input',filterFeatures);document.getElementById('feature-filter').addEventListener('change',filterFeatures);
for(const button of document.querySelectorAll('[data-video]'))button.addEventListener('click',async()=>{const video=document.getElementById(button.dataset.video);for(const other of document.querySelectorAll('video'))if(other!==video)other.pause();video.scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth',block:'center'});const seek=()=>{video.currentTime=Number(button.dataset.time);video.play().catch(()=>{});};if(video.readyState>=1)seek();else{video.addEventListener('loadedmetadata',seek,{once:true});video.load();}});
const sections=[...document.querySelectorAll('main>section')];const nav=[...document.querySelectorAll('.rail nav a')];function updateNavigation(){const y=window.scrollY+Math.min(260,window.innerHeight*.3);let current=sections[0].id;for(const section of sections)if(section.offsetTop<=y)current=section.id;for(const a of nav){const active=a.hash==='#'+current;a.classList.toggle('active',active);if(active)a.setAttribute('aria-current','location');else a.removeAttribute('aria-current');}const extent=document.documentElement.scrollHeight-window.innerHeight;document.getElementById('progress').style.width=(extent>0?Math.min(100,100*window.scrollY/extent):0)+'%';}window.addEventListener('scroll',updateNavigation,{passive:true});window.addEventListener('resize',updateNavigation);updateNavigation();
for(const anchor of document.querySelectorAll('a[href^="#"]'))anchor.addEventListener('click',()=>{const target=document.getElementById(anchor.hash.slice(1));if(target?.tagName==='DETAILS')target.open=true;});
'''
from engineering_additions import extend
body, css = extend(body, css, ROOT, r, n, nm, link, ex, ev, SIM, LAB)

document = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="RoboSimStudio features, implementation, ecosystem position and an engineering roadmap for integration with Newton."><link rel="icon" href="data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 64 64%22%3E%3Crect width=%2264%22 height=%2264%22 rx=%2212%22 fill=%22%2321533d%22/%3E%3Ctext x=%2218%22 y=%2247%22 font-size=%2242%22 fill=%22white%22%3ER%3C/text%3E%3C/svg%3E"><title>RoboSimStudio & Newton — Features and Integration</title><style>'+css+'</style></head><body>'+body+'<script>'+js+'</script></body></html>'
soup = BeautifulSoup(document, 'html.parser')
sources = {}
for anchor in soup.select('a[href^="https://"]'):
    sources[anchor['href']] = anchor.get_text(' ', strip=True).removesuffix(' ↗')
curated = [
    (RSS+'README.md','RoboSimStudio: release scope'),
    (RSS+'robosimstudio/sim/scene.py','RSS scene construction'),
    (RSS+'robosimstudio_studio/session.py','RSS studio session'),
    (RSS+'robosimstudio/core/draggers.py','RSS interaction physics'),
    (RSS+'robosimstudio_studio/tuning/apply.py','RSS parameter updates'),
    (RSS+'robosimstudio_agent/server.py','RSS live agent server'),
    (NEWTON+'newton/_src/viewer/viewer_viser.py','Newton ViewerViser'),
    (NEWTON+'newton/_src/viewer/picking.py','Newton picking'),
    (LAB+'docs/source/concepts/backend_architecture.rst','Isaac Lab backend ownership'),
    (SIM+'source/extensions/isaacsim.physics.newton/python/impl/newton_stage.py','Isaac Sim Newton integration'),
    (ARENA+'isaaclab_arena/environments/arena_env_builder.py','Arena environment construction'),
    ('https://reports.eric-heiden.com/newton-live-mcp/','Ongoing Newton live MCP study'),
]
source_html = '<ul class="source-list">'+''.join('<li>'+link(u,l)+'</li>' for u,l in curated)+'</ul>'
document = document.replace('<div id="source-list-placeholder"></div>', source_html)
syntax_css = '''
/* Build-time syntax highlighting: works offline and with JavaScript disabled. */
.syntax-python .tok-comment{color:#546452;font-style:italic}
.syntax-python .tok-keyword{color:#743c70;font-weight:600}
.syntax-python .tok-string{color:#735214}
.syntax-python .tok-number{color:#315b93}
.syntax-python .tok-function{color:#286446}
.syntax-python .tok-name{color:#244d3d}
.syntax-python .tok-operator{color:#3d6672}
.syntax-python .tok-punctuation{color:#546452}
pre .syntax-python .tok-comment{color:#abc0ae}
pre .syntax-python .tok-keyword{color:#e7a5d0}
pre .syntax-python .tok-string{color:#e7d79a}
pre .syntax-python .tok-number{color:#b8d7ff}
pre .syntax-python .tok-function{color:#ccecab}
pre .syntax-python .tok-name{color:#e7f0e4}
pre .syntax-python .tok-operator{color:#b6dfe5}
pre .syntax-python .tok-punctuation{color:#c0cebf}
@media print{pre{background:#f0f3ed;color:#22372e}pre .syntax-python .tok-comment{color:#546452}pre .syntax-python .tok-keyword{color:#743c70}pre .syntax-python .tok-string{color:#735214}pre .syntax-python .tok-number{color:#315b93}pre .syntax-python .tok-function{color:#286446}pre .syntax-python .tok-name{color:#244d3d}pre .syntax-python .tok-operator{color:#3d6672}pre .syntax-python .tok-punctuation{color:#546452}}
'''
document = document.replace('</style></head>', syntax_css + '</style></head>')
document = re.sub(r'<code>(.*?)</code>', highlight_snippet, document, flags=re.S)
document = document.replace('<pre><code class="language-python', '<pre tabindex="0" aria-label="Python source example"><code class="language-python')
(ROOT/'index.html').write_text(document)
(ROOT/'evidence/four-part-source-index.json').write_text(json.dumps(sources,indent=2)+'\n')

reproduction = (WORK/'report-before-four-part-reproduce.txt').read_text()
reproduction = reproduction.split('\nReport revision (')[0]
reproduction = reproduction.replace('  cd /absolute/path/to/report\n', '  cd /absolute/path/to/report/..\n')
reproduction = reproduction.replace('The optional server supports byte-range requests for video seeking.', 'Open http://127.0.0.1:8093/rss-newton-report/ (using your report directory name).\nServing the parent also makes the sibling zip bundle downloadable.\nThe optional server supports byte-range requests for video seeking.')
reproduction += '''
Four-part presentation revision:
- The main report now has four sections: RSS overview/implementation, ecosystem/audiences,
  integration options/components, and final recommendations/roadmap.
- Added source-linked architecture and approximate ecosystem SVG diagrams in media/.
- The old integration-track section is absent. The ecosystem map describes product roles
  and ownership, rather than reproducing that section's integration-track narrative.
- Newton's interactive ViewerViser is the report baseline; exact original snapshot details
  remain above and in the raw evidence for reproducibility.
- Four feature walkthroughs replace the lengthy capability subsections. All 17 feature-matrix
  rows, three original recordings, five examples and original simulation evidence are retained.
- The report explains that RSS's local command protocol is not an MCP server. The user's
  Newton live MCP report is linked as ongoing research, not a shipped Newton API or RSS benchmark.
- media/rss-architecture.svg and media/ecosystem-map.svg are standalone copies of the diagrams.
- evidence/four-part-* files describe this revision. All older report-* QA files/screenshots
  are historical renderings, not verification of the current report.
- Original simulation tests were not rerun for this editorial and layout revision.
- Python blocks and inline code use embedded, build-time syntax highlighting. No external
  scripts or styles are needed; source text and indentation are preserved. Commit IDs stay plain.
- evidence/syntax-highlighting-* records the highlighting revision's browser and source checks.
- Embedded videos now use the shorter *-highlights.mp4 edits, with new poster frames and
  chapter times. Opening/interstitial title cards and idle intervals are removed. Small corner
  labels identify the action and playback speed. No simulated motion was synthesized.
- The three original MP4 captures and posters remain unchanged. evidence/video-edit-map.json
  records the original hashes, retained intervals, speed changes and new chapter starts.
- The live report uses the edited files; original captures remain linked under Evidence.

Read-only take inspection (no simulation process required):
  uv run --no-project --with numpy python /absolute/path/to/report/examples/inspect_recorded_take.py
  Original output: evidence/capability-take-inspection.json.
'''
(ROOT/'REPRODUCE.txt').write_text(reproduction)
print(json.dumps({'html_bytes':len(document.encode()),'main_sections':[s['id'] for s in soup.select('main>section')],'feature_rows':len(table.select('tbody tr')),'sources':len(sources)},indent=2))

with (ROOT/'REPRODUCE.txt').open('a') as out:
    out.write("""
Engineering API follow-up:
- API comparison: RSS Scene/BuildRequest/Sim/TaskSpec, Newton ModelBuilder and Example
  runner, plus ovnewton internal main 3e6576c2e4dcf24deda0d0c7b6b2aa43b87c33fb and
  multiple-scenes 1877c4902c98161e28dfd8401cfc66d60eb6a48c source review.
- New standalone example, in the same RSS environment as the original audit:
    uv run python /absolute/path/to/report/examples/rss_cloth_cameras.py
  It starts the studio at http://127.0.0.1:8097 with explicit MuJoCo/VBD routing.
  Stop it with the Simulation / Stop server button. Camera views use the tiled renderer.
- The recorded session used the same scene through the live server:
    uv run python -m robosimstudio_agent.serve --scene /absolute/path/to/report/examples/rss_cloth_cameras.py --port 8097 --run-dir /tmp/rss-camera-study-run --camera-view --no-watch
- Camera demo: media/rss-cloth-cameras.mp4 is a continuous 9.8-second recording
  at normal speed, without title cards. The source WebM, recording actions,
  clip provenance, GUI probes and runtime logs are bundled under media/ and evidence/.
- GUI probe: inspector tri_ke log10 4 -> 4.3 changed the model array. Material
  areal_density 0.3 -> 0.45 stayed pending; Apply & rebuild refused this hand-built
  scene and left particle mass unchanged. No captured-graph/fresh-build equivalence claim.
  camera-gui-probe.py records the post-interaction assertions from the original
  host session; it expects that session/socket and its saved before/after JSON inputs.
- Four-part-browser-qa.json verifies all four short clips over Tailscale, mobile
  layout, navigation, images and filtering. engineering-browser-qa.json checks
  the new API, screenshots and syntax highlighting.
- The component-integration table now explains eight components, their Newton work,
  recommended ownership and an initial acceptance check. Three annotated thumbnails
  enlarge in a keyboard-accessible dialog; component-table-browser-qa.json records
  desktop/mobile, image loading, close controls and syntax-highlighting checks.
- Rebuild/package sources are in tools/. Run rebuild_four_part_report.py from this
  report's tools directory to rebuild HTML; its companion module contains the
  API/GUI revision. Paths in these reproduction scripts reflect the study host.
""")
