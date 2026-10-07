from bs4 import BeautifulSoup
from pathlib import Path
import html

OVSHA='3e6576c2e4dcf24deda0d0c7b6b2aa43b87c33fb'
OV='https://github.com/NVIDIA-Omniverse/ovnewton-internal/blob/'+OVSHA+'/'
OVMULTI='https://github.com/NVIDIA-Omniverse/ovnewton-internal/blob/1877c4902c98161e28dfd8401cfc66d60eb6a48c/'

def extend(body,css,ROOT,r,n,nm,link,ex,ev,SIM,LAB):
    s=BeautifulSoup(body,'html.parser')
    def fragment(markup): return BeautifulSoup(markup,'html.parser')
    def after(node,markup): node.insert_after(fragment(markup))
    def append(node,markup): node.append(fragment(markup))
    def heading(selector,text): s.select_one(selector).string=text
    heading('h1','RoboSimStudio and Newton')
    heading('#overview>h2','1. RSS implementation and API')
    heading('#ecosystem>h2','2. Ecosystem and users')
    heading('#integration>h2','3. Integration with Newton')
    heading('#recommendations>h2','4. Recommendations and next steps')
    heading('#feature-scene h3','Scene construction and robot controls')
    heading('#feature-handles h3','Persistent rigid-body and cloth handles')
    heading('#feature-tuning h3','GUI model editing and recording')
    heading('#feature-agent h3','Source editing in a running session')
    for tag in s.select('.tag'): tag.decompose()
    for label in s.select('.feature p>strong'):
        if label.get_text() in ('For the user:','Underneath:','What Newton gains:'): label.decompose()
    s.select_one('header .lede').string='Implementation notes, API overlap, user workflows, and a proposed integration plan.'
    s.select_one('header .meta').string='Source audit, original experiments, API follow-up and camera demo: 2026-10-07 UTC. Four short videos; six runnable examples. Existing logs remain available.'
    for selector in ['#feature-scene details']:
        s.select_one(selector).decompose()
    api=f'''
<div id="api-comparison">
<h3>How close is RSS Scene to ModelBuilder?</h3>
<p><strong>They overlap during authoring, but return different levels of abstraction.</strong> Both accumulate a description before building GPU data. RSS keeps named entity specifications, workcell-relative poses and robot presets. Its compiler uses Newton builders, creates solver islands and coupling, then wraps the result in a <code>Sim</code> with controls, cameras and an episode-like lifecycle. Newton’s <code>ModelBuilder.finalize()</code> produces a <code>Model</code>; the application chooses its solver, allocates state and control, and owns the loop.</p>
<div class="table-wrap"><table><thead><tr><th>Concern</th><th>RSS</th><th>Newton / application</th><th>Integration implication</th></tr></thead><tbody>
<tr><td>Authoring</td><td><code>Scene.add(spec)</code> stores named boxes, sheets, robot and camera specifications; returns a named reference for entities.</td><td><code>ModelBuilder</code> assembles bodies, joints, shapes, particles and imported assets.</td><td>RSS is a higher-level authoring option. A Newton user should be able to attach studio tools to an existing model without converting it into RSS specs.</td></tr>
<tr><td>Construction result</td><td><code>Scene.build()</code> → compiler → <code>CompiledScene</code> → <code>Sim</code>. Warmup/reset runs by default.</td><td><code>builder.finalize()</code> → <code>Model</code>; <code>model.state()</code> and <code>model.control()</code> allocate runtime buffers.</td><td>A model and a runnable application are distinct products. Do not make finalize open a viewer or choose an application loop.</td></tr>
<tr><td>Solver choice and scheduling</td><td><code>ArrangementSpec</code> maps entity kinds to backend factories and specifies coupling, control rate and substeps.</td><td>The application creates solver instances, collision handling, timestep, state swaps and any multi-solver schedule.</td><td>Useful convenience, but RSS’s domain routing and default coupling are policy. Retain explicit escape hatches for existing solver pipelines.</td></tr>
<tr><td>Control and observation</td><td><code>Sim.step(action, mode=...)</code> selects a controller; observations come from providers. Robot presets supply tool frames and gripper conventions.</td><td><code>Control</code> is simulation input data; examples or a learning framework interpret actions and assemble observations.</td><td>A common runtime must not prescribe an RL action space, gripper range or observation dictionary.</td></tr>
<tr><td>Viewer ownership</td><td><code>rss.view(sim)</code> creates a StudioSession that drives the loop. <code>rss.viewer(sim)</code> supports a caller-owned loop.</td><td>An <code>Example</code> owns simulation resources; the example runner calls its <code>step()</code> and <code>render()</code>.</td><td>Provide an attach mode as well as standalone RSS. Existing Example or Lab code must remain the sole step owner.</td></tr>
<tr><td>Rebuild and identity</td><td>Building the same Scene again rebinds its entity references to the new build. Two Sims from one Scene do not have independent references.</td><td>Body/particle indices and state buffers belong to a particular finalized model; reconstruction invalidates consumers’ cached references.</td><td>Record model generation, world and entity identity. Rebind panels, renderers, handles and agent requests together.</td></tr>
<tr><td>Task definition</td><td><code>TaskSpec</code> adds instruction, evaluator, routine, variation and episode rules to scene fields. <code>BuildRequest</code> supplies the scene-only path.</td><td>Newton models describe physics. Task semantics live in application code, Lab, Arena or another framework.</td><td>Reuse scene construction without adopting RSS’s task representation or benchmark conventions.</td></tr>
</tbody></table></div>
<p class="sources">{r('robosimstudio/sim/scene.py#L320','Scene.build and reference lifetime')} · {r('robosimstudio/scene/request.py#L99','BuildRequest')} · {r('robosimstudio/scene/compiler.py#L228','Compiler entry point')} · {r('robosimstudio/sim/sim.py#L47','Sim')} · {n('newton/_src/sim/builder.py','ModelBuilder')} · {n('newton/examples/__init__.py#L532','Example runner')}</p>
<h3>RSS end to end: cloth, Newton solvers and camera views</h3>
<p>This is the actual new demo, with its introductory docstring removed. Solver objects are constructed inside <code>build()</code>: the articulated robot goes to Newton’s MuJoCo solver, the cloth and block to VBD, with RSS’s default proxy coupling between them. A camera specification declares a view; the studio’s camera panel creates the tiled renderer on demand. <code>share=False</code> avoids a public Viser tunnel.</p>
<pre><code>{html.escape('import robosimstudio as rss'+(ROOT/'examples/rss_cloth_cameras.py').read_text().split('import robosimstudio as rss',1)[1])}</code></pre>
<p>The program creates two solver islands: 16 robot bodies in MuJoCo and one rigid body plus 361 cloth particles in VBD. The measured build took about 21 seconds on this shared machine. Eight substeps at 60 control Hz means a nominal 1/480 s physics step; it is not a claim of 60 Hz wall-clock throughput. The GUI run was around 18–20 control steps/s with live images enabled. These observations are specific to this scene and host.</p>
<p class="sources">{ex('rss_cloth_cameras.py','Run this example')} · {r('docs/source/guides/physics.md','Backend and coupling choices')} · {r('robosimstudio/scene/physics/compose.py','Physics composition')} · {r('robosimstudio/sim/draw.py','Viewer entry points')} · {ev('camera-demo-probe.json','Runtime probe')}</p>
</div>'''
    after(s.select_one('#feature-scene'),api)
    tuning=s.select_one('#feature-tuning')
    tuning.select_one('.cards').decompose()
    tuning.find_all('p',recursive=False)[1].decompose()
    # Detailed GUI walkthrough, directly beside the feature it documents.
    gui=f'''
<div id="gui-editing">
<p><strong>What happens when someone edits the model?</strong> Clicking a mesh selects an entity and opens an inspector. The inspector asks the entity for its tunables; field metadata specifies units, ranges, logarithmic scaling and the cost of applying the change. A cloth and a rigid body therefore expose different controls. The separate Material panel adds presets, revert and save. Selection alone is not a scene rewrite.</p>
<figure class="cutout"><img src="media/rss-inspector-cutout.png" loading="lazy" alt="RSS cloth inspector showing generated material parameter controls"><figcaption>Actual inspector crop. The controls come from entity tunables and dataclass field metadata, rather than a fixed editor schema for every object.</figcaption></figure>
<ol class="engineering-steps">
<li><strong>The browser queues intent.</strong> A Viser callback records a desired value. <code>SpecBinder.drain()</code> consumes it on the simulation thread, produces an updated immutable specification and invokes the appropriate writer. Physics writes do not happen on the Viser callback thread.</li>
<li><strong>Live fields update existing arrays.</strong> Cloth stretch/bend parameters write the selected cloth’s triangle or edge material data. A captured CUDA graph reads those buffers on later launches. This probe checks array writes, not fresh-build equivalence or captured execution. In this run, changing the inspector’s log10 tri_ke from 4 to 4.3 changed the model value from 10,000 to 19,952.623 N/m; particle mass was unchanged. Rigid-body, shape and joint writers have their own destinations and solver change flags; there is no universal “set any field” operation.</li>
<li><strong>A tilde means recapture.</strong> Soft-contact stiffness, damping and friction are Python model scalars passed to kernels by value. A captured graph needs invalidation/recapture; eager execution reads the new value directly. These contact fields are model-wide, whereas the cloth constitutive arrays are selected per cloth.</li>
<li><strong>An asterisk means rebuild.</strong> Density, geometry or solver allocation changes can require fresh model and solver buffers. The UI accumulates them in “Pending rebuild.” In the public hand-written Scene path used here, “Apply &amp; rebuild” reports that the change belongs in the source script; it does not apply the structural edit. The task-backed reload path exists in source, but its withheld task tier was not validated.</li>
<li><strong>Saving is scoped.</strong> “Save as preset file” saves a material record. Camera pose edits are session-local overrides. Neither operation is a general save of the edited scene, solver, controller, camera, handles and task into a portable document.</li>
</ol>
<div class="cutout-pair"><figure><img src="media/rss-material-cutout.png" loading="lazy" alt="RSS material controls with pending rebuild field"><figcaption>Material presets and the edit-cost legend. Density is marked as a rebuild field.</figcaption></figure><figure><img src="media/rss-rebuild-cutout.png" loading="lazy" alt="RSS status says a hand-built scene must be rebuilt by editing its script"><figcaption>The status field reports “nothing to rebuild from”—the full message says to edit the source script. The field clips its long message; the complete value is in the linked probe.</figcaption></figure></div>
<p><strong>Newton work implied by this UI:</strong> make field scope, validation and solver invalidation explicit; preserve a recipe if rebuild is supported; apply or reject an edit atomically; then rebind every consumer. The reusable contribution is that edit contract and its tests. RSS’s panel layout can remain in the companion. The inspector and Material panel also maintain separate working specifications: after the inspector edit, the Material panel still displayed its earlier value in this run. Cross-panel synchronization needs a regression test. An agent needs the same outcome as structured data: applied, requires recapture, requires rebuild, or unsupported—not just a status-line string.</p>
<p class="sources">{r('robosimstudio_studio/inspector.py','Inspector')} · {r('robosimstudio_studio/tuning/binder.py','Widget metadata and queue')} · {r('robosimstudio/core/tuning.py','Model writers')} · {r('robosimstudio_studio/tuning/apply.py','Live / recapture / rebuild')} · {r('robosimstudio_studio/tuning/panels.py#L182','Rebuild behavior')} · {ev('camera-demo-probe.json','GUI edit probe')}</p>
</div>'''
    append(s.select_one('#feature-tuning'),gui)
    cameras=f'''
<article id="feature-cameras" class="feature"><h3>Cloth manipulation with live camera images</h3>
<p>A practitioner can arrange the cloth while checking whether two observation cameras can see the relevant fold or grasp point. RSS displays server-rendered RGB images inside Viser; selecting a camera, looking through it and changing its pose are part of the same session. A coding agent can use the same declared cameras to request observations, but benefits from explicit camera IDs and numerical state rather than the panel layout.</p>
<div class="video-card"><div class="video-title">RSS: cloth handles and two camera views <small>0:10 · normal speed</small></div><video id="camera-video" controls playsinline preload="metadata" poster="media/rss-cloth-cameras-poster.jpg"><source src="media/rss-cloth-cameras.mp4" type="video/mp4"></video><p>A particle-patch pin lifts the towel while front and overhead images update; the browser switches to the overhead pose, then releases the pin. This is a new reproduction of RSS’s camera/manipulation workflow using the bundled example, not an authors’ promotional clip. It starts directly in the UI. The pin’s per-substep movement cap explains why the cloth follows the target gradually.</p></div>
<figure class="cutout"><img src="media/rss-cameras-cutout.png" loading="lazy" alt="RSS camera panel displaying front and overhead RGB renders of the manipulated cloth"><figcaption>Camera panel crop: both images come from the tiled backend. The main browser viewport and these image observations are different rendering paths.</figcaption></figure>
<p><strong>Implementation:</strong> <code>PanelCamera</code> selects the declared rig or owns a renderer created through <code>Sim.make_renderer()</code>. It renders the current state at a budgeted cadence, publishes images with Viser image widgets and can texture camera frustums with the same frames. “Look through” moves the browser camera; “Set from view” stores a session-local pose override and rebuilds the renderer when its camera configuration changes. RGB, depth and segmentation switches are available according to backend support.</p>
<p>The optional “World model” mode is a separate path: OVRTX renders the browser camera and sends an image background, while Viser draws interaction overlays. That is how RSS can show a path-traced scene without implementing a path tracer in the browser. OVRTX was not installed in this run; the recording demonstrates the tiled camera views only. Decorative browser cloth shading does not automatically become a sensor material.</p>
<p><strong>What is worth sharing:</strong> camera identity and pose editing, image publication and observation timing can connect to Newton’s existing viewers and sensors. A scene/image timestamp should accompany the pixels, especially when the UI renders more slowly than simulation. Image widgets and a second renderer abstraction alone are not a reason to merge RSS.</p>
<p class="sources">{r('robosimstudio_studio/cameras.py','Camera panel and pose lifetime')} · {r('robosimstudio_studio/render_view.py','OVRTX viewport path')} · {r('robosimstudio/sim/sim.py#L281','Renderer factory')} · {ex('rss_cloth_cameras.py','Demo source')} · {ev('camera-demo-probe.json','Probe and rendering dimensions')}</p>
</article>'''
    # Insert before the live source feature, following existing manipulation video.
    s.select_one('#feature-agent').insert_before(fragment(cameras))
    lifecycle=f'''
<div id="application-structure">
<h3>Application structure: RSS, Example classes and USD</h3>
<p><strong>There is a useful common problem here: constructing a runnable experiment and exposing it to tools.</strong> Today, scene loading, solver setup, stepping, reset, GUI callbacks and recordings are often coupled in one script. RSS packages these responsibilities. Newton’s Example convention already provides part of the application shape; the USD/ovnewton work addresses another part. A shared structure could let the studio and an agent operate either kind of application.</p>
<div class="table-wrap"><table><thead><tr><th>Existing mechanism</th><th>What it supplies</th><th>What it does not establish</th></tr></thead><tbody>
<tr><td>Newton Example convention</td><td>A Python object with <code>viewer</code>, <code>step()</code> and <code>render()</code>; optional <code>gui(ui)</code> and test hooks. The runner handles viewer transport and example switching.</td><td>A portable scene/solver document, a general model-replacement protocol, or a task schema. Reset in the interactive browser reconstructs the example; it should not be assumed equivalent to every application’s own reset method.</td></tr>
<tr><td>RSS Scene / Sim</td><td>A scene recipe plus solver arrangement, controller setup, observation providers, reset/step/close and studio integration. <code>compose</code> and <code>pipeline</code> let callers change physics assembly or the executor.</td><td>A universal replacement for existing Newton loops. Its control assumptions, entity metadata and solver scheduling need adapters when the host already owns a model.</td></tr>
<tr><td>ovnewton internal main, 6 October</td><td>USD → ovstage → <code>add_ovstage(builder, stage)</code>. The caller registers solver attributes, imports, performs preparation such as VBD coloring, finalizes, then attaches a stage binding. The example selects XPBD, MuJoCo or VBD. Limited surface-cloth import is present.</td><td>Automatic construction of an entire solver program from USD. The example’s solver factory, solver options and step loop are still Python application code. Its temporary VBD contact override explicitly anticipates more solver-specific PhysicsScene settings.</td></tr>
<tr><td>ovnewton multiple-scene branch</td><td><code>StageBinding.scenes</code> maps PhysicsScene paths to per-scene bindings/models. Ownership follows USD simulationOwner. The application allocates a solver, state/control and collision pipeline per selected scene.</td><td>RSS-style interaction between solver islands. These are independent simulations; cross-scene joints are rejected and cross-scene objects do not contact each other.</td></tr>
<tr><td>Newton USD scene attributes</td><td>The schema resolver reads timestep-rate and maximum-iteration attributes, plus gravity and per-object physics properties. The USD importer returns physics_dt and max_solver_iterations metadata for the application to consume.</td><td>A complete application configuration: selecting several solver islands, their coupling/order, control rate, controllers, reset policy, cameras and arbitrary task evaluation requires additional semantics.</td></tr>
</tbody></table></div>
<p class="sources">{n('newton/examples/__init__.py#L532','Example runner and reconstruction')} · {r('robosimstudio/sim/scene.py#L320','RSS assembly hooks')} · {link(OV+'ovnewton/examples/example_ovnewton_basic.py#L144','ovnewton main: import, finalize, create solver')} · {link(OVMULTI+'ovnewton/examples/example_ovnewton_multiscene.py#L193','Multiple-scene application loop')} · {link(OV+'ovnewton/_src/ovnewton.py','StageBinding implementation')} · {link(SIM+'source/libraries/isaacsim/physics_engines/ovnewton/python/impl/newton_stage.py#L406','Isaac Sim solver construction')} · {n('newton/_src/usd/schemas.py#L114','Newton scene attributes')} · {n('newton/_src/utils/import_usd.py#L2822','Returned simulation settings')}</p>
<p class="small">This comparison uses internal main <code>3e6576c2e4dc</code> and the updated multiple-scene branch <code>1877c4902c98</code> ({link('https://github.com/NVIDIA-Omniverse/ovnewton-internal/pull/209','ovnewton #209')}); those source links require repository access. The public mirror is older and omits these newer capabilities. Neither branch was executed here.</p>
<p><strong>The key overlap is construction and ownership, not identical scene semantics.</strong> RSS’s MuJoCo and VBD islands exchange motion/contact effects through a coupler. ovnewton’s multiple PhysicsScenes partition independent models. Splitting RSS’s robot and cloth into two USD PhysicsScenes would remove their interaction unless an application adds an explicit coupling scheme. A common application structure must represent both independent worlds and coupled solver components without confusing them.</p>
<figure class="figure"><div class="diagram-scroll"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 370" role="img" aria-labelledby="app-title"><title id="app-title">Proposed convergence of Python and USD construction at an application-owned Newton runtime</title><style>text{{font-family:system-ui;fill:#253343;font-size:17px}}.bx{{fill:#f3f5f7;stroke:#8091a3}}.arr{{stroke:#617285;stroke-width:2;fill:none}}.sm{{font-size:14px}}</style><rect class="bx" x="15" y="15" width="285" height="80" rx="5"/><text x="34" y="45">Python construction</text><text class="sm" x="34" y="73">ModelBuilder / optional RSS recipe</text><rect class="bx" x="337" y="15" width="285" height="80" rx="5"/><text x="356" y="45">USD construction</text><text class="sm" x="356" y="73">Newton importer / ovnewton adapter</text><rect class="bx" x="659" y="15" width="285" height="80" rx="5"/><text x="678" y="45">Existing application</text><text class="sm" x="678" y="73">Borrow Lab or Example resources</text><path class="arr" d="M155 95V123H480V150M480 95V150M803 95V123H480"/><rect class="bx" x="150" y="151" width="660" height="95" rx="5"/><text x="173" y="183">One application owns model, states, control and solver(s)</text><text class="sm" x="173" y="211">step / render · reset or rebuild · generation · edit notifications</text><text class="sm" x="173" y="233">Solver recipe and lifecycle contract: proposed work, not a new API today</text><path class="arr" d="M480 246V273H157V291M480 273V291M480 273H803V291"/><rect class="bx" x="15" y="292" width="285" height="58" rx="5"/><text x="44" y="327">RSS panels / Viser</text><rect class="bx" x="337" y="292" width="285" height="58" rx="5"/><text x="369" y="327">Agent commands / MCP</text><rect class="bx" x="659" y="292" width="285" height="58" rx="5"/><text x="696" y="327">Tests / recording</text></svg></div><figcaption>Proposed common point: the application’s owned runtime. Python and USD remain valid construction paths; tools attach after construction. This does not require a second Scene graph in Newton.</figcaption></figure>
<p><strong>Start by adapting the Example convention, not by committing to another large base class.</strong> Keep existing <code>step()</code> and <code>render()</code> methods. Add only the information the pilot actually needs: where to obtain the current model and state, how to submit edits between steps, how reset differs from rebuild, and when old references become invalid. Optional setup/close hooks and explicit viewer attachment may be enough. Do not force examples to inherit RSS Sim or force a Lab environment through a second runner.</p>
<p><strong>Coordinate the solver recipe with ovnewton, keeping configuration separate from running buffers.</strong> It needs solver type(s), supported options, timestep/substeps, collision policy and coupling order. Resolve USD defaults and explicit Python/CLI overrides once, document precedence, and reject unsupported combinations. Both readers should feed Newton-owned physics interpretation, rather than duplicate material and joint semantics in RSS and ovnewton. Keep native USD/ovstage paths available so visual materials and stage state are not lost in an RSS round trip.</p>
<p>RSS provides a concrete design to compare against: <code>BuildRequest</code> and <code>TaskSpec</code> both satisfy the compiler’s <code>SceneRequest</code> protocol. The shared compiler needs the construction subset; task instruction, evaluator, scripted routine, episode variations, action conventions and completion rules sit above it. That separation is worth preserving. Copying the whole TaskSpec into Newton would import benchmark semantics into an engine API and compete with Lab/Arena’s own task definitions.</p>
<p><strong>Proposed validation:</strong> create the same small rigid scene through Python and USD; verify model properties, explicit solver choice, timing, initial state and reset. Attach the same panel and bounded-step command adapter to both. Test a rejected structural edit and a successful rebuild, including stale callbacks, state-buffer swaps and renderer rebinding. Then repeat the attach path in one existing Example and one Lab-owned runtime. Then add a surface-cloth case within ovnewton’s supported subset: identity world transform, constant thickness, density-derived mass and supported stretch/bend material inputs. Match the authored mesh, mass and material values before calling the two paths equivalent; RSS’s full cloth spec contains more fields.</p>
<p class="sources">{r('robosimstudio/scene/request.py','Shared scene request protocol')} · {r('robosimstudio/specs/task.py#L381','TaskSpec fields')} · {r('robosimstudio/core/env.py','Control / observation lifecycle')} · {link(OV+'docs/site/support.md','Current ovnewton cloth limits')} · {link(OVMULTI+'docs/ovnewton/multi-scene-design.md','Multiple-scene ownership and restrictions')} · {link('https://reports.eric-heiden.com/newton-ovstage-rtx/','Related USD preservation / import study')} · {link(LAB+'docs/source/concepts/native-physics-api/newton.rst','Lab runtime ownership')}</p>
</div>'''
    s.select_one('#integration .flow-diagram').insert_before(fragment(lifecycle))
    phases=s.select('#roadmap .phase')
    if len(phases)==4:
        phases[1].find('h3').string='Test the workflow and application boundary'
        append(phases[1].find_all('div',recursive=False)[1],'<p>Include a native Example adapter and a small Python-versus-USD construction comparison. Agree on solver configuration, reset/rebuild and resource ownership with the ovnewton effort before proposing a new public application class.</p>')
        phases[2].find('h3').string='Upstream reusable interaction and lifecycle contracts'
    s.select_one('#evidence>p').string='Original suite results and captures are retained below. This revision additionally runs the explicit-solver camera example and probes live GUI edits. The bundled script was also launched directly to verify its scene/build/view entry point. No Isaac or ovnewton runtime was executed for the API comparison; those findings are source reviews.'
    append(s.select_one('#evidence tbody'),f'<tr><td>{ex("rss_cloth_cameras.py")}</td><td>Explicit MuJoCo/VBD routing; two 640×400 camera views; cloth pin and release; live material edit and unsupported structural rebuild.</td><td>{ev("camera-demo-probe.json","Results")} · {ev("rss-camera-session.log","Session log")} · {ev("rss-camera-direct.log","Direct entry point")} · {ev("camera-video-provenance.json","Video provenance")}</td></tr>')
    s.select_one('footer').string='Engineering assessment. Proposed APIs and integrations are distinguished from executed examples. Original evidence is retained in the bundle.'
    s.select_one('#matrix>summary').string='Feature matrix: 17 capabilities'
    s.select_one('#matrix>p').string='Additional capabilities, overlap and evidence limits.'
    diagram=s.select_one('#application-structure svg')
    (ROOT/'media/application-runtime.svg').write_text(str(diagram))
    append(s.select_one('#application-structure figcaption'),'<a href="media/application-runtime.svg"> Open full-size diagram.</a>')
    append(s.select_one('#sources'), '<p><a href="evidence/component-table-browser-qa.json">Component table and screenshot-dialog checks</a>.</p>')
    # Keep four navigational parts but remove presentation slogans and repeated marketing chrome.
    for eyebrow in s.select('main>.part>.eyebrow'): eyebrow.decompose()
    for item in s.select('.choice .verdict'): item.decompose()
    for phase in s.select('.phase h3'):
        txt=phase.get_text().rstrip('.')
        phase.string=txt
    gui_node=s.select_one('#gui-editing')
    grid=s.new_tag('div',attrs={'class':'gui-explainer'})
    shot=gui_node.select_one('.cutout')
    steps=gui_node.select_one('ol')
    shot.insert_before(grid)
    grid.append(shot.extract())
    grid.append(steps.extract())
    css+='''
/* Engineering report revision: compact, plain headings and neutral surfaces. */
:root{--paper:#fff;--ink:#252c33;--green:#205b85;--muted:#596571;--line:#d9dee4;--white:#fff}
body{font-size:16px;line-height:1.62}h1,h2,h3,.phase-num,.stat b{font-family:system-ui,-apple-system,sans-serif;letter-spacing:0}h1{font-size:38px;line-height:1.2}h2{font-size:29px;line-height:1.3}h3,.feature h3{font-size:21px;line-height:1.4}.lede{font-size:20px;max-width:850px}.part-lede{font-size:17px}.part{padding-top:32px;margin-top:40px}.rail{background:#f3f5f7;color:#253343;border-right:1px solid #d9dee4}.brand{font-size:22px;color:#253343}.rail nav a{color:#52616f}.rail nav a.active{color:#173f60;background:#e4ebf1;border-color:#386987}.rail small,.rail-bottom,.rail nav span{color:#52616f}.rail-bottom a{color:#205b85}.eyebrow{letter-spacing:.7px}.feature{border:0;border-top:1px solid #d9dee4;border-radius:0;background:transparent;padding:24px 0;margin-top:30px}.feature p{font-size:16px}.card,.phase,.figure,.flow-diagram,.video-card{border-radius:4px;box-shadow:none}.card,.choice.recommended,.phase{background:#fafbfc}.card .title{font-family:system-ui;font-size:19px}.phase{padding:20px}.phase .gate{background:#eef2f5}.phase-num{background:#e3eaf0;color:#294e6a}.callout{background:#f1f5f8;border-color:#7096b0;border-radius:3px}main{padding-top:35px}pre{border-radius:3px}.gui-explainer{display:grid;grid-template-columns:322px 1fr;gap:24px;align-items:start}.gui-explainer .cutout{margin-top:15px}.gui-explainer ol{padding-left:22px}.cutout{max-width:340px;margin:20px auto}.cutout img,.cutout-pair img{display:block;max-width:100%;height:auto;border:1px solid #d9dee4}.cutout figcaption,.cutout-pair figcaption{font-size:13px;line-height:1.5;margin-top:8px;color:#596571}.cutout-pair{display:grid;grid-template-columns:1fr 1fr;gap:22px;max-width:830px;margin:22px auto}.cutout-pair figure{margin:0}.engineering-steps li{padding-left:4px;margin:14px 0}#api-comparison,#application-structure{margin-top:30px}#application-structure .figure{padding:12px}.video-title small{font-size:12px}#feature-cameras .cutout{max-width:430px}.jump{padding:8px 14px;border-radius:3px}.mechanism{border-radius:3px}
@media(max-width:1150px){.gui-explainer{grid-template-columns:1fr}}@media(max-width:780px){h1{font-size:31px}h2{font-size:25px}.lede{font-size:18px}.rail{border-right:0;border-bottom:1px solid #d9dee4}.cutout-pair{grid-template-columns:1fr}.feature{padding:20px 0}.feature h3{font-size:21px}.phase h3{font-size:19px}}
'''
    append(s, """
<dialog class="screenshot-dialog" id="component-image-dialog" aria-labelledby="component-image-title" aria-describedby="component-image-caption">
<div class="screenshot-dialog-bar"><strong id="component-image-title">RSS screenshot</strong><button type="button" data-close-image aria-label="Close enlarged screenshot">Close ×</button></div>
<img id="component-image" alt="">
<p id="component-image-caption"></p><a id="component-image-original" target="_blank" rel="noopener">Open original image ↗</a>
</dialog>
<script>
(() => {
  const dialog = document.getElementById('component-image-dialog');
  if (typeof dialog.showModal !== 'function') return;
  let trigger = null;
  let overflow = '';
  for (const link of document.querySelectorAll('.component-shot')) {
    link.addEventListener('click', event => {
      if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      trigger = link;
      const thumbnail = link.querySelector('img');
      document.getElementById('component-image').src = link.href;
      document.getElementById('component-image').alt = thumbnail.alt;
      document.getElementById('component-image-caption').textContent = link.dataset.caption;
      document.getElementById('component-image-original').href = link.href;
      overflow = document.documentElement.style.overflow;
      document.documentElement.style.overflow = 'hidden';
      dialog.showModal();
    });
  }
  dialog.querySelector('[data-close-image]').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', event => {
    const rect = dialog.getBoundingClientRect();
    if (event.target === dialog && (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom)) dialog.close();
  });
  dialog.addEventListener('close', () => {
    document.documentElement.style.overflow = overflow;
    if (trigger) trigger.focus({preventScroll:true});
  });
})();
</script>
""")
    css += """
.component-table{min-width:780px;table-layout:fixed}.component-table th:nth-child(1),.component-table td:first-child{width:29%}.component-table th:nth-child(2){width:36%}.component-table th:nth-child(3){width:35%}.component-table td:first-child{font-weight:400}.component-table td{vertical-align:top;line-height:1.55}.component-table td>p{margin:9px 0}.component-table .sources{font-size:12px;line-height:1.5}.component-shot{display:block;max-width:185px;margin:12px 0 10px;text-decoration:none}.component-shot img{display:block;width:100%;height:123px;object-fit:cover;border:1px solid #ccd3da;border-radius:2px}.component-shot img.show-top{object-position:top}.component-shot img.show-bottom{object-position:bottom}.component-shot span{display:block;margin-top:5px;font-size:12px;line-height:1.4;color:#405b70}.component-shot:hover img{outline:2px solid #507a99;outline-offset:2px}.screenshot-dialog{width:min(1100px,95vw);max-height:94vh;padding:18px;border:1px solid #697886;border-radius:3px;color:#253343;background:white;overflow:auto}.screenshot-dialog::backdrop{background:rgba(17,26,34,.72)}.screenshot-dialog-bar{display:flex;align-items:center;justify-content:space-between;gap:20px;margin-bottom:12px}.screenshot-dialog-bar button{background:#eef2f5;border:1px solid #b3bec8;border-radius:3px;padding:6px 12px;cursor:pointer}.screenshot-dialog>img{display:block;max-width:100%;max-height:70vh;width:auto;height:auto;margin:0 auto;object-fit:contain}.screenshot-dialog p{margin:12px 0 6px;font-size:15px;line-height:1.5}.screenshot-dialog>a{font-size:13px}@media(max-width:780px){.screenshot-dialog{padding:12px}.screenshot-dialog>img{max-height:62vh}.screenshot-dialog p{font-size:13px}.component-table{font-size:14px}.component-shot{max-width:165px}}@media print{.component-table{min-width:0}.screenshot-dialog{display:none}}
"""
    return str(s),css
