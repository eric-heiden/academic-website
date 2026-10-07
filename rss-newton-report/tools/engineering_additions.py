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
    s.select_one('header .lede').string='RoboSimStudio’s scene API, interactive tools, and relationship to Newton, Isaac Lab, and USD-based applications.'
    s.select_one('header .meta').decompose()
    for selector in ['#feature-scene details']:
        s.select_one(selector).decompose()
    api=f'''
<div id="api-comparison">
<h3>Comparing RSS Scene with Newton ModelBuilder</h3>
<p>Both APIs collect a scene description before constructing simulation data, but they return different objects. RSS stores named entities, poses relative to the workcell, and robot presets. Its compiler uses Newton builders and assigns groups of objects to solvers. These groups, called solver islands, can interact through a coupling mechanism. RSS wraps the resulting model, solver setup, controls, and cameras in a runnable <code>Sim</code>. By contrast, Newton’s <code>ModelBuilder.finalize()</code> returns a <code>Model</code>. The application chooses the solver, allocates state and control buffers, and runs the simulation loop.</p>
<div class="table-wrap"><table><thead><tr><th>Concern</th><th>RSS</th><th>Newton / application</th><th>Integration implication</th></tr></thead><tbody>
<tr><td>Authoring</td><td><code>Scene.add(spec)</code> stores specifications for named boxes, sheets, robots, and cameras. It returns named references for entities.</td><td><code>ModelBuilder</code> assembles bodies, joints, shapes, particles, and imported assets.</td><td>RSS provides a higher-level authoring option. Newton users should be able to attach studio tools to an existing model without converting it into RSS specifications.</td></tr>
<tr><td>Construction result</td><td><code>Scene.build()</code> → compiler → <code>CompiledScene</code> → <code>Sim</code>. Warmup/reset runs by default.</td><td><code>builder.finalize()</code> → <code>Model</code>; <code>model.state()</code> and <code>model.control()</code> allocate runtime buffers.</td><td>A model and a runnable application are distinct products. Do not make finalize open a viewer or choose an application loop.</td></tr>
<tr><td>Solver choice and scheduling</td><td><code>ArrangementSpec</code> assigns entity types to physics backends and specifies coupling, control rate, and substeps.</td><td>The application creates solvers and defines collision handling, the timestep, state-buffer swaps, and any multi-solver schedule.</td><td>RSS simplifies setup, but its solver assignments and coupling defaults are application choices. An integration must let applications retain their existing solver configuration and stepping sequence.</td></tr>
<tr><td>Control and observation</td><td><code>Sim.step(action, mode=...)</code> selects a controller; observations come from providers. Robot presets supply tool frames and gripper conventions.</td><td><code>Control</code> is simulation input data; examples or a learning framework interpret actions and assemble observations.</td><td>A common runtime must not prescribe an RL action space, gripper range or observation dictionary.</td></tr>
<tr><td>Viewer ownership</td><td><code>rss.view(sim)</code> creates a StudioSession that drives the loop. <code>rss.viewer(sim)</code> supports a caller-owned loop.</td><td>An <code>Example</code> owns simulation resources; the example runner calls its <code>step()</code> and <code>render()</code>.</td><td>Provide an attach mode as well as standalone RSS. Existing Example or Lab code must remain the sole step owner.</td></tr>
<tr><td>Rebuild and identity</td><td>Building the same Scene again rebinds its entity references to the new build. Two Sims from one Scene do not have independent references.</td><td>Body/particle indices and state buffers belong to a particular finalized model; reconstruction invalidates consumers’ cached references.</td><td>Record model generation, world and entity identity. Rebind panels, renderers, handles and agent requests together.</td></tr>
<tr><td>Task definition</td><td><code>TaskSpec</code> adds instruction, evaluator, routine, variation and episode rules to scene fields. <code>BuildRequest</code> supplies the scene-only path.</td><td>Newton models describe physics. Task semantics live in application code, Lab, Arena or another framework.</td><td>Reuse scene construction without adopting RSS’s task representation or benchmark conventions.</td></tr>
</tbody></table></div>
<p class="sources">{r('robosimstudio/sim/scene.py#L320','Scene.build and reference lifetime')} · {r('robosimstudio/scene/request.py#L99','BuildRequest')} · {r('robosimstudio/scene/compiler.py#L228','Compiler entry point')} · {r('robosimstudio/sim/sim.py#L47','Sim')} · {n('newton/_src/sim/builder.py','ModelBuilder')} · {n('newton/examples/__init__.py#L532','Example runner')}</p>
<h3>Complete scene setup with cloth, solvers, and cameras</h3>
<p>The example below defines a robot, cloth, a block, and two cameras, then builds the simulation and opens the studio. Inside <code>build()</code>, RSS creates Newton’s MuJoCo solver for the articulated robot and its vertex block descent (VBD) solver for the cloth and block. RSS’s default proxy coupling transfers interactions between those solver islands. The camera specifications define the views, and the camera panel creates a tiled renderer when needed. Setting <code>share=False</code> disables the public Viser tunnel.</p>
<pre><code>{html.escape('import robosimstudio as rss'+(ROOT/'examples/rss_cloth_cameras.py').read_text().split('import robosimstudio as rss',1)[1])}</code></pre>
<p>The simulation contains 16 robot bodies in the MuJoCo island and one rigid body plus 361 cloth particles in the VBD island. Building it took about 21 seconds on the shared test machine. The configuration specifies 60 control steps per simulated second and eight physics substeps per control step, giving a physics timestep of 1/480 s. With live camera images enabled, the application ran at approximately 18–20 control steps per wall-clock second. These timings describe this scene on that host, not general RSS performance.</p>
<p class="sources">{ex('rss_cloth_cameras.py','Run this example')} · {r('docs/source/guides/physics.md','Backend and coupling choices')} · {r('robosimstudio/scene/physics/compose.py','Physics composition')} · {r('robosimstudio/sim/draw.py','Viewer entry points')} · {ev('camera-demo-probe.json','Runtime probe')}</p>
</div>'''
    after(s.select_one('#feature-scene'),api)
    tuning=s.select_one('#feature-tuning')
    tuning.select_one('.cards').decompose()
    tuning.find_all('p',recursive=False)[1].decompose()
    # Detailed GUI walkthrough, directly beside the feature it documents.
    gui=f'''
<div id="gui-editing">
<p>Clicking a mesh selects an entity and opens its inspector. The inspector uses metadata to determine which parameters users can edit, their units and ranges, and how to apply changes. Cloth and rigid bodies therefore expose different controls. A separate Material panel provides presets and save and revert operations.</p>
<figure class="cutout"><img src="media/rss-inspector-cutout.png" loading="lazy" alt="RSS cloth inspector showing generated material parameter controls"><figcaption>The cloth inspector generates controls from the entity’s editable parameters and field metadata.</figcaption></figure>
<ol class="engineering-steps">
<li><strong>The browser queues parameter changes.</strong> A Viser callback records the requested value. On the simulation thread, <code>SpecBinder.drain()</code> creates an updated immutable specification and calls the appropriate parameter writer. The browser callback does not write physics state directly.</li>
<li><strong>Some edits update existing arrays.</strong> Cloth stretch and bend parameters update material arrays for the selected cloth’s triangles or edges. Later launches of a captured CUDA graph read those buffers. In the GUI test, changing log10 tri_ke from 4 to 4.3 changed the model value from 10,000 to 19,952.623 N/m without changing particle mass. The test confirmed the array update, but did not establish equivalence to a freshly built model or verify captured execution. Rigid-body, shape, and joint parameters use separate writers and solver notifications.</li>
<li><strong>Fields marked with a tilde require graph recapture.</strong> Soft-contact stiffness, damping, and friction are Python scalar values passed into GPU kernels. A CUDA graph records kernel launches for later reuse, including these scalar arguments. The application must capture the graph again to use changed scalar values; uncaptured execution reads the new values directly. These contact fields affect the entire model, unlike the material arrays for an individual cloth.</li>
<li><strong>Fields marked with an asterisk require a rebuild.</strong> Changes to density, geometry, or solver allocation can require new model and solver buffers. The interface lists these edits under “Pending rebuild.” For the tested Python-authored scene, “Apply &amp; rebuild” instructed the user to edit the source script and did not apply the change. RSS also contains a task-backed reload path, but the public release withholds the task tier needed to validate it.</li>
<li><strong>Saving a preset does not save the whole scene.</strong> “Save as preset file” writes a material record. Camera pose edits remain local to the session. Neither operation saves the complete scene, solver, controller, cameras, manipulation handles, and task in a portable document.</li>
</ol>
<div class="cutout-pair"><figure><img src="media/rss-material-cutout.png" loading="lazy" alt="RSS material controls with pending rebuild field"><figcaption>The Material panel marks density changes as requiring a rebuild.</figcaption></figure><figure><img src="media/rss-rebuild-cutout.png" loading="lazy" alt="RSS status says a hand-built scene must be rebuilt by editing its script"><figcaption>The rebuild command rejects the structural edit and directs the user to the scene’s source script.</figcaption></figure></div>
<p>A Newton editing interface should identify each field’s scope, validate the requested value, and notify the affected solver. If an edit requires reconstruction, the application must provide a way to rebuild the model and reconnect every component that uses it. Each edit should either succeed completely or leave the current model unchanged. RSS’s panel layout can remain separate from these rules.</p>
<p>The inspector and Material panel currently maintain separate working specifications. After the inspector changed a value in the test, the Material panel still displayed the old value. Shared readback from the active model and a regression test should prevent that disagreement. Coding agents also need structured results that distinguish an applied edit from one that requires recapture, requires reconstruction, or is unsupported.</p>
<p class="sources">{r('robosimstudio_studio/inspector.py','Inspector')} · {r('robosimstudio_studio/tuning/binder.py','Widget metadata and queue')} · {r('robosimstudio/core/tuning.py','Model writers')} · {r('robosimstudio_studio/tuning/apply.py','Live / recapture / rebuild')} · {r('robosimstudio_studio/tuning/panels.py#L182','Rebuild behavior')} · {ev('camera-demo-probe.json','GUI edit probe')}</p>
</div>'''
    append(s.select_one('#feature-tuning'),gui)
    cameras=f'''
<article id="feature-cameras" class="feature"><h3>Cloth manipulation with live camera images</h3>
<p>A practitioner can arrange the cloth while checking whether two observation cameras can see the relevant fold or grasp point. RSS displays server-rendered RGB images inside Viser; selecting a camera, looking through it and changing its pose are part of the same session. A coding agent can use the same declared cameras to request observations, but benefits from explicit camera IDs and numerical state rather than the panel layout.</p>
<div class="video-card"><div class="video-title">RSS: cloth handles and two camera views <small>0:10 · normal speed</small></div><video id="camera-video" controls playsinline preload="metadata" poster="media/rss-cloth-cameras-poster.jpg"><source src="media/rss-cloth-cameras.mp4" type="video/mp4"></video><p>A pin lifts a towel patch while the front and overhead camera images update. The user then switches to the overhead view and releases the pin. The cloth follows the target gradually because the pin limits movement during each physics substep.</p></div>
<figure class="cutout"><img src="media/rss-cameras-cutout.png" loading="lazy" alt="RSS camera panel displaying front and overhead RGB renders of the manipulated cloth"><figcaption>The tiled renderer produces both camera images. These sensor views use a different rendering path from the main browser viewport.</figcaption></figure>
<p><code>PanelCamera</code> uses the declared camera rig or creates a renderer through <code>Sim.make_renderer()</code>. It renders the current simulation state at a limited update rate and displays the images in Viser. “Look through” moves the browser camera to the selected camera’s pose. “Set from view” stores the current browser pose as a session-local override; a changed camera configuration causes the panel to rebuild its renderer. Available outputs include RGB, depth, and segmentation, depending on the backend.</p>
<p>The optional “World model” mode uses a separate rendering path. OVRTX renders an image from the browser camera’s pose, and Viser displays that image behind the interaction controls. This allows path-traced rendering without a browser-side path tracer. The demonstration covers tiled camera views, not the optional OVRTX integration. Cloth appearance settings in the browser do not automatically apply to sensor rendering.</p>
<p>Newton’s existing viewers and sensors could support the same camera selection, pose editing, and image display. Each image should identify its camera and simulation time, especially when rendering runs more slowly than the simulation. An integration should add only the missing public interfaces, rather than introduce a second renderer abstraction.</p>
<p class="sources">{r('robosimstudio_studio/cameras.py','Camera panel and pose lifetime')} · {r('robosimstudio_studio/render_view.py','OVRTX viewport path')} · {r('robosimstudio/sim/sim.py#L281','Renderer factory')} · {ex('rss_cloth_cameras.py','Demo source')} · {ev('camera-demo-probe.json','Probe and rendering dimensions')}</p>
</article>'''
    # Insert before the live source feature, following existing manipulation video.
    s.select_one('#feature-agent').insert_before(fragment(cameras))
    lifecycle=f'''
<div id="application-structure">
<h3>Application structure: RSS, Example classes and USD</h3>
<p>RSS, Newton examples, and ovnewton all need to construct simulations and expose them to tools. They organize those responsibilities differently. RSS manages scene loading, solver setup, stepping, reset, GUI callbacks, and recording. Newton’s <code>Example</code> convention defines a common simulation loop, while ovnewton connects USD scenes to Newton models. A shared application interface could let studio controls and coding agents work with any of these construction paths.</p>
<div class="table-wrap"><table><thead><tr><th>Existing mechanism</th><th>What it supplies</th><th>What it does not establish</th></tr></thead><tbody>
<tr><td>Newton Example convention</td><td>A Python object with <code>viewer</code>, <code>step()</code> and <code>render()</code>; optional <code>gui(ui)</code> and test hooks. The runner handles viewer transport and example switching.</td><td>A portable scene/solver document, a general model-replacement protocol, or a task schema. Reset in the interactive browser reconstructs the example; it should not be assumed equivalent to every application’s own reset method.</td></tr>
<tr><td>RSS Scene / Sim</td><td>A scene recipe plus solver arrangement, controller setup, observation providers, reset/step/close and studio integration. <code>compose</code> and <code>pipeline</code> let callers change physics assembly or the executor.</td><td>A universal replacement for existing Newton loops. Its control assumptions, entity metadata and solver scheduling need adapters when the host already owns a model.</td></tr>
<tr><td>ovnewton internal main, 6 October</td><td>USD → ovstage → <code>add_ovstage(builder, stage)</code>. The caller registers solver attributes, imports, performs preparation such as VBD coloring, finalizes, then attaches a stage binding. The example selects XPBD, MuJoCo or VBD. Limited surface-cloth import is present.</td><td>Automatic construction of an entire solver program from USD. The example’s solver factory, solver options and step loop are still Python application code. Its temporary VBD contact override explicitly anticipates more solver-specific PhysicsScene settings.</td></tr>
<tr><td>ovnewton multiple-scene branch</td><td><code>StageBinding.scenes</code> maps PhysicsScene paths to per-scene bindings/models. Ownership follows USD simulationOwner. The application allocates a solver, state/control and collision pipeline per selected scene.</td><td>RSS-style interaction between solver islands. These are independent simulations; cross-scene joints are rejected and cross-scene objects do not contact each other.</td></tr>
<tr><td>Newton USD scene attributes</td><td>The schema resolver reads timestep-rate and maximum-iteration attributes, plus gravity and per-object physics properties. The USD importer returns physics_dt and max_solver_iterations metadata for the application to consume.</td><td>A complete application configuration: selecting several solver islands, their coupling/order, control rate, controllers, reset policy, cameras and arbitrary task evaluation requires additional semantics.</td></tr>
</tbody></table></div>
<p class="sources">{n('newton/examples/__init__.py#L532','Example runner and reconstruction')} · {r('robosimstudio/sim/scene.py#L320','RSS assembly hooks')} · {link(OV+'ovnewton/examples/example_ovnewton_basic.py#L144','ovnewton main: import, finalize, create solver')} · {link(OVMULTI+'ovnewton/examples/example_ovnewton_multiscene.py#L193','Multiple-scene application loop')} · {link(OV+'ovnewton/_src/ovnewton.py','StageBinding implementation')} · {link(SIM+'source/libraries/isaacsim/physics_engines/ovnewton/python/impl/newton_stage.py#L406','Isaac Sim solver construction')} · {n('newton/_src/usd/schemas.py#L114','Newton scene attributes')} · {n('newton/_src/utils/import_usd.py#L2822','Returned simulation settings')}</p>
<p class="small">The ovnewton comparison covers internal main revision <code>3e6576c2e4dc</code> and multiple-scene revision <code>1877c4902c98</code> ({link('https://github.com/NVIDIA-Omniverse/ovnewton-internal/pull/209','ovnewton #209')}). The links require repository access. These source-level findings do not establish runtime compatibility; the older public mirror also lacks these capabilities.</p>
<p>RSS and ovnewton both organize scene construction, but their scene divisions have different physical meanings. RSS’s MuJoCo and VBD islands exchange motion and contact effects through a coupler. In ovnewton, separate PhysicsScenes produce independent models. Placing an RSS robot and its cloth in different USD PhysicsScenes would remove their interaction unless the application explicitly coupled them. A common application interface must distinguish independent simulations from solver components that interact within one simulation.</p>
<figure class="figure"><div class="diagram-scroll"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 370" role="img" aria-labelledby="app-title"><title id="app-title">Proposed convergence of Python and USD construction at an application-owned Newton runtime</title><style>text{{font-family:system-ui;fill:#253343;font-size:17px}}.bx{{fill:#f3f5f7;stroke:#8091a3}}.arr{{stroke:#617285;stroke-width:2;fill:none}}.sm{{font-size:14px}}</style><rect class="bx" x="15" y="15" width="285" height="80" rx="5"/><text x="34" y="45">Python construction</text><text class="sm" x="34" y="73">ModelBuilder / optional RSS recipe</text><rect class="bx" x="337" y="15" width="285" height="80" rx="5"/><text x="356" y="45">USD construction</text><text class="sm" x="356" y="73">Newton importer / ovnewton adapter</text><rect class="bx" x="659" y="15" width="285" height="80" rx="5"/><text x="678" y="45">Existing application</text><text class="sm" x="678" y="73">Borrow Lab or Example resources</text><path class="arr" d="M155 95V123H480V150M480 95V150M803 95V123H480"/><rect class="bx" x="150" y="151" width="660" height="95" rx="5"/><text x="173" y="183">One application owns model, states, control and solver(s)</text><text class="sm" x="173" y="211">step / render · reset or rebuild · generation · edit notifications</text><text class="sm" x="173" y="233">Solver recipe and lifecycle contract: proposed work, not a new API today</text><path class="arr" d="M480 246V273H157V291M480 273V291M480 273H803V291"/><rect class="bx" x="15" y="292" width="285" height="58" rx="5"/><text x="44" y="327">RSS panels / Viser</text><rect class="bx" x="337" y="292" width="285" height="58" rx="5"/><text x="369" y="327">Agent commands / MCP</text><rect class="bx" x="659" y="292" width="285" height="58" rx="5"/><text x="696" y="327">Tests / recording</text></svg></div><figcaption>Proposed common point: the application’s owned runtime. Python and USD remain valid construction paths; tools attach after construction. This does not require a second Scene graph in Newton.</figcaption></figure>
<p>The existing <code>Example</code> convention provides a practical starting point. Keep its <code>step()</code> and <code>render()</code> methods, and add only the information tools need: access to the current model and state, a safe point for submitting edits between steps, reset and rebuild operations, and notification when references become invalid. Optional setup and close hooks, together with explicit viewer attachment, may be sufficient. Examples should not need to inherit RSS’s <code>Sim</code>, and Isaac Lab environments should retain their existing runner.</p>
<p>Newton and ovnewton should use a consistent description of solver configuration, separate from runtime buffers. That description needs solver types and options, timesteps and substeps, collision handling, and the order of coupled operations. It should define how explicit Python or command-line settings override USD defaults and reject unsupported combinations. Both construction paths should use Newton’s interpretation of physical materials and joints. Preserving the native USD/ovstage path also avoids losing visual materials or stage state during conversion to RSS specifications.</p>
<p>RSS already separates scene construction from task evaluation: <code>BuildRequest</code> and <code>TaskSpec</code> both implement the compiler’s <code>SceneRequest</code> protocol. The compiler uses their scene-construction fields, while task instructions, evaluators, scripted routines, episode variations, action conventions, and completion rules belong to the task layer. Newton should preserve this separation. Adopting the whole <code>TaskSpec</code> would introduce benchmark conventions into the engine API and overlap with Isaac Lab and Arena.</p>
<p>Validation should begin with the same small rigid-body scene built through Python and USD. Compare model properties, solver selection, timing, initial state, and reset behavior, then attach the same panel and bounded-step commands to both. Test rejected edits and successful rebuilds, including references to the old model, state-buffer swaps, and renderer updates. Repeat these checks in an existing Newton example and an Isaac Lab-owned runtime.</p>
<p>A subsequent cloth comparison should stay within ovnewton’s supported subset: an identity world transform, constant thickness, density-derived mass, and supported stretch and bend parameters. The two construction paths must agree on the mesh, mass, and material values before they can be considered equivalent for that case. RSS supports additional cloth fields that this comparison would not cover.</p>
<p class="sources">{r('robosimstudio/scene/request.py','Shared scene request protocol')} · {r('robosimstudio/specs/task.py#L381','TaskSpec fields')} · {r('robosimstudio/core/env.py','Control / observation lifecycle')} · {link(OV+'docs/site/support.md','Current ovnewton cloth limits')} · {link(OVMULTI+'docs/ovnewton/multi-scene-design.md','Multiple-scene ownership and restrictions')} · {link('https://reports.eric-heiden.com/newton-ovstage-rtx/','Related USD preservation / import study')} · {link(LAB+'docs/source/concepts/native-physics-api/newton.rst','Lab runtime ownership')}</p>
</div>'''
    s.select_one('#integration .flow-diagram').insert_before(fragment(lifecycle))
    s.select_one('#evidence>p').string='The tests cover RSS scene construction, interactive controls, live editing, camera images, and a native Newton viewer example. They establish specific behavior in those configurations, not general physical accuracy or compatibility across applications. The Isaac and ovnewton comparisons describe source-level interfaces; their runtime integration remains untested.'
    append(s.select_one('#evidence tbody'),f'<tr><td>{ex("rss_cloth_cameras.py")}</td><td>The example assigns objects to MuJoCo and VBD, displays two 640×400 camera views, and supports cloth pinning and release. Tests confirm a live material edit and rejection of an unsupported structural rebuild.</td><td>{ev("camera-demo-probe.json","Results")} · {ev("rss-camera-session.log","Session log")} · {ev("rss-camera-direct.log","Standalone launch")}</td></tr>')
    s.select_one('footer').decompose()
    s.select_one('#matrix>summary').string='Feature matrix: 17 capabilities'
    s.select_one('#matrix>p').string='The table compares RSS capabilities with existing Newton functionality and identifies the proposed integration boundary for each.'
    diagram=s.select_one('#application-structure svg')
    (ROOT/'media/application-runtime.svg').write_text(str(diagram))
    append(s.select_one('#application-structure figcaption'),'<a href="media/application-runtime.svg"> Open full-size diagram.</a>')
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
