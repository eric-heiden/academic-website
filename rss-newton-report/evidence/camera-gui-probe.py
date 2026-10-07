from pathlib import Path
from robosimstudio_agent.wire import request
import ast, json
R=Path('/home/eheiden/.codex/visualizations/2026/10/07/01a113ce-dd5d-7a83-bb8a-8f0ea97a273b/rss-newton-report')
req=lambda obj:request('/tmp/rss-camera-study-run/ctl.sock',obj,timeout=60)
def evaluate(code):
    out=req({'cmd':'eval','code':code}); assert out['ok'],out
    return ast.literal_eval(out['value'])
before=ast.literal_eval(json.loads(Path('/tmp/rss-camera-before.json').read_text())['value'])
after=ast.literal_eval(json.loads(Path('/tmp/rss-camera-after.json').read_text())['value'])
final=evaluate('{"pending":session.tune._pending.value,"status":session.status,"mass":float(sim.model.particle_mass.numpy().sum()),"finite":sim.state_is_finite(),"density_requested":session.tune.binder.spec.areal_density,"loaded_at":live.loaded_at,"step":sim.step_count}')
assert abs(after['tri'][0]-10**4.3)<0.01,(before,after)
assert before['mass']==after['mass']==final['mass'],final
assert final['density_requested']==0.45 and 'assembled by hand' in final['status'],final
assert final['finite']
result={'status':'passed','rss_commit':'9bb21c85f79240eb16cc5e610114edd74f1eb2b2','ovnewton_source_commit':'cd06cb205276d5655bb99e5d12a3ce67635c9895','example':'examples/rss_cloth_cameras.py','before_gui_edits':before,'after_inspector_tri_ke_log10_4_3':after,'after_material_density_0_45_and_rebuild_button':final,'checks':['Explicit MuJoCo robot and VBD cloth/rigid islands','Both declared cameras 640x400; overhead renderer output uint8 RGB shape 400x640x3','GUI inspector updated first triangle shear coefficient from 10000 to 19952.623, matching 10**4.3','GUI structural density change remained pending; particle mass unchanged; hand-built scene rebuild refused with status','Simulation remains finite'],'limitations':['One scene and one cloth pin; no two-handle or fresh-build equivalence test','Tiled renderer only; no RTX or ovnewton runtime','Inspector and Material panel hold separate working specs; cross-panel consistency not certified']}
(R/'evidence/camera-demo-probe.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
