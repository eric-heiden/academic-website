import hashlib,json
from pathlib import Path
import numpy as np
from tools.mcp_evaluation.real_robot_model import validate_config,physical_coefficients
root=Path(__file__).resolve().parent
config=json.loads((root/'config.json').read_text())
validate_config(config)
folders=sorted(root.glob('candidate-[0-9][0-9][0-9]'))
metrics=[json.loads((f/'metrics.json').read_text()) for f in folders]
last=metrics[-1]
assert last['success'] and last['frames']==1800
assert config==last['config']==json.loads((root/'weighted-wrist2.json').read_text())
task=json.loads((root/'task.json').read_text())
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
hashes={name:digest(root/name) for name in task['input_hashes']}
assert hashes==task['input_hashes']
manifest=json.loads((root/'training-regressor.manifest.json').read_text())
assert digest(task['geometry_file'])==manifest['geometry_sha256']
assert len(folders)<=60
assert len({m['pid'] for m in metrics})==len(metrics)
assert all(m['frames']==1800 and m['finite'] for m in metrics)
keys=list(last['thresholds'])
worst={k:max(e[k] for e in last['per_episode']) for k in keys}
cv=[]
for n,ep in zip([17,18,19],[2,3,4]):
 m=metrics[n-1]
 assert m['success']
 cv.append({'omitted_episode':ep,'metrics_file':str(folders[n-1]/'metrics.json'),'omitted_episode_scores':next(e for e in m['per_episode'] if e['episode']==ep),'all_training_checks_pass':m['success']})
report={
 'final_config':'config.json','final_config_sha256':digest(root/'config.json'),
 'final_metrics':str(folders[-1]/'metrics.json'),'final_trace':last['trace_path'],
 'physical_candidates':len(folders),'passing_candidates':sum(m['success'] for m in metrics),
 'final_success':last['success'],'thresholds':last['thresholds'],
 'pooled':{k:last[k] for k in keys},'worst_recording':worst,
 'per_recording':last['per_episode'],'leave_one_recording_out':cv,
 'method':{'coefficients':98,'estimator':'CVXPY CLARABEL convex constrained weighted least squares',
  'inertial_constraints':'For every link, [[0.5*trace(I_origin)*identity-I_origin,h],[h.T,m]] >= 0.0001*identity, plus mass, first-moment/COM and radius bounds.',
  'joint_constraints':'Bounded viscous and Coulomb loss, torque bias and armature.',
  'regularization':{'weight':.0001,'type':'scale-aware zero-centered squared norm; no nominal dynamic parameters'},
  'joint_weights':[1,1,1,1,2,2,2],
  'episode_normalization':'Each joint weight divided by min(max(recording torque standard deviation,0.5),1.0).',
  'armature_lower_bounds':[.025,.015,.025,.015,.03,.015,.02],
  'selection':'Training torque/forward tradeoff, checked with three leave-one-recording-out refits. All candidate simulations used the prescribed fresh-process command.'},
 'immutable_input_hashes_verified':hashes,'geometry_sha256_verified':manifest['geometry_sha256'],
 'limitations':['Effective dynamics for filtered measured joint torque, not recovered motor command.','Individual link parameters are nonunique; fit selects a physically feasible representative.','Only the three supplied recordings were used. The 16-recording independent test fold has not been accessed or evaluated.','Forward agreement is established for the prescribed 100 ms windows.']}
(root/'identification-report.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['Final configuration: config.json.','',f'Final verification: {folders[-1].name}/metrics.json, success=true, 1800 steps across 36 windows.',f'Physical evaluations: {len(folders)}; passing: {report["passing_candidates"]}. Every trace and process log is retained.','','Worst result over the three supplied recordings:','', '| Metric | Fitted | Limit |','|---|---:|---:|']
for k in keys:lines.append(f'| {k} | {worst[k]:.6f} | {last["thresholds"][k]:.6f} |')
lines+=['','All pooled and per-recording checks passed. Three leave-one-recording-out refits using the final method also passed on their omitted recordings.','','Fit: full 98-coefficient weighted least squares with convex physical moment constraints and weak zero-centered regularization. Small armature floors were selected using the measured forward windows. No nominal Panda dynamic parameters were used.','','The link parameters are nonunique effective dynamics. The measured torque is filtered, and verification here covers 100 ms windows. The independent 16-recording test fold remains untested.']
(root/'identification-report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'candidate_count':len(folders),'success':last['success'],'worst':worst,'config_sha256':report['final_config_sha256'],'input_integrity':'verified'},indent=2))
