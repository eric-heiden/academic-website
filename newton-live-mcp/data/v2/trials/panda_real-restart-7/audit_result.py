from pathlib import Path
import hashlib,json
import numpy as np
from tools.mcp_evaluation.real_robot_model import validate_config,physical_coefficients
root=Path.cwd();task=json.loads(Path('task.json').read_text());cfg=json.loads(Path('config.json').read_text());validate_config(cfg)
metrics=json.loads(Path('candidate-018/metrics.json').read_text())
assert metrics['config']==cfg
assert json.loads(Path('candidate-018/config.json').read_text())==cfg
assert metrics['success'] and metrics['frames']==1800 and metrics['finite']
inputs={}
for name,expected in task['input_hashes'].items():
    actual=hashlib.sha256(Path(name).read_bytes()).hexdigest();assert actual==expected,name;inputs[name]=actual
manifest=json.loads(Path('training-regressor.manifest.json').read_text())
geometry=Path(task['geometry_file']);assert hashlib.sha256(geometry.read_bytes()).hexdigest()==manifest['geometry_sha256']
all_runs=[]
for i in range(1,19):
    path=Path(f'candidate-{i:03d}/metrics.json');d=json.loads(path.read_text());validate_config(d['config'])
    assert d['frames']==1800 and Path(d['trace_path']).exists()
    trace=np.load(d['trace_path']);assert trace['q'].shape==(1800,7) and trace['qd'].shape==(1800,7)
    assert np.isfinite(trace['q']).all() and np.isfinite(trace['qd']).all()
    all_runs.append({'candidate':i,'success':d['success'],'pid':d['pid'],'trace':d['trace_path']})
assert len(set(x['pid'] for x in all_runs))==18
cv=[]
for n,eid in [(15,2),(16,3),(17,4)]:
    d=json.loads(Path(f'candidate-{n:03d}/metrics.json').read_text())
    e=next(e for e in d['per_episode'] if e['episode']==eid);assert e['success'];cv.append(e)
keys=list(task['thresholds'])
worst={k:max(e[k] for e in metrics['per_episode']) for k in keys}
cvworst={k:max(e[k] for e in cv) for k in keys}
audit={'final_candidate':18,'configuration_sha256':hashlib.sha256(Path('config.json').read_bytes()).hexdigest(),'training_passed':True,'final_numeric_configuration_matches_measurement':True,'immutable_inputs_match':inputs,'geometry_sha256':manifest['geometry_sha256'],'complete_candidates':len(all_runs),'candidate_budget':60,'worst_recording_metrics':worst,'cross_validation_worst_excluded_recording':cvworst,'candidates':all_runs,'heldout_test':'Not accessed; independent verification pending.'}
Path('identification-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
lines=['# Effective Panda dynamics identification','','The submitted `config.json` exactly matches completed candidate 018. Its 1,800-step measurement passed every threshold pooled and in each of the three supplied physical recordings. All 18 fresh-process candidates, including the failed initial candidate, retain their numeric configuration, metrics, full NPZ trace, and process log.','','## Method','','Fitted the supplied 3,150-by-98 Newton regressor directly using weighted least squares and CVXPY/Clarabel. Each link has a full symmetric origin inertia, mass, and first moment. Positive semidefinite pseudo-inertia constraints enforce physically valid COM tensors and triangle inequalities; mass, COM, second-moment, loss, bias, and armature bounds are enforced. A small coefficient regularizer (1e-4) uses only the homogeneous placeholders. Joint loss coefficients are nonnegative.','','The torque-only optimum made several effective rotational inertias too small for accurate measured forward motion. Candidate comparisons selected armature lower bounds [0.025, 0.025, 0.025, 0.025, 0.030, 0.025, 0.025] kg m², refitting all other parameters jointly. These constraints balance torque and motion prediction. No nominal Panda dynamics or external data were used.','','## Final measured quality','','| Metric | Pooled | Worst recording | Required maximum |','|---|---:|---:|---:|']
for k in keys:lines.append(f'| {k} | {metrics[k]:.6f} | {worst[k]:.6f} | {task["thresholds"][k]:g} |')
lines+=['','## Validation and artifacts','','The final fitting procedure was also trained on each pair of recordings (candidates 015–017). Each excluded recording passed every torque and forward-motion threshold. Worst excluded-recording torque RMSE was %.6f N m, normalized torque RMSE %.6f, position RMSE %.6f rad, and velocity RMSE %.6f rad/s.'%(cvworst['max_joint_torque_rmse_nm'],cvworst['max_joint_torque_normalized_rmse'],cvworst['max_joint_position_rmse_rad'],cvworst['max_joint_velocity_rmse_rad_s']),'','- Final configuration: `config.json`','- Final completed measurement and trace: `candidate-018/metrics.json`, `candidate-018/metrics.npz`','- Fitting code: `fit_model.py`; final fit and validation driver: `finalize_fit.py`','- Audit with input hashes and candidate records: `identification-audit.json`','','## Limits','','These are effective dynamics for filtered measured joint torques, not a unique recovery of individual link properties or raw motor commands. The supplied data do not uniquely identify every link coefficient. Forward validation covers 100 ms windows; longer horizons are untested. The independent 16-recording test fold was not accessed and remains for the post-submission verifier.','']
Path('identification-report.md').write_text('\n'.join(lines))
print(json.dumps({k:audit[k] for k in ['final_candidate','configuration_sha256','training_passed','final_numeric_configuration_matches_measurement','complete_candidates','worst_recording_metrics','cross_validation_worst_excluded_recording']},indent=2))
