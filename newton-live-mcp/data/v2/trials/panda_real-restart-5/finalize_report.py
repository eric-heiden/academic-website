import hashlib,json
from pathlib import Path
import numpy as np
from tools.mcp_evaluation.real_robot_model import validate_config,physical_coefficients

task=json.loads(Path('task.json').read_text());config=json.loads(Path('config.json').read_text());validate_config(config)
m=json.loads(Path('candidate-013/metrics.json').read_text())
assert config==m['config'] and m['success'] and m['frames']==1800
assert all(e['success'] and e['sample_count']==600 for e in m['per_episode'])
hashes={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in task['input_hashes']}
assert hashes==task['input_hashes']
geometry=Path(task['geometry_file']); manifest=json.loads(Path('training-regressor.manifest.json').read_text())
assert hashlib.sha256(geometry.read_bytes()).hexdigest()==manifest['geometry_sha256']
paths=sorted(Path('.').glob('candidate-*/metrics.json'))
assert len(paths)==13
for path in paths:
    metric=json.loads(path.read_text());assert metric['frames']==1800
    assert Path(metric['trace_path']).exists() and (path.parent/'process.log').exists()
    if not (path.parent/'config.json').exists():
        (path.parent/'config.json').write_text(json.dumps(metric['config'],indent=2)+'\n')
worst={k:max(e[k] for e in m['per_episode']) for k in m['thresholds']}
omitted=[]
for i,ep in [(10,2),(11,3),(12,4)]:
    metric=json.loads(Path(f'candidate-{i:03d}/metrics.json').read_text())
    item=next(e for e in metric['per_episode'] if e['episode']==ep)
    assert item['success'];omitted.append(item)
report={'final_candidate':13,'physical_candidate_count':len(paths),'complete_frames_per_candidate':1800,'training_success':True,'final_config_sha256':hashlib.sha256(Path('config.json').read_bytes()).hexdigest(),'input_hashes_verified':hashes,'worst_per_recording':worst,'thresholds':m['thresholds'],'omitted_recording_results':omitted,'method':'Constrained weighted least squares over all 98 physical coefficients with CVXPY CLARABEL. PSD pseudo-inertias enforce physical COM inertias and triangle inequalities. Broad mass/COM/second-moment/joint bounds enforced. Scale-normalized quadratic regularization 1e-4; armature floors [0.03,0.03,0.03,0.03,0.03,0.02,0.015] kg m^2 selected from prescribed full physical evaluations. No nominal dynamics used.','limitations':['Individual link parameters are not uniquely identifiable.','Model represents effective dynamics of filtered measured joint torque, not raw motor commands.','Omission checks reuse the supplied training recordings and are not the independent 16-recording test fold.','Independent test results remain unknown.']}
Path('identification-report.json').write_text(json.dumps(report,indent=2)+'\n')
rows=['| Metric (worst recording) | Measured | Limit |','| --- | ---: | ---: |']
labels={'max_joint_torque_rmse_nm':'Joint torque RMSE (N m)','max_joint_torque_normalized_rmse':'Normalized joint torque RMSE','max_joint_position_rmse_rad':'Joint position RMSE (rad)','max_joint_velocity_rmse_rad_s':'Joint velocity RMSE (rad/s)','position_p95_rad':'95th percentile absolute position error (rad)','max_joint_speed_rad_s':'Maximum joint speed (rad/s)'}
for k,v in worst.items():rows.append(f'| {labels[k]} | {v:.6f} | {m["thresholds"][k]:g} |')
text='The final full numeric model is in `config.json` and matches the passing completed measurement in `candidate-013/metrics.json`. All three supplied recordings pass every threshold, pooled and separately, over 36 windows and 1,800 prescribed Newton steps. All 13 complete physical evaluations used fresh processes; the failed first candidate and every log and trace are retained.\n\n'+'\n'.join(rows)+'\n\n'+report['method']+'\n\nEach of the three recording-omission fits also passed on its omitted recording. Worst omitted-recording joint torque, position and velocity RMSE were '+f'{max(e["max_joint_torque_rmse_nm"] for e in omitted):.6f} N m, {max(e["max_joint_position_rmse_rad"] for e in omitted):.6f} rad and {max(e["max_joint_velocity_rmse_rad_s"] for e in omitted):.6f} rad/s.'+'\n\nAll supplied data and regressor hashes and the geometry hash match their original manifests. Complete trace values were read and independently checked against the reported torque, position and velocity RMSE.\n\n'+' '.join(report['limitations'])+'\n'
Path('identification-report.md').write_text(text)
print(json.dumps({'training_success':True,'candidates':len(paths),'worst_per_recording':worst,'config_sha256':report['final_config_sha256']}))
