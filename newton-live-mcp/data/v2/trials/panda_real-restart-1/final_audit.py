import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import hashlib,json
from pathlib import Path
import numpy as np
from fit_model import fit,report
from tools.mcp_evaluation.real_robot_model import validate_config,physical_coefficients

config=json.loads(Path('config.json').read_text())
metrics=json.loads(Path('candidate-002/metrics.json').read_text())
assert config==metrics['config'] and metrics['success'] and metrics['frames']==1800
validate_config(config)
task=json.loads(Path('task.json').read_text())
integrity={}
for name,expected in task['input_hashes'].items():
 actual=hashlib.sha256(Path(name).read_bytes()).hexdigest()
 assert actual==expected,name
 integrity[name]=actual
D=np.load('training-regressor.npz');A,b,ep=D['A'],D['b'],D['sample_episode_ids']
x=physical_coefficients(config)
assert np.allclose(np.sqrt(np.mean((A@x-b).reshape(-1,7)**2,axis=0)),metrics['torque_rmse_per_joint_nm'],atol=1e-12,rtol=0)
cv={}
for e in np.unique(ep):
 xc,st,ob=fit(A,b,ep,lam=1e-4,arm_floor=.03,mask=np.repeat(ep!=e,7))
 cv[str(e)]=report(A,b,ep,xc)['episodes'][str(e)]
tr=np.load(metrics['trace_path'])
for k in ['q','qd','reference_q','reference_qd','measured_torque']:
 assert tr[k].shape==(1800,7) and np.isfinite(tr[k]).all()
checks={k:{'limit':limit,'pooled':metrics[k],'worst_recording':max(e[k] for e in metrics['per_episode'])} for k,limit in task['thresholds'].items()}
record={
 'selected_candidate':2,'complete_physical_evaluations':5,'selected_config_sha256':hashlib.sha256(Path('config.json').read_bytes()).hexdigest(),
 'estimator':'Weighted linear least squares with seven positive-semidefinite 4x4 pseudo-inertia constraints, mass/COM/second-moment and joint coefficient bounds; weak scaled ridge (lambda=1e-4) toward homogeneous placeholders. Joint armature floor 0.03 kg*m^2 chosen by comparing torque and prescribed forward motion checks.',
 'training_samples_for_torque':450,'training_forward_windows':36,'steps':1800,
 'approximate_regressor_rank_at_absolute_tolerance_1e-3':int(np.linalg.matrix_rank(A,tol=1e-3)),
 'checks':checks,'leave_one_recording_out_torque_rmse_per_joint_nm':cv,
 'min_COM_inertia_eigenvalue':min(float(np.linalg.eigvalsh(i).min()) for i in config['inertia']),
 'input_hashes_verified':integrity,'trace_path':metrics['trace_path'],
 'limitations':['Effective dynamics for filtered measured torque; individual link parameters are not uniquely identified.','Only the supplied three recordings were used. Independent held-out evaluation is pending and was not accessed.','The positive armature floor trades torque residual against sensitivity of forward motion to residual torque.'],
 'data_attribution':'LIP4RobotInverseDynamics, MERL (Giacomuzzo, Carli, Romeres, Dalla Libera), 2024, DOI 10.5281/zenodo.12516500; CC-BY-SA-4.0.'
}
Path('identification-report.json').write_text(json.dumps(record,indent=2)+'\n')
lines=['Selected candidate 002; config.json is byte-identical to candidate-002/config.json and numerically identical to its passing completed measurement.','', 'Five fresh-process physical candidate evaluations were completed. Candidate 001 failed motion checks; candidates 002–005 passed all pooled and per-recording thresholds. All outputs are retained.','', '| Check | Pooled | Worst recording | Limit |','|---|---:|---:|---:|']
for k,v in checks.items():lines.append(f"| {k} | {v['pooled']:.6f} | {v['worst_recording']:.6f} | {v['limit']} |")
lines.extend(['',record['estimator'],'', 'Leave-one-recording-out validation used only the three supplied training recordings. Worst validation torque RMSE: '+str(max(max(v) for v in cv.values()))+' N m.','',*record['limitations'],'',record['data_attribution']])
Path('identification-report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(record,indent=2))
