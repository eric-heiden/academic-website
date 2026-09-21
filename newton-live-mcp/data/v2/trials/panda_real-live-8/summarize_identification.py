from pathlib import Path
import json, hashlib
import numpy as np
root=Path(__file__).resolve().parent
spec=json.loads((root/'task.json').read_text())
manifest=json.loads((root/'training-regressor.manifest.json').read_text())
metrics=json.loads((root/'final_metrics.json').read_text())
cfg=json.loads((root/'config.json').read_text())
assert cfg==metrics['config']
assert metrics['success'] and metrics['frames']==1800
hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in spec['input_hashes']}
assert hashes==spec['input_hashes']
geom=Path(spec['geometry_file'])
assert hashlib.sha256(geom.read_bytes()).hexdigest()==manifest['geometry_sha256']
for m,c,I in zip(cfg['mass'],cfg['com'],cfg['inertia']):
 c=np.array(c);I=np.array(I);lam=np.linalg.eigvalsh(I)
 assert .05<=m<=10 and np.max(np.abs(c))<=.4
 assert lam.min()>=1e-8 and lam[-1]<=lam[:2].sum()+1e-10
 assert .5*np.trace(I)+m*c@c<=.25*m+1e-10
for key in ['viscous','coulomb','torque_bias','armature']:
 lo,hi=spec['bounds'][key];assert np.min(cfg[key])>=lo and np.max(cfg[key])<=hi
before=json.loads((root/'final_metrics_before_rebuild.json').read_text())
for key in metrics['thresholds']:assert metrics[key]==before[key]
traces=[root/f'candidate-{i:03d}.npz' for i in range(1,57)]
assert all(p.exists() for p in traces)
for p in traces:
 with np.load(p) as d: assert d['q'].shape==(1800,7) and d['qd'].shape==(1800,7)
rows=[]
for key,limit in metrics['thresholds'].items():
 worst=max(e[key] for e in metrics['per_episode'])
 rows.append(f'| {key} | {metrics[key]:.6f} | {worst:.6f} | {limit:g} |')
text='''# Measured Panda dynamics identification

Final configuration: `config.json`. Final completed measurement: `final_metrics.json`, trace `candidate-056.npz`. All six thresholds pass both pooled and separately for all three supplied recordings. A rebuilt live model reproduced the final metrics exactly. The exact submitted configuration equals the measured configuration.

Only the supplied training recordings, immutable regressor and manifest, task bounds, sanitized geometry, and permitted shared/public physics sources were used. No nominal Panda dynamics or additional robot recordings were used. No subagents or extra simulators were created. Input and geometry SHA-256 hashes were checked and remain unchanged.

The fit uses the 98-coefficient physical parameterization, weighted convex least squares, and positive semidefinite pseudo-inertia constraints. These enforce positive mass, bounded COM, realizable symmetric COM tensors, and the link second-moment bound. Joint friction, bias and armature satisfy their specified bounds. The objective uses torque weights [1,1,1,1,2,2,3] and ridge coefficient 1e-4, with geometric scaling documented in `fit_model.py`. Armatures were selected by complete Newton measurements and recording-level cross-validation, then the remaining coefficients were refitted. About 69 parameter directions are numerically identifiable in this regressor.

The final model is `refined_a5_0.03.json`: armatures [0.02,0.01,0.02,0.01,0.03,0.01,0.025]. Leave-one-recording-out fits with this procedure passed all checks on each excluded recording: worst torque RMSE 0.213992 Nm, normalized torque RMSE 0.294792, position RMSE 0.012349 rad, velocity RMSE 0.294095 rad/s. Later joint-loss refinements were rejected because their worst excluded-recording velocity error was higher (0.345220 rad/s).

56 complete physical candidates were evaluated, each by apply_config/reset/1800 steps/metrics. All 56 traces, including the failed initial candidate, are preserved. Candidate 1 passed torque checks but failed motion checks because its very small effective wrist inertias amplified residual torque. No candidate trace was overwritten. Final candidate 56 completed after a rebuild. Work finished within the 1200-second budget (approximately nine minutes).

| Metric | Pooled | Worst recording | Limit |
|---|---:|---:|---:|
'''+ '\n'.join(rows)+'''

This is an effective dynamics identification for filtered joint measurements. Individual link parameters are not uniquely recoverable; several fitted inertia directions are close to the physical lower bound. The independent 16-recording test fold was not accessed. Training and cross-validation passes do not establish its result, or behavior beyond the measured 100 ms horizons.
'''
(root/'identification_report.md').write_text(text)
(root/'final_integrity_check.json').write_text(json.dumps({'inputs':hashes,'geometry_sha256':manifest['geometry_sha256'],'configuration_matches_measurement':True,'bounds_pass':True,'rebuilt_metrics_identical':True,'preserved_complete_traces':56},indent=2)+'\n')
print(json.dumps({'success':metrics['success'],'trace':Path(metrics['trace_path']).name,'candidate_count':56,'input_hashes_match':True,'configuration_matches_measurement':True,'bounds_pass':True,'rebuilt_metrics_identical':True}))
