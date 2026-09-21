from pathlib import Path
import json, hashlib
import numpy as np
root=Path(__file__).resolve().parent
m=json.loads((root/'final_measurement.json').read_text())
task=json.loads((root/'task.json').read_text())
for name,sha in task['input_hashes'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha
assert m['config']==json.loads((root/'config.json').read_text())
keys=list(task['thresholds'])
rows=['| Metric | Pooled | Worst recording | Limit |','|---|---:|---:|---:|']
for k in keys:
    rows.append(f"| {k} | {m[k]:.6f} | {max(e[k] for e in m['per_episode']):.6f} | {task['thresholds'][k]} |")
cv=[]
for i,e in [(15,2),(16,3),(17,4)]:
    p=next(root.glob(f'measurement_{i:03d}_*.json'))
    v=json.loads(p.read_text()); held=next(x for x in v['per_episode'] if x['episode']==e)
    cv.append({'omitted_training_episode':e,**{k:held[k] for k in keys},'success':held['success']})
(root/'cross_validation.json').write_text(json.dumps(cv,indent=2)+'\n')
report='''# Physical Panda identification\n\nThe submitted full numeric model is `config.json`. Its final complete Newton measurement passed all six thresholds, pooled and in every supplied recording (episodes 2, 3, and 4). Final trace: `candidate-022.npz`; full metrics: `final_measurement.json`. There were 22 complete physical candidate evaluations, including the final repeat. All candidate traces and evaluation logs are retained.\n\n## Estimation\n\nCustom CVXPY estimation fits all 98 physical coefficients using the supplied immutable Newton regressor. Each link is represented by mass, first mass moment, and symmetric inertia about the link origin. A positive semidefinite pseudo-inertia constraint guarantees physical COM inertias and the triangle inequalities. Mass, COM, second moment, friction, torque bias, and armature obey the task bounds. A small positive second-moment margin avoids singular tensors. A weak quadratic regularizer selects a representative among poorly identifiable link parameter combinations; its prior uses only the supplied homogeneous placeholders. No nominal Panda dynamic parameters were used.\n\nThe first least-squares candidate fit torque well but failed wrist motion checks. Complete prescribed Newton evaluations were used to select modest joint-armature lower bounds. The final fit uses lower bounds [0.025, 0.01, 0.025, 0.015, 0.03, 0.02, 0.015] kg*m^2, a 1e-4 pseudo-inertia margin, and 1e-4 quadratic regularization. Loss-parameter alternatives and broader armature sweeps were retained but not selected. `fit_physical.py` and `fit_balanced.py` reproduce the selected offline estimate.\n\n## Final measured quality\n\n'''+ '\n'.join(rows)+'''\n\n## Validation and limitations\n\nThree separate fits each omitted one entire training recording. All three omitted-recording torque and forward-motion measurements passed; details are in `cross_validation.json`. This is an internal robustness check, not the independent published test fold. The 16 independent recordings remain inaccessible and unverified until the external verification process runs.\n\nThe regressor has approximately 69 numerically resolved directions (singular values above 1e-3), so individual link masses, centers, and inertia tensors are not uniquely determined. The result is an effective physical link/joint model. Publisher-filtered torque and state derivatives introduce model mismatch; measured joint torque is not a recovered raw motor command. The selected armature regularization balances torque agreement with short-horizon motion prediction.\n\nThe three immutable input hashes were rechecked successfully after fitting. No shared sources, geometry, target windows, callbacks, or solver settings were modified.\n'''
(root/'identification_report.md').write_text(report)
print(json.dumps({'success':m['success'],'candidate_count':22,'cross_validation':cv,'config_sha256':hashlib.sha256((root/'config.json').read_bytes()).hexdigest(),'preserved_traces':len(list(root.glob('candidate-*.npz')))},indent=2))
