import json,hashlib
from pathlib import Path
import numpy as np
from tools.mcp_evaluation.real_robot_model import validate_config,physical_coefficients
files=sorted(Path('.').glob('candidate-*/metrics.json'))
last=files[-1];m=json.loads(last.read_text());c=json.loads(Path('config.json').read_text());validate_config(c)
assert c==m['config'];assert m['success'] and m['frames']==1800 and all(e['success'] for e in m['per_episode'])
assert c==json.loads(Path('candidate-016/config.json').read_text())
summary=[]
for f in files:
 q=json.loads(f.read_text());summary.append({'candidate':f.parent.name,'success':q['success'],'frames':q['frames'],'pid':q['pid'],'metrics':str(f),'trace':q['trace_path'],'worst_per_recording':{k:max(e[k] for e in q['per_episode']) for k in q['thresholds']}})
Path('all-candidate-results.json').write_text(json.dumps(summary,indent=2)+'\n')
worst={k:max(e[k] for e in m['per_episode']) for k in m['thresholds']}
config_sha=hashlib.sha256(Path('config.json').read_bytes()).hexdigest()
text=f'''# Identified effective Panda dynamics

The exact final numeric model is `config.json` (SHA-256 `{config_sha}`). Fresh-process validation: `{last}`, complete 1,800-step trace: `{m['trace_path']}`. Every pooled and individual-recording threshold passed. All {len(files)} complete physical candidate evaluations, including failures, remain in their separate candidate directories with process logs, configurations, metrics and traces. Every candidate used a separate invocation of the prescribed `real_rollout` process. No persistent simulator or additional simulator was used.

## Estimation

Only the supplied three measured recordings, immutable 3,150-by-98 numerical regressor, mapping manifest and sanitized geometry were used. The model was estimated from the supplied homogeneous placeholders without nominal Panda dynamic parameters. `fit_model.py` contains the estimator. The measured regressor has 69 well-resolved singular directions; individual physical link parameters are not uniquely identified.

The convex fit estimates all 70 link coefficients and 28 joint coefficients. For each link, its pseudo-inertia matrix [[S,h],[h^T,m]] is positive definite, where S = trace(I_origin)/2 * identity - I_origin and h = mass * COM. This enforces positive COM inertia and the physical triangle inequalities. Explicit constraints also enforce the supplied mass, COM, second-moment and joint-loss bounds. Exported tensors are validated using the public exact physical mapping.

The final fit uses a Huber loss with transition 0.1 after dividing each recording/joint's torque residual by its torque standard deviation clipped to [0.5,1] Nm. A small dimensionless quadratic penalty (coefficient 1e-5) regularizes the parameter representation. Pseudo-inertia second-moment eigenvalues are bounded below by 1e-5. Effective armature lower bounds are [0.03,0.02,0.03,0.02,0.025,0.025,0.015] kg*m^2. These regularization choices were assessed by the prescribed physical candidates; unconstrained torque fitting alone produced inadequate wrist motion. Candidate 016 was selected for its balance of torque and motion error, with reduced armature at joint 7 to limit acceleration-related torque error.

## Final training quality

| Criterion | Pooled | Worst recording | Required maximum |
|---|---:|---:|---:|
'''
for k,limit in m['thresholds'].items():text+=f'| {k} | {m[k]:.6f} | {worst[k]:.6f} | {limit:g} |\n'
text+='\nLeave-one-recording-out refits of the final regularization also passed all thresholds, including each omitted recording (candidates 022–024). These use only the three supplied training recordings and are not the independent published test fold.\n'
text+='\n## Limitations and provenance\n\nThese are effective dynamics estimates for publisher-filtered measured torque, not uniquely recovered physical link properties or raw motor commands. Some weakly identified parameters meet their regularization bounds. The 100 ms prediction checks do not establish long-horizon simulation accuracy. The 16-recording independent fold was not accessed or evaluated; its final verification is external to this run.\n\nTraining source: LIP4RobotInverseDynamics, DOI 10.5281/zenodo.12516500, CC-BY-SA-4.0 (as supplied in the immutable input manifest). Supplied hashes of training.npz, training-regressor.npz, training-regressor.manifest.json and sanitized geometry were checked unchanged. No scoring code, reference data, geometry, solver algorithm or shared source files were edited.\n'
Path('identification-report.md').write_text(text)
print(json.dumps({'final_candidate':last.parent.name,'count':len(files),'unique_pids':len(set(q['pid'] for q in summary)),'success':m['success'],'config_sha256':config_sha,'worst_recording':worst},indent=2))
