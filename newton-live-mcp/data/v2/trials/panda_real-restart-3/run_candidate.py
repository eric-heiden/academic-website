import argparse,json,subprocess,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--note',default='');a=p.parse_args()
existing=list(Path('.').glob('candidate-*/metrics.json')); nums=[int(p.name.split('-')[1]) for p in Path('.').glob('candidate-*') if p.is_dir()]; n=max(nums,default=0)+1
if n>60:raise RuntimeError('Candidate budget exhausted')
out=Path(f'candidate-{n:03d}');out.mkdir(); config=Path(a.source).read_text();Path('config.json').write_text(config);(out/'config.json').write_text(config);(out/'note.txt').write_text(a.note+'\n')
cmd=['uv','run','--no-sync','--project','/home/horde/apps/newton-live-mcp','python','-m','tools.mcp_evaluation.real_rollout','--config','config.json','--reference','training.npz','--output',str(out/'metrics.json')]
start=time.time()
with (out/'process.log').open('w') as stream:r=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT)
if r.returncode:print('FAIL',out,'exit',r.returncode);print((out/'process.log').read_text()[-4000:]);raise SystemExit(r.returncode)
m=json.loads((out/'metrics.json').read_text());summary={'candidate':n,'note':a.note,'wall_seconds':time.time()-start,'success':m['success']}
for k in m['thresholds']:summary[k]=max(e[k] for e in [m]+m['per_episode'])
summary['position_per_joint']=np.max([e['position_rmse_per_joint_rad'] for e in m['per_episode']],axis=0).tolist();summary['velocity_per_joint']=np.max([e['velocity_rmse_per_joint_rad_s'] for e in m['per_episode']],axis=0).tolist()
with Path('candidate-summary.jsonl').open('a') as f:f.write(json.dumps(summary)+'\n')
print(json.dumps(summary),flush=True)
