"""One fresh prescribed rollout process per retained numeric configuration."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

p=argparse.ArgumentParser()
p.add_argument('number',type=int)
p.add_argument('source')
args=p.parse_args()
directory=Path(f'candidate-{args.number:03d}')
directory.mkdir(exist_ok=False)
shutil.copyfile(args.source,'config.json')
shutil.copyfile('config.json',directory/'config.json')
started=time.time()
with (directory/'process.log').open('w') as log:
    result=subprocess.run(['uv','run','--no-sync','--project','/home/horde/apps/newton-live-mcp',
                           'python','-m','tools.mcp_evaluation.real_rollout','--config','config.json',
                           '--reference','training.npz','--output',str(directory/'metrics.json')],
                          stdout=log,stderr=subprocess.STDOUT)
if result.returncode:
    raise RuntimeError(f'{directory}: rollout failed with code {result.returncode}; see process.log')
m=json.loads((directory/'metrics.json').read_text())
summary={'candidate':args.number,'wall_seconds':time.time()-started,
         'scores':[{k:v for k,v in e.items() if k in ['episode','success',
            'max_joint_torque_rmse_nm','max_joint_torque_normalized_rmse',
            'max_joint_position_rmse_rad','max_joint_velocity_rmse_rad_s',
            'position_p95_rad','max_joint_speed_rad_s']} for e in [m]+m['per_episode']]}
with Path('candidate-summary.jsonl').open('a') as f:
    f.write(json.dumps(summary)+'\n')
print(json.dumps(summary,indent=2))
