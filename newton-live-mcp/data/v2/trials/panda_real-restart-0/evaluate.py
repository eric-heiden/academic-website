"""Evaluate one saved fit through the prescribed fresh-process workflow."""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('fit')
p.add_argument('number', type=int)
a = p.parse_args()
destination = Path(f'candidate-{a.number:03d}')
destination.mkdir(exist_ok=False)
shutil.copyfile(a.fit, 'config.json')
shutil.copyfile('config.json', destination/'config.json')
command = ['uv','run','--no-sync','--project','/home/horde/apps/newton-live-mcp','python','-m',
           'tools.mcp_evaluation.real_rollout','--config','config.json','--reference','training.npz',
           '--output',str(destination/'metrics.json')]
with (destination/'process.log').open('w') as f:
    result = subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
if result.returncode: raise RuntimeError(f'Candidate failed; see {destination}/process.log')
m = json.loads((destination/'metrics.json').read_text())
keys = ['success',*m['thresholds']]
print('candidate',a.number, {k:m[k] for k in keys},flush=True)
for e in m['per_episode']:
    print('episode',e['episode'],{k:round(e[k],6) if isinstance(e[k],float) else e[k] for k in keys},flush=True)
print('q',m['position_rmse_per_joint_rad'],'qd',m['velocity_rmse_per_joint_rad_s'],flush=True)
