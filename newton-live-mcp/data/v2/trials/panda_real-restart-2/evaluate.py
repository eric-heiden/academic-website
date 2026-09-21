import argparse,subprocess,json,shutil,time,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('number',type=int);a=p.parse_args()
d=Path(f'candidate-{a.number:03d}');d.mkdir(exist_ok=False)
shutil.copyfile(a.config,'config.json');shutil.copyfile('config.json',d/'config.json')
with (d/'process.log').open('w') as log:
 r=subprocess.run(['uv','run','--no-sync','--project','/home/horde/apps/newton-live-mcp','python','-m','tools.mcp_evaluation.real_rollout','--config','config.json','--reference','training.npz','--output',str(d/'metrics.json')],stdout=log,stderr=subprocess.STDOUT)
if r.returncode: print('FAILED',a.number,r.returncode);raise SystemExit(r.returncode)
m=json.loads((d/'metrics.json').read_text())
print('CANDIDATE',a.number,'success',m['success'],flush=True)
for e in [m]+m['per_episode']:
 print(e.get('episode','pooled'),'torque',round(e['max_joint_torque_rmse_nm'],4),'norm',round(e['max_joint_torque_normalized_rmse'],4),'pos',[round(v,4) for v in e['position_rmse_per_joint_rad']],'vel',[round(v,4) for v in e['velocity_rmse_per_joint_rad_s']],flush=True)
