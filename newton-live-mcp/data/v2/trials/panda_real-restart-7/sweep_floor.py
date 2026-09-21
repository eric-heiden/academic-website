from pathlib import Path
import json, subprocess, sys, time
import numpy as np
from fit_model import solve,to_config,summary
keys=['max_joint_torque_rmse_nm','max_joint_torque_normalized_rmse','max_joint_position_rmse_rad','max_joint_velocity_rmse_rad_s','position_p95_rad','max_joint_speed_rad_s']
for number,floor in [(3,.025),(4,.015),(5,.035)]:
    out=Path(f'candidate-{number:03d}');out.mkdir()
    x,status=solve(amin=floor)
    cfg=to_config(x);s=json.dumps(cfg,indent=2)+'\n';Path('config.json').write_text(s);(out/'config.json').write_text(s)
    (out/'fit.json').write_text(json.dumps({'floor':floor,'status':status,'scores':summary(x)},indent=2)+'\n')
    with (out/'process.log').open('w') as f:
        result=subprocess.run(['uv','run','--no-sync','--project','/home/horde/apps/newton-live-mcp','python','-m','tools.mcp_evaluation.real_rollout','--config','config.json','--reference','training.npz','--output',str(out/'metrics.json')],stdout=f,stderr=subprocess.STDOUT)
    if result.returncode: print(number,'ERROR',flush=True);continue
    d=json.loads((out/'metrics.json').read_text())
    print(number,floor,d['success'],'worst episodes',{k:max(e[k] for e in d['per_episode']) for k in keys},flush=True)
    print('pooled torque',d['torque_rmse_per_joint_nm'],'position',d['position_rmse_per_joint_rad'],'velocity',d['velocity_rmse_per_joint_rad_s'],flush=True)
