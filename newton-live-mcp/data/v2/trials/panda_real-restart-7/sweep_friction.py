from pathlib import Path
import json,subprocess
import numpy as np
from fit_model import solve,to_config,summary
keys=['max_joint_torque_rmse_nm','max_joint_torque_normalized_rmse','max_joint_position_rmse_rad','max_joint_velocity_rmse_rad_s','position_p95_rad','max_joint_speed_rad_s']
for number,floor,cmax in [(10,.025,.1),(11,.015,.1),(12,.025,0),(13,.015,0),(14,[.025,.025,.025,.025,.035,.025,.025],5)]:
    out=Path(f'candidate-{number:03d}');out.mkdir()
    x,status=solve(amin=np.asarray(floor),cmax=cmax)
    cfg=to_config(x);s=json.dumps(cfg,indent=2)+'\n';Path('config.json').write_text(s);(out/'config.json').write_text(s)
    (out/'fit.json').write_text(json.dumps({'floor':floor,'cmax':cmax,'status':status,'scores':summary(x)},indent=2)+'\n')
    with (out/'process.log').open('w') as f:
        r=subprocess.run(['uv','run','--no-sync','--project','/home/horde/apps/newton-live-mcp','python','-m','tools.mcp_evaluation.real_rollout','--config','config.json','--reference','training.npz','--output',str(out/'metrics.json')],stdout=f,stderr=subprocess.STDOUT)
    if r.returncode:print(number,'ERROR',flush=True);continue
    d=json.loads((out/'metrics.json').read_text())
    print(number,'floor',floor,'cmax',cmax,'pass',d['success'],{k:max(e[k] for e in d['per_episode']) for k in keys},flush=True)
    print('pooled torque',d['torque_rmse_per_joint_nm'],'position',d['position_rmse_per_joint_rad'],'velocity',d['velocity_rmse_per_joint_rad_s'],flush=True)
