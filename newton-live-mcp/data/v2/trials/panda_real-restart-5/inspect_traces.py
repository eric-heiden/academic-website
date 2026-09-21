import json
from pathlib import Path
import numpy as np
report=[]
for path in sorted(Path('.').glob('candidate-*/metrics.json')):
    m=json.loads(path.read_text());t=np.load(m['trace_path']);qerr=t['q']-t['reference_q'];verr=t['qd']-t['reference_qd'];terr=t['torque_prediction']-t['torque_target']
    assert t['q'].shape==(1800,7) and t['qd'].shape==(1800,7)
    assert np.allclose(np.sqrt(np.mean(qerr**2,axis=0)),m['position_rmse_per_joint_rad'])
    assert np.allclose(np.sqrt(np.mean(verr**2,axis=0)),m['velocity_rmse_per_joint_rad_s'])
    assert np.allclose(np.sqrt(np.mean(terr**2,axis=0)),m['torque_rmse_per_joint_nm'])
    windows=np.sqrt(np.mean(qerr.reshape(36,50,7)**2,axis=1)); ve=np.sqrt(np.mean(verr.reshape(36,50,7)**2,axis=1))
    r={'candidate':path.parent.name,'success':m['success'],'worst_episode_ratio':max(max(e[k]/v for k,v in m['thresholds'].items()) for e in m['per_episode']), 'worst_window_position':windows.max(axis=0).tolist(),'worst_window_velocity':ve.max(axis=0).tolist(),'torque_rmse':m['torque_rmse_per_joint_nm'],'worst_episode':{k:max(e[k] for e in m['per_episode']) for k in m['thresholds']}}
    report.append(r);print(json.dumps(r),flush=True)
Path('trace-diagnostics.json').write_text(json.dumps(report,indent=2)+'\n')
