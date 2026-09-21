import json, subprocess, sys, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent

def run(config, label):
    existing=list(ROOT.glob('candidate-[0-9][0-9][0-9]'))
    num=max([int(p.name.split('-')[1]) for p in existing]+[0])+1
    folder=ROOT/f'candidate-{num:03d}';folder.mkdir()
    (folder/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (ROOT/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (folder/'label.txt').write_text(label+'\n')
    with (folder/'process.log').open('w') as log:
        p=subprocess.run(['uv','run','--no-sync','--project','/home/horde/apps/newton-live-mcp','python','-m','tools.mcp_evaluation.real_rollout','--config','config.json','--reference','training.npz','--output',str(folder/'metrics.json')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if p.returncode:
        print('FAILED',folder.name,p.returncode,flush=True);return None
    m=json.loads((folder/'metrics.json').read_text())
    keys=list(m['thresholds'])
    worst={k:max(v[k] for v in m['per_episode']) for k in keys}
    print(folder.name,label,'success',m['success'],'worst',json.dumps(worst),'pos',m['position_rmse_per_joint_rad'],'vel',m['velocity_rmse_per_joint_rad_s'],flush=True)
    with (ROOT/'candidate-summary.jsonl').open('a') as f:f.write(json.dumps({'candidate':folder.name,'label':label,'success':m['success'],'worst':worst})+'\n')
    return m

if __name__=='__main__':
    base=json.load(open('fit-001.json'))
    for arm in [.025,.05,.075]:
        c=json.loads(json.dumps(base));c['armature']=(np.array(c['armature'])+arm).tolist()
        run(c,f'base fit plus uniform armature {arm}')
