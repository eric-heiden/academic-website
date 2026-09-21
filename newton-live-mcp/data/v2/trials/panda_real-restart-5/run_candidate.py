import argparse,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('fit');p.add_argument('index',type=int);a=p.parse_args()
out=Path(f'candidate-{a.index:03d}')
out.mkdir(exist_ok=False)
config=json.loads(Path(a.fit).read_text())
Path('config.json').write_text(json.dumps(config,indent=2)+'\n')
(out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
env=os.environ.copy();env['NEWTON_EVAL_REAL_GEOMETRY']='/home/horde/artifacts/newton-live-mcp-v2/real-robot/public/geometry/panda_geometry.xml';env['OPENBLAS_NUM_THREADS']='1'
cmd=['uv','run','--no-sync','--project','/home/horde/apps/newton-live-mcp','python','-m','tools.mcp_evaluation.real_rollout','--config','config.json','--reference','training.npz','--output',str(out/'metrics.json')]
with (out/'process.log').open('w') as log:
    code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env).returncode
if code:print('FAILED PROCESS',code,flush=True);sys.exit(code)
m=json.loads((out/'metrics.json').read_text());keys=['success',*m['thresholds']]
print(json.dumps({'index':a.index,'pooled':{k:m[k] for k in keys},'worst_episode':{k:max(e[k] for e in m['per_episode']) for k in m['thresholds']},'position':m['position_rmse_per_joint_rad'],'velocity':m['velocity_rmse_per_joint_rad_s']}),flush=True)
