import subprocess,json
from pathlib import Path
import numpy as np
r=np.load('training-regressor.npz');ids=r['sample_episode_ids'];b=r['b'].reshape(-1,7)
specs={
 'ls':['--arm-min','.03'],
 'huber10':['--arm-min','.03','--huber','.1'],
 'huber20':['--arm-min','.03','--huber','.2'],
 'wrist2':['--arm-min','.03','--weights','1,1,1,1,2,2,2'],
 'wrist4':['--arm-min','.03','--weights','1,1,1,1,4,4,4'],
 'ridge01':['--arm-min','.03','--ridge','.1'],
 'ridge1':['--arm-min','.03','--ridge','1'],
}
folder=Path('cross-validation');folder.mkdir(exist_ok=True); summaries={}
for label,args in specs.items():
 errors=[]
 for id in np.unique(ids):
  name=folder/f'{label}-exclude-{id}.json'
  with name.with_suffix('.log').open('w') as log:
   subprocess.run(['uv','run','--no-sync','--project','/home/horde/apps/newton-live-mcp','python','fit.py','--output',str(name),'--exclude',str(id)]+args,stdout=log,stderr=subprocess.STDOUT,check=True)
  e=np.load(name.with_suffix('.npz'))['errors'][ids==id];rmse=np.sqrt(np.mean(e*e,axis=0));norm=rmse/np.maximum(b[ids==id].std(axis=0),.5)
  errors.append({'episode':int(id),'rmse':rmse.tolist(),'normalized':norm.tolist()})
 summaries[label]=errors
 print(label,'worst held-record torque',np.round(np.max([x['rmse'] for x in errors],axis=0),4),'norm',np.round(np.max([x['normalized'] for x in errors],axis=0),4),flush=True)
(folder/'summary.json').write_text(json.dumps(summaries,indent=2))
