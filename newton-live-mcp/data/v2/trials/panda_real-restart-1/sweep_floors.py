import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,subprocess,sys
import numpy as np
from fit_model import fit,to_config,report

d=np.load('training-regressor.npz');A,b,ep=d['A'],d['b'],d['sample_episode_ids']
for number,arm_floor in [(3,.02),(4,.04),(5,.05)]:
 directory=Path(f'candidate-{number:03d}');directory.mkdir()
 x,status,objective=fit(A,b,ep,lam=1e-4,arm_floor=arm_floor)
 c=to_config(x)
 Path('config.json').write_text(json.dumps(c,indent=2)+'\n');(directory/'config.json').write_text(json.dumps(c,indent=2)+'\n')
 fit_report={'armature_floor':arm_floor,'status':status,'objective':objective,'cv':{}}
 for e in np.unique(ep):
  xc,st,ob=fit(A,b,ep,lam=1e-4,mask=np.repeat(ep!=e,7),arm_floor=arm_floor)
  fit_report['cv'][str(e)]=report(A,b,ep,xc)['episodes'][str(e)]
 (directory/'fit-report.json').write_text(json.dumps(fit_report,indent=2)+'\n')
 env=os.environ.copy();env['NEWTON_EVAL_REAL_GEOMETRY']='/home/horde/artifacts/newton-live-mcp-v2/real-robot/public/geometry/panda_geometry.xml'
 with (directory/'process.log').open('x') as log:
  result=subprocess.run(['uv','run','--no-sync','--project','/home/horde/apps/newton-live-mcp','python','-m','tools.mcp_evaluation.real_rollout','--config','config.json','--reference','training.npz','--output',str(directory/'metrics.json')],env=env,stdout=log,stderr=subprocess.STDOUT)
 if result.returncode:raise RuntimeError(result.returncode)
 subprocess.run([sys.executable,'summarize.py',str(directory/'metrics.json')],check=True)
