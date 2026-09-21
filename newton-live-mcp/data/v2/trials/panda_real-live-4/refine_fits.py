import json
from pathlib import Path
import numpy as np
from fit_physical import fit, episodes

specs=[]
for loss in ['square','huber']:
    for a5 in [.025,.035,.045]:
        floors=np.array([.03,.01,.03,.01,a5,.02,.015])
        specs.append((loss,floors))
out=[]
for i,(loss,floors) in enumerate(specs):
    name=f'refined_{i:02d}_{loss}'
    cfg,info=fit(ridge=1e-5,armature_floor=floors,loss=loss)
    Path(name+'.json').write_text(json.dumps(cfg,indent=2)+'\n')
    cvs=[]
    for e in np.unique(episodes):
        cc,ii=fit(ridge=1e-5,armature_floor=floors,loss=loss,exclude=e)
        Path(name+f'_exclude_{e}.json').write_text(json.dumps(cc,indent=2)+'\n')
        cvs.append(ii['torque_rmse'][str(e)])
    out.append({'name':name,'loss':loss,'floors':floors.tolist(),'fit':info,'cv_rmse':cvs})
    print(json.dumps(out[-1]),flush=True)
    Path('refined_comparison.json').write_text(json.dumps(out,indent=2)+'\n')
