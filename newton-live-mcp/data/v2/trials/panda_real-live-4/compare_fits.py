import json
from pathlib import Path
import numpy as np
from fit_physical import fit, physical_coefficients, scores, episodes, A, b

specs=[('square',1e-5,a) for a in [.01,.02,.03,.04,.05]]
specs += [('huber',1e-5,a) for a in [.02,.03,.04]]
specs += [('square',r,.03) for r in [1e-7,.0001,.001]]
out=[]
for i,(loss,ridge,floor) in enumerate(specs):
    name=f'fit_{i:02d}_{loss}_{ridge}_{floor}'
    cfg,info=fit(ridge=ridge,armature_floor=floor,loss=loss)
    Path(name+'.json').write_text(json.dumps(cfg,indent=2)+'\n')
    cvs=[]
    for e in np.unique(episodes):
        cc,ii=fit(ridge=ridge,armature_floor=floor,loss=loss,exclude=e)
        cvs.append(ii['torque_rmse'][str(e)])
    entry={'name':name,'loss':loss,'ridge':ridge,'floor':floor,'fit':info,'cv_rmse':cvs}
    out.append(entry)
    print(json.dumps(entry),flush=True)
    Path('fit_comparison.json').write_text(json.dumps(out,indent=2)+'\n')
