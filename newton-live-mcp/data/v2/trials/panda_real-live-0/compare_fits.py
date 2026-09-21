import json
import numpy as np
from fit_physical import ROOT, fit, episodes

reports=[]
for amin in [.01,.02,.03,.04,.05]:
    cfg,x,rep=fit(lam=1e-5,arm_min=amin)
    name=f'fit-arm-{amin:g}'
    (ROOT/(name+'.json')).write_text(json.dumps(cfg,indent=2)+'\n')
    np.save(ROOT/(name+'.npy'),x)
    rep['arm_min']=amin
    cv=[]
    for ep in np.unique(episodes):
        _,_,rr=fit(lam=1e-5,arm_min=amin,mask=episodes!=ep)
        cv.append({'episode':int(ep),'rmse':rr['per_episode_rmse'][str(ep)]})
    rep['leave_one_episode_out']=cv
    reports.append(rep)
    print(json.dumps(rep),flush=True)
(ROOT/'arm-fit-reports.json').write_text(json.dumps(reports,indent=2)+'\n')
