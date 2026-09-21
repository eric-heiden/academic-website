import json
import numpy as np
from fit_physical import ROOT, fit, episodes

reports=[]
for amin in [.015,.025,.035]:
    for damping in [.2,.3,.4,.5]:
        arms=np.array([.025,.025,.025,.025,amin,.025,.02])
        visc=np.array([0.,0.,0.,0.,damping,0.,0.])
        cfg,x,rep=fit(lam=1e-5,arm_min=arms,visc_min=visc)
        name=f'fit-damping-{amin:g}-{damping:g}'
        (ROOT/(name+'.json')).write_text(json.dumps(cfg,indent=2)+'\n')
        np.save(ROOT/(name+'.npy'),x)
        rep['arm_min']=arms.tolist();rep['damping_min']=damping;rep['name']=name
        cv=[]
        for ep in np.unique(episodes):
            _,_,rr=fit(lam=1e-5,arm_min=arms,visc_min=visc,mask=episodes!=ep)
            cv.append({'episode':int(ep),'rmse':rr['per_episode_rmse'][str(ep)]})
        rep['leave_one_episode_out']=cv
        reports.append(rep)
        print(json.dumps({'name':name,'rmse':rep['rmse'],'cv_worst':np.max([r['rmse'] for r in cv],axis=0).tolist()}),flush=True)
(ROOT/'damping-fit-reports.json').write_text(json.dumps(reports,indent=2)+'\n')
