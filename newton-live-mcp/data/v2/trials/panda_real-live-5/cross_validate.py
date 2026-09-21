"""Leave-one-recording-out model selection, using only training recordings."""
import json
import numpy as np
from fit_model import ROOT, fit, b

floors=[.02,.01,.02,.01,.04,.02,.02]
std=np.maximum(b.reshape(3,-1,7).std(axis=1),.5)
summary=[]
for ridge in (1e-7,1e-6,1e-5,1e-4,1e-3):
    for weight in (0.,.5):
        validation=[]
        reports=[]
        for idx,ep in enumerate((2,3,4)):
            cfg,report,x=fit(ridge=ridge,weight=weight,armfloor=floors,exclude=ep)
            reports.append(report)
            e=np.array(report['per_episode_rmse'])[idx]
            validation.append(e)
            name=f'cv_r{ridge}_w{weight}_ex{ep}'
            (ROOT/f'{name}.json').write_text(json.dumps(cfg,indent=2)+'\n')
            (ROOT/f'{name}.fit.json').write_text(json.dumps(report,indent=2)+'\n')
        validation=np.array(validation)
        summary.append({'ridge':ridge,'weight':weight,'max_validation_torque_rmse':validation.max(),'max_validation_normalized':(validation/std).max(),'validation_rmse':validation.tolist(),'mean_squared':np.mean(validation**2)})
(ROOT/'cross_validation.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary))
