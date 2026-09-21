import json
import numpy as np
from fit_model import ROOT, fit

summary=[]
for weight in (0.,.5):
    for floor in (.01,.02,.03,.04):
        name=f'floor_{floor:.2f}_w{weight:.1f}'
        cfg,report,x=fit(ridge=1e-5,weight=weight,armfloor=floor)
        (ROOT/f'{name}.json').write_text(json.dumps(cfg,indent=2)+'\n')
        (ROOT/f'{name}.fit.json').write_text(json.dumps(report,indent=2)+'\n')
        np.save(ROOT/f'{name}.coeff.npy',x)
        summary.append({'name':name,**report})
print(json.dumps(summary))
