import json
import numpy as np
from fit_model import ROOT, fit

# Rotor lower bounds chosen from training torque/forward tradeoff and checked
# through leave-one-recording-out fits. All link parameters remain estimated.
floors=[.02,.01,.02,.01,.04,.02,.02]
cfg,report,x=fit(ridge=1e-5,weight=0.,armfloor=floors)
(ROOT/'selected_model.json').write_text(json.dumps(cfg,indent=2)+'\n')
(ROOT/'selected_model.fit.json').write_text(json.dumps(report,indent=2)+'\n')
np.save(ROOT/'selected_model.coeff.npy',x)
print(json.dumps(report))
