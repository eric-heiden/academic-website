from fit_physical import *
import contextlib
results=[]
with (BASE/'crossvalidation-fit.log').open('w') as log, contextlib.redirect_stdout(log):
 for robust in [False,True]:
  for lam in [1e-6,1e-5,1e-4,1e-3,1e-2]:
   validation=np.zeros((450,7))
   for ep in np.unique(episodes):
    mask=np.repeat(episodes!=ep,7)
    p,c=fit(lam=lam,mask=mask,arm_min=np.array([.025,.005,.025,.005,.03,.01,.02]),robust=robust)
    validation[episodes==ep]=(A@p-b).reshape(-1,7)[episodes==ep]
   rmse=np.sqrt(np.mean(validation**2,axis=0))
   val_ep=[np.sqrt(np.mean(validation[episodes==ep]**2,axis=0)) for ep in np.unique(episodes)]
   norm=np.array(val_ep)/np.maximum(b.reshape(3,150,7).std(axis=1),.5)
   results.append(dict(robust=robust,lam=lam,rmse=rmse.tolist(),worst_nm=float(np.max(val_ep)),worst_normalized=float(norm.max())))
(BASE/'crossvalidation.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
