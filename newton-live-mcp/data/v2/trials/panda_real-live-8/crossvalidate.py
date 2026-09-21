from fit_model import *
import time
rows=[]
for ridge in [1e-6,1e-4,1e-3]:
 for af in [.015,.025,.04]:
  arms=[.01]*4+[af,.01,af]
  name=f'cv_r{ridge}_a{af}'
  cfg,x,stats=fit([1,1,1,1,2,2,3],ridge,armature_fixed=arms)
  errs=[]
  for e in np.unique(ep):
   train=np.repeat(ep!=e,7)
   cf,xx,st=fit([1,1,1,1,2,2,3],ridge,train=train,armature_fixed=arms)
   errs.append(np.asarray(st['per_episode'][str(e)]))
  cverr=np.array(errs)
  stats['crossvalidation_rmse']=cverr.tolist();stats['name']=name
  rows.append(stats)
  (ROOT/(name+'.json')).write_text(json.dumps(cfg,indent=2)+'\n')
  np.save(ROOT/(name+'.npy'),x)
  print(name,'train',np.round(stats['rmse'],4),'CVmax',np.round(cverr.max(0),4),flush=True)
(ROOT/'crossvalidation.json').write_text(json.dumps(rows,indent=2)+'\n')
