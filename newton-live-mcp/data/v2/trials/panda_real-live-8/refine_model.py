from fit_model import *
configs=[]
for a5 in [.025,.03,.035]:
 arms=[.02,.01,.02,.01,a5,.01,.025]
 name=f'refined_a5_{a5}'
 cfg,x,stats=fit([1,1,1,1,2,2,3],1e-4,armature_fixed=arms)
 (ROOT/(name+'.json')).write_text(json.dumps(cfg,indent=2)+'\n');np.save(ROOT/(name+'.npy'),x)
 print(name,json.dumps(stats),flush=True)
 for e in np.unique(ep):
  cf,xx,st=fit([1,1,1,1,2,2,3],1e-4,train=np.repeat(ep!=e,7),armature_fixed=arms)
  cvname=name+f'_leave_{e}'
  (ROOT/(cvname+'.json')).write_text(json.dumps(cf,indent=2)+'\n');np.save(ROOT/(cvname+'.npy'),xx)
  print(cvname,st['per_episode'][str(e)],flush=True)
