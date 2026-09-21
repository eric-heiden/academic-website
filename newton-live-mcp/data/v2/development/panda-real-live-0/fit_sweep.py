from fit_model import *
D=np.load(ROOT/'training.npz');print('data',[(k,D[k].shape) for k in D.files]);print('qdd std',D['qdd'].std(axis=0));print('qd std',D['qd'].std(axis=0))
for arm in [.005,.01,.02,.04,.08,.15]:
    x,status,obj=solve(1e-5,arm_min=arm)
    name='arm_'+str(arm)
    (ROOT/(name+'.json')).write_text(json.dumps(config_from_x(x),indent=2)+'\n')
    np.save(ROOT/(name+'.npy'),x)
    print(name,status,'rmse',score(x),flush=True)
for lam in [1e-6,1e-5,1e-4,1e-3]:
    for e in ids:
        x,status,obj=solve(lam,train=E!=e)
        print('cross_validation',lam,int(e),score(x)[str(e)],flush=True)
