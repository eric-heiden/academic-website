import json, copy
from pathlib import Path
import numpy as np
from scipy.optimize import lsq_linear
from fit_physical import A,b,episode,physical_coefficients,validate_config
pairs=[(key,j) for key in ('viscous','coulomb','torque_bias','armature') for j in (4,5,6)]
h=np.array([.005]*9+[.002]*3)

def residual(config,trace):
    err=(A@physical_coefficients(config)-b).reshape(-1,7)
    denom=np.empty_like(err)
    for ep in np.unique(episode):
        mask=episode==ep
        denom[mask]=np.minimum(.5,.5*np.maximum(.5,b.reshape(-1,7)[mask].std(axis=0)))
    torque=err/denom*np.sqrt(.6/len(err))
    vel=(trace['qd']-trace['reference_qd'])/.5*np.sqrt(.3/len(trace['qd']))
    pos=(trace['q']-trace['reference_q'])/.025*np.sqrt(.1/len(trace['q']))
    return np.r_[torque.ravel(),vel.ravel(),pos.ravel()]

def prepare(base_path,tag):
    base=json.loads(Path(base_path).read_text())
    paths=[]
    for i,(key,j) in enumerate(pairs):
        cfg=copy.deepcopy(base);cfg[key][j]+=h[i]
        validate_config(cfg)
        path=f'{tag}-fd-{i:02d}.json';Path(path).write_text(json.dumps(cfg,indent=2)+'\n');paths.append(path)
    Path(f'{tag}-paths.json').write_text(json.dumps(paths))

def update(base_path,base_trace,tag,fd_trace_start):
    base=json.loads(Path(base_path).read_text());r=residual(base,np.load(base_trace))
    J=np.empty((len(r),len(pairs)))
    for i in range(len(pairs)):
        cfg=json.loads(Path(f'{tag}-fd-{i:02d}.json').read_text());tr=np.load(f'candidate-{fd_trace_start+i:03d}.npz')
        J[:,i]=(residual(cfg,tr)-r)/h[i]
    z=np.array([base[k][j] for k,j in pairs])
    trust=np.array([.08]*3+[.05]*3+[.05]*3+[.015]*3)
    lo=np.maximum(-trust,np.r_[np.zeros(6),-np.ones(3)*2,np.ones(3)*.005]-z)
    hi=np.minimum(trust,np.r_[np.ones(6)*5,np.ones(3)*2,np.ones(3)*.08]-z)
    # Mild shrinkage relative to the inverse-dynamics fit limits window-specific changes.
    augJ=np.vstack([J,np.diag(.07/trust)])
    augr=np.r_[r,np.zeros(len(z))]
    opt=lsq_linear(augJ,-augr,bounds=(lo,hi),tol=1e-10,max_iter=200)
    print('base objective',float(r@r),'predicted objective',float(np.sum((r+J@opt.x)**2)))
    print('updates',dict(zip([f'{k}{j+1}' for k,j in pairs],opt.x.tolist())))
    paths=[]
    for i,alpha in enumerate([.5,1.,1.5]):
        cfg=copy.deepcopy(base)
        for (k,j),delta in zip(pairs,opt.x):cfg[k][j]+=float(alpha*delta)
        validate_config(cfg)
        path=f'{tag}-update-{i}.json';Path(path).write_text(json.dumps(cfg,indent=2)+'\n');paths.append(path)
    Path(f'{tag}-update-paths.json').write_text(json.dumps(paths))
    np.savez_compressed(f'{tag}-sensitivity.npz',J=J,residual=r,delta=opt.x)

if __name__=='__main__':
    import sys
    if sys.argv[1]=='prepare':prepare(sys.argv[2],sys.argv[3])
    else:update(sys.argv[2],sys.argv[3],sys.argv[4],int(sys.argv[5]))
