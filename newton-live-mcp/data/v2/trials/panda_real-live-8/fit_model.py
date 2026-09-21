import json
from pathlib import Path
import numpy as np
import cvxpy as cp

ROOT=Path(__file__).resolve().parent
D=np.load(ROOT/'training-regressor.npz')
A=D['A']; b=D['b']; ep=D['sample_episode_ids']; joints=np.tile(np.arange(7),len(ep))


def unpack(x):
    cfg={k:[] for k in ['mass','com','inertia','viscous','coulomb','torque_bias','armature']}
    for i in range(7):
        v=x[i*10:i*10+10];m=float(v[0]);h=v[1:4];c=h/m
        I=np.array([[v[4],v[7],v[8]],[v[7],v[5],v[9]],[v[8],v[9],v[6]]])
        Ic=I-m*(np.dot(c,c)*np.eye(3)-np.outer(c,c))
        Ic=(Ic+Ic.T)*.5
        cfg['mass'].append(m);cfg['com'].append(c.tolist());cfg['inertia'].append(Ic.tolist())
    for k,sl in zip(['viscous','coulomb','torque_bias','armature'],[slice(70,77),slice(77,84),slice(84,91),slice(91,98)]):cfg[k]=x[sl].tolist()
    return cfg


def fit(weights=np.ones(7),ridge=1e-4,train=None,robust=None,armature_floor=0,prior=None,armature_fixed=None):
    x=cp.Variable(98)
    cons=[]
    for i in range(7):
        v=x[i*10:i*10+10];m=v[0];h=v[1:4]
        I=cp.bmat([[v[4],v[7],v[8]],[v[7],v[5],v[9]],[v[8],v[9],v[6]]]);S=.5*cp.trace(I)*np.eye(3)-I
        P=cp.bmat([[S,cp.reshape(h,(3,1),order='C')],[cp.reshape(h,(1,3),order='C'),cp.reshape(m,(1,1),order='C')]])
        cons += [m>=.050001,m<=9.999999,h<=.399999*m,h>=-.399999*m,P >> np.diag([1e-6]*3+[0]),cp.trace(S)<=.249999*m]
    cons += [x[70:84]>=1e-8,x[70:84]<=4.999999,x[84:91]>=-1.999999,x[84:91]<=1.999999,x[91:98]>=armature_floor+1e-8,x[91:98]<=.999999]
    if armature_fixed is not None: cons += [x[91:98] == np.asarray(armature_fixed)]
    if train is None:train=np.ones(len(b),dtype=bool)
    scale=np.tile(weights,len(ep))
    z=cp.multiply(scale[train],A[train]@x-b[train])
    loss=cp.sum_squares(z) if robust is None else cp.sum(cp.huber(z,robust))
    pscale=np.r_[np.tile([5,.5,.5,.5,.2,.2,.2,.2,.2,.2],7),np.ones(21),np.ones(7)*.1]
    reg=cp.sum_squares(cp.multiply(1/pscale,x if prior is None else x-prior))
    problem=cp.Problem(cp.Minimize(loss/train.sum()+ridge*reg),cons)
    problem.solve(solver='CLARABEL',tol_gap_abs=1e-9,tol_feas=1e-9,tol_gap_rel=1e-9,max_iter=250)
    if x.value is None:raise RuntimeError(problem.status)
    v=x.value
    err=(A@v-b).reshape(-1,7)
    stats={'status':problem.status,'objective':problem.value,'rmse':np.sqrt((err**2).mean(0)).tolist(),'per_episode':{str(e):np.sqrt((err[ep==e]**2).mean(0)).tolist() for e in np.unique(ep)}}
    return unpack(v),v,stats

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--name',default='fit_001');p.add_argument('--ridge',type=float,default=1e-4);p.add_argument('--weights',type=float,nargs=7,default=[1,1,1,1,2,2,3]);p.add_argument('--robust',type=float);p.add_argument('--armature-floor',type=float,default=0)
    args=p.parse_args();cfg,x,stats=fit(args.weights,args.ridge,robust=args.robust,armature_floor=args.armature_floor)
    (ROOT/(args.name+'.json')).write_text(json.dumps(cfg,indent=2)+'\n')
    np.save(ROOT/(args.name+'.npy'),x)
    (ROOT/(args.name+'_fit.json')).write_text(json.dumps(stats,indent=2)+'\n')
    print(json.dumps(stats)); print('mass',cfg['mass']);print('joint', {k:cfg[k] for k in ['viscous','coulomb','torque_bias','armature']})
