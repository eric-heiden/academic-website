"""Constrained estimation using only the supplied measured-data regressor."""
import json
import sys
from pathlib import Path
import numpy as np
import cvxpy as cp

ROOT = Path(__file__).resolve().parent
REG = np.load(ROOT / 'training-regressor.npz')
A, b = REG['A'], REG['b']
EPISODES = REG['sample_episode_ids']

def decode(x):
    out = {k: [] for k in ('mass','com','inertia')}
    for j in range(7):
        p = x[10*j:10*j+10]
        mass = p[0]
        com = p[1:4] / mass
        inertia = np.array([[p[4],p[7],p[8]], [p[7],p[5],p[9]], [p[8],p[9],p[6]]])
        inertia -= mass * (com@com*np.eye(3) - np.outer(com,com))
        out['mass'].append(float(mass))
        out['com'].append(com.tolist())
        out['inertia'].append(inertia.tolist())
    for k, off in [('viscous',70),('coulomb',77),('torque_bias',84),('armature',91)]:
        out[k] = x[off:off+7].tolist()
    return out

def encode(c):
    x=[]
    for m,cc,ii in zip(c['mass'],c['com'],c['inertia']):
        cc=np.asarray(cc);ii=np.asarray(ii)+m*((cc@cc)*np.eye(3)-np.outer(cc,cc))
        x.extend([m,*(m*cc),ii[0,0],ii[1,1],ii[2,2],ii[0,1],ii[0,2],ii[1,2]])
    for k in ['viscous','coulomb','torque_bias','armature']: x.extend(c[k])
    return np.asarray(x)

def fit(ridge=1e-4, weight_power=0., exclude=None, loss='square', arm_min=1e-6, eig_min=1e-6, visc_min=1e-7):
    x=cp.Variable(98)
    cons=[]
    for j in range(7):
        p=x[10*j:10*j+10]
        mass,h=p[0],p[1:4]
        inertia=cp.bmat([[p[4],p[7],p[8]],[p[7],p[5],p[9]],[p[8],p[9],p[6]]])
        second=.5*cp.trace(inertia)*np.eye(3)-inertia
        pseudo=cp.bmat([[second,cp.reshape(h,(3,1),order='C')],[cp.reshape(h,(1,3),order='C'),cp.reshape(mass,(1,1),order='C')]])
        cons += [mass>=.05001,mass<=9.99999,h<=.39999*mass,h>=-.39999*mass,
                 pseudo-np.diag([eig_min,eig_min,eig_min,0]) >> 0,
                 cp.trace(second)<=.24999*mass]
    cons += [x[70:84]>=1e-7,x[70:84]<=4.99999,x[84:91]>=-1.99999,x[84:91]<=1.99999,x[91:98]>=arm_min,x[91:98]<=.99999]
    cons += [x[70:77]>=visc_min]
    sel=np.ones(len(b),dtype=bool) if exclude is None else np.repeat(EPISODES!=exclude,7)
    scale=np.maximum(b[sel].reshape(-1,7).std(axis=0),.5)**(-weight_power)
    weights=np.tile(scale,len(b)//7)[sel]
    residual=cp.multiply(weights,A[sel]@x-b[sel])
    # Dimension scales define a generic compact physical regularizer, not Panda priors.
    center=np.zeros(98)
    metric=np.ones(98)
    for j in range(7):
        center[10*j]=1.
        center[10*j+4:10*j+7]=.01
        metric[10*j:10*j+10]=[.2,1,1,1,3,3,3,3,3,3]
    objective=cp.sum_squares(residual)/np.sum(sel) if loss=='square' else cp.sum(cp.huber(residual,.15))/np.sum(sel)
    objective+=ridge*cp.sum_squares(cp.multiply(metric,x-center))
    problem=cp.Problem(cp.Minimize(objective),cons)
    problem.solve(solver='CLARABEL',tol_gap_abs=1e-9,tol_feas=1e-9,tol_gap_rel=1e-9,max_iter=300)
    if x.value is None: raise RuntimeError(problem.status)
    value=x.value
    err=(A@value-b).reshape(-1,7)
    report={'ridge':ridge,'weight_power':weight_power,'exclude':exclude,'loss':loss,'status':problem.status,'objective':problem.value,'rmse':np.sqrt(np.mean(err**2,axis=0)).tolist(),'per_episode':{str(e):np.sqrt(np.mean(err[EPISODES==e]**2,axis=0)).tolist() for e in np.unique(EPISODES)}}
    return decode(value),report

if __name__=='__main__':
    ridge=float(sys.argv[1]) if len(sys.argv)>1 else 1e-4
    power=float(sys.argv[2]) if len(sys.argv)>2 else 0
    name=sys.argv[3] if len(sys.argv)>3 else 'fit_initial'
    c,r=fit(ridge,power)
    (ROOT/(name+'.json')).write_text(json.dumps(c,indent=2)+'\n')
    (ROOT/(name+'-fit.json')).write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r));print('mass',c['mass']);print('joint parameters',{k:c[k] for k in ['viscous','coulomb','torque_bias','armature']})
