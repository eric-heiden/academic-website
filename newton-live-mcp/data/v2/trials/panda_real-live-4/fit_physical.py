"""Estimate full physical parameters from the supplied immutable design matrix."""
import json
import sys
from pathlib import Path
import numpy as np
import cvxpy as cp

sys.path.insert(0, '/home/horde/apps/newton-live-mcp')
from tools.mcp_evaluation.real_robot_model import validate_config, physical_coefficients

DATA = np.load('training-regressor.npz')
A, b = DATA['A'], DATA['b']
episodes = DATA['sample_episode_ids']
COMP = ((0,0),(1,1),(2,2),(0,1),(0,2),(1,2))

def to_config(x):
    cfg = {k: [] for k in ['mass','com','inertia','viscous','coulomb','torque_bias','armature']}
    for i in range(7):
        v = x[10*i:10*i+10]
        m = float(v[0]); c = v[1:4]/m
        io = np.zeros((3,3))
        for t,(a,d) in zip(v[4:],COMP): io[a,d] = io[d,a] = t
        ic = io - m*((c@c)*np.eye(3)-np.outer(c,c))
        cfg['mass'].append(m); cfg['com'].append(c.tolist()); cfg['inertia'].append(ic.tolist())
    for j,k in enumerate(['viscous','coulomb','torque_bias','armature']):
        lo,hi = (-2,2) if k=='torque_bias' else (0,1 if k=='armature' else 5)
        cfg[k] = np.clip(x[70+j*7:77+j*7],lo,hi).tolist()
    return validate_config(cfg)

def scores(x):
    r = (A@x-b).reshape(-1,7)
    result={}
    for e in [None,*np.unique(episodes)]:
        mask=np.ones(len(episodes),bool) if e is None else episodes==e
        result[str(e)] = np.sqrt(np.mean(r[mask]**2,axis=0)).tolist()
    return result

def fit(ridge=1e-5, exclude=None, armature_floor=0, loss='square', weights=None, center=None):
    x=cp.Variable(98)
    cons=[]
    for i in range(7):
        v=x[10*i:10*i+10]; m=v[0]; h=v[1:4]
        io=cp.bmat([[v[4],v[7],v[8]],[v[7],v[5],v[9]],[v[8],v[9],v[6]]])
        sigma=.5*cp.trace(io)*np.eye(3)-io
        pseudo=cp.bmat([[sigma,cp.reshape(h,(3,1),order='C')],[cp.reshape(h,(1,3),order='C'),cp.reshape(m,(1,1),order='C')]])
        cons += [m>=.050001,m<=9.999999,h>=-.399999*m,h<=.399999*m,
                 pseudo >> 1e-6*np.eye(4),cp.trace(sigma)<=.249999*m]
    cons += [x[70:84]>=0,x[70:84]<=5,x[84:91]>=-2,x[84:91]<=2,x[91:]>=armature_floor,x[91:]<=1]
    select=np.ones(len(b),bool) if exclude is None else np.repeat(episodes!=exclude,7)
    if weights is None:
        weights=np.ones_like(b)
        for e in np.unique(episodes):
            em=episodes==e
            scale=np.minimum(1,np.maximum(.5,b.reshape(-1,7)[em].std(axis=0)))
            weights.reshape(-1,7)[em]=1/scale
    residual=cp.multiply(weights[select],A[select]@x-b[select])
    scale=np.r_[np.tile([2,.2,.2,.2,.1,.1,.1,.1,.1,.1],7),np.ones(28)]
    if center is None: center=np.zeros(98)
    data_loss=cp.sum_squares(residual) if loss=='square' else cp.sum(cp.huber(residual,.2))
    objective=data_loss/np.sum(select)+ridge*cp.sum_squares(cp.multiply(1/scale,x-center))
    problem=cp.Problem(cp.Minimize(objective),cons)
    problem.solve(solver='CLARABEL',tol_gap_abs=1e-9,tol_feas=1e-9,tol_gap_rel=1e-9,max_iter=200)
    if x.value is None: raise RuntimeError(problem.status)
    cfg=to_config(x.value)
    return cfg,{'status':problem.status,'objective':problem.value,'torque_rmse':scores(physical_coefficients(cfg))}

if __name__=='__main__':
    cfg,info=fit()
    Path('fit_initial.json').write_text(json.dumps(cfg,indent=2)+'\n')
    Path('fit_initial_summary.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info))
    print(json.dumps(cfg))
