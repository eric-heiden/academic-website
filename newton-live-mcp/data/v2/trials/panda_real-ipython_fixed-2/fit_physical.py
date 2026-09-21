import json, time
from pathlib import Path
import numpy as np
import cvxpy as cp
from tools.mcp_evaluation.real_robot_model import validate_config, physical_coefficients

ROOT = Path('/home/horde/artifacts/newton-live-mcp-v2/confirmation/panda_real-ipython_fixed-2')
D = np.load(ROOT/'training-regressor.npz')
A, b = D['A'], D['b']
ep = D['sample_episode_ids']

def decode(x):
    out = {k: [] for k in ('mass','com','inertia')}
    for i in range(7):
        m, hx, hy, hz, xx, yy, zz, xy, xz, yz = x[10*i:10*i+10]
        c = np.array([hx,hy,hz])/m
        I0 = np.array([[xx,xy,xz],[xy,yy,yz],[xz,yz,zz]])
        Ic = I0-m*(np.dot(c,c)*np.eye(3)-np.outer(c,c))
        out['mass'].append(float(m)); out['com'].append(c.tolist()); out['inertia'].append(Ic.tolist())
    for k, start in [('viscous',70),('coulomb',77),('torque_bias',84),('armature',91)]:
        out[k] = x[start:start+7].tolist()
    return validate_config(out)

def fit(lam=1e-5, weights=None, mask=None, robust=False, amin=1e-6):
    x=cp.Variable(98)
    cons=[]
    for i in range(7):
        v=x[10*i:10*i+10]; m=v[0]; h=v[1:4]
        I=cp.bmat([[v[4],v[7],v[8]],[v[7],v[5],v[9]],[v[8],v[9],v[6]]])
        S=0.5*cp.trace(I)*np.eye(3)-I
        P=cp.bmat([[S,cp.reshape(h,(3,1),order='C')],[cp.reshape(h,(1,3),order='C'),cp.reshape(m,(1,1),order='C')]])
        cons += [P >> 1e-6*np.eye(4),m>=.05001,m<=9.99999,h>=-.39999*m,h<=.39999*m,cp.trace(S)<=.24999*m]
    cons += [x[70:84]>=1e-8,x[70:84]<=4.999999,x[84:91]>=-1.999999,x[84:91]<=1.999999,x[91:98]>=amin,x[91:98]<=.999999]
    if weights is None:
        weights=np.tile(np.array([1,1,1,1,1,1,2.]),len(b)//7)
    if mask is None: mask=np.ones(len(b),bool)
    residual=cp.multiply(weights[mask], A[mask]@x-b[mask])
    scales=np.r_[np.tile([1,.2,.2,.2,.1,.1,.1,.1,.1,.1],7),np.ones(21),np.full(7,.1)]
    loss=cp.sum(cp.huber(residual,.3)) if robust else cp.sum_squares(residual)
    prob=cp.Problem(cp.Minimize(loss/mask.sum()+lam*cp.sum_squares(cp.multiply(1/scales,x))),cons)
    t=time.time()
    prob.solve(solver='CLARABEL',tol_gap_abs=1e-9,tol_feas=1e-9,tol_gap_rel=1e-9,max_iter=300)
    cfg=decode(x.value)
    err=(A@physical_coefficients(cfg)-b).reshape(-1,7)
    print('fit',lam,prob.status,'seconds',time.time()-t,'RMSE',np.sqrt(np.mean(err**2,axis=0)))
    print('per episode',[(int(e),np.sqrt(np.mean(err[ep==e]**2,axis=0)).round(4).tolist()) for e in np.unique(ep)])
    return cfg, x.value.copy(), err
