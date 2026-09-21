import json
from pathlib import Path
import numpy as np
import cvxpy as cp

COMPONENTS = ((0,0),(1,1),(2,2),(0,1),(0,2),(1,2))

def coefficient_config(v):
    out = {k: [] for k in ('mass','com','inertia')}
    for j in range(7):
        p = np.asarray(v[j*10:(j+1)*10]); m = p[0]; c = p[1:4]/m
        I = np.zeros((3,3))
        for z,(a,b) in zip(p[4:], COMPONENTS): I[a,b] = I[b,a] = z
        Ic = I - m*(c@c*np.eye(3)-np.outer(c,c))
        out['mass'].append(float(m)); out['com'].append(c.tolist()); out['inertia'].append(Ic.tolist())
    for k, start in zip(('viscous','coulomb','torque_bias','armature'),(70,77,84,91)):
        out[k] = np.asarray(v[start:start+7]).tolist()
    return out

def fit_physical(A,b, ridge=1e-4, weights=None, armature_min=1e-7, inertia_min=1e-6, rows=None, robust=None):
    if rows is not None: A,b = A[rows],b[rows]
    x = cp.Variable(98)
    con = []
    for j in range(7):
        p=x[10*j:10*j+10]; m=p[0]; h=p[1:4]
        I=cp.bmat([[p[4],p[7],p[8]],[p[7],p[5],p[9]],[p[8],p[9],p[6]]])
        S=0.5*cp.trace(I)*np.eye(3)-I
        P=cp.bmat([[S-cp.Constant(inertia_min*np.eye(3)), cp.reshape(h,(3,1),order='C')],[cp.reshape(h,(1,3),order='C'),cp.reshape(m,(1,1),order='C')]])
        con += [m>=0.05001, m<=9.99999, h>=-0.39999*m,h<=0.39999*m,P>>0,cp.trace(S)<=0.24999*m]
    con += [x[70:84]>=1e-8,x[70:84]<=4.99999,x[84:91]>=-1.99999,x[84:91]<=1.99999,x[91:98]>=armature_min,x[91:98]<=0.99999]
    residual=A@x-b
    if weights is not None: residual=cp.multiply(np.tile(weights,len(b)//7),residual)
    scale=np.r_[np.tile([3,1,1,1,.3,.3,.3,.3,.3,.3],7),np.ones(28)]
    loss=cp.sum_squares(residual) if robust is None else cp.sum(cp.huber(residual,robust))
    objective=loss/(len(b)//7)+ridge*cp.sum_squares(cp.multiply(1/scale,x))
    prob=cp.Problem(cp.Minimize(objective),con)
    prob.solve(solver='CLARABEL',tol_gap_abs=1e-9,tol_feas=1e-9,tol_gap_rel=1e-9,max_iter=200)
    return coefficient_config(x.value),x.value,{'status':prob.status,'objective':prob.value,'rmse':np.sqrt(np.mean((A@x.value-b).reshape(-1,7)**2,axis=0)).tolist()}
