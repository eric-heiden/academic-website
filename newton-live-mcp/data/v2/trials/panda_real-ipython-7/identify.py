"""Estimation from the supplied immutable regressor, without nominal dynamics."""
import json
import time
from pathlib import Path
import numpy as np
import scipy.linalg as la
import cvxpy as cp

COMPONENTS = ((0,0),(1,1),(2,2),(0,1),(0,2),(1,2))

def decode(x):
    cfg = {k: [] for k in ('mass','com','inertia')}
    for j in range(7):
        p = x[10*j:10*j+10]
        m = float(p[0])
        c = p[1:4]/m
        Io = np.zeros((3,3))
        for v, (a,b) in zip(p[4:], COMPONENTS):
            Io[a,b] = Io[b,a] = v
        Ic = Io-m*(c@c*np.eye(3)-np.outer(c,c))
        cfg['mass'].append(m)
        cfg['com'].append(c.tolist())
        cfg['inertia'].append(Ic.tolist())
    for k,s in zip(('viscous','coulomb','torque_bias','armature'),(70,77,84,91)):
        lo,hi = (-2.,2.) if k == 'torque_bias' else (0.,1. if k=='armature' else 5.)
        cfg[k] = np.clip(x[s:s+7],lo,hi).tolist()
    return cfg

def fit_physical(A,b, *, mask=None, ridge=1e-5, weights=None, armfloor=0., armprior=None, armreg=0., anchor=None, anchorreg=0., robust=None):
    if mask is not None:
        A,b = A[mask],b[mask]
    if weights is None:
        weights = np.ones(len(b))
    elif mask is not None:
        weights = np.asarray(weights)[mask]
    n = len(b)
    x = cp.Variable(98)
    constraints = []
    for j in range(7):
        p=x[j*10:(j+1)*10]
        m,h = p[0],p[1:4]
        Io=cp.bmat([[p[4],p[7],p[8]],[p[7],p[5],p[9]],[p[8],p[9],p[6]]])
        Sigma=0.5*cp.trace(Io)*np.eye(3)-Io
        pseudo=cp.bmat([[Sigma-1e-6*np.eye(3),cp.reshape(h,(3,1),order='C')],[cp.reshape(h,(1,3),order='C'),cp.reshape(m,(1,1),order='C')]])
        constraints += [m>=0.050001,m<=9.999999,h>=-.399999*m,h<=.399999*m,pseudo>>0,cp.trace(Sigma)<=.249999*m]
    constraints += [x[70:84]>=0,x[70:84]<=5,x[84:91]>=-2,x[84:91]<=2,x[91:]>=armfloor,x[91:]<=1]
    scale=np.r_[np.tile([5,.5,.5,.5,.2,.2,.2,.2,.2,.2],7),np.ones(21),np.ones(7)*.2]
    if robust is None:
        Q,R=la.qr(A*weights[:,None]/np.sqrt(n),mode='economic')
        y=Q.T@(b*weights/np.sqrt(n))
        objective=cp.sum_squares(R@x-y)
    else:
        objective=cp.sum(cp.huber(cp.multiply(weights,A@x-b),robust))/n
    objective += ridge*cp.sum_squares(cp.multiply(1/scale,x))
    if armprior is not None:
        objective += armreg*cp.sum_squares(x[91:]-armprior)
    if anchor is not None:
        objective += anchorreg*cp.sum_squares(cp.multiply(1/scale,x-anchor))
    problem=cp.Problem(cp.Minimize(objective),constraints)
    started=time.monotonic()
    problem.solve(solver='CLARABEL',tol_gap_abs=1e-9,tol_feas=1e-9,tol_gap_rel=1e-9,max_iter=150)
    if x.value is None:
        raise RuntimeError(problem.status)
    return decode(x.value),x.value.copy(),{'status':problem.status,'objective':problem.value,'seconds':time.monotonic()-started}

def torque_summary(A,b,x):
    e=(A@x-b).reshape(-1,7)
    return np.sqrt(np.mean(e*e,axis=0)).tolist()
