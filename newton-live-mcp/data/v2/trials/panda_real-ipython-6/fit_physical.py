"""Constrained estimation using only the supplied measured-data regressor."""
import json
from pathlib import Path
import numpy as np
import cvxpy as cp

ROOT = Path('/home/horde/artifacts/newton-live-mcp-v2/confirmation/panda_real-ipython-6')

def coefficients_to_config(x):
    cfg = {k: [] for k in ('mass', 'com', 'inertia')}
    for i in range(7):
        v = x[10*i:10*i+10]
        m, h = float(v[0]), v[1:4]
        c = h/m
        io = np.array([[v[4],v[7],v[8]], [v[7],v[5],v[9]], [v[8],v[9],v[6]]])
        ic = io-m*(c@c*np.eye(3)-np.outer(c,c))
        cfg['mass'].append(m)
        cfg['com'].append(c.tolist())
        cfg['inertia'].append(ic.tolist())
    for k, start, lo, hi in [('viscous',70,0,5),('coulomb',77,0,5),('torque_bias',84,-2,2),('armature',91,0,1)]:
        cfg[k] = np.clip(x[start:start+7],lo,hi).tolist()
    return cfg

def fit_physical(A, b, episode_ids, ridge=1e-5, loss='square', mask=None, weights=None, min_moment=1e-5, prior=None, armature_min=0):
    x = cp.Variable(98)
    cons = []
    for i in range(7):
        v = x[10*i:10*i+10]
        m, h = v[0], v[1:4]
        io = cp.bmat([[v[4],v[7],v[8]], [v[7],v[5],v[9]], [v[8],v[9],v[6]]])
        s = 0.5*cp.trace(io)*np.eye(3)-io
        j = cp.bmat([[s, cp.reshape(h,(3,1),order='C')], [cp.reshape(h,(1,3),order='C'),cp.reshape(m,(1,1),order='C')]])
        cons += [m >= .050001, m <= 9.999999, h >= -.399999*m, h <= .399999*m,
                 j >> min_moment*np.eye(4), cp.trace(s) <= .249999*m]
    cons += [x[70:84]>=0, x[70:84]<=5, x[84:91]>=-2, x[84:91]<=2, x[91:98]>=armature_min, x[91:98]<=1]
    if weights is None:
        scales = np.empty((len(b)//7,7))
        for ep in np.unique(episode_ids):
            sel = episode_ids==ep
            scales[sel] = np.minimum(np.maximum(b.reshape(-1,7)[sel].std(axis=0),.5),1.)
        weights = 1/scales.ravel()
    if mask is None:
        mask = np.ones(len(b), dtype=bool)
    residual = cp.multiply(weights[mask], A[mask]@x-b[mask])
    regularizer_scale = np.r_[np.tile([.2,2,2,2,10,10,10,10,10,10],7), np.ones(21)*.5,np.ones(7)*3]
    center = np.zeros(98) if prior is None else prior
    regularizer = ridge*cp.sum_squares(cp.multiply(regularizer_scale,x-center))
    error = cp.sum_squares(residual) if loss=='square' else cp.sum(cp.huber(residual,float(loss)))
    problem = cp.Problem(cp.Minimize(error/np.sum(mask)+regularizer),cons)
    problem.solve(solver='CLARABEL',max_iter=200,tol_gap_abs=1e-9,tol_feas=1e-9,tol_gap_rel=1e-9)
    if x.value is None:
        raise RuntimeError(problem.status)
    return coefficients_to_config(x.value), x.value, {'status':problem.status,'objective':problem.value,'solver_time':problem.solver_stats.solve_time}

def torque_summary(A,b,x,episodes):
    err = (A@x-b).reshape(-1,7)
    result = {'pooled':np.sqrt(np.mean(err**2,axis=0)).tolist()}
    for ep in np.unique(episodes):
        sel = episodes==ep
        result[str(ep)] = np.sqrt(np.mean(err[sel]**2,axis=0)).tolist()
    return result
