"""Constrained identification using only the supplied measured-data regressor."""
import argparse
import json
from pathlib import Path

import cvxpy as cp
import numpy as np


DATA = np.load('training-regressor.npz')
A, B = DATA['A'], DATA['b']
EPISODES = DATA['sample_episode_ids']
SCALES = np.ones((len(EPISODES), 7))
for episode in np.unique(EPISODES):
    rows = EPISODES == episode
    SCALES[rows] = np.minimum(1., np.maximum(.5, B.reshape(-1, 7)[rows].std(0)))


def to_config(x):
    result = {'mass': [], 'com': [], 'inertia': []}
    for k in range(7):
        p = x[k*10:(k+1)*10]
        mass = float(p[0])
        center = p[1:4] / mass
        origin = np.array([[p[4], p[7], p[8]], [p[7], p[5], p[9]], [p[8], p[9], p[6]]])
        inertia = origin - mass * (center@center*np.eye(3) - np.outer(center, center))
        result['mass'].append(mass)
        result['com'].append(center.tolist())
        result['inertia'].append(inertia.tolist())
    for i, key in enumerate(('viscous', 'coulomb', 'torque_bias', 'armature')):
        result[key] = x[70+i*7:77+i*7].tolist()
    return result


def torque_report(x):
    residual = (A@x-B).reshape(-1, 7)
    out = {}
    for episode in np.unique(EPISODES):
        rows = EPISODES == episode
        out[str(episode)] = {'rmse': np.sqrt(np.mean(residual[rows]**2, axis=0)).tolist(),
                             'scaled_rmse': np.sqrt(np.mean((residual[rows]/SCALES[rows])**2, axis=0)).tolist()}
    return out


def fit(regularization=1e-5, exclude=None, armature_min=0., inertia_floor=1e-6, weights=None,
        robust=0., armature_prior=0., armature_penalty=0., viscous_min=0.):
    x = cp.Variable(98)
    constraints = []
    for k in range(7):
        p = x[k*10:(k+1)*10]
        m, h = p[0], p[1:4]
        inertia = cp.bmat([[p[4],p[7],p[8]], [p[7],p[5],p[9]], [p[8],p[9],p[6]]])
        sigma = .5*cp.trace(inertia)*np.eye(3)-inertia
        pseudo = cp.bmat([[sigma-inertia_floor*np.eye(3), cp.reshape(h,(3,1),order='C')],
                          [cp.reshape(h,(1,3),order='C'), cp.reshape(m,(1,1),order='C')]])
        constraints += [m >= .050001, m <= 9.999999, h >= -.399999*m, h <= .399999*m,
                        pseudo >> 0, cp.trace(sigma) <= .249999*m]
    constraints += [x[70:84] >= 1e-8, x[70:84] <= 4.999999,
                    x[70:77] >= np.maximum(1e-8,viscous_min),
                    x[84:91] >= -1.999999, x[84:91] <= 1.999999,
                    x[91:] >= np.maximum(1e-8, armature_min), x[91:] <= .999999]
    rows = np.ones(len(B), dtype=bool) if exclude is None else np.repeat(EPISODES != exclude, 7)
    scale = SCALES.ravel().copy()
    if weights is not None:
        scale /= np.tile(np.sqrt(weights),len(EPISODES))
    design, target = A[rows]/scale[rows,None], B[rows]/scale[rows]
    residual = design@x-target
    # All scales are generic physical magnitudes, with no nominal robot dynamics.
    penalty_scale = np.tile([3., .3, .3, .3, .1, .1, .1, .1, .1, .1],7)
    cost = (cp.sum(cp.huber(residual, robust)) if robust else cp.sum_squares(residual))/rows.sum()
    cost += regularization*cp.sum_squares(cp.multiply(1/penalty_scale,x[:70]))
    if armature_penalty:
        cost += armature_penalty*cp.sum_squares(x[91:]-armature_prior)
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(solver='CLARABEL',tol_gap_abs=1e-9,tol_feas=1e-9,tol_gap_rel=1e-9,max_iter=300)
    if x.value is None:
        raise RuntimeError(problem.status)
    value = x.value.copy()
    return value, {'status':problem.status,'objective':problem.value,'torque':torque_report(value)}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--regularization',type=float,default=1e-5)
    p.add_argument('--exclude',type=int)
    p.add_argument('--armature-min',type=float,default=0.)
    p.add_argument('--inertia-floor',type=float,default=1e-6)
    p.add_argument('--robust',type=float,default=0.)
    p.add_argument('--armature-prior',type=float,default=0.)
    p.add_argument('--armature-penalty',type=float,default=0.)
    p.add_argument('--viscous-min',type=float,default=0.)
    p.add_argument('--output',default='fit-001')
    args=p.parse_args()
    kwargs=vars(args).copy(); output=Path(kwargs.pop('output'))
    x,report=fit(**kwargs)
    output.mkdir(exist_ok=False)
    np.save(output/'coefficients.npy',x)
    (output/'config.json').write_text(json.dumps(to_config(x),indent=2)+'\n')
    (output/'fit.json').write_text(json.dumps({'arguments':kwargs,**report},indent=2)+'\n')
    print(json.dumps(report,indent=2))
