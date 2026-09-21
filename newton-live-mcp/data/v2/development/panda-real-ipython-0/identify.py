"""Constrained estimation using only the supplied measured-data design matrix."""
import json
import time
import numpy as np
import cvxpy as cp
from tools.mcp_evaluation.real_robot_model import validate_config, physical_coefficients


def coefficients_to_config(x):
    config = {k: [] for k in ('mass', 'com', 'inertia')}
    for j in range(7):
        p = x[j*10:(j+1)*10]
        m, h = float(p[0]), p[1:4]
        c = h/m
        Io = np.array([[p[4], p[7], p[8]], [p[7], p[5], p[9]], [p[8], p[9], p[6]]])
        Ic = Io - m*(np.dot(c,c)*np.eye(3)-np.outer(c,c))
        config['mass'].append(m)
        config['com'].append(c.tolist())
        config['inertia'].append(Ic.tolist())
    for i, key in enumerate(('viscous', 'coulomb', 'torque_bias', 'armature')):
        bounds = (-2,2) if key =[redacted] 'torque_bias' else (0,1 if key =[redacted] 'armature' else 5)
        config[key] = np.clip(x[70+i*7:77+i*7], *bounds).tolist()
    return validate_config(config)


def fit_physical(A, b, weights=None, ridge=1e-5, min_arm=0., robust=None, prior=None, prior_weight=0., min_visc=0.):
    x = cp.Variable(98)
    constraints = []
    for j in range(7):
        p = x[j*10:(j+1)*10]
        m = p[0]
        h = cp.reshape(p[1:4], (3,1), order='C')
        Io = cp.bmat([[p[4],p[7],p[8]], [p[7],p[5],p[9]], [p[8],p[9],p[6]]])
        S = .5*cp.trace(Io)*np.eye(3)-Io
        pseudo = cp.bmat([[S-1e-6*np.eye(3), h],[h.T,cp.reshape(m,(1,1),order='C')]])
        constraints.extend([m>=.05001, m<=9.99999, p[1:4]>=-.39999*m, p[1:4]<=.39999*m,
                            cp.trace(S)<=.24999*m, pseudo>>0])
    constraints.extend([x[70:84]>=0, x[70:77]>=min_visc, x[70:84]<=5, x[84:91]>=-2, x[84:91]<=2,
                        x[91:98]>=min_arm, x[91:98]<=1])
    if weights is None:
        weights = np.ones(len(b))
    residual = cp.multiply(weights, A@x-b)
    loss = cp.sum_squares(residual) if robust is None else cp.sum(cp.huber(residual,robust))
    scaling = np.r_[np.tile([1,3,3,3,10,10,10,10,10,10],7), np.ones(28)]
    penalty = ridge*cp.sum_squares(cp.multiply(scaling,x))
    if prior is not None and prior_weight:
        penalty += prior_weight*cp.sum_squares(A@(x-prior))/len(b)
    prob = cp.Problem(cp.Minimize(loss/(len(b)/7)+penalty),constraints)
    prob.solve(solver='CLARABEL', max_iter=200, tol_gap_abs=1e-9, tol_feas=1e-9)
    if x.value is None:
        raise RuntimeError(prob.status)
    config = coefficients_to_config(x.value)
    actual = physical_coefficients(config)
    return config, actual, {'status':prob.status, 'objective':prob.value,
        'torque_rmse':np.sqrt(np.mean((A@actual-b).reshape(-1,7)**2,axis=0)).tolist()}


def summary_metrics(m):
    keys = ('success','max_joint_torque_rmse_nm','max_joint_torque_normalized_rmse',
            'max_joint_position_rmse_rad','max_joint_velocity_rmse_rad_s','position_p95_rad','max_joint_speed_rad_s')
    return {**{k:m[k] for k in keys},'per_episode':[{k:p[k] for k in ('episode',)+keys} for p in m['per_episode']],
            'trace_path':m['trace_path']}


def evaluate_candidate(config, label):
    # This is the prescribed physical workflow; all built-in logging stays intact.
    candidate_number = len(candidate_history)+1
    if candidate_number > 60:
        raise RuntimeError('Physical candidate budget exhausted')
    path = workdir/f'fit-{candidate_number:03d}-{label}.json'
    path.write_text(json.dumps(config,indent=2)+'\n')
    session.scenario.apply_config(config)
    session.dispatch('reset')
    session.dispatch('step', {'count':1800})
    result = session.scenario.metrics()
    candidate_history.append({'label':label,'config':config,'metrics':result})
    (workdir/f'fit-{candidate_number:03d}-{label}-metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Candidate',candidate_number,label,summary_metrics(result),flush=True)
    return result
