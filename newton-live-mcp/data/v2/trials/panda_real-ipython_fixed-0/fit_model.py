import json, time
from pathlib import Path
import numpy as np
import cvxpy as cp
from tools.mcp_evaluation.real_robot_model import validate_config, physical_coefficients

WORK = Path('/home/horde/artifacts/newton-live-mcp-v2/confirmation/panda_real-ipython_fixed-0')
REG = np.load(WORK/'training-regressor.npz')
FIT_A, FIT_B = REG['A'], REG['b']
FIT_EP = REG['sample_episode_ids']

def coefficients_to_config(x):
    result = {'mass': [], 'com': [], 'inertia': []}
    for j in range(7):
        v = x[10*j:10*j+10]
        m, h = v[0], v[1:4]
        c = h/m
        Io = np.array([[v[4],v[7],v[8]], [v[7],v[5],v[9]], [v[8],v[9],v[6]]])
        Ic = Io-m*(np.dot(c,c)*np.eye(3)-np.outer(c,c))
        result['mass'].append(float(m))
        result['com'].append(c.tolist())
        result['inertia'].append(Ic.tolist())
    for k, key in enumerate(('viscous','coulomb','torque_bias','armature')):
        result[key] = x[70+7*k:77+7*k].tolist()
    return validate_config(result)

def fit_physical(lam=1e-5, joint_weights=None, selection=None, robust=None, extra=None, min_armature=1e-7, fixed_armature=None):
    mat, target = FIT_A, FIT_B
    if selection is not None:
        rows = np.repeat(selection, 7)
        mat, target = mat[rows], target[rows]
    n = len(target)//7
    jw = np.ones(7) if joint_weights is None else np.asarray(joint_weights)
    weights = np.tile(jw, n)
    if robust is not None:
        weights = weights*robust
    mat, target = mat*weights[:,None], target*weights
    x = cp.Variable(98)
    constraints = []
    for j in range(7):
        v = x[j*10:j*10+10]
        m, h = v[0], v[1:4]
        Io = cp.bmat([[v[4],v[7],v[8]], [v[7],v[5],v[9]], [v[8],v[9],v[6]]])
        sigma = .5*cp.trace(Io)*np.eye(3)-Io
        pseudo = cp.bmat([[sigma-1e-6*np.eye(3),cp.reshape(h,(3,1),order='C')], [cp.reshape(h,(1,3),order='C'), cp.reshape(m,(1,1),order='C')]])
        constraints += [m >= .05001, m <= 9.99999, h <= .39999*m, h >= -.39999*m, pseudo >> 0, cp.trace(sigma) <= .24999*m]
    constraints += [x[70:84] >= 1e-7, x[70:84] <= 4.9999999, x[84:91] >= -1.9999999, x[84:91] <= 1.9999999, x[91:98] >= min_armature, x[91:98] <= .9999999]
    if fixed_armature is not None:
        constraints += [x[91:98] == np.asarray(fixed_armature)]
    scales = np.array([5,1,1,1,.5,.5,.5,.5,.5,.5]*7+[5]*14+[2]*7+[1]*7)
    # Compress the least-squares objective without changing its minimizer.
    Q,R = np.linalg.qr(mat, mode='reduced')
    z = Q.T@target
    objective = cp.sum_squares(R@x-z)/n + lam*cp.sum_squares(cp.multiply(1/scales,x))
    if extra is not None:
        objective += extra(x)
    problem = cp.Problem(cp.Minimize(objective), constraints)
    t0 = time.monotonic()
    problem.solve(solver='CLARABEL', tol_gap_abs=1e-8, tol_feas=1e-9, tol_gap_rel=1e-8, max_iter=200)
    if x.value is None:
        raise RuntimeError(problem.status)
    config = coefficients_to_config(x.value)
    e = (FIT_A@physical_coefficients(config)-FIT_B).reshape(-1,7)
    info = {'status':problem.status, 'seconds':time.monotonic()-t0, 'lambda':lam, 'weights':jw.tolist(), 'rmse':np.sqrt(np.mean(e*e,axis=0)).tolist(), 'per_episode_rmse':{str(ep):np.sqrt(np.mean(e[FIT_EP==ep]**2,axis=0)).tolist() for ep in np.unique(FIT_EP)}}
    return config, info

def measure_candidate(config, label):
    global candidate_results
    if 'candidate_results' not in globals():
        candidate_results = []
    i = len(candidate_results)+1
    path = WORK/f'fit-candidate-{i:03d}-{label}.json'
    if path.exists():
        raise RuntimeError('Refusing to overwrite candidate configuration')
    path.write_text(json.dumps(config,indent=2)+'\n')
    session.scenario.apply_config(config)
    session.dispatch('reset')
    session.dispatch('step', {'count':1800})
    result = session.scenario.metrics()
    candidate_results.append({'label':label,'config':config,'metrics':result})
    with (WORK/'fitting-log.jsonl').open('a') as f:
        f.write(json.dumps(candidate_results[-1])+'\n')
    keys = ('success','frames','trace_path','max_joint_torque_rmse_nm','max_joint_torque_normalized_rmse','max_joint_position_rmse_rad','max_joint_velocity_rmse_rad_s','position_p95_rad','max_joint_speed_rad_s')
    print(label, json.dumps({k:result[k] for k in keys}))
    print('Worst episode metrics:', {k:max(ep[k] for ep in result['per_episode']) for k in ('max_joint_torque_rmse_nm','max_joint_torque_normalized_rmse','max_joint_position_rmse_rad','max_joint_velocity_rmse_rad_s')})
    return result
