import argparse
import json
from pathlib import Path
import numpy as np
import cvxpy as cp

parser = argparse.ArgumentParser()
parser.add_argument('--output', default='fit-001.json')
parser.add_argument('--reg', type=float, default=1e-5)
parser.add_argument('--weights', default='1,1,1,1,1,1,1')
parser.add_argument('--exclude', type=int, default=-1)
parser.add_argument('--armature-min', default='0')
parser.add_argument('--robust', type=float, default=0.)
args = parser.parse_args()
armature_min = np.array([float(x) for x in args.armature_min.split(',')])
if len(armature_min) == 1: armature_min = np.repeat(armature_min,7)
z = np.load('training-regressor.npz')
A, b, episodes = z['A'], z['b'], z['sample_episode_ids']
select = np.repeat(episodes != args.exclude, 7)
weights = np.tile(np.array([float(x) for x in args.weights.split(',')]), len(b)//7)
x = cp.Variable(98)
constraints = []
I3 = np.eye(3)
for j in range(7):
    p = x[j*10:(j+1)*10]
    m, h = p[0], p[1:4]
    Io = cp.bmat([[p[4], p[7], p[8]], [p[7], p[5], p[9]], [p[8], p[9], p[6]]])
    S = .5*cp.trace(Io)*I3-Io
    P = cp.bmat([[S-1e-6*I3, cp.reshape(h,(3,1),order='C')], [cp.reshape(h,(1,3),order='C'), cp.reshape(m,(1,1),order='C')]])
    constraints += [m >= .050001, m <= 9.999999, h >= -.399999*m, h <= .399999*m,
                    P >> 0, cp.trace(S) <= .249999*m]
constraints += [x[70:84] >= 0, x[70:84] <= 5, x[84:91] >= -2, x[84:91] <= 2,
                x[91:] >= armature_min, x[91:] <= 1]
scale = np.r_[np.tile([1, .2, .2, .2, .1, .1, .1, .1, .1, .1], 7), np.ones(28)]
prior = np.zeros(98)
prior[np.arange(7)*10] = 1
for j in range(7): prior[j*10+4:j*10+7] = .01
residual = cp.multiply(weights[select], A[select]@x-b[select])
loss = cp.sum(cp.huber(residual, args.robust)) if args.robust else cp.sum_squares(residual)
objective = loss / select.sum() + args.reg * cp.sum_squares(cp.multiply(1/scale,x-prior))
problem = cp.Problem(cp.Minimize(objective), constraints)
problem.solve(solver='CLARABEL', tol_gap_abs=1e-9, tol_feas=1e-10, tol_gap_rel=1e-9, max_iter=300)
print('status', problem.status, 'objective', problem.value, flush=True)
v = x.value
if v is None: raise RuntimeError(problem.status)
out = {k: [] for k in ('mass','com','inertia','viscous','coulomb','torque_bias','armature')}
for j in range(7):
    p = v[j*10:(j+1)*10]
    m, c = p[0], p[1:4]/p[0]
    Io = np.array([[p[4],p[7],p[8]],[p[7],p[5],p[9]],[p[8],p[9],p[6]]])
    Ic = Io-m*(c@c*I3-np.outer(c,c))
    out['mass'].append(float(m)); out['com'].append(c.tolist()); out['inertia'].append(Ic.tolist())
for k,start,lo,hi in [('viscous',70,0,5),('coulomb',77,0,5),('torque_bias',84,-2,2),('armature',91,0,1)]:
    out[k] = np.clip(v[start:start+7],lo,hi).tolist()
Path(args.output).write_text(json.dumps(out,indent=2)+'\n')
error = (A@v-b).reshape(-1,7)
report = {'args':vars(args),'status':problem.status,'objective':problem.value,'rmse':np.sqrt(np.mean(error**2,axis=0)).tolist(),
          'per_episode': {str(e): np.sqrt(np.mean(error[episodes==e]**2,axis=0)).tolist() for e in np.unique(episodes)}}
Path(args.output).with_suffix('.fit-log.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
print('mass',out['mass']);print('joint parameters', {k:out[k] for k in ('viscous','coulomb','torque_bias','armature')})
