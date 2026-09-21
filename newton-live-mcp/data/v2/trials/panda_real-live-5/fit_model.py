"""Physical identification using only the supplied immutable design matrix."""
import json
import argparse
from pathlib import Path
import numpy as np
import cvxpy as cp

ROOT = Path(__file__).resolve().parent
reg = np.load(ROOT / 'training-regressor.npz')
A, b = reg['A'], reg['b']
episodes = reg['sample_episode_ids']

def decode(x):
    out = {'mass': [], 'com': [], 'inertia': []}
    for i in range(7):
        z = x[10*i:10*i+10]
        m, h = float(z[0]), z[1:4]
        c = h/m
        I = np.array([[z[4],z[7],z[8]],[z[7],z[5],z[9]],[z[8],z[9],z[6]]])
        Ic = I - m*(np.dot(c,c)*np.eye(3)-np.outer(c,c))
        out['mass'].append(m)
        out['com'].append(c.tolist())
        out['inertia'].append(Ic.tolist())
    for j,k in enumerate(('viscous','coulomb','torque_bias','armature')):
        lo,hi = (-2.,2.) if k == 'torque_bias' else ((0.,1.) if k == 'armature' else (0.,5.))
        out[k] = np.clip(x[70+7*j:77+7*j],lo,hi).tolist()
    return out

def fit(ridge=1e-4, weight=0., exclude=None, margin=1e-5, robust=False, armfloor=0.):
    x = cp.Variable(98)
    constraints = []
    for i in range(7):
        z = x[10*i:10*i+10]
        m,h = z[0],z[1:4]
        I = cp.bmat([[z[4],z[7],z[8]],[z[7],z[5],z[9]],[z[8],z[9],z[6]]])
        S = cp.trace(I)*.5*np.eye(3)-I
        P = cp.bmat([[S-cp.Constant(margin*np.eye(3)),cp.reshape(h,(3,1),order='C')],[cp.reshape(h,(1,3),order='C'),cp.reshape(m,(1,1),order='C')]])
        constraints += [m>=.05, m<=10, h>=-.4*m,h<=.4*m, P >> 0,cp.trace(S)<=.25*m]
    constraints += [x[70:84]>=0, x[70:84]<=5, x[84:91]>=-2, x[84:91]<=2, x[91:98]>=armfloor, x[91:98]<=1]
    mask = np.ones(len(b),dtype=bool) if exclude is None else np.repeat(episodes != exclude,7)
    scales=np.maximum(b.reshape(3,-1,7).std(axis=1),.5)
    weights = np.repeat(scales[:,None,:],150,axis=1).ravel()**(-weight)
    residual = cp.multiply(weights[mask],A[mask]@x-b[mask])
    # Weak dimension-aware generic prior resolves unobservable parameter transfers.
    prior = np.zeros(98)
    scale = np.ones(98)
    for i in range(7):
        prior[10*i]=1.
        prior[10*i+4:10*i+7]=.01
        scale[10*i]=1.
        scale[10*i+1:10*i+4]=.2
        scale[10*i+4:10*i+10]=.1
    scale[91:]=.1
    loss=cp.sum(cp.huber(residual,.2)) if robust else cp.sum_squares(residual)
    problem = cp.Problem(cp.Minimize(loss/np.sum(mask)+ridge*cp.sum_squares(cp.multiply(1/scale,x-prior))), constraints)
    problem.solve(solver='CLARABEL', max_iter=300, tol_gap_abs=1e-9, tol_feas=1e-9, tol_gap_rel=1e-9)
    if x.value is None: raise RuntimeError(problem.status)
    cfg=decode(x.value)
    err=(A@x.value-b).reshape(3,-1,7)
    report={'status':problem.status,'objective':problem.value,'ridge':ridge,'weight':weight,'exclude':exclude,'margin':margin,'robust':robust,'armfloor':armfloor,'rmse':np.sqrt(np.mean(err**2,axis=(0,1))).tolist(),'per_episode_rmse':np.sqrt(np.mean(err**2,axis=1)).tolist(),'mass':cfg['mass'],'armature':cfg['armature']}
    return cfg,report,x.value

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--name',default='fit001')
    p.add_argument('--ridge',type=float,default=1e-4)
    p.add_argument('--weight',type=float,default=0.)
    p.add_argument('--margin',type=float,default=1e-5)
    p.add_argument('--armfloor',type=float,default=0.)
    p.add_argument('--exclude',type=int)
    p.add_argument('--robust',action='store_true')
    args=p.parse_args()
    name=args.name
    cfg,report,x=fit(**{k:v for k,v in vars(args).items() if k!='name'})
    (ROOT / f'{name}.json').write_text(json.dumps(cfg,indent=2)+'\n')
    (ROOT / f'{name}.fit.json').write_text(json.dumps(report,indent=2)+'\n')
    np.save(ROOT / f'{name}.coeff.npy',x)
    print(json.dumps(report))
