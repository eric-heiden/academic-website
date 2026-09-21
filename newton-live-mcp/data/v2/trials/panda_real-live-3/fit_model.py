"""Physical semidefinite identification from the supplied measured-data regressor."""
import argparse
import json
from pathlib import Path
import numpy as np
import cvxpy as cp

PAIRS = [(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]

def decode(x):
    out = {k: [] for k in ['mass','com','inertia']}
    for j in range(7):
        v = x[j*10:(j+1)*10]
        m, c = v[0], v[1:4]/v[0]
        I = np.zeros((3,3))
        for a,(r,s) in enumerate(PAIRS):
            I[r,s] = I[s,r] = v[4+a]
        Ic = I - m*(c@c*np.eye(3)-np.outer(c,c))
        out['mass'].append(float(m))
        out['com'].append(c.tolist())
        out['inertia'].append(Ic.tolist())
    for j,k in enumerate(['viscous','coulomb','torque_bias','armature']):
        out[k] = x[70+j*7:77+j*7].tolist()
    return out

def fit(A,b,reg=1e-5,weights=None,train=None,arm_floor=0.0):
    n = len(b)//7
    if train is None: train=np.ones(n,dtype=bool)
    if weights is None: weights=np.ones((n,7))
    weights=np.broadcast_to(weights,(n,7)).reshape(-1)
    mask=np.repeat(train,7)
    x = cp.Variable(98)
    cons=[]
    for j in range(7):
        v=x[j*10:(j+1)*10]; m=v[0]; h=v[1:4]
        I=cp.bmat([[v[4],v[7],v[8]],[v[7],v[5],v[9]],[v[8],v[9],v[6]]])
        S=cp.trace(I)*.5*np.eye(3)-I
        P=cp.bmat([[S-1e-6*np.eye(3),cp.reshape(h,(3,1),order='C')], [cp.reshape(h,(1,3),order='C'),cp.reshape(m,(1,1),order='C')]])
        cons += [m>=.05001,m<=9.99999,h<=.39999*m,h>=-.39999*m,P>>0,cp.trace(S)<=.24999*m]
    cons += [x[70:84]>=1e-8,x[70:84]<=4.999999,x[84:91]>=-1.999999,x[84:91]<=1.999999,x[91:98]>=max(1e-8,arm_floor),x[91:98]<=.999999]
    # Generic scales, independent of any nominal Panda dynamics.
    scale=np.r_[np.tile([3,.5,.5,.5,.15,.15,.15,.1,.1,.1],7),np.ones(21),np.ones(7)*.3]
    prior=np.zeros(98); prior[np.arange(7)*10]=1
    prior[np.concatenate([np.arange(7)*10+i for i in [4,5,6]])]=.01
    residual=cp.multiply(weights[mask],A[mask]@x-b[mask])
    obj=cp.sum_squares(residual)/int(train.sum())+reg*cp.sum_squares(cp.multiply(1/scale,x-prior))
    problem=cp.Problem(cp.Minimize(obj),cons)
    problem.solve(solver='CLARABEL',tol_gap_abs=1e-9,tol_feas=1e-9,tol_gap_rel=1e-9,max_iter=300)
    if x.value is None: raise RuntimeError(problem.status)
    return x.value, {'status':problem.status,'objective':problem.value,'iterations':problem.solver_stats.num_iters}

def diagnostics(x,A,b,episodes):
    err=(A@x-b).reshape(-1,7)
    result={}
    for e in np.unique(episodes):
        mask=episodes==e
        rmse=np.sqrt(np.mean(err[mask]**2,axis=0))
        norm=rmse/np.maximum(b.reshape(-1,7)[mask].std(axis=0),.5)
        result[str(e)]={'rmse':rmse.tolist(),'nrmse':norm.tolist()}
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--reg',type=float,default=1e-5)
    parser.add_argument('--output',default='fit_001.json')
    parser.add_argument('--crossval',action='store_true')
    parser.add_argument('--arm-floor',type=float,default=0.0)
    args=parser.parse_args()
    data=np.load('training-regressor.npz'); A=data['A']; b=data['b']; episodes=data['sample_episode_ids']
    x,info=fit(A,b,args.reg,arm_floor=args.arm_floor)
    Path(args.output).write_text(json.dumps(decode(x),indent=2)+'\n')
    np.save(Path(args.output).with_suffix('.coefficients.npy'),x)
    print(json.dumps({'fit':info,'torque':diagnostics(x,A,b,episodes),'joint':x[70:].tolist(),'mass':x[:70:10].tolist()},indent=2),flush=True)
    if args.crossval:
        for ep in np.unique(episodes):
            xc,ic=fit(A,b,args.reg,train=episodes!=ep,arm_floor=args.arm_floor)
            print(json.dumps({'excluded':int(ep),'torque':diagnostics(xc,A,b,episodes)}),flush=True)
