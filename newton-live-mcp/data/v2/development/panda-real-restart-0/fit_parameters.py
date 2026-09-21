"""Estimate full physical dynamics using only the supplied immutable regressor."""
import argparse
import json
from pathlib import Path
import numpy as np
import cvxpy as cp


def coeff_to_config(x):
    out = {k: [] for k in ('mass', 'com', 'inertia')}
    for i in range(7):
        m = x[10*i]
        c = x[10*i+1:10*i+4]/m
        v = x[10*i+4:10*i+10]
        io = np.array([[v[0],v[3],v[4]],[v[3],v[1],v[5]],[v[4],v[5],v[2]]])
        ic = io - m*(c@c*np.eye(3)-np.outer(c,c))
        out['mass'].append(float(m));out['com'].append(c.tolist());out['inertia'].append(ic.tolist())
    for k, start, bounds in [('viscous',70,(0,5)),('coulomb',77,(0,5)),('torque_bias',84,(-2,2)),('armature',91,(0,1))]:
        out[k] = np.clip(x[start:start+7], *bounds).tolist()
    return out


def fit(A, b, reg=1e-5, weight=0., min_arm=0., huber=0., prior=None):
    x = cp.Variable(98)
    constraints=[]
    prior_x = np.zeros(98)
    scales = np.ones(98)
    for i in range(7):
        z=x[10*i:10*i+10]
        m=z[0];h=z[1:4]
        io=cp.bmat([[z[4],z[7],z[8]],[z[7],z[5],z[9]],[z[8],z[9],z[6]]])
        s=cp.trace(io)*.5*np.eye(3)-io
        pseudo=cp.bmat([[s,cp.reshape(h,(3,1),order='C')],[cp.reshape(h,(1,3),order='C'),cp.reshape(m,(1,1),order='C')]])
        constraints += [m>=.050001,m<=9.999999,h>=-.399999*m,h<=.399999*m,
                        pseudo-np.diag([1e-6,1e-6,1e-6,0]) >> 0,cp.trace(s)<=.249999*m]
        prior_x[10*i]=1.
        prior_x[10*i+4:10*i+7]=.01
        scales[10*i]=1.
        scales[10*i+1:10*i+4]=5.
        scales[10*i+4:10*i+10]=10.
    constraints += [x[70:84]>=0,x[70:84]<=5,x[84:91]>=-2,x[84:91]<=2,x[91:98]>=min_arm,x[91:98]<=1]
    if prior is not None:
        prior_x=prior
    std=np.maximum(b.reshape(-1,7).std(0),.5)
    weights=np.tile(std**(-weight),len(b)//7)
    residual=cp.multiply(weights,A@x-b)
    loss=cp.sum(cp.huber(residual,huber)) if huber else cp.sum_squares(residual)
    objective=loss/len(b)+reg*cp.sum_squares(cp.multiply(scales,x-prior_x))
    prob=cp.Problem(cp.Minimize(objective),constraints)
    prob.solve(solver='CLARABEL',tol_gap_abs=1e-9,tol_feas=1e-9,tol_gap_rel=1e-9,max_iter=300)
    if x.value is None:raise RuntimeError(prob.status)
    return x.value, {'status':prob.status,'objective':prob.value}


def score(A,b,x,episodes):
    e=(A@x-b).reshape(-1,7)
    out={'rmse':np.sqrt(np.mean(e*e,0)).tolist(),'per_episode':{}}
    for ep in np.unique(episodes):
        mask=episodes==ep
        rmse=np.sqrt(np.mean(e[mask]**2,0))
        out['per_episode'][str(ep)]={'rmse':rmse.tolist(),'normalized':(rmse/np.maximum(b.reshape(-1,7)[mask].std(0),.5)).tolist()}
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reg',type=float,default=1e-5);p.add_argument('--weight',type=float,default=0.)
    p.add_argument('--min-arm',type=float,default=0.);p.add_argument('--huber',type=float,default=0.)
    p.add_argument('--out',default='fit-001');p.add_argument('--cv',action='store_true');a=p.parse_args()
    r=np.load('training-regressor.npz');A,b,ep=r['A'],r['b'],r['sample_episode_ids']
    x,info=fit(A,b,a.reg,a.weight,a.min_arm,a.huber)
    c=coeff_to_config(x)
    report=vars(a)|info|score(A,b,x,ep)
    if a.cv:
        report['cross_validation']={}
        for held in np.unique(ep):
            train=np.repeat(ep!=held,7);test=~train
            xx,_=fit(A[train],b[train],a.reg,a.weight,a.min_arm,a.huber)
            report['cross_validation'][str(held)]=score(A[test],b[test],xx,ep[ep==held])
    dest=Path(a.out);dest.mkdir(exist_ok=False)
    (dest/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    (dest/'fit.json').write_text(json.dumps(report,indent=2)+'\n')
    np.save(dest/'coefficients.npy',x)
    print(json.dumps(report,indent=2));print('parameters',json.dumps(c))
