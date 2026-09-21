"""Fit physical dynamics from immutable measured Newton regressors only."""
import json, time
from pathlib import Path
import numpy as np
import scipy.linalg as la
import cvxpy as cp

root=Path(__file__).resolve().parent
reg=np.load(root/'training-regressor.npz')
A=reg['A']; b=reg['b']; epi=reg['sample_episode_ids']
U,s,Vt=la.svd(A,full_matrices=False)
Aclean=(U[:,s>0.01]*s[s>0.01])@Vt[s>0.01]


def coefficient_config(x):
    cfg={k:[] for k in ['mass','com','inertia','viscous','coulomb','torque_bias','armature']}
    for i in range(7):
        z=x[10*i:10*i+10]; m=z[0]; c=z[1:4]/m
        Io=np.array([[z[4],z[7],z[8]],[z[7],z[5],z[9]],[z[8],z[9],z[6]]])
        Ic=Io-m*((c@c)*np.eye(3)-np.outer(c,c))
        cfg['mass'].append(float(m));cfg['com'].append(c.tolist());cfg['inertia'].append(Ic.tolist())
    for i,k in enumerate(['viscous','coulomb','torque_bias','armature']):
        v=x[70+i*7:77+i*7].copy()
        if k!='torque_bias':v=np.maximum(v,0)
        cfg[k]=v.tolist()
    return cfg


def fit_physical(weights=None, ridge=1e-5, selection=None, robust=None, min_armature=0, inertia_eps=1e-6):
    x=cp.Variable(98)
    cons=[]
    for i in range(7):
        z=x[10*i:10*i+10];m=z[0];h=z[1:4]
        Io=cp.bmat([[z[4],z[7],z[8]],[z[7],z[5],z[9]],[z[8],z[9],z[6]]])
        S=.5*cp.trace(Io)*np.eye(3)-Io
        P=cp.bmat([[S,cp.reshape(h,(3,1),order='C')],[cp.reshape(h,(1,3),order='C'),cp.reshape(m,(1,1),order='C')]])
        cons += [m>=.05001,m<=9.99999,h<=.39999*m,h>=-.39999*m,cp.trace(S)<=.24999*m,P>>inertia_eps*np.eye(4)]
    cons += [x[70:84]>=0,x[70:84]<=5,x[84:91]>=-2,x[84:91]<=2,x[91:98]>=min_armature,x[91:98]<=1]
    scale=np.tile(np.array([3,.3,.3,.3,.1,.1,.1,.1,.1,.1]),7)
    scale=np.r_[scale,np.ones(21),np.ones(7)*.2]
    if selection is None:selection=np.ones(len(b),dtype=bool)
    if weights is None:weights=np.ones(len(b))
    Aw=Aclean[selection]*np.asarray(weights)[selection,None];bw=b[selection]*np.asarray(weights)[selection]
    if robust is None:
        Q,R=la.qr(Aw,mode='economic');y=Q.T@bw
        loss=cp.sum_squares(R@x-y)/sum(selection)
    else:
        loss=cp.sum(cp.huber(Aw@x-bw,robust))/sum(selection)
    prob=cp.Problem(cp.Minimize(loss+ridge*cp.sum_squares(cp.multiply(1/scale,x))),cons)
    t=time.perf_counter()
    prob.solve(solver='CLARABEL',tol_gap_abs=1e-9,tol_feas=1e-9,tol_gap_rel=1e-9,max_iter=250)
    result={'status':prob.status,'objective':prob.value,'seconds':time.perf_counter()-t,'x':x.value,'config':coefficient_config(x.value)}
    return result


def torque_summary(x):
    err=(A@x-b).reshape(-1,7)
    return {str(e):np.sqrt(np.mean(err[epi==e]**2,axis=0)).tolist() for e in np.unique(epi)}

if __name__ == '__main__':
    # Reproduce the chosen estimator offline. This does not apply a candidate.
    chosen=fit_physical(
        weights=np.tile([1,1,1,1,3,2,3],len(b)//7),
        ridge=1e-5,
        min_armature=np.array([.03,.03,.03,.03,.03,.03,.02]),
        inertia_eps=1e-5,
    )
    print(json.dumps(chosen['config'],indent=2))
