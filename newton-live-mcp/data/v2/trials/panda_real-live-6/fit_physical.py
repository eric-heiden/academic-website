import json, time
from pathlib import Path
import numpy as np
import cvxpy as cp
BASE=Path(__file__).resolve().parent
Z=np.load(BASE/'training-regressor.npz'); A=Z['A']; b=Z['b']; episodes=Z['sample_episode_ids']; sample_indices=Z['sample_indices']

def to_config(x):
    masses=[]; centers=[]; inertias=[]
    for i in range(7):
        z=x[i*10:i*10+10]; m=z[0]; c=z[1:4]/m
        Io=np.array([[z[4],z[7],z[8]],[z[7],z[5],z[9]],[z[8],z[9],z[6]]])
        Ic=Io-m*(np.dot(c,c)*np.eye(3)-np.outer(c,c))
        masses.append(float(m)); centers.append(c.tolist()); inertias.append(Ic.tolist())
    config=dict(mass=masses,com=centers,inertia=inertias)
    for k,j,lo,hi in [('viscous',70,0,5),('coulomb',77,0,5),('torque_bias',84,-2,2),('armature',91,0,1)]:
        config[k]=np.clip(x[j:j+7],lo,hi).tolist()
    return config

def fit(lam=1e-5, mask=None, weights=None, arm_min=0, robust=False, prior=None, fixed=None):
    x=cp.Variable(98)
    cons=[]
    for i in range(7):
        z=x[10*i:10*i+10]; m=z[0]; h=z[1:4]
        I=cp.bmat([[z[4],z[7],z[8]],[z[7],z[5],z[9]],[z[8],z[9],z[6]]])
        S=cp.trace(I)/2*np.eye(3)-I
        P=cp.bmat([[S,cp.reshape(h,(3,1),order='C')],[cp.reshape(h,(1,3),order='C'),cp.reshape(m,(1,1),order='C')]])
        cons += [m>=0.050001,m<=9.999999,h>=-0.399999*m,h<=0.399999*m,P-np.diag([1e-6,1e-6,1e-6,0])>>0,cp.trace(S)<=0.249999*m]
    cons += [x[70:84]>=0,x[70:84]<=5,x[84:91]>=-2,x[84:91]<=2,x[91:]>=arm_min,x[91:]<=1]
    if fixed is not None:
        for index,value in fixed.items(): cons.append(x[index]==value)
    if mask is None: mask=np.ones(len(b),dtype=bool)
    if weights is None: weights=np.tile([1,1,1,1,1,1,2],len(b)//7)
    residual=cp.multiply(weights[mask],A[mask]@x-b[mask])
    scale=np.r_[np.tile([1/3,1/.3,1/.3,1/.3,1/.1,1/.1,1/.1,1/.1,1/.1,1/.1],7),np.ones(21),np.ones(7)*10]
    if prior is None: prior=np.zeros(98)
    error=cp.sum(cp.huber(residual,.25)) if robust else cp.sum_squares(residual)
    objective=error/(mask.sum()/7)+lam*cp.sum_squares(cp.multiply(scale,x-prior))
    problem=cp.Problem(cp.Minimize(objective),cons)
    problem.solve(solver='CLARABEL',tol_gap_abs=1e-9,tol_feas=1e-9,tol_gap_rel=1e-9,max_iter=200)
    if x.value is None: raise RuntimeError(problem.status)
    p=x.value; residual=(A@p-b).reshape(-1,7)
    print(json.dumps(dict(lam=lam,status=problem.status,objective=problem.value,rmse=np.sqrt(np.mean(residual**2,axis=0)).tolist(),episode_rmse=[np.sqrt(np.mean(residual[episodes==e]**2,axis=0)).tolist() for e in np.unique(episodes)],joint=p[70:].reshape(4,7).tolist())),flush=True)
    return p,to_config(p)

if __name__=='__main__':
    x,c=fit()
    (BASE/'fit-001.json').write_text(json.dumps(c,indent=2)+'\n')
    np.save(BASE/'fit-001-coeff.npy',x)
