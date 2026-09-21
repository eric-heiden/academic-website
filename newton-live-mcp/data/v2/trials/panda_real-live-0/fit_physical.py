import json
from pathlib import Path
import numpy as np
import cvxpy as cp

ROOT = Path(__file__).resolve().parent
data = np.load(ROOT / 'training-regressor.npz')
A, b = data['A'], data['b']
episodes = data['sample_episode_ids']

def unpack(x):
    cfg = {k: [] for k in ('mass', 'com', 'inertia')}
    for i in range(7):
        z=x[10*i:10*i+10]
        m=z[0]; c=z[1:4]/m
        I=np.array([[z[4],z[7],z[8]],[z[7],z[5],z[9]],[z[8],z[9],z[6]]])
        Ic=I-m*(c@c*np.eye(3)-np.outer(c,c))
        cfg['mass'].append(float(m));cfg['com'].append(c.tolist());cfg['inertia'].append(Ic.tolist())
    for k,s in [('viscous',70),('coulomb',77),('torque_bias',84),('armature',91)]:
        cfg[k]=x[s:s+7].tolist()
    return cfg

def fit(lam=1e-5, mask=None, weights=None, arm_min=0.0, extra=None, visc_min=0.):
    x=cp.Variable(98)
    con=[]
    for i in range(7):
        z=x[10*i:10*i+10];m=z[0];h=z[1:4]
        I=cp.bmat([[z[4],z[7],z[8]],[z[7],z[5],z[9]],[z[8],z[9],z[6]]])
        S=.5*cp.trace(I)*np.eye(3)-I
        J=cp.bmat([[S,cp.reshape(h,(3,1),order='C')],[cp.reshape(h,(1,3),order='C'),cp.reshape(m,(1,1),order='C')]])
        con += [m>=.050001,m<=9.999999,h>=-.399999*m,h<=.399999*m,J>>1e-6*np.eye(4),cp.trace(S)<=.249999*m]
    con += [x[70:84]>=1e-8,x[70:84]<=4.999999,x[84:91]>=-1.999999,x[84:91]<=1.999999,x[91:]>=arm_min+1e-8,x[91:]<=.999999]
    con += [x[70:77]>=np.asarray(visc_min)+1e-8]
    use=np.ones(len(b),dtype=bool) if mask is None else np.repeat(mask,7)
    w=np.ones(len(b)) if weights is None else np.broadcast_to(weights,(len(b)//7,7)).ravel()
    residual=cp.multiply(w[use], A[use]@x-b[use])
    scale=np.tile([3.,.5,.5,.5,.2,.2,.2,.1,.1,.1],7).tolist()+[1.]*21+[.1]*7
    objective=cp.sum_squares(residual)/use.sum()+lam*cp.sum_squares(cp.multiply(1/np.array(scale),x))
    if extra is not None:
        objective += extra(x)
    prob=cp.Problem(cp.Minimize(objective),con)
    prob.solve(solver='CLARABEL',tol_gap_abs=1e-9,tol_feas=1e-9,tol_gap_rel=1e-9,max_iter=300)
    if x.value is None: raise RuntimeError(prob.status)
    xx=x.value
    err=(A@xx-b).reshape(-1,7)
    report={'lambda':lam,'status':prob.status,'objective':prob.value,'rmse':np.sqrt(np.mean(err**2,axis=0)).tolist(),
            'per_episode_rmse':{str(ep):np.sqrt(np.mean(err[episodes==ep]**2,axis=0)).tolist() for ep in np.unique(episodes)}}
    return unpack(xx),xx,report

if __name__=='__main__':
    reports=[]
    for lam in [0,1e-6,1e-5,1e-4,1e-3]:
        cfg,x,report=fit(lam)
        name=f'fit-lambda-{lam:g}'
        (ROOT / (name+'.json')).write_text(json.dumps(cfg,indent=2)+'\n')
        np.save(ROOT/(name+'.npy'),x)
        reports.append(report)
        print(json.dumps(report),flush=True)
    (ROOT/'fit-reports.json').write_text(json.dumps(reports,indent=2)+'\n')
