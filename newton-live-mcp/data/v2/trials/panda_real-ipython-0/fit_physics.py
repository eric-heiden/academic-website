import numpy as np
import cvxpy as cp

def coefficients_config(x):
    cfg = {key: [redacted] for key in ('mass','com','inertia','viscous','coulomb','torque_bias','armature')}
    for i in range(7):
        v = x[i*10:(i+1)*10]
        m, h = float(v[0]), v[1:4]
        c = h/m
        I0 = np.array([[v[4],v[7],v[8]],[v[7],v[5],v[9]],[v[8],v[9],v[6]]])
        Ic = I0-m*(np.dot(c,c)*np.eye(3)-np.outer(c,c))
        cfg['mass'].append(m)
        cfg['com'].append(c.tolist())
        cfg['inertia'].append(Ic.tolist())
    for k,key in enumerate(('viscous','coulomb','torque_bias','armature')):
        lo,hi = (-2,2) if k == 2 else (0,1 if k == 3 else 5)
        cfg[key] = np.clip(x[70+7*k:77+7*k],lo,hi).tolist()
    return cfg

def fit_physics(A,b,weights=None,ridge=1e-4,mask=None,prior=None,arm_min=0.0,arm_max=1.0,pseudo_floor=1e-6):
    if mask is not None:
        A,b = A[mask], b[mask]
        if weights is not None: weights = weights[mask]
    if weights is None: weights = np.ones(len(b))
    Aw, bw = A*weights[:,None], b*weights
    x = cp.Variable(98)
    cons=[]
    for i in range(7):
        v=x[i*10:(i+1)*10]
        m,h=v[0],v[1:4]
        I=cp.bmat([[v[4],v[7],v[8]],[v[7],v[5],v[9]],[v[8],v[9],v[6]]])
        S=0.5*cp.trace(I)*np.eye(3)-I
        J=cp.bmat([[S,cp.reshape(h,(3,1),order='C')],[cp.reshape(h,(1,3),order='C'),cp.reshape(m,(1,1),order='C')]])
        cons.extend([m>=0.05001,m<=9.99999,h<=0.39999*m,h>=-0.39999*m,cp.trace(S)<=0.24999*m,J>>pseudo_floor*np.eye(4)])
    cons.extend([x[70:84]>=0,x[70:84]<=5,x[84:91]>=-2,x[84:91]<=2,x[91:98]>=arm_min,x[91:98]<=arm_max])
    scales=np.array(([.1,1,1,1,5,5,5,5,5,5]*7)+[.2]*21+[1]*7)
    if prior is None: prior=np.zeros(98)
    obj=cp.sum_squares(Aw@x-bw)/(len(b)/7)+ridge*cp.sum_squares(cp.multiply(scales,x-prior))
    prob=cp.Problem(cp.Minimize(obj),cons)
    prob.solve(solver='CLARABEL',tol_gap_abs=1e-9,tol_feas=1e-9,tol_gap_rel=1e-9,max_iter=150)
    return x.value, {'status':prob.status,'objective':prob.value,'solve_time':prob.solver_stats.solve_time}

def torque_summary(A,b,x,episode_ids):
    err=(A@x-b).reshape(-1,7)
    out={'pooled':np.sqrt(np.mean(err**2,axis=0)).tolist()}
    for ep in np.unique(episode_ids):
        out[str(ep)]=np.sqrt(np.mean(err[episode_ids==ep]**2,axis=0)).tolist()
    return out
