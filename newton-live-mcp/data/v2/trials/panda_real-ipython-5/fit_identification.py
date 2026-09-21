"""Physical identification using only the supplied immutable training matrix."""
import json
import numpy as np
import cvxpy as cp

COMPONENTS = ((0,0),(1,1),(2,2),(0,1),(0,2),(1,2))

def coefficients_to_config(x):
    result = {k: [] for k in ('mass','com','inertia')}
    for i in range(7):
        z = x[10*i:10*i+10]
        m, h = z[0], z[1:4]
        c = h/m
        I = np.zeros((3,3))
        for v,(a,b) in zip(z[4:], COMPONENTS):
            I[a,b]=I[b,a]=v
        Ic=I-m*(np.dot(c,c)*np.eye(3)-np.outer(c,c))
        result['mass'].append(float(m))
        result['com'].append(c.tolist())
        result['inertia'].append(Ic.tolist())
    for i,k in enumerate(('viscous','coulomb','torque_bias','armature')):
        v=x[70+7*i:77+7*i].copy()
        lo,hi = (-2,2) if k=='torque_bias' else (0,1 if k=='armature' else 5)
        result[k]=np.clip(v,lo,hi).tolist()
    return result

def fit_physical(A,b,weights=None,ridge=1e-5,armature_min=0.,inertia_min=1e-5,mask=None,robust=None):
    x=cp.Variable(98)
    constraints=[]
    for i in range(7):
        z=x[10*i:10*i+10]
        m,h=z[0],z[1:4]
        I=cp.bmat([[z[4],z[7],z[8]],[z[7],z[5],z[9]],[z[8],z[9],z[6]]])
        S=0.5*cp.trace(I)*np.eye(3)-I
        P=cp.bmat([[S,cp.reshape(h,(3,1),order='C')],[cp.reshape(h,(1,3),order='C'),cp.reshape(m,(1,1),order='C')]])
        constraints += [m>=0.05001,m<=9.99999,h>=-0.39999*m,h<=0.39999*m,cp.trace(S)<=0.24999*m,P >> np.diag([inertia_min]*3+[0.])]
    constraints += [x[70:84]>=0,x[70:84]<=5,x[84:91]>=-2,x[84:91]<=2,x[91:98]>=armature_min,x[91:98]<=1]
    if weights is None: weights=np.ones(7)
    w=np.tile(np.sqrt(weights),len(b)//7)
    if mask is None: mask=np.ones(len(b),dtype=bool)
    target=(A*w[:,None])[mask]
    y=(b*w)[mask]
    # SVD compression keeps the fit small and preserves the complete LS objective.
    u,s,v=np.linalg.svd(target,full_matrices=False)
    resid=cp.multiply(s,v@x)-u.T@y
    scale=np.tile([1.,5.,5.,5.,10.,10.,10.,10.,10.,10.],7)
    penalty=cp.sum_squares(cp.multiply(scale,x[:70]))+cp.sum_squares(x[70:])
    problem=cp.Problem(cp.Minimize(cp.sum_squares(resid)/(mask.sum()/7)+ridge*penalty),constraints)
    problem.solve(solver='CLARABEL',tol_gap_abs=1e-9,tol_feas=1e-9,tol_gap_rel=1e-9,max_iter=250)
    if x.value is None: raise RuntimeError(problem.status)
    return coefficients_to_config(x.value), x.value, {'status':problem.status,'objective':problem.value}
