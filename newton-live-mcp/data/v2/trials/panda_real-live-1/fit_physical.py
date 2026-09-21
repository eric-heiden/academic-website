import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import json, time
import numpy as np
import scipy.linalg as la
import cvxpy as cp

ROOT=Path(__file__).resolve().parent
D=np.load(ROOT/'training-regressor.npz'); A=D['A']; b=D['b']; ep=D['sample_episode_ids']
REF=np.load(ROOT/'training.npz')
COMP=[(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]

def to_config(x):
    out={k:[] for k in ['mass','com','inertia']}
    for j in range(7):
        z=x[10*j:10*j+10]; m=z[0]; c=z[1:4]/m
        io=np.zeros((3,3))
        for v,(a,b_) in zip(z[4:],COMP): io[a,b_]=io[b_,a]=v
        ic=io-m*(np.dot(c,c)*np.eye(3)-np.outer(c,c))
        out['mass'].append(float(m)); out['com'].append(c.tolist()); out['inertia'].append(ic.tolist())
    for i,k in enumerate(['viscous','coulomb','torque_bias','armature']):
        lo,hi=(-2,2) if k=='torque_bias' else (0,1 if k=='armature' else 5)
        out[k]=np.clip(x[70+i*7:77+i*7],lo,hi).tolist()
    return out

def stats(x):
    e=(A@x-b).reshape(-1,7)
    return {str(k): np.sqrt(np.mean(e[ep==k]**2,axis=0)).round(5).tolist() for k in np.unique(ep)}

def fit(name='fit_001',reg=1e-5,weights=None,mask=None,prior=None,margin=1e-6,arm_min=0.0):
    start=time.time(); x=cp.Variable(98); cons=[]
    for j in range(7):
        z=x[10*j:10*j+10]; m=z[0]; h=z[1:4]
        io=cp.bmat([[z[4],z[7],z[8]],[z[7],z[5],z[9]],[z[8],z[9],z[6]]])
        S=cp.trace(io)*.5*np.eye(3)-io
        P=cp.bmat([[S-margin*np.eye(3),cp.reshape(h,(3,1),order='C')],[cp.reshape(h,(1,3),order='C'),cp.reshape(m,(1,1),order='C')]])
        cons.extend([m>=.050001,m<=9.999999,h>=-.399999*m,h<=.399999*m,P>>0,cp.trace(S)<=.249999*m])
    cons.extend([x[70:84]>=0,x[70:84]<=5,x[84:91]>=-2,x[84:91]<=2,x[91:98]>=arm_min,x[91:98]<=1])
    if weights is None: weights=np.ones(7)
    if mask is None: mask=np.ones(len(b),dtype=bool)
    rowweight=np.tile(weights,len(b)//7)[mask]
    aa=A[mask]*rowweight[:,None]; bb=b[mask]*rowweight
    # Stable reduced residual through QR, preserving the least-squares minimizer.
    Q,R=la.qr(aa,mode='economic'); yy=Q.T@bb
    if prior is None:
        prior=np.zeros(98)
        for j in range(7): prior[j*10]=1; prior[j*10+4:j*10+7]=.01
    sc=np.r_[np.tile([.2,1,1,1,5,5,5,5,5,5],7),np.ones(28)]
    obj=cp.sum_squares(R@x-yy)/len(bb)+reg*cp.sum_squares(cp.multiply(sc,x-prior))
    prob=cp.Problem(cp.Minimize(obj),cons)
    prob.solve(solver='CLARABEL',tol_gap_abs=1e-9,tol_feas=1e-9,tol_gap_rel=1e-9,max_iter=300)
    if x.value is None: raise RuntimeError(prob.status)
    xx=x.value; cfg=to_config(xx)
    (ROOT/(name+'.json')).write_text(json.dumps(cfg,indent=2)+'\n')
    np.save(ROOT/(name+'_coeff.npy'),xx)
    report={'name':name,'status':prob.status,'seconds':time.time()-start,'objective':prob.value,'reg':reg,'weights':weights.tolist(),'rmse':stats(xx),'mass':cfg['mass'],'viscous':cfg['viscous'],'coulomb':cfg['coulomb'],'armature':cfg['armature'],'bias':cfg['torque_bias']}
    (ROOT/(name+'_fit.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    return xx

if __name__=='__main__':
    print('Data', {k:REF[k].shape for k in REF.files},'solvers',cp.installed_solvers(),flush=True)
    s=la.svdvals(A); print('Singular values',s.tolist(),flush=True)
    fit()
