import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import json, argparse, time
from pathlib import Path
import numpy as np
import scipy.linalg as la
import cvxpy as cp

KEYS=('viscous','coulomb','torque_bias','armature')

def to_config(x):
    c={'mass':[],'com':[],'inertia':[]}
    for j in range(7):
        m=x[j*10]; h=x[j*10+1:j*10+4]; v=x[j*10+4:j*10+10]
        I=np.array([[v[0],v[3],v[4]],[v[3],v[1],v[5]],[v[4],v[5],v[2]]])
        center=h/m
        Ic=I-m*(center@center*np.eye(3)-np.outer(center,center))
        c['mass'].append(float(m));c['com'].append(center.tolist());c['inertia'].append(Ic.tolist())
    for i,k in enumerate(KEYS):
        c[k]=x[70+i*7:77+i*7].tolist()
    return c

def fit(A,b,ep,lam=1e-4, mask=None, weights=None, floor=1e-5, arm_floor=1e-8):
    if mask is None: mask=np.ones(len(b),dtype=bool)
    if weights is None:
        weights=np.ones_like(b)
        for e in np.unique(ep):
            rows=np.repeat(ep==e,7)
            scale=np.clip(b[rows].reshape(-1,7).std(axis=0),.5,1.)
            weights[rows]=np.tile(1/scale, (ep==e).sum())
    Aw=A[mask]*weights[mask,None]; bw=b[mask]*weights[mask]
    R=la.qr(np.column_stack([Aw,bw]),mode='r')[0][:99]
    x=cp.Variable(98)
    cons=[]
    for j in range(7):
        m=x[j*10];h=x[j*10+1:j*10+4];v=x[j*10+4:j*10+10]
        I=cp.bmat([[v[0],v[3],v[4]],[v[3],v[1],v[5]],[v[4],v[5],v[2]]])
        S=.5*cp.trace(I)*np.eye(3)-I
        J=cp.bmat([[S-floor*np.eye(3),cp.reshape(h,(3,1),order='C')],[cp.reshape(h,(1,3),order='C'),cp.reshape(m,(1,1),order='C')]])
        cons += [J >> 0, m>=.050001,m<=9.999999,h>=-.399999*m,h<=.399999*m,cp.trace(S)<=.249999*m]
    cons += [x[70:84]>=1e-8,x[70:84]<=4.999999,x[84:91]>=-1.999999,x[84:91]<=1.999999,x[91:]>=arm_floor,x[91:]<=.999999]
    p0=np.zeros(98); scales=np.ones(98)
    for j in range(7):
        p0[10*j]=1.;p0[10*j+4:10*j+7]=.01
        scales[10*j]=3.; scales[10*j+1:10*j+4]=.3;scales[10*j+4:10*j+10]=.1
    scales[91:]=.3
    obj=cp.sum_squares(R[:,:98]@x-R[:,98])/mask.sum()+lam*cp.sum_squares(cp.multiply(1/scales,x-p0))
    prob=cp.Problem(cp.Minimize(obj),cons)
    prob.solve(solver='CLARABEL',tol_gap_abs=1e-9,tol_feas=1e-9,tol_gap_rel=1e-9,max_iter=300)
    if x.value is None:raise RuntimeError(prob.status)
    return x.value,prob.status,prob.value

def report(A,b,ep,x):
    err=(A@x-b).reshape(-1,7)
    r={'pooled':np.sqrt(np.mean(err**2,axis=0)).tolist(),'episodes':{}}
    for e in np.unique(ep):r['episodes'][str(e)]=np.sqrt(np.mean(err[ep==e]**2,axis=0)).tolist()
    return r

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--lam',type=float,default=1e-4);parser.add_argument('--output',default='fit-001');parser.add_argument('--cv',action='store_true');parser.add_argument('--arm-floor',type=float,default=1e-8);parser.add_argument('--inertia-floor',type=float,default=1e-5);args=parser.parse_args()
    d=np.load('training-regressor.npz');A=d['A'];b=d['b'];ep=d['sample_episode_ids']
    print('regressor',d.files,A.shape,'rank',np.linalg.matrix_rank(A,tol=1e-3),flush=True)
    start=time.time();x,status,objective=fit(A,b,ep,args.lam,arm_floor=args.arm_floor,floor=args.inertia_floor)
    c=to_config(x);Path(args.output+'.json').write_text(json.dumps(c,indent=2)+'\n')
    np.save(args.output+'.npy',x)
    out={'lambda':args.lam,'status':status,'objective':objective,'torque':report(A,b,ep,x),'seconds':time.time()-start}
    if args.cv:
        out['cv']={}
        for e in np.unique(ep):
            xc,st,ob=fit(A,b,ep,args.lam,mask=np.repeat(ep!=e,7),arm_floor=args.arm_floor,floor=args.inertia_floor)
            out['cv'][str(e)]=report(A,b,ep,xc)['episodes'][str(e)]
    Path(args.output+'-report.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
