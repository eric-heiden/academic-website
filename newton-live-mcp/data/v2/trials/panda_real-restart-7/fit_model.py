import json, argparse, time
from pathlib import Path
import numpy as np
import cvxpy as cp

D=np.load('training-regressor.npz'); A=D['A']; b=D['b']; episodes=D['sample_episode_ids']; ids=np.unique(episodes)
S=np.tile([3,.5,.5,.5,.2,.2,.2,.1,.1,.1],7).tolist()+[2]*7+[2]*7+[.5]*7+[.2]*7
S=np.array(S)
x0=np.zeros(98)
for k in range(7): x0[k*10]=1; x0[k*10+4:k*10+7]=.01
scale=np.minimum(np.maximum(b.reshape(-1,7).std(0),.5),1.)

def to_config(x):
    out={k:[] for k in ['mass','com','inertia']}
    for k in range(7):
        m=x[10*k]; h=x[10*k+1:10*k+4]; v=x[10*k+4:10*k+10]
        Io=np.array([[v[0],v[3],v[4]],[v[3],v[1],v[5]],[v[4],v[5],v[2]]]); c=h/m
        Ic=Io-m*(c@c*np.eye(3)-np.outer(c,c))
        out['mass'].append(float(m));out['com'].append(c.tolist());out['inertia'].append(Ic.tolist())
    for n,k in enumerate(['viscous','coulomb','torque_bias','armature']):
        lo,hi=((-2,2) if k=='torque_bias' else (0,1) if k=='armature' else (0,5))
        out[k]=np.clip(x[70+n*7:77+n*7],lo,hi).tolist()
    return out

def solve(alpha=1e-4, train=None, robust=None, weights=None, amin=0, imargin=1e-6, cmax=5, friction_penalty=0):
    x=cp.Variable(98); con=[]
    for k in range(7):
        m=x[k*10]; h=x[k*10+1:k*10+4]; v=x[k*10+4:k*10+10]
        I=cp.bmat([[v[0],v[3],v[4]],[v[3],v[1],v[5]],[v[4],v[5],v[2]]])
        sig=.5*cp.trace(I)*np.eye(3)-I
        P=cp.bmat([[sig-imargin*np.eye(3),cp.reshape(h,(3,1),order='C')],[cp.reshape(h,(1,3),order='C'),cp.reshape(m,(1,1),order='C')]])
        con.extend([m>=.050001,m<=9.99999,h<=.39999*m,h>=-.39999*m,cp.trace(sig)<=.24999*m,P>>0])
    con.extend([x[70:84]>=0,x[70:84]<=5,x[77:84]<=cmax,x[84:91]>=-2,x[84:91]<=2,x[91:98]>=amin,x[91:98]<=1])
    mask=np.ones(len(b),bool) if train is None else np.repeat(np.isin(episodes,train),7)
    w=np.tile(1/scale,len(b)//7) if weights is None else weights
    residual=cp.multiply(w[mask],A[mask]@x-b[mask])
    loss=cp.sum_squares(residual) if robust is None else cp.sum(cp.huber(residual,robust))
    obj=loss/mask.sum()+alpha*cp.sum_squares(cp.multiply(1/S,x-x0))/98+friction_penalty*cp.sum_squares(x[77:84])/7
    prob=cp.Problem(cp.Minimize(obj),con)
    prob.solve(solver='CLARABEL',max_iter=250,tol_gap_abs=1e-9,tol_feas=1e-9,tol_gap_rel=1e-9)
    if x.value is None:raise RuntimeError(prob.status)
    return x.value, prob.status

def summary(x):
    e=(A@x-b).reshape(-1,7); out={}
    for ep in [None,*ids]:
        sel=np.ones(len(e),bool) if ep is None else episodes==ep
        rmse=np.sqrt(np.mean(e[sel]**2,axis=0)); normal=rmse/np.maximum(b.reshape(-1,7)[sel].std(0),.5)
        out[str(ep)]={'rmse':rmse.tolist(),'ratio':max(rmse.max()/.5,normal.max()/.5)}
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--alpha',type=float,default=1e-4);p.add_argument('--robust',type=float);p.add_argument('--cv',action='store_true');p.add_argument('--output',default='config.json');p.add_argument('--amin',type=float,default=0);args=p.parse_args()
    st=time.time();x,status=solve(args.alpha,robust=args.robust,amin=args.amin)
    cfg=to_config(x);Path(args.output).write_text(json.dumps(cfg,indent=2)+'\n')
    print('full',status,json.dumps(summary(x)),flush=True)
    print('params',json.dumps({k:cfg[k] for k in ['mass','viscous','coulomb','torque_bias','armature']}),flush=True)
    if args.cv:
        for ep in ids:
            xx,stat=solve(args.alpha,train=ids[ids!=ep],robust=args.robust,amin=args.amin)
            print('validation',int(ep),stat,json.dumps(summary(xx)[str(ep)]),flush=True)
    print('seconds',time.time()-st,flush=True)
