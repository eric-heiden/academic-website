"""Constrained identification using only the supplied immutable regression data."""
import argparse
import json
from pathlib import Path
import numpy as np
import cvxpy as cp

COMPONENTS = ((0,0),(1,1),(2,2),(0,1),(0,2),(1,2))

def decode(x):
    cfg = {k: [] for k in ('mass','com','inertia','viscous','coulomb','torque_bias','armature')}
    for i in range(7):
        z=x[10*i:10*i+10]; m=float(z[0]); c=z[1:4]/m
        io=np.zeros((3,3))
        for v,(a,b) in zip(z[4:],COMPONENTS): io[a,b]=io[b,a]=v
        ic=io-m*(c@c*np.eye(3)-np.outer(c,c))
        cfg['mass'].append(m); cfg['com'].append(c.tolist());cfg['inertia'].append(ic.tolist())
    for i,k in enumerate(('viscous','coulomb','torque_bias','armature')):
        lo,hi=(-2.,2.) if k=='torque_bias' else (0.,1. if k=='armature' else 5.)
        cfg[k]=np.clip(x[70+7*i:77+7*i],lo,hi).tolist()
    return cfg

def fit(A,b,mask,reg,weights=None,prior=None,robust=False,armature_floor=None):
    x=cp.Variable(98)
    constraints=[]
    for i in range(7):
        z=x[10*i:10*i+10]; m=z[0]; h=z[1:4]
        io=cp.bmat([[z[4],z[7],z[8]],[z[7],z[5],z[9]],[z[8],z[9],z[6]]])
        s=cp.trace(io)*.5*np.eye(3)-io
        p=cp.bmat([[s,cp.reshape(h,(3,1),order='C')],[cp.reshape(h,(1,3),order='C'),cp.reshape(m,(1,1),order='C')]])
        constraints += [m>=.05001,m<=9.99999,h>=-.39999*m,h<=.39999*m,
                        p-np.diag([1e-6,1e-6,1e-6,0]) >> 0,cp.trace(s)<=.24999*m]
    constraints += [x[70:84]>=1e-8,x[70:84]<=4.999999,x[84:91]>=-1.999999,
                    x[84:91]<=1.999999,x[91:98]>=1e-8,x[91:98]<=.999999]
    if armature_floor is not None: constraints.append(x[91:98]>=np.asarray(armature_floor)+1e-8)
    scales=np.r_[np.tile([3.,.3,.3,.3,.1,.1,.1,.1,.1,.1],7),np.ones(21),np.ones(7)*.2]
    if weights is None: weights=np.tile([1,1,1,1,1.5,1,2],len(b)//7)
    am=A[mask]*weights[mask,None]; bm=b[mask]*weights[mask]
    if robust:
        loss=cp.sum(cp.huber(am@x-bm,.4))/mask.sum()
    else:
        # Compression preserves the least-squares minimizer and reduces cone size.
        q,r=np.linalg.qr(am,mode='reduced'); y=q.T@bm
        loss=cp.sum_squares(r@x-y)/mask.sum()
    penalty=cp.sum_squares(cp.multiply(1/scales,x if prior is None else x-prior))
    problem=cp.Problem(cp.Minimize(loss+reg*penalty),constraints)
    problem.solve(solver='CLARABEL',tol_gap_abs=1e-9,tol_feas=1e-9,tol_gap_rel=1e-9,max_iter=250)
    if x.value is None: raise RuntimeError(problem.status)
    return x.value,problem.status

def scores(A,b,x,eids):
    e=(A@x-b).reshape(-1,7); target=b.reshape(-1,7)
    out={'pooled':np.sqrt((e*e).mean(axis=0)).tolist()}
    for ep in np.unique(eids):
        s=eids==ep; rmse=np.sqrt((e[s]**2).mean(axis=0))
        out[str(ep)]={'rmse':rmse.tolist(),'normalized':(rmse/np.maximum(target[s].std(axis=0),.5)).tolist()}
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--reg',type=float,default=1e-5);p.add_argument('--output',default='fit-001.json');p.add_argument('--cv',action='store_true');p.add_argument('--robust',action='store_true');p.add_argument('--armature-floor',type=float,nargs=7);args=p.parse_args()
    d=np.load('training-regressor.npz'); raw=d['A'];b=d['b'];eids=d['sample_episode_ids']
    u,s,v=np.linalg.svd(raw,full_matrices=False);s[s<.01]=0;A=(u*s)@v
    records={}
    if args.cv:
        for ep in np.unique(eids):
            x,status=fit(A,b,np.repeat(eids!=ep,7),args.reg,robust=args.robust,armature_floor=args.armature_floor)
            records[str(ep)]={'status':status,'scores':scores(raw,b,x,eids)}
            print('leave out',ep,records[str(ep)],flush=True)
    x,status=fit(A,b,np.ones(len(b),bool),args.reg,robust=args.robust,armature_floor=args.armature_floor)
    cfg=decode(x)
    from tools.mcp_evaluation.real_robot_model import validate_config
    validate_config(cfg)
    Path(args.output).write_text(json.dumps(cfg,indent=2)+'\n')
    report={'regularization':args.reg,'status':status,'scores':scores(raw,b,x,eids),'cross_validation':records}
    Path(args.output).with_suffix('.report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
