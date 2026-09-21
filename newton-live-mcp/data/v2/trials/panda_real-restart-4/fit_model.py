import argparse, json, time
from pathlib import Path
import numpy as np
import cvxpy as cp

ROOT=Path(__file__).resolve().parent
D=np.load(ROOT/'training-regressor.npz')
A,b=D['A'],D['b']; episodes=D['sample_episode_ids']
REF=np.load(ROOT/'training.npz')

def constraints(x):
    cons=[]
    for i in range(7):
        k=10*i;m=x[k];h=x[k+1:k+4]
        z=x[k+4:k+10]
        I=cp.bmat([[z[0],z[3],z[4]],[z[3],z[1],z[5]],[z[4],z[5],z[2]]])
        S=0.5*cp.trace(I)*np.eye(3)-I
        P=cp.bmat([[S,cp.reshape(h,(3,1),order='C')],[cp.reshape(h,(1,3),order='C'),cp.reshape(m,(1,1),order='C')]])
        cons += [m>=0.05001,m<=9.99999,h<=.39999*m,h>=-.39999*m,cp.trace(S)<=.24999*m,P>>1e-6*np.eye(4)]
    cons += [x[70:84]>=0,x[70:84]<=5,x[84:91]>=-2,x[84:91]<=2,x[91:98]>=0,x[91:98]<=1]
    return cons

def to_config(x):
    out={key:[redacted] for key in ['mass','com','inertia','viscous','coulomb','torque_bias','armature']}
    for i in range(7):
        m=x[i*10];c=x[i*10+1:i*10+4]/m;z=x[i*10+4:i*10+10]
        I=np.array([[z[0],z[3],z[4]],[z[3],z[1],z[5]],[z[4],z[5],z[2]]])-m*(c@c*np.eye(3)-np.outer(c,c))
        out['mass'].append(float(m));out['com'].append(c.tolist());out['inertia'].append(I.tolist())
    for key,k,lo,hi in [('viscous',70,0,5),('coulomb',77,0,5),('torque_bias',84,-2,2),('armature',91,0,1)]:
        out[key]=np.clip(x[k:k+7],lo,hi).tolist()
    return out

def score(x):
    err=(A@x-b).reshape(-1,7)
    ret={'pooled':np.sqrt(np.mean(err**2,axis=0)).tolist()}
    for ep in np.unique(episodes):
        mask=episodes==ep
        rmse=np.sqrt(np.mean(err[mask]**2,axis=0))
        norm=rmse/np.maximum(b.reshape(-1,7)[mask].std(axis=0),.5)
        ret[str(ep)]={'rmse':rmse.tolist(),'nrmse':norm.tolist()}
    return ret

def fit(reg=1e-4, weight=1, omit=None, huber=0, prior=None, armprior=0):
    x=cp.Variable(98)
    w=np.ones((len(b)//7,7))
    for ep in np.unique(episodes):
        mask=episodes==ep
        w[mask]=1/np.minimum(np.maximum(b.reshape(-1,7)[mask].std(axis=0),.5),1)**weight
        if omit==ep:w[mask]=0
    w=w.ravel()
    residual=cp.multiply(w,A@x-b)
    loss=cp.sum_squares(residual)/np.sum(w>0)*7 if huber==0 else cp.sum(cp.huber(residual,huber))/np.sum(w>0)*7
    # Weak scale-aware regularization selects bounded representatives of unidentifiable dynamics.
    scale=np.tile([.2,1,1,1,3,3,3,4,4,4],7).tolist()+[.2]*7+[.2]*7+[.2]*7+[1]*7
    if prior is None: prior=np.zeros(98)
    loss+=reg*cp.sum_squares(cp.multiply(scale,x-prior))
    if armprior:loss+=armprior*cp.sum_squares(x[91:98]-.08)
    p=cp.Problem(cp.Minimize(loss),constraints(x)); t=time.time()
    p.solve(solver='CLARABEL',tol_gap_abs=1e-9,tol_feas=1e-9,tol_gap_rel=1e-9,max_iter=300)
    print('solve',p.status,p.value,'seconds',time.time()-t,flush=True)
    if x.value is None:raise ValueError(p.status)
    return x.value

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reg',type=float,default=1e-4);p.add_argument('--weight',type=float,default=1);p.add_argument('--omit',type=int);p.add_argument('--huber',type=float,default=0);p.add_argument('--out',default='fit-001');p.add_argument('--armprior',type=float,default=0)
    args=p.parse_args()
    print('solvers',cp.installed_solvers(),flush=True)
    print('reference', {k:REF[k].shape for k in REF.files},flush=True)
    print('spectrum',np.linalg.svd(A,compute_uv=False).tolist(),flush=True)
    x=fit(args.reg,args.weight,args.omit,args.huber,armprior=args.armprior)
    Path(args.out+'.json').write_text(json.dumps(to_config(x),indent=2)+'\n')
    np.save(args.out+'.npy',x)
    result=score(x);Path(args.out+'-scores.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    print(json.dumps(to_config(x)),flush=True)
