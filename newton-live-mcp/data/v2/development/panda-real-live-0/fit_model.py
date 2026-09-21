import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import numpy as np, scipy.linalg as la, cvxpy as cp, json, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
reg=np.load(ROOT/'training-regressor.npz'); A=reg['A']; b=reg['b']; episodes=reg['sample_episode_ids']; ids=np.unique(episodes)
E=np.repeat(episodes,7); J=np.tile(np.arange(7),len(episodes))
stds=np.array([np.maximum(b.reshape(-1,7)[episodes==e].std(axis=0),.5) for e in ids])
weights=np.empty_like(b)
for e,s in zip(ids,stds): weights[E==e]=np.tile(1/np.minimum(s,1.),sum(episodes==e))

def config_from_x(x):
    config={k:[] for k in ['mass','com','inertia','viscous','coulomb','torque_bias','armature']}
    for i in range(7):
        v=x[i*10:(i+1)*10]; m=v[0]; c=v[1:4]/m
        I=np.array([[v[4],v[7],v[8]],[v[7],v[5],v[9]],[v[8],v[9],v[6]]])-m*((c@c)*np.eye(3)-np.outer(c,c))
        config['mass'].append(float(m));config['com'].append(c.tolist());config['inertia'].append(I.tolist())
    for i,k in enumerate(['viscous','coulomb','torque_bias','armature']):config[k]=x[70+i*7:77+i*7].tolist()
    return config

def score(x):
    err=(A@x-b).reshape(-1,7)
    return {str(e):np.sqrt(np.mean(err[episodes==e]**2,axis=0)).round(6).tolist() for e in ids}

def solve(lam=1e-4, train=None, loss='l2', arm_min=1e-6, visc_min=1e-7):
    if train is None:train=np.ones(len(b),bool)
    x=cp.Variable(98)
    constraints=[]
    for i in range(7):
        v=x[i*10:(i+1)*10];m=v[0];h=v[1:4]
        I=cp.bmat([[v[4],v[7],v[8]],[v[7],v[5],v[9]],[v[8],v[9],v[6]]])
        S=.5*cp.trace(I)*np.eye(3)-I
        P=cp.bmat([[S,cp.reshape(h,(3,1),order='C')],[cp.reshape(h,(1,3),order='C'),cp.reshape(m,(1,1),order='C')]])
        constraints.extend([m>=.050001,m<=9.999999,h<=.399999*m,h>=-.399999*m,P>>1e-6*np.eye(4),cp.trace(S)<=.249999*m])
    constraints.extend([x[70:84]>=1e-7,x[70:84]<=4.999999,x[84:91]>=-1.999999,x[84:91]<=1.999999,x[91:]>=arm_min,x[91:]<=.999999])
    constraints.append(x[70:77]>=visc_min)
    aw=A[train]*weights[train,None]; bw=b[train]*weights[train]
    # QR retains the exact least-squares optimum with a small conic problem.
    Q,R=la.qr(aw,mode='economic'); y=Q.T@bw
    scales=np.tile([1,3,3,3,10,10,10,10,10,10],7)
    prior=np.tile([1,0,0,0,.01,.01,.01,0,0,0],7)
    residual=R@x-y
    objective=cp.sum_squares(residual)/sum(train)+lam*cp.sum_squares(cp.multiply(scales,x[:70]-prior))+lam*.1*cp.sum_squares(x[70:])
    p=cp.Problem(cp.Minimize(objective),constraints)
    p.solve(solver='CLARABEL',tol_gap_abs=1e-9,tol_feas=1e-9,tol_gap_rel=1e-9,max_iter=200)
    if x.value is None:raise RuntimeError(p.status)
    return x.value,p.status,float(p.value)

if __name__=='__main__':
    print('solvers',cp.installed_solvers(),'shape',A.shape,'episodes',ids,'stds',stds,flush=True)
    s=la.svdvals(A);print('singular values',s.round(6),flush=True)
    for lam in [1e-5,1e-4,1e-3,0.0]:
        t=time.time();x,status,obj=solve(lam)
        name='fit_'+str(lam)
        (ROOT/(name+'.json')).write_text(json.dumps(config_from_x(x),indent=2)+'\n')
        np.save(ROOT/(name+'.npy'),x)
        print(name,status,obj,'secs',time.time()-t,'rmse',score(x),'joint',x[70:].round(4).tolist(),'m',x[::10][:7].round(4).tolist(),flush=True)
