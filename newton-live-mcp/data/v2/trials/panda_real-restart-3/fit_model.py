import argparse,json,time
from pathlib import Path
import numpy as np
import cvxpy as cp

p=argparse.ArgumentParser(); p.add_argument('--ridge',type=float,default=1e-5); p.add_argument('--arm-min',type=float,default=0); p.add_argument('--inertia-min',type=float,default=1e-5); p.add_argument('--exclude',type=int); p.add_argument('--output',default='config.json'); p.add_argument('--mode',default='weighted'); p.add_argument('--arm-vector'); p.add_argument('--huber',type=float,default=0); a=p.parse_args(); arm_floor=np.array(json.loads(a.arm_vector)) if a.arm_vector else np.full(7,a.arm_min)
d=np.load('training-regressor.npz'); A=d['A']; b=d['b']; eps=d['sample_episode_ids']; Y=b.reshape(-1,7)
scales=np.ones_like(Y)
if a.mode=='weighted':
 for ep in np.unique(eps):
  sel=eps==ep; scales[sel]=np.clip(Y[sel].std(0),.5,1)
mask=np.ones(len(b),dtype=bool) if a.exclude is None else np.repeat(eps!=a.exclude,7)
Aw=A[mask]/scales.ravel()[mask,None]; bw=b[mask]/scales.ravel()[mask]
# Compress residual algebra to improve the conic solve; discard float32 gauge noise.
u,s,vt=np.linalg.svd(Aw,full_matrices=False); keep=s>1e-2; R=s[keep,None]*vt[keep]; y=u[:,keep].T@bw
x=cp.Variable(98); constraints=[]
for link in range(7):
 z=x[10*link:10*(link+1)]; m=z[0]; h=z[1:4]; I=cp.bmat([[z[4],z[7],z[8]],[z[7],z[5],z[9]],[z[8],z[9],z[6]]]); S=.5*cp.trace(I)*np.eye(3)-I
 J=cp.bmat([[S,cp.reshape(h,(3,1),order='C')],[cp.reshape(h,(1,3),order='C'),cp.reshape(m,(1,1),order='C')]])
 constraints += [m>=.05001,m<=9.99999,h<=.39999*m,h>=-.39999*m,cp.trace(S)<=.24999*m,J >> np.diag([a.inertia_min]*3+[1e-8])]
constraints += [x[70:84]>=1e-8,x[70:84]<=4.999999,x[84:91]>=-1.999999,x[84:91]<=1.999999,x[91:98]>=arm_floor+1e-8,x[91:98]<=.999999]
regscale=np.array(([5]+[.5]*3+[.1]*6)*7+[1]*21+[.1]*7)
loss=cp.sum(cp.huber(Aw@x-bw,a.huber)) if a.huber>0 else cp.sum_squares(R@x-y)
obj=loss/(mask.sum()/7)+a.ridge*cp.sum_squares(cp.multiply(1/regscale,x))
prob=cp.Problem(cp.Minimize(obj),constraints); start=time.time(); prob.solve(solver='CLARABEL',tol_gap_abs=1e-9,tol_feas=1e-9,tol_gap_rel=1e-9,max_iter=300)
if x.value is None: raise RuntimeError(prob.status)
z=x.value; cfg={key:[redacted] for key in ['mass','com','inertia','viscous','coulomb','torque_bias','armature']}
for i in range(7):
 v=z[10*i:10*(i+1)]; m=v[0]; h=v[1:4]; c=h/m; I=np.array([[v[4],v[7],v[8]],[v[7],v[5],v[9]],[v[8],v[9],v[6]]]); Ic=I-m*((c@c)*np.eye(3)-np.outer(c,c)); cfg['mass'].append(m);cfg['com'].append(c.tolist());cfg['inertia'].append(Ic.tolist())
for k,st in [('viscous',70),('coulomb',77),('torque_bias',84),('armature',91)]:cfg[k]=z[st:st+7].tolist()
# Validation uses the exact public physical parameter mapping, no simulator is built.
from tools.mcp_evaluation.real_robot_model import validate_config,physical_coefficients
validate_config(cfg); err=(A@physical_coefficients(cfg)-b).reshape(-1,7)
stats={'status':prob.status,'seconds':time.time()-start,'args':vars(a),'objective':prob.value,'rmse':np.sqrt((err**2).mean(0)).tolist(),'per_episode':{str(ep):np.sqrt((err[eps==ep]**2).mean(0)).tolist() for ep in np.unique(eps)}}
Path(a.output).write_text(json.dumps(cfg,indent=2)+'\n');Path(a.output+'.fit.json').write_text(json.dumps(stats,indent=2)+'\n');print(json.dumps(stats)); print('mass',cfg['mass']);print('joint params',{k:cfg[k] for k in ['viscous','coulomb','torque_bias','armature']})
