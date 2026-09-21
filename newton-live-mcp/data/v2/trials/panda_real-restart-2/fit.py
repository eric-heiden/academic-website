import argparse,json,time
from pathlib import Path
import numpy as np
import cvxpy as cp

parser=argparse.ArgumentParser()
parser.add_argument('--output',default='fit-001.json')
parser.add_argument('--ridge',type=float,default=0.001)
parser.add_argument('--arm-min',default='0')
parser.add_argument('--weights',default='1,1,1,1,1,1,1')
parser.add_argument('--exclude',type=int,default=-1)
parser.add_argument('--huber',type=float,default=0.)
a=parser.parse_args()
arm_min=np.array([float(v) for v in a.arm_min.split(',')])
if arm_min.size==1: arm_min=np.full(7,arm_min[0])
if arm_min.size!=7: raise ValueError('arm-min requires one or seven values')
r=np.load('training-regressor.npz'); A,b=r['A'],r['b']; ids=r['sample_episode_ids']
weights=np.array([float(v) for v in a.weights.split(',')]); rowweights=np.tile(weights,len(b)//7)
mask=np.repeat(ids!=a.exclude,7)
x=cp.Variable(98); cons=[]
for j in range(7):
 k=10*j; m=x[k]; h=x[k+1:k+4]
 I=cp.bmat([[x[k+4],x[k+7],x[k+8]],[x[k+7],x[k+5],x[k+9]],[x[k+8],x[k+9],x[k+6]]])
 S=.5*cp.trace(I)*np.eye(3)-I
 J=cp.bmat([[S,cp.reshape(h,(3,1),order='C')],[cp.reshape(h,(1,3),order='C'),cp.reshape(m,(1,1),order='C')]])
 cons += [J >> 1e-6*np.eye(4),m>=.05001,m<=9.99999,h>=-.39999*m,h<=.39999*m,cp.trace(S)<=.24999*m]
cons += [x[70:84]>=1e-9,x[70:84]<=4.999999,x[84:91]>=-1.999999,x[84:91]<=1.999999,x[91:]>=arm_min+1e-9,x[91:]<=.999999]
scale=np.r_[np.tile([1,.2,.2,.2,.05,.05,.05,.05,.05,.05],7),np.ones(28)*.2]
e=cp.multiply(rowweights[mask],A[mask]@x-b[mask])
loss=cp.sum_squares(e) if a.huber<=0 else cp.sum(cp.huber(e,a.huber))
obj=loss+a.ridge*cp.sum_squares(cp.multiply(1/scale,x))
p=cp.Problem(cp.Minimize(obj),cons); t=time.time();p.solve(solver='CLARABEL',tol_gap_abs=1e-8,tol_feas=1e-9,tol_gap_rel=1e-9,max_iter=300)
print('solve',p.status,p.value,'seconds',time.time()-t,flush=True)
z=x.value
if z is None: raise RuntimeError(p.status)
out={k:[] for k in ['mass','com','inertia','viscous','coulomb','torque_bias','armature']}
for j in range(7):
 k=10*j; m=z[k]; c=z[k+1:k+4]/m
 Io=np.array([[z[k+4],z[k+7],z[k+8]],[z[k+7],z[k+5],z[k+9]],[z[k+8],z[k+9],z[k+6]]])
 Ic=Io-m*(c@c*np.eye(3)-np.outer(c,c))
 out['mass'].append(m);out['com'].append(c.tolist());out['inertia'].append(Ic.tolist())
for n,k in enumerate(['viscous','coulomb','torque_bias','armature']):out[k]=z[70+7*n:77+7*n].tolist()
err=(A@z-b).reshape(-1,7); print('pooled RMSE',np.sqrt(np.mean(err**2,axis=0)))
for id in np.unique(ids):print('episode',id,'RMSE',np.sqrt(np.mean(err[ids==id]**2,axis=0)))
print('joint',z[70:]);print('mass',out['mass']);print('com',out['com'])
Path(a.output).write_text(json.dumps(out,indent=2)+'\n')
np.savez(Path(a.output).with_suffix('.npz'),x=z,prediction=(A@z).reshape(-1,7),errors=err)
