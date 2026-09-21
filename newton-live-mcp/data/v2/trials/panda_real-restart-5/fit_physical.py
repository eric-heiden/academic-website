import argparse, json, time
from pathlib import Path
import numpy as np
import cvxpy as cp

p=argparse.ArgumentParser()
p.add_argument('--reg',type=float,default=1e-4)
p.add_argument('--out',default='fit-001.json')
p.add_argument('--exclude',type=int,default=-1)
p.add_argument('--armature-min',type=float,default=0.0)
p.add_argument('--armature-floor',type=str,default=None)
p.add_argument('--inertia-min',type=float,default=1e-5)
p.add_argument('--weight',type=float,default=1.0)
a=p.parse_args()
armature_floor=np.full(7,a.armature_min) if a.armature_floor is None else np.array(json.loads(a.armature_floor))
assert armature_floor.shape==(7,)
data=np.load('training-regressor.npz'); A=data['A']; b=data['b']; eps=data['sample_episode_ids']; unique=np.unique(eps)
weights=np.ones_like(b).reshape(-1,7)
for ep in unique:
    selected=eps==ep
    weights[selected]=1/np.maximum(np.minimum(b.reshape(-1,7)[selected].std(axis=0),1.0),0.5)**a.weight
mask=np.repeat(eps!=a.exclude,7)
weights=weights.ravel()
x=cp.Variable(98)
cons=[]
regterms=[]
for k in range(7):
    v=x[10*k:10*k+10]; m=v[0]; h=v[1:4]
    I=cp.bmat([[v[4],v[7],v[8]],[v[7],v[5],v[9]],[v[8],v[9],v[6]]])
    S=0.5*cp.trace(I)*np.eye(3)-I
    P=cp.bmat([[S,cp.reshape(h,(3,1),order='C')],[cp.reshape(h,(1,3),order='C'),cp.reshape(m,(1,1),order='C')]])
    cons += [m>=0.05001,m<=9.99999,h>=-0.39999*m,h<=0.39999*m,cp.trace(S)<=0.24999*m,
             P-cp.diag([a.inertia_min,a.inertia_min,a.inertia_min,0.0]) >> 0]
    regterms += [cp.sum_squares(v[0]/5),cp.sum_squares(h),cp.sum_squares(v[4:]/0.2)]
cons += [x[70:84]>=0,x[70:84]<=5,x[84:91]>=-2,x[84:91]<=2,x[91:]>=armature_floor,x[91:]<=1]
regterms += [cp.sum_squares(x[70:84]/5),cp.sum_squares(x[84:91]/2),cp.sum_squares(x[91:]/0.5)]
Aw=A[mask]*weights[mask,None]; bw=b[mask]*weights[mask]
obj=cp.sum_squares(Aw@x-bw)/len(bw)+a.reg*sum(regterms)
problem=cp.Problem(cp.Minimize(obj),cons)
t=time.time();problem.solve(solver='CLARABEL',tol_gap_abs=1e-9,tol_feas=1e-9,tol_gap_rel=1e-9,max_iter=300)
print('solver',problem.status,'seconds',time.time()-t,'objective',problem.value,flush=True)
if x.value is None: raise RuntimeError('No solution')
v=x.value.copy()
config={k:[] for k in ['mass','com','inertia','viscous','coulomb','torque_bias','armature']}
for k in range(7):
    z=v[k*10:k*10+10]; m=z[0]; c=z[1:4]/m
    I=np.array([[z[4],z[7],z[8]],[z[7],z[5],z[9]],[z[8],z[9],z[6]]])-m*(c@c*np.eye(3)-np.outer(c,c))
    config['mass'].append(float(m));config['com'].append(c.tolist());config['inertia'].append(I.tolist())
for key,start,lo,hi in [('viscous',70,0,5),('coulomb',77,0,5),('torque_bias',84,-2,2),('armature',91,0,1)]:
    config[key]=np.clip(v[start:start+7],lo,hi).tolist()
Path(a.out).write_text(json.dumps(config,indent=2)+'\n')
pred=A@v; err=(pred-b).reshape(-1,7)
report={'reg':a.reg,'exclude':a.exclude,'armature_floor':armature_floor.tolist(),'objective':problem.value,'status':problem.status,'rmse':np.sqrt(np.mean(err**2,axis=0)).tolist(),'per_episode':[]}
for ep in unique:
    e=err[eps==ep]; std=np.maximum(b.reshape(-1,7)[eps==ep].std(axis=0),0.5)
    report['per_episode'].append({'episode':int(ep),'rmse':np.sqrt(np.mean(e**2,axis=0)).tolist(),'normalized_rmse':(np.sqrt(np.mean(e**2,axis=0))/std).tolist()})
Path(a.out+'.report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
print('mass',config['mass'],'armature',config['armature'],'viscous',config['viscous'],'coulomb',config['coulomb'],flush=True)
