import json, sys, time
from pathlib import Path
import numpy as np
import cvxpy as cp
sys.path.insert(0, '/home/horde/apps/newton-live-mcp')
from tools.mcp_evaluation.real_robot_model import validate_config, physical_coefficients
D=np.load('training-regressor.npz'); A=D['A']; b=D['b']; episode=D['sample_episode_ids'];

def decode(x):
    cfg={k:[] for k in ('mass','com','inertia')}
    for k in range(7):
        z=x[k*10:(k+1)*10]; m=z[0]; c=z[1:4]/m
        J=np.array([[z[4],z[7],z[8]],[z[7],z[5],z[9]],[z[8],z[9],z[6]]])
        I=J-m*(c@c*np.eye(3)-np.outer(c,c))
        cfg['mass'].append(float(m));cfg['com'].append(c.tolist());cfg['inertia'].append(I.tolist())
    for i,key in enumerate(('viscous','coulomb','torque_bias','armature')):
        cfg[key]=x[70+i*7:77+i*7].tolist()
    return validate_config(cfg)

def fit(weights=None, ridge=1e-5, train=None, robust=False, arm_min=0., inertia_margin=1e-6):
    if weights is None: weights=np.ones(7)
    rows=np.ones(len(b),dtype=bool) if train is None else np.repeat(np.isin(episode,train),7)
    w=np.tile(weights,len(b)//7)[rows]
    x=cp.Variable(98)
    constraints=[]
    for k in range(7):
        z=x[k*10:(k+1)*10];m=z[0];h=z[1:4]
        J=cp.bmat([[z[4],z[7],z[8]],[z[7],z[5],z[9]],[z[8],z[9],z[6]]])
        S=0.5*cp.trace(J)*np.eye(3)-J
        P=cp.bmat([[S,cp.reshape(h,(3,1),order='C')],[cp.reshape(h,(1,3),order='C'),cp.reshape(m,(1,1),order='C')]])
        constraints += [m>=.050001,m<=9.999999,h>=-.399999*m,h<=.399999*m,P>>inertia_margin*np.eye(4),cp.trace(S)<=.249999*m]
    for sl,lo,hi in [(slice(70,84),0,5),(slice(84,91),-2,2),(slice(91,98),arm_min,1)]:
        constraints += [x[sl]>=lo+1e-8,x[sl]<=hi-1e-8]
    res=cp.multiply(w,A[rows]@x-b[rows]); scales=np.r_[np.tile([3,.5,.5,.5,.2,.2,.2,.1,.1,.1],7),np.ones(28)]
    loss=cp.sum(cp.huber(res,.25)) if robust else cp.sum_squares(res)
    prob=cp.Problem(cp.Minimize(loss/rows.sum()+ridge*cp.sum_squares(cp.multiply(1/scales,x))),constraints)
    prob.solve(solver='CLARABEL',tol_gap_abs=1e-9,tol_feas=1e-9,tol_gap_rel=1e-9,max_iter=200)
    if x.value is None: raise RuntimeError(prob.status)
    cfg=decode(x.value)
    r=(A@physical_coefficients(cfg)-b).reshape(3,150,7)
    report={'status':prob.status,'objective':prob.value,'rmse':np.sqrt(np.mean(r*r,axis=(0,1))).tolist(),'episode_rmse':np.sqrt(np.mean(r*r,axis=1)).tolist(),'mass':cfg['mass'],'armature':cfg['armature']}
    return cfg,report

if __name__=='__main__':
    cfg,report=fit()
    Path('fit-001.json').write_text(json.dumps(cfg,indent=2)+'\n')
    Path('fit-001-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
