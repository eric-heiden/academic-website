import json, time
from pathlib import Path
import numpy as np
import scipy.linalg as la
import cvxpy as cp
from tools.mcp_evaluation.real_robot_model import physical_coefficients, validate_config

work = Path('/home/horde/artifacts/newton-live-mcp-v2/confirmation/panda_real-ipython-4')
reg = dict(np.load(work/'training-regressor.npz'))
data = dict(np.load(work/'training.npz'))
A, b = reg['A'], reg['b']
episodes = reg['sample_episode_ids']
fit_history = []
eval_history = []

def coefficients_config(x):
    cfg = {k:[] for k in ['mass','com','inertia','viscous','coulomb','torque_bias','armature']}
    for j in range(7):
        v=x[j*10:j*10+10]
        m=v[0]; c=v[1:4]/m
        I=np.array([[v[4],v[7],v[8]],[v[7],v[5],v[9]],[v[8],v[9],v[6]]])
        Ic=I-m*(c@c*np.eye(3)-np.outer(c,c))
        cfg['mass'].append(float(m)); cfg['com'].append(c.tolist()); cfg['inertia'].append(Ic.tolist())
    for k, s in [('viscous',70),('coulomb',77),('torque_bias',84),('armature',91)]:
        bounds=(-2,2) if k=='torque_bias' else (0,1 if k=='armature' else 5)
        cfg[k]=np.clip(x[s:s+7],*bounds).tolist()
    return validate_config(cfg)

def torque_stats(x, matrix=A, target=b, eps=episodes):
    err=(matrix@x-target).reshape(-1,7)
    return {str(ep):np.sqrt(np.mean(err[eps==ep]**2,axis=0)).round(5).tolist() for ep in np.unique(eps)}

def fit_physical(lam=1e-4, weights=None, keep=None, prior=None, arm_floor=0., loss='square', huber_delta=.5):
    started=time.perf_counter()
    if keep is None: keep=np.ones(len(b),dtype=bool)
    if weights is None: weights=np.ones(len(b))
    if np.size(weights)==7: weights=np.tile(weights,len(b)//7)
    weights=np.asarray(weights)
    aa=A[keep]*weights[keep,None]; bb=b[keep]*weights[keep]
    x=cp.Variable(98)
    cons=[]
    for j in range(7):
        v=x[j*10:j*10+10]; m=v[0]; h=v[1:4]
        I=cp.bmat([[v[4],v[7],v[8]],[v[7],v[5],v[9]],[v[8],v[9],v[6]]])
        S=.5*cp.trace(I)*np.eye(3)-I
        P=cp.bmat([[S,cp.reshape(h,(3,1),order='C')],[cp.reshape(h,(1,3),order='C'),cp.reshape(m,(1,1),order='C')]])
        cons.extend([m>=.050001,m<=9.999999,h<=.399999*m,h>=-.399999*m,P-np.diag([1e-6]*3+[0]) >> 0,cp.trace(S)<=.249999*m])
    cons.extend([x[70:84]>=0,x[70:84]<=5,x[84:91]>=-2,x[84:91]<=2,x[91:98]>=arm_floor,x[91:98]<=1])
    if loss=='square':
        qa,ra=la.qr(aa,mode='economic')
        err_obj=cp.sum_squares(ra@x-qa.T@bb)/len(bb)
    else:
        err_obj=cp.sum(cp.huber(aa@x-bb,huber_delta))/len(bb)
    scales=np.r_[np.tile([3,.5,.5,.5,.2,.2,.2,.1,.1,.1],7),np.ones(21)*2,np.ones(7)*.2]
    ref=np.zeros(98) if prior is None else prior
    problem=cp.Problem(cp.Minimize(err_obj+lam*cp.sum_squares(cp.multiply(1/scales,x-ref))),cons)
    problem.solve(solver='CLARABEL',max_iter=200,tol_gap_abs=1e-9,tol_feas=1e-9,tol_gap_rel=1e-9)
    xx=x.value.copy(); cfg=coefficients_config(xx)
    item={'fit':len(fit_history)+1,'lambda':lam,'loss':loss,'arm_floor':np.asarray(arm_floor).tolist(),'status':problem.status,'objective':problem.value,'seconds':time.perf_counter()-started,'torque':torque_stats(physical_coefficients(cfg)),'config':cfg}
    fit_history.append(item)
    (work/f'fit-{len(fit_history):03d}.json').write_text(json.dumps(item,indent=2)+'\n')
    return cfg,xx,item

def brief(metrics):
    keys=['success','max_joint_torque_rmse_nm','max_joint_torque_normalized_rmse','max_joint_position_rmse_rad','max_joint_velocity_rmse_rad_s','position_p95_rad','max_joint_speed_rad_s']
    return {k:metrics.get(k) for k in ['frames','trace_path']+keys} | {'per_episode':[{k:e.get(k) for k in ['episode']+keys} for e in metrics['per_episode']]}

def evaluate(cfg,label):
    if len(eval_history)>=60: raise RuntimeError('Candidate budget reached')
    session.scenario.apply_config(cfg)
    session.dispatch('reset')
    session.dispatch('step',{'count':1800})
    metrics=session.scenario.metrics()
    eval_history.append({'label':label,**metrics})
    (work/f'evaluation-{len(eval_history):03d}.json').write_text(json.dumps(eval_history[-1],indent=2)+'\n')
    return brief(metrics)
