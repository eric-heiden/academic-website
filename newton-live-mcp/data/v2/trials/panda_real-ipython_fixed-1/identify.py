import numpy as np
import cvxpy as cp
import json, time
from pathlib import Path
from tools.mcp_evaluation.real_robot_model import validate_config, physical_coefficients

work = Path('/home/horde/artifacts/newton-live-mcp-v2/confirmation/panda_real-ipython_fixed-1')
reg = np.load(work/'training-regressor.npz')
A, b = reg['A'], reg['b']
episode_ids = reg['sample_episode_ids']
unique_episodes = np.unique(episode_ids)
threshold_scale = np.empty((len(b)//7,7))
for ep in unique_episodes:
    mask = episode_ids == ep
    threshold_scale[mask] = np.minimum(0.5, 0.5*np.maximum(b.reshape(-1,7)[mask].std(axis=0),0.5))

def decode_coefficients(x):
    values = {key: [redacted] for key in ['mass','com','inertia']}
    for i in range(7):
        m = float(x[10*i]); h = x[10*i+1:10*i+4]
        xx,yy,zz,xy,xz,yz=x[10*i+4:10*i+10]
        origin = np.array([[xx,xy,xz],[xy,yy,yz],[xz,yz,zz]])
        c=h/m
        tensor = origin-m*(np.dot(c,c)*np.eye(3)-np.outer(c,c))
        values['mass'].append(m); values['com'].append(c.tolist()); values['inertia'].append(tensor.tolist())
    for key, start, lo, hi in [('viscous',70,0,5),('coulomb',77,0,5),('torque_bias',84,-2,2),('armature',91,0,1)]:
        values[key]=np.clip(x[start:start+7],lo,hi).tolist()
    return validate_config(values)


def fit_physical(ridge=1e-5, weights=None, mask=None, robust=False, armature_floor=0., extra_penalty=None, viscous_floor=0.):
    x=cp.Variable(98)
    cons=[]
    for i in range(7):
        m=x[10*i]; h=x[10*i+1:10*i+4]
        xx,yy,zz,xy,xz,yz=[x[10*i+k] for k in range(4,10)]
        I=cp.bmat([[xx,xy,xz],[xy,yy,yz],[xz,yz,zz]])
        S=.5*cp.trace(I)*np.eye(3)-I
        P=cp.bmat([[S,cp.reshape(h,(3,1),order='C')],[cp.reshape(h,(1,3),order='C'),cp.reshape(m,(1,1),order='C')]])
        cons += [m>=.050001,m<=9.999999,h>=-.399999*m,h<=.399999*m,P >> np.diag([1e-6,1e-6,1e-6,0]),cp.trace(S)<=.249999*m]
    cons += [x[70:77]>=viscous_floor+1e-8,x[70:84]>=1e-8,x[70:84]<=4.999999,x[84:91]>=-1.999999,x[84:91]<=1.999999,x[91:98]>=armature_floor+1e-8,x[91:98]<=.999999]
    if weights is None: weights = 1/threshold_scale.ravel()
    if mask is None: mask = np.ones(len(b),dtype=bool)
    design=A[mask]*weights[mask,None]; target=b[mask]*weights[mask]
    residual=design@x-target
    scales = np.tile([3.,.5,.5,.5,.15,.15,.15,.1,.1,.1],7).tolist()+[2.]*7+[2.]*7+[1.]*7+[.2]*7
    objective = (cp.sum(cp.huber(residual,1.0)) if robust else cp.sum_squares(residual))/len(target)
    objective += ridge*cp.sum_squares(cp.multiply(1/np.array(scales),x))
    if extra_penalty is not None: objective += extra_penalty(x)
    problem=cp.Problem(cp.Minimize(objective),cons)
    t=time.time()
    problem.solve(solver='CLARABEL',tol_gap_abs=1e-9,tol_feas=1e-9,tol_gap_rel=1e-9,max_iter=300)
    if x.value is None: raise RuntimeError(problem.status)
    cfg=decode_coefficients(x.value)
    err=(A@physical_coefficients(cfg)-b).reshape(-1,7)
    report={'status':problem.status,'objective':float(problem.value),'seconds':time.time()-t,'torque_rmse':np.sqrt(np.mean(err**2,axis=0)).tolist(),'episodes':{str(ep):np.sqrt(np.mean(err[episode_ids==ep]**2,axis=0)).tolist() for ep in unique_episodes}}
    return cfg, report

candidate_history=[]
def evaluate_candidate(config, label):
    index=len(candidate_history)+1
    (work/f'fit-{index:03d}-{label}.json').write_text(json.dumps(config,indent=2)+'\n')
    session.scenario.apply_config(config)
    session.dispatch('reset')
    step_response=session.dispatch('step',{'count':1800})
    metrics=session.scenario.metrics()
    candidate_history.append({'label':label,'config':config,'metrics':metrics})
    with (work/'identification-history.jsonl').open('a') as f: f.write(json.dumps(candidate_history[-1])+'\n')
    print(json.dumps({'candidate':index,'label':label, 'success':metrics['success'], 'trace_path':metrics['trace_path'], 'frames':metrics['frames'], **{k:metrics[k] for k in metrics['thresholds']}, 'per_episode':metrics['per_episode']}))
    return metrics
