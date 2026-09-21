"""Physical SDP identification using only the supplied measured regressor."""
import json
import time
from pathlib import Path
import numpy as np
import scipy.linalg as la
import cvxpy as cp
from threadpoolctl import threadpool_limits

WORK = Path('/home/horde/artifacts/newton-live-mcp-v2/confirmation/panda_real-ipython-2')
REG = np.load(WORK / 'training-regressor.npz')
A, b = REG['A'], REG['b']
EIDS = REG['sample_episode_ids']
SCALE = np.r_[np.tile([3., .3, .3, .3, .1, .1, .1, .1, .1, .1],7), np.ones(21), np.ones(7)*.2]

def decode(theta):
    out = {k: [] for k in ('mass','com','inertia')}
    for j in range(7):
        m, hx, hy, hz, xx, yy, zz, xy, xz, yz = theta[10*j:10*j+10]
        c = np.array([hx,hy,hz])/m
        origin = np.array([[xx,xy,xz],[xy,yy,yz],[xz,yz,zz]])
        ic = origin - m*(c@c*np.eye(3)-np.outer(c,c))
        out['mass'].append(float(m))
        out['com'].append(c.tolist())
        out['inertia'].append(((ic+ic.T)/2).tolist())
    for i,key in enumerate(('viscous','coulomb','torque_bias','armature')):
        out[key] = theta[70+i*7:77+i*7].tolist()
    return out

def encode(config):
    result=[]
    for m,c,ic in zip(config['mass'],config['com'],config['inertia']):
        c=np.asarray(c); io=np.asarray(ic)+m*(c@c*np.eye(3)-np.outer(c,c))
        result.extend([m,*(m*c),io[0,0],io[1,1],io[2,2],io[0,1],io[0,2],io[1,2]])
    for key in ('viscous','coulomb','torque_bias','armature'):
        result.extend(config[key])
    return np.array(result)

def fit(alpha=1e-5, weights=None, mask=None, target=None, robust=None, armature_floor=0., prior=None):
    """Fit origin spatial moments with a positive pseudo-inertia for each link."""
    if mask is None: mask=np.ones(len(b),bool)
    if weights is None: weights=np.ones(len(b))
    if target is None: target=b
    if prior is None:
        prior=np.zeros(98)
        for j in range(7): prior[10*j]=1.; prior[10*j+4:10*j+7]=.01
    z=cp.Variable(98)
    theta=cp.multiply(z,SCALE)
    cons=[]
    for j in range(7):
        t=theta[j*10:(j+1)*10]
        m=t[0]; h=t[1:4]
        io=cp.bmat([[t[4],t[7],t[8]],[t[7],t[5],t[9]],[t[8],t[9],t[6]]])
        second=.5*cp.trace(io)*np.eye(3)-io
        pseudo=cp.bmat([[second,cp.reshape(h,(3,1),order='C')],[cp.reshape(h,(1,3),order='C'),cp.reshape(m,(1,1),order='C')]])
        cons.extend([m>=.050001,m<=9.999999,h>=-.399999*m,h<=.399999*m,
                     pseudo >> 1e-6*np.eye(4),cp.trace(second)<=.249999*m])
    cons.extend([theta[70:84]>=1e-8,theta[70:84]<=4.999999,
                 theta[84:91]>=-1.999999,theta[84:91]<=1.999999,
                 theta[91:98]>=armature_floor+1e-8,theta[91:98]<=.999999])
    aw=A[mask]*weights[mask,None]; bw=target[mask]*weights[mask]
    if robust is None:
        # QR removes thousands of residual variables without changing the optimum.
        with threadpool_limits(limits=1):
            q,r=la.qr(aw*SCALE,mode='economic')
            y=q.T@bw
        loss=cp.sum_squares(r@z-y)/sum(mask)*7
    else:
        loss=cp.sum(cp.huber(aw@theta-bw,robust))/sum(mask)*7
    problem=cp.Problem(cp.Minimize(loss+alpha*cp.sum_squares(z-prior/SCALE)),cons)
    with threadpool_limits(limits=1):
        problem.solve(solver='CLARABEL',tol_gap_abs=1e-9,tol_feas=1e-10,tol_gap_rel=1e-9,max_iter=200)
    if theta.value is None: raise RuntimeError(problem.status)
    v=np.array(theta.value)
    return decode(v), {'status':problem.status,'objective':float(problem.value),'alpha':alpha,
                      'torque_rmse':np.sqrt(np.mean((A@v-b).reshape(-1,7)**2,axis=0)).tolist()}

def brief(metrics):
    keys=['success','max_joint_torque_rmse_nm','max_joint_torque_normalized_rmse','max_joint_position_rmse_rad',
          'max_joint_velocity_rmse_rad_s','position_p95_rad','max_joint_speed_rad_s']
    return {k:metrics[k] for k in keys} | {'per_episode':[{k:e[k] for k in ['episode']+keys} for e in metrics['per_episode']],
                                        'trace_path':metrics['trace_path']}

def evaluate(session, config, label):
    session.scenario.validate_config(config)
    before=time.perf_counter()
    session.scenario.apply_config(config)
    session.dispatch('reset')
    session.dispatch('step',{'count':1800})
    met=session.scenario.metrics()
    with (WORK/'identification-candidates.jsonl').open('a') as f:
        f.write(json.dumps({'label':label,'config':config,'metrics':met})+'\n')
    print(label,'seconds',round(time.perf_counter()-before,3),json.dumps(brief(met)))
    return met

def motion_residual(metrics):
    trace=np.load(metrics['trace_path'])
    # Fixed measured traces remain untouched. Scale by sample count and thresholds.
    return np.r_[((trace['qd']-trace['reference_qd'])/.5/np.sqrt(len(trace['qd']))).ravel(),
                 ((trace['q']-trace['reference_q'])/.025/np.sqrt(2*len(trace['q']))).ravel()]

def torque_residual(config):
    error=(A@encode(config)-b).reshape(-1,7)
    scales=np.empty_like(error)
    for ep in np.unique(EIDS):
        sel=EIDS==ep
        scales[sel]=.5*np.clip(b.reshape(-1,7)[sel].std(axis=0),.5,1)
    return (error/scales/np.sqrt(len(error))).ravel()
