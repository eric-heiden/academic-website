"""Measured-data identification with convex physical inertia constraints."""
import json
import time
from pathlib import Path
import numpy as np
import cvxpy as cp

ROOT = Path(__file__).resolve().parent
R = np.load(ROOT / 'training-regressor.npz')
A, b = R['A'], R['b']
episodes = R['sample_episode_ids']
rows_episode = np.repeat(episodes, 7)
scales = np.tile([3., .5, .5, .5, .2, .2, .2, .15, .15, .15], 7)
scales = np.r_[scales, np.full(7, 2.), np.full(7, 2.), np.full(7, .5), np.full(7, .2)]

def to_config(p):
    cfg = {k: [] for k in ['mass', 'com', 'inertia']}
    for j in range(7):
        z = p[10*j:10*j+10]
        m, h = z[0], z[1:4]
        I = np.array([[z[4],z[7],z[8]],[z[7],z[5],z[9]],[z[8],z[9],z[6]]])
        c = h/m
        Ic = I - m*(np.dot(c,c)*np.eye(3)-np.outer(c,c))
        cfg['mass'].append(float(m))
        cfg['com'].append(c.tolist())
        cfg['inertia'].append(((Ic+Ic.T)/2).tolist())
    for key, offset, low, high in [('viscous',70,0,5),('coulomb',77,0,5),('torque_bias',84,-2,2),('armature',91,0,1)]:
        cfg[key] = np.clip(p[offset:offset+7],low,high).tolist()
    return cfg

def torque_report(p):
    e = (A@p-b).reshape(-1,7)
    y = b.reshape(-1,7)
    return {str(ep): {'rmse':np.sqrt(np.mean(e[episodes==ep]**2,axis=0)).tolist(), 'nrmse':(np.sqrt(np.mean(e[episodes==ep]**2,axis=0))/np.maximum(y[episodes==ep].std(axis=0),.5)).tolist()} for ep in np.unique(episodes)}

def fit(lam=1e-5, excluded=None, weights=None, anchor=None, robust=None, armature=None, min_armature=None, inertia_floor=1e-6):
    start = time.monotonic()
    p = cp.Variable(98)
    constraints = []
    for j in range(7):
        z = p[10*j:10*j+10]
        m, h = z[0], cp.reshape(z[1:4],(3,1),order='C')
        I = cp.bmat([[z[4],z[7],z[8]],[z[7],z[5],z[9]],[z[8],z[9],z[6]]])
        S = .5*cp.trace(I)*np.eye(3)-I
        J = cp.bmat([[S,h],[h.T,cp.reshape(m,(1,1),order='C')]])
        constraints += [J-np.diag([inertia_floor,inertia_floor,inertia_floor,0.]) >> 0, m>=.050001, m<=9.999999,
                        z[1:4]>=-.399999*m,z[1:4]<=.399999*m,cp.trace(S)<=.249999*m]
    constraints += [p[70:84]>=0,p[70:84]<=5,p[84:91]>=-2,p[84:91]<=2,p[91:]>=0,p[91:]<=1]
    if armature is not None: constraints += [p[91:] == np.asarray(armature)]
    if min_armature is not None: constraints += [p[91:] >= np.asarray(min_armature)]
    sel = np.ones(len(b),dtype=bool) if excluded is None else rows_episode!=excluded
    w = np.ones(len(b)) if weights is None else np.asarray(weights)
    Aw, bw = A[sel]*w[sel,None], b[sel]*w[sel]
    if robust is None:
        Q, T = np.linalg.qr(np.c_[Aw,bw],mode='reduced')
        loss = cp.sum_squares(T[:,:98]@p-T[:,98])/sel.sum()
    else:
        loss = cp.sum(cp.huber(Aw@p-bw,robust))/sel.sum()
    anchor = np.zeros(98) if anchor is None else anchor
    obj = loss + lam*cp.sum_squares(cp.multiply(1/scales,p-anchor))
    problem = cp.Problem(cp.Minimize(obj),constraints)
    problem.solve(solver='CLARABEL',max_iter=200,tol_gap_abs=1e-9,tol_feas=1e-9,tol_gap_rel=1e-9)
    if p.value is None: raise RuntimeError(problem.status)
    result = {'lam':lam,'excluded':excluded,'status':problem.status,'objective':problem.value,'seconds':time.monotonic()-start,'torque':torque_report(p.value)}
    return p.value, to_config(p.value), result

if __name__ == '__main__':
    p, cfg, report = fit()
    (ROOT/'fit-001.json').write_text(json.dumps(cfg,indent=2)+'\n')
    (ROOT/'fit-001-report.json').write_text(json.dumps(report,indent=2)+'\n')
    np.save(ROOT/'fit-001-coeff.npy',p)
    print(json.dumps(report))
    print('mass',cfg['mass'],'viscous',cfg['viscous'],'coulomb',cfg['coulomb'],'bias',cfg['torque_bias'],'armature',cfg['armature'])
