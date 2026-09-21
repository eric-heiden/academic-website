# This helper evaluates every physical candidate only through the prescribed live workflow.
import copy
import cvxpy as cp
from tools.mcp_evaluation.real_robot_model import physical_coefficients

fit_data=np.load(work/'training-regressor.npz')
fit_A,fit_b=fit_data['A'],fit_data['b']
fit_eps=fit_data['sample_episode_ids']
param_keys=[('armature',4),('viscous',4),('coulomb',4),('torque_bias',4),('armature',6),('viscous',6),('coulomb',6),('torque_bias',6)]
param_steps=np.array([.002,.015,.015,.01,.002,.015,.015,.01])
param_trust=np.array([.012,.06,.06,.035,.01,.06,.06,.035])

def values_of(c):
    return np.array([c[k][j] for k,j in param_keys])

def config_of(c,v):
    new=copy.deepcopy(c)
    for (k,j),u in zip(param_keys,v): new[k][j]=float(u)
    return new

def compact_measurement(m):
    worst={k:max(e[k] for e in m['per_episode']) for k in m['thresholds']}
    return dict(candidate=int(Path(m['trace_path']).stem.split('-')[-1]),success=m['success'],worst=worst,ratio=max(worst[k]/m['thresholds'][k] for k in worst),trace=m['trace_path'])

def measure_cfg(c,label):
    session.scenario.apply_config(c)
    session.dispatch('reset')
    session.dispatch('step',{'count':1800})
    m=session.scenario.metrics()
    idx=int(Path(m['trace_path']).stem.split('-')[-1])
    (work/f'measurement-{idx:03d}.json').write_text(json.dumps(m,indent=2)+'\n')
    (work/f'candidate-config-{idx:03d}.json').write_text(json.dumps(c,indent=2)+'\n')
    with (work/'refinement-log.jsonl').open('a') as stream: stream.write(json.dumps({'label':label,**compact_measurement(m)})+'\n')
    with np.load(m['trace_path']) as trace:
        e_q=trace['q']-trace['reference_q']; e_qd=trace['qd']-trace['reference_qd']
    return m,e_q,e_qd

def refine_step(c,base_result,label):
    m,eq,ev=base_result
    v=values_of(c)
    derivatives_q=[]; derivatives_v=[]; derivatives_tau=[]
    tau0=(fit_A@physical_coefficients(c)-fit_b).reshape(-1,7)
    diagnostics=[]
    for i,step in enumerate(param_steps):
        perturb=v.copy(); perturb[i]+=step
        ci=config_of(c,perturb)
        mi,qi,vi=measure_cfg(ci,f'{label}-derivative-{i}')
        derivatives_q.append((qi-eq)/step)
        derivatives_v.append((vi-ev)/step)
        derivatives_tau.append(((fit_A@physical_coefficients(ci)-fit_b).reshape(-1,7)-tau0)/step)
        diagnostics.append(compact_measurement(mi))
    Jq=np.stack(derivatives_q,axis=-1); Jv=np.stack(derivatives_v,axis=-1); Jt=np.stack(derivatives_tau,axis=-1)
    d=cp.Variable(8); t=cp.Variable()
    delta=cp.multiply(param_trust,d)
    cons=[d>=-1,d<=1]
    for i,(k,j) in enumerate(param_keys):
        lo,hi=(-2,2) if k=='torque_bias' else ((.005,.08) if k=='armature' else (0,5))
        cons += [v[i]+delta[i]>=lo,v[i]+delta[i]<=hi]
    objective_terms=[]
    for ep in range(3):
        tm=slice(ep*150,(ep+1)*150); fm=slice(ep*600,(ep+1)*600)
        sd=np.maximum(fit_b.reshape(-1,7)[tm].std(axis=0),.5)
        for j in range(7):
            tr=(tau0[tm,j]+Jt[tm,j]@delta)/(min(.5,.5*sd[j])*np.sqrt(150))
            qr=(eq[fm,j]+Jq[fm,j]@delta)/(.025*np.sqrt(600))
            vr=(ev[fm,j]+Jv[fm,j]@delta)/(.5*np.sqrt(600))
            cons += [cp.norm(tr)<=t,cp.norm(qr)<=t,cp.norm(vr)<=t]
            objective_terms += [cp.sum_squares(tr),.5*cp.sum_squares(qr),.5*cp.sum_squares(vr)]
    problem=cp.Problem(cp.Minimize(t+.004*cp.sum(objective_terms)+.003*cp.sum_squares(d)),cons)
    problem.solve(solver='CLARABEL',max_iter=100,tol_gap_abs=1e-8,tol_feas=1e-8)
    proposal=config_of(c,v+param_trust*d.value)
    (work/f'refinement-{label}-proposal.json').write_text(json.dumps(proposal,indent=2)+'\n')
    actual=measure_cfg(proposal,f'{label}-proposal')
    report=dict(label=label,predicted_ratio=float(t.value),old=compact_measurement(m),new=compact_measurement(actual[0]),old_parameters=v.tolist(),new_parameters=values_of(proposal).tolist(),status=problem.status)
    (work/f'refinement-{label}-report.json').write_text(json.dumps(report,indent=2)+'\n')
    return proposal,actual,report
