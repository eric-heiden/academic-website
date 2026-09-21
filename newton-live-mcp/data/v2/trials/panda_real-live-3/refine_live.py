"""Finite-difference refinement using only completed prescribed live evaluations."""
import scipy.optimize

regressor_data = np.load(workdir/'training-regressor.npz')
regressor_A = regressor_data['A']
regressor_b = regressor_data['b']
torque_scales = np.empty((450,7))
for ep in np.unique(regressor_data['sample_episode_ids']):
    selected=regressor_data['sample_episode_ids']==ep
    torque_scales[selected]=np.minimum(.5,.5*np.maximum(regressor_b.reshape(-1,7)[selected].std(axis=0),.5))

def residual_of(m):
    with np.load(m['trace_path']) as trace:
        torque=(trace['torque_prediction']-trace['torque_target'])/torque_scales/np.sqrt(450)
        position=(trace['q']-trace['reference_q'])/.025/np.sqrt(1800)
        velocity=(trace['qd']-trace['reference_qd'])/.5/np.sqrt(1800)
    return np.r_[torque.ravel(),position.ravel(),velocity.ravel()]

def propose_refinement(base_index, suffix):
    base_measurement=candidate_results[base_index-1]
    cfg=copy.deepcopy(base_measurement['config'])
    r0=residual_of(base_measurement)
    parameters=[(k,j) for k in ['viscous','coulomb','torque_bias','armature'] for j in [4,5,6]]
    probe_sizes={'viscous':.015,'coulomb':.015,'torque_bias':.01,'armature':.003}
    step_sizes={'viscous':.08,'coulomb':.08,'torque_bias':.04,'armature':.02}
    derivatives=[]
    for key,joint in parameters:
        perturbed=copy.deepcopy(cfg)
        perturbed[key][joint]+=probe_sizes[key]
        evaluate_candidate(perturbed,f'{suffix}: derivative {key}[{joint}]')
        derivatives.append((residual_of(candidate_results[-1])-r0)/probe_sizes[key]*step_sizes[key])
    jac=np.stack(derivatives,axis=1)
    lower=[];upper=[]
    for key,joint in parameters:
        lo,hi={'viscous':(0,5),'coulomb':(0,5),'torque_bias':(-2,2),'armature':(.008,.12)}[key]
        lower.append(max(-1,(lo-cfg[key][joint])/step_sizes[key]))
        upper.append(min(1,(hi-cfg[key][joint])/step_sizes[key]))
    solved=scipy.optimize.lsq_linear(np.r_[jac,np.eye(len(parameters))*.08],np.r_[-r0,np.zeros(len(parameters))],bounds=(lower,upper),tol=1e-9)
    candidates=[]
    for fraction in [1.0,.5]:
        proposed=copy.deepcopy(cfg)
        for value,(key,joint) in zip(solved.x,parameters):
            proposed[key][joint]+=fraction*value*step_sizes[key]
        summary=evaluate_candidate(proposed,f'{suffix}: step {fraction}')
        summary['objective']=float(residual_of(candidate_results[-1])@residual_of(candidate_results[-1]))
        candidates.append(summary)
    info={'base':base_index,'base_objective':float(r0@r0),'parameters':parameters,'step':solved.x.tolist(),'proposals':candidates}
    (workdir/f'refinement_{suffix}.json').write_text(json.dumps(info,indent=2)+'\n')
    return info
