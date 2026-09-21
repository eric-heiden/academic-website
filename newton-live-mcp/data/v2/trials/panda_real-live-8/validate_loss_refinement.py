cv_loss_results=[]
for excluded in [2,3,4]:
    loss_base=json.loads((root_dir/f'refined_a5_0.03_leave_{excluded}.json').read_text())
    loss_axes=[(key,4) for key in ['viscous','coulomb','torque_bias','armature']]
    loss_vector=np.array([loss_base[key][j] for key,j in loss_axes])
    cv_r0,cv_s0=loss_eval(loss_vector,f'cv_loss_{excluded}_base')
    cv_J=np.zeros((len(cv_r0),4))
    for j,step in enumerate([.005,.005,.002,.001]):
        trial=loss_vector.copy();trial[j]+=step
        residual,summary=loss_eval(trial,f'cv_loss_{excluded}_fd_{j}')
        cv_J[:,j]=(residual-cv_r0)/step
    cv_d=cp.Variable(4);cv_t=cp.Variable()
    cv_trust=np.array([.075,.075,.035,.012])
    cv_lb=np.array([0,0,-2,.008])-loss_vector
    cv_ub=np.array([5,5,2,.055])-loss_vector
    cv_cons=[cv_d>=np.maximum(cv_lb,-cv_trust),cv_d<=np.minimum(cv_ub,cv_trust)]
    for block,start,n,fac in [('torque',0,450,1),('velocity',3150,1800,1),('position',15750,1800,2)]:
        episode_rows=loss_ep if block=='torque' else np.repeat([2,3,4],600)
        for e in [2,3,4]:
            if e==excluded:continue
            rows=np.where(episode_rows==e)[0]
            for j in range(7):
                ids=start+7*rows+j
                cv_cons.append(cp.norm((cv_r0[ids]+cv_J[ids]@cv_d)*np.sqrt(3)*fac)<=cv_t)
    cv_prob=cp.Problem(cp.Minimize(cv_t+.002*cp.sum_squares(cp.multiply(1/cv_trust,cv_d))),cv_cons)
    cv_prob.solve(solver='CLARABEL')
    cv_delta=cv_d.value
    residual,summary=loss_eval(loss_vector+cv_delta,f'cv_loss_{excluded}_refined')
    cv_metrics=session.scenario.metrics()
    held=next(e for e in cv_metrics['per_episode'] if e['episode']==excluded)
    cv_loss_results.append({'excluded':excluded,'delta':cv_delta.tolist(),'initial':loss_vector.tolist(),'final':(loss_vector+cv_delta).tolist(),'held':held,'trace':summary['trace']})
(root_dir/'forward_crossvalidation.json').write_text(json.dumps(cv_loss_results,indent=2)+'\n')
result=cv_loss_results
