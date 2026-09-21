from scipy.optimize import lsq_linear

loss_base=json.loads((root_dir/'refined_a5_0.03.json').read_text())
loss_axes=[(key,j) for j in [4,6] for key in ['viscous','coulomb','torque_bias','armature']]
loss_vector=np.array([loss_base[key][j] for key,j in loss_axes])
loss_data=np.load(root_dir/'training-regressor.npz')
loss_A=loss_data['A'];loss_b=loss_data['b'];loss_ep=loss_data['sample_episode_ids']
loss_scale=np.empty((450,7))
for e in np.unique(loss_ep):
    sel=loss_ep==e
    loss_scale[sel]=.5*np.minimum(1,np.maximum(.5,loss_b.reshape(-1,7)[sel].std(0)))

def pack_physical(cfg):
    values=[]
    for mass,c,Ic in zip(cfg['mass'],cfg['com'],cfg['inertia']):
        c=np.array(c);Io=np.array(Ic)+mass*(np.dot(c,c)*np.eye(3)-np.outer(c,c))
        values.extend([mass,*(mass*c),Io[0,0],Io[1,1],Io[2,2],Io[0,1],Io[0,2],Io[1,2]])
    for key in ['viscous','coulomb','torque_bias','armature']:values.extend(cfg[key])
    return np.array(values)

def loss_config(p):
    cfg=copy.deepcopy(loss_base)
    for val,(key,j) in zip(p,loss_axes):cfg[key][j]=float(val)
    return cfg

def loss_eval(p,name):
    cfg=loss_config(p)
    (root_dir/(name+'.json')).write_text(json.dumps(cfg,indent=2)+'\n')
    summary=evaluate_candidate(cfg,name)
    trace=np.load(root_dir/summary['trace'])
    rt=(loss_A@pack_physical(cfg)-loss_b).reshape(-1,7)/loss_scale/np.sqrt(450)
    rv=(trace['qd']-trace['reference_qd'])/.5/np.sqrt(1800)
    rp=(trace['q']-trace['reference_q'])/.025/np.sqrt(1800)*.5
    residual=np.r_[rt.ravel(),rv.ravel(),rp.ravel()]
    summary['objective']=float(residual@residual)
    return residual,summary

loss_r0,loss_s0=loss_eval(loss_vector,'loss_base')
loss_fd=np.tile([.005,.005,.002,.001],2)
loss_J=np.zeros((len(loss_r0),8))
loss_fd_results=[]
for j,step in enumerate(loss_fd):
    trial=loss_vector.copy();trial[j]+=step
    residual,summary=loss_eval(trial,f'loss_fd_{j}')
    loss_J[:,j]=(residual-loss_r0)/step
    loss_fd_results.append(summary)
loss_lb=np.array([0,0,-2,.008]*2)-loss_vector
loss_ub=np.array([5,5,2,.055]*2)-loss_vector
loss_trust=np.tile([.075,.075,.035,.012],2)
loss_delta=lsq_linear(loss_J,-loss_r0,bounds=(np.maximum(loss_lb,-loss_trust),np.minimum(loss_ub,loss_trust)),tol=1e-10).x
loss_trials=[]
for scale in [1,.5]:
    residual,summary=loss_eval(loss_vector+scale*loss_delta,f'loss_step_{scale}')
    loss_trials.append(summary)
result={'base':loss_s0,'axes':loss_axes,'initial':loss_vector.tolist(),'delta':loss_delta.tolist(),'trials':loss_trials}
(root_dir/'loss_refinement.json').write_text(json.dumps(result,indent=2)+'\n')
