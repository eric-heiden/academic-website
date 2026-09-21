"""Tune six distal joint coefficients using only complete prescribed candidates.
Execute in the persistent IPython application namespace after identify.py fits.
"""
from scipy.optimize import least_squares
from tools.mcp_evaluation.real_robot_model import physical_coefficients

motion_base = json.loads((workdir/'fit-refined-02.json').read_text())
def unpack_motion(x):
    cfg = copy.deepcopy(motion_base)
    cfg['armature'][4],cfg['armature'][6] = float(x[0]),float(x[1])
    cfg['viscous'][4],cfg['coulomb'][4] = float(x[2]),float(x[3])
    cfg['viscous'][6],cfg['coulomb'][6] = float(x[4]),float(x[5])
    return cfg

torque_scale = np.ones((450,7))
for ep in np.unique(reg['sample_episode_ids']):
    mask = reg['sample_episode_ids']==ep
    torque_scale[mask] = np.minimum(.5,.5*np.maximum(reg['b'].reshape(-1,7)[mask].std(axis=0),.5))
motion_objectives = []
def motion_residual(x):
    cfg = unpack_motion(x)
    met = evaluate_candidate(cfg,'combined torque-motion fit')
    tr = np.load(met['trace_path'])
    et = (tr['torque_prediction']-tr['torque_target'])/torque_scale
    eq = (tr['q']-tr['reference_q'])/.025
    ev = (tr['qd']-tr['reference_qd'])/.5
    residual = np.r_[et.ravel()/np.sqrt(len(et)),eq.ravel()/np.sqrt(len(eq)),ev.ravel()/np.sqrt(len(ev))]
    motion_objectives.append({'candidate':candidate_count,'x':x.tolist(),'loss':float(residual@residual)})
    return residual

x0 = np.array([motion_base['armature'][4],motion_base['armature'][6],motion_base['viscous'][4],motion_base['coulomb'][4],motion_base['viscous'][6],motion_base['coulomb'][6]])
motion_opt = least_squares(motion_residual,x0,bounds=([.005,.005,.01,.05,0,.05],[.06,.04,.5,.6,.3,.4]),diff_step=.003,x_scale=[.03,.02,.15,.3,.08,.2],max_nfev=4,ftol=1e-5,xtol=1e-5,gtol=1e-5)
(workdir/'motion-optimization.json').write_text(json.dumps({'result':motion_opt.x.tolist(),'message':motion_opt.message,'history':motion_objectives},indent=2)+'\n')
(workdir/'fit-motion.json').write_text(json.dumps(unpack_motion(motion_opt.x),indent=2)+'\n')
print('motion optimum',motion_opt.x,'cost',motion_opt.cost,'nfev',motion_opt.nfev,'candidate count',candidate_count)
