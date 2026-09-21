from fit_model import *
from candidate_runner import run
for arm in [.015,.025,.04,.055]:
 x=cp.Variable(98)
 w=np.ones((len(b)//7,7))
 for ep in np.unique(episodes):
  mask=episodes==ep
  w[mask]=1/np.minimum(np.maximum(b.reshape(-1,7)[mask].std(axis=0),.5),1)
 cons=constraints(x)+[x[91:98]>=arm]
 scale=np.tile([.2,1,1,1,3,3,3,4,4,4],7).tolist()+[.2]*21+[1]*7
 loss=cp.sum_squares(cp.multiply(w.ravel(),A@x-b))/450+1e-4*cp.sum_squares(cp.multiply(scale,x))
 p=cp.Problem(cp.Minimize(loss),cons)
 p.solve(solver='CLARABEL',tol_gap_abs=1e-9,tol_feas=1e-9,tol_gap_rel=1e-9,max_iter=300)
 conf=to_config(x.value)
 Path(f'refit-arm-{arm}.json').write_text(json.dumps(conf,indent=2)+'\n')
 print('offline',arm,json.dumps(score(x.value)),flush=True)
 run(conf,f'refit lower armature {arm}')
