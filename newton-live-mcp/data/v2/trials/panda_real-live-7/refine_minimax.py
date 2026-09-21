from refine_forward import *
import cvxpy as cp
base=json.loads(Path('fit-014.json').read_text());d=np.load('refine1-sensitivity.npz');J=d['J'];r=d['residual'];z=np.array([base[k][j] for k,j in pairs])
trust=np.array([.08]*3+[.05]*3+[.05]*3+[.012]*3)
lo=np.maximum(-trust,np.r_[np.zeros(6),-np.ones(3)*2,np.ones(3)*.01]-z)
hi=np.minimum(trust,np.r_[np.ones(6)*5,np.ones(3)*2,np.ones(3)*.08]-z)
u=cp.Variable(12); t=cp.Variable(); constraints=[u>=lo,u<=hi];pred=r+J@u
# Each constraint is one joint and recording, normalized to its acceptance limit.
for start,N,weight in [(0,450,.6),(3150,1800,.3),(15750,1800,.1)]:
    for ep in range(3):
        for joint in range(7):
            indices=start+np.arange(ep*N//3,(ep+1)*N//3)*7+joint
            constraints += [cp.norm(pred[indices],2)*np.sqrt(3/weight)<=t]
prob=cp.Problem(cp.Minimize(t+.002*cp.sum_squares(cp.multiply(1/trust,u))),constraints)
prob.solve(solver='CLARABEL',tol_gap_abs=1e-9,tol_feas=1e-9)
print('predicted max ratio',t.value,'delta',u.value)
paths=[]
for i,alpha in enumerate([.5,1.,1.5]):
    cfg=copy.deepcopy(base)
    for (k,j),delta in zip(pairs,u.value):cfg[k][j]+=float(alpha*delta)
    validate_config(cfg)
    path=f'minimax-update-{i}.json';Path(path).write_text(json.dumps(cfg,indent=2)+'\n');paths.append(path)
Path('minimax-paths.json').write_text(json.dumps(paths))
