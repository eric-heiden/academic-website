from fit_physical import *
p,c=fit(lam=1e-4,arm_min=np.array([.03,.01,.03,.01,.027,.015,.02]),robust=True)
(BASE/'fit-refine.json').write_text(json.dumps(c,indent=2)+'\n'); np.save(BASE/'fit-refine-coeff.npy',p)
