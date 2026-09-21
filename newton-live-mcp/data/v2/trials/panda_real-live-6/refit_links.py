from fit_physical import *
c=json.loads((BASE/'candidate-config-030.json').read_text())
fixed={}
for key,start in [('viscous',70),('coulomb',77),('torque_bias',84),('armature',91)]:
 for j in range(7): fixed[start+j]=c[key][j]
for label,w in [('refitlinks',np.array([1,1,1,1,1,1,2])),('refitweighted',np.array([1,1,1,1,2,1,2]))]:
 p,cf=fit(lam=1e-4,robust=True,fixed=fixed,weights=np.tile(w,450))
 (BASE/f'fit-{label}.json').write_text(json.dumps(cf,indent=2)+'\n')
