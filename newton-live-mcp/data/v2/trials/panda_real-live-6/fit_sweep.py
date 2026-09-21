from fit_physical import *
records=[]
for label,amin,lam,robust in [
 ('arm015',.015,1e-5,False),('arm030',.03,1e-5,False),('arm050',.05,1e-5,False),
 ('mixed',np.array([.015,.005,.015,.005,.035,.015,.020]),1e-5,False),
 ('robust',np.array([.015,.005,.015,.005,.035,.015,.020]),1e-5,True),
 ('ridge',np.array([.015,.005,.015,.005,.035,.015,.020]),1e-3,False)]:
 p,c=fit(arm_min=amin,lam=lam,robust=robust)
 (BASE/f'fit-{label}.json').write_text(json.dumps(c,indent=2)+'\n'); np.save(BASE/f'fit-{label}-coeff.npy',p)
 records.append(dict(label=label,arm_min=np.asarray(amin).tolist(),lambda_reg=lam,robust=robust))
(BASE/'fit-sweep-settings.json').write_text(json.dumps(records,indent=2)+'\n')
