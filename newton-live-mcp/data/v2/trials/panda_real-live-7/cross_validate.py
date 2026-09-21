from fit_physical import *
reports=[]
for arm in [.01,.02,.025,.035,.05]:
    for ep in np.unique(episode):
        cfg,rep=fit(arm_min=arm,train=[int(e) for e in np.unique(episode) if e!=ep])
        r=(A@physical_coefficients(cfg)-b).reshape(-1,7)[episode==ep]
        rms=np.sqrt(np.mean(r*r,axis=0)); norms=rms/np.maximum(.5,b.reshape(-1,7)[episode==ep].std(axis=0))
        reports.append({'arm':arm,'episode':int(ep),'rmse':rms.tolist(),'normalized_rmse':norms.tolist()})
for arm in [.01,.02,.025,.035,.05]:
    rr=[r for r in reports if r['arm']==arm]
    print(json.dumps({'arm':arm,'max_nm':np.max([r['rmse'] for r in rr]),'max_normalized':np.max([r['normalized_rmse'] for r in rr]),'rms_per_joint':np.max([r['rmse'] for r in rr],axis=0).tolist()}))
Path('cross-validation.json').write_text(json.dumps(reports,indent=2)+'\n')
