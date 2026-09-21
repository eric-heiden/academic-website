from fit_physical import *
settings=[(np.array([.025,.01,.025,.01,.03,.02,.02]),1e-6,1e-5,False),
(np.array([.025,.01,.025,.01,.03,.02,.02]),1e-4,1e-5,False),
(np.array([.025,.01,.025,.01,.03,.02,.02]),1e-3,1e-5,False),
(np.array([.025,.01,.025,.01,.03,.02,.02]),1e-4,1e-4,False),
(np.array([.025,.01,.025,.01,.03,.02,.02]),1e-4,1e-5,True)]
for i,(arm,margin,ridge,robust) in enumerate(settings,13):
    cfg,rep=fit(arm_min=arm,inertia_margin=margin,ridge=ridge,robust=robust)
    Path(f'fit-{i:03d}.json').write_text(json.dumps(cfg,indent=2)+'\n')
    print(json.dumps({'fit':i,'margin':margin,'ridge':ridge,'robust':robust,'rmse':rep['rmse']}))
