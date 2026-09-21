from fit_physical import *
settings=[
[.01,.005,.01,.005,.02,.01,.015],
[.01,.005,.01,.005,.03,.01,.015],
[.01,.005,.01,.005,.04,.01,.015],
[.01,.005,.01,.005,.03,.01,.025],
[.02,.01,.02,.01,.03,.01,.02],
[.005,.005,.005,.005,.03,.005,.015],
]
for i,arm in enumerate(settings,7):
    cfg,rep=fit(arm_min=np.asarray(arm));
    Path(f'fit-{i:03d}.json').write_text(json.dumps(cfg,indent=2)+'\n')
    print(json.dumps({'fit':i,'arm':arm,'rmse':rep['rmse']}))
