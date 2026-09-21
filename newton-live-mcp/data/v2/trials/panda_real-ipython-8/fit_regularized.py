from identify import *
results=[]
for arm in [.005,.01,.015,.02,.03,.04]:
    for wj in [np.ones(7),np.array([1,1,1,1,1.5,1,2])]:
        weights=np.tile(wj,len(b)//7)
        p,cfg,rep=fit(armature=np.full(7,arm),weights=weights)
        name=f'fit-arm-{arm:g}-w{wj[-1]:g}'
        (ROOT/f'{name}.json').write_text(json.dumps(cfg,indent=2)+'\n')
        results.append({'name':name,'report':rep})
        print(name, 'torque',np.round(np.sqrt(np.mean((A@p-b).reshape(-1,7)**2,axis=0)),4).tolist(),flush=True)
(ROOT/'regularized-reports.json').write_text(json.dumps(results,indent=2)+'\n')
