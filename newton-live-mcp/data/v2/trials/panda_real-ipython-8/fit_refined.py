from identify import *
results=[]
arm=np.array([.02,.005,.02,.005,.03,.01,.015])
weights=np.tile([1,1,1,1,1.5,1,2],len(b)//7)
for lam,robust,floor in [(1e-5,None,1e-6),(1e-4,None,1e-6),(1e-5,.2,1e-6),(1e-5,.1,1e-6),(1e-5,None,1e-4),(1e-4,None,1e-4)]:
    p,cfg,rep=fit(lam=lam,armature=arm,weights=weights,robust=robust,inertia_floor=floor)
    name=f'fit-refined-{len(results)+1:02d}'
    (ROOT/f'{name}.json').write_text(json.dumps(cfg,indent=2)+'\n')
    results.append({'name':name,'robust':robust,'floor':floor,'report':rep})
    print(name,'torque',np.round(np.sqrt(np.mean((A@p-b).reshape(-1,7)**2,axis=0)),4).tolist(),flush=True)
(ROOT/'refined-reports.json').write_text(json.dumps(results,indent=2)+'\n')
