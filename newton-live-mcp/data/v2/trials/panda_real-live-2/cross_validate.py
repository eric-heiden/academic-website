from fit_model import *

settings=[]
for ridge in [1e-6,1e-5,1e-4,1e-3]:
    for power in [0,.5]:
        settings.append(dict(ridge=ridge,weight_power=power,arm_min=np.array([.02,.01,.02,.01,.035,.02,.015])))
for eig in [.00025,.001,.003]:
    settings.append(dict(ridge=1e-4,weight_power=.5,arm_min=np.array([.02,.01,.02,.01,.03,.02,.015]),eig_min=eig))
records=[]
for i,kw in enumerate(settings):
    cv=[]
    for ep in [2,3,4]:
        c,r=fit(**kw,exclude=ep)
        cv.append(r['per_episode'][str(ep)])
    c,r=fit(**kw)
    name='cvfit'+str(i)
    (ROOT/(name+'.json')).write_text(json.dumps(c,indent=2)+'\n')
    kw={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in kw.items()}
    record={'name':name,'settings':kw,'cv':cv,'train':r}
    records.append(record)
    print(name,kw,'crossval',np.round(np.sqrt(np.mean(np.array(cv)**2,axis=0)),4),'train',np.round(r['rmse'],4),flush=True)
(ROOT/'cross-validation.json').write_text(json.dumps(records,indent=2)+'\n')
