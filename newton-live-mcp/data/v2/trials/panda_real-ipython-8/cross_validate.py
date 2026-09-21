from identify import *
records = []
for lam in [0,1e-6,1e-5,1e-4,1e-3,1e-2]:
    for exclude in [2,3,4,None]:
        p,cfg,report = fit(lam=lam,excluded=exclude)
        record = {'lam':lam,'exclude':exclude,'report':report}
        records.append(record)
        if exclude is None:
            (ROOT/f'fit-lambda-{lam:g}.json').write_text(json.dumps(cfg,indent=2)+'\n')
        target = str(exclude) if exclude is not None else '4'
        print(lam,exclude, np.round(report['torque'][target]['rmse'],3).tolist(),flush=True)
(ROOT/'cross-validation.json').write_text(json.dumps(records,indent=2)+'\n')
