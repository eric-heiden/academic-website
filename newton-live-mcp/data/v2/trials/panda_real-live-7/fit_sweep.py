from fit_physical import *
reports=[]
for i,arm in enumerate([.01,.025,.05,.075,.1],2):
    cfg,rep=fit(arm_min=arm)
    rep.update(name=f'fit-{i:03d}',arm_min=arm)
    Path(f'fit-{i:03d}.json').write_text(json.dumps(cfg,indent=2)+'\n');reports.append(rep)
    print(json.dumps(rep),flush=True)
Path('sweep-report.json').write_text(json.dumps(reports,indent=2)+'\n')
