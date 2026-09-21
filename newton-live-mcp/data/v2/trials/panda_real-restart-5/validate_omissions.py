import subprocess,json
from pathlib import Path
for i,ep in [(10,2),(11,3),(12,4)]:
    fit=f'fit-{i:03d}.json'
    with open(fit+'.log','w') as log:
        subprocess.run(['python','fit_physical.py','--armature-floor','[0.03,0.03,0.03,0.03,0.03,0.02,0.015]','--exclude',str(ep),'--out',fit],stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run(['python','run_candidate.py',fit,str(i)],check=True)
    m=json.loads(Path(f'candidate-{i:03d}/metrics.json').read_text())
    print('OMITTED',ep,json.dumps(next(e for e in m['per_episode'] if e['episode']==ep)),flush=True)
