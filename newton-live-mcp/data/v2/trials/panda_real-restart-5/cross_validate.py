import subprocess, json
from pathlib import Path
import numpy as np
results=[]
for reg in [0.0,1e-5,1e-4,1e-3,1e-2]:
    for ep in [2,3,4]:
        name=f'cv-reg{reg:g}-ep{ep}.json'
        with open(name+'.log','w') as out:
            subprocess.run(['python','fit_physical.py','--reg',str(reg),'--exclude',str(ep),'--out',name],stdout=out,stderr=subprocess.STDOUT,check=True)
        report=json.loads(Path(name+'.report.json').read_text())
        val=next(e for e in report['per_episode'] if e['episode']==ep)
        results.append({'reg':reg,'ep':ep,**val})
        print(json.dumps(results[-1]),flush=True)
Path('cross-validation.json').write_text(json.dumps(results,indent=2)+'\n')
