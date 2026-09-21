import subprocess,json
from pathlib import Path
specs=[(4,.01),(5,.015),(6,.02),(7,.04),(8,.05)]
for i,arm in specs:
    fit=f'fit-{i:03d}.json'
    with open(fit+'.log','w') as log:
        subprocess.run(['python','fit_physical.py','--armature-min',str(arm),'--out',fit],stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run(['python','run_candidate.py',fit,str(i)],check=True)
