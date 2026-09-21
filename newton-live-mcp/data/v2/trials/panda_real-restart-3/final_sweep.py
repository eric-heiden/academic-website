import subprocess,json
from pathlib import Path
cases=[('robust-select-a',[.03,.02,.03,.02,.025,.025,.015],.1),('robust-select-b',[.03,.03,.03,.03,.03,.03,.015],.1),('robust-select-c',[.03,.02,.03,.02,.025,.025,.015],.05),('robust-select-d',[.04,.03,.04,.03,.025,.03,.015],.1)]
base=['uv','run','--no-sync','--project','/home/horde/apps/newton-live-mcp','python']
for name,arms,huber in cases:
 with open(name+'.log','w') as log:
  subprocess.run(base+['fit_model.py','--arm-vector',json.dumps(arms),'--huber',str(huber),'--output',name+'.json'],stdout=log,stderr=subprocess.STDOUT,check=True)
 subprocess.run(base+['run_candidate.py',name+'.json','--note',name],check=True)
