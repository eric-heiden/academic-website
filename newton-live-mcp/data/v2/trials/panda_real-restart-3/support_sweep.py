import subprocess,json
base=['uv','run','--no-sync','--project','/home/horde/apps/newton-live-mcp','python']
for support in [.06,.09]:
 name=f'support-{support}';arms=[support,.03,support,.03,.02,.03,.015]
 with open(name+'.log','w') as log:subprocess.run(base+['fit_model.py','--arm-vector',json.dumps(arms),'--huber','0.1','--output',name+'.json'],stdout=log,stderr=subprocess.STDOUT,check=True)
 subprocess.run(base+['run_candidate.py',name+'.json','--note',f'Proximal armature support {support}; distal floors .02,.03,.015'],check=True)
