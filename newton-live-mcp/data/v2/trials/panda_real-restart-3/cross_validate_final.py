import subprocess,json
base=['uv','run','--no-sync','--project','/home/horde/apps/newton-live-mcp','python']
for episode in [2,3,4]:
 name=f'cv-robust-select-a-ep{episode}'
 with open(name+'.log','w') as log:subprocess.run(base+['fit_model.py','--arm-vector','[0.03,0.02,0.03,0.02,0.025,0.025,0.015]','--huber','0.1','--exclude',str(episode),'--output',name+'.json'],stdout=log,stderr=subprocess.STDOUT,check=True)
 subprocess.run(base+['run_candidate.py',name+'.json','--note',f'Robust selected regularization; leave episode {episode} out'],check=True)
