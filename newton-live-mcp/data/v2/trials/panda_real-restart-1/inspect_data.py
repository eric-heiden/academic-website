import numpy as np,json
r=np.load('training.npz')
for k in r.files:
 v=r[k]
 print(k,v.shape)
 if k in ('q','qd','qdd','tau'):
  print('mean',v.mean(0),'std',v.std(0),'range',np.min(v,axis=0),np.max(v,axis=0))
m=json.load(open('candidate-001/metrics.json'))
t=np.load(m['trace_path'])
for k in ('q','qd'):
 e=(t[k]-t['reference_'+k]).reshape(36,50,7)
 print(k,'end mean',e[:,-1].mean(0),'end rms',np.sqrt((e[:,-1]**2).mean(0)))
