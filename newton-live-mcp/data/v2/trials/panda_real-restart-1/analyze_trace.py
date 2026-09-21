import json,sys
import numpy as np
for f in sys.argv[1:]:
 m=json.load(open(f));t=np.load(m['trace_path'])
 print(f)
 print('worst per joint torque',np.max([e['torque_rmse_per_joint_nm'] for e in m['per_episode']],axis=0))
 print('worst per joint norm',np.max([e['torque_normalized_rmse_per_joint'] for e in m['per_episode']],axis=0))
 print('worst per joint position',np.max([e['position_rmse_per_joint_rad'] for e in m['per_episode']],axis=0))
 print('worst per joint velocity',np.max([e['velocity_rmse_per_joint_rad_s'] for e in m['per_episode']],axis=0))
 print('final window velocity rms',np.sqrt(np.mean((t['qd'].reshape(36,50,7)[:,-1]-t['reference_qd'].reshape(36,50,7)[:,-1])**2,axis=0)))
