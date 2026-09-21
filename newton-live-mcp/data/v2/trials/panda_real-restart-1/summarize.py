import json,sys
from pathlib import Path
keys=['max_joint_torque_rmse_nm','max_joint_torque_normalized_rmse','max_joint_position_rmse_rad','max_joint_velocity_rmse_rad_s','position_p95_rad','max_joint_speed_rad_s']
for file in sys.argv[1:]:
 m=json.load(open(file));print(file,'success',m['success'],'frames',m['frames'])
 print('pooled', {k:round(m[k],6) for k in keys})
 print('worst recording',{k:round(max(e[k] for e in m['per_episode']),6) for k in keys})
 print('position', [round(v,6) for v in m['position_rmse_per_joint_rad']])
 print('velocity', [round(v,6) for v in m['velocity_rmse_per_joint_rad_s']])
