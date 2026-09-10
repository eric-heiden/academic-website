"""Audit kinematic centroidal momentum during flight; does not simulate control."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
import newton.utils
from newton.examples.robot.wbc_controller import MotionReference

parser=argparse.ArgumentParser()
parser.add_argument('motion',type=Path)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
m=mujoco.MjModel.from_xml_path(str(newton.utils.download_asset('unitree_g1')/'mjcf/g1_29dof_rev_1_0.xml'))
d=mujoco.MjData(m)
r=MotionReference(m,np.loadtxt(args.motion,delimiter=','))
feet=[m.body(s+'_ankle_roll_link').id for s in ('left','right')]
local=np.array([[x,y,-.035] for x in (-.05,.12) for y in (-.025,.025)])
d.qpos[:]=r.qpos[0];mujoco.mj_kinematics(m,d)
shift=.002-min((d.xpos[b]+local@d.xmat[b].reshape(3,3).T)[:,2].min() for b in feet)
r.qpos[:,2]+=shift
rows=[]
for i,q in enumerate(r.qpos):
 d.qpos[:]=q;d.qvel[:]=r.velocity[i]
 mujoco.mj_forward(m,d);mujoco.mj_subtreeVel(m,d)
 clearance=min((d.xpos[b]+local@d.xmat[b].reshape(3,3).T)[:,2].min() for b in feet)
 contacts=sum(0 in m.geom_bodyid[[c.geom1,c.geom2]] for c in d.contact)
 rows.append([i/r.fps,clearance,*d.subtree_com[1],*d.subtree_linvel[1],*d.subtree_angmom[1],contacts])
a=np.array(rows);mask=(a[:,1]>.03)&(a[:,-1]==0)
intervals=[]
indices=np.flatnonzero(mask)
for group in np.split(indices,np.flatnonzero(np.diff(indices)>1)+1):
 if len(group)<4:continue
 core=group[1:-1]
 acceleration=np.gradient(a[:,5:8],1/r.fps,axis=0)
 residual=m.body_mass.sum()*(acceleration[core]-m.opt.gravity)
 angular=a[core,8:11]
 intervals.append(dict(start=float(a[group[0],0]),end=float(a[group[-1],0]),frames=len(group),
 mass_kg=float(m.body_mass.sum()),unexplained_force_rms_N=float(np.sqrt(np.mean(np.sum(residual**2,axis=1)))),
 angular_momentum_mean_kg_m2_s=angular.mean(axis=0).tolist(),angular_momentum_range_kg_m2_s=np.ptp(angular,axis=0).tolist()))
result=dict(method='Centroidal momentum from the 30 Hz kinematic reference; both sole clearances above 3 cm and no floor contacts; omit one boundary frame each side.',intervals=intervals)
args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
