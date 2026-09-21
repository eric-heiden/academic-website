from fit_model import *
from candidate_runner import run
import argparse

def estimate(arms,omit=None,reg=1e-4,huber=0,physical_margin=1e-6,joint_weights=None):
 x=cp.Variable(98); w=np.ones((len(b)//7,7))
 for ep in np.unique(episodes):
  mask=episodes==ep
  w[mask]=1/np.minimum(np.maximum(b.reshape(-1,7)[mask].std(axis=0),.5),1)
  if ep==omit:w[mask]=0
 if joint_weights is not None:w*=np.asarray(joint_weights)[None,:]
 cons=constraints(x)+[x[91:98]>=arms]
 if physical_margin>1e-6:
  for i in range(7):
   k=10*i;m=x[k];h=x[k+1:k+4];z=x[k+4:k+10]
   I=cp.bmat([[z[0],z[3],z[4]],[z[3],z[1],z[5]],[z[4],z[5],z[2]]]); S=.5*cp.trace(I)*np.eye(3)-I
   P=cp.bmat([[S,cp.reshape(h,(3,1),order='C')],[cp.reshape(h,(1,3),order='C'),cp.reshape(m,(1,1),order='C')]])
   cons += [P>>physical_margin*np.eye(4)]
 scale=np.tile([.2,1,1,1,3,3,3,4,4,4],7).tolist()+[.2]*21+[1]*7
 res=cp.multiply(w.ravel(),A@x-b)
 loss=(cp.sum_squares(res) if not huber else cp.sum(cp.huber(res,huber)))/(np.sum(w>0)/7)+reg*cp.sum_squares(cp.multiply(scale,x))
 p=cp.Problem(cp.Minimize(loss),cons)
 p.solve(solver='CLARABEL',tol_gap_abs=1e-9,tol_feas=1e-9,tol_gap_rel=1e-9,max_iter=300)
 print('fit',p.status,p.value,flush=True)
 return x.value

if __name__=='__main__':
 arms=np.array([.025,.015,.025,.015,.03,.015,.02])
 for omit in [None,2,3,4]:
  x=estimate(arms,omit)
  conf=to_config(x)
  Path(f'balanced-omit-{omit}.json').write_text(json.dumps(conf,indent=2)+'\n')
  print('scores',omit,json.dumps(score(x)),flush=True)
  run(conf,f'balanced arms omit {omit}')
