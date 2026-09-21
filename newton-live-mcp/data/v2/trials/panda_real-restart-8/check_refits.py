"""Fit leave-one-recording-out configurations for transfer checks."""
import json
from pathlib import Path
import numpy as np
from fit_model import fit, decode, scores
from tools.mcp_evaluation.real_robot_model import validate_config

d=np.load('training-regressor.npz');raw=d['A'];b=d['b'];eids=d['sample_episode_ids']
u,s,v=np.linalg.svd(raw,full_matrices=False);s[s<.01]=0;A=(u*s)@v
for ep in np.unique(eids):
    x,status=fit(A,b,np.repeat(eids!=ep,7),1e-5,robust=True,armature_floor=[.02,.01,.02,.01,.03,.015,.02])
    c=validate_config(decode(x))
    Path(f'refit-without-{ep}.json').write_text(json.dumps(c,indent=2)+'\n')
    print(ep,status,scores(raw,b,x,eids),flush=True)
