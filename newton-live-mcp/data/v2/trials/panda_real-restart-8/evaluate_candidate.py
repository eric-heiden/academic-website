"""Archive and evaluate one configuration in a fresh prescribed rollout process."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

def main():
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('number',type=int);args=p.parse_args()
    destination=Path(f'candidate-{args.number:03d}')
    destination.mkdir(exist_ok=False)
    shutil.copyfile(args.config,'config.json')
    shutil.copyfile('config.json',destination/'config.json')
    command=['uv','run','--no-sync','--project','/home/horde/apps/newton-live-mcp','python','-m','tools.mcp_evaluation.real_rollout','--config','config.json','--reference','training.npz','--output',str(destination/'metrics.json')]
    start=time.time()
    with (destination/'process.log').open('w') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError(f'Candidate failed: {destination}')
    m=json.loads((destination/'metrics.json').read_text())
    print('candidate',args.number,'seconds',time.time()-start,'success',m['success'],flush=True)
    for e in [m]+m['per_episode']:
        print(e.get('episode','pooled'),{k:e[k] for k in m['thresholds']},flush=True)

if __name__=='__main__':main()
