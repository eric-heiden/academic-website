"""Run complete physical candidates only through the prescribed live workflow."""
from pathlib import Path
import json
import numpy as np
root = Path('/home/horde/artifacts/newton-live-mcp-v2/confirmation/panda_real-live-2')

def evaluate_candidate(name):
    candidate = json.loads((root/(name+'.json')).read_text())
    session.scenario.apply_config(candidate)
    session.dispatch('reset')
    session.dispatch('step', {'count':1800})
    metrics = session.scenario.metrics()
    (root/(name+'-metrics.json')).write_text(json.dumps(metrics,indent=2)+'\n')
    summary = {k:metrics[k] for k in ['success','trace_path','torque_rmse_per_joint_nm','position_rmse_per_joint_rad','velocity_rmse_per_joint_rad_s','max_joint_speed_rad_s']}
    summary['name']=name
    summary['worst_episode']={k:max(ep[k] for ep in metrics['per_episode']) for k in metrics['thresholds']}
    with (root/'search-history.jsonl').open('a') as f: f.write(json.dumps(summary)+'\n')
    return summary
