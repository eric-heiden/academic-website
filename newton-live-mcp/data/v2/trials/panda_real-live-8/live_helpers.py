from pathlib import Path
import json, copy
import numpy as np
root_dir=Path('/home/horde/artifacts/newton-live-mcp-v2/confirmation/panda_real-live-8')

def evaluate_candidate(candidate,name,held_episode=None):
    session.scenario.apply_config(candidate)
    session.dispatch('reset')
    session.dispatch('step', {'count':1800})
    measured=session.scenario.metrics()
    with (root_dir/'fit_diagnostics.jsonl').open('a') as stream:
        stream.write(json.dumps({'name':name,'metrics':measured})+'\n')
    selected=measured['per_episode'] if held_episode is None else [e for e in measured['per_episode'] if e['episode']==held_episode]
    return {'name':name,'trace':Path(measured['trace_path']).name,'success':measured['success'],
            'torque':measured['torque_rmse_per_joint_nm'],'position':measured['position_rmse_per_joint_rad'],
            'velocity':measured['velocity_rmse_per_joint_rad_s'],
            'worst_ratio':max(e[k]/v for e in selected for k,v in measured['thresholds'].items()),
            'selected_worst':{k:max(e[k] for e in selected) for k in measured['thresholds']}}

def evaluate_file(name,held_episode=None):
    return evaluate_candidate(json.loads((root_dir/(name+'.json')).read_text()),name,held_episode)
