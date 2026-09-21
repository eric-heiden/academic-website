"""Run one prescribed fresh-process physical candidate and preserve its artifacts."""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

KEYS = ['max_joint_torque_rmse_nm', 'max_joint_torque_normalized_rmse',
        'max_joint_position_rmse_rad', 'max_joint_velocity_rmse_rad_s',
        'position_p95_rad', 'max_joint_speed_rad_s']


def evaluate(config, note):
    used = sorted(p for p in Path('.').glob('candidate-*') if p.is_dir() and p.name.split('-')[1].isdigit())
    index = max([int(p.name.split('-')[1]) for p in used] + [0]) + 1
    if index > 60:
        raise RuntimeError('Physical candidate budget exhausted')
    dest = Path(f'candidate-{index:03d}')
    dest.mkdir(exist_ok=False)
    s = json.dumps(config, indent=2) + '\n'
    Path('config.json').write_text(s)
    (dest/'config.json').write_text(s)
    (dest/'estimation_note.json').write_text(json.dumps(note,indent=2)+'\n')
    with (dest/'process.log').open('x') as log:
        proc = subprocess.run(['uv','run','--no-sync','--project','/home/horde/apps/newton-live-mcp',
            'python','-m','tools.mcp_evaluation.real_rollout','--config','config.json',
            '--reference','training.npz','--output',str(dest/'metrics.json')], stdout=log,stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError(f'{dest} failed; see preserved process.log')
    m=json.loads((dest/'metrics.json').read_text())
    groups=[m]+m['per_episode']
    ratios=np.array([[g[k]/m['thresholds'][k] for k in KEYS] for g in groups])
    summary={'candidate':index,'success':m['success'],'worst_threshold_ratios':dict(zip(KEYS,ratios.max(0).tolist())),
             'torque':m['torque_rmse_per_joint_nm'],'position':m['position_rmse_per_joint_rad'],
             'velocity':m['velocity_rmse_per_joint_rad_s'], 'note':note}
    with Path('candidate-summary.jsonl').open('a') as stream:
        stream.write(json.dumps(summary)+'\n')
    print(json.dumps(summary),flush=True)
    return m,summary


if __name__=='__main__':
    evaluate(json.loads(Path(sys.argv[1]).read_text()), {'source_config':sys.argv[1]})
