"""Run a pinned WBC evaluation manifest, retaining trajectories locally."""
import argparse
import json
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--manifest', type=Path, required=True)
parser.add_argument('--motions', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
manifest = json.loads(args.manifest.read_text())
actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
if actual != manifest['commit']:
    raise SystemExit('Check out Newton commit ' + manifest['commit'] + ' before evaluating.')
for run in manifest['runs']:
    command = ['uv', 'run', '--extra', 'wbc', '-m', 'newton.examples', 'robot_g1_wbc',
               '--viewer', 'null', '--test', '--num-frames', str(run['frames']),
               '--output', str(args.output / run['tag']), *run['arguments']]
    if run['motion'] != 'stand':
        command += ['--motion', str(args.motions / (run['motion'] + '.csv'))]
    print('Running', run['tag'], flush=True)
    with (args.output / (run['tag'] + '.log')).open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
