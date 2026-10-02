#!/bin/bash
# Foot separation and interpenetration of a G1 checkpoint under fixed commands (foot_sep.py), JSON lines to
# results/loop/footsep/<run>.json.  $1: run name, $2: model edits, $3/$4: left/right foot bodies, $5...: commands.
set -euo pipefail
cd "$(dirname "$0")"
PY=/home/horde/repos/mujoco_warp-pr1535-357a75d/.venv/bin/python
mkdir -p results/loop/footsep
RUN=$1; EDIT=$2
out=results/loop/footsep/$1.json; : > $out
LEFT=${3:-left_ankle_roll_link}; RIGHT=${4:-right_ankle_roll_link}
shift $(( $# < 4 ? $# : 4 ))
CMDS=("$@"); [ ${#CMDS[@]} -eq 0 ] && CMDS=("0 0.3 0" "0 -0.3 0" "1.0 0 0" "0.5 0 0.5")
for cmd in "${CMDS[@]}"; do
  $PY record2.py --ckpt runs/loop/$RUN.pt --out /tmp/sep_$RUN.npz --envs 8 --steps 300 --cmd $cmd > /dev/null 2>&1
  $PY foot_sep.py --npz /tmp/sep_$RUN.npz --model-edit "$EDIT" --left $LEFT --right $RIGHT --label "$cmd" >> $out
done
cat $out
