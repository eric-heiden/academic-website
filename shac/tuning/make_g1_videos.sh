#!/bin/bash
# Records and renders the G1 report videos in the reference simulator.
#   $1: final G1 checkpoint (gait prior, SAPO), $2: comparison checkpoint (no prior, SAPO)
# Outputs media/tuning_g1_walk_seq.mp4 (command sequence) and media/tuning_g1_compare.mp4 (side by side).
set -euo pipefail
cd "$(dirname "$0")"
PY=/home/horde/repos/mujoco_warp-pr1535-357a75d/.venv/bin/python
export MUJOCO_GL=glfw DISPLAY=:99
FINAL=$1
NOPRIOR=$2
SEQ=("3 0.5 0 0" "3 0.5 0 0.5" "3 0 0.3 0" "3 0 0 0")
CMP=("4 0.5 0 0" "3 0.5 0 0.5" "3 0 0 0")
# Side view of episode 0 relative to its initial heading.
azimuth() {
  $PY - "$1" <<'EOF'
import sys, numpy as np
q = np.load(sys.argv[1], allow_pickle=True)["qpos"][0, 0]
w, x, y, z = q[3:7]
print(round(float(np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))) + 90.0, 1))
EOF
}
poster() { $PY -c "
import imageio.v2 as io, sys
r = io.get_reader(sys.argv[1])
for i, f in enumerate(r):
    if i == int(sys.argv[3]): io.imwrite(sys.argv[2], f, quality=88); break
" "$1" "$2" "$3"; }

$PY record2.py --ckpt "$FINAL" --out media/g1_walk_seq.npz --envs 8 --cmd-schedule "${SEQ[@]}"
AZ=$(azimuth media/g1_walk_seq.npz)
$PY render.py --npz media/g1_walk_seq.npz --episode 0 --out media/g1_walk_seq_raw.mp4 --width 640 --height 360 \
  --distance 3.0 --elevation -10 --azimuth "$AZ" --poster-time 2
$PY overlay_cmds.py --video media/g1_walk_seq_raw.mp4 --out media/tuning_g1_walk_seq.mp4 --size 16 --cmd-schedule "${SEQ[@]}"
poster media/tuning_g1_walk_seq.mp4 media/tuning_g1_walk_seq_poster.jpg 100

for tag in noprior final; do
  ck=$NOPRIOR; [ $tag = final ] && ck=$FINAL
  $PY record2.py --ckpt "$ck" --out media/g1_cmp_$tag.npz --envs 8 --cmd-schedule "${CMP[@]}"
done
AZ=$(azimuth media/g1_cmp_final.npz)
for tag in noprior final; do
  $PY render.py --npz media/g1_cmp_$tag.npz --episode 0 --out media/g1_cmp_${tag}_raw.mp4 --width 640 --height 360 \
    --distance 2.6 --elevation -8 --azimuth "$AZ" --poster-time 2
  $PY overlay_cmds.py --video media/g1_cmp_${tag}_raw.mp4 --out media/g1_cmp_$tag.mp4 --size 16 --cmd-schedule "${CMP[@]}"
done
FF=$($PY -c "import imageio_ffmpeg as f; print(f.get_ffmpeg_exe())")
$FF -y -loglevel error -i media/g1_cmp_noprior.mp4 -i media/g1_cmp_final.mp4 -filter_complex "[0:v][1:v]hstack=inputs=2" \
  -c:v libx264 -pix_fmt yuv420p -crf 24 -movflags +faststart media/tuning_g1_compare.mp4
poster media/tuning_g1_compare.mp4 media/tuning_g1_compare_poster.jpg 100
ls -la media/tuning_g1_walk_seq* media/tuning_g1_compare*
