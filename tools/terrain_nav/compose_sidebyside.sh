#!/usr/bin/env bash
# Side-by-side, frame-locked (same clock from t = 0): the flat-only policy and the rough-trained policy on the same
# downhill trail, same path follower, same 0.6 m/s command. The left recording stops at its fall; its last frame
# (with the FELL banner from trail_video.py) is held for HOLD seconds while the right one keeps walking.
# Real time: both inputs are 25 fps real-time renders; nothing is sped up.
# usage: compose_sidebyside.sh <left.mp4 flat-only> <right.mp4 rough-trained> <out.mp4> [hold seconds, default 8]
set -euo pipefail
L=$1; R=$2; OUT=$3; HOLD=${4:-8}
F=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
FB=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
DL=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$L")
T=$(python3 -c "print(f'{$DL + $HOLD:.2f}')")
ffmpeg -y -v error -i "$L" -i "$R" -filter_complex "
[0:v]tpad=stop_mode=clone:stop_duration=${HOLD},scale=960:540,setsar=1[l];
[1:v]trim=duration=${T},setpts=PTS-STARTPTS,scale=960:540,setsar=1[r];
[l][r]hstack=inputs=2[s];
[s]pad=1920:1080:0:270:color=0x15181c,
drawtext=fontfile=${FB}:text='Same scanned trail, same path follower, same 0.6 m/s command  -  downhill':fontcolor=white:fontsize=38:x=(w-text_w)/2:y=70,
drawtext=fontfile=${F}:text='real time 1x  |  MuJoCo physics, Isaac Sim RTX render':fontcolor=0xb8c0cc:fontsize=26:x=(w-text_w)/2:y=130,
drawtext=fontfile=${FB}:text='trained on flat ground only (v11a)':fontcolor=0xff8a7a:fontsize=34:x=(960-text_w)/2:y=215,
drawtext=fontfile=${FB}:text='+ rough-terrain training':fontcolor=0x7ee08a:fontsize=34:x=960+(960-text_w)/2:y=215,
drawtext=fontfile=${F}:text='falls at 28 m':fontcolor=0xff8a7a:fontsize=30:x=(960-text_w)/2:y=845,
drawtext=fontfile=${F}:text='walks all 182 m (both directions)':fontcolor=0x7ee08a:fontsize=30:x=960+(960-text_w)/2:y=845[v]" \
  -map "[v]" -r 25 -c:v libx264 -crf 24 -preset medium -pix_fmt yuv420p -movflags +faststart "$OUT"
ffprobe -v error -show_entries format=duration -show_entries stream=nb_frames,r_frame_rate -of compact "$OUT"
