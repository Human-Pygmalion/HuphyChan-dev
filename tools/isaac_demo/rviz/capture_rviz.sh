#!/usr/bin/env bash
# Headless RViz capture on the 3090 VM (10.8.0.14).   usage: capture_rviz.sh <rviz config> <out.png> [wait seconds]
# needs: ~/pyg_demo/world_dump.npz, ~/pyg_demo/rviz/{grid.npz,publish_scene.py}, ~/pyg_demo/assets/huphy_full_shell_loop.urdf (+ meshes)
set -uo pipefail
CFG=${1:?rviz config}
OUT=${2:?out png}
WAIT=${3:-30}
set +u
source ~/ros_env_rviz.sh
set -u
export DISPLAY=:99
H=$HOME/pyg_demo
URDF=$H/rviz/huphy_rviz.urdf
# RViz needs absolute mesh paths
sed "s#filename=\"meshes_v30_armfix_shell/#filename=\"file://$H/assets/meshes_v30_armfix_shell/#g" "$H/assets/huphy_full_shell_loop.urdf" > "$URDF"
python3 - "$URDF" "$H/rviz/rsp_params.yaml" <<'PY'
import re, sys
t = open(sys.argv[1], encoding='utf-8').read()
# RViz needs the colour inside every visual: copy the top-level material colours into the references, grey for visuals without material
cols = dict(re.findall(r'<material name="([^"]+)">\s*<color rgba="([^"]+)"/>\s*</material>', t))
t = re.sub(r'<material name="([^"]+)"/>', lambda m: '<material name="%s"><color rgba="%s"/></material>' % (m.group(1), cols.get(m.group(1), '0.72 0.73 0.76 1')), t)
t = re.sub(r'<visual.*?</visual>', lambda m: m.group(0) if '<material' in m.group(0) else m.group(0)[:-len('</visual>')] + '<material name="rv_grey"><color rgba="0.72 0.73 0.76 1"/></material></visual>', t, flags=re.S)
open(sys.argv[1], 'w', encoding='utf-8').write(t)
body = '\n'.join('      ' + l for l in t.splitlines())
open(sys.argv[2], 'w', encoding='utf-8').write('robot_state_publisher:\n  ros__parameters:\n    robot_description: |\n' + body + '\n')
PY
pkill -f "Xvfb :99" 2>/dev/null
sleep 1
Xvfb :99 -screen 0 2560x1440x24 >/tmp/xvfb.log 2>&1 &
XP=$!
sleep 2
ros2 run robot_state_publisher robot_state_publisher --ros-args --params-file "$H/rviz/rsp_params.yaml" >/tmp/rsp.log 2>&1 &
RS=$!
python3 "$H/rviz/publish_scene.py" --dump "$H/world_dump.npz" --grid "$H/rviz/grid.npz" --urdf "$URDF" ${PUB_ARGS:-} >/tmp/pub.log 2>&1 &
PB=$!
sleep 4
rviz2 -d "$CFG" >/tmp/rviz_cap.log 2>&1 &
RV=$!
sleep "$WAIT"
import -window root "$OUT"
pkill -P $RS 2>/dev/null; kill $RV $PB $RS $XP 2>/dev/null   # pkill -P: "ros2 run" leaves the robot_state_publisher node running otherwise
echo "saved $OUT"
