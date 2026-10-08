#!/usr/bin/env bash
# Interactive mountain-trail navigation viewer on the 3090 VM (10.8.0.14): RViz2 on a virtual screen, shared over VNC.
#   start:  bash run_nav_viewer.sh            stop:  bash run_nav_viewer.sh stop
#   connect: any VNC viewer -> 10.8.0.14:5901  (password in ~/pyg_demo/nav/VNC_PASSWORD.txt, readable only by syaro)
# In RViz: "2D Pose Estimate" = start, "2D Goal Pose" = goal -> A* re-plans over the terrain classes.
# needs: ~/pyg_demo/nav/{nav_server.py,nav.rviz,terrain.npz,map_010.npy,map_005.npy,map_full.npy}, ~/pyg_demo/world_dump.npz,
#        ~/pyg_demo/assets/huphy_full_shell_loop.urdf (+ meshes) - the RViz URDF is written by ../rviz/capture_rviz.sh logic below
set -uo pipefail
H=$HOME/pyg_demo; N=$H/nav; DISP=:97; PORT=5901
PIDF=$N/pids
if [ "${1:-}" = stop ]; then
  [ -f "$PIDF" ] && for p in $(cat "$PIDF"); do pkill -P "$p" 2>/dev/null; kill "$p" 2>/dev/null; done; rm -f "$PIDF"; echo stopped; exit 0   # also the children: "ros2 run" leaves its node behind otherwise
fi
[ -f "$PIDF" ] && { echo "already running (pids $(cat "$PIDF")); stop first"; exit 1; }
set +u; source ~/ros_env_rviz.sh; set -u
export DISPLAY=$DISP
# RViz URDF: absolute mesh paths + colour inside every visual (same transform as capture_rviz.sh)
URDF=$N/huphy_rviz.urdf
sed "s#filename=\"meshes_v30_armfix_shell/#filename=\"file://$H/assets/meshes_v30_armfix_shell/#g" "$H/assets/huphy_full_shell_loop.urdf" > "$URDF"
python3 - "$URDF" "$N/rsp_params.yaml" <<'PY'
import re, sys
t = open(sys.argv[1], encoding='utf-8').read()
cols = dict(re.findall(r'<material name="([^"]+)">\s*<color rgba="([^"]+)"/>\s*</material>', t))
t = re.sub(r'<material name="([^"]+)"/>', lambda m: '<material name="%s"><color rgba="%s"/></material>' % (m.group(1), cols.get(m.group(1), '0.72 0.73 0.76 1')), t)
t = re.sub(r'<visual.*?</visual>', lambda m: m.group(0) if '<material' in m.group(0) else m.group(0)[:-len('</visual>')] + '<material name="rv_grey"><color rgba="0.72 0.73 0.76 1"/></material></visual>', t, flags=re.S)
open(sys.argv[1], 'w', encoding='utf-8').write(t)
open(sys.argv[2], 'w', encoding='utf-8').write('robot_state_publisher:\n  ros__parameters:\n    robot_description: |\n' + '\n'.join('      ' + l for l in t.splitlines()) + '\n')
PY
# VNC password: generated once, kept in a file only this user can read
if [ ! -f "$N/vnc.passwd" ]; then
  python3 -c "import secrets;print(secrets.token_urlsafe(9))" > "$N/VNC_PASSWORD.txt"; chmod 600 "$N/VNC_PASSWORD.txt"
  x11vnc -storepasswd "$(cat "$N/VNC_PASSWORD.txt")" "$N/vnc.passwd" >/dev/null 2>&1; chmod 600 "$N/vnc.passwd"
fi
Xvfb $DISP -screen 0 1920x1080x24 >"$N/xvfb.log" 2>&1 & echo $! >> "$PIDF"
sleep 2
ros2 run robot_state_publisher robot_state_publisher --ros-args --params-file "$N/rsp_params.yaml" >"$N/rsp.log" 2>&1 & echo $! >> "$PIDF"
python3 "$N/nav_server.py" --data "$N" --urdf "$URDF" --dump "$H/world_dump.npz" >"$N/nav.log" 2>&1 & echo $! >> "$PIDF"
sleep 8
rviz2 -d "$N/nav.rviz" >"$N/rviz.log" 2>&1 & echo $! >> "$PIDF"
x11vnc -display $DISP -rfbport $PORT -listen 0.0.0.0 -forever -shared -rfbauth "$N/vnc.passwd" -noxdamage -quiet >"$N/vnc.log" 2>&1 & echo $! >> "$PIDF"
sleep 3
echo "running: VNC 10.8.0.14:$PORT (0.0.0.0), password file $N/VNC_PASSWORD.txt; logs in $N/*.log"
