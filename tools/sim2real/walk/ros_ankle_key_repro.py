#!/usr/bin/env python3
"""Offline reproduction: in HUPHY biped's ROS path the four ankle crank motor angles reach the
policy as 0.0. Uses HUPHY's own ros/huphy_ros/huphy_ros/convert.py (read from git, SEP stubbed).

  python3 tools/sim2real/walk/ros_ankle_key_repro.py [~/external_repos/HUPHY] [origin/biped]

Path: control_node.py:141 publishes robot.joint_names = leg.JOINT_NAMES (ankle_pitch/ankle_roll,
leg.py:92) -> policy_node.py:134 arrays_to_observation -> policy_node.py:227 and
generalized_walk.py:170 read f"{j}.pos" for j in ORDER (ankle_a/ankle_b) with default 0.0.
"""
import subprocess, sys, types, os
repo = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/external_repos/HUPHY")
ref = sys.argv[2] if len(sys.argv) > 2 else "origin/biped"
src = subprocess.check_output(["git", "-C", repo, "show", f"{ref}:ros/huphy_ros/huphy_ros/convert.py"], text=True)
src = src.replace("from huphy.robots.biped import SEP", 'SEP = "/"  # huphy.robots.biped.SEP')
convert = types.ModuleType("convert"); exec(compile(src, "convert.py", "exec"), convert.__dict__)
LEGS = ("right_leg", "left_leg")
JOINT_NAMES = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")      # leg.py:92
ORDER = [f"{l}/{j}" for l in LEGS for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_a", "ankle_b")]
names = tuple(f"{l}/{j}" for l in LEGS for j in JOINT_NAMES)                                # control_node.py:141
obs = {f"{l}/{m}.pos": 10.0 for l in LEGS
       for m in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_a", "ankle_b", "ankle_pitch", "ankle_roll")}
n, p, _, _ = convert.observation_to_arrays(obs, names)                                     # control_node.py:254
back = convert.arrays_to_observation(n, p)                                                  # policy_node.py:134
seen = {k: back.get(f"{k}.pos", 0.0) for k in ORDER}                                        # policy_node.py:227
print("every motor at 10.0 deg; policy sees:")
for k, v in seen.items():
    print(f"  {k:22s} {v:6.2f}{'   <-- missing, defaulted to 0.0' if v == 0.0 else ''}")
