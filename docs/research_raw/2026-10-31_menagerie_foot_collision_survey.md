# Raw excerpts: how real humanoid MJCF/URDF files define the foot (fetched 2026-10-31, main branch)
Sources: https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/<robot>/<xml>; unitreerobotics/unitree_rl_gym g1_29dof(.rev_1_0).urdf/xml; unitreerobotics/unitree_ros h1.urdf

## H1 (unitree_h1/h1.xml) default classes + ankle
```
      <default class="collision">
        <geom group="3" mass="0" density="0"/>
        <default class="foot">
          <geom type="capsule" size=".014"/>
          <default class="foot1">
            <geom fromto="-.035 0 -0.056 .02 0 -0.045"/>
          </default>
          <default class="foot2">
            <geom fromto=".02 0 -0.045 .115 0 -0.056"/>
          </default>
          <default class="foot3">
            <geom fromto=".14 -.03 -0.056 .14 .03 -0.056"/>
          </default>
        </default>
      </default>
              <body name="left_ankle_link" pos="0 0 -0.4">
                <inertial pos="0.06722 0.00015 -0.04497" quat="0.489101 0.503197 0.565782 0.432972" mass="0.446"
                  diaginertia="0.00220848 0.00218961 0.000214202"/>
                <joint name="left_ankle" axis="0 1 0" range="-0.87 0.52"/>
                <geom class="visual" mesh="left_ankle_link"/>
                <geom class="foot1"/>
                <geom class="foot2"/>
                <geom class="foot3"/>
              </body>
```
## G1 (unitree_g1/g1.xml)
```
      <default class="visual">
        <geom group="2" type="mesh" contype="0" conaffinity="0" density="0" material="metal"/>
      </default>
      <default class="collision">
        <geom group="3" type="mesh"/>
        <default class="foot">
          <geom type="sphere" size="0.005" priority="1" friction="0.6" condim="3"/>
        </default>
      </default>
                <body name="left_ankle_roll_link" pos="0 0 -0.017558">
                  <inertial pos="0.026505 0 -0.016425" quat="-0.000481092 0.728482 -0.000618967 0.685065" mass="0.608"
                    diaginertia="0.00167218 0.0016161 0.000217621"/>
                  <joint name="left_ankle_roll_joint" axis="1 0 0" range="-0.2618 0.2618" actuatorfrcrange="-50 50"/>
                  <geom class="visual" material="black" mesh="left_ankle_roll_link"/>
                  <geom class="foot" pos="-0.05 0.025 -0.03"/>
                  <geom class="foot" pos="-0.05 -0.025 -0.03"/>
                  <geom class="foot" pos="0.12 0.03 -0.03"/>
                  <geom class="foot" pos="0.12 -0.03 -0.03"/>
```
## G1 unitree_rl_gym g1_29dof.urdf left_ankle_roll_link (4 spheres r=0.005 at (-0.05,+-0.025,-0.03),(0.12,+-0.03,-0.03)); G1 foot STL bbox x[-0.0658,0.1424] y[-0.0377,0.0378] z[-0.0354,0.0236]
## T1
```
                  <body name="left_foot_link" pos="0 0.00025 -0.012">
                    <inertial pos="-0.000249 0 -0.00914" quat="0 0.620755 0 0.784005" mass="0.685"
                      diaginertia="0.00269786 0.002385 0.00218714"/>
                    <joint name="Left_Ankle_Roll" axis="1 0 0" range="-0.44 0.44" actuatorfrcrange="-15 15"/>
                    <geom class="visual" mesh="left_foot_link"/>
                    <geom class="collision" size="0.1115 0.05 0.015" pos="0.01 0 -0.015" type="box"/>
                  </body>
```
## Berkeley
```
      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" density="0" group="2" material="metal"/>
      </default>
      <default class="collision">
        <geom group="3"/>
        <default class="foot1">
          <geom friction="0.8 0.02 0.01" size="0.08 0.0275 0.006" pos="0 -0.0429038 0.0133277" type="box"/>
        </default>
        <default class="foot2">
          <geom size="0.027 0.0325" pos="0 9.61687e-05 0.0388277" type="cylinder"/>
        </default>
                <body name="ll_faa" pos="0 0.0175 0" quat="0.5 -0.5 -0.5 -0.5">
                  <inertial pos="-2.9e-05 -0.0163978 0.0153387" quat="0.985258 -0.171058 -0.00255986 0.000455342"
                    mass="0.507" diaginertia="0.00109422 0.000978734 0.000375399"/>
                  <joint name="LL_FAA" class="faa"/>
                  <geom pos="-0.16 0.503096 -0.0286723" quat="0.5 -0.5 -0.5 -0.5" class="visual" mesh="ll_faa"/>
                  <geom class="foot1" quat="0.45452 -0.541675 -0.541675 -0.45452"/>
                  <geom class="foot2" quat="0 -1 -1 0"/>
                  <site name="LL_FOOT" pos="0 -0.048689 0.014485" quat="-0.454519 0.541675 0.541675 0.454519"/>
```
## Cassie
```
  <default>
    <geom contype="0" conaffinity="0" condim="1" solref="0.005 1"/>
    <equality solref="0.005 1"/>
    <default class="cassie">
      <geom material="cassie" group="2"/>
    </default>
    <default class="collision">
      <geom contype="1" group="3" type="capsule"/>
      <default class="collision-left">
        <geom contype="2" conaffinity="4"/>
      </default>
      <default class="collision-right">
        <geom contype="4" conaffinity="2"/>
      </default>
                  <body name="left-foot" pos="0.408 -0.04 0">
                    <inertial pos="0.00474 0.02748 -0.00014" mass="0.1498"
                      fullinertia="0.00017388 0.00016793 0.00033261 0.00011814 1.36e-06 -4e-07"/>
                    <joint name="left-foot" type="hinge" range="-140 -30" damping="1" armature="0.01225"/>
                    <geom type="mesh" mesh="left-foot" euler="0 0 90"/>
                    <geom size="0.02" fromto="-0.052821 0.092622 0 0.069746 -0.010224 0" class="collision-left"/>
                  </body>
```
## Talos
```
  <default>
    <default class="talos">
      <site rgba="1 0 0 1" size="0.01" group="5"/>
      <joint frictionloss="1" damping="1"/>
      <geom type="mesh" solref="0.001 1"/>
      <default class="visual">
        <geom group="2" contype="0" conaffinity="0" density="0"/>
      </default>
      <default class="collision">
        <geom group="3"/>
      </default>
    </default>
  </default>
                  <inertial pos="-0.0201699 -0.000510665 -0.0602836" quat="0.343485 0.610901 0.618186 0.3559"
                    mass="1.61177" diaginertia="0.00773279 0.006503 0.00387932"/>
                  <joint name="leg_left_6_joint" axis="1 0 0" range="-0.5236 0.5236" actuatorfrcrange="-100 100"/>
                  <geom class="visual" rgba="0.7 0.7 0.7 1" mesh="l_ankle_X_lo_res"/>
                  <geom class="collision" contype="6" type="box" size=".1 .06 .006" pos="-0.005 0 -.1"/>
                  <site name="left_foot" pos="0 0 -0.1"/>
```
## Apollo
```
  <default>
    <geom contype="0" conaffinity="0" condim="1" margin="0.000500" solref="0.005 1" group="1"/>
    <equality solref="0.005 1" solimp="0.99 0.999 0.00001"/>
    </default>
    <default class="collision">
      <geom contype="0" conaffinity="0" group="3" rgba="0 1 0 0.4"/>
                <body name="l_foot_link" quat="0.965926 0 0.258819 0">
                  <frame pos="0.0614891 -0.00342931 -0.0265273" quat="0.987443 -0.0797175 -0.132426 -0.0326442">
                    <site name="l_foot_fr" pos="0.1 0.0425 -0.0215"/>
                    <site name="l_foot_br" pos="-0.1 0.0425 -0.0215"/>
                    <site name="l_foot_fl" pos="0.1 -0.0425 -0.0215"/>
                    <site name="l_foot_bl" pos="-0.1 -0.0425 -0.0215"/>
                  </frame>
                  <inertial pos="0.056607 -0.00200339 -0.0304112" quat="0.432893 0.587516 0.510475 0.454802"
                    mass="1.1808" diaginertia="0.0088235 0.00838261 0.00210633"/>
                  <joint name="l_ankle_pd" axis="0 1 0" range="-1.5708 0.436332" actuatorfrcrange="-150 150"
                    damping="21" frictionloss="1.87" armature="0.205456"/>
                  <geom class="visual_dark" mesh="l_foot_link"/>
                  <geom name="collision_l_sole" size="0.1 0.0425 0.009" pos="0.0646931 -0.00550529 -0.03843"
                    quat="0.987443 -0.0797175 -0.132426 -0.0326442" type="box" class="collision"/>
scene.xml: <pair geom1="collision_l_sole" geom2="floor" condim="6" solref="0.005 1" friction="2 2 0.01 0.0001 0.0001"/>
```
## OP3
```
  <default>
    <mesh scale="0.001 0.001 0.001"/>
    <geom type="mesh" solref=".004 1"/>
    <joint damping="1.084" armature="0.045" frictionloss="0.03"/>
    <site group="5" type="sphere"/>
    <position kp="21.1" ctrlrange="-3.141592 3.141592" forcerange="-5 5"/>
    <default class="collision">
      <geom group="3"/>
      <default class="foot">
        <geom mass="0" type="box"/>
      </default>
    </default>
    <default class="visual">
      <geom material="black" contype="0" conaffinity="0" group="2"/>
    </default>
  </default>
                    diaginertia="0.000115818 7.87135e-05 4.03389e-05"/>
                  <joint name="l_ank_roll" axis="1 0 0"/>
                  <geom mesh="ll6" class="visual"/>
                  <!-- <geom mesh="ll6c" class="collision" /> -->
                  <geom class="foot" pos="0.024 0.013 -0.0265" size="0.0635 0.028 0.004"/>
                  <geom class="foot" pos="0.024 0.0125 -0.0265" size="0.057 0.039 0.004"/>
                </body>
```
## ToddlerBot 2XC
```
                    <body name="left_ankle_roll_link" pos="-0.023 0.019 0" quat="1 0 0 0">
                      <joint axis="1 0 0" name="left_ankle_roll" range="-1.221730476395941 1.2217304763961203"
                        class="XC430"/>
                      <inertial pos="0.0358432 -7e-08 -0.0267229" mass="0.039"
                        fullinertia="2.88034e-05 3.04307e-05 8.87438e-06 6.52846e-11 2.12102e-06 2.65224e-10"/>
                      <geom type="mesh" class="visual" pos="0.0358432 -7e-08 -0.0267229" quat="-1 0 0 0"
                        mesh="left_ankle_roll_link_visual" material="ankle_roll_link_visual_material"/>
                      <geom type="mesh" class="collision" mesh="left_ankle_roll_link_collision"
                        pos="0.0358432 -7e-08 -0.0267229" quat="-1 0 0 0" name="left_ankle_roll_link_collision"
                        material="ankle_roll_link_visual_material" contype="1" conaffinity="1"/>
                      <site name="left_foot_center" pos="0.0358432 -7e-08 -0.0374345" type="sphere" size="0.005"
                        rgba="0.9 0.1 0.1 0.8"/>
collision STL: 56 tris, bbox x[-0.0510,0.0563] y[-0.0201,0.0201] z[-0.0122,-0.0022] m
```
## Fourier N1
```
                <body name="left_foot_pitch_link">
                  <inertial pos="0.031083 -0.00011033 -0.031859" quat="5.38314e-05 0.70921 -0.0030347 0.704991"
                    mass="0.528" diaginertia="0.00183246 0.00168042 0.000253637"/>
                  <joint class="joint_4530E" name="left_ankle_pitch_joint" pos="0 0 0" axis="0 1 0"
                    range="-0.785 0.785"/>
                  <geom class="visual" mesh="left_foot_pitch_link"/>
                  <geom class="collision" type="mesh" mesh="left_foot_pitch_link"/>
foot_pitch STL: 22960 tris bbox x[-0.0735,0.1405] y[-0.0385,0.0385] z[-0.0445,0.0115]
```
## PNDbotics Adam Lite
```
                <body name="toeLeft">
                  <inertial pos="0.0370066 -2.063e-05 -0.0419868" quat="0.00469469 0.673963 -0.00294814 0.738744"
                    mass="0.4758" diaginertia="0.00159292 0.00152195 0.000273726"/>
                  <joint name="ankleRoll_Left" pos="0 0 0" axis="1 0 0" range="-0.3491 0.3491"
                    actuatorfrcrange="-12 12"/>
                  <geom mesh="toeLeft_0" material="pnd_logo" class="visual"/>
                  <geom mesh="toeLeft_1" class="visual"/>
                  <geom mesh="toeLeft_2" material="pnd_logo" class="visual"/>
                  <geom mesh="toeLeft_3" class="visual"/>
                  <geom name="toe1_left" size="0.015 0.11" pos="0.045 0.025 -0.049" quat="0.703845 0 0.710353 0"
                    type="cylinder" class="collision"/>
                  <geom name="toe2_left" size="0.015 0.11" pos="0.045 -0.025 -0.049" quat="0.703845 0 0.710353 0"
                    type="cylinder" class="collision"/>
                </body>
```
## H1 URDF unitree_ros left_ankle_link: mass 0.725; collision origin xyz=0.05 0 -0.05, box size 0.28 0.03 0.024
