# Raw sources: finding PD gains by running real motors, and matching the real joint to the simulated joint (2026-10-16)

Question (from the user, 2026-10-16): "While running the real motors I want to find suitable PD gains. Survey the methods and tell me what to do. Also help me acquire the knowledge needed."
Scope of this file: sources and verbatim excerpts only. The synthesis is in `docs/172_real_motor_pd_tuning_guide.md`.

Method note
- arXiv PDFs and manuals were downloaded and converted with `pdftotext` (raw reading order). Every excerpt marked with a quotation mark was checked by a script against the extracted text (whitespace and ligatures normalised). Source files were code files, Markdown files or PDFs fetched on 2026-10-16.
- GitHub files were fetched from raw.githubusercontent.com at the commit named in the entry.
- A claim line is my one-line reading with numbers. "UNVERIFIED" marks anything not checked against the primary text.
- Our robot (from the task brief, not re-measured here): 23.6 kg lower body; RS04 (hip pitch, hip roll, knee) and RS03 (hip yaw, two ankle cranks per leg); MIT-mode driver; policy 50 Hz; recommended training gains hip 135/6, hip yaw 75/4, knee 130/6, crank 60/2.7; measured output-side rotor inertia RS04 0.0773 and RS03 0.0638 kg m^2.
- Related earlier note (not repeated here): `docs/research_raw/2026-10-14_pd_gain_criteria_sources.md` (how to choose gains: BeyondMimic rule, Hwangbo, Margolis, Bronars).

---

## A. Matching a real joint or actuator to the simulation

### S1. Impedance Matching: Enabling an RL-Based Running Jump in a Quadruped Robot
Neil Guan, Shangqun Yu, Shifan Zhu, Donghyun Kim. arXiv 2404.15096v2, 30 Apr 2024. https://arxiv.org/abs/2404.15096
- Method: per-joint chirp sweep on the real robot (body fixed), same sweep in simulation, grid search of simulated Kp/Kd (and rotor inertia) so that the Bode magnitude matches near the natural frequency. Phase is ignored.
  > "We conduct frequency response experiments on each joint individually using a chirp signal, sweeping the joint position command from 0.1 Hz to 25 Hz with an amplitude of 0.25 rad." <!--v:guan-->
  > "The phase plot was disregarded for two reasons: 1) the time delay, a significant determinant of the phase plot, behaves differently in simulations than in physical systems" <!--v:guan-->
  > "We randomize stiffness and damping by adding a value sampled from U12 (−2.0, 2.0) and U12 (−0.05, 0.05)N m s/rad to our commanded stiffness and damping" <!--v:guan-->
- Numbers (from the text): hardware gains Kp 17 / Kd 0.4; best-fit simulated gains Kp 20 / 17.5 / 21.5 and Kd 0.45 / 0.4 / 0.55 (hip ab-ad, hip flex-ext, knee); fit band 0.1-15 Hz (knee) and 1-10 Hz (hip); DR = additive uniform ±2 N m/rad on Kp (about ±10 %) and ±0.05 N m s/rad on Kd (about ±11 %), chosen "based off our impedance analysis" (the spread of the best-fit pairs).
- Rotor inertia is essential:
  > "The rotor inertia markedly influences the best fitted gains; thus, careful selection is crucial." <!--v:guan-->
  (with rotor inertia set to zero "our grid-search across a wide range of gains yields no match to real life")
- Caveat: quadruped, 50 Hz policy, Mini-Cheetah-class actuators; the chirp goes to 25 Hz only because the policy rate is 50 Hz (Nyquist).

### S2. Towards bridging the gap: Systematic sim-to-real transfer for diverse legged robots (PACE)
Filip Bjelonic, Fabian Tischhauser, Marco Hutter. IJRR, DOI 10.1177/02783649261459628; arXiv 2509.06342v2, 31 Jul 2026. https://arxiv.org/abs/2509.06342
- Method: fixed-base, in-air chirps at the joint-position-target level tracked by the PD controller; fit per-joint armature, viscous damping, Coulomb friction, joint bias and one global delay by CMA-ES against joint-position trajectories.
  > "We excite all joints simultaneously using chirp signals of 20 – 60 s per sequence, sweeping from a minimum to a maximum frequency." <!--v:bjel-->
  > "The chirp is applied at the joint position target level and tracked by a PD controller, such that both the real system and the simulator are driven by identical reference trajectories" <!--v:bjel-->
  > "identification and deployment should therefore use identical firmware modes." <!--v:bjel-->
- Numbers: parameter vector p = [Ia, d, tau_f, q_bias, Td] (4n+1 values); idealised joint model H(s) = e^(-s Td) P / (Ia s^2 + (d + D) s + P); Figure 1 uses a 0.1-2.0 Hz chirp; ANYmal and Tytan both fit a global delay of 7.5 ms; policies were deployed on 10 more robots "without randomization of dynamic parameters".
  > "Both ANY MAL and T YTAN identify a global delay of 7.5 ms." <!--v:bjel-->
- Identification accuracy and gain level (the only numeric tolerance found in this survey):
  > fitted Ia agrees with analytic targets within 2 % (rotor), 6 % (interface), and 15 % (added mass at radius) <!--v:bjel-->
  > higher gains did not improve identification quality but instead pushed the closed-loop poles beyond the feasible excitation bandwidth. <!--v:bjel-->
  (the same sentence gives the standard deviation across three PD settings: 0.67 %, 3.7 %, 14 %.)
- Caveat: firmware compensations add a load-independent "virtual inertia", so identify in the same firmware mode as deployment.

### S3. Berkeley Humanoid: A Research Platform for Learning-based Control
Qiayuan Liao, Bike Zhang, Xuanyu Huang, Xiaoyu Huang, Zhongyu Li, Koushil Sreenath. arXiv 2407.21781v1, 31 Jul 2024. https://arxiv.org/abs/2407.21781
- Method: "identify, do not randomise": joint-level actuators (no linkage) so rotor inertia enters as armature; identify friction and armature on hardware; narrow DR; no PD-gain or motor-strength randomisation.
  > "we opt not to randomize properties that cannot be identified in these two categories, such as a general “motor strength” ratio or PD gains" <!--v:bh-->
  > "excessive randomization may hinder successful policy learning or lead to exceedingly conservative policies" <!--v:bh-->
- Numbers (Table 6, read from the flattened table text, column order inferred): joint friction x0.9-1.1, joint armature x1.0-1.05, linkage mass x0.9-1.1, default joint position ±0.05 rad, base mass ±1 kg; no PD gain row.
- Safety context (why a hanging rig matters for a full-size humanoid):
  > "do not require a gantry crane due to their lightweight nature" <!--v:bh-->
- Caveat: the torque-current mapping near saturation was named as remaining sim-to-real gap by the authors.

### S4. Learning agile and dynamic motor skills for legged robots
Jemin Hwangbo, Joonho Lee, Alexey Dosovitskiy, Dario Bellicoso, Vassilios Tsounis, Vladlen Koltun, Marco Hutter. Science Robotics 4(26), 2019. arXiv 1901.08652. https://arxiv.org/abs/1901.08652
- Method: actuator network (MLP) that maps the history of position error and velocity to joint torque, trained on measured torque; used inside the simulator instead of an ideal PD.
  > "an actuator network that outputs an estimated torque at the joints given a history of position errors (the actual position subtracted from the commanded position) and velocities" <!--v:hwang-->
  > "constructing an analytical actuator model for ANYmal takes at least three weeks" <!--v:hwang-->
- Numbers: history sampled at t-0.01 s and t-0.02 s; needs a joint torque sensor (series-elastic actuator); fixed gains Kp 50 / Kd 0.1 N m/rad used for the policy output.
- Caveat: needs torque ground truth. Our RS04/RS03 report torque from current, not from a torque sensor (UNVERIFIED accuracy: the project note `docs/experiments/2026-09-10_motor_value_validity.md` treats reported torque as scaled).

### S5. Sim-to-Real: Learning Agile Locomotion For Quadruped Robots
Jie Tan, Tingnan Zhang, Erwin Coumans, Atil Iscen, Yunfei Bai, Danijar Hafner, Steven Bohez, Vincent Vanhoucke. arXiv 1804.10332v2, 16 May 2018 (venue RSS 2018 from memory: UNVERIFIED). https://arxiv.org/abs/1804.10332
- Method: identify mass, motor friction, battery voltage; replace the simulator's ideal position servo with a DC-motor model; measure and simulate latency; randomise.
  > "if large gains are used, the motors can remain stable in simulation but oscillate in reality." <!--v:tan-->
  > "The PD servo running on the microcontroller has a lower latency (3ms) while the locomotion controller executed on TX2 has a higher latency (typically 15-19ms)." <!--v:tan-->
  > "lower bound 80% 0Nm 50% 80% 3ms 0ms 14.0V 0.5 -0.05 radian 0 radian upper bound 120% 0.05Nm 150% 120% 20ms 40ms 16.8V" <!--v:tan-->
  (Table I column order: mass, motor friction, inertia, motor strength, control step, latency, battery voltage, contact friction, IMU bias, IMU noise.)
- Claim: motor strength ±20 % (equivalent to scaling Kp and Kd together), friction 0-0.05 N m, latency 0-40 ms.

### S6. Sim-to-Real Transfer for Biped Locomotion
Wenhao Yu, Visak C. V. Kumar, Greg Turk, C. Karen Liu. arXiv 1903.01390v2, 25 Aug 2019 (venue IROS 2019 from memory: UNVERIFIED). https://arxiv.org/abs/1903.01390
- Method: two-stage system identification (before and after policy learning) of a Darwin OP2 biped; the PD actuator is augmented by a neural network that outputs state-dependent kp, kd because torque cannot be measured.
  > "we augment the conventional PDbased actuator model with a neural network to increase the expressiveness of the model, which we name Neural Network PD Actuator (NN-PD)" <!--v:yu-->
  > "the actuators on Darwin OP2 are not capable of accurately measuring the actual torque being applied" <!--v:yu-->
- Claim: when torque is not measurable, identify the effective kp(.), kd(.) from position/velocity logs instead of torque.

### S7. Dynamics Randomization Revisited: A Case Study for Quadrupedal Locomotion
Zhaoming Xie, Xingye Da, Michiel van de Panne, Buck Babich, Animesh Garg. arXiv 2011.02404v3, 25 Mar 2021. https://arxiv.org/abs/2011.02404
- Claim: randomise only the parameters that were found to cause the gap; DR is "neither necessary nor sufficient". P gain was one of four randomised items.
  > "unnecessary randomization of parameters can produce conservative policies with limited actual gains in robustness, while randomization in truly problematic parameters bridges the reality gap" <!--v:xie21-->
  > "Parameter Mass Inertia P gain Latency Range [0.8, 1.2] × default [0.5, 1.5] × default [−20 Nm/rad, 20 Nm/rad] + default [0 ms, 20 ms]" <!--v:xie21-->
- Numbers: default Kp 40 / Kd 0.5 (so ±20 N m/rad is ±50 %); a Kp 160 policy was much harder to transfer than Kp 40; the policy-robustness test lowers kp in simulation until the policy fails (a proxy for unmodelled friction).

### S8. Sim-to-Real Transfer of Robotic Control with Dynamics Randomization
Xue Bin Peng, Marcin Andrychowicz, Wojciech Zaremba, Pieter Abbeel. arXiv 1710.06537v3, 3 Mar 2018 (venue ICRA 2018 from memory: UNVERIFIED). https://arxiv.org/abs/1710.06537
  > "Parameters such as mass, damping, friction, and controller gains are logarithmically sampled, while other parameters are uniformly sampled." <!--v:peng-->
  > "[0.5, 2]× default gains" <!--v:peng-->
- Claim: the widest published gain range (0.5x-2x, log-uniform) is for a Fetch-like arm pushing a puck, with joint damping 0.2x-20x; not a legged robot.

---

## B. How an MIT-mode driver relates to a simulated PD

### S9. RobStride RS04 User Manual (English, CAN 2.0)
RobStride (Lingzu Times). File `Product Literature/CAN2.0/RS04/RS04User Manual260713.pdf`, repo https://github.com/RobStride/Product_Information at commit 384cc561e1ff079c22f68b19908526094430febd. The "Version History" table in the PDF text lists "1.0 Initial Release November 25, 2025". RS03 manual is the same layout (`RS03User Manual261013.pdf`; the RS03 numbers below are not separately checked: UNVERIFIED).
- Control law:
  > "The control logic of the operation and control mode is t_ref=Kd * (v_vset-v_actual)+Kp * (p_set-p_actual)+t_ff. Tref is converted to the expected iq current through an internal formula and output through the current loop" <!--v:rs04-->
- MIT-protocol frame (Command 3, 11-bit ID): position 16 bit over ±12.57 rad (motor side), velocity 12 bit over ±15 rad/s, Kp 12 bit over 0-5000, Kd 12 bit over 0-100, torque 12 bit over ±120 N m.
  > "Byte4: Low 8 bits: Kp [0~4096], corresponding to 0 ~ 5000" <!--v:rs04-->
  > "Kd [0~4096], corresponding to 0 ~ 100" <!--v:rs04-->
  - Derived here (not in the manual): Kp step 5000/4095 = 1.22 N m/rad, Kd step 100/4095 = 0.0244 N m s/rad, torque step 240/4095 = 0.0586 N m; with Kp 60 the step is 2 %, with Kd 2.7 it is 0.9 %.
- Feedback frame: angle ±4 pi (16 bit), velocity ±15 rad/s (16 bit), torque ±120 N m (16 bit); active reporting is off by default and, when switched on, the default interval is 10 ms.
  > "Target speed [0~4096], corresponding to (-15 rad/s ~ 15 rad/s)" <!--v:rs04-->
  > "01 To enable active reporting, the default reporting interval is 10ms" <!--v:rs04-->
- Hardware facts: reduction 9:1, 14-bit encoder, torque constant 2.1 N m/Arms.
  > "Deceleration ratio: 9:1" <!--v:rs04-->
- Safety remark in the manual (regeneration):
  > "the motor generates electricity under this condition and requires power supply to prevent overvoltage" <!--v:rs04-->
- NOT in the manual: the driver's internal loop rate and velocity filter. The manual lists parameters `spd_filt_gain` (default 0.1) and `cur_filt_gain` but does not say whether MIT-mode velocity uses them: UNVERIFIED.

### S10. bgkatz/motorcontrol (open-source MIT-Cheetah / Mini-Cheetah motor firmware)
Ben Katz. https://github.com/bgkatz/motorcontrol at commit 93bbc60676ba4c2c0c64466c2b921eeb9d9fb1df; files `Core/Src/foc.c`, `Core/Src/position_sensor.c`, `Core/Inc/hw_config.h`, `Core/Inc/position_sensor.h`.
- Reference implementation of the "MIT mode" law, run in the 40 kHz current-loop interrupt:
  > `float torque_des = controller->kp*(controller->p_des - controller->theta_mech) + controller->t_ff_filt + controller->kd*(controller->v_des - controller->dtheta_mech);` <!--v:katzfw-->
  > `#define DT .000025f // Loop period` <!--v:katzfw-->
  > `controller->t_ff_filt = 0.9f*controller->t_ff_filt + 0.1f*controller->t_ff;` <!--v:katzfw-->
- Velocity is a finite difference over N_POS_SAMPLES = 20 position samples (a 19-sample span, 0.475 ms at 40 kHz) of a 16-bit encoder (ENC_CPR 65536).
  > `encoder->velocity = TWO_PI_F*((float)(encoder->count_buff[0] - encoder->count_buff[N_POS_SAMPLES-1]))` <!--v:katzfw-->
- Safety: torque command is cut above the soft bus-voltage limit.
  > `if(controller->v_bus > V_BUS_MAX){controller->i_q_des = 0;}` <!--v:katzfw-->
- Caveat: RobStride's firmware is closed. This file shows the design the protocol comes from, not RobStride's exact numbers.

### S11. A Low Cost Modular Actuator for Dynamic Robots (Katz thesis)
Benjamin G. Katz. S.M. thesis, MIT Mechanical Engineering, 2018 (advisor Sangbae Kim). https://dspace.mit.edu/handle/1721.1/118671 ; PDF https://dspace.mit.edu/bitstream/handle/1721.1/118671/1057343368-MIT.pdf
- UNVERIFIED: the PDF download returned HTTP 429 (Retry-After 900 s) on 2026-10-16, so the text was not opened. Bibliographic data only (title page seen in the DSpace record via search). Use S10 for the controller details until the thesis is read.

### S12. go_motor_control README (Unitree GO-M8010-6 driver, third party)
MuShibo, https://github.com/MuShibo/go_motor_control at commit 05faaf3cbd056f4aad2453bfeb0bc89305eb1860 (secondary source, not Unitree documentation).
  > "we need to convert kp and kd on the output side by dividing them by the square of r" <!--v:gorm-->
- Claim: some MIT-style drivers (Unitree GO-M8010-6, ratio 6.33) take gains in rotor-side units; output-side Kp / r^2 = rotor-side Kp. The RobStride manual states angles in "motor" units for the MIT frame (position ±12.57 rad = ±4 pi): whether the RS04 Kp is rotor-side or output-side is not stated in the text I read: UNVERIFIED. Test it on the bench (apply a known torque by hanging a known mass).

### S13. Proprioceptive Actuator Design in the MIT Cheetah
Patrick M. Wensing, Albert Wang, Sangok Seok, David Otten, Jeffrey Lang, Sangbae Kim. IEEE Trans. Robotics 33(3), 2017, DOI 10.1109/TRO.2016.2640183 (author's proof PDF read).
  > "Proprioceptive actuation uses collocated force control at the joints to effectively control contact interactions at the feet under dynamic conditions." <!--v:wensing-->
- Claim: torque is controlled by motor current at the joint (low gear ratio, backdrivable); the same idea as our quasi-direct-drive RS04/RS03.

---

## C. Gain randomisation ranges in published locomotion work

### S14. Li et al. 2024: Reinforcement Learning for Versatile, Dynamic, and Robust Bipedal Locomotion Control (Cassie)
Zhongyu Li, Xue Bin Peng, Pieter Abbeel, Sergey Levine, Glen Berseth, Koushil Sreenath. arXiv 2401.16889v2, 26 Aug 2024. https://arxiv.org/abs/2401.16889
  > "adding a range of ±30% deviation from their default values. This randomization is applied independently to each joint-level PD controller." <!--v:li24-->
  > "PD gains used on each motor (40% higher than the default value)" <!--v:li24-->
- Claim: Cassie PD-gain DR = ±30 % of default, independent per joint, justified as "diversity in motor responses that can mimic ... uncertain motor dynamics"; a +40 % gain shift is used as a held-out stress test.

### S15. Li et al. 2023: Robust and Versatile Bipedal Jumping Control through Reinforcement Learning (Cassie)
Zhongyu Li, Xue Bin Peng, Pieter Abbeel, Sergey Levine, Glen Berseth, Koushil Sreenath. arXiv 2302.09450v2, 1 Jun 2023 (venue RSS 2023 from memory: UNVERIFIED). https://arxiv.org/abs/2302.09450
  > "We also randomize the PD gains used in the joint-level PD controllers (since our policy outputs target motor positions). The range is ±30% of the default value." <!--v:li23-->
  > "(4 times the default value) to approximate the motor aging issues on the hardware" <!--v:li23-->
- Claim: same ±30 % on Kp and Kd; joint damping is randomised with an upper bound of 4x default to cover motor ageing.

### S16. Booster Gym (Booster T1) paper and configuration file
Yushi Wang, Penghui Chen, Xinyu Han, Feng Wu, Mingguo Zhao. arXiv 2506.15132v1, 18 Jun 2025. https://arxiv.org/abs/2506.15132 ; config https://github.com/BoosterRobotics/booster_gym (`envs/T1.yaml`, commit da396a06d6eed99e2de72d7749c48ee8748950f9).
  > "The motor’s PD controller is employed with predefined Kp and Kd gains, while joint velocity targets and feedforward torque are set to zero, aligning with the simulation configuration." <!--v:booster-->
  > "The policy operates at a rate of 50 Hz, while the internal PD controller runs at a higher frequency" <!--v:booster-->
  > dof_stiffness: range: [0.95, 1.05] operation: "scaling" distribution: "uniform" <!--v:boosteryaml-->
  > dof_damping: range: [0.95, 1.05] operation: "scaling" distribution: "uniform" <!--v:boosteryaml-->
- Numbers: T1 gains Hip 200/5, Knee 200/5, Ankle 50/1 (N m/rad, N m s/rad); stiffness and damping DR ±5 % uniform; joint friction additive 0-2.0 (units not stated in the file: UNVERIFIED). The paper gives no numeric justification for the ±5 % (UNVERIFIED: not found in the text).

### S17. Walk These Ways (Margolis) paper and config
Gabriel B. Margolis, Pulkit Agrawal. arXiv 2212.03238v1, 6 Dec 2022 (venue CoRL 2022 from memory: UNVERIFIED). https://arxiv.org/abs/2212.03238 ; code https://github.com/Improbable-AI/walk-these-ways at commit 0e7236bdc81ce855cbe3d70345a7899452bdeb1c (`go1_gym/envs/base/legged_robot_config.py`).
  > "The position targets are tracked using a proportionalderivative controller with kp = 20, kd = 0.5." <!--v:wtw-->
  > "we train a policy that is robust to a range of robot’s body mass, motor strength, joint position calibration, the ground friction" <!--v:wtw-->
  > Kp_factor_range = [0.8, 1.3] <!--v:wtwcode-->
  > Kd_factor_range = [0.5, 1.5] <!--v:wtwcode-->
- Claim: the paper randomises motor strength 90-110 % and joint calibration ±0.02 rad (Table of ranges) and keeps kp 20 / kd 0.5 fixed. The code has Kp 0.8-1.3 and Kd 0.5-1.5 factor ranges, but `randomize_Kp_factor = False` and `randomize_Kd_factor = False` in the base config: whether they were on for the paper is UNVERIFIED.

### S18. NVIDIA Isaac Lab: `randomize_actuator_gains` and task configurations
https://github.com/isaac-sim/IsaacLab at commit 9b6d2d822aa2879c1ef593bcca6118b04429d0cb (`source/isaaclab/isaaclab/envs/mdp/events.py`, `source/isaaclab_tasks/...`).
  > "Randomize the actuator gains in an articulation by adding, scaling, or setting random values." <!--v:ilev-->
  > "stiffness_distribution_params": (0.8, 1.2), <!--v:ilcfg-->
  > "damping_distribution_params": (0.8, 1.2), <!--v:ilcfg-->
- Numbers (grep of all uses, `randomize_actuator_gains`): legged `contrib/dr_legs/hold_pose_env_cfg.py` stiffness x0.8-1.2, damping x0.8-1.2 (armature x0.8-1.2); arms and hands x0.5-2.0, x0.75-1.5 (stiffness), x0.3-3.0 (damping), `contrib/deploy/reach` x0.9-1.1 and x0.75-1.5. No locomotion velocity task in this checkout uses it (no match in grep): the "default" for Isaac Lab locomotion is no gain DR.
- Caveat: ranges are task examples, not recommendations.

### S19. Cassie sim-to-real: Siekmann et al. 2021 (two papers)
Jonah Siekmann, Yesh Godse, Alan Fern, Jonathan Hurst. "Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition", arXiv 2011.01387v2. And Siekmann, Green, Warila, Fern, Hurst, "Blind Bipedal Stair Traversal via Sim-to-Real Reinforcement Learning", RSS 2021, arXiv 2105.08328v1. https://arxiv.org/abs/2011.01387 ; https://arxiv.org/abs/2105.08328
  > "Joint damping Nms/rad [0.3, 4.0] × default values Joint mass kg [0.5, 1.5] × default values" <!--v:sk-->
  > "Range [0.5, 3.5] × default values [0.5, 1.7] × default values" <!--v:sk2-->
- Claim: the Oregon State / Agility Cassie recipe randomises joint damping (0.3-4x, later 0.5-3.5x) and joint mass, plus encoder offset ±0.05 rad and execution rate 37-42 Hz, and does not randomise PD gains in these tables.

### S20. Humanoid-Gym (Gu et al. 2024)
Xinyang Gu, Yen-Jen Wang, Jianyu Chen. arXiv 2404.05695v2, 18 May 2024. https://arxiv.org/abs/2404.05695
  > "Motor Strength Payload Unit rad rad/s rad/s rad ms % kg Range [-0.05, 0.05] [-0.5, 0.5] [-0.1, 0.1] [-0.03, 0.03] [0, 10] [0.1, 2.0] [95, 105] [-5, 5]" <!--v:hgym-->
- Claim: Humanoid-Gym (RobotEra XBot): system delay 0-10 ms, friction 0.1-2.0, motor strength 95-105 %, payload ±5 kg; no separate Kp/Kd row.

---

## D. Safety practice for first motor tests

### S21. moteus PID tuning guide (mjbots)
https://mjbots.github.io/moteus/guides/pid-tuning/ ; source `docs/guides/pid-tuning.md` at commit 8f747f4ac448d117c1390b77ca2d80c27e751a56.
  > "Increase kp by 50% at a time until the desired stiffness is reached." <!--v:moteus-->
  > "Set kd initially to 1/100th of the kp value." <!--v:moteus-->
  > "It is measured in Nm per revolution of error as measured at the output of the reducer." <!--v:moteus-->
  > "recommended to configure and use an acceleration limit with moteus that is physically achievable in your system" <!--v:moteus-->
- Claim: start with tiny kp, hand-test, raise by 50 % steps, halve on vibration; kd starts at kp/100; always limit the commanded acceleration. (Revolution units: moteus kp is N m per revolution, not per radian.)

### S22. Unitree `unitree_rl_gym` deployment README
https://github.com/unitreerobotics/unitree_rl_gym , file `deploy/deploy_real/README.md`, commit 276801e46c5d433564f24658bac64f254b7d2d4b.
  > "Start the robot in the hoisting state and wait for the robot to enter `zero torque mode`" <!--v:unitree-->
  > "the robot joints are in the damping state in the `debugging mode`." <!--v:unitree-->
- Claim: vendor deploy procedure for a humanoid: hoist, zero-torque, damping mode, then command the default pose, then the policy.

### Safety items already listed above
- S3 (gantry crane for full-size humanoids), S9 (regenerative overvoltage warning), S10 (bus-voltage torque cut).

---

## E. Classical references

### S23. Åström and Hägglund, Advanced PID Control (ISA, 2006), Chapter 6 "PID Design"
Karl J. Åström, Tore Hägglund. Chapter 6 PDF (course copy): https://skoge.folk.ntnu.no/puublications_others/books/%C3%85strom-2006_Advanced%20PID%20Control/6.PID%20Design.pdf
  > "Increase the gain slowly until the process starts to oscillate." <!--v:ah6-->
  > "the closed-loop systems obtained have poor damping and high sensitivity" <!--v:ah6-->
  > "Two classical methods for determining the parameters of PID controllers were presented by Ziegler and Nichols in 1942." <!--v:ah6-->
- Claim: Ziegler-Nichols frequency method needs the loop to be driven to sustained oscillation, and the resulting loop has poor damping. This is why it is not used for a motor that carries a human-scale leg.

### S24. Step-response and damping definitions (secondary, Wikipedia raw text, fetched 2026-10-16)
"Logarithmic decrement", "Settling time", "Rise time". https://en.wikipedia.org/wiki/Logarithmic_decrement ; https://en.wikipedia.org/wiki/Settling_time ; https://en.wikipedia.org/wiki/Rise_time
  > The method of logarithmic decrement becomes less and less precise as the damping ratio increases past about 0.5 <!--v:wplog-->
  > the time required for the response curve to reach and stay within a range of certain percentage (usually 5% or 2%) of the final value. <!--v:wpset-->
- Claim: log-decrement estimates zeta from two successive peaks and is unreliable for zeta above about 0.5; settling time uses a 2 % or 5 % band. Our hip and knee at the recommended gains have zeta of 0.9 or more for the rotor alone, so overshoot and log-decrement are weak tools there; use Bode or chirp fits (S1, S2).
- Formulas used in the guide (standard second-order results; textbook source not opened: UNVERIFIED): overshoot Mp = exp(-pi zeta / sqrt(1 - zeta^2)); zeta from Mp = -ln Mp / sqrt(pi^2 + ln^2 Mp).

### S25. Carried from the 2026-10-14 note (excerpts verified there)
- BeyondMimic (Liao et al., arXiv 2508.08241): "kp,j = Ij ωn², kd,j = 2Ij ζωn", zeta 2, 10 Hz, rotor-only inertia. (`docs/research_raw/2026-10-14_pd_gain_criteria_sources.md` A1)
- Bronars, Park, Agrawal (arXiv 2604.02523): "stiff, overdamped controllers amplify small modeling errors because they respond to position and velocity deviations with high torques." (same file, C1; hardware was a Franka arm, not a legged robot)
- Humanoid (company) talk: "If you're updating your PD controller at 40,000 Hz, and your action only updates at 50. That's more or less a static action for the PD controller to track." (same file, B5; talk transcript, speaker UNVERIFIED)

### S26. Textbooks and courses recommended in the guide
Textbooks were NOT opened (titles and editions UNVERIFIED against the publisher; chapter numbers differ by edition). Web pages below returned HTTP 200 on 2026-10-16 and their `<title>` was read.
- K. J. Åström, R. M. Murray, "Feedback Systems: An Introduction for Scientists and Engineers", 2nd ed., Princeton University Press, 2021 (ISBN 978-0-691-19398-4 per bookseller listings). The official site moved to https://fbsbook.org (redirects to fbswiki.org, which returned HTTP 500 on 2026-10-16; the old Caltech wiki says it is no longer maintained).
- G. F. Franklin, J. D. Powell, A. Emami-Naeini, "Feedback Control of Dynamic Systems", 8th ed., Pearson, 2019 (UNVERIFIED edition year).
- Steve Brunton, "Control Bootcamp", YouTube playlist, page title "Control Bootcamp - YouTube": https://www.youtube.com/playlist?list=PLMrJAkhIeNNR20Mz-VpzgfQs5zrYi085m
- MIT OpenCourseWare 2.004 "Dynamics and Control II": https://ocw.mit.edu/courses/2-004-dynamics-and-control-ii-spring-2008/ (page title read).
- MIT OpenCourseWare 2.14 "Analysis and Design of Feedback Control Systems": https://ocw.mit.edu/courses/2-14-analysis-and-design-of-feedback-control-systems-spring-2014/ (page title read).
- Stanford EE263 (Boyd) course page, directory index only: https://web.stanford.edu/class/ee263/ (least squares; contents UNVERIFIED).
- L. Ljung, "System Identification: Theory for the User", 2nd ed., Prentice Hall, 1999 (UNVERIFIED).
- K. J. Åström, B. Wittenmark, "Computer-Controlled Systems", 3rd ed., Prentice Hall, 1997 (UNVERIFIED).
- Control Tutorials for MATLAB and Simulink (University of Michigan, https://ctms.engin.umich.edu/CTMS/): returned HTTP 403 to a script; not opened.

### S27. MuJoCo XML reference: position actuator `kv` and integrators
Google DeepMind. `doc/XMLreference.rst` (main branch, fetched 2026-10-16). https://mujoco.readthedocs.io/en/stable/XMLreference.html
  > Implicit integration works for :at:`damping` when using the Euler integrator but not for :at:`kv`. <!--v:mjxml-->
  > To get implicit integration for :at:`kv`, implicit or implicitfast is required <!--v:mjxml-->
- Claim: the simulated PD velocity term (`kv`) is integrated implicitly only with the implicit or implicitfast integrator; the driver's PD is continuous (about 40 kHz in the reference firmware, S10). The training step is 5 ms with 4 sub-steps per policy action (local `velocity_env_cfg.py` lines 455 and 460): so the simulated PD is sampled at 200 Hz.

---

## Claims table (for the guide)

| # | Claim | Number | Source |
|---|---|---|---|
| 1 | Chirp sweep at the position-target level, same in sim and real, is the standard way to compare closed-loop joint response | 0.1-25 Hz, 0.25 rad (S1); 20-60 s per chirp (S2) | S1, S2 |
| 2 | Match magnitude (bandwidth, peak) near the natural frequency; skip phase because delay is hard to match | 0.1-15 Hz knee, 1-10 Hz hip | S1 |
| 3 | Identify armature, viscous damping, Coulomb friction, bias and one global delay per joint | 4n+1 parameters; delay 7.5 ms (ANYmal, Tytan) | S2 |
| 3b | Identified inertia accuracy: 2 % rotor, 6 % interface, 15 % added mass; moderate gains identify better than high gains | 2 / 6 / 15 % | S2 |
| 4 | Identify, do not randomise PD gains, if you can identify | armature x1.0-1.05, friction x0.9-1.1 | S3 |
| 5 | Gain DR in published legged work | Cassie ±30 % (S14, S15); T1 ±5 % (S16); Guan ±2 N m/rad about ±10 % (S1); Xie ±20 N m/rad on 40 (±50 %) (S7); Walk These Ways code Kp 0.8-1.3, Kd 0.5-1.5 (S17); Isaac Lab legged example 0.8-1.2 (S18); Peng 0.5-2x (S8) | as listed |
| 6 | MIT mode computes PD inside the driver at a rate far above the policy rate | 40 kHz reference firmware (S10); policy 50 Hz (S16) | S10, S16 |
| 7 | RS04 MIT frame quantisation | Kp step 1.22, Kd step 0.0244 | S9 (derived) |
| 8 | Ziegler-Nichols needs sustained oscillation | - | S23 |
| 9 | Simulated PD is sampled at the physics step; the driver loop is far faster | 5 ms step (local), 25 us (S10) | S27, S10 |
| 10 | First tests: tiny kp, 50 % steps, kd = kp/100, acceleration limit, hoist, zero-torque first | - | S21, S22 |
