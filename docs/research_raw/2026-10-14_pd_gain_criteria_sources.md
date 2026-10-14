# Raw sources: criteria for choosing joint PD gains (Kp/Kd) of RL position-target legged policies (2026-10-14)

Question (from the user): what criteria define an appropriate Kp/Kd for a position-target RL policy with a PD loop at the motor?
Is it "the value with which the robot barely stands"? This file only collects sources and verbatim excerpts. No synthesis beyond the claims table at the end.

Method note: arXiv PDFs were downloaded and run through `pdftotext`; the quoted strings were read from that text (line breaks joined, math symbols as extracted).
GitHub files were fetched from raw.githubusercontent.com at the commit named in each entry. Anything not checked against the primary text is marked UNVERIFIED.

Our robot (for later comparison, from the task brief, not re-measured here): 23.6 kg lower body; RS04 hips/knee Kp 150/220, Kd 6; RS03 ankle cranks Kp 22.3, Kd 1.41.

---

## A. Rule-based derivations (criteria 1, 2, 7)

### A1. BeyondMimic: From Motion Tracking to Versatile Humanoid Control via Guided Diffusion
Qiayuan Liao, Takara E. Truong, Xiaoyu Huang, Yuman Gao, Guy Tevet, Koushil Sreenath, C. Karen Liu. arXiv 2025 (v4). https://arxiv.org/abs/2508.08241 (PDF read: v2, https://arxiv.org/pdf/2508.08241v2)

- Claim: gains come from a natural frequency / damping ratio rule on reflected inertia, citing a workshop talk (Raibert and Farshidian) as origin.
  > "we heuristically set joint stiffness and damping following Raibert et. al.: kp,j = Ij ωn², kd,j = 2Ij ζωn"
  > "Ij = kg,j² Imotor,j is the reflected inertia of the jth joint"
- Claim: zeta = 2 chosen because inertia is underestimated (rotor only, link inertia ignored); 10 Hz chosen "relatively low" for compliance.
  > "We choose a damping ratio ζ = 2 (overdamped) rather than 1 (critically damped) as in Raibert et. al. [38], since the inertia is typically underestimated by considering only the motor armature and ignoring the apparent link inertia."
  > "The natural frequency is set to a relatively low value of 10 Hz, which promotes compliance with moderate gains."
- Claim: action scale tied to gains and torque limit.
  > "αj = 0.25 τj,max / kp,j, with τj,max representing the maximum allowable joint torque for joint j."
  > "This heuristic assumes that contacts generally occur around q̄j and that the robot hardware design ensures the maximum joint torque is proportional to the expected load."
  > "At low gains, these setpoints are not intended as desired position targets; rather, they serve as intermediate variables to generate desired torques"
- Claim (high-gain downsides): 
  > "policies trained under such high-impedance settings are often impractical for hardware deployment, as they amplify sensor noise, reduce the passive compliance needed for impact absorption, and hinder implicit torque information"
  (the same paragraph says prior works [9], [16], [33] "adopt high impedance for precise tracking", i.e. the high-gain camp)
- Origin ref [38]: M. Raibert and F. Farshidian, "Workshop on whole-body control and bimanual manipulation: Applications in humanoids and beyond", RSS 2025 workshop talk. No public text found: UNVERIFIED beyond the citation.

### A2. mjlab G1 constants (local file) and upstream BeyondMimic code
- Local: `/home/syaro/MikuchanRemote/Human-Pygmalion/mujoco-sim/mjlab/src/mjlab/asset_zoo/robots/unitree_g1/g1_constants.py` (mjlab submodule HEAD b2d68f9eb4e47bd1f62ff84d149fd6247907aa95, lines ~105-125 and 278-286)
- Upstream: https://github.com/HybridRobotics/whole_body_tracking (main, ls-remote HEAD cd65172032893724b445448818c34165846d847d at fetch time), file `source/whole_body_tracking/whole_body_tracking/robots/g1.py`. Raw fetch: https://raw.githubusercontent.com/HybridRobotics/whole_body_tracking/main/source/whole_body_tracking/whole_body_tracking/robots/g1.py
- Claim: the code implements A1 exactly; identical constants upstream and in mjlab.
  > `NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz` ; `DAMPING_RATIO = 2.0`
  > `STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2` ; `DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ`
  > local: `G1_ACTION_SCALE[n] = 0.25 * e / s`  (e = effort_limit, s = stiffness)
- Claim: in mjlab the "inertia" is rotor inertia only, reflected through the gearbox (no link inertia): local `mjlab/utils/actuator.py` `reflected_inertia_from_two_stage_planetary` = `r1 + r2 + r3` with `r1 = rotor_inertia[0] * (gear_ratio[1]*gear_ratio[2])**2` etc. Ankle/waist 4-bar joints: armature approximated as the sum of the two actuators (comment in g1_constants.py: "we assume a nominal 1:1 gear ratio").
- Computed here from the local file (not a source claim): armature -> Kp / Kd at 10 Hz, zeta 2 -> action_scale (0.25*effort/Kp):
  | actuator | armature kg m^2 | Kp N m/rad | Kd N m s/rad | effort N m | action_scale |
  |---|---|---|---|---|---|
  | 5020 | 0.003609725 | 14.25 | 0.907 | 25 | 0.439 |
  | 7520-14 | 0.010177520 | 40.18 | 2.558 | 88 | 0.548 |
  | 7520-22 | 0.025101925 | 99.10 | 6.309 | 139 | 0.351 |
  | 4010 | 0.004250000 | 16.78 | 1.068 | 5 | 0.075 |

### A3. OmniXtreme (re-uses A1)
Yunshen Wang, Shaohang Zhu, Peiyuan Zhi, Yuhan Li, Jiaxin Li, Yong-Lu Li, Yuchen Xiao, Xingxing Wang et al. arXiv 2602.23843 (2026). https://arxiv.org/abs/2602.23843
- Claim: BeyondMimic gain rule adopted unchanged by another group on Unitree G1.
  > "Following the methodology of BeyondMimic, we calculate the PD gains, kp and kd, based on the natural frequency (ω), damping ratio (ζ), and the armature inertia (I) of the motors"
  > "the parameters consist of a 10 Hz natural frequency and a damping ratio of 2."

### A4. Isaac Lab URDF converter docstring (same rule, different convention)
https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/sim/converters/urdf_converter_cfg.py (main, HEAD b9a673a29f3a7f09508f339c3ec370fdda2f542f at fetch)
- Claim: standard formula with "total equivalent inertia at the joint" (i.e. link side included), damping-ratio taxonomy.
  > "P = m · f², D = 2 · r · f · m where f is the natural frequency, r is the damping ratio, and m is the total equivalent inertia at the joint."
  > "r = 1.0 is a critically damped system, r < 1.0 is underdamped, r > 1.0 is overdamped."
- Caveat: docstring writes f without 2*pi; whether f is Hz or rad/s is not stated there: UNVERIFIED (BeyondMimic uses 2*pi*10).

### A5. Winter et al. 1998, human ankle stiffness from the same formula (biological analog of K = I w^2)
D. A. Winter, A. E. Patla, F. Prince, M. Ishac, K. Gielo-Perczak. "Stiffness control of balance in quiet standing." J Neurophysiol 80(3):1211-21, 1998. doi:10.1152/jn.1998.80.3.1211. https://pubmed.ncbi.nlm.nih.gov/9744933/
- Claim: effective ankle stiffness estimated as Ke = I wn^2 with I = body inertia about the ankle; the inverted pendulum is severely underdamped.
  > "The effective stiffness of the system, Ke, is then estimated from Ke = I omega n2"
  > "the inverted pendulum model is severely underdamped, approaching the undamped condition"

---

## B. Load / gravity / standing criterion (criterion 3 and the "barely stands" idea)

### B1. Loram and Lakie 2002 (J Physiol)
I. D. Loram, M. Lakie. "Direct measurement of human ankle stiffness during quiet standing: the intrinsic mechanical stiffness is insufficient for stability." J Physiol 545(3):1041-53, 2002. doi:10.1113/jphysiol.2002.025049. PMC2290720. https://pubmed.ncbi.nlm.nih.gov/12482906/
- Claim: a minimal stabilising stiffness exists (gravity toppling stiffness m g h); passive ankle stiffness measured at 91% of it, so extra active torque modulation is needed.
  > "On average the directly determined intrinsic stiffness is 91 +/- 23 % (mean +/- S.D.) of that necessary to provide minimal stabilisation."
  > "intrinsic ankle stiffness is not quite sufficient to stabilise the body or pendulum"

### B2. Casadio, Morasso, Sanguineti 2005 (Gait Posture)
"Direct measurement of ankle stiffness during quiet standing: implications for control modelling and clinical application." Gait Posture 21(4):410-24, 2005. doi:10.1016/j.gaitpost.2004.05.005. https://pubmed.ncbi.nlm.nih.gov/15886131/
- Claim: with larger (1 deg) disturbances, intrinsic stiffness is 64% of critical; "critical stiffness" is the normalising quantity.
  > "The intrinsic ankle stiffness was evaluated to be 64+/-8% of the critical stiffness for test disturbances of the order of 1 degrees."

### B3. Loram, Maganaris, Lakie 2007 (J Physiol): defines the m g h normaliser in plain words
"The passive, human calf muscles in relation to standing: the non-linear decrease from short range to long range stiffness." PMC2277155. https://pmc.ncbi.nlm.nih.gov/articles/PMC2277155
- Claim: m g h = gravitational toppling torque per unit angle; below 100% of it static stability is lost without active control; reported 91% (Loram and Lakie 2002) and 65% (Casadio 2005); the passive share is contested (10-15% vs 65-90%).
  > "found the stiffness to be less than 100% of the gravitational toppling torque per unit angle 'mgh' and thus insufficient for static stability"
  > "Some groups have revealed a substantial intrinsic stiffness (65-90%) normalized to load stiffness, 'mgh'."
  (the same abstract adds that others "believe the passive contribution to be small (10-15%)")
- Relevance note for the planner: this is the biological inverted-pendulum bound (stiffness >= m g h about the ankle), not an RL-robot gain recipe. No robot paper in this file cites it as a gain criterion.

### B4. Morgenstein, Masetty, Welch, Sentis: "Too Much of a Good Thing: When sim2real Efforts Impede Policy Learning (And What to Do About It)"
Apptronik and UT Austin. arXiv 2606.02636 (v2, 3 Jun 2026), position paper, Apptronik Apollo. https://arxiv.org/abs/2606.02636
- Claim: gravity-driven lower bound on gains grows with robot mass and height, and conflicts with exploration.
  > "For a larger, heavier humanoid, higher gains are required to track desired joint positions in the presence of gravity."
  > "the gains required to compensate for gravity and track desired joint positions grows with both the mass and height of the robot" (paper's scaling result: Kp proportional to m*L, from Kp = I wn^2)
  > "either set the gains appropriately for sim2real transfer, but suffer poor sample efficiency ... or intentionally set lower gains, allowing for better policy learning but with poor sim2real transfer."
- Caveat: position/opinion paper; scaling argument is the authors', cites Raibert and Farshidian talk for Kp = I wn^2.

### B5. Humanoid (company) talk transcript: "Deployment-Ready RL: Pitfalls, Lessons, and Best Practices"
https://thehumanoid.ai/deployment-ready-rl-pitfalls-lessons-and-best-practices/ (page published 2025-08-22, modified 2026-08-12; byline "Humanoid's team"; speaker not named on the page, speaker mentions a Spot at "my lab": speaker identity UNVERIFIED). Transcript text, timestamps in the page.
- Claim: two ways to choose gains: (i) classic PD "natural frequency, critical damping" gives very high gains that stand up easily but hurt exploration; (ii) low gains + gravity-comp feed-forward view.
  > "you're gonna pick KP to be the square of your natural frequency, and use that KD to be critical damping ... the resulting gains are very, very high."
  > "that will transfer to hardware, you'll be able to stand the robot up no problem ... But you have substantially reduced exploration performance in simulation."
- Claim: explicit heuristic from torque limit and range of motion, and Kd = Kp/20.
  > "we set Kp such that when we're at one extreme of that range of motion, you can't apply more than your torque limit."
  > "if you divide Kp by about 20, I find that works really nicely in practice."
- Claim: policy at 50 Hz is quasi-static for the firmware PD loop.
  > "If you're updating your PD controller at 40,000 Hz, and your action only updates at 50. That's more or less a static action for the PD controller to track."
- Claim: train low, deploy high via distillation if hardware cannot track low-gain torque.
  > "The reason we use low torques in sim is primarily for exploration. ... you can then distill that ... into a policy that uses the higher gains that do work on your hardware"
  > "Lower PD gains will give you better exploration"
- Caveat: talk transcript, not peer reviewed; the speaker calls the Kp rule "just a heuristic".

---

## C. Low-gain / compliance and sim-to-real (criterion 4)

### C1. Tune to Learn: How Controller Gains Shape Robot Policy Learning
Antonia Bronars, Younghyo Park, Pulkit Agrawal (MIT Improbable AI Lab). arXiv 2604.02523 (2 Apr 2026). https://arxiv.org/abs/2604.02523
- Claim: the only controlled gain-grid study found; RL training succeeds at any gain if hyperparameters are tuned, but sim-to-real worsens for stiff and overdamped gains.
  > "reinforcement learning can succeed across all gain regimes given compatible hyperparameter tuning, and (3) sim-to-real transfer is harmed by stiff and overdamped gain regimes."
  > "gain selection should instead be guided by learnability"
  > "stiff, overdamped controllers amplify small modeling errors because they respond to position and velocity deviations with high torques."
  > "Lowering the policy frequency ... reduces the prevalence of high-frequency oscillation during sim-to-real transfer."
- Scope caveat (checked in the text): hardware sim-to-real experiments use a Franka Research 3 arm (reaching), not a legged robot; locomotion appears only as a simulated Unitree G1 RL task.
  > "the dominant failure mode is high-frequency oscillation, which persists even with domain randomization"

### C2. Hwangbo et al. 2019 (ANYmal)
"Learning agile and dynamic motor skills for legged robots." Science Robotics 4(26), 2019. arXiv 1901.08652. https://arxiv.org/abs/1901.08652
- Claim: Kp chosen as torque range / motion range; performance robust to gain change; PD with low gains on the SEA.
  > "fixed gains (k p = 50 Nm/rad and k d = 0.1 Nm/rad/s) ... The position gain is chosen to be roughly the nominal range of torque (±30 Nm) divided by the nominal range of motion (±0.6 rad)."
  > "From our experience, the final locomotion performance is robust against a small change in gains. For instance, increasing the position gain to 80 Nm/rad does not noticeably change the performance."
  > "The velocity gain is chosen to be sufficiently high to prevent unwanted oscillation on the real robot."
- Claim: position-target policies start as a standing controller (relevant to the "stand" idea, but as an RL-training advantage, not a gain criterion).
  > "a position policy has an advantage in training since it starts as a standing controller whereas a torque controller initially creates many trajectories that result in falling."
- Note: the quoted fixed-gain PD is how the policy's position output is converted to torque in the paper's policy-network description (the actuator network is a separate module); supplement, "Policy training details".

### C3. Rudin et al. 2021 + legged_gym configs
N. Rudin, D. Hoeller, P. Reist, M. Hutter. "Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning." CoRL 2021. arXiv 2109.11978. https://arxiv.org/abs/2109.11978
- Claim: sim step limited by PD discretisation; gains reduced for the smaller A1.
  > "This limit is imposed by the actuator network (approximating a discrete-time PD-controller) becoming unstable and not by the simulation itself."
  > "we remove the actuator model of the ANYdrive motors, reduce PD gains and the torque penalties"
- legged_gym configs (https://github.com/leggedrobotics/legged_gym, HEAD 8fa29acc6fd1910c3d9659eef6310bdd301cde0a at fetch), `legged_gym/envs/a1/a1_config.py`: `stiffness = {'joint': 20.}`, `damping = {'joint': 0.5}`, `action_scale = 0.25`, `decimation = 4`. `legged_gym/envs/anymal_c/mixed_terrains/anymal_c_rough_config.py`: `stiffness = {'HAA': 80., 'HFE': 80., 'KFE': 80.}`, `damping = {... 2.}`, `action_scale = 0.5`, `use_actuator_network = True`. (Note: the "Kp 20 / Kd 0.5" figure is the A1 config, not ANYmal; ANYmal-C is 80/2.)

### C4. Margolis et al. 2022 (Mini Cheetah)
"Rapid Locomotion via Reinforcement Learning." RSS 2022. arXiv 2205.02824. https://arxiv.org/abs/2205.02824
- Claim: deliberately low fixed gains, not tuned.
  > "The proportional gain is 20, and the derivative gain is 0.5. We chose these low gains to promote smooth motions and did not tune them during our experiments."

### C5. Tan et al. 2018 (Minitaur)
"Sim-to-Real: Learning Agile Locomotion For Quadruped Robots." RSS 2018. arXiv 1804.10332. https://arxiv.org/abs/1804.10332
- Claim: large gains can be stable in an implicit sim servo but oscillate on hardware.
  > "if large gains are used, the motors can remain stable in simulation but oscillate in reality."

### C6. Peng and van de Panne 2017
"Learning Locomotion Skills Using DeepRL: Does the Choice of Action Space Matter?" arXiv 1611.01055 (SCA 2017). https://arxiv.org/abs/1611.01055
- Claim: position-target (PD) action space beats torque; copes better with low policy query rates; gains are hand-specified, no gain sweep.
  > "PD performs well across all examples, achieving comparable-to-the-best performance for all motions."
  > "Actuation models that incorporate low-level feedback such as PD and Vel, appear to cope more effectively to lower query rates, while the Tor degrades more rapidly at lower query rates."
  > "kpi and kdi are manually-specified gains"
- Caveat: simulated characters only.

### C7. ECO: Energy-Constrained Optimization with RL for Humanoid Walking
Weidong Huang, Jingwen Zhang, Jiongye Li, Shibowen Zhang, Jiayang Wu, Jiayi Wang, Hangxin Liu, Yaodong Yang, Yao Su et al. arXiv 2602.06445 (Feb 2026, T-ASE accepted). https://arxiv.org/abs/2602.06445
- Claim: lowering ankle gains fixed ground-contact instability; gain alignment sim/hardware is essential. Platform BRUCE 4.8 kg.
  > "high PD gains at the ankle lead to instability upon ground contact due to sensor noise. To address this, we reduce the ankle's PD gains, making the joints more compliant and the policy less sensitive to noise."
  > "carefully tuning the proportional (Kp) and derivative (Kd) gains of the low-level PD controller is essential to align the joint-position tracking performance between simulation and hardware."
  > "BRUCE is 70cm tall, weighs 4.8kg" ; "Kp = diag(7, 10, 7, 10, 1.5) and Kd = diag(0.2, 0.4, 0.2, 0.4, 0.08) for the hip yaw, hip pitch, hip roll, knee pitch, and ankle pitch joints"

### C8. Masuda and Takahashi 2022
"Sim-to-Real Transfer of Compliant Bipedal Locomotion on Torque Sensor-Less Gear-Driven Humanoid." arXiv 2204.03897. https://arxiv.org/abs/2204.03897
- Claim: low D gain forced by latency made the joint spring-like and hard for the policy; policy outputs P gain (6 to 0.1) with D fixed at 0.1 (variable-impedance action).
  > "we could not ensure the position control's stability if the D-gain was increased beyond a certain level, so the D-gain had to be set excessively low."
  > "The low D-gain made the position control behave like a spring and was difficult for the policy to handle"
  > "The P gain is commanded at a value between 6 and 0.1, and the D gain is fixed at 0.1."

### C9. Kim, Berseth, Schwartz, Park 2023 (torque vs position action space on bipeds)
"Torque-based Deep Reinforcement Learning for Task-and-Robot Agnostic Learning on Bipedal Robots Using Sim-to-Real Transfer." arXiv 2304.09434. https://arxiv.org/abs/2304.09434
- Claim: a gain sweep (Kp = Kdefault/{1,2,4,8}) on the human-sized TOCABI: the default (high) gain failed to learn walking; the gain does not transfer between robots; gravity-bearing joints need larger gain.
  > "It is observed that the robot cannot learn to walk when a high gain of Kp = Kdefault is used."
  > "finding an appropriate value that balances compliance and tracking performance for the given task is necessary."
  > "the gain had to be tuned for each joint by allocating a larger gain for joints that would sustain gravity."
- Caveat: argues for torque action space, so the paper is partisan; masses stated: Atlas 90 kg, TOCABI 104 kg (order as in paper: "mass of 90 kg and 104 kg, respectively", Atlas then TOCABI).

### C10. Variable-impedance policies
- Spoljaric, Yan, Lee. "Variable Stiffness for Robust Locomotion through Reinforcement Learning." arXiv 2502.09436 (2025). https://arxiv.org/abs/2502.09436
  > "fixed stiffness and damping values (Kp and Kd), requiring extensive PD gain tuning that is often task- and robot-specific"
  > "higher stiffness is required during contact with the ground, but less is needed during the swing phase."
- Radosavovic, Xiao, Zhang, Darrell, Malik, Sreenath. "Real-World Humanoid Locomotion with Reinforcement Learning" (Digit; Science Robotics 2024). arXiv 2303.03381. https://arxiv.org/abs/2303.03381
  > "The action space consists of the PD setpoints for 16 actuated joints and the predicted PD gains for 8 actuated leg joints."
  > "we run the neural network policy at 50 Hz and the joint PD controller at 1 kHz."
  (gain ranges not extracted: UNVERIFIED)

### C11. Xie et al. (Cassie), and Li et al. (Cassie IJRR)
- Z. Xie, P. Clary, J. Dao, P. Morais, J. Hurst, M. van de Panne. "Iterative Reinforcement Learning Based Design of Dynamic Locomotion Skills for Cassie." arXiv 1903.09537 (2019). https://arxiv.org/abs/1903.09537 (CoRL version: https://proceedings.mlr.press/v100/xie20a/xie20a.pdf)
  > "The policy updates its output joint PD targets once every 30 ms" ; "The embedded computer executes a PD control loop for each joint at the full 2 kHz rate"
  Claim: policy ~33 Hz vs PD 2 kHz on a real biped. Default Cassie Kp/Kd values were NOT found in the text (grep for gains empty): UNVERIFIED.
- Z. Li, X. B. Peng, P. Abbeel, S. Levine, G. Berseth, K. Sreenath. "Reinforcement Learning for Versatile, Dynamic, and Robust Bipedal Locomotion Control." IJRR 2024. arXiv 2401.16889. https://arxiv.org/abs/2401.16889
  > "incorporate randomization into the PD gains of the joint-level PD controllers, adding a range of ±30% deviation from their default values."
  Claim: gains randomised +-30% around a default, as a robustness measure. Default values not extracted: UNVERIFIED.

---

## D. High-gain / discrete-time arguments (criterion 5)

### D1. Tan, Liu, Turk 2011: Stable Proportional-Derivative Controllers
IEEE Computer Graphics and Applications 2011. https://www.cc.gatech.edu/~turk/my_papers/stable_pd.pdf (text read)
- Claim: explicit PD with high gain is numerically unstable at a given step; high-gain tracking forces small steps.
  > "when precise tracking with high gains is desired, PD servos typically produce unstable control forces which require the numerical simulation to take small time steps."
- The bound itself (dt*Kd/I < 2 etc.): a search assistant derived it (forward Euler, one joint); no citable primary source found. UNVERIFIED as a published rule. DART "Control Gain Tuning (Discrete-Time)" page (omega*dt rule of thumb ~0.1) was reported by search but the page returned 404 on fetch: UNVERIFIED.

### D2. MuJoCo documentation, "Computation > Numerical integration"
https://mujoco.readthedocs.io/en/stable/computation/index.html (fetched 2026-10-14)
- Claim: implicit-in-velocity integrators target velocity-dependent (damping) instabilities, including actuator damping; timestep too large -> instability.
  > "This approach is particularly effective in systems where instabilities are caused by velocity-dependent forces: multi-joint pendulums, bodies tumbling through space, systems with lift and drag forces, and systems with substantial damping in tendons and actuators."
  > "If it is too large, the simulation will become unstable."
  (implicitfast is the documented recommended integrator; position-gain terms are not covered by implicit-in-velocity)

### D3. Humanoid talk (B5) and Peng (C6)
Policy-rate argument: PD loop sees the 50 Hz target as quasi-static (B5); PD action spaces degrade less at low query rates than torque (C6); Tune to Learn (C1) found lower policy frequency reduced jitter for stiff gains. No source found stating a quantitative "policy at 50 Hz cannot correct faster than N Hz" rule.

---

## E. Published gain tables of comparable robots (criterion 6)

All Kp in N m/rad, Kd in N m s/rad. "Mass" is as stated; computed masses are my sums of URDF `<mass>` tags, labelled.

| Robot | Source (file/paper, commit) | hip | knee | ankle | policy / PD rate | mass |
|---|---|---|---|---|---|---|
| Unitree H1 | unitree_rl_gym `legged_gym/envs/h1/h1_config.py` @276801e4 | yaw/roll/pitch 150 / 2 | 200 / 4 | 40 / 2 | decimation 4, dt 0.005 (so 50 Hz) | URDF sum 51.6 kg (computed, whole robot) |
| Unitree G1 (12 dof) | unitree_rl_gym `legged_gym/envs/g1/g1_config.py` @276801e4 | yaw/roll/pitch 100 / 2 | 150 / 4 | 40 / 2 | decimation 4, dt 0.005 | URDF sum 32.1 kg (computed, g1_12dof.urdf) |
| Unitree G1 (mjlab/BeyondMimic) | A2 above | derived from armature | derived | derived | n/a | n/a (see A2 table: 99 for 7520-22 motors, 40 for 7520-14, 14 for 5020) |
| Booster T1 | booster_gym `envs/T1.yaml` (HEAD da396a06...) | 200 / 5 | 200 / 5 | 50 / 1 | decimation 10, dt 0.002 (50 Hz); randomization dof_stiffness and dof_damping x[0.95,1.05] | UNVERIFIED (not in the fetched yaml) |
| Berkeley Humanoid | isaac_berkeley_humanoid `.../assets/berkeley_humanoid.py` (HEAD ffc7b269...) | HR/HAA 10 / 1.5 ; HFE 15 / 1.5 | KFE 15 / 1.5 | FFE 1 / 0.1 ; FAA 1 / 0.1 | n/a | 16 kg (paper arXiv 2407.21781: "a 16 kg, fully electric drive mid-scale robot", 0.85 m) |
| MIT Humanoid | Jeon, Heim, Khazoom, Kim, arXiv 2307.10142: "Kp = 30 Nm/rad and Kd = 5 Nms/rad are fixed" (all 10 leg joints) | 30 / 5 | 30 / 5 | 30 / 5 | n/a | UNVERIFIED |
| BRUCE | ECO, arXiv 2602.06445 (C7) | yaw 7 / 0.2, pitch 10 / 0.4, roll 7 / 0.2 | 10 / 0.4 | 1.5 / 0.08 | n/a | 4.8 kg |
| ANYmal-C | legged_gym (C3) | 80 / 2 (HAA, HFE) | 80 / 2 | n/a | decimation 4 | UNVERIFIED |
| ANYmal (Hwangbo) | C2 | 50 / 0.1 all | 50 / 0.1 | n/a | n/a | UNVERIFIED |
| A1 / Mini Cheetah | C3 (A1 20 / 0.5) ; C4 (Mini Cheetah 20 / 0.5) | 20 / 0.5 | 20 / 0.5 | n/a | n/a | UNVERIFIED |
| Fourier GR1 | HiLo (arXiv 2502.03122) gives the control law tau = Kp(q_tgt - q) - Kd qdot at 1000 Hz but the gain table was not found in the text | UNVERIFIED | UNVERIFIED | UNVERIFIED | 1000 Hz torque loop | UNVERIFIED |
| Cassie | defaults not found (C11) | UNVERIFIED | UNVERIFIED | UNVERIFIED | policy ~33 Hz, PD 2 kHz | UNVERIFIED |
| Digit | gains are policy outputs (C10), ranges UNVERIFIED | | | | policy 50 Hz, PD 1 kHz | UNVERIFIED |

Primary-file links: https://github.com/unitreerobotics/unitree_rl_gym/blob/276801e46c5d433564f24658bac64f254b7d2d4b/legged_gym/envs/h1/h1_config.py ; https://github.com/unitreerobotics/unitree_rl_gym/blob/276801e46c5d433564f24658bac64f254b7d2d4b/legged_gym/envs/g1/g1_config.py ; https://github.com/BoosterRobotics/booster_gym/blob/main/envs/T1.yaml ; https://github.com/HybridRobotics/isaac_berkeley_humanoid (assets/berkeley_humanoid.py under exts/berkeley_humanoid/berkeley_humanoid/). Booster Gym paper (Wang, Chen, Han, Wu, Zhao; arXiv 2506.15132) says: "The motor's PD controller is employed with predefined Kp and Kd gains, while joint velocity targets and feedforward torque are set to zero" and that it randomizes "joint stiffness, damping, and friction, as well as communication delays."

Other relevant line from Berkeley Humanoid (arXiv 2407.21781, https://arxiv.org/abs/2407.21781): they do not randomise gains.
  > "we opt not to randomize properties that cannot be identified ... such as a general 'motor strength' ratio or PD gains, which were often used as a 'lazy approach' to approximate actuation uncertainties."

Also relevant: Bjelonic, Tischhauser, Hutter. "Towards bridging the gap: Systematic sim-to-real transfer for diverse legged robots." arXiv 2509.06342 (2025). https://arxiv.org/abs/2509.06342 : identifies per-joint armature, damping, friction and a delay (ANYmal and TYTAN both a "global delay of 7.5 ms"). Supports measuring armature rather than assuming it. Not a gain-selection recipe.

---

## F. Search for "minimum gain at which the robot barely stands" as a criterion

- No robot-RL source found that selects Kp as the minimum value at which the robot stands. A targeted search for a downward Kp sweep reporting the lowest standing Kp returned nothing (search assistant statement: no study found that "sweeps PD stiffness downward and reports the lowest Kp at which a policy can still stand"). This is a negative result of a limited search, not proof of absence.
- Closest items:
  - Gravity-driven lower bound on gain, qualitative: B4 ("higher gains are required ... in the presence of gravity"), C9 ("larger gain for joints that would sustain gravity").
  - Biological minimum-stiffness-for-stability criterion (stiffness >= m g h): B1-B3.
  - The opposite direction as default advice: B5 ("Lower PD gains will give you better exploration"), C4, C1.
  - Torque-range based choice: C2 (Kp = torque range / motion range), B5 (Kp s.t. one end of range hits torque limit), A1 (action scale = 0.25 tau_max / Kp).
- Derivation (not from a source; generic statics, flagged): holding a static gravity torque tau_g with PD target error e requires Kp*e = tau_g, so e = tau_g / Kp. The Apptronik paper (B4) and the Kim paper (C9) say gravity raises the needed gain but give no formula.

---

## G. Claims table (criterion | direction it pushes Kp | sources)

| Criterion | Pushes Kp | Sources |
|---|---|---|
| Natural frequency 10 Hz and zeta 2 on reflected rotor inertia: Kp = I wn^2, Kd = 2 I zeta wn | Sets a modest value that scales with armature (rotor-only in mjlab) | A1, A2, A3; Isaac Lab form with total equivalent inertia A4; human analog Ke = I wn^2 in A5 |
| Action scale = 0.25 * effort_limit / Kp (tau_max/Kp rule) | Couples Kp to torque limit; does not choose Kp | A1, A2 |
| Torque range / motion range (Kp ~ tau_max / range) | Moderate; set by actuator limits | C2 (50 = 30/0.6), B5 |
| Gravity / load / tracking error budget (Kp >= tau / e_allowed) | Up (floor grows with m and L) | B4, C9; biological m g h bound B1-B3; generic statics derivation only in F |
| Static toppling stiffness m g h (human ankle) | Up, as a floor for pure-stiffness balance; passive human value measured 65-91% of it, hence active control needed | B1, B2, B3 |
| Hardware sensor noise, impact absorption, compliance | Down | A1, C7, C1, C5 |
| Sim-to-real gap grows for stiff and overdamped gains | Down (and not too damped) | C1 (Franka hardware only), C5 |
| Exploration / learnability of RL | Down for exploration; RL can learn at any gain if hyperparameters are tuned | B5, B4, C1 (success at all gains), C9 (high gain failed to learn walking) |
| Fixed deployment gains are robot-specific and typically hand-tuned; policy-learned stiffness as alternative | Variable | C9, C10, C8 |
| Tracking accuracy and fast disturbance rejection by the inner loop (policy at 50 Hz is slow vs PD kHz) | Up (the PD must supply the fast loop) | B5 (quasi-static action), C11 (33 Hz vs 2 kHz), C6 (PD more robust at low query rates); high-impedance precision tracking is what BeyondMimic cites as the opposite camp (A1) |
| Discrete-time stability at PD/sim rate | Up-bounded by dt and inertia; implicit/damped integrators relax it | D1, D2, C3 (dt 0.005 limit), C5 |
| Latency limits usable Kd | Down on Kd | C8 |
| Damping ratio: zeta 2 overdamped because link inertia is ignored; critical (zeta 1) per Raibert; human standing is underdamped | Kd up | A1, A4, A5; stiff and overdamped hurts transfer C1 |
| "Minimum gain at which the robot stands" | No source found using it as a criterion | F |
