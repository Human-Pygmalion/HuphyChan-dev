# Academic criteria for joint PD gains (Kp/Kd) of a legged robot: raw research note

Date: 2026-10-14. Author: researcher agent (Sonnet). Requested by the planner.
Scope: the academic side only (human posture and gait stiffness, impedance and sampled-data theory, actuator design, RL gain studies, control-theory sizing). The mjlab/BeyondMimic natural-frequency rule, robot gain tables, legged_gym/ANYmal, Peng and van de Panne 2017 and Hwangbo 2019 are covered by another agent and are not repeated.

## 0. Findings first

1. No academic source found defines "appropriate gain" as "the minimum gain at which the system barely stands". The nearest statements are physiological, not design rules:
   - Fitzpatrick, Taylor, McCloskey 1992 call the gravity torque-angle slope "the minimal ankle stiffness required to stand" (a physics fact about an inverted pendulum, K_crit = m g h, h = height of the centre of mass above the ankle).
   - Loram and Lakie 2002 use "minimal stabilisation" in the same sense.
   - In the RL papers, "the minimum Kp a policy can cope with" is used only as a robustness test, never as the selection rule (Xie et al. 2021, entry D2).
2. Humans do not stand with stiffness at or above m g h from passive tissue. Direct measurements of intrinsic (reflex-free) ankle stiffness give 0.91 (Loram 2002), 0.64 (Casadio 2005) and 0.93 down to 0.44 depending on perturbation size (Vlutters 2015), all as a fraction of m g h. So passive stiffness is below the toppling stiffness and active, time-varying control is required.
3. The total effective stiffness of the human postural controller (intrinsic plus neural feedback, fitted to sway transfer functions) is above m g h: Peterka 2002 reports "always about 1/3 larger than necessary", i.e. about 1.33 x m g h, with damping scaled up together with stiffness.
4. In the RL literature the gain criterion is two-sided and empirical. Too low: large tracking error, the policy behaves like a torque controller (Xie 2021, Chen 2023). Too high: instability on hardware, stiff contact-sensitive gaits, larger reality gap (Xie 2021: kp=160 fails sim-to-real; Tan 2018: large gains stable in sim but oscillate in reality). Bogdanovic 2020 found an interior optimum (Kp=5) for a hopper.
5. The hard upper bounds come from sampled-data and delay theory (Colgate), not from standing: b > K T/2 + B for a spring of stiffness K sampled at period T (device damping b, virtual damping B), and b > (2n+1) K T/2 + B when the loop has n samples of extra delay.
6. The classical "inner loop 4 to 10 times faster than the outer loop" cascade rule (Skogestad, Shinskey) is not satisfied by RL locomotion stacks (50 Hz policy, joint PD closed-loop bandwidth typically well under 25 Hz). No source found that sets a PD-to-policy bandwidth ratio for RL. Flagged as an inference in section 5.

## 1. How the sources were checked

- Abstracts of journal papers: NCBI E-utilities efetch (PubMed) and OpenAlex abstract index; DOIs and pages cross-checked against Crossref.
- RL papers: full text of arXiv PDFs read with pdftotext (arXiv IDs given). Excerpts below are copied from that text; line breaks joined.
- Roberts, Moore, Colgate 2024: author's accepted manuscript at par.nsf.gov (purl 10533850).
- Wensing 2017: author's proof at fab.cba.mit.edu (mit-cheetah-actuator.pdf).
- Wittenmark, Astrom, Arzen: IFAC Professional Brief PDF hosted at kth.se.
- Every excerpt is under 40 words. Items I could not verify against a primary text are marked UNVERIFIED.
- Numbers marked "my arithmetic" are not from a source.

## 2. Topic A: human stiffness as the reference for "barely standing"

### A1. Fitzpatrick RC, Taylor JL, McCloskey DI (1992)
Ankle stiffness of standing humans in response to imperceptible perturbation: reflex and task-dependent components. J Physiol 454:533-547. DOI 10.1113/jphysiol.1992.sp019278 (PMID 1474502).
- "This relationship determines the minimal ankle stiffness required to stand, and reflex muscle stiffness is a necessary component of this ankle stiffness."
- "the local sensory reflexes alone are sufficient to stand."
- Claim: the gravity (load) stiffness m g h is the minimal ankle stiffness to stand, and reflex stiffness is needed on top of passive tissue. No numeric fraction in the abstract. This is the closest thing to "barely standing" in the literature and it is a physical lower bound, not a tuning rule.

### A2. Loram ID, Lakie M (2002)
Direct measurement of human ankle stiffness during quiet standing: the intrinsic mechanical stiffness is insufficient for stability. J Physiol 545(3):1041-1053. DOI 10.1113/jphysiol.2002.025049 (PMID 12482906, PMC2290720).
- "intrinsic ankle stiffness is not quite sufficient to stabilise the body or pendulum. On average the directly determined intrinsic stiffness is 91 +/- 23 % (mean +/- S.D.) of that necessary to provide minimal stabilisation."
- "The implication is that the brain cannot set ankle stiffness and then ignore the control task because additional modulation of torque is required to maintain balance."
- Claim: intrinsic ankle stiffness = 0.91 +/- 0.23 of the critical value (m g h). Stiffness alone cannot hold the pendulum.

### A3. Casadio M, Morasso PG, Sanguineti V (2005)
Direct measurement of ankle stiffness during quiet standing: implications for control modelling and clinical application. Gait Posture 21(4):410-424. DOI 10.1016/j.gaitpost.2004.05.005 (PMID 15886131).
- "The intrinsic ankle stiffness was evaluated to be 64+/-8% of the critical stiffness for test disturbances of the order of 1 degrees."
- "the upper bound is given by the estimates performed with much smaller test disturbances [1] which yield a higher value: 91+/-23%."
- Claim: 64 +/- 8 % of critical for 1 degree steps (lower bound), 91 +/- 23 % for tiny perturbations (upper bound). Stiffness is amplitude dependent (softer for larger motions).

### A4. Vlutters M, Boonstra TA, Schouten AC, van der Kooij H (2015)
Direct measurement of the intrinsic ankle stiffness during standing. J Biomech 48(7):1258-1263. DOI 10.1016/j.jbiomech.2015.03.004 (PMID 25843262).
- "The intrinsic ankle-foot stiffness appears insufficient to ensure balance, ranging from 0.93+-0.09 to 0.44+-0.06 (normalized to critical stiffness 'mgh')."
- "This implies that changes in muscle activation are required to maintain balance."
- Claim: 0.93 down to 0.44 of m g h as rotation amplitude grows from 0.005 to 0.08 rad (40 ms ramp-and-hold, reflexes excluded). Independent confirmation of A2 and A3.

### A5. Winter DA, Patla AE, Prince F, Ishac M, Gielo-Perczak K (1998)
Stiffness control of balance in quiet standing. J Neurophysiol 80(3):1211-1221. DOI 10.1152/jn.1998.80.3.1211 (PMID 9744933).
- "The model assumes that muscles act as springs to cause the center-of-pressure (COP) to move in phase with the center-of-mass (COM) as the body sways about some desired position."
- "the inverted pendulum model is severely underdamped, approaching the undamped condition."
- Claim: proposes that pure muscle stiffness (a 0th order spring) stabilises stance; estimates Ke = I wn^2 from sway spectra. This is the strongest "stiffness alone stands" claim in the literature, and it was challenged (A6, A8, A2). Numeric Ke values are in the full text (not extracted here, UNVERIFIED).

### A6. Morasso PG, Schieppati M (1999)
Can muscle stiffness alone stabilize upright standing? J Neurophysiol 82(3):1622-1626. DOI 10.1152/jn.1999.82.3.1622 (PMID 10482776).
- "we show that physiological values of stiffness of the ankle muscles are insufficient to stabilize the body \"inverted pendulum.\""
- "the in-phase relation between the trajectories of the center of pressure and the center of mass is determined by physics, not by control patterns."
- Claim: critique of A5. No numeric threshold in the abstract.

### A7. Peterka RJ (2002)
Sensorimotor integration in human postural control. J Neurophysiol 88(3):1097-1118. DOI 10.1152/jn.2002.88.3.1097 (PMID 12205132).
- "In normal subjects, stiffness and damping were highly correlated with body mass and moment of inertia, with stiffness always about 1/3 larger than necessary to resist the destabilizing torque due to gravity."
- "To maintain postural stability and avoid resonant behavior, an increase in stiffness should be accompanied by a corresponding increase in damping."
- "Increased damping was achieved primarily by decreasing the apparent time delay of feedback control rather than by changing the damping coefficient"
- Claim: total effective stiffness is about 1.33 x m g h (my reading of "1/3 larger"), stiffness rose up to 60 % with larger support-surface perturbation, and damping had to rise with it, mostly through a shorter effective feedback delay.

### A8. Bottaro A, Casadio M, Morasso PG, Sanguineti V (2005)  [supporting]
Body sway during quiet standing: is it the residual chattering of an intermittent stabilization process? Hum Mov Sci 24(4):588-615. DOI 10.1016/j.humov.2005.07.006 (PMID 16143414).
- "can guarantee asymptotic stability of controlled posture at the expense of unrealistic assumptions: the level of intrinsic muscle stiffness in the former case"
- Claim: review. Stiffness control needs an intrinsic stiffness level that is unrealistic; PID needs unrealistic noise; proposes intermittent stabilisation. Supports "active control needed". No number.

### A9. Rouse EJ, Hargrove LJ, Perreault EJ, Kuiken TA (2014)
Estimation of human ankle impedance during the stance phase of walking. IEEE Trans Neural Syst Rehabil Eng 22(4):870-878. DOI 10.1109/TNSRE.2014.2307256 (PMID 24760937).
- "the stiffness component of impedance was found to increase linearly from 1.5 to 6.5 Nm/rad/kg between 20% and 70% of stance phase."
- "The damping component was found to be statistically greater than zero only for the estimate at 70% of stance phase, with a value of 0.03 Nms/rad/kg."
- Claim: walking-stance ankle stiffness 1.5 to 6.5 N m/rad/kg, damping essentially zero except 0.03 N m s/rad/kg late in stance. Per kg, the standing critical value is g h (my arithmetic: 9.81 x 1.0 m about 9.8 N m/rad/kg for h = 1 m), so the walking stiffness is about 15 to 66 % of the standing critical value; walking is not stabilised by ankle stiffness alone. The model damping is negligible, so humans run joints low-damped in stance.

### A10. Shamaei K, Sawicki GS, Dollar AM (2013), ankle, knee, hip  [one entry]
- Ankle: Estimation of quasi-stiffness and propulsive work of the human ankle in the stance phase of walking. PLoS ONE 8(3):e59935. DOI 10.1371/journal.pone.0059935 (PMID 23555839).
- Knee: Estimation of quasi-stiffness of the human knee in the stance phase of walking. PLoS ONE 8(3):e59993. DOI 10.1371/journal.pone.0059993 (PMID 23533662).
- Hip: Estimation of quasi-stiffness of the human hip in the stance phase of walking. PLoS ONE 8(12):e81841. DOI 10.1371/journal.pone.0081841 (PMID 24349136).
- Ankle: "the ankle joint propels the body through three distinctive phases of nearly constant stiffness known as the quasi-stiffness of each phase."
- Ankle full text (PMC3605342): "The average values range from 201 Nm/rad to 685 Nm/rad for" (symbol for the phase lost in text extraction; which phase is UNVERIFIED).
- Knee full text (PMC3606171): "The stiffness that designers use range from ~50 Nm/rad to ~430 Nm/rad"
- Hip full text (PMC3857237): "K e ranged from a minimum value of 91 Nm/rad ... to a maximum value of 1775 Nm/rad" (extension stage; flexion stage 9 to 1400 N m/rad).
- Claim: gait joint quasi-stiffness is the slope of the moment-angle curve (a descriptive, work-loop quantity), scales with body weight and height, gait speed and excursion. It is a measured output, not a controller gain; use as an order-of-magnitude reference per body mass only. Note quasi-stiffness is not impedance: Rouse 2014 reports that for the ankle the two agree, but that is not shown for knee or hip.

## 3. Topic B: impedance control theory, passivity, sample rate and delay

### B1. Hogan N (1985)
Impedance control: an approach to manipulation, Part I Theory. J Dyn Syst Meas Control 107(1):1-7. DOI 10.1115/1.3140702. (Part II Implementation: DOI 10.1115/1.3140713.)
- "Control of position or force alone is inadequate; control of dynamic behavior is also required."
- "given the environment contains inertial objects, the manipulator must be an impedance."
- Claim: a stiffness/damping setting is a choice of interaction dynamics, not a position-accuracy knob. Joint Kp, Kd are the joint-space impedance. No numeric criterion.

### B2. Colgate JE, Brown JM (1994)
Factors affecting the Z-width of a haptic display. Proc IEEE ICRA 1994, pp. 3205-3210. DOI 10.1109/ROBOT.1994.351077.
- "an important measure of performance is the dynamic range of achievable impedances-\"Z-Width\"-and that an impedance is achievable if it satisfies a robustness property such as passivity."
- "A striking result is that inherent interface damping exerts an overwhelming influence on Z-Width."
- Claim: the largest stiffness a sampled controller can render is limited by physical damping of the interface (and by sample-and-hold, quantisation, velocity filtering). Source of the b > K T/2 family of bounds.

### B3. Colgate JE, Schenkel GG (1997)
Passivity of a class of sampled-data systems: application to haptic interfaces. J Robot Syst 14(1):37-47. DOI 10.1002/(SICI)1097-4563(199701)14:1<37::AID-ROB4>3.0.CO;2-V. (Conference version: Proc American Control Conf 1994, pp. 3236-3240, DOI 10.1109/ACC.1994.735172.)
- "Necessary conditions for passivity are found via a small gain condition, and sufficient conditions are found via an application of Parseval's theorem"
- Claim: gives the lower bound on physical viscous damping for passivity of a continuous plant with a sampled controller; virtual-wall example. The explicit formula is quoted from B4 below (the 1997 paper itself was not opened, UNVERIFIED).

### B4. Roberts RG, Moore CA Jr, Colgate JE (2024)
A new expression for the passivity bound for a class of sampled-data systems. IEEE Trans Robot (accepted 2024). DOI 10.1109/TRO.2024.3433869.
- "Their passivity condition was given by a lower bound on the viscous damping of the actuator, with necessity proved using Nyquist plane methods and sufficiency proved using an energy-based argument."
- "A conservative bound for (53) is b > (2n+1)kT /2 + B."  (the case H(z) = K + B (z-1)/(T z), n extra samples of delay)
- For H(z) = K + B (z-1)/(T z) with no delay the paper gives "KT/2 + B" as the bound on b (the inequality sign was lost in text extraction; the form b > K T/2 + B is stated in Ghavifekr et al., arXiv 2008.11099, as "calculated as b >= KT/2 + B, where T is the sampling time, K and B are the stiffness and damping of the virtual wall, and b is the damping of the haptic interface").
- Delay of one sample (td = T), from the paper text: b > K T/2 + B + (K T - 2B) if K T >= 4B, else b > K T/2 + B + K^2 T^2/(8B). For K T >= 4B this equals b > 3 K T/2 - B (my arithmetic, consistent with the conservative form with n = 1).
- Claim: K is the stiffness, B the virtual damping, T the loop period, b the physical damping. A one-sample delay triples the K T/2 term. Passivity is sufficient, not necessary, for stability (see B6).

### B5. Abbott JJ, Okamura AM (2005)  [supporting]
Effects of position quantization and sampling rate on virtual-wall passivity. IEEE Trans Robot 21(5):952-964. DOI 10.1109/TRO.2005.851377.
- "we find an explicit upper bound on virtual wall stiffness that is a necessary and sufficient condition for virtual wall passivity."
- Claim: with Coulomb-plus-viscous friction and encoder resolution, a stiffness ceiling exists that depends on friction, quantisation and sampling period. Formula not extracted (UNVERIFIED).

### B6. Diolaiti N, Niemeyer G, Barbagli F, Salisbury JK (2006)  [supporting]
Stability of haptic rendering: discretization, quantization, time delay, and Coulomb effects. IEEE Trans Robot 22(2):256-268. DOI 10.1109/TRO.2005.862487.
- "Rendering stiff virtual objects remains a core challenge in the field of haptics. A study of this problem is presented, which relates the maximum achievable object stiffness to the elements of the control loop."
- "Nonlinear effects create distinct stability regions, and many common devices operate stably, yet in violation of passivity criteria."
- Claim: sampling rate, quantisation, computational delay and amplifier dynamics set the maximum stable stiffness together with inertia and natural viscous plus Coulomb damping. Passivity bounds (B4) are conservative: real devices run stable beyond them.

Note on units for this project (my arithmetic, not from a source): the PD loop in a motor driver at 10 kHz has T = 0.1 ms; with Kp = 40 N m/rad the term K T/2 is 2 x 10^-3 N m s/rad, which is small compared with typical joint friction, so the driver-rate passivity bound is not the binding constraint. It matters if the effective loop period is the 20 ms policy period (not the case for a PD in the driver) or if delay n is large.

## 4. Topic C: actuator design and gain implications

### C1. Wensing PM, Wang A, Seok S, Otten D, Lang J, Kim S (2017)
Proprioceptive actuator design in the MIT Cheetah: impact mitigation and high-bandwidth physical interaction for dynamic legged robots. IEEE Trans Robot 33(3):509-522. DOI 10.1109/TRO.2016.2640183.
- "Proprioceptive actuation uses collocated force control at the joints to effectively control contact interactions at the feet under dynamic conditions."
- "this control approach used with high-gear-ratio actuators has not demonstrated robustness to impacts or high-bandwidth force control capabilities for dynamic legged locomotion."
- "This structural compliance effectively decreases the bandwidth of the proprioceptive force control below that of the electrical dynamics."
- Claim: low gear ratio plus backdrivability (impact mitigation factor, IMF) rather than high-gain position or impedance control with a high-ratio gearbox is what mitigates impacts. Numbers: 20 kHz motor current loop, 100 N step force test, contact times down to 85 ms, peak forces over 450 N. Implication for PD gains: with a high-ratio geared actuator, reflected rotor inertia and friction, not the pendulum, dominate what Kp can do.

### C2. Kenneally G, De A, Koditschek DE (2016)  [supporting]
Design principles for a family of direct-drive legged robots. IEEE Robot Autom Lett 1(2):900-907. DOI 10.1109/LRA.2016.2528294.
- "We present a methodology that achieves the well-known benefits of DD robot design (transparency, mechanical robustness/efficiency, high-actuation bandwidth, and increased specific power)"
- Claim: direct drive gives transparency and bandwidth at low specific force. Qualitative; no gain value in the abstract. (Seok et al. 2015, DOI 10.1109/TMECH.2014.2339013, was checked: abstract is about energy-loss design principles and a single-stage low-ratio transmission, no gain statement; not used.)

### C3. Pratt GA, Williamson MM (1995)
Series elastic actuators. Proc IEEE/RSJ IROS 1995, pp. 399-406. DOI 10.1109/IROS.1995.525827.
- "It is traditional to make the interface between an actuator and its load as stiff as possible."
- "reducing interface stiffness offers a number of advantages, including greater shock tolerance, lower reflected inertia, more accurate and stable force control"
- "As a trade-off, reducing interface stiffness also lowers zero motion force bandwidth."
- Claim: compliance trades bandwidth for shock tolerance and force-control stability. The PD-in-the-driver analogue is a soft Kp. Abstract only (full PDF not retrieved).

### C4. Haddadin S, Albu-Schaffer A, Eiberger O, Hirzinger G (2010)  [supporting]
New insights concerning intrinsic joint elasticity for safety. Proc IEEE/RSJ IROS 2010, pp. 2181-2187. DOI 10.1109/IROS.2010.5652037. (Related: Haddadin, Albu-Schaffer, Hirzinger, Requirements for safe robots, Int J Robot Res 28(11-12):1507-1527, 2009, DOI 10.1177/0278364909343970.)
- "intrinsic joint elasticity does not reduce the Head Injury Criterion or impact forces compared to conventional actuation with some considerable elastic behavior in the joint, if considering full scale robots."
- Claim: a counterweight to C3: for full-scale robots, joint softness does not by itself cut impact forces. So "lower stiffness means safer impacts" is not general. Human-robot safety setting, not legged locomotion (caveat). Abstract read via TUM portal.

## 5. Topic D: RL studies that sweep, learn or discuss PD gains

### D1. Bogdanovic M, Khadiv M, Righetti L (2020)
Learning variable impedance control for contact sensitive tasks. IEEE Robot Autom Lett 5(4):6129-6136. DOI 10.1109/LRA.2020.3011379. arXiv 1907.07500.
- "For fixed gain control we can observe a drop in performance when gains are too low and when they are too high, with a medium value of Kp = 5.0 having the best average performance."
- "the maximum admissible gains in contact will always be larger than during the flying phase due to the changing reflected inertia at the joints in loaded/unloaded conditions."
- Claim: single-leg hopper (PyBullet, DDPG): performance is an interior optimum in fixed Kp (best Kp = 5.0; units not stated, UNVERIFIED); direct torque policies failed; the variable-gain policy reached Kp = 10 without the instability of fixed Kp = 10 because it raises gain only in stance. The max stable gain depends on loaded vs unloaded reflected inertia. A regulariser keeps desired position close to the real trajectory so gains stay interpretable for real-robot transfer.

### D2. Xie Z, Da X, van de Panne M, Babich B, Garg A (2021)
Dynamics randomization revisited: a case study for quadrupedal locomotion. Proc IEEE ICRA 2021, pp. 4955-4961. DOI 10.1109/ICRA48506.2021.9560837. arXiv 2011.02404.
- "In [2], the authors use stiff proportional gains (kp = 220) for the PD controllers on the motors, while we use a soft gain (kp = 40)."
- "low proportional gains will result in a policy behaving like a torque controller, i.e., using the target angle as a proxy to achieve a desired force, while high proportional gains will behave like a position controller."
- "Policies without velocity feedback or with kp = 160 all fail the sim-to-real tests."
- "we use kp = 160 instead of kp = 220 as used in [2] because the higher gain can cause instability on the robot."
- Claim: Laikago, kd = 0.5, kp = 40 default works with no dynamics randomization on hardware; kp = 160 policy shows stiff motion, contact-timing sensitivity, fails sim-to-real. They also test robustness by lowering kp at test time and recording "the minimum value the policy can cope with" (the table values were not extracted, UNVERIFIED): a robustness probe, not a selection rule. Units (N m/rad) assumed from the figure axis "P gain (Nm/rad)".

### D3. Chen S, Zhang B, Mueller MW, Rai A, Sreenath K (2023)
Learning torque control for quadrupedal locomotion. Proc IEEE-RAS Humanoids 2023. DOI 10.1109/HUMANOIDS57100.2023.10375154. arXiv 2203.05194.
- "This policy uses joint positions as the action space and runs at 50 Hz, followed by a low-level PD controller tracking at 200 Hz"
- "high proportional gains can cause instability on the robot while low proportional gains result in a policy behaving like a torque controller that leads to a more robust policy."
- "a small derivative gain leads to learning instabilities and a large derivative gain prevents the robot from tracking a target velocity."
- Claim: baseline position policy uses Kp = 20, Kd = 0.5; learning curves differ for Kp = 40, 60, 80 (Kd = 0.5), so Kp changes learning; torque policy reached a higher reward and recovered from kicks the position policy did not. The two quoted gain statements are the authors' review of Xie 2021 and Smith 2022, not new experiments. A 50 Hz policy over a 200 Hz PD (4:1), not 10:1.

### D4. Smith L, Kostrikov I, Levine S (2022)
A walk in the park: learning to walk in 20 minutes with model-free reinforcement learning. arXiv 2208.07860 (demo version: RSS 2023, DOI 10.15607/RSS.2023.XIX.056).
- "Small damping values (Kd = 1) lead to instabilities, which are not desirable for a policy executed on a real robot, while large values prevent the agent from reaching the target velocity (Kd = 20)."
- Claim: A1 robot in MuJoCo; sweep of Kd gave a usable band; Kd = 10 chosen for the remaining ablations. The Kp value is not stated in the text (UNVERIFIED) and the units of Kd are not stated.

### D5. Guan N, Yu S, Zhu S, Kim D (2024)
Impedance matching: enabling an RL-based running jump in a quadruped robot. Proc 21st Int Conf Ubiquitous Robots (UR) 2024, pp. 755-761. DOI 10.1109/UR61395.2024.10597522. arXiv 2404.15096.
- "Instead of naively copying the feedback gains used in the real robot, we conduct a frequency sweep in both real and simulated robots and select appropriate actuator parameters for simulation by matching their bode plots."
- "frequency sim-to-real transfer with robots with geared actuators is very difficult without addressing rotor inertia"
- Claim: a quantitative gain-selection recipe: chirp the real and the simulated joint, grid-search sim gains (Kp 13 to 27 N m/rad, Kd 0.1 to 0.7 N m s/rad, 50 x 50 grid) to match magnitude Bode (bandwidth and overshoot) near the natural frequency, then randomise around it. Real hardware gains were 17 / 0.4 (Kp / Kd); the matched simulation gains were Kp 20, 17.5, 21.5 and Kd 0.45, 0.4, 0.55 for hip abduction, hip flexion, knee; randomisation +/- 2.0 on Kp and +/- 0.05 on Kd. Policy rate 50 Hz sets the sweep Nyquist. The authors say "perhaps choosing a set of critically damped gains would be more stable".

### D6. Spoljaric D, Yan Y, Lee D (2025)  [supporting]
Variable stiffness for robust locomotion through reinforcement learning. arXiv 2502.09436 (venue given in arXiv comment: IFAC joint symposia on mechatronics and robotics; UNVERIFIED which year's proceedings).
- "We show that variable stiffness policies, with grouping in per-leg stiffness (PLS), outperform position-based control in velocity tracking and push recovery."
- "Our approach simplifies design by eliminating per-joint stiffness tuning while keeping competitive results with various metrics."
- Claim: policy outputs Kp together with joint targets on a quadruped; per-leg grouping best. Intro restates Xie and Smith: large Kp makes training unstable, low Kp is torque-like. The text also remarks that higher stiffness during contact "seems reasonable" (context not checked).

### D7. Kim D, Berseth G, Schwartz M, Park J (2023)  [supporting]
Torque-based deep reinforcement learning for task-and-robot agnostic learning on bipedal robots using sim-to-real transfer. IEEE Robot Autom Lett 8(10):6251-6258. DOI 10.1109/LRA.2023.3304561. arXiv 2304.09434.
- "for position control, gain tuning is required to achieve the best possible policy performance."
- Claim: argues that a torque action space needs less parameter tuning and exploits "torque control's inherent compliance" on a human-sized biped. Abstract only.

### D8. Siekmann J, Godse Y, Fern A, Hurst J (2021)  [supporting]
Sim-to-real learning of all common bipedal gaits via periodic reward composition. Proc IEEE ICRA 2021, pp. 7309-7315. DOI 10.1109/ICRA48506.2021.9561814. arXiv 2011.01387.
- "The policy is evaluated at 40Hz, and the robot's PD controllers are run at 2000Hz."
- "the P gains and D gains are also summed with a fixed 'neutral' set of gains."
- Claim: Cassie policy outputs 10 joint targets and 10 PD gain deltas around neutral gains; 40 Hz policy over a 2 kHz PD (50:1). Neutral gain values not found in the text (UNVERIFIED). No ablation of the gain output.

### D9. Tan J, Zhang T, Coumans E, Iscen A, Bai Y, Hafner D, Bohez S, Vanhoucke V (2018)  [supporting]
Sim-to-real: learning agile locomotion for quadruped robots. Proc Robotics: Science and Systems XIV. DOI 10.15607/RSS.2018.XIV.010. arXiv 1804.10332.
- "if large gains are used, the motors can remain stable in simulation but oscillate in reality."
- Claim: Bullet's position constraint (an implicit PD that uses end-of-step state) stays stable at large gains while a real PD servo (current state, delay, saturation) oscillates; fixed by an explicit DC-motor actuator model with torque saturation plus latency. Inference for this project (not stated by the paper): any simulator that integrates PD implicitly can accept Kp values that the real explicit, delayed driver cannot.

### D10. Li Z, Peng XB, Abbeel P, Levine S, Berseth G, Sreenath K (2024)  [supporting]
Reinforcement learning for versatile, dynamic, and robust bipedal locomotion control. Int J Robot Res 44(5):840-888. DOI 10.1177/02783649241285161. arXiv 2401.16889.
- Table row: "Motor PD Gains [0.7, 1.3] x default"
- "adding a range of +-30% deviation from their default values. This randomization is applied independently to each joint-level PD controller."
- Claim: Cassie training randomises each joint's PD gains by +/- 30 % to cover motor uncertainty; a +40 % gain change is used as a one-at-a-time probe in the adaptation analysis. Absolute gains not extracted.

## 6. Topic E: control-theory sizing rules

### E1. Wittenmark B, Astrom KJ, Arzen K-E, Computer control: an overview (IFAC Professional Brief, about 2002; PDF undated, year from search engine, UNVERIFIED)
URL: https://www.kth.se/social/upload/4f311a97f2765461cf00000d/IFAC_control.pdf
- "for small h, the hold can be approximated by a time delay of half a sampling interval."
- "hw_c ~ 0.05 to 0.14 ... where w_c is the crossover frequency (in radians per second) of the continuous-time system. This rule gives a Nyquist frequency that is about 23 to 70 times higher than the crossover frequency."  (my transcription of the symbols: the text is "hω_c ≈ 0.05 to 0.14")
- Claim: sample-and-hold costs half a sample of phase delay; to lose only 5 to 15 degrees of phase margin the loop period h must satisfy h x crossover <= 0.05 to 0.14. Applied to the driver PD loop (my arithmetic): a 10 kHz driver (h = 0.1 ms) supports a crossover up to 500 to 1400 rad/s, about 80 to 220 Hz, above any joint PD crossover in the papers above; the constraint binds at the 50 Hz policy level, not in the driver.

### E2. Skogestad S, lecture notes "Advanced regulatory control elements and switching" (NTNU, process control course 2025)  [supporting]
URL: https://skoge.folk.ntnu.no/vgprosessregulering/lectures-slides-2025/APC6%20-%20Advanced%20regulatory%20control%20elements%20and%20switching.pdf
- "Inner loop (slave) should be at least 4 times* faster than the outer loop (master)"  (* Shinskey, Controlling Multivariable Processes, ISA, 1981, p. 12; book not opened, UNVERIFIED)
- "But recommend 10 times faster"
- Claim: time-scale separation of 4 (minimum) to 10 (recommended) between slave and master loops. These are process-control lecture notes, not peer-reviewed. For legged RL the policy is not a slow outer PID; joint PD bandwidth is not 4 to 10 times above 25 Hz. No source found that sets a PD-to-policy ratio (inference).

### UNVERIFIED items (searched, not confirmed from a primary text)
- Franklin, Powell, Workman, Digital Control of Dynamic Systems: sample at least 20 times the closed-loop bandwidth. Seen only second-hand.
- Astrom and Wittenmark, Computer-Controlled Systems: h x w_c between 0.15 and 0.5. Not found; the IFAC brief (E1) gives 0.05 to 0.14 for a 5 to 15 degree phase-margin loss.
- Spong, Hutchinson, Vidyasagar, Robot Modeling and Control: Kp = I_eff wn^2, Kd = 2 I_eff wn (critical damping), I_eff including N^2 x motor inertia. Standard formula, not verified against the book. A 2026 paper (arXiv 2602.00401, ZEST) uses this recipe with damping ratio 1; not read in full.
- "Servo bandwidth should be 1/3 to 1/2 of the first structural resonance": no primary source found.
- Winter 2001 (J Neurophysiol 85:2630-2633) numeric Ke values: not retrieved.

## 7. Claims table

Units: Kp in N m/rad, Kd in N m s/rad where stated. "Up" means the criterion argues for a higher Kp, "Down" a lower Kp, "Ceiling" an upper limit, "Floor" a lower limit.

| Criterion | Pushes Kp | Quantitative threshold if any | Sources |
|---|---|---|---|
| Static gravity: a pure-stiffness inverted pendulum stands only if K > m g h | Floor (up) | K_crit = m g h (normalised 1.0); a physical fact, not a design target | Fitzpatrick 1992; Loram 2002; Casadio 2005; Vlutters 2015; Morasso 1999 |
| Human intrinsic (passive) stiffness is below critical, so active control is needed | Floor is not met by stiffness alone | 0.91 +/- 0.23 (Loram); 0.64 +/- 0.08 (Casadio, 1 deg); 0.93 to 0.44 vs amplitude (Vlutters) | Loram 2002; Casadio 2005; Vlutters 2015; Bottaro 2005 |
| Total human controller stiffness (passive plus neural) with matched damping | Up, with Kd up | about 1.33 x m g h; stiffness up to +60 % with larger perturbation | Peterka 2002 |
| Pure stiffness control sufficient for stance | Up (claim), disputed | none; "severely underdamped"; refuted by the 0.44 to 0.93 data | Winter 1998; refuted by Morasso 1999, Loram 2002 |
| Walking-stance joint stiffness (descriptive reference) | Reference only | ankle 1.5 to 6.5 N m/rad/kg (about 15 to 66 % of g h per kg, my arithmetic); ankle quasi-stiffness 201 to 685 N m/rad; knee designers ~50 to ~430; hip 9 to 1775 N m/rad across subjects | Rouse 2014; Shamaei 2013 |
| Low Kp gives large tracking error, torque-like behaviour, robust to impacts and sim mismatch | Down | kp = 40 works on Laikago; kp 160 to 220 stiff, fails transfer or unstable | Xie 2021; Chen 2023 |
| High Kp gives better tracking but instability on hardware and larger reality gap | Down (Ceiling) | kp = 160 fails sim-to-real, kp = 220 unstable on robot (Laikago) | Xie 2021; Tan 2018 |
| Interior optimum for fixed Kp | Both | Kp = 5.0 best on a hopper (units not stated); worse both sides | Bogdanovic 2020 |
| Kd too low destabilises learning, Kd too high blocks velocity tracking | Kd band | Kd = 1 unstable, Kd = 20 too slow, Kd = 10 chosen (A1); Kd = 0.5 with Kp 20 to 80 (quadruped) | Smith 2022; Chen 2023; Xie 2021 |
| Damping must scale with stiffness; shorter effective delay raises achievable damping | Kp up requires Kd up or delay down | no number | Peterka 2002; Bogdanovic 2020 (Kd tied to Kp by critical-damping scaling) |
| Sampled-data passivity: device damping must exceed K T/2 + B | Ceiling | b > K T/2 + B; with n samples of delay b > (2n+1) K T/2 + B | Colgate and Brown 1994; Colgate and Schenkel 1997; Roberts, Moore, Colgate 2024 |
| Physical interface damping controls the maximum renderable stiffness (Z-width) | Ceiling | none | Colgate and Brown 1994 |
| Passivity bounds are conservative; real devices run stable above them | Relaxes the ceiling | none | Diolaiti 2006; Abbott and Okamura 2005 |
| Hold plus anti-alias filter costs phase; loop period vs crossover | Ceiling on crossover | h x w_c <= 0.05 to 0.14 (5 to 15 deg phase-margin loss); hold = half-sample delay | Wittenmark, Astrom, Arzen (IFAC brief) |
| Reflected inertia: max admissible gain is larger in stance (loaded) than in flight | Ceiling varies with phase | none | Bogdanovic 2020; Guan 2024 (rotor inertia essential to match Bode) |
| Simulation gains should reproduce the real closed-loop Bode (bandwidth, overshoot), not copy hardware numbers | Sets sim Kp, Kd | hardware 17 / 0.4; matched sim Kp 17.5 to 21.5, Kd 0.4 to 0.55; sweep Kp 13 to 27, Kd 0.1 to 0.7 | Guan 2024 |
| Simulator implicit PD stays stable at gains a real explicit, delayed driver cannot | Ceiling for real hardware | none | Tan 2018 |
| Gain randomisation as protection against motor uncertainty | Spread around chosen Kp | +/- 30 % of default (Cassie); +/- 2.0 Kp and +/- 0.05 Kd (quadruped, Kp about 20) | Li 2024; Guan 2024 |
| Actuator design: backdrivable low-ratio drives mitigate impacts without high gain | Down | none (IMF metric; 20 kHz current loop) | Wensing 2017; Kenneally 2016; Pratt and Williamson 1995 |
| Joint softness does not by itself reduce impact forces for full-scale robots | Neutral (against "softer is safer") | none | Haddadin 2010 |
| Learning gains or torque instead of fixing them | Removes the fixed choice | none | Bogdanovic 2020; Siekmann 2021; Spoljaric 2025; Kim 2023; Chen 2023 |
| Cascade separation: inner loop 4 to 10 times faster than outer | Up (PD bandwidth) | 4 (minimum), 10 (recommended); not met by 50 Hz RL stacks; no RL source sets a ratio | Skogestad notes (Shinskey 1981, UNVERIFIED) |

## 8. Caveats

1. The human data are for a rigid-link-like inverted pendulum pivoting at the ankle with compliant tendon and foot; the robot's equivalent K_crit is m g h of its own mass and CoM height. The 0.44 to 0.93 fractions do not transfer to a robot, whose actuator is a stiff servo.
2. Standing is not the binding constraint for RL gains in the papers above. They choose Kp from sim-to-real stability and learning behaviour, which differ per robot and units (N m/rad vs scaled units). Do not copy absolute numbers across robots.
3. Colgate's bounds are passivity (sufficient) conditions derived for haptic virtual walls with a human on the other end; for a legged joint the effective environment is the ground and body, so treat them as scaling arguments, not tuned limits. Diolaiti 2006 shows stable operation above passivity.
4. Several RL claims on gains are the authors' literature review (Chen 2023 on Xie 2021 and Smith 2022; Spoljaric 2025 likewise), not independent experiments. Primary experiments are Xie 2021, Bogdanovic 2020, Smith 2022, Guan 2024.
5. Not covered: structural compliance and backlash as a gain ceiling (no primary source found), ZMP/LIP ankle controllers that need Kp > m g h in position-controlled biped literature (not searched), the Seok 2015 and Haddadin 2009 full texts.
