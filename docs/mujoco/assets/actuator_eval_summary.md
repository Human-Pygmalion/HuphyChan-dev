
### [v1_slow]  (N=1500, 30s, dt=0.020s, sim→real ×1.15)

| joint | motor | RMS/rated | P99/peak | Peak/peak | spd P99/Peak rpm | out-env% | 열 S / T_ss | P_mech pk | verdict |
|---|---|--:|--:|--:|--:|--:|:--:|--:|:--:|
| hip_pitch | RS04 | 12.5/40 (31%) OK | 35.8 (30%) | 41.7 (35%) OK | 26/32 | 0.00 | S=0.00 / 37C OK | 111W | OK |
| hip_roll | RS04 | 25.0/40 (62%) OK | 63.2 (53%) | 82.5 (69%) OK | 29/38 | 0.00 | S=0.02 / 72C OK | 232W | OK |
| hip_yaw | RS03 | 5.3/20 (26%) OK | 22.2 (37%) | 31.0 (52%) OK | 13/19 | 0.00 | S=0.00 / 33C OK | 24W | OK |
| knee | RS04 | 19.6/40 (49%) OK | 34.1 (28%) | 43.7 (36%) OK | 26/31 | 0.00 | S=0.00 / 54C OK | 111W | OK |
| crank_A | RS03 | 11.0/20 (55%) OK | 28.3 (47%) | 29.1 (49%) OK | 49/59 | 0.00 | S=0.00 / 61C OK | 90W | OK |
| crank_B | RS03 | 10.9/20 (54%) OK | 26.2 (44%) | 28.1 (47%) OK | 48/64 | 0.00 | S=0.00 / 60C OK | 100W | OK |

**v1_slow 종합**: 전 관절 안전권 ✅  · 시스템 피크: P_mech 667W / P_elec(추정) 530W (η=0.8)

**[v1_slow] ankle-axis equivalent** — tauank_eq = Jᶜᵀ·τ_crank, the joint-axis demand behind the crank motors (interpretation only, ×1.15; the sizing motor is the RS03 crank above):
| ankle axis | RMS | P99 | Peak [N·m] |
|---|--:|--:|--:|
| pitch | 4.0 | 17.4 | 21.0 |
| roll | 14.8 | 39.0 | 40.3 |

### [v2_slow]  (N=1500, 30s, dt=0.020s, sim→real ×1.15)

| joint | motor | RMS/rated | P99/peak | Peak/peak | spd P99/Peak rpm | out-env% | 열 S / T_ss | P_mech pk | verdict |
|---|---|--:|--:|--:|--:|--:|:--:|--:|:--:|
| hip_pitch | RS04 | 9.3/40 (23%) OK | 19.1 (16%) | 20.8 (17%) OK | 16/18 | 0.00 | S=0.00 / 31C OK | 49W | OK |
| hip_roll | RS04 | 20.6/40 (52%) OK | 40.2 (34%) | 41.3 (34%) OK | 7/9 | 0.00 | S=0.00 / 57C OK | 33W | OK |
| hip_yaw | RS03 | 5.5/20 (28%) OK | 19.6 (33%) | 22.1 (37%) OK | 11/22 | 0.00 | S=0.00 / 34C OK | 14W | OK |
| knee | RS04 | 7.1/40 (18%) OK | 23.9 (20%) | 30.5 (25%) OK | 38/53 | 0.00 | S=0.00 / 29C OK | 65W | OK |
| crank_A | RS03 | 6.8/20 (34%) OK | 26.3 (44%) | 27.2 (45%) OK | 36/61 | 0.00 | S=0.00 / 39C OK | 52W | OK |
| crank_B | RS03 | 10.2/20 (51%) OK | 32.7 (54%) | 33.3 (55%) OK | 35/73 | 0.00 | S=0.01 / 56C OK | 89W | OK |

**v2_slow 종합**: 전 관절 안전권 ✅  · 시스템 피크: P_mech 301W / P_elec(추정) 240W (η=0.8)

**[v2_slow] ankle-axis equivalent** — tauank_eq = Jᶜᵀ·τ_crank, the joint-axis demand behind the crank motors (interpretation only, ×1.15; the sizing motor is the RS03 crank above):
| ankle axis | RMS | P99 | Peak [N·m] |
|---|--:|--:|--:|
| pitch | 4.3 | 11.3 | 14.6 |
| roll | 12.4 | 44.2 | 44.6 |

### [v3_slow]  (N=1500, 30s, dt=0.020s, sim→real ×1.15)

| joint | motor | RMS/rated | P99/peak | Peak/peak | spd P99/Peak rpm | out-env% | 열 S / T_ss | P_mech pk | verdict |
|---|---|--:|--:|--:|--:|--:|:--:|--:|:--:|
| hip_pitch | RS04 | 13.3/40 (33%) OK | 40.3 (34%) | 43.7 (36%) OK | 31/33 | 0.00 | S=0.00 / 38C OK | 135W | OK |
| hip_roll | RS04 | 20.9/40 (52%) OK | 46.8 (39%) | 55.8 (47%) OK | 11/21 | 0.00 | S=0.00 / 58C OK | 30W | OK |
| hip_yaw | RS03 | 10.0/20 (50%) OK | 36.2 (60%) | 40.1 (67%) OK | 24/56 | 0.00 | S=0.01 / 55C OK | 39W | OK |
| knee | RS04 | 11.8/40 (30%) OK | 46.6 (39%) | 57.2 (48%) OK | 69/86 | 0.00 | S=0.00 / 35C OK | 226W | OK |
| crank_A | RS03 | 5.8/20 (29%) OK | 18.4 (31%) | 21.5 (36%) OK | 56/83 | 0.00 | S=0.00 / 35C OK | 119W | OK |
| crank_B | RS03 | 7.0/20 (35%) OK | 24.8 (41%) | 31.8 (53%) OK | 94/135 | 0.00 | S=0.00 / 40C OK | 355W | OK |

**v3_slow 종합**: 전 관절 안전권 ✅  · 시스템 피크: P_mech 904W / P_elec(추정) 847W (η=0.8)

**[v3_slow] ankle-axis equivalent** — tauank_eq = Jᶜᵀ·τ_crank, the joint-axis demand behind the crank motors (interpretation only, ×1.15; the sizing motor is the RS03 crank above):
| ankle axis | RMS | P99 | Peak [N·m] |
|---|--:|--:|--:|
| pitch | 4.7 | 15.1 | 20.4 |
| roll | 8.3 | 29.1 | 34.1 |