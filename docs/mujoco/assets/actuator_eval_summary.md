
### [clean]  (N=90750, 1815s, dt=0.020s, sim→real ×1.15)

| joint | motor | RMS/rated | P99/peak | Peak/peak | spd P99/Peak rpm | out-env% | 열 S / T_ss | P_mech pk | verdict |
|---|---|--:|--:|--:|--:|--:|:--:|--:|:--:|
| hip_pitch | RS04 | 18.4/40 (46%) OK | 65.8 (55%) | 82.3 (69%) OK | 40/60 | 0.00 | S=0.20 / 50C OK | 354W | OK |
| hip_roll | RS04 | 29.1/40 (73%) OK | 71.9 (60%) | 100.5 (84%) OK | 19/44 | 0.00 | S=0.42 / 88C OK | 224W | OK |
| hip_yaw | RS03 | 10.9/20 (55%) OK | 32.7 (54%) | 63.8 (106%) ⚠ | 30/77 | 0.00 | S=0.15 / 61C OK | 158W | **FAIL** |
| knee | RS04 | 15.9/40 (40%) OK | 54.4 (45%) | 108.6 (90%) OK | 65/110 | 0.00 | S=0.09 / 44C OK | 452W | OK |
| crank_A | RS03 | 8.7/20 (44%) OK | 31.0 (52%) | 38.8 (65%) OK | 59/89 | 0.00 | S=0.04 / 48C OK | 222W | OK |
| crank_B | RS03 | 12.2/20 (61%) OK | 33.0 (55%) | 36.7 (61%) OK | 55/85 | 0.00 | S=0.09 / 70C OK | 149W | OK |

**clean 종합**: 미달 1 — hip_yaw ⚠  · 시스템 피크: P_mech 1560W / P_elec(추정) 1086W (η=0.8)

**[clean] ankle-axis equivalent** — tauank_eq = Jᶜᵀ·τ_crank, the joint-axis demand behind the crank motors (interpretation only, ×1.15; the sizing motor is the RS03 crank above):
| ankle axis | RMS | P99 | Peak [N·m] |
|---|--:|--:|--:|
| pitch | 7.6 | 19.2 | 25.8 |
| roll | 14.2 | 46.7 | 53.5 |

### [push]  (N=90750, 1815s, dt=0.020s, sim→real ×1.15)

| joint | motor | RMS/rated | P99/peak | Peak/peak | spd P99/Peak rpm | out-env% | 열 S / T_ss | P_mech pk | verdict |
|---|---|--:|--:|--:|--:|--:|:--:|--:|:--:|
| hip_pitch | RS04 | 19.2/40 (48%) OK | 70.0 (58%) | 99.4 (83%) OK | 43/64 | 0.00 | S=0.32 / 53C OK | 436W | OK |
| hip_roll | RS04 | 30.4/40 (76%) OK | 75.4 (63%) | 138.0 (115%) ⚠ | 20/70 | 0.01 | S=0.56 / 94C OK | 635W | **FAIL** |
| hip_yaw | RS03 | 11.1/20 (55%) OK | 34.7 (58%) | 68.6 (114%) ⚠ | 30/93 | 0.01 | S=0.23 / 62C OK | 445W | **FAIL** |
| knee | RS04 | 15.8/40 (40%) OK | 54.3 (45%) | 121.7 (101%) ⚠ | 67/122 | 0.00 | S=0.09 / 44C OK | 1020W | **FAIL** |
| crank_A | RS03 | 9.4/20 (47%) OK | 31.2 (52%) | 42.7 (71%) OK | 58/139 | 0.00 | S=0.05 / 52C OK | 400W | OK |
| crank_B | RS03 | 13.1/20 (66%) OK | 34.1 (57%) | 43.3 (72%) OK | 57/115 | 0.00 | S=0.12 / 77C OK | 223W | OK |

**push 종합**: 미달 3 — hip_roll, hip_yaw, knee ⚠  · 시스템 피크: P_mech 3159W / P_elec(추정) 2815W (η=0.8)

**[push] ankle-axis equivalent** — tauank_eq = Jᶜᵀ·τ_crank, the joint-axis demand behind the crank motors (interpretation only, ×1.15; the sizing motor is the RS03 crank above):
| ankle axis | RMS | P99 | Peak [N·m] |
|---|--:|--:|--:|
| pitch | 7.6 | 19.6 | 38.8 |
| roll | 15.6 | 47.7 | 57.5 |