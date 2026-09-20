
### [v6b]  (N=90750, 1815s, dt=0.020s, sim→real ×1.15)

| joint | motor | RMS/rated | P99/peak | Peak/peak | spd P99/Peak rpm | out-env% | 열 S / T_ss | P_mech pk | verdict |
|---|---|--:|--:|--:|--:|--:|:--:|--:|:--:|
| hip_pitch | RS04 | 16.5/40 (41%) OK | 65.7 (55%) | 87.7 (73%) OK | 41/67 | 0.00 | S=0.22 / 46C OK | 441W | OK |
| hip_roll | RS04 | 25.8/40 (64%) OK | 69.0 (57%) | 113.7 (95%) OK | 19/36 | 0.00 | S=0.33 / 75C OK | 252W | OK |
| hip_yaw | RS03 | 9.3/20 (46%) OK | 36.1 (60%) | 66.3 (111%) ⚠ | 31/87 | 0.00 | S=0.36 / 51C OK | 174W | **FAIL** |
| knee | RS04 | 12.6/40 (31%) OK | 45.8 (38%) | 81.6 (68%) OK | 66/118 | 0.00 | S=0.03 / 37C OK | 649W | OK |
| crank_A | RS03 | 9.1/20 (45%) OK | 29.7 (49%) | 41.5 (69%) OK | 52/91 | 0.00 | S=0.03 / 50C OK | 246W | OK |
| crank_B | RS03 | 10.3/20 (51%) OK | 32.2 (54%) | 43.3 (72%) OK | 56/93 | 0.00 | S=0.10 / 57C OK | 267W | OK |

**v6b 종합**: 미달 1 — hip_yaw ⚠  · 시스템 피크: P_mech 2028W / P_elec(추정) 1360W (η=0.8)

**[v6b] ankle-axis equivalent** — tauank_eq = Jᶜᵀ·τ_crank, the joint-axis demand behind the crank motors (interpretation only, ×1.15; the sizing motor is the RS03 crank above):
| ankle axis | RMS | P99 | Peak [N·m] |
|---|--:|--:|--:|
| pitch | 7.3 | 19.2 | 26.9 |
| roll | 12.7 | 44.6 | 57.6 |

### [v7]  (N=90750, 1815s, dt=0.020s, sim→real ×1.15)

| joint | motor | RMS/rated | P99/peak | Peak/peak | spd P99/Peak rpm | out-env% | 열 S / T_ss | P_mech pk | verdict |
|---|---|--:|--:|--:|--:|--:|:--:|--:|:--:|
| hip_pitch | RS04 | 16.6/40 (42%) OK | 65.3 (54%) | 84.4 (70%) OK | 49/67 | 0.00 | S=0.26 / 46C OK | 510W | OK |
| hip_roll | RS04 | 26.0/40 (65%) OK | 66.3 (55%) | 112.0 (93%) OK | 33/63 | 0.00 | S=0.31 / 76C OK | 321W | OK |
| hip_yaw | RS03 | 10.7/20 (54%) OK | 41.2 (69%) | 68.6 (114%) ⚠ | 48/85 | 0.01 | S=0.42 / 60C OK | 313W | **FAIL** |
| knee | RS04 | 12.2/40 (31%) OK | 43.3 (36%) | 108.1 (90%) OK | 84/199 | 0.02 | S=0.03 / 36C OK | 982W | OK |
| crank_A | RS03 | 8.6/20 (43%) OK | 28.2 (47%) | 45.3 (75%) OK | 47/98 | 0.00 | S=0.06 / 47C OK | 289W | OK |
| crank_B | RS03 | 9.8/20 (49%) OK | 28.3 (47%) | 46.0 (77%) OK | 63/142 | 0.00 | S=0.02 / 54C OK | 318W | OK |

**v7 종합**: 미달 1 — hip_yaw ⚠  · 시스템 피크: P_mech 2734W / P_elec(추정) 1788W (η=0.8)

**[v7] ankle-axis equivalent** — tauank_eq = Jᶜᵀ·τ_crank, the joint-axis demand behind the crank motors (interpretation only, ×1.15; the sizing motor is the RS03 crank above):
| ankle axis | RMS | P99 | Peak [N·m] |
|---|--:|--:|--:|
| pitch | 7.4 | 20.0 | 29.0 |
| roll | 11.8 | 35.9 | 62.2 |