
### [v10a_clean]  (N=90750, 1815s, dt=0.020s, sim→real ×1.15)

| joint | motor | RMS/rated | P99/peak | Peak/peak | spd P99/Peak rpm | out-env% | 열 S / T_ss | P_mech pk | verdict |
|---|---|--:|--:|--:|--:|--:|:--:|--:|:--:|
| hip_pitch | RS04 | 19.3/40 (48%) OK | 68.0 (57%) | 89.5 (75%) OK | 51/89 | 0.00 | S=0.40 / 53C OK | 501W | OK |
| hip_roll | RS04 | 28.0/40 (70%) OK | 68.0 (57%) | 106.7 (89%) OK | 37/67 | 0.00 | S=0.35 / 84C OK | 321W | OK |
| hip_yaw | RS03 | 12.2/20 (61%) OK | 43.8 (73%) | 65.9 (110%) ⚠ | 54/111 | 0.00 | S=0.54 / 69C OK | 432W | **FAIL** |
| knee | RS04 | 14.1/40 (35%) OK | 47.3 (39%) | 137.1 (114%) ⚠ | 104/183 | 0.00 | S=0.09 / 40C OK | 1174W | **FAIL** |
| crank_A | RS03 | 8.1/20 (40%) OK | 26.2 (44%) | 42.9 (72%) OK | 48/98 | 0.00 | S=0.02 / 45C OK | 302W | OK |
| crank_B | RS03 | 9.3/20 (47%) OK | 26.5 (44%) | 41.6 (69%) OK | 64/130 | 0.00 | S=0.01 / 51C OK | 241W | OK |

**v10a_clean 종합**: 미달 2 — hip_yaw, knee ⚠  · 시스템 피크: P_mech 2971W / P_elec(추정) 2123W (η=0.8)

**[v10a_clean] ankle-axis equivalent** — tauank_eq = Jᶜᵀ·τ_crank, the joint-axis demand behind the crank motors (interpretation only, ×1.15; the sizing motor is the RS03 crank above):
| ankle axis | RMS | P99 | Peak [N·m] |
|---|--:|--:|--:|
| pitch | 7.6 | 20.4 | 32.7 |
| roll | 10.7 | 31.1 | 53.5 |

### [v10a_push]  (N=90750, 1815s, dt=0.020s, sim→real ×1.15)

| joint | motor | RMS/rated | P99/peak | Peak/peak | spd P99/Peak rpm | out-env% | 열 S / T_ss | P_mech pk | verdict |
|---|---|--:|--:|--:|--:|--:|:--:|--:|:--:|
| hip_pitch | RS04 | 19.3/40 (48%) OK | 66.5 (55%) | 89.2 (74%) OK | 51/91 | 0.00 | S=0.34 / 53C OK | 689W | OK |
| hip_roll | RS04 | 29.6/40 (74%) OK | 71.4 (59%) | 138.0 (115%) ⚠ | 36/76 | 0.01 | S=0.44 / 91C OK | 546W | **FAIL** |
| hip_yaw | RS03 | 12.6/20 (63%) OK | 44.6 (74%) | 68.6 (114%) ⚠ | 53/109 | 0.02 | S=0.69 / 72C OK | 650W | **FAIL** |
| knee | RS04 | 14.5/40 (36%) OK | 48.4 (40%) | 134.2 (112%) ⚠ | 106/206 | 0.00 | S=0.08 / 41C OK | 1457W | **FAIL** |
| crank_A | RS03 | 8.4/20 (42%) OK | 27.8 (46%) | 43.5 (73%) OK | 50/120 | 0.00 | S=0.02 / 46C OK | 344W | OK |
| crank_B | RS03 | 9.4/20 (47%) OK | 27.3 (45%) | 49.4 (82%) OK | 65/134 | 0.00 | S=0.01 / 52C OK | 328W | OK |

**v10a_push 종합**: 미달 3 — hip_roll, hip_yaw, knee ⚠  · 시스템 피크: P_mech 4013W / P_elec(추정) 2628W (η=0.8)

**[v10a_push] ankle-axis equivalent** — tauank_eq = Jᶜᵀ·τ_crank, the joint-axis demand behind the crank motors (interpretation only, ×1.15; the sizing motor is the RS03 crank above):
| ankle axis | RMS | P99 | Peak [N·m] |
|---|--:|--:|--:|
| pitch | 7.8 | 20.8 | 45.4 |
| roll | 11.0 | 35.0 | 58.2 |