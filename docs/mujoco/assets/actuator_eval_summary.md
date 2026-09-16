
### [v6b_push]  (N=90750, 1815s, dt=0.020s, sim→real ×1.15)

| joint | motor | RMS/rated | P99/peak | Peak/peak | spd P99/Peak rpm | out-env% | 열 S / T_ss | P_mech pk | verdict |
|---|---|--:|--:|--:|--:|--:|:--:|--:|:--:|
| hip_pitch | RS04 | 16.1/40 (40%) OK | 62.2 (52%) | 89.5 (75%) OK | 39/84 | 0.00 | S=0.21 / 44C OK | 599W | OK |
| hip_roll | RS04 | 25.5/40 (64%) OK | 70.8 (59%) | 138.0 (115%) ⚠ | 21/94 | 0.01 | S=0.38 / 74C OK | 922W | **FAIL** |
| hip_yaw | RS03 | 8.6/20 (43%) OK | 33.3 (56%) | 68.6 (114%) ⚠ | 36/108 | 0.01 | S=0.25 / 47C OK | 437W | **FAIL** |
| knee | RS04 | 12.1/40 (30%) OK | 43.4 (36%) | 133.0 (111%) ⚠ | 66/184 | 0.00 | S=0.02 / 36C OK | 1337W | **FAIL** |
| crank_A | RS03 | 9.5/20 (47%) OK | 30.3 (51%) | 47.5 (79%) OK | 52/118 | 0.00 | S=0.04 / 52C OK | 312W | OK |
| crank_B | RS03 | 11.1/20 (56%) OK | 32.4 (54%) | 44.0 (73%) OK | 56/111 | 0.00 | S=0.11 / 62C OK | 261W | OK |

**v6b_push 종합**: 미달 3 — hip_roll, hip_yaw, knee ⚠  · 시스템 피크: P_mech 3869W / P_elec(추정) 2638W (η=0.8)

**[v6b_push] ankle-axis equivalent** — tauank_eq = Jᶜᵀ·τ_crank, the joint-axis demand behind the crank motors (interpretation only, ×1.15; the sizing motor is the RS03 crank above):
| ankle axis | RMS | P99 | Peak [N·m] |
|---|--:|--:|--:|
| pitch | 7.1 | 19.6 | 43.3 |
| roll | 13.9 | 44.3 | 59.1 |