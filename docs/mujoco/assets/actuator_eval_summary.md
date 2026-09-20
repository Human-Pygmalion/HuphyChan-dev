
### [connectome]  (N=90750, 1815s, dt=0.020s, sim→real ×1.15)

| joint | motor | RMS/rated | P99/peak | Peak/peak | spd P99/Peak rpm | out-env% | 열 S / T_ss | P_mech pk | verdict |
|---|---|--:|--:|--:|--:|--:|:--:|--:|:--:|
| hip_pitch | RS04 | 11.4/40 (29%) OK | 34.8 (29%) | 62.7 (52%) OK | 31/82 | 0.00 | S=0.00 / 35C OK | 316W | OK |
| hip_roll | RS04 | 22.8/40 (57%) OK | 67.5 (56%) | 120.3 (100%) ⚠ | 29/70 | 0.00 | S=0.24 / 64C OK | 687W | **FAIL** |
| hip_yaw | RS03 | 5.8/20 (29%) OK | 20.4 (34%) | 44.6 (74%) OK | 16/39 | 0.00 | S=0.00 / 35C OK | 115W | OK |
| knee | RS04 | 11.2/40 (28%) OK | 36.4 (30%) | 100.7 (84%) OK | 69/127 | 0.00 | S=0.01 / 34C OK | 604W | OK |
| crank_A | RS03 | 9.7/20 (49%) OK | 31.1 (52%) | 49.5 (82%) OK | 51/137 | 0.00 | S=0.07 / 53C OK | 231W | OK |
| crank_B | RS03 | 11.8/20 (59%) OK | 33.2 (55%) | 43.2 (72%) OK | 54/130 | 0.00 | S=0.11 / 67C OK | 235W | OK |

**connectome 종합**: 미달 1 — hip_roll ⚠  · 시스템 피크: P_mech 2188W / P_elec(추정) 1407W (η=0.8)

**[connectome] ankle-axis equivalent** — tauank_eq = Jᶜᵀ·τ_crank, the joint-axis demand behind the crank motors (interpretation only, ×1.15; the sizing motor is the RS03 crank above):
| ankle axis | RMS | P99 | Peak [N·m] |
|---|--:|--:|--:|
| pitch | 6.5 | 19.2 | 31.6 |
| roll | 14.8 | 45.6 | 66.1 |