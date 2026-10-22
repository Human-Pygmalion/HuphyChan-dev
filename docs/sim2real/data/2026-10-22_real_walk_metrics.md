## timing

| run | duration_s | loop_hz_setting | policy_interval_median_ms | policy_interval_p95_ms | policy_interval_max_ms | frac_interval_gt_30ms | loop_period_median_ms | loop_rate_mean_hz | reply_latency_median_ms_R | reply_latency_median_ms_L | reply_latency_p99_ms_R | reply_latency_p99_ms_L | non_ontime_frac | pairing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1019 stand g0.8 | 6.37 |  | 26.3 | 37.1 | 47.2 | 0.219 |  |  |  |  |  |  |  |  |
| 1019 stand g1 #1 | 4.03 |  | 28.1 | 37 | 41.7 | 0.326 |  |  |  |  |  |  |  |  |
| 1019 stand g1 #2 | 1.04 |  | 27.3 | 34.6 | 36.4 | 0.184 |  |  |  |  |  |  |  |  |
| 1019 stand g1 #3 | 0.795 |  | 27.8 | 35.7 | 41.3 | 0.25 |  |  |  |  |  |  |  |  |
| 1020 vx0 g1 hz100 | 80.8 | 100 | 30.9 | 52.5 | 121 | 0.544 | 14.3 | 60.6 | 0.594 | 0.571 | 8.05 | 5.03 | 0 | paired |
| 1021 vx0 g0.6/1 A | 47.4 | 200 | 62.6 | 91.5 | 126 | 1 | 13.4 | 63.4 | 0.595 | 0.567 | 7.64 | 5.08 | 0 | paired |
| 1021 vx0 g1 vib-check | 209 | 200 | 53 | 71 | 145 | 1 | 12.4 | 73.6 | 0.653 | 0.653 | 5.62 | 5.07 | 0 | paired |
| 1021 vx0 g1 Kd x1.5 | 101 | 200 | 52.8 | 68 | 117 | 1 | 12.6 | 74.5 | 0.665 | 0.668 | 4.52 | 4.18 | 4.43e-05 | paired |
| 1021 vx0.3 g1 | 3.66 | 200 | 54.8 | 68.3 | 79.3 | 1 | 13.6 | 71 | 0.667 | 0.677 | 3.49 | 2.96 | 0 | paired |
| 1021 vx0.5 g1 | 5 | 200 | 55.9 | 65.9 | 75.1 | 1 | 13.5 | 71.9 | 0.68 | 0.67 | 3.27 | 2.86 | 0 | paired |
| 1021 vx0.5 g0.6/1.5 broken | 4.68 | 200 | 54.4 | 65.5 | 72.2 | 1 | 13.2 | 72.7 | 0.665 | 0.662 | 3.12 | 3.27 | 0 | paired |
| 1021 vx0 g0.6/1 B | 67.6 | 200 | 54.7 | 70.1 | 104 | 1 | 12.9 | 72.3 | 0.674 | 0.673 | 5.82 | 5.68 | 0 | paired |
| 1021 wz0.5 g1 | 35.8 | 200 | 50 | 62 | 101 | 1 | 11.9 | 78.9 | 0.62 | 0.614 | 3.47 | 3.16 | 0 | paired |
| 1021 wz0.5 g1 soles | 21.3 | 200 | 51.2 | 65 | 88.8 | 1 | 12.2 | 76.5 | 0.632 | 0.632 | 4.09 | 3.66 | 0 | paired |
| 1021 wz0.5 g0.6/1 | 44.4 | 200 | 49.7 | 62.1 | 88.8 | 1 | 11.8 | 79.3 | 0.623 | 0.62 | 3.69 | 3.35 | 0 | paired |

## freeze

| run | gyro_repeat_frac | gyro_repeat_longest_rows | gyro_repeat_longest_s | grav_repeat_frac | q_repeat_frac | frozen_time_frac | heading_change_deg | heading_live_only_deg | heading_live_rate_x_duration_deg |
|---|---|---|---|---|---|---|---|---|---|
| 1019 stand g0.8 | 0 | 0 | 0 | 0 | 0 | 0 | 3.34 | 3.34 | 3.34 |
| 1019 stand g1 #1 | 0.00704 | 1 | 0.0345 | 0.00704 | 0 | 0.00855 | -11.9 | -11.9 | -12 |
| 1019 stand g1 #2 | 0 | 0 | 0 | 0 | 0 | 0 | -23.8 | -23.8 | -23.8 |
| 1019 stand g1 #3 | 0 | 0 | 0 | 0 | 0 | 0 | -2.48 | -2.48 | -2.48 |
| 1020 vx0 g1 hz100 | 0.0458 | 110 | 3.36 | 0.0388 | 0.0237 | 0.0424 | 56.1 | 56.1 | 58.6 |
| 1021 vx0 g0.6/1 A | 0.206 | 138 | 8.13 | 0.178 | 0.0146 | 0.19 | -15.3 | -15.3 | -18.9 |
| 1021 vx0 g1 vib-check | 0.0558 | 44 | 2.33 | 0.028 | 0.00675 | 0.0549 | -57.7 | -57.7 | -61.1 |
| 1021 vx0 g1 Kd x1.5 | 0.121 | 25 | 1.31 | 0.043 | 0.0117 | 0.114 | -49 | -49 | -55.4 |
| 1021 vx0.3 g1 | 0 | 0 | 0 | 0 | 0 | 0 | -4.86 | -4.86 | -4.86 |
| 1021 vx0.5 g1 | 0 | 0 | 0 | 0 | 0 | 0 | 2.89 | 2.89 | 2.89 |
| 1021 vx0.5 g0.6/1.5 broken | 0 | 0 | 0 | 0 | 0 | 0 | -51.6 | -51.6 | -51.6 |
| 1021 vx0 g0.6/1 B | 0.13 | 49 | 2.49 | 0.0736 | 0.0147 | 0.127 | 2.35 | 2.35 | 2.69 |
| 1021 wz0.5 g1 | 0 | 0 | 0 | 0 | 0 | 0 | 597 | 597 | 597 |
| 1021 wz0.5 g1 soles | 0 | 0 | 0 | 0 | 0 | 0 | 254 | 254 | 254 |
| 1021 wz0.5 g0.6/1 | 0 | 0 | 0 | 0 | 0 | 0 | 716 | 716 | 716 |

## heading

| run | cmd_vx | cmd_wz | heading_change_deg | yaw_rate_mean_deg_s | gyro_z_bias_quiet2s_deg_s | tilt_fwd_quiet_deg | tilt_left_quiet_deg | tilt_fwd_run_deg | tilt_left_run_deg | gyro_z_rms_deg_s |
|---|---|---|---|---|---|---|---|---|---|---|
| 1019 stand g0.8 | 0 | 0 | 3.34 | 0.524 | -2.9 | -1.66 | -0.265 | -2.67 | 0.152 | 52.5 |
| 1019 stand g1 #1 | 0 | 0 | -11.9 | -2.95 | -1.41 | -2.22 | 0.151 | -2.64 | 0.196 | 48.1 |
| 1019 stand g1 #2 | 0 | 0 | -23.8 | -22.9 |  |  |  | -0.318 | 1.55 | 68.3 |
| 1019 stand g1 #3 | 0 | 0 | -2.48 | -3.11 |  |  |  | -1.19 | 0.638 | 55.1 |
| 1020 vx0 g1 hz100 | 0 | 0 | 56.1 | 0.695 | -0.0406 | 0.223 | 0.681 | 0.483 | 0.612 | 21.9 |
| 1021 vx0 g0.6/1 A | 0 | 0 | -15.3 | -0.323 | 0.0139 | 0.916 | 0.268 | -1.54 | 1.13 | 8.03 |
| 1021 vx0 g1 vib-check | 0 | 0 | -57.7 | -0.276 | -0.0228 | 1.32 | 0.556 | 0.729 | 1.02 | 23.3 |
| 1021 vx0 g1 Kd x1.5 | 0 | 0 | -49 | -0.485 | 0.0133 | 0.433 | 0.00164 | -0.487 | 0.389 | 25.4 |
| 1021 vx0.3 g1 | 0.3 | 0 | -4.86 | -1.33 | 2.64 | -1.52 | 0.638 | -1.9 | -0.199 | 45.5 |
| 1021 vx0.5 g1 | 0.5 | 0 | 2.89 | 0.577 | -4.45 | -3.67 | 0.205 | -2.39 | -0.584 | 42.1 |
| 1021 vx0.5 g0.6/1.5 broken | 0.5 | 0 | -51.6 | -11 | 1.27 | 0.498 | -0.02 | -2.32 | -2.09 | 73.6 |
| 1021 vx0 g0.6/1 B | 0 | 0 | 2.35 | 0.0348 | 0.0848 | -0.772 | -1.03 | -1.23 | 0.301 | 18.5 |
| 1021 wz0.5 g1 | 0 | 0.5 | 597 | 16.7 | 17.2 | -2.1 | 2.39 | -2.03 | 0.672 | 48.5 |
| 1021 wz0.5 g1 soles | 0 | 0.5 | 254 | 11.9 | 15 | -2.97 | 0.552 | -2.34 | 1.07 | 55.5 |
| 1021 wz0.5 g0.6/1 | 0 | 0.5 | 716 | 16.1 | 14.9 | -2.48 | -0.123 | -2.94 | -0.38 | 32.7 |

## gait

| run | f_gait_hz | LR_phase_deg | LR_xcorr_coef | rms_deg.hip_pitch | rms_deg.knee | asym_rms.hip_pitch | asym_rms.knee | offset_mirror_deg.hip_pitch | offset_mirror_deg.knee |
|---|---|---|---|---|---|---|---|---|---|
| 1019 stand g0.8 | 0.418 | 1.14 | 0.689 | 5.08 | 6.21 | 0.0983 | -0.274 | 1.1 | -0.429 |
| 1019 stand g1 #1 | 1.23 | 37.3 | 0.676 | 2.29 | 2.37 | 0.471 | -0.193 | -0.808 | 0.141 |
| 1019 stand g1 #2 | 1.02 |  |  | 3.07 | 4.2 | -0.227 | -0.231 | 0.504 | -2.11 |
| 1019 stand g1 #3 | 1.24 |  |  | 5.47 | 6.71 | 0.00366 | 0.441 | 2.25 | -0.694 |
| 1020 vx0 g1 hz100 | 0.85 | 72.4 | 0.733 | 1.12 | 1.84 | -0.162 | -1 | 0.26 | 0.0583 |
| 1021 vx0 g0.6/1 A | 0.4 | -3.47 | 0.186 | 1.82 | 3.02 | 1 | -0.246 | 0.159 | -0.176 |
| 1021 vx0 g1 vib-check | 5.1 | 158 | 0.0302 | 0.84 | 0.962 | 0.147 | 0.16 | -0.348 | 0.0738 |
| 1021 vx0 g1 Kd x1.5 | 0.32 | -98.1 | 0.388 | 1.99 | 3 | 0.123 | -0.0897 | -1.39 | -0.0751 |
| 1021 vx0.3 g1 | 1.05 | -145 | 0.317 | 4.91 | 6.31 | -0.573 | 0.332 | 4.96 | -0.243 |
| 1021 vx0.5 g1 | 0.8 | -144 | 0.519 | 4.29 | 6.52 | 0.0762 | 0.169 | 0.545 | -0.67 |
| 1021 vx0.5 g0.6/1.5 broken | 0.727 | -168 | 0.266 | 6.46 | 9.8 | 0.203 | 0.374 | -1.94 | 2.57 |
| 1021 vx0 g0.6/1 B | 0.36 | 32.1 | 0.0701 | 3.35 | 4.64 | -0.352 | -0.468 | -0.152 | -0.723 |
| 1021 wz0.5 g1 | 1.32 | -171 | 0.169 | 3.12 | 6.15 | -0.0886 | -0.224 | -1.18 | -2.77 |
| 1021 wz0.5 g1 soles | 1.54 | -173 | 0.286 | 3.27 | 6.81 | -0.0282 | -0.235 | -1.07 | -3.4 |
| 1021 wz0.5 g0.6/1 | 1.28 | -122 | 0.54 | 2.61 | 5.16 | 0.00404 | 0.0725 | -0.904 | 1.61 |

## jitter

| run | f_dominant_hz | f_hi_hz | pos_var_total_deg2 | pos_band_frac.0-1 | pos_band_frac.1-3 | pos_band_frac.3-6 | pos_band_frac.6-Nyq | track_err_rms_deg_mean | act_step_rms_mean | gyro_band_frac.3-6 | gyro_live_band_frac.3-6 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1019 stand g0.8 | 1.57 | 5.34 | 94.4 | 0.507 | 0.257 | 0.151 | 0.0843 | 9.55 | 0.549 | 0.318 | 0.318 |
| 1019 stand g1 #1 | 6.92 | 6.92 | 40.7 | 0.215 | 0.215 | 0.302 | 0.268 | 7.54 | 0.496 | 0.253 | 0.253 |
| 1019 stand g1 #2 |  |  |  |  |  |  |  |  |  |  |  |
| 1019 stand g1 #3 |  |  |  |  |  |  |  |  |  |  |  |
| 1020 vx0 g1 hz100 | 0.379 | 5.56 | 1.1 | 0.499 | 0.0979 | 0.298 | 0.106 | 3.46 | 0.148 | 0.671 | 0.671 |
| 1021 vx0 g0.6/1 A | 0.437 | 3.12 | 8.97 | 0.985 | 0.0142 | 0.00104 | 0.000113 | 4.16 | 0.114 | 0.133 | 0.134 |
| 1021 vx0 g1 vib-check | 5.09 | 5.09 | 5.37 | 0.62 | 0.178 | 0.191 | 0.0116 | 3.37 | 0.157 | 0.778 | 0.777 |
| 1021 vx0 g1 Kd x1.5 | 0.444 | 4.96 | 6.26 | 0.877 | 0.081 | 0.0399 | 0.00198 | 4 | 0.183 | 0.584 | 0.577 |
| 1021 vx0.3 g1 |  |  |  |  |  |  |  |  |  |  |  |
| 1021 vx0.5 g1 |  |  |  |  |  |  |  |  |  |  |  |
| 1021 vx0.5 g0.6/1.5 broken |  |  |  |  |  |  |  |  |  |  |  |
| 1021 vx0 g0.6/1 B | 0.643 | 3.14 | 55.6 | 0.777 | 0.189 | 0.0315 | 0.00247 | 5.09 | 0.197 | 0.383 | 0.372 |
| 1021 wz0.5 g1 | 1.41 | 3.12 | 165 | 0.23 | 0.684 | 0.0779 | 0.0081 | 7.63 | 0.487 | 0.498 | 0.498 |
| 1021 wz0.5 g1 soles | 1.5 | 3.1 | 176 | 0.237 | 0.68 | 0.0754 | 0.00818 | 7.69 | 0.495 | 0.47 | 0.47 |
| 1021 wz0.5 g0.6/1 | 1.26 | 3.69 | 118 | 0.145 | 0.804 | 0.0472 | 0.00395 | 7.2 | 0.351 | 0.527 | 0.527 |

## contrast gain 1 vs 0.6 (vx0, standing command): 1021 vx0 g1 vib-check -> 1021 vx0 g0.6/1 B

| metric | a | b | b-a | ratio |
|---|---|---|---|---|
| duration_s | 209.2 | 67.57 | -141.7 | 0.323 |
| loop_hz_setting | 200 | 200 | 0 | 1 |
| policy_interval_median_ms | 53 | 54.7 | 1.7 | 1.03 |
| policy_interval_p95_ms | 71 | 70.1 | -0.9 | 0.987 |
| policy_interval_max_ms | 145 | 103.7 | -41.3 | 0.715 |
| frac_interval_gt_30ms | 1 | 1 | 0 | 1 |
| loop_period_median_ms | 12.4 | 12.9 | 0.5 | 1.04 |
| loop_rate_mean_hz | 73.62 | 72.34 | -1.285 | 0.983 |
| reply_latency_median_ms_R | 0.6531 | 0.6736 | 0.0205 | 1.03 |
| reply_latency_median_ms_L | 0.6533 | 0.6727 | 0.01945 | 1.03 |
| reply_latency_p99_ms_R | 5.617 | 5.816 | 0.199 | 1.04 |
| reply_latency_p99_ms_L | 5.072 | 5.684 | 0.6119 | 1.12 |
| non_ontime_frac | 0 | 0 | 0 |  |
| gyro_repeat_frac | 0.05582 | 0.13 | 0.07419 | 2.33 |
| gyro_repeat_longest_rows | 44 | 49 | 5 | 1.11 |
| gyro_repeat_longest_s | 2.335 | 2.493 | 0.1581 | 1.07 |
| grav_repeat_frac | 0.02804 | 0.07359 | 0.04555 | 2.62 |
| q_repeat_frac | 0.00675 | 0.01472 | 0.007968 | 2.18 |
| frozen_time_frac | 0.05492 | 0.1267 | 0.07183 | 2.31 |
| heading_change_deg | -57.72 | 2.351 | 60.07 | -0.0407 |
| heading_live_only_deg | -57.72 | 2.351 | 60.07 | -0.0407 |
| heading_live_rate_x_duration_deg | -61.07 | 2.693 | 63.77 | -0.0441 |
| cmd_vx | 0 | 0 | 0 |  |
| cmd_wz | 0 | 0 | 0 |  |
| yaw_rate_mean_deg_s | -0.2759 | 0.0348 | 0.3107 | -0.126 |
| gyro_z_bias_quiet2s_deg_s | -0.0228 | 0.0848 | 0.1076 | -3.72 |
| tilt_fwd_quiet_deg | 1.32 | -0.7717 | -2.092 | -0.584 |
| tilt_left_quiet_deg | 0.5555 | -1.03 | -1.586 | -1.85 |
| tilt_fwd_run_deg | 0.7293 | -1.234 | -1.963 | -1.69 |
| tilt_left_run_deg | 1.02 | 0.3012 | -0.7185 | 0.295 |
| gyro_z_rms_deg_s | 23.28 | 18.48 | -4.8 | 0.794 |
| f_gait_hz | 5.1 | 0.36 | -4.74 | 0.0706 |
| LR_phase_deg | 158.2 | 32.08 | -126.1 | 0.203 |
| LR_xcorr_coef | 0.03015 | 0.0701 | 0.03995 | 2.32 |
| rms_deg.hip_pitch | 0.84 | 3.345 | 2.505 | 3.98 |
| rms_deg.knee | 0.9624 | 4.639 | 3.676 | 4.82 |
| asym_rms.hip_pitch | 0.1467 | -0.3525 | -0.4992 | -2.4 |
| asym_rms.knee | 0.16 | -0.4677 | -0.6277 | -2.92 |
| offset_mirror_deg.hip_pitch | -0.3484 | -0.152 | 0.1964 | 0.436 |
| offset_mirror_deg.knee | 0.07376 | -0.7235 | -0.7972 | -9.81 |
| f_dominant_hz | 5.085 | 0.6427 | -4.443 | 0.126 |
| f_hi_hz | 5.085 | 3.142 | -1.943 | 0.618 |
| pos_var_total_deg2 | 5.369 | 55.61 | 50.24 | 10.4 |
| pos_band_frac.0-1 | 0.62 | 0.7769 | 0.1569 | 1.25 |
| pos_band_frac.1-3 | 0.1776 | 0.1891 | 0.01153 | 1.06 |
| pos_band_frac.3-6 | 0.1908 | 0.03151 | -0.1593 | 0.165 |
| pos_band_frac.6-Nyq | 0.0116 | 0.002472 | -0.009124 | 0.213 |
| track_err_rms_deg_mean | 3.369 | 5.094 | 1.725 | 1.51 |
| act_step_rms_mean | 0.1567 | 0.1972 | 0.04051 | 1.26 |
| gyro_band_frac.3-6 | 0.7782 | 0.3835 | -0.3947 | 0.493 |
| gyro_live_band_frac.3-6 | 0.7766 | 0.3719 | -0.4047 | 0.479 |

## contrast gain 1 vs 0.6, first 0.6 run: 1021 vx0 g1 vib-check -> 1021 vx0 g0.6/1 A

| metric | a | b | b-a | ratio |
|---|---|---|---|---|
| duration_s | 209.2 | 47.35 | -161.9 | 0.226 |
| loop_hz_setting | 200 | 200 | 0 | 1 |
| policy_interval_median_ms | 53 | 62.6 | 9.6 | 1.18 |
| policy_interval_p95_ms | 71 | 91.48 | 20.49 | 1.29 |
| policy_interval_max_ms | 145 | 126.4 | -18.6 | 0.872 |
| frac_interval_gt_30ms | 1 | 1 | 0 | 1 |
| loop_period_median_ms | 12.4 | 13.4 | 1 | 1.08 |
| loop_rate_mean_hz | 73.62 | 63.36 | -10.26 | 0.861 |
| reply_latency_median_ms_R | 0.6531 | 0.5953 | -0.05785 | 0.911 |
| reply_latency_median_ms_L | 0.6533 | 0.5669 | -0.0864 | 0.868 |
| reply_latency_p99_ms_R | 5.617 | 7.637 | 2.02 | 1.36 |
| reply_latency_p99_ms_L | 5.072 | 5.082 | 0.009718 | 1 |
| non_ontime_frac | 0 | 0 | 0 |  |
| gyro_repeat_frac | 0.05582 | 0.2064 | 0.1506 | 3.7 |
| gyro_repeat_longest_rows | 44 | 138 | 94 | 3.14 |
| gyro_repeat_longest_s | 2.335 | 8.134 | 5.799 | 3.48 |
| grav_repeat_frac | 0.02804 | 0.1784 | 0.1504 | 6.36 |
| q_repeat_frac | 0.00675 | 0.01465 | 0.007897 | 2.17 |
| frozen_time_frac | 0.05492 | 0.1902 | 0.1353 | 3.46 |
| heading_change_deg | -57.72 | -15.29 | 42.43 | 0.265 |
| heading_live_only_deg | -57.72 | -15.29 | 42.43 | 0.265 |
| heading_live_rate_x_duration_deg | -61.07 | -18.88 | 42.2 | 0.309 |
| cmd_vx | 0 | 0 | 0 |  |
| cmd_wz | 0 | 0 | 0 |  |
| yaw_rate_mean_deg_s | -0.2759 | -0.3228 | -0.04696 | 1.17 |
| gyro_z_bias_quiet2s_deg_s | -0.0228 | 0.0139 | 0.0367 | -0.61 |
| tilt_fwd_quiet_deg | 1.32 | 0.9161 | -0.4044 | 0.694 |
| tilt_left_quiet_deg | 0.5555 | 0.2682 | -0.2873 | 0.483 |
| tilt_fwd_run_deg | 0.7293 | -1.538 | -2.268 | -2.11 |
| tilt_left_run_deg | 1.02 | 1.135 | 0.1152 | 1.11 |
| gyro_z_rms_deg_s | 23.28 | 8.029 | -15.25 | 0.345 |
| f_gait_hz | 5.1 | 0.4 | -4.7 | 0.0784 |
| LR_phase_deg | 158.2 | -3.467 | -161.6 | -0.0219 |
| LR_xcorr_coef | 0.03015 | 0.1863 | 0.1561 | 6.18 |
| rms_deg.hip_pitch | 0.84 | 1.817 | 0.9769 | 2.16 |
| rms_deg.knee | 0.9624 | 3.018 | 2.056 | 3.14 |
| asym_rms.hip_pitch | 0.1467 | 1.002 | 0.8555 | 6.83 |
| asym_rms.knee | 0.16 | -0.2458 | -0.4058 | -1.54 |
| offset_mirror_deg.hip_pitch | -0.3484 | 0.1591 | 0.5075 | -0.457 |
| offset_mirror_deg.knee | 0.07376 | -0.1763 | -0.25 | -2.39 |
| f_dominant_hz | 5.085 | 0.4368 | -4.649 | 0.0859 |
| f_hi_hz | 5.085 | 3.12 | -1.965 | 0.614 |
| pos_var_total_deg2 | 5.369 | 8.969 | 3.6 | 1.67 |
| pos_band_frac.0-1 | 0.62 | 0.9847 | 0.3646 | 1.59 |
| pos_band_frac.1-3 | 0.1776 | 0.0142 | -0.1634 | 0.08 |
| pos_band_frac.3-6 | 0.1908 | 0.00104 | -0.1898 | 0.00545 |
| pos_band_frac.6-Nyq | 0.0116 | 0.0001131 | -0.01148 | 0.00975 |
| track_err_rms_deg_mean | 3.369 | 4.158 | 0.7884 | 1.23 |
| act_step_rms_mean | 0.1567 | 0.1142 | -0.04253 | 0.729 |
| gyro_band_frac.3-6 | 0.7782 | 0.1333 | -0.6449 | 0.171 |
| gyro_live_band_frac.3-6 | 0.7766 | 0.1343 | -0.6423 | 0.173 |

## contrast Kd x1 vs x1.5 (vx0): 1021 vx0 g1 vib-check -> 1021 vx0 g1 Kd x1.5

| metric | a | b | b-a | ratio |
|---|---|---|---|---|
| duration_s | 209.2 | 101.1 | -108.2 | 0.483 |
| loop_hz_setting | 200 | 200 | 0 | 1 |
| policy_interval_median_ms | 53 | 52.8 | -0.2 | 0.996 |
| policy_interval_p95_ms | 71 | 68 | -3 | 0.958 |
| policy_interval_max_ms | 145 | 116.5 | -28.5 | 0.803 |
| frac_interval_gt_30ms | 1 | 1 | 0 | 1 |
| loop_period_median_ms | 12.4 | 12.6 | 0.2 | 1.02 |
| loop_rate_mean_hz | 73.62 | 74.48 | 0.8574 | 1.01 |
| reply_latency_median_ms_R | 0.6531 | 0.665 | 0.0119 | 1.02 |
| reply_latency_median_ms_L | 0.6533 | 0.6679 | 0.0146 | 1.02 |
| reply_latency_p99_ms_R | 5.617 | 4.522 | -1.095 | 0.805 |
| reply_latency_p99_ms_L | 5.072 | 4.183 | -0.8887 | 0.825 |
| non_ontime_frac | 0 | 4.427e-05 | 4.427e-05 |  |
| gyro_repeat_frac | 0.05582 | 0.1206 | 0.06474 | 2.16 |
| gyro_repeat_longest_rows | 44 | 25 | -19 | 0.568 |
| gyro_repeat_longest_s | 2.335 | 1.315 | -1.02 | 0.563 |
| grav_repeat_frac | 0.02804 | 0.04302 | 0.01498 | 1.53 |
| q_repeat_frac | 0.00675 | 0.01168 | 0.004934 | 1.73 |
| frozen_time_frac | 0.05492 | 0.1145 | 0.05958 | 2.08 |
| heading_change_deg | -57.72 | -49.05 | 8.672 | 0.85 |
| heading_live_only_deg | -57.72 | -49.05 | 8.672 | 0.85 |
| heading_live_rate_x_duration_deg | -61.07 | -55.39 | 5.684 | 0.907 |
| cmd_vx | 0 | 0 | 0 |  |
| cmd_wz | 0 | 0 | 0 |  |
| yaw_rate_mean_deg_s | -0.2759 | -0.4853 | -0.2094 | 1.76 |
| gyro_z_bias_quiet2s_deg_s | -0.0228 | 0.0133 | 0.0361 | -0.583 |
| tilt_fwd_quiet_deg | 1.32 | 0.4334 | -0.8871 | 0.328 |
| tilt_left_quiet_deg | 0.5555 | 0.00164 | -0.5539 | 0.00295 |
| tilt_fwd_run_deg | 0.7293 | -0.4873 | -1.217 | -0.668 |
| tilt_left_run_deg | 1.02 | 0.3887 | -0.6309 | 0.381 |
| gyro_z_rms_deg_s | 23.28 | 25.41 | 2.133 | 1.09 |
| f_gait_hz | 5.1 | 0.32 | -4.78 | 0.0627 |
| LR_phase_deg | 158.2 | -98.14 | -256.3 | -0.62 |
| LR_xcorr_coef | 0.03015 | 0.3885 | 0.3583 | 12.9 |
| rms_deg.hip_pitch | 0.84 | 1.99 | 1.15 | 2.37 |
| rms_deg.knee | 0.9624 | 2.997 | 2.034 | 3.11 |
| asym_rms.hip_pitch | 0.1467 | 0.1231 | -0.02365 | 0.839 |
| asym_rms.knee | 0.16 | -0.08967 | -0.2496 | -0.561 |
| offset_mirror_deg.hip_pitch | -0.3484 | -1.391 | -1.043 | 3.99 |
| offset_mirror_deg.knee | 0.07376 | -0.07506 | -0.1488 | -1.02 |
| f_dominant_hz | 5.085 | 0.4439 | -4.642 | 0.0873 |
| f_hi_hz | 5.085 | 4.957 | -0.1287 | 0.975 |
| pos_var_total_deg2 | 5.369 | 6.265 | 0.8957 | 1.17 |
| pos_band_frac.0-1 | 0.62 | 0.8772 | 0.2571 | 1.41 |
| pos_band_frac.1-3 | 0.1776 | 0.08098 | -0.09658 | 0.456 |
| pos_band_frac.3-6 | 0.1908 | 0.03988 | -0.1509 | 0.209 |
| pos_band_frac.6-Nyq | 0.0116 | 0.001985 | -0.009611 | 0.171 |
| track_err_rms_deg_mean | 3.369 | 3.996 | 0.6269 | 1.19 |
| act_step_rms_mean | 0.1567 | 0.1834 | 0.02669 | 1.17 |
| gyro_band_frac.3-6 | 0.7782 | 0.5836 | -0.1946 | 0.75 |
| gyro_live_band_frac.3-6 | 0.7766 | 0.5774 | -0.1992 | 0.743 |

## contrast gain 1 vs 0.6 (wz0.5): 1021 wz0.5 g1 -> 1021 wz0.5 g0.6/1

| metric | a | b | b-a | ratio |
|---|---|---|---|---|
| duration_s | 35.84 | 44.42 | 8.576 | 1.24 |
| loop_hz_setting | 200 | 200 | 0 | 1 |
| policy_interval_median_ms | 50 | 49.7 | -0.3 | 0.994 |
| policy_interval_p95_ms | 62.04 | 62.1 | 0.0597 | 1 |
| policy_interval_max_ms | 101.5 | 88.8 | -12.7 | 0.875 |
| frac_interval_gt_30ms | 1 | 1 | 0 | 1 |
| loop_period_median_ms | 11.9 | 11.8 | -0.1 | 0.992 |
| loop_rate_mean_hz | 78.91 | 79.33 | 0.4251 | 1.01 |
| reply_latency_median_ms_R | 0.6196 | 0.6227 | 0.0031 | 1.01 |
| reply_latency_median_ms_L | 0.6141 | 0.6201 | 0.00605 | 1.01 |
| reply_latency_p99_ms_R | 3.468 | 3.688 | 0.22 | 1.06 |
| reply_latency_p99_ms_L | 3.16 | 3.352 | 0.1921 | 1.06 |
| non_ontime_frac | 0 | 0 | 0 |  |
| gyro_repeat_frac | 0 | 0 | 0 |  |
| gyro_repeat_longest_rows | 0 | 0 | 0 |  |
| gyro_repeat_longest_s | 0 | 0 | 0 |  |
| grav_repeat_frac | 0 | 0 | 0 |  |
| q_repeat_frac | 0 | 0 | 0 |  |
| frozen_time_frac | 0 | 0 | 0 |  |
| heading_change_deg | 597.1 | 715.8 | 118.7 | 1.2 |
| heading_live_only_deg | 597.1 | 715.8 | 118.7 | 1.2 |
| heading_live_rate_x_duration_deg | 597.1 | 715.8 | 118.7 | 1.2 |
| cmd_vx | 0 | 0 | 0 |  |
| cmd_wz | 0.5 | 0.5 | 0 | 1 |
| yaw_rate_mean_deg_s | 16.66 | 16.12 | -0.5444 | 0.967 |
| gyro_z_bias_quiet2s_deg_s | 17.21 | 14.92 | -2.294 | 0.867 |
| tilt_fwd_quiet_deg | -2.102 | -2.478 | -0.3768 | 1.18 |
| tilt_left_quiet_deg | 2.388 | -0.1233 | -2.511 | -0.0516 |
| tilt_fwd_run_deg | -2.03 | -2.941 | -0.9106 | 1.45 |
| tilt_left_run_deg | 0.6723 | -0.3799 | -1.052 | -0.565 |
| gyro_z_rms_deg_s | 48.54 | 32.7 | -15.85 | 0.674 |
| f_gait_hz | 1.32 | 1.28 | -0.04 | 0.97 |
| LR_phase_deg | -170.8 | -122.1 | 48.67 | 0.715 |
| LR_xcorr_coef | 0.1688 | 0.5399 | 0.3711 | 3.2 |
| rms_deg.hip_pitch | 3.122 | 2.613 | -0.5084 | 0.837 |
| rms_deg.knee | 6.146 | 5.156 | -0.9898 | 0.839 |
| asym_rms.hip_pitch | -0.08862 | 0.004042 | 0.09266 | -0.0456 |
| asym_rms.knee | -0.2236 | 0.07249 | 0.2961 | -0.324 |
| offset_mirror_deg.hip_pitch | -1.182 | -0.9036 | 0.2788 | 0.764 |
| offset_mirror_deg.knee | -2.768 | 1.614 | 4.382 | -0.583 |
| f_dominant_hz | 1.406 | 1.258 | -0.1487 | 0.894 |
| f_hi_hz | 3.125 | 3.694 | 0.569 | 1.18 |
| pos_var_total_deg2 | 164.6 | 117.7 | -46.91 | 0.715 |
| pos_band_frac.0-1 | 0.2303 | 0.1448 | -0.08555 | 0.629 |
| pos_band_frac.1-3 | 0.6837 | 0.8041 | 0.1204 | 1.18 |
| pos_band_frac.3-6 | 0.07786 | 0.0472 | -0.03067 | 0.606 |
| pos_band_frac.6-Nyq | 0.008101 | 0.003953 | -0.004148 | 0.488 |
| track_err_rms_deg_mean | 7.627 | 7.199 | -0.4278 | 0.944 |
| act_step_rms_mean | 0.4874 | 0.351 | -0.1364 | 0.72 |
| gyro_band_frac.3-6 | 0.4984 | 0.5267 | 0.02834 | 1.06 |
| gyro_live_band_frac.3-6 | 0.4984 | 0.5267 | 0.02834 | 1.06 |

## contrast soles attached (wz0.5): 1021 wz0.5 g1 -> 1021 wz0.5 g1 soles

| metric | a | b | b-a | ratio |
|---|---|---|---|---|
| duration_s | 35.84 | 21.33 | -14.51 | 0.595 |
| loop_hz_setting | 200 | 200 | 0 | 1 |
| policy_interval_median_ms | 50 | 51.2 | 1.2 | 1.02 |
| policy_interval_p95_ms | 62.04 | 64.96 | 2.924 | 1.05 |
| policy_interval_max_ms | 101.5 | 88.84 | -12.66 | 0.875 |
| frac_interval_gt_30ms | 1 | 1 | 0 | 1 |
| loop_period_median_ms | 11.9 | 12.2 | 0.3 | 1.03 |
| loop_rate_mean_hz | 78.91 | 76.5 | -2.407 | 0.97 |
| reply_latency_median_ms_R | 0.6196 | 0.632 | 0.01235 | 1.02 |
| reply_latency_median_ms_L | 0.6141 | 0.6322 | 0.0181 | 1.03 |
| reply_latency_p99_ms_R | 3.468 | 4.09 | 0.6227 | 1.18 |
| reply_latency_p99_ms_L | 3.16 | 3.658 | 0.4982 | 1.16 |
| non_ontime_frac | 0 | 0 | 0 |  |
| gyro_repeat_frac | 0 | 0 | 0 |  |
| gyro_repeat_longest_rows | 0 | 0 | 0 |  |
| gyro_repeat_longest_s | 0 | 0 | 0 |  |
| grav_repeat_frac | 0 | 0 | 0 |  |
| q_repeat_frac | 0 | 0 | 0 |  |
| frozen_time_frac | 0 | 0 | 0 |  |
| heading_change_deg | 597.1 | 254 | -343.1 | 0.425 |
| heading_live_only_deg | 597.1 | 254 | -343.1 | 0.425 |
| heading_live_rate_x_duration_deg | 597.1 | 254 | -343.1 | 0.425 |
| cmd_vx | 0 | 0 | 0 |  |
| cmd_wz | 0.5 | 0.5 | 0 | 1 |
| yaw_rate_mean_deg_s | 16.66 | 11.91 | -4.752 | 0.715 |
| gyro_z_bias_quiet2s_deg_s | 17.21 | 15.05 | -2.168 | 0.874 |
| tilt_fwd_quiet_deg | -2.102 | -2.966 | -0.8645 | 1.41 |
| tilt_left_quiet_deg | 2.388 | 0.5517 | -1.836 | 0.231 |
| tilt_fwd_run_deg | -2.03 | -2.341 | -0.3101 | 1.15 |
| tilt_left_run_deg | 0.6723 | 1.067 | 0.3945 | 1.59 |
| gyro_z_rms_deg_s | 48.54 | 55.48 | 6.94 | 1.14 |
| f_gait_hz | 1.32 | 1.542 | 0.2222 | 1.17 |
| LR_phase_deg | -170.8 | -173 | -2.19 | 1.01 |
| LR_xcorr_coef | 0.1688 | 0.2864 | 0.1176 | 1.7 |
| rms_deg.hip_pitch | 3.122 | 3.271 | 0.1497 | 1.05 |
| rms_deg.knee | 6.146 | 6.812 | 0.6656 | 1.11 |
| asym_rms.hip_pitch | -0.08862 | -0.02816 | 0.06045 | 0.318 |
| asym_rms.knee | -0.2236 | -0.2354 | -0.01182 | 1.05 |
| offset_mirror_deg.hip_pitch | -1.182 | -1.07 | 0.1125 | 0.905 |
| offset_mirror_deg.knee | -2.768 | -3.395 | -0.6269 | 1.23 |
| f_dominant_hz | 1.406 | 1.502 | 0.09615 | 1.07 |
| f_hi_hz | 3.125 | 3.099 | -0.02629 | 0.992 |
| pos_var_total_deg2 | 164.6 | 175.9 | 11.27 | 1.07 |
| pos_band_frac.0-1 | 0.2303 | 0.2367 | 0.006369 | 1.03 |
| pos_band_frac.1-3 | 0.6837 | 0.6798 | -0.003957 | 0.994 |
| pos_band_frac.3-6 | 0.07786 | 0.07537 | -0.002494 | 0.968 |
| pos_band_frac.6-Nyq | 0.008101 | 0.008183 | 8.148e-05 | 1.01 |
| track_err_rms_deg_mean | 7.627 | 7.689 | 0.06187 | 1.01 |
| act_step_rms_mean | 0.4874 | 0.4951 | 0.007667 | 1.02 |
| gyro_band_frac.3-6 | 0.4984 | 0.4703 | -0.02802 | 0.944 |
| gyro_live_band_frac.3-6 | 0.4984 | 0.4703 | -0.02802 | 0.944 |

## contrast vx 0.3 vs 0.5: 1021 vx0.3 g1 -> 1021 vx0.5 g1

| metric | a | b | b-a | ratio |
|---|---|---|---|---|
| duration_s | 3.664 | 5.003 | 1.339 | 1.37 |
| loop_hz_setting | 200 | 200 | 0 | 1 |
| policy_interval_median_ms | 54.84 | 55.93 | 1.09 | 1.02 |
| policy_interval_p95_ms | 68.28 | 65.93 | -2.351 | 0.966 |
| policy_interval_max_ms | 79.33 | 75.07 | -4.262 | 0.946 |
| frac_interval_gt_30ms | 1 | 1 | 0 | 1 |
| loop_period_median_ms | 13.6 | 13.5 | -0.1 | 0.993 |
| loop_rate_mean_hz | 71 | 71.91 | 0.9071 | 1.01 |
| reply_latency_median_ms_R | 0.6666 | 0.6802 | 0.0135 | 1.02 |
| reply_latency_median_ms_L | 0.6774 | 0.6704 | -0.007 | 0.99 |
| reply_latency_p99_ms_R | 3.489 | 3.269 | -0.2203 | 0.937 |
| reply_latency_p99_ms_L | 2.962 | 2.856 | -0.1061 | 0.964 |
| non_ontime_frac | 0 | 0 | 0 |  |
| gyro_repeat_frac | 0 | 0 | 0 |  |
| gyro_repeat_longest_rows | 0 | 0 | 0 |  |
| gyro_repeat_longest_s | 0 | 0 | 0 |  |
| grav_repeat_frac | 0 | 0 | 0 |  |
| q_repeat_frac | 0 | 0 | 0 |  |
| frozen_time_frac | 0 | 0 | 0 |  |
| heading_change_deg | -4.858 | 2.888 | 7.747 | -0.594 |
| heading_live_only_deg | -4.858 | 2.888 | 7.747 | -0.594 |
| heading_live_rate_x_duration_deg | -4.858 | 2.888 | 7.747 | -0.594 |
| cmd_vx | 0.3 | 0.5 | 0.2 | 1.67 |
| cmd_wz | 0 | 0 | 0 |  |
| yaw_rate_mean_deg_s | -1.326 | 0.5773 | 1.903 | -0.435 |
| gyro_z_bias_quiet2s_deg_s | 2.643 | -4.455 | -7.098 | -1.69 |
| tilt_fwd_quiet_deg | -1.522 | -3.67 | -2.148 | 2.41 |
| tilt_left_quiet_deg | 0.6385 | 0.205 | -0.4334 | 0.321 |
| tilt_fwd_run_deg | -1.895 | -2.389 | -0.4942 | 1.26 |
| tilt_left_run_deg | -0.1991 | -0.5844 | -0.3854 | 2.94 |
| gyro_z_rms_deg_s | 45.54 | 42.06 | -3.48 | 0.924 |
| f_gait_hz | 1.051 | 0.7997 | -0.2509 | 0.761 |
| LR_phase_deg | -144.8 | -143.6 | 1.246 | 0.991 |
| LR_xcorr_coef | 0.3172 | 0.5193 | 0.2021 | 1.64 |
| rms_deg.hip_pitch | 4.915 | 4.293 | -0.6212 | 0.874 |
| rms_deg.knee | 6.305 | 6.52 | 0.2144 | 1.03 |
| asym_rms.hip_pitch | -0.5732 | 0.07622 | 0.6494 | -0.133 |
| asym_rms.knee | 0.3323 | 0.1686 | -0.1637 | 0.507 |
| offset_mirror_deg.hip_pitch | 4.956 | 0.5447 | -4.412 | 0.11 |
| offset_mirror_deg.knee | -0.2428 | -0.6698 | -0.427 | 2.76 |

## contrast vx 0 vs 0.5 (gain 1): 1021 vx0 g1 vib-check -> 1021 vx0.5 g1

| metric | a | b | b-a | ratio |
|---|---|---|---|---|
| duration_s | 209.2 | 5.003 | -204.2 | 0.0239 |
| loop_hz_setting | 200 | 200 | 0 | 1 |
| policy_interval_median_ms | 53 | 55.93 | 2.93 | 1.06 |
| policy_interval_p95_ms | 71 | 65.93 | -5.069 | 0.929 |
| policy_interval_max_ms | 145 | 75.07 | -69.93 | 0.518 |
| frac_interval_gt_30ms | 1 | 1 | 0 | 1 |
| loop_period_median_ms | 12.4 | 13.5 | 1.1 | 1.09 |
| loop_rate_mean_hz | 73.62 | 71.91 | -1.715 | 0.977 |
| reply_latency_median_ms_R | 0.6531 | 0.6802 | 0.02705 | 1.04 |
| reply_latency_median_ms_L | 0.6533 | 0.6704 | 0.01705 | 1.03 |
| reply_latency_p99_ms_R | 5.617 | 3.269 | -2.348 | 0.582 |
| reply_latency_p99_ms_L | 5.072 | 2.856 | -2.216 | 0.563 |
| non_ontime_frac | 0 | 0 | 0 |  |
| gyro_repeat_frac | 0.05582 | 0 | -0.05582 | 0 |
| gyro_repeat_longest_rows | 44 | 0 | -44 | 0 |
| gyro_repeat_longest_s | 2.335 | 0 | -2.335 | 0 |
| grav_repeat_frac | 0.02804 | 0 | -0.02804 | 0 |
| q_repeat_frac | 0.00675 | 0 | -0.00675 | 0 |
| frozen_time_frac | 0.05492 | 0 | -0.05492 | 0 |
| heading_change_deg | -57.72 | 2.888 | 60.61 | -0.05 |
| heading_live_only_deg | -57.72 | 2.888 | 60.61 | -0.05 |
| heading_live_rate_x_duration_deg | -61.07 | 2.888 | 63.96 | -0.0473 |
| cmd_vx | 0 | 0.5 | 0.5 |  |
| cmd_wz | 0 | 0 | 0 |  |
| yaw_rate_mean_deg_s | -0.2759 | 0.5773 | 0.8532 | -2.09 |
| gyro_z_bias_quiet2s_deg_s | -0.0228 | -4.455 | -4.432 | 195 |
| tilt_fwd_quiet_deg | 1.32 | -3.67 | -4.991 | -2.78 |
| tilt_left_quiet_deg | 0.5555 | 0.205 | -0.3505 | 0.369 |
| tilt_fwd_run_deg | 0.7293 | -2.389 | -3.118 | -3.28 |
| tilt_left_run_deg | 1.02 | -0.5844 | -1.604 | -0.573 |
| gyro_z_rms_deg_s | 23.28 | 42.06 | 18.78 | 1.81 |
| f_gait_hz | 5.1 | 0.7997 | -4.3 | 0.157 |
| LR_phase_deg | 158.2 | -143.6 | -301.8 | -0.908 |
| LR_xcorr_coef | 0.03015 | 0.5193 | 0.4892 | 17.2 |
| rms_deg.hip_pitch | 0.84 | 4.293 | 3.453 | 5.11 |
| rms_deg.knee | 0.9624 | 6.52 | 5.557 | 6.77 |
| asym_rms.hip_pitch | 0.1467 | 0.07622 | -0.07049 | 0.52 |
| asym_rms.knee | 0.16 | 0.1686 | 0.008681 | 1.05 |
| offset_mirror_deg.hip_pitch | -0.3484 | 0.5447 | 0.893 | -1.56 |
| offset_mirror_deg.knee | 0.07376 | -0.6698 | -0.7435 | -9.08 |

## contrast vx 0 vs wz 0.5 (gain 1): 1021 vx0 g1 vib-check -> 1021 wz0.5 g1

| metric | a | b | b-a | ratio |
|---|---|---|---|---|
| duration_s | 209.2 | 35.84 | -173.4 | 0.171 |
| loop_hz_setting | 200 | 200 | 0 | 1 |
| policy_interval_median_ms | 53 | 50 | -3 | 0.943 |
| policy_interval_p95_ms | 71 | 62.04 | -8.96 | 0.874 |
| policy_interval_max_ms | 145 | 101.5 | -43.5 | 0.7 |
| frac_interval_gt_30ms | 1 | 1 | 0 | 1 |
| loop_period_median_ms | 12.4 | 11.9 | -0.5 | 0.96 |
| loop_rate_mean_hz | 73.62 | 78.91 | 5.285 | 1.07 |
| reply_latency_median_ms_R | 0.6531 | 0.6196 | -0.0335 | 0.949 |
| reply_latency_median_ms_L | 0.6533 | 0.6141 | -0.0392 | 0.94 |
| reply_latency_p99_ms_R | 5.617 | 3.468 | -2.15 | 0.617 |
| reply_latency_p99_ms_L | 5.072 | 3.16 | -1.912 | 0.623 |
| non_ontime_frac | 0 | 0 | 0 |  |
| gyro_repeat_frac | 0.05582 | 0 | -0.05582 | 0 |
| gyro_repeat_longest_rows | 44 | 0 | -44 | 0 |
| gyro_repeat_longest_s | 2.335 | 0 | -2.335 | 0 |
| grav_repeat_frac | 0.02804 | 0 | -0.02804 | 0 |
| q_repeat_frac | 0.00675 | 0 | -0.00675 | 0 |
| frozen_time_frac | 0.05492 | 0 | -0.05492 | 0 |
| heading_change_deg | -57.72 | 597.1 | 654.9 | -10.3 |
| heading_live_only_deg | -57.72 | 597.1 | 654.9 | -10.3 |
| heading_live_rate_x_duration_deg | -61.07 | 597.1 | 658.2 | -9.78 |
| cmd_vx | 0 | 0 | 0 |  |
| cmd_wz | 0 | 0.5 | 0.5 |  |
| yaw_rate_mean_deg_s | -0.2759 | 16.66 | 16.94 | -60.4 |
| gyro_z_bias_quiet2s_deg_s | -0.0228 | 17.21 | 17.24 | -755 |
| tilt_fwd_quiet_deg | 1.32 | -2.102 | -3.422 | -1.59 |
| tilt_left_quiet_deg | 0.5555 | 2.388 | 1.832 | 4.3 |
| tilt_fwd_run_deg | 0.7293 | -2.03 | -2.76 | -2.78 |
| tilt_left_run_deg | 1.02 | 0.6723 | -0.3473 | 0.659 |
| gyro_z_rms_deg_s | 23.28 | 48.54 | 25.27 | 2.09 |
| f_gait_hz | 5.1 | 1.32 | -3.78 | 0.259 |
| LR_phase_deg | 158.2 | -170.8 | -328.9 | -1.08 |
| LR_xcorr_coef | 0.03015 | 0.1688 | 0.1386 | 5.6 |
| rms_deg.hip_pitch | 0.84 | 3.122 | 2.281 | 3.72 |
| rms_deg.knee | 0.9624 | 6.146 | 5.184 | 6.39 |
| asym_rms.hip_pitch | 0.1467 | -0.08862 | -0.2353 | -0.604 |
| asym_rms.knee | 0.16 | -0.2236 | -0.3835 | -1.4 |
| offset_mirror_deg.hip_pitch | -0.3484 | -1.182 | -0.8341 | 3.39 |
| offset_mirror_deg.knee | 0.07376 | -2.768 | -2.842 | -37.5 |
| f_dominant_hz | 5.085 | 1.406 | -3.679 | 0.277 |
| f_hi_hz | 5.085 | 3.125 | -1.96 | 0.614 |
| pos_var_total_deg2 | 5.369 | 164.6 | 159.2 | 30.7 |
| pos_band_frac.0-1 | 0.62 | 0.2303 | -0.3897 | 0.371 |
| pos_band_frac.1-3 | 0.1776 | 0.6837 | 0.5062 | 3.85 |
| pos_band_frac.3-6 | 0.1908 | 0.07786 | -0.1129 | 0.408 |
| pos_band_frac.6-Nyq | 0.0116 | 0.008101 | -0.003495 | 0.699 |
| track_err_rms_deg_mean | 3.369 | 7.627 | 4.258 | 2.26 |
| act_step_rms_mean | 0.1567 | 0.4874 | 0.3307 | 3.11 |
| gyro_band_frac.3-6 | 0.7782 | 0.4984 | -0.2798 | 0.64 |
| gyro_live_band_frac.3-6 | 0.7766 | 0.4984 | -0.2782 | 0.642 |

## contrast loop 100 Hz (1020) vs 200 Hz (1021), vx0 gain 1: 1020 vx0 g1 hz100 -> 1021 vx0 g1 vib-check

| metric | a | b | b-a | ratio |
|---|---|---|---|---|
| duration_s | 80.75 | 209.2 | 128.5 | 2.59 |
| loop_hz_setting | 100 | 200 | 100 | 2 |
| policy_interval_median_ms | 30.9 | 53 | 22.1 | 1.72 |
| policy_interval_p95_ms | 52.5 | 71 | 18.5 | 1.35 |
| policy_interval_max_ms | 120.9 | 145 | 24.1 | 1.2 |
| frac_interval_gt_30ms | 0.5442 | 1 | 0.4558 | 1.84 |
| loop_period_median_ms | 14.3 | 12.4 | -1.9 | 0.867 |
| loop_rate_mean_hz | 60.59 | 73.62 | 13.03 | 1.22 |
| reply_latency_median_ms_R | 0.5935 | 0.6531 | 0.0596 | 1.1 |
| reply_latency_median_ms_L | 0.5708 | 0.6533 | 0.0825 | 1.14 |
| reply_latency_p99_ms_R | 8.053 | 5.617 | -2.436 | 0.698 |
| reply_latency_p99_ms_L | 5.035 | 5.072 | 0.03728 | 1.01 |
| non_ontime_frac | 0 | 0 | 0 |  |
| gyro_repeat_frac | 0.04577 | 0.05582 | 0.01004 | 1.22 |
| gyro_repeat_longest_rows | 110 | 44 | -66 | 0.4 |
| gyro_repeat_longest_s | 3.358 | 2.335 | -1.023 | 0.695 |
| grav_repeat_frac | 0.03882 | 0.02804 | -0.01079 | 0.722 |
| q_repeat_frac | 0.0237 | 0.00675 | -0.01695 | 0.285 |
| frozen_time_frac | 0.04243 | 0.05492 | 0.01248 | 1.29 |
| heading_change_deg | 56.14 | -57.72 | -113.9 | -1.03 |
| heading_live_only_deg | 56.14 | -57.72 | -113.9 | -1.03 |
| heading_live_rate_x_duration_deg | 58.63 | -61.07 | -119.7 | -1.04 |
| cmd_vx | 0 | 0 | 0 |  |
| cmd_wz | 0 | 0 | 0 |  |
| yaw_rate_mean_deg_s | 0.6952 | -0.2759 | -0.9711 | -0.397 |
| gyro_z_bias_quiet2s_deg_s | -0.0406 | -0.0228 | 0.0178 | 0.562 |
| tilt_fwd_quiet_deg | 0.2228 | 1.32 | 1.098 | 5.93 |
| tilt_left_quiet_deg | 0.6806 | 0.5555 | -0.125 | 0.816 |
| tilt_fwd_run_deg | 0.483 | 0.7293 | 0.2462 | 1.51 |
| tilt_left_run_deg | 0.6118 | 1.02 | 0.4079 | 1.67 |
| gyro_z_rms_deg_s | 21.85 | 23.28 | 1.424 | 1.07 |
| f_gait_hz | 0.85 | 5.1 | 4.25 | 6 |
| LR_phase_deg | 72.45 | 158.2 | 85.73 | 2.18 |
| LR_xcorr_coef | 0.7325 | 0.03015 | -0.7024 | 0.0412 |
| rms_deg.hip_pitch | 1.119 | 0.84 | -0.2794 | 0.75 |
| rms_deg.knee | 1.837 | 0.9624 | -0.8749 | 0.524 |
| asym_rms.hip_pitch | -0.1621 | 0.1467 | 0.3088 | -0.905 |
| asym_rms.knee | -1.002 | 0.16 | 1.161 | -0.16 |
| offset_mirror_deg.hip_pitch | 0.2595 | -0.3484 | -0.6079 | -1.34 |
| offset_mirror_deg.knee | 0.05827 | 0.07376 | 0.01549 | 1.27 |
| f_dominant_hz | 0.3792 | 5.085 | 4.706 | 13.4 |
| f_hi_hz | 5.562 | 5.085 | -0.4768 | 0.914 |
| pos_var_total_deg2 | 1.1 | 5.369 | 4.269 | 4.88 |
| pos_band_frac.0-1 | 0.4986 | 0.62 | 0.1214 | 1.24 |
| pos_band_frac.1-3 | 0.09785 | 0.1776 | 0.0797 | 1.81 |
| pos_band_frac.3-6 | 0.298 | 0.1908 | -0.1072 | 0.64 |
| pos_band_frac.6-Nyq | 0.1055 | 0.0116 | -0.09391 | 0.11 |
| track_err_rms_deg_mean | 3.458 | 3.369 | -0.08868 | 0.974 |
| act_step_rms_mean | 0.1484 | 0.1567 | 0.008271 | 1.06 |
| gyro_band_frac.3-6 | 0.6714 | 0.7782 | 0.1068 | 1.16 |
| gyro_live_band_frac.3-6 | 0.6714 | 0.7766 | 0.1052 | 1.16 |

## contrast vx0.5 gain 1 vs gain 0.6/1.5 (broken): 1021 vx0.5 g1 -> 1021 vx0.5 g0.6/1.5 broken

| metric | a | b | b-a | ratio |
|---|---|---|---|---|
| duration_s | 5.003 | 4.682 | -0.3207 | 0.936 |
| loop_hz_setting | 200 | 200 | 0 | 1 |
| policy_interval_median_ms | 55.93 | 54.35 | -1.58 | 0.972 |
| policy_interval_p95_ms | 65.93 | 65.53 | -0.4045 | 0.994 |
| policy_interval_max_ms | 75.07 | 72.18 | -2.888 | 0.962 |
| frac_interval_gt_30ms | 1 | 1 | 0 | 1 |
| loop_period_median_ms | 13.5 | 13.2 | -0.3 | 0.978 |
| loop_rate_mean_hz | 71.91 | 72.72 | 0.8124 | 1.01 |
| reply_latency_median_ms_R | 0.6802 | 0.6647 | -0.0154 | 0.977 |
| reply_latency_median_ms_L | 0.6704 | 0.6623 | -0.0081 | 0.988 |
| reply_latency_p99_ms_R | 3.269 | 3.12 | -0.1492 | 0.954 |
| reply_latency_p99_ms_L | 2.856 | 3.272 | 0.4157 | 1.15 |
| non_ontime_frac | 0 | 0 | 0 |  |
| gyro_repeat_frac | 0 | 0 | 0 |  |
| gyro_repeat_longest_rows | 0 | 0 | 0 |  |
| gyro_repeat_longest_s | 0 | 0 | 0 |  |
| grav_repeat_frac | 0 | 0 | 0 |  |
| q_repeat_frac | 0 | 0 | 0 |  |
| frozen_time_frac | 0 | 0 | 0 |  |
| heading_change_deg | 2.888 | -51.61 | -54.5 | -17.9 |
| heading_live_only_deg | 2.888 | -51.61 | -54.5 | -17.9 |
| heading_live_rate_x_duration_deg | 2.888 | -51.61 | -54.5 | -17.9 |
| cmd_vx | 0.5 | 0.5 | 0 | 1 |
| cmd_wz | 0 | 0 | 0 |  |
| yaw_rate_mean_deg_s | 0.5773 | -11.02 | -11.6 | -19.1 |
| gyro_z_bias_quiet2s_deg_s | -4.455 | 1.268 | 5.722 | -0.285 |
| tilt_fwd_quiet_deg | -3.67 | 0.4983 | 4.168 | -0.136 |
| tilt_left_quiet_deg | 0.205 | -0.02 | -0.2251 | -0.0976 |
| tilt_fwd_run_deg | -2.389 | -2.324 | 0.06564 | 0.973 |
| tilt_left_run_deg | -0.5844 | -2.092 | -1.508 | 3.58 |
| gyro_z_rms_deg_s | 42.06 | 73.64 | 31.58 | 1.75 |
| f_gait_hz | 0.7997 | 0.7272 | -0.07256 | 0.909 |
| LR_phase_deg | -143.6 | -168.2 | -24.62 | 1.17 |
| LR_xcorr_coef | 0.5193 | 0.2658 | -0.2535 | 0.512 |
| rms_deg.hip_pitch | 4.293 | 6.463 | 2.17 | 1.51 |
| rms_deg.knee | 6.52 | 9.802 | 3.282 | 1.5 |
| asym_rms.hip_pitch | 0.07622 | 0.2034 | 0.1272 | 2.67 |
| asym_rms.knee | 0.1686 | 0.3741 | 0.2054 | 2.22 |
| offset_mirror_deg.hip_pitch | 0.5447 | -1.943 | -2.488 | -3.57 |
| offset_mirror_deg.knee | -0.6698 | 2.572 | 3.242 | -3.84 |

