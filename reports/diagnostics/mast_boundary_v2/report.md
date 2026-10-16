# MAST boundary diagnostic (flags v2)

Run of 2026-10-16T05:44:59+00:00, repository commit 64d128ebfbd029545619ea24f935ce16ac87af20. 1206 shots, 69075 frames per set at stride 1. Map sets: arm_g (`/Users/kamia/Desktop/PROJECTS/fusion-challenge/fusion-equilibrium-challenge/submissions/2026-10-15_arm_g_solhalf/mast_public_test.npz`), set_b (`/Users/kamia/Desktop/PROJECTS/fusion-challenge/fusion-equilibrium-challenge/submissions/2026-10-15_arm_setb_g_solhalf/mast_public_test.npz`), control (coil flux of sol_half plus Ip phi_mean, c = 0, no remainder). No MAST equilibrium label read.

## Method

- Boundary and axis: the organizers' `extract_lcfs_with_sign` and `derive_frame`, unmodified, machine MAST, axis sign +1, on the float16 maps read as float64 (as the scorer reads them).
- Wall: the 0.5 isoline of the scorer's MAST envelope mask (the boundary must stay inside it); centre column: its points at R <= 0.3 m. `machines.py` holds no wall or centre column.
- Saddles: zeros of the gradient of a bicubic spline of the map (Newton from minima of |grad psi|^2), classified by the Hessian; the scorer has no saddle finder. dpsi_sep uses the two saddles inside the envelope closest in flux to the boundary.
- Phase: `coil_scales.phase_of` on the frame table's ip_over_peak (flat-top at or above its level of peak Ip; rising and falling by position in the shot).
- Clip counts: each arm's model columns scaled by its stored DIII-D scaler, beyond its training clip bound.
- Known shots: submission keys mapped to MAST shot numbers through the writing order of the submission's run record (`folds.MAST.shot_checks`); they choose what is plotted and enter the overlap only.

## Flag definitions and frozen thresholds

Flags: `diag_mast_boundary_v2.yaml` (version 2), SHA-256 `82bb03ce9b45e33a263c3e5477fa0d5952f0a116d6238be4a373e8a07b9d0f20`. Written after the quick run's base rates and before any overlap with the known shots. Reason: |dpsi_sep| < 0.02 fired on about 0.9 of the frames of all three sets, the physics-only control included, so F2 and F5 flagged every shot. MAST runs routinely near double null; a small dpsi_sep is its normal state, not an anomaly. dpsi_sep stays a continuous per-frame column.

Comparison: `diag_mast_boundary_v1.yaml` (version 1), SHA-256 `7bfd7b0ba4c4843894ec65d3431bde5bc4302d2e818470c19fbfde907cca5c44`. Written before any comparison with known shots.

| threshold | v2 | v1 |
| --- | --- | --- |
| axis_sign | 1.0 | 1.0 |
| robust_z_max | 3.5 | 3.5 |
| a_band_m | (0.35, 0.75) | (0.35, 0.75) |
| area_band_m2 | (0.5, 4.2) | (0.5, 4.2) |
| dpsi_sep_max | None | 0.02 |
| xpoint_flux_tol | 0.02 | 0.02 |
| xpoint_contour_max_m | 0.08 | 0.08 |
| wall_tol_m | 0.03 | 0.03 |
| centre_column_max_r_m | 0.3 | 0.3 |
| grid_edge_m | 0.04 | 0.04 |
| da_max_m | 0.05 | 0.05 |
| dr0_max_m | 0.05 | 0.05 |
| flag_fraction | 0.2 | 0.2 |
| coil_active_ka | 5.0 | 5.0 |

- F1 size: |robust z| of a or area above robust_z_max (over the set's frames), or a or area outside its band.
- F2 topology: the defining saddle changes side (upper or lower) between consecutive frames. (v1: F2 topology: |dpsi_sep| below 0.02, or the defining saddle changes side between consecutive frames.)
- F3 limiting: limited on the centre column, within grid_edge_m of the grid edge, or no boundary found.
- F4 arm disagreement: |a_g - a_b| above da_max_m or |R0_g - R0_b| above dr0_max_m, or only one found.
- F5 control agreement: F1, F2 or F3 in arm_g, in set_b and in the control.
- A shot is flagged for a flag when at least 0.2 of its frames carry it.

## Boundaries per set

| set | frames | found | type_diverted | type_limited | type_unresolved | type_none | limited_cc | fail_axis_on_rim | fail_open_at_axis | two_peaks | second_gap_median | xpt_switch | a_median_m | r0_median_m | kappa_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arm_g | 69075 | 1.000 | 0.954 | 0.045 | 0.001 | 0.000 | 0.045 | 0.000 | 0.000 | 0.000 | n/a | 0.078 | 0.598 | 0.848 | 1.807 |
| set_b | 69075 | 1.000 | 0.961 | 0.038 | 0.001 | 0.000 | 0.038 | 0.000 | 0.000 | 0.000 | 0.000 | 0.068 | 0.619 | 0.853 | 1.753 |
| control | 69075 | 1.000 | 0.970 | 0.030 | 0.001 | 0.000 | 0.030 | 0.000 | 0.000 | 0.000 | n/a | 0.069 | 0.600 | 0.866 | 1.850 |

### Flags v2

Frame share per flag:

| set | F1 | F2 | F3 | any_flag | F4 | F5 |
| --- | --- | --- | --- | --- | --- | --- |
| arm_g | 0.004 | 0.078 | 0.045 | 0.127 | n/a | n/a |
| set_b | 0.008 | 0.068 | 0.038 | 0.113 | n/a | n/a |
| control | 0.002 | 0.069 | 0.030 | 0.100 | n/a | n/a |
| arm pair | n/a | n/a | n/a | n/a | 0.026 | 0.046 |

Shots flagged per flag:

| flag | shots |
| --- | --- |
| arm_g_F1 | 3 |
| arm_g_F2 | 143 |
| arm_g_F3 | 42 |
| arm_g_any_flag | 274 |
| arm_g_fail | 0 |
| set_b_F1 | 13 |
| set_b_F2 | 100 |
| set_b_F3 | 37 |
| set_b_any_flag | 223 |
| set_b_fail | 0 |
| control_F1 | 0 |
| control_F2 | 69 |
| control_F3 | 21 |
| control_any_flag | 126 |
| control_fail | 0 |
| F4 | 19 |
| F5 | 39 |

### Flags v1

Frame share per flag:

| set | F1 | F2 | F3 | any_flag | F4 | F5 |
| --- | --- | --- | --- | --- | --- | --- |
| arm_g | 0.004 | 0.921 | 0.045 | 0.935 | n/a | n/a |
| set_b | 0.008 | 0.953 | 0.038 | 0.961 | n/a | n/a |
| control | 0.002 | 0.955 | 0.030 | 0.961 | n/a | n/a |
| arm pair | n/a | n/a | n/a | n/a | 0.026 | 0.928 |

Shots flagged per flag:

| flag | shots |
| --- | --- |
| arm_g_F1 | 3 |
| arm_g_F2 | 1162 |
| arm_g_F3 | 42 |
| arm_g_any_flag | 1167 |
| arm_g_fail | 0 |
| set_b_F1 | 13 |
| set_b_F2 | 1163 |
| set_b_F3 | 37 |
| set_b_any_flag | 1170 |
| set_b_fail | 0 |
| control_F1 | 0 |
| control_F2 | 1159 |
| control_F3 | 21 |
| control_any_flag | 1161 |
| control_fail | 0 |
| F4 | 19 |
| F5 | 1154 |

## dpsi_sep by set and phase

|dpsi_sep| = |psi_X2 - psi_X1| / |psi_axis - psi_boundary| of the two saddles inside the envelope closest in flux to the boundary; with_value is the share of frames with two such saddles.

| set | phase | frames | with_value | p5 | p25 | p50 | p75 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arm_g | all | 69075 | 0.9747 | 0.0004 | 0.0019 | 0.0035 | 0.0060 | 0.0222 |
| control | all | 69075 | 0.9838 | 0.0002 | 0.0009 | 0.0019 | 0.0030 | 0.0144 |
| set_b | all | 69075 | 0.9800 | 0.0002 | 0.0011 | 0.0021 | 0.0035 | 0.0130 |
| arm_g | falling | 8189 | 0.9828 | 0.0006 | 0.0026 | 0.0047 | 0.0093 | 0.0366 |
| arm_g | flat-top | 51915 | 0.9760 | 0.0004 | 0.0017 | 0.0032 | 0.0056 | 0.0214 |
| arm_g | rising | 8971 | 0.9598 | 0.0008 | 0.0024 | 0.0043 | 0.0062 | 0.0121 |
| control | falling | 8189 | 0.9996 | 0.0008 | 0.0024 | 0.0033 | 0.0064 | 0.0185 |
| control | flat-top | 51915 | 0.9809 | 0.0002 | 0.0008 | 0.0017 | 0.0027 | 0.0145 |
| control | rising | 8971 | 0.9861 | 0.0002 | 0.0007 | 0.0016 | 0.0032 | 0.0066 |
| set_b | falling | 8189 | 0.9801 | 0.0005 | 0.0017 | 0.0026 | 0.0040 | 0.0161 |
| set_b | flat-top | 51915 | 0.9800 | 0.0002 | 0.0010 | 0.0019 | 0.0031 | 0.0130 |
| set_b | rising | 8971 | 0.9796 | 0.0005 | 0.0016 | 0.0031 | 0.0046 | 0.0106 |

## Flags v2 by phase (share of frames)

| set | phase | frames | F1 | F2 | F3 | F4 | F5 | fail |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arm_g | falling | 8189 | 0.029 | 0.084 | 0.007 | 0.100 | 0.017 | 0.000 |
| arm_g | flat-top | 51915 | 0.001 | 0.080 | 0.014 | 0.004 | 0.024 | 0.000 |
| arm_g | rising | 8971 | 0.002 | 0.062 | 0.258 | 0.086 | 0.202 | 0.000 |
| set_b | falling | 8189 | 0.063 | 0.069 | 0.008 | 0.100 | 0.017 | 0.000 |
| set_b | flat-top | 51915 | 0.000 | 0.069 | 0.013 | 0.004 | 0.024 | 0.000 |
| set_b | rising | 8971 | 0.001 | 0.063 | 0.207 | 0.086 | 0.202 | 0.000 |
| control | falling | 8189 | 0.005 | 0.050 | 0.002 | 0.100 | 0.017 | 0.000 |
| control | flat-top | 51915 | 0.001 | 0.069 | 0.009 | 0.004 | 0.024 | 0.000 |
| control | rising | 8971 | 0.003 | 0.084 | 0.176 | 0.086 | 0.202 | 0.000 |

## Flagged against unflagged shots, flags v2 (medians of shot-level inputs)

| flag | n_flagged | ip_mean_ka flagged / not | ip_max_ka flagged / not | ip_rate_rel_max flagged / not | sol_swing_ka flagged / not | clip_rate_arm_g flagged / not | clip_rate_set_b flagged / not |
| --- | --- | --- | --- | --- | --- | --- | --- |
| arm_g_F1 | 3 | 513.580 / 640.959 | 843.992 / 755.774 | 12.010 / 9.687 | 49.092 / 49.730 | 0.149 / 0.062 | 0.448 / 0.062 |
| arm_g_F2 | 143 | 712.832 / 630.581 | 770.419 / 745.764 | 11.853 / 9.271 | 48.555 / 50.067 | 0.068 / 0.062 | 0.068 / 0.062 |
| arm_g_F3 | 42 | 619.529 / 641.243 | 666.627 / 756.099 | 8.672 / 9.743 | 43.195 / 49.940 | 0.066 / 0.062 | 0.066 / 0.062 |
| arm_g_any_flag | 274 | 699.923 / 626.171 | 768.149 / 739.749 | 11.335 / 9.165 | 48.789 / 50.145 | 0.069 / 0.060 | 0.070 / 0.060 |
| arm_g_fail | 0 | n/a / 640.767 | n/a / 755.798 | n/a / 9.697 | n/a / 49.728 | n/a / 0.062 | n/a / 0.062 |
| set_b_F1 | 13 | 543.683 / 641.915 | 759.111 / 755.503 | 6.653 / 9.792 | 51.654 / 49.609 | 0.162 / 0.062 | 0.277 / 0.062 |
| set_b_F2 | 100 | 720.783 / 634.334 | 773.889 / 748.671 | 12.627 / 9.398 | 47.041 / 50.041 | 0.070 / 0.062 | 0.073 / 0.062 |
| set_b_F3 | 37 | 627.644 / 641.075 | 669.000 / 755.915 | 8.324 / 9.745 | 43.250 / 49.845 | 0.071 / 0.062 | 0.071 / 0.062 |
| set_b_any_flag | 223 | 658.533 / 639.637 | 766.701 / 744.323 | 11.140 / 9.388 | 49.114 / 49.842 | 0.082 / 0.059 | 0.083 / 0.060 |
| set_b_fail | 0 | n/a / 640.767 | n/a / 755.798 | n/a / 9.697 | n/a / 49.728 | n/a / 0.062 | n/a / 0.062 |
| control_F1 | 0 | n/a / 640.767 | n/a / 755.798 | n/a / 9.697 | n/a / 49.728 | n/a / 0.062 | n/a / 0.062 |
| control_F2 | 69 | 696.039 / 639.637 | 757.878 / 755.564 | 13.366 / 9.537 | 41.925 / 50.117 | 0.069 / 0.062 | 0.069 / 0.062 |
| control_F3 | 21 | 651.905 / 640.613 | 778.095 / 755.583 | 6.868 / 9.767 | 46.338 / 49.730 | 0.100 / 0.062 | 0.100 / 0.062 |
| control_any_flag | 126 | 679.160 / 636.836 | 759.073 / 754.190 | 12.038 / 9.580 | 46.075 / 50.155 | 0.071 / 0.061 | 0.071 / 0.062 |
| control_fail | 0 | n/a / 640.767 | n/a / 755.798 | n/a / 9.697 | n/a / 49.728 | n/a / 0.062 | n/a / 0.062 |
| F4 | 19 | 685.336 / 640.115 | 766.227 / 755.503 | 8.495 / 9.709 | 50.736 / 49.718 | 0.109 / 0.062 | 0.109 / 0.062 |
| F5 | 39 | 676.439 / 639.637 | 761.782 / 754.270 | 11.281 / 9.676 | 46.108 / 49.764 | 0.103 / 0.062 | 0.103 / 0.062 |

## Known shots

24 entries, 24 matched to a shot of the submission, 24 analysed in this run (the others lie outside --shots). Ordered by the board's d_lcfs; groups cluster 0.3-0.45 m (0.3-0.45 m), cluster 0.6-0.86 m (0.6-0.86 m).

| rank | key | shot | d_lcfs | group | arm_g diverted/limited/none | set_b diverted/limited/none | control diverted/limited/none | a_g median | R0_g median | |dpsi_sep|_g median | F3_g | F4 | F5 | files |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | shot_1170 | f999ce3a53 | 0.197 | other | 1.00/0.00/0.00 | 1.00/0.00/0.00 | 1.00/0.00/0.00 | 0.582 | 0.847 | 0.002 | 0.000 | 0.162 | 0.000 | [frames](figures/known/k01_shot_1170_f999ce3a53_frames.md), [maps](figures/known/k01_shot_1170_f999ce3a53_maps.png), [traces](figures/known/k01_shot_1170_f999ce3a53_traces.png) |
| 2 | shot_0788 | a9f4999e0f | 0.219 | other | 0.99/0.01/0.00 | 0.99/0.01/0.00 | 0.99/0.01/0.00 | 0.585 | 0.847 | 0.003 | 0.014 | 0.145 | 0.014 | [frames](figures/known/k02_shot_0788_a9f4999e0f_frames.md), [maps](figures/known/k02_shot_0788_a9f4999e0f_maps.png), [traces](figures/known/k02_shot_0788_a9f4999e0f_traces.png) |
| 3 | shot_0751 | a1ecfca9b3 | 0.350 | cluster 0.3-0.45 m | 0.96/0.04/0.00 | 0.91/0.09/0.00 | 0.97/0.03/0.00 | 0.598 | 0.844 | 0.003 | 0.039 | 0.065 | 0.052 | [frames](figures/known/k03_shot_0751_a1ecfca9b3_frames.md), [maps](figures/known/k03_shot_0751_a1ecfca9b3_maps.png), [traces](figures/known/k03_shot_0751_a1ecfca9b3_traces.png) |
| 4 | shot_0784 | a944663b2b | 0.370 | cluster 0.3-0.45 m | 0.97/0.00/0.00 | 1.00/0.00/0.00 | 1.00/0.00/0.00 | 0.600 | 0.823 | 0.024 | 0.000 | 0.000 | 0.281 | [frames](figures/known/k04_shot_0784_a944663b2b_frames.md), [maps](figures/known/k04_shot_0784_a944663b2b_maps.png), [traces](figures/known/k04_shot_0784_a944663b2b_traces.png) |
| 5 | shot_0157 | 1f910176f4 | 0.376 | cluster 0.3-0.45 m | 1.00/0.00/0.00 | 1.00/0.00/0.00 | 1.00/0.00/0.00 | 0.644 | 0.825 | 0.019 | 0.000 | 0.000 | 0.000 | [frames](figures/known/k05_shot_0157_1f910176f4_frames.md), [maps](figures/known/k05_shot_0157_1f910176f4_maps.png), [traces](figures/known/k05_shot_0157_1f910176f4_traces.png) |
| 6 | shot_0497 | 6751b420f2 | 0.391 | cluster 0.3-0.45 m | 0.91/0.09/0.00 | 0.96/0.04/0.00 | 0.96/0.04/0.00 | 0.642 | 0.828 | 0.026 | 0.087 | 0.000 | 0.043 | [frames](figures/known/k06_shot_0497_6751b420f2_frames.md), [maps](figures/known/k06_shot_0497_6751b420f2_maps.png), [traces](figures/known/k06_shot_0497_6751b420f2_traces.png) |
| 7 | shot_1055 | e00d9b8ce4 | 0.421 | cluster 0.3-0.45 m | 1.00/0.00/0.00 | 1.00/0.00/0.00 | 1.00/0.00/0.00 | 0.554 | 0.859 | 0.054 | 0.000 | 0.000 | 0.000 | [frames](figures/known/k07_shot_1055_e00d9b8ce4_frames.md), [maps](figures/known/k07_shot_1055_e00d9b8ce4_maps.png), [traces](figures/known/k07_shot_1055_e00d9b8ce4_traces.png) |
| 8 | shot_0219 | 2b59f6137b | 0.606 | cluster 0.6-0.86 m | 1.00/0.00/0.00 | 1.00/0.00/0.00 | 1.00/0.00/0.00 | 0.583 | 0.838 | 0.020 | 0.000 | 0.000 | 0.000 | [frames](figures/known/k08_shot_0219_2b59f6137b_frames.md), [maps](figures/known/k08_shot_0219_2b59f6137b_maps.png), [traces](figures/known/k08_shot_0219_2b59f6137b_traces.png) |
| 9 | shot_0534 | 6fae0bc163 | 0.623 | cluster 0.6-0.86 m | 0.86/0.14/0.00 | 0.86/0.14/0.00 | 0.86/0.14/0.00 | 0.573 | 0.849 | 0.016 | 0.143 | 0.071 | 0.214 | [frames](figures/known/k09_shot_0534_6fae0bc163_frames.md), [maps](figures/known/k09_shot_0534_6fae0bc163_maps.png), [traces](figures/known/k09_shot_0534_6fae0bc163_traces.png) |
| 10 | shot_0654 | 8a4c7ddbb8 | 0.633 | cluster 0.6-0.86 m | 0.93/0.07/0.00 | 0.93/0.07/0.00 | 0.93/0.07/0.00 | 0.569 | 0.881 | 0.018 | 0.067 | 0.000 | 0.067 | [frames](figures/known/k10_shot_0654_8a4c7ddbb8_frames.md), [maps](figures/known/k10_shot_0654_8a4c7ddbb8_maps.png), [traces](figures/known/k10_shot_0654_8a4c7ddbb8_traces.png) |
| 11 | shot_0117 | 182f1b368d | 0.637 | cluster 0.6-0.86 m | 0.98/0.02/0.00 | 0.98/0.02/0.00 | 0.98/0.02/0.00 | 0.580 | 0.868 | 0.013 | 0.024 | 0.000 | 0.024 | [frames](figures/known/k11_shot_0117_182f1b368d_frames.md), [maps](figures/known/k11_shot_0117_182f1b368d_maps.png), [traces](figures/known/k11_shot_0117_182f1b368d_traces.png) |
| 12 | shot_0394 | 4f7d4c4095 | 0.639 | cluster 0.6-0.86 m | 0.92/0.08/0.00 | 0.92/0.08/0.00 | 0.92/0.08/0.00 | 0.587 | 0.862 | 0.024 | 0.081 | 0.000 | 0.108 | [frames](figures/known/k12_shot_0394_4f7d4c4095_frames.md), [maps](figures/known/k12_shot_0394_4f7d4c4095_maps.png), [traces](figures/known/k12_shot_0394_4f7d4c4095_traces.png) |
| 13 | shot_0510 | 6a783ae330 | 0.641 | cluster 0.6-0.86 m | 0.91/0.09/0.00 | 0.91/0.09/0.00 | 0.91/0.09/0.00 | 0.590 | 0.861 | 0.012 | 0.087 | 0.000 | 0.087 | [frames](figures/known/k13_shot_0510_6a783ae330_frames.md), [maps](figures/known/k13_shot_0510_6a783ae330_maps.png), [traces](figures/known/k13_shot_0510_6a783ae330_traces.png) |
| 14 | shot_1130 | eec0d5e8fe | 0.657 | cluster 0.6-0.86 m | 0.94/0.06/0.00 | 0.86/0.14/0.00 | 0.96/0.04/0.00 | 0.585 | 0.862 | 0.011 | 0.060 | 0.000 | 0.040 | [frames](figures/known/k14_shot_1130_eec0d5e8fe_frames.md), [maps](figures/known/k14_shot_1130_eec0d5e8fe_maps.png), [traces](figures/known/k14_shot_1130_eec0d5e8fe_traces.png) |
| 15 | shot_0814 | aeda9210f7 | 0.663 | cluster 0.6-0.86 m | 0.94/0.06/0.00 | 0.97/0.03/0.00 | 0.97/0.03/0.00 | 0.593 | 0.859 | 0.009 | 0.061 | 0.000 | 0.030 | [frames](figures/known/k15_shot_0814_aeda9210f7_frames.md), [maps](figures/known/k15_shot_0814_aeda9210f7_maps.png), [traces](figures/known/k15_shot_0814_aeda9210f7_traces.png) |
| 16 | shot_0354 | 484e426647 | 0.667 | cluster 0.6-0.86 m | 1.00/0.00/0.00 | 1.00/0.00/0.00 | 1.00/0.00/0.00 | 0.589 | 0.850 | 0.012 | 0.000 | 0.000 | 0.045 | [frames](figures/known/k16_shot_0354_484e426647_frames.md), [maps](figures/known/k16_shot_0354_484e426647_maps.png), [traces](figures/known/k16_shot_0354_484e426647_traces.png) |
| 17 | shot_0304 | 3ee0cf22a1 | 0.681 | cluster 0.6-0.86 m | 0.95/0.05/0.00 | 0.98/0.02/0.00 | 0.98/0.02/0.00 | 0.593 | 0.836 | 0.009 | 0.049 | 0.000 | 0.024 | [frames](figures/known/k17_shot_0304_3ee0cf22a1_frames.md), [maps](figures/known/k17_shot_0304_3ee0cf22a1_maps.png), [traces](figures/known/k17_shot_0304_3ee0cf22a1_traces.png) |
| 18 | shot_0963 | ce013f07e5 | 0.683 | cluster 0.6-0.86 m | 0.90/0.10/0.00 | 0.90/0.10/0.00 | 0.90/0.10/0.00 | 0.589 | 0.862 | 0.016 | 0.100 | 0.000 | 0.100 | [frames](figures/known/k18_shot_0963_ce013f07e5_frames.md), [maps](figures/known/k18_shot_0963_ce013f07e5_maps.png), [traces](figures/known/k18_shot_0963_ce013f07e5_traces.png) |
| 19 | shot_0379 | 4d1732d0b5 | 0.702 | cluster 0.6-0.86 m | 0.96/0.04/0.00 | 0.96/0.04/0.00 | 0.96/0.04/0.00 | 0.595 | 0.854 | 0.011 | 0.040 | 0.000 | 0.040 | [frames](figures/known/k19_shot_0379_4d1732d0b5_frames.md), [maps](figures/known/k19_shot_0379_4d1732d0b5_maps.png), [traces](figures/known/k19_shot_0379_4d1732d0b5_traces.png) |
| 20 | shot_0302 | 3e892292d7 | 0.702 | cluster 0.6-0.86 m | 1.00/0.00/0.00 | 1.00/0.00/0.00 | 1.00/0.00/0.00 | 0.593 | 0.855 | 0.011 | 0.000 | 0.000 | 0.000 | [frames](figures/known/k20_shot_0302_3e892292d7_frames.md), [maps](figures/known/k20_shot_0302_3e892292d7_maps.png), [traces](figures/known/k20_shot_0302_3e892292d7_traces.png) |
| 21 | shot_0833 | b33cd6c421 | 0.718 | cluster 0.6-0.86 m | 0.89/0.11/0.00 | 0.89/0.11/0.00 | 0.89/0.11/0.00 | 0.606 | 0.820 | 0.005 | 0.111 | 0.000 | 0.111 | [frames](figures/known/k21_shot_0833_b33cd6c421_frames.md), [maps](figures/known/k21_shot_0833_b33cd6c421_maps.png), [traces](figures/known/k21_shot_0833_b33cd6c421_traces.png) |
| 22 | shot_0332 | 4383644ab6 | 0.732 | cluster 0.6-0.86 m | 0.89/0.11/0.00 | 0.89/0.11/0.00 | 0.89/0.11/0.00 | 0.602 | 0.829 | 0.009 | 0.111 | 0.000 | 0.111 | [frames](figures/known/k22_shot_0332_4383644ab6_frames.md), [maps](figures/known/k22_shot_0332_4383644ab6_maps.png), [traces](figures/known/k22_shot_0332_4383644ab6_traces.png) |
| 23 | shot_1070 | e420b64254 | 0.787 | cluster 0.6-0.86 m | 0.99/0.01/0.00 | 1.00/0.00/0.00 | 1.00/0.00/0.00 | 0.576 | 0.860 | 0.015 | 0.013 | 0.184 | 0.000 | [frames](figures/known/k23_shot_1070_e420b64254_frames.md), [maps](figures/known/k23_shot_1070_e420b64254_maps.png), [traces](figures/known/k23_shot_1070_e420b64254_traces.png) |
| 24 | shot_0678 | 8eeaa36612 | 0.857 | cluster 0.6-0.86 m | 0.91/0.06/0.00 | 0.87/0.11/0.00 | 0.94/0.06/0.00 | 0.578 | 0.832 | 0.026 | 0.061 | 0.171 | 0.061 | [frames](figures/known/k24_shot_0678_8eeaa36612_frames.md), [maps](figures/known/k24_shot_0678_8eeaa36612_maps.png), [traces](figures/known/k24_shot_0678_8eeaa36612_traces.png) |

## Overlap with the known shots, flags v2 and v1 (reading)

Computed after the flags were frozen; no threshold was tuned to these shots.

| flag | known_in_table | known_caught | known_caught_share | other_raised | known_not_in_table |
| --- | --- | --- | --- | --- | --- |
| shot_arm_g_F1 | 24 | 0 | 0.000 | 3 | 0 |
| shot_arm_g_F2 | 24 | 2 | 0.083 | 141 | 0 |
| shot_arm_g_F3 | 24 | 0 | 0.000 | 42 | 0 |
| shot_arm_g_any_flag | 24 | 5 | 0.208 | 269 | 0 |
| shot_arm_g_fail | 24 | 0 | 0.000 | 0 | 0 |
| shot_set_b_F1 | 24 | 0 | 0.000 | 13 | 0 |
| shot_set_b_F2 | 24 | 1 | 0.042 | 99 | 0 |
| shot_set_b_F3 | 24 | 0 | 0.000 | 37 | 0 |
| shot_set_b_any_flag | 24 | 5 | 0.208 | 218 | 0 |
| shot_set_b_fail | 24 | 0 | 0.000 | 0 | 0 |
| shot_control_F1 | 24 | 0 | 0.000 | 0 | 0 |
| shot_control_F2 | 24 | 1 | 0.042 | 68 | 0 |
| shot_control_F3 | 24 | 0 | 0.000 | 21 | 0 |
| shot_control_any_flag | 24 | 2 | 0.083 | 124 | 0 |
| shot_control_fail | 24 | 0 | 0.000 | 0 | 0 |
| shot_F4 | 24 | 0 | 0.000 | 19 | 0 |
| shot_F5 | 24 | 2 | 0.083 | 37 | 0 |
| shot_arm_g_F1_v1 | 24 | 0 | 0.000 | 3 | 0 |
| shot_arm_g_F2_v1 | 24 | 22 | 0.917 | 1140 | 0 |
| shot_arm_g_F3_v1 | 24 | 0 | 0.000 | 42 | 0 |
| shot_arm_g_any_flag_v1 | 24 | 22 | 0.917 | 1145 | 0 |
| shot_arm_g_fail_v1 | 24 | 0 | 0.000 | 0 | 0 |
| shot_set_b_F1_v1 | 24 | 0 | 0.000 | 13 | 0 |
| shot_set_b_F2_v1 | 24 | 22 | 0.917 | 1141 | 0 |
| shot_set_b_F3_v1 | 24 | 0 | 0.000 | 37 | 0 |
| shot_set_b_any_flag_v1 | 24 | 22 | 0.917 | 1148 | 0 |
| shot_set_b_fail_v1 | 24 | 0 | 0.000 | 0 | 0 |
| shot_control_F1_v1 | 24 | 0 | 0.000 | 0 | 0 |
| shot_control_F2_v1 | 24 | 22 | 0.917 | 1137 | 0 |
| shot_control_F3_v1 | 24 | 0 | 0.000 | 21 | 0 |
| shot_control_any_flag_v1 | 24 | 22 | 0.917 | 1139 | 0 |
| shot_control_fail_v1 | 24 | 0 | 0.000 | 0 | 0 |
| shot_F4_v1 | 24 | 0 | 0.000 | 19 | 0 |
| shot_F5_v1 | 24 | 21 | 0.875 | 1133 | 0 |

## Figures (flag-based selection)

- 6fae0bc163 (cluster 0.6-0.86 m): [6fae0bc163_maps.png](figures/6fae0bc163_maps.png), [6fae0bc163_traces.png](figures/6fae0bc163_traces.png)
- ce013f07e5 (cluster 0.6-0.86 m): [ce013f07e5_maps.png](figures/ce013f07e5_maps.png), [ce013f07e5_traces.png](figures/ce013f07e5_traces.png)
- 4d1732d0b5 (cluster 0.6-0.86 m): [4d1732d0b5_maps.png](figures/4d1732d0b5_maps.png), [4d1732d0b5_traces.png](figures/4d1732d0b5_traces.png)
- 92e0049416 (flagged): [92e0049416_maps.png](figures/92e0049416_maps.png), [92e0049416_traces.png](figures/92e0049416_traces.png)
- 38ea522170 (flagged): [38ea522170_maps.png](figures/38ea522170_maps.png), [38ea522170_traces.png](figures/38ea522170_traces.png)
- c05a2bc697 (flagged): [c05a2bc697_maps.png](figures/c05a2bc697_maps.png), [c05a2bc697_traces.png](figures/c05a2bc697_traces.png)
- a944663b2b (cluster 0.3-0.45 m): [a944663b2b_maps.png](figures/a944663b2b_maps.png), [a944663b2b_traces.png](figures/a944663b2b_traces.png)
- 1b066e3e47 (flagged): [1b066e3e47_maps.png](figures/1b066e3e47_maps.png), [1b066e3e47_traces.png](figures/1b066e3e47_traces.png)
- 2b1233aa20 (flagged): [2b1233aa20_maps.png](figures/2b1233aa20_maps.png), [2b1233aa20_traces.png](figures/2b1233aa20_traces.png)
- 44a170ef3b (flagged): [44a170ef3b_maps.png](figures/44a170ef3b_maps.png), [44a170ef3b_traces.png](figures/44a170ef3b_traces.png)
- 0ad8c81cc2 (unflagged): [0ad8c81cc2_maps.png](figures/0ad8c81cc2_maps.png), [0ad8c81cc2_traces.png](figures/0ad8c81cc2_traces.png)
- 62481e61fd (unflagged): [62481e61fd_maps.png](figures/62481e61fd_maps.png), [62481e61fd_traces.png](figures/62481e61fd_traces.png)
- ac9a3493e3 (unflagged): [ac9a3493e3_maps.png](figures/ac9a3493e3_maps.png), [ac9a3493e3_traces.png](figures/ac9a3493e3_traces.png)
- f981976184 (unflagged): [f981976184_maps.png](figures/f981976184_maps.png), [f981976184_traces.png](figures/f981976184_traces.png)

## Readings

Left for the review of the full run.
