# Evaluation of run 20261015T033844_e82e8704_s0

Validation list `working_validation` (100 shots, 5551 frames at stride 4); chain: checkpoints 20261015T033844_e82e8704_s0, mirror averaging on, q95 from the map, betaN from the DIII-D raw-input ridge (fitted on `working_train` at stride 4). Corner: q_cyl < 1.789 and ts_beta_p < 0.1698 (relaxed), 629 frames.

MAST floor (F100, mean_current_v0 on Codabench): S 0.4833, R2 psi 0.7447, D_LCFS 0.263 m.

## Checks

| Id | Check | Status |
| --- | --- | --- |
| Z4 | Warm start | review: trained below the warm start (all 0.8588 against 0.8613, no near-repeat 0.8338 against 0.8325) |
| Z7 | Slices present | pass |
| Z8 | Clip rates | review: largest share 0.0733 |
| Z11 | Mirror averaging | review: on 0.8740, off 0.8681 |
| Z12 | MAST sign | pass (asserted before the MAST maps) |

## Score terms by slice

Identity: `0.55 max(0, R2 psi) + 0.15 max(0, R2 scalars) + 0.10 (1 - D_LCFS) + 0.20 consistency`, beside the scorer's S.

| Slice | Shots | S | Identity | R2 psi | R2 psi per shot | R2 q95 | R2 betaN | 1 - D_LCFS | Consistency | R_axis | Z_axis | kappa | tri_top | tri_bot | volume | li |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 100 | 0.8740 | 0.8740 | 0.9971 | 0.9964 | 0.8605 | 0.9408 | 0.9365 | 0.4841 | 0.5859 | 0.6315 | 0.0469 | 0.6463 | 0.6653 | 0.2160 | 0.5971 |
| no_near_repeat | 32 | 0.8563 | 0.8563 | 0.9958 | 0.9952 | 0.8658 | 0.9184 | 0.9342 | 0.4069 | 0.4127 | 0.3823 | -0.5044 | 0.6380 | 0.5848 | 0.2735 | 0.5569 |
| ip_negative | 9 | 0.8274 | 0.8274 | 0.9959 | 0.9956 | 0.9182 | 0.8926 | 0.9432 | 0.2477 | -0.6943 | 0.0104 | -0.1603 | 0.7012 | 0.5900 | -1.0676 | 0.4325 |
| corner | 25 | 0.8123 | 0.8123 | 0.9953 | 0.9946 | 0.5715 | 0.8404 | 0.9384 | 0.3258 | 0.2114 | 0.5094 | -1.6190 | 0.6942 | 0.4519 | 0.2198 | 0.1941 |

MAST floor (F100, mean_current_v0 on Codabench): S 0.4833, R2 psi 0.7447, D_LCFS 0.263 m.

## Score terms by phase

| Slice | Shots | S | Identity | R2 psi | R2 psi per shot | R2 q95 | R2 betaN | 1 - D_LCFS | Consistency | R_axis | Z_axis | kappa | tri_top | tri_bot | volume | li |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase_rising | 100 | 0.8454 | 0.8454 | 0.9968 | 0.9951 | 0.8579 | 0.9433 | 0.9306 | 0.3450 | 0.5710 | -0.1577 | 0.1561 | 0.5045 | 0.6875 | 0.0604 | 0.4355 |
| phase_flat_top | 99 | 0.8687 | 0.8687 | 0.9964 | 0.9957 | 0.7904 | 0.9267 | 0.9395 | 0.4898 | 0.5454 | 0.8018 | -0.0430 | 0.6765 | 0.6564 | 0.2136 | 0.5349 |
| phase_late | 49 | 0.8645 | 0.8645 | 0.9975 | 0.9926 | 0.6075 | 0.9008 | 0.9280 | 0.5500 | 0.5655 | 0.4694 | 0.3858 | 0.7508 | 0.6787 | 0.3216 | 0.6785 |
| no_near_repeat_phase_rising | 32 | 0.8353 | 0.8353 | 0.9959 | 0.9945 | 0.8292 | 0.9359 | 0.9290 | 0.3115 | 0.6524 | -0.4467 | 0.0438 | 0.3790 | 0.6025 | 0.0684 | 0.4348 |
| no_near_repeat_phase_flat_top | 31 | 0.8524 | 0.8524 | 0.9946 | 0.9945 | 0.8141 | 0.8925 | 0.9405 | 0.4166 | 0.2169 | 0.7172 | -0.9394 | 0.6891 | 0.5607 | 0.2567 | 0.4758 |
| no_near_repeat_phase_late | 14 | 0.7772 | 0.7772 | 0.9971 | 0.9937 | -0.3780 | 0.7787 | 0.9262 | 0.5308 | 0.4008 | 0.2344 | 0.4741 | 0.9419 | 0.8535 | 0.5535 | 0.2571 |

MAST floor (F100, mean_current_v0 on Codabench): S 0.4833, R2 psi 0.7447, D_LCFS 0.263 m.

## Worst 20 validation shots by R2 psi

| Shot | R2 psi | D_LCFS | Boundary failures |
| --- | --- | --- | --- |
| b6f9f193bf | 0.9729 | 0.1521 | 0 of 11 |
| adf87cc682 | 0.9851 | 0.0524 | 0 of 61 |
| baa80ca344 | 0.9874 | 0.0663 | 0 of 61 |
| f6a63db6cd | 0.9880 | 0.0693 | 0 of 62 |
| 9918b551a1 | 0.9906 | 0.0864 | 0 of 61 |
| c0ec44362c | 0.9911 | 0.1167 | 0 of 75 |
| 03eb17ba13 | 0.9911 | 0.0741 | 0 of 54 |
| 4469483a2a | 0.9912 | 0.0341 | 0 of 62 |
| 9470d80a8c | 0.9920 | 0.0342 | 0 of 46 |
| 06c410b6f2 | 0.9925 | 0.1030 | 0 of 51 |
| a34195b1cc | 0.9926 | 0.0595 | 0 of 67 |
| 957c0a3776 | 0.9930 | 0.0977 | 0 of 48 |
| 1ec9bfdc1b | 0.9932 | 0.0352 | 0 of 46 |
| 953c07270c | 0.9939 | 0.0387 | 0 of 46 |
| 391e8b2e6c | 0.9940 | 0.0362 | 0 of 70 |
| 5aa5308ccf | 0.9941 | 0.0391 | 0 of 46 |
| 0150a57d75 | 0.9945 | 0.0391 | 0 of 59 |
| 804475f2f1 | 0.9948 | 0.0886 | 0 of 61 |
| 2c37a9ef75 | 0.9948 | 0.0967 | 0 of 43 |
| 597caf2da6 | 0.9949 | 0.0839 | 0 of 41 |

## Overfit panel

| Epoch | Train loss | Validation loss | No-near-repeat loss |
| --- | --- | --- | --- |
| 116 (best epoch) | 0.29208 | 0.27745 | 0.34606 |
| 131 (stopped epoch) | 0.28807 | 0.27562 | 0.34631 |

Trained: S all 0.8588, no near-repeat 0.8338, gap 0.0250.

Warm start (ridge): S all 0.8613, no near-repeat 0.8325, gap 0.0288.

Seed spread of S over 1 run(s): mean 0.8588, standard deviation -, range 0.8588 to 0.8588.
- 20261015T033844_e82e8704_s0 (seed 0): S 0.8588, no near-repeat 0.8338

## Warm start beside the trained model

Training-time scorer (fixed ridge scalars):

| Slice | Warm start | Trained | Difference |
| --- | --- | --- | --- |
| all | 0.8613 | 0.8588 | -0.0025 |
| no_near_repeat | 0.8325 | 0.8338 | 0.0014 |
| ip_negative | 0.8164 | 0.8162 | -0.0003 |
| corner | 0.6859 | 0.7064 | 0.0205 |

Chain score, mirror averaging on: S 0.8740 (no near-repeat 0.8563); off: S 0.8681 (no near-repeat 0.8452).

## Clip rates

Share of scaled values beyond 10 per column.

DIII-D validation (22052 frames): `ip_rate_rel` 0.0054, `q_cyl` 0.0002, `resistive_age` 0.0001.

DIII-D test (193359 frames): `ip_rate_rel` 0.0059, `q_cyl` 0.0004, `resistive_age` 0.0000.

MAST test (69075 frames): `ip_rate_rel` 0.0733, `q_cyl` 0.0002, `ts_beta_p` 0.0000.

## Grouped permutation importance (validation, all shots)

| Group | Columns | S | drop S | drop r2_psi | drop r2_qb | drop one_minus_dlcfs | drop consistency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| coil_pattern | 22 | 0.7972 | 0.0768 | 0.0049 | 0.0143 | 0.0272 | 0.3465 |
| plasma_content | 3 | 0.8429 | 0.0311 | 0.0044 | -0.0034 | 0.0047 | 0.1437 |
| field_current | 1 | 0.8477 | 0.0263 | 0.0007 | 0.0231 | 0.0015 | 0.1116 |
| shot_position | 3 | 0.8569 | 0.0171 | 0.0008 | 0.0000 | 0.0002 | 0.0834 |

Drop of each consistency value:

| Group | R_axis | Z_axis | kappa | tri_top | tri_bot | volume | li |
| --- | --- | --- | --- | --- | --- | --- | --- |
| coil_pattern | 0.2985 | 2.0719 | 0.7800 | 0.2290 | 0.4479 | 0.3385 | 0.5556 |
| plasma_content | 1.8793 | 0.0055 | 0.0194 | 0.0084 | 0.0228 | 0.6045 | 0.1481 |
| field_current | 0.0867 | 0.1867 | 0.1715 | 0.0185 | 0.0144 | 0.0453 | 0.3828 |
| shot_position | 0.0432 | 0.0119 | 0.2283 | 0.0115 | -0.0044 | 0.0135 | 0.4613 |

## MAST sensitivity (label-free)

61 MAST test shots, 904 frames; unpermuted maps: boundary on 1.000 of frames, map RMS 0.08001 Wb/rad; medians R_axis 0.895, Z_axis -0.005, kappa 2.282, R_geo 0.867, a 0.467. No MAST label is read.

| Group | Map change / RMS | Boundary share | Frames with both | median abs change R_axis | median abs change Z_axis | median abs change kappa | median abs change R_geo | median abs change a |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| coil_pattern | 0.046 | 1.000 | 904 | 0.0353 | 0.3595 | 0.1227 | 0.0126 | 0.0231 |
| plasma_content | 0.037 | 1.000 | 904 | 0.0336 | 0.0115 | 0.0569 | 0.0197 | 0.0135 |
| field_current | 0.011 | 1.000 | 904 | 0.0035 | 0.0062 | 0.0125 | 0.0023 | 0.0023 |
| shot_position | 0.014 | 1.000 | 904 | 0.0090 | 0.0203 | 0.0420 | 0.0028 | 0.0075 |

MAST floor (F100, mean_current_v0 on Codabench): S 0.4833, R2 psi 0.7447, D_LCFS 0.263 m.
