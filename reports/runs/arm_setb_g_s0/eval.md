# Evaluation of run arm_setb_g_s0

Validation list `validation` (870 shots, 48186 frames at stride 4); chain: checkpoints arm_setb_g_s0, mirror averaging on, q95 from the map, betaN from the DIII-D raw-input ridge (fitted on `train` at stride 4). Corner: q_cyl < 1.780 and ts_beta_p < 0.1088, 2356 frames.

MAST floor (F100, mean_current_v0 on Codabench): S 0.4833, R2 psi 0.7447, D_LCFS 0.263 m.

## Checks

| Id | Check | Status |
| --- | --- | --- |
| Z4 | Warm start | review: trained at or above the warm start (all 0.9795 against 0.9243, no near-repeat 0.9697 against 0.9069) |
| Z7 | Slices present | pass |
| Z8 | Clip rates | review: largest share 0.0733 |
| Z11 | Mirror averaging | review: on 0.9795, off 0.9792 |
| Z12 | MAST sign | pass (asserted before the MAST maps) |

## Score terms by slice

Identity: `0.55 max(0, R2 psi) + 0.15 max(0, R2 scalars) + 0.10 (1 - D_LCFS) + 0.20 consistency`, beside the scorer's S.

| Slice | Shots | S | Identity | R2 psi | R2 psi per shot | R2 q95 | R2 betaN | 1 - D_LCFS | Consistency | R_axis | Z_axis | kappa | tri_top | tri_bot | volume | li |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 870 | 0.9795 | 0.9795 | 0.9996 | 0.9992 | 0.9484 | 0.9338 | 0.9852 | 0.9504 | 0.9441 | 0.9398 | 0.9156 | 0.9845 | 0.9803 | 0.9691 | 0.9191 |
| no_near_repeat | 286 | 0.9697 | 0.9697 | 0.9993 | 0.9983 | 0.9311 | 0.9294 | 0.9811 | 0.9124 | 0.8918 | 0.8581 | 0.8679 | 0.9830 | 0.9729 | 0.9585 | 0.8544 |
| ip_negative | 74 | 0.9610 | 0.9610 | 0.9996 | 0.9996 | 0.9283 | 0.8258 | 0.9821 | 0.9074 | 0.8520 | 0.9249 | 0.8651 | 0.9574 | 0.9525 | 0.9519 | 0.8476 |
| corner | 151 | 0.9704 | 0.9704 | 0.9996 | 0.9995 | 0.9720 | 0.8038 | 0.9884 | 0.9433 | 0.9286 | 0.9880 | 0.8581 | 0.9839 | 0.9797 | 0.9393 | 0.9256 |

MAST floor (F100, mean_current_v0 on Codabench): S 0.4833, R2 psi 0.7447, D_LCFS 0.263 m.

## Score terms by phase

| Slice | Shots | S | Identity | R2 psi | R2 psi per shot | R2 q95 | R2 betaN | 1 - D_LCFS | Consistency | R_axis | Z_axis | kappa | tri_top | tri_bot | volume | li |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase_rising | 858 | 0.9736 | 0.9736 | 0.9992 | 0.9984 | 0.9355 | 0.9474 | 0.9837 | 0.9223 | 0.9178 | 0.9014 | 0.8769 | 0.9821 | 0.9768 | 0.9327 | 0.8681 |
| phase_flat_top | 862 | 0.9851 | 0.9851 | 0.9996 | 0.9995 | 0.9860 | 0.9297 | 0.9865 | 0.9647 | 0.9545 | 0.9829 | 0.9246 | 0.9865 | 0.9826 | 0.9768 | 0.9448 |
| phase_late | 427 | 0.9160 | 0.9160 | 0.9980 | 0.9970 | 0.6006 | 0.7360 | 0.9728 | 0.8483 | 0.7672 | 0.6264 | 0.8572 | 0.9566 | 0.9574 | 0.9567 | 0.8162 |
| no_near_repeat_phase_rising | 281 | 0.9679 | 0.9679 | 0.9990 | 0.9982 | 0.9310 | 0.9540 | 0.9802 | 0.8955 | 0.9015 | 0.8268 | 0.8385 | 0.9828 | 0.9698 | 0.9165 | 0.8323 |
| no_near_repeat_phase_flat_top | 279 | 0.9771 | 0.9771 | 0.9994 | 0.9992 | 0.9774 | 0.9114 | 0.9841 | 0.9370 | 0.9063 | 0.9763 | 0.8781 | 0.9851 | 0.9758 | 0.9648 | 0.8726 |
| no_near_repeat_phase_late | 151 | 0.8998 | 0.8998 | 0.9956 | 0.9950 | 0.5380 | 0.8642 | 0.9702 | 0.7501 | 0.6113 | 0.2189 | 0.7930 | 0.9517 | 0.9510 | 0.9629 | 0.7618 |

MAST floor (F100, mean_current_v0 on Codabench): S 0.4833, R2 psi 0.7447, D_LCFS 0.263 m.

## Worst 20 validation shots by R2 psi

| Shot | R2 psi | D_LCFS | Boundary failures |
| --- | --- | --- | --- |
| d42a0ed0c0 | 0.8267 | 0.5210 | 0 of 1 |
| 269e33625f | 0.9152 | 0.0564 | 1 of 22 |
| b6f9f193bf | 0.9889 | 0.0481 | 0 of 11 |
| b2df6a7008 | 0.9951 | 0.0330 | 0 of 17 |
| ec216f0d01 | 0.9951 | 0.0314 | 0 of 12 |
| 533cba65c5 | 0.9952 | 0.0298 | 0 of 53 |
| 316f541fd6 | 0.9953 | 0.0300 | 0 of 1 |
| a2ae4f3c2b | 0.9957 | 0.0287 | 0 of 71 |
| 1b54ce9916 | 0.9957 | 0.0223 | 0 of 61 |
| 2427c80a5d | 0.9959 | 0.0362 | 0 of 65 |
| ced938a97c | 0.9964 | 0.0290 | 0 of 37 |
| 46c4bc5f9c | 0.9966 | 0.0157 | 0 of 65 |
| 8496e7cfe6 | 0.9973 | 0.0272 | 0 of 48 |
| 2b9b50b9b4 | 0.9973 | 0.0166 | 0 of 75 |
| 4274201181 | 0.9974 | 0.0191 | 0 of 19 |
| f4889b292e | 0.9974 | 0.0266 | 0 of 62 |
| e365322b54 | 0.9975 | 0.0283 | 0 of 76 |
| 3b2025b28b | 0.9976 | 0.0136 | 0 of 53 |
| 0b92fb631b | 0.9976 | 0.0372 | 0 of 20 |
| 4c146d6cd7 | 0.9977 | 0.0152 | 0 of 28 |

## Overfit panel

| Epoch | Train loss | Validation loss | No-near-repeat loss |
| --- | --- | --- | --- |
| 187 (best epoch) | 0.09606 | 0.09950 | 0.17230 |
| 199 (stopped epoch) | 0.09630 | 0.09952 | 0.17235 |

Trained: S all 0.9795, no near-repeat 0.9697, gap 0.0098.

Warm start (ridge): S all 0.9243, no near-repeat 0.9069, gap 0.0174.

Seed spread of S over 2 run(s): mean 0.9796, standard deviation 0.0001, range 0.9795 to 0.9797.
- arm_setb_g_s0 (seed 0): S 0.9795, no near-repeat 0.9697
- arm_setb_g_s1 (seed 1): S 0.9797, no near-repeat 0.9701

## Warm start beside the trained model

Prediction chain (warm-start checkpoint `warm.pt` against the best checkpoint; q95 from the map, betaN from the DIII-D head, mirror averaging on):

| Slice | Warm start | Trained | Difference |
| --- | --- | --- | --- |
| all | 0.9243 | 0.9795 | 0.0553 |
| no_near_repeat | 0.9069 | 0.9697 | 0.0628 |
| ip_negative | 0.8766 | 0.9610 | 0.0844 |
| corner | 0.8883 | 0.9704 | 0.0822 |

The run record holds no training-time warm-start or best score (trained with --no-score).

Chain score, mirror averaging on: S 0.9795 (no near-repeat 0.9697); off: S 0.9792 (no near-repeat 0.9695).

## Clip rates

Share of scaled values beyond 10 per column.

DIII-D validation (191411 frames): `coil_c12` 0.0104, `ip_rate_rel` 0.0061, `resistive_age` 0.0016, `q_cyl` 0.0006, `ts_beta_p` 0.0000, `shafranov_lambda` 0.0000, `shaping` 0.0000, `coil_c21` 0.0000.

DIII-D test (193359 frames): `coil_c12` 0.0162, `ip_rate_rel` 0.0059, `q_cyl` 0.0004, `coil_c11` 0.0001, `resistive_age` 0.0000, `shaping` 0.0000.

MAST test (69075 frames): `ip_rate_rel` 0.0733, `coil_c30` 0.0081, `shaping` 0.0006, `q_cyl` 0.0002, `ts_beta_p` 0.0000.

## Grouped permutation importance (validation, all shots)

| Group | Columns | S | drop S | drop r2_psi | drop r2_qb | drop one_minus_dlcfs | drop consistency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| coil_terms | 8 | 0.7968 | 0.1828 | 0.0101 | 0.0679 | 0.0716 | 0.7993 |
| plasma_content | 3 | 0.9311 | 0.0484 | 0.0023 | 0.0032 | 0.0104 | 0.2282 |
| field_current | 1 | 0.9559 | 0.0236 | 0.0007 | 0.0165 | 0.0043 | 0.1017 |
| shot_position | 3 | 0.9512 | 0.0283 | 0.0008 | 0.0095 | 0.0049 | 0.1298 |

Drop of each consistency value:

| Group | R_axis | Z_axis | kappa | tri_top | tri_bot | volume | li |
| --- | --- | --- | --- | --- | --- | --- | --- |
| coil_terms | 1.2187 | 2.1064 | 1.3036 | 0.4414 | 0.4658 | 0.9921 | 0.9785 |
| plasma_content | 0.9058 | 0.0224 | 0.2781 | 0.0248 | 0.0275 | 0.1594 | 0.1791 |
| field_current | 0.1490 | 0.0330 | 0.1780 | 0.0066 | 0.0086 | 0.0551 | 0.2818 |
| shot_position | 0.0935 | 0.0388 | 0.3511 | 0.0070 | 0.0085 | 0.0390 | 0.3704 |

## MAST sensitivity (label-free)

61 MAST test shots, 904 frames; unpermuted maps: boundary on 1.000 of frames, map RMS 0.07613 Wb/rad; medians R_axis 0.948, Z_axis -0.000, kappa 1.735, R_geo 0.899, a 0.592. No MAST label is read.

| Group | Map change / RMS | Boundary share | Frames with both | median abs change R_axis | median abs change Z_axis | median abs change kappa | median abs change R_geo | median abs change a |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| coil_terms | 0.087 | 1.000 | 904 | 0.0188 | 0.0024 | 0.1018 | 0.0219 | 0.0301 |
| plasma_content | 0.022 | 1.000 | 904 | 0.0087 | 0.0001 | 0.0071 | 0.0086 | 0.0019 |
| field_current | 0.014 | 1.000 | 904 | 0.0019 | 0.0001 | 0.0093 | 0.0017 | 0.0022 |
| shot_position | 0.010 | 1.000 | 904 | 0.0023 | 0.0001 | 0.0114 | 0.0015 | 0.0027 |

MAST floor (F100, mean_current_v0 on Codabench): S 0.4833, R2 psi 0.7447, D_LCFS 0.263 m.
