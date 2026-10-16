# Evaluation of run rung2_g_s0

Validation list `validation` (870 shots, 48186 frames at stride 4); chain: checkpoints rung2_g_s0, mirror averaging on, q95 from the map, betaN from the DIII-D raw-input ridge (fitted on `train` at stride 4). Corner: q_cyl < 1.780 and ts_beta_p < 0.1088, 2356 frames.

MAST floor (F100, mean_current_v0 on Codabench): S 0.4833, R2 psi 0.7447, D_LCFS 0.263 m.

## Checks

| Id | Check | Status |
| --- | --- | --- |
| Z4 | Warm start | review: trained at or above the warm start (all 0.9750 against 0.9302, no near-repeat 0.9653 against 0.9166) |
| Z7 | Slices present | pass |
| Z8 | Clip rates | review: largest share 0.0733 |
| Z11 | Mirror averaging | review: on 0.9750, off 0.9740 |
| Z12 | MAST sign | pass (asserted before the MAST maps) |

## Score terms by slice

Identity: `0.55 max(0, R2 psi) + 0.15 max(0, R2 scalars) + 0.10 (1 - D_LCFS) + 0.20 consistency`, beside the scorer's S.

| Slice | Shots | S | Identity | R2 psi | R2 psi per shot | R2 q95 | R2 betaN | 1 - D_LCFS | Consistency | R_axis | Z_axis | kappa | tri_top | tri_bot | volume | li |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 870 | 0.9750 | 0.9750 | 0.9996 | 0.9992 | 0.9397 | 0.9338 | 0.9826 | 0.9322 | 0.9326 | 0.8916 | 0.8991 | 0.9763 | 0.9714 | 0.9553 | 0.8991 |
| no_near_repeat | 286 | 0.9653 | 0.9653 | 0.9993 | 0.9986 | 0.9086 | 0.9294 | 0.9783 | 0.8999 | 0.8868 | 0.8572 | 0.8448 | 0.9746 | 0.9598 | 0.9419 | 0.8345 |
| ip_negative | 74 | 0.9418 | 0.9418 | 0.9996 | 0.9995 | 0.8296 | 0.8258 | 0.9803 | 0.8494 | 0.8218 | 0.6325 | 0.8518 | 0.9416 | 0.9407 | 0.9304 | 0.8271 |
| corner | 151 | 0.9686 | 0.9686 | 0.9996 | 0.9996 | 0.9739 | 0.8038 | 0.9872 | 0.9340 | 0.9287 | 0.9618 | 0.8497 | 0.9803 | 0.9751 | 0.9236 | 0.9188 |

MAST floor (F100, mean_current_v0 on Codabench): S 0.4833, R2 psi 0.7447, D_LCFS 0.263 m.

## Score terms by phase

| Slice | Shots | S | Identity | R2 psi | R2 psi per shot | R2 q95 | R2 betaN | 1 - D_LCFS | Consistency | R_axis | Z_axis | kappa | tri_top | tri_bot | volume | li |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase_rising | 858 | 0.9572 | 0.9572 | 0.9990 | 0.9980 | 0.9196 | 0.9474 | 0.9797 | 0.8491 | 0.8765 | 0.6208 | 0.8203 | 0.9732 | 0.9695 | 0.8885 | 0.7948 |
| phase_flat_top | 862 | 0.9832 | 0.9832 | 0.9996 | 0.9994 | 0.9846 | 0.9297 | 0.9843 | 0.9569 | 0.9458 | 0.9598 | 0.9174 | 0.9793 | 0.9736 | 0.9705 | 0.9518 |
| phase_late | 427 | 0.9134 | 0.9134 | 0.9983 | 0.9967 | 0.5535 | 0.7360 | 0.9694 | 0.8536 | 0.8106 | 0.7936 | 0.8382 | 0.9310 | 0.9424 | 0.9277 | 0.7318 |
| no_near_repeat_phase_rising | 281 | 0.9530 | 0.9530 | 0.9988 | 0.9979 | 0.9094 | 0.9540 | 0.9765 | 0.8312 | 0.8704 | 0.6060 | 0.7763 | 0.9734 | 0.9630 | 0.8690 | 0.7600 |
| no_near_repeat_phase_flat_top | 279 | 0.9753 | 0.9753 | 0.9994 | 0.9990 | 0.9742 | 0.9114 | 0.9817 | 0.9302 | 0.8914 | 0.9443 | 0.8681 | 0.9785 | 0.9610 | 0.9581 | 0.9098 |
| no_near_repeat_phase_late | 151 | 0.9035 | 0.9035 | 0.9969 | 0.9959 | 0.3710 | 0.8642 | 0.9673 | 0.8293 | 0.7839 | 0.7979 | 0.7594 | 0.9230 | 0.9338 | 0.9304 | 0.6763 |

MAST floor (F100, mean_current_v0 on Codabench): S 0.4833, R2 psi 0.7447, D_LCFS 0.263 m.

## Worst 20 validation shots by R2 psi

| Shot | R2 psi | D_LCFS | Boundary failures |
| --- | --- | --- | --- |
| d42a0ed0c0 | 0.9011 | 0.4485 | 0 of 1 |
| 269e33625f | 0.9517 | 0.0276 | 0 of 22 |
| b6f9f193bf | 0.9741 | 0.0509 | 0 of 11 |
| ec216f0d01 | 0.9898 | 0.0308 | 0 of 12 |
| 316f541fd6 | 0.9921 | 0.0280 | 0 of 1 |
| 4339e91e8b | 0.9932 | 0.0214 | 0 of 35 |
| 533cba65c5 | 0.9933 | 0.0468 | 0 of 53 |
| ced938a97c | 0.9939 | 0.0419 | 0 of 37 |
| ac04e52963 | 0.9944 | 0.0717 | 0 of 1 |
| e2932f4823 | 0.9949 | 0.0570 | 0 of 2 |
| 2427c80a5d | 0.9962 | 0.0543 | 0 of 65 |
| 9200260c25 | 0.9967 | 0.0347 | 0 of 83 |
| 2b9b50b9b4 | 0.9968 | 0.0193 | 0 of 75 |
| 8496e7cfe6 | 0.9969 | 0.0269 | 0 of 48 |
| 6208fa6b91 | 0.9970 | 0.0186 | 0 of 26 |
| b2df6a7008 | 0.9973 | 0.0294 | 0 of 17 |
| 4274201181 | 0.9974 | 0.0181 | 0 of 19 |
| 698f1cc29c | 0.9974 | 0.0155 | 0 of 21 |
| b8e064051c | 0.9975 | 0.0268 | 0 of 57 |
| a1a627dfdf | 0.9975 | 0.0400 | 0 of 61 |

## Overfit panel

| Epoch | Train loss | Validation loss | No-near-repeat loss |
| --- | --- | --- | --- |
| 168 (best epoch) | 0.09646 | 0.11490 | 0.17985 |
| 183 (stopped epoch) | 0.09521 | 0.11469 | 0.18077 |

Trained: S all 0.9750, no near-repeat 0.9653, gap 0.0097.

Warm start (ridge): S all 0.9302, no near-repeat 0.9166, gap 0.0136.

Seed spread of S over 2 run(s): mean 0.9749, standard deviation 0.0002, range 0.9748 to 0.9750.
- rung2_g_s0 (seed 0): S 0.9750, no near-repeat 0.9653
- rung2_g_s1 (seed 1): S 0.9748, no near-repeat 0.9652

## Warm start beside the trained model

Prediction chain (warm-start checkpoint `warm.pt` against the best checkpoint; q95 from the map, betaN from the DIII-D head, mirror averaging on):

| Slice | Warm start | Trained | Difference |
| --- | --- | --- | --- |
| all | 0.9302 | 0.9750 | 0.0448 |
| no_near_repeat | 0.9166 | 0.9653 | 0.0487 |
| ip_negative | 0.8915 | 0.9418 | 0.0503 |
| corner | 0.8799 | 0.9686 | 0.0887 |

The run record holds no training-time warm-start or best score (trained with --no-score).

Chain score, mirror averaging on: S 0.9750 (no near-repeat 0.9653); off: S 0.9740 (no near-repeat 0.9641).

## Clip rates

Share of scaled values beyond 10 per column.

DIII-D validation (191411 frames): `ip_rate_rel` 0.0061, `resistive_age` 0.0016, `q_cyl` 0.0006, `ts_beta_p` 0.0000, `coil_pt_00` 0.0000, `coil_pt_04` 0.0000, `coil_pt_05` 0.0000, `coil_pt_19` 0.0000, `coil_pt_20` 0.0000, `coil_pt_21` 0.0000.

DIII-D test (193359 frames): `ip_rate_rel` 0.0059, `q_cyl` 0.0004, `resistive_age` 0.0000.

MAST test (69075 frames): `ip_rate_rel` 0.0733, `q_cyl` 0.0002, `ts_beta_p` 0.0000.

## Grouped permutation importance (validation, all shots)

| Group | Columns | S | drop S | drop r2_psi | drop r2_qb | drop one_minus_dlcfs | drop consistency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| coil_pattern | 22 | 0.8142 | 0.1608 | 0.0095 | 0.0437 | 0.0672 | 0.7114 |
| plasma_content | 3 | 0.9087 | 0.0663 | 0.0035 | 0.0089 | 0.0133 | 0.3084 |
| field_current | 1 | 0.9383 | 0.0367 | 0.0015 | 0.0233 | 0.0054 | 0.1592 |
| shot_position | 3 | 0.9476 | 0.0274 | 0.0012 | 0.0024 | 0.0057 | 0.1288 |

Drop of each consistency value:

| Group | R_axis | Z_axis | kappa | tri_top | tri_bot | volume | li |
| --- | --- | --- | --- | --- | --- | --- | --- |
| coil_pattern | 1.3768 | 2.1731 | 0.8281 | 0.3916 | 0.4140 | 0.9710 | 0.5664 |
| plasma_content | 0.9465 | 0.0938 | 0.5066 | 0.0221 | 0.0222 | 0.2134 | 0.3678 |
| field_current | 0.1375 | 0.0609 | 0.3220 | 0.0076 | 0.0077 | 0.0610 | 0.5176 |
| shot_position | 0.1337 | 0.0328 | 0.3405 | 0.0111 | 0.0079 | 0.0710 | 0.3046 |

## MAST sensitivity (label-free)

61 MAST test shots, 904 frames; unpermuted maps: boundary on 1.000 of frames, map RMS 0.08011 Wb/rad; medians R_axis 0.943, Z_axis -0.000, kappa 1.785, R_geo 0.874, a 0.560. No MAST label is read.

| Group | Map change / RMS | Boundary share | Frames with both | median abs change R_axis | median abs change Z_axis | median abs change kappa | median abs change R_geo | median abs change a |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| coil_pattern | 0.057 | 1.000 | 904 | 0.0211 | 0.0041 | 0.0714 | 0.0183 | 0.0205 |
| plasma_content | 0.035 | 1.000 | 904 | 0.0160 | 0.0002 | 0.0195 | 0.0159 | 0.0063 |
| field_current | 0.021 | 1.000 | 904 | 0.0051 | 0.0003 | 0.0348 | 0.0043 | 0.0088 |
| shot_position | 0.033 | 1.000 | 904 | 0.0120 | 0.0004 | 0.0495 | 0.0095 | 0.0099 |

MAST floor (F100, mean_current_v0 on Codabench): S 0.4833, R2 psi 0.7447, D_LCFS 0.263 m.
