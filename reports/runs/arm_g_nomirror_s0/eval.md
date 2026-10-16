# Evaluation of run arm_g_nomirror_s0

Validation list `validation` (870 shots, 48186 frames at stride 4); chain: checkpoints arm_g_nomirror_s0, mirror averaging on, q95 from the map, betaN from the DIII-D raw-input ridge (fitted on `train` at stride 4). Corner: q_cyl < 1.780 and ts_beta_p < 0.1088, 2356 frames.

MAST floor (F100, mean_current_v0 on Codabench): S 0.4833, R2 psi 0.7447, D_LCFS 0.263 m.

## Checks

| Id | Check | Status |
| --- | --- | --- |
| Z4 | Warm start | review: trained at or above the warm start (all 0.9698 against 0.9302, no near-repeat 0.9625 against 0.9166) |
| Z7 | Slices present | pass |
| Z8 | Clip rates | review: largest share 0.0733 |
| Z11 | Mirror averaging | review: on 0.9698, off 0.9798 |
| Z12 | MAST sign | pass (asserted before the MAST maps) |

## Score terms by slice

Identity: `0.55 max(0, R2 psi) + 0.15 max(0, R2 scalars) + 0.10 (1 - D_LCFS) + 0.20 consistency`, beside the scorer's S.

| Slice | Shots | S | Identity | R2 psi | R2 psi per shot | R2 q95 | R2 betaN | 1 - D_LCFS | Consistency | R_axis | Z_axis | kappa | tri_top | tri_bot | volume | li |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 870 | 0.9698 | 0.9698 | 0.9992 | 0.9988 | 0.9432 | 0.9338 | 0.9805 | 0.9069 | 0.8700 | 0.9154 | 0.8312 | 0.9715 | 0.9705 | 0.9415 | 0.8480 |
| no_near_repeat | 286 | 0.9625 | 0.9625 | 0.9990 | 0.9982 | 0.9271 | 0.9294 | 0.9769 | 0.8803 | 0.8238 | 0.8882 | 0.7883 | 0.9739 | 0.9606 | 0.9300 | 0.7969 |
| ip_negative | 74 | 0.9473 | 0.9473 | 0.9992 | 0.9991 | 0.9439 | 0.8258 | 0.9798 | 0.8353 | 0.5953 | 0.8776 | 0.7574 | 0.9432 | 0.9563 | 0.9167 | 0.8005 |
| corner | 151 | 0.9522 | 0.9522 | 0.9990 | 0.9989 | 0.9637 | 0.8038 | 0.9851 | 0.8585 | 0.7318 | 0.9348 | 0.7533 | 0.9758 | 0.9660 | 0.8859 | 0.7618 |

MAST floor (F100, mean_current_v0 on Codabench): S 0.4833, R2 psi 0.7447, D_LCFS 0.263 m.

## Score terms by phase

| Slice | Shots | S | Identity | R2 psi | R2 psi per shot | R2 q95 | R2 betaN | 1 - D_LCFS | Consistency | R_axis | Z_axis | kappa | tri_top | tri_bot | volume | li |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase_rising | 858 | 0.9629 | 0.9629 | 0.9987 | 0.9979 | 0.9285 | 0.9474 | 0.9807 | 0.8743 | 0.8725 | 0.8442 | 0.7921 | 0.9770 | 0.9728 | 0.9035 | 0.7580 |
| phase_flat_top | 862 | 0.9727 | 0.9727 | 0.9991 | 0.9988 | 0.9795 | 0.9297 | 0.9814 | 0.9091 | 0.8572 | 0.9315 | 0.8293 | 0.9714 | 0.9711 | 0.9470 | 0.8559 |
| phase_late | 427 | 0.9207 | 0.9207 | 0.9987 | 0.9970 | 0.5924 | 0.7360 | 0.9711 | 0.8733 | 0.8063 | 0.9060 | 0.8131 | 0.9424 | 0.9506 | 0.9392 | 0.7557 |
| no_near_repeat_phase_rising | 281 | 0.9603 | 0.9603 | 0.9987 | 0.9979 | 0.9229 | 0.9540 | 0.9774 | 0.8627 | 0.8689 | 0.7928 | 0.7625 | 0.9797 | 0.9669 | 0.8918 | 0.7761 |
| no_near_repeat_phase_flat_top | 279 | 0.9638 | 0.9638 | 0.9989 | 0.9985 | 0.9662 | 0.9114 | 0.9798 | 0.8781 | 0.7860 | 0.9131 | 0.7851 | 0.9743 | 0.9597 | 0.9313 | 0.7975 |
| no_near_repeat_phase_late | 151 | 0.9226 | 0.9226 | 0.9980 | 0.9967 | 0.5588 | 0.8642 | 0.9676 | 0.8510 | 0.7748 | 0.9114 | 0.7468 | 0.9400 | 0.9459 | 0.9410 | 0.6972 |

MAST floor (F100, mean_current_v0 on Codabench): S 0.4833, R2 psi 0.7447, D_LCFS 0.263 m.

## Worst 20 validation shots by R2 psi

| Shot | R2 psi | D_LCFS | Boundary failures |
| --- | --- | --- | --- |
| d42a0ed0c0 | 0.8685 | 0.4950 | 0 of 1 |
| b6f9f193bf | 0.9859 | 0.0514 | 0 of 11 |
| 8448e4c175 | 0.9873 | 0.0287 | 0 of 48 |
| 269e33625f | 0.9887 | 0.0194 | 0 of 22 |
| 316f541fd6 | 0.9892 | 0.0381 | 0 of 1 |
| 5bbd1db534 | 0.9915 | 0.0245 | 0 of 71 |
| 39b1adf166 | 0.9928 | 0.0228 | 0 of 71 |
| 533cba65c5 | 0.9939 | 0.0343 | 0 of 53 |
| 3b2025b28b | 0.9940 | 0.0239 | 0 of 53 |
| e53563abd5 | 0.9942 | 0.0487 | 0 of 58 |
| 6208fa6b91 | 0.9944 | 0.0666 | 0 of 26 |
| 2427c80a5d | 0.9948 | 0.0700 | 0 of 65 |
| 5f4430874c | 0.9949 | 0.0573 | 0 of 57 |
| 1489ba3bad | 0.9951 | 0.0185 | 0 of 29 |
| ec216f0d01 | 0.9951 | 0.0397 | 0 of 12 |
| d2014e5e0a | 0.9951 | 0.0552 | 0 of 45 |
| 5cccde4618 | 0.9952 | 0.0814 | 0 of 61 |
| add4fc7429 | 0.9953 | 0.0227 | 0 of 86 |
| 5b0f6e1965 | 0.9953 | 0.0262 | 0 of 23 |
| 375c9ed705 | 0.9955 | 0.0196 | 0 of 65 |

## Overfit panel

| Epoch | Train loss | Validation loss | No-near-repeat loss |
| --- | --- | --- | --- |
| 185 (best epoch) | 0.10522 | 0.10590 | 0.17638 |
| 199 (stopped epoch) | 0.10525 | 0.10590 | 0.17646 |

Trained: S all 0.9698, no near-repeat 0.9625, gap 0.0073.

Warm start (ridge): S all 0.9302, no near-repeat 0.9166, gap 0.0136.

Seed spread of S over 1 run(s): mean 0.9698, standard deviation -, range 0.9698 to 0.9698.
- arm_g_nomirror_s0 (seed 0): S 0.9698, no near-repeat 0.9625

## Warm start beside the trained model

Prediction chain (warm-start checkpoint `warm.pt` against the best checkpoint; q95 from the map, betaN from the DIII-D head, mirror averaging on):

| Slice | Warm start | Trained | Difference |
| --- | --- | --- | --- |
| all | 0.9302 | 0.9698 | 0.0396 |
| no_near_repeat | 0.9166 | 0.9625 | 0.0459 |
| ip_negative | 0.8915 | 0.9473 | 0.0558 |
| corner | 0.8799 | 0.9522 | 0.0723 |

The run record holds no training-time warm-start or best score (trained with --no-score).

Chain score, mirror averaging on: S 0.9698 (no near-repeat 0.9625); off: S 0.9798 (no near-repeat 0.9722).

## Clip rates

Share of scaled values beyond 10 per column.

DIII-D validation (191411 frames): `ip_rate_rel` 0.0061, `resistive_age` 0.0016, `q_cyl` 0.0006, `ts_beta_p` 0.0000, `coil_pt_00` 0.0000, `coil_pt_04` 0.0000, `coil_pt_05` 0.0000, `coil_pt_19` 0.0000, `coil_pt_20` 0.0000, `coil_pt_21` 0.0000.

DIII-D test (193359 frames): `ip_rate_rel` 0.0059, `q_cyl` 0.0004, `resistive_age` 0.0000.

MAST test (69075 frames): `ip_rate_rel` 0.0733, `q_cyl` 0.0002, `ts_beta_p` 0.0000.

## Grouped permutation importance (validation, all shots)

| Group | Columns | S | drop S | drop r2_psi | drop r2_qb | drop one_minus_dlcfs | drop consistency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| coil_pattern | 22 | 0.7980 | 0.1718 | 0.0114 | 0.0603 | 0.0660 | 0.7496 |
| plasma_content | 3 | 0.9209 | 0.0489 | 0.0029 | 0.0032 | 0.0096 | 0.2293 |
| field_current | 1 | 0.9480 | 0.0218 | 0.0006 | 0.0123 | 0.0032 | 0.0962 |
| shot_position | 3 | 0.9402 | 0.0295 | 0.0008 | 0.0133 | 0.0041 | 0.1333 |

Drop of each consistency value:

| Group | R_axis | Z_axis | kappa | tri_top | tri_bot | volume | li |
| --- | --- | --- | --- | --- | --- | --- | --- |
| coil_pattern | 1.0539 | 2.1226 | 1.1239 | 0.4265 | 0.4411 | 0.9148 | 0.8609 |
| plasma_content | 1.1262 | 0.0542 | 0.2762 | 0.0221 | 0.0277 | 0.1787 | 0.1758 |
| field_current | 0.1480 | 0.0640 | 0.1405 | 0.0078 | 0.0064 | 0.0530 | 0.2537 |
| shot_position | 0.0850 | 0.0639 | 0.3617 | 0.0085 | 0.0078 | 0.0382 | 0.3683 |

## MAST sensitivity (label-free)

61 MAST test shots, 904 frames; unpermuted maps: boundary on 1.000 of frames, map RMS 0.07954 Wb/rad; medians R_axis 0.941, Z_axis -0.000, kappa 1.824, R_geo 0.879, a 0.562. No MAST label is read.

| Group | Map change / RMS | Boundary share | Frames with both | median abs change R_axis | median abs change Z_axis | median abs change kappa | median abs change R_geo | median abs change a |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| coil_pattern | 0.057 | 1.000 | 904 | 0.0215 | 0.0046 | 0.0561 | 0.0162 | 0.0149 |
| plasma_content | 0.026 | 1.000 | 904 | 0.0054 | 0.0002 | 0.0102 | 0.0107 | 0.0030 |
| field_current | 0.015 | 1.000 | 904 | 0.0047 | 0.0002 | 0.0158 | 0.0041 | 0.0029 |
| shot_position | 0.024 | 1.000 | 904 | 0.0063 | 0.0002 | 0.0259 | 0.0045 | 0.0059 |

MAST floor (F100, mean_current_v0 on Codabench): S 0.4833, R2 psi 0.7447, D_LCFS 0.263 m.
