# Evaluation of run rung2_f_s0

Validation list `validation` (870 shots, 48186 frames at stride 4); chain: checkpoints rung2_f_s0, mirror averaging on, q95 from the map, betaN from the DIII-D raw-input ridge (fitted on `train` at stride 4). Corner: q_cyl < 1.780 and ts_beta_p < 0.1088, 2356 frames.

MAST floor (F100, mean_current_v0 on Codabench): S 0.4833, R2 psi 0.7447, D_LCFS 0.263 m.

## Checks

| Id | Check | Status |
| --- | --- | --- |
| Z4 | Warm start | review: trained at or above the warm start (all 0.9733 against 0.9302, no near-repeat 0.9643 against 0.9166) |
| Z7 | Slices present | pass |
| Z8 | Clip rates | review: largest share 0.0733 |
| Z11 | Mirror averaging | review: on 0.9733, off 0.9722 |
| Z12 | MAST sign | pass (asserted before the MAST maps) |

## Score terms by slice

Identity: `0.55 max(0, R2 psi) + 0.15 max(0, R2 scalars) + 0.10 (1 - D_LCFS) + 0.20 consistency`, beside the scorer's S.

| Slice | Shots | S | Identity | R2 psi | R2 psi per shot | R2 q95 | R2 betaN | 1 - D_LCFS | Consistency | R_axis | Z_axis | kappa | tri_top | tri_bot | volume | li |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 870 | 0.9733 | 0.9733 | 0.9995 | 0.9992 | 0.9395 | 0.9338 | 0.9814 | 0.9247 | 0.9238 | 0.8828 | 0.8926 | 0.9732 | 0.9692 | 0.9464 | 0.8846 |
| no_near_repeat | 286 | 0.9643 | 0.9643 | 0.9993 | 0.9985 | 0.9131 | 0.9294 | 0.9774 | 0.8936 | 0.8838 | 0.8481 | 0.8418 | 0.9720 | 0.9565 | 0.9319 | 0.8212 |
| ip_negative | 74 | 0.9429 | 0.9429 | 0.9996 | 0.9995 | 0.8590 | 0.8258 | 0.9794 | 0.8440 | 0.7755 | 0.6397 | 0.8398 | 0.9434 | 0.9424 | 0.9240 | 0.8432 |
| corner | 151 | 0.9658 | 0.9658 | 0.9995 | 0.9995 | 0.9664 | 0.8038 | 0.9859 | 0.9234 | 0.9177 | 0.9609 | 0.8337 | 0.9734 | 0.9691 | 0.9044 | 0.9044 |

MAST floor (F100, mean_current_v0 on Codabench): S 0.4833, R2 psi 0.7447, D_LCFS 0.263 m.

## Score terms by phase

| Slice | Shots | S | Identity | R2 psi | R2 psi per shot | R2 q95 | R2 betaN | 1 - D_LCFS | Consistency | R_axis | Z_axis | kappa | tri_top | tri_bot | volume | li |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase_rising | 858 | 0.9563 | 0.9563 | 0.9990 | 0.9980 | 0.9162 | 0.9474 | 0.9794 | 0.8456 | 0.8660 | 0.6212 | 0.8190 | 0.9720 | 0.9695 | 0.8772 | 0.7944 |
| phase_flat_top | 862 | 0.9812 | 0.9812 | 0.9996 | 0.9994 | 0.9806 | 0.9297 | 0.9830 | 0.9493 | 0.9361 | 0.9508 | 0.9095 | 0.9757 | 0.9708 | 0.9621 | 0.9399 |
| phase_late | 427 | 0.9133 | 0.9133 | 0.9983 | 0.9965 | 0.5904 | 0.7360 | 0.9672 | 0.8400 | 0.8048 | 0.7691 | 0.8267 | 0.9279 | 0.9406 | 0.9153 | 0.6955 |
| no_near_repeat_phase_rising | 281 | 0.9520 | 0.9520 | 0.9988 | 0.9979 | 0.9063 | 0.9540 | 0.9761 | 0.8274 | 0.8674 | 0.5988 | 0.7758 | 0.9705 | 0.9624 | 0.8577 | 0.7593 |
| no_near_repeat_phase_flat_top | 279 | 0.9732 | 0.9732 | 0.9993 | 0.9990 | 0.9657 | 0.9114 | 0.9806 | 0.9235 | 0.8866 | 0.9363 | 0.8655 | 0.9764 | 0.9565 | 0.9476 | 0.8954 |
| no_near_repeat_phase_late | 151 | 0.9084 | 0.9084 | 0.9970 | 0.9957 | 0.4646 | 0.8642 | 0.9649 | 0.8194 | 0.7922 | 0.7789 | 0.7467 | 0.9134 | 0.9329 | 0.9202 | 0.6512 |

MAST floor (F100, mean_current_v0 on Codabench): S 0.4833, R2 psi 0.7447, D_LCFS 0.263 m.

## Worst 20 validation shots by R2 psi

| Shot | R2 psi | D_LCFS | Boundary failures |
| --- | --- | --- | --- |
| d42a0ed0c0 | 0.9014 | 0.4523 | 0 of 1 |
| 269e33625f | 0.9520 | 0.0263 | 0 of 22 |
| b6f9f193bf | 0.9785 | 0.0410 | 0 of 11 |
| 4339e91e8b | 0.9893 | 0.0234 | 0 of 35 |
| ec216f0d01 | 0.9913 | 0.0343 | 0 of 12 |
| 316f541fd6 | 0.9918 | 0.0284 | 0 of 1 |
| ac04e52963 | 0.9939 | 0.0719 | 0 of 1 |
| 533cba65c5 | 0.9948 | 0.0401 | 0 of 53 |
| e2932f4823 | 0.9950 | 0.0557 | 0 of 2 |
| f4889b292e | 0.9952 | 0.0360 | 0 of 62 |
| ced938a97c | 0.9955 | 0.0413 | 0 of 37 |
| b2df6a7008 | 0.9956 | 0.0328 | 0 of 17 |
| 49a34e411a | 0.9962 | 0.0277 | 0 of 49 |
| 2b9b50b9b4 | 0.9966 | 0.0246 | 0 of 75 |
| 2427c80a5d | 0.9967 | 0.0567 | 0 of 65 |
| bd46ca6cea | 0.9968 | 0.0307 | 0 of 26 |
| 8496e7cfe6 | 0.9968 | 0.0268 | 0 of 48 |
| ab04959aa2 | 0.9969 | 0.0232 | 0 of 81 |
| b9f17546e8 | 0.9970 | 0.0176 | 0 of 60 |
| 1b1a5c5531 | 0.9970 | 0.0195 | 0 of 38 |

## Overfit panel

| Epoch | Train loss | Validation loss | No-near-repeat loss |
| --- | --- | --- | --- |
| 193 (best epoch) | 0.02969 | 0.03635 | 0.05738 |
| 199 (stopped epoch) | 0.02965 | 0.03636 | 0.05741 |

Trained: S all 0.9733, no near-repeat 0.9643, gap 0.0091.

Warm start (ridge): S all 0.9302, no near-repeat 0.9166, gap 0.0136.

Seed spread of S over 2 run(s): mean 0.9734, standard deviation 0.0001, range 0.9733 to 0.9735.
- rung2_f_s0 (seed 0): S 0.9733, no near-repeat 0.9643
- rung2_f_s1 (seed 1): S 0.9735, no near-repeat 0.9648

## Warm start beside the trained model

Prediction chain (warm-start checkpoint `warm.pt` against the best checkpoint; q95 from the map, betaN from the DIII-D head, mirror averaging on):

| Slice | Warm start | Trained | Difference |
| --- | --- | --- | --- |
| all | 0.9302 | 0.9733 | 0.0431 |
| no_near_repeat | 0.9166 | 0.9643 | 0.0477 |
| ip_negative | 0.8915 | 0.9429 | 0.0514 |
| corner | 0.8799 | 0.9658 | 0.0859 |

The run record holds no training-time warm-start or best score (trained with --no-score).

Chain score, mirror averaging on: S 0.9733 (no near-repeat 0.9643); off: S 0.9722 (no near-repeat 0.9635).

## Clip rates

Share of scaled values beyond 10 per column.

DIII-D validation (191411 frames): `ip_rate_rel` 0.0061, `resistive_age` 0.0016, `q_cyl` 0.0006, `ts_beta_p` 0.0000, `coil_pt_00` 0.0000, `coil_pt_04` 0.0000, `coil_pt_05` 0.0000, `coil_pt_19` 0.0000, `coil_pt_20` 0.0000, `coil_pt_21` 0.0000.

DIII-D test (193359 frames): `ip_rate_rel` 0.0059, `q_cyl` 0.0004, `resistive_age` 0.0000.

MAST test (69075 frames): `ip_rate_rel` 0.0733, `q_cyl` 0.0002, `ts_beta_p` 0.0000.

## Grouped permutation importance (validation, all shots)

| Group | Columns | S | drop S | drop r2_psi | drop r2_qb | drop one_minus_dlcfs | drop consistency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| coil_pattern | 22 | 0.8112 | 0.1621 | 0.0090 | 0.0431 | 0.0658 | 0.7206 |
| plasma_content | 3 | 0.9118 | 0.0616 | 0.0031 | 0.0076 | 0.0115 | 0.2879 |
| field_current | 1 | 0.9296 | 0.0437 | 0.0017 | 0.0255 | 0.0052 | 0.1923 |
| shot_position | 3 | 0.9494 | 0.0240 | 0.0010 | -0.0001 | 0.0049 | 0.1148 |

Drop of each consistency value:

| Group | R_axis | Z_axis | kappa | tri_top | tri_bot | volume | li |
| --- | --- | --- | --- | --- | --- | --- | --- |
| coil_pattern | 1.2813 | 2.1132 | 0.8744 | 0.3900 | 0.4123 | 0.9422 | 0.6184 |
| plasma_content | 0.9164 | 0.0811 | 0.4282 | 0.0212 | 0.0242 | 0.2043 | 0.3398 |
| field_current | 0.1480 | 0.0851 | 0.4035 | 0.0058 | 0.0087 | 0.0631 | 0.6315 |
| shot_position | 0.1413 | 0.0451 | 0.3006 | 0.0094 | 0.0069 | 0.0675 | 0.2324 |

## MAST sensitivity (label-free)

61 MAST test shots, 904 frames; unpermuted maps: boundary on 1.000 of frames, map RMS 0.07767 Wb/rad; medians R_axis 0.968, Z_axis 0.000, kappa 1.732, R_geo 0.903, a 0.583. No MAST label is read.

| Group | Map change / RMS | Boundary share | Frames with both | median abs change R_axis | median abs change Z_axis | median abs change kappa | median abs change R_geo | median abs change a |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| coil_pattern | 0.070 | 1.000 | 904 | 0.0272 | 0.0042 | 0.0716 | 0.0227 | 0.0242 |
| plasma_content | 0.033 | 1.000 | 904 | 0.0145 | 0.0002 | 0.0202 | 0.0137 | 0.0070 |
| field_current | 0.021 | 1.000 | 904 | 0.0041 | 0.0003 | 0.0283 | 0.0030 | 0.0077 |
| shot_position | 0.040 | 1.000 | 904 | 0.0138 | 0.0004 | 0.0409 | 0.0100 | 0.0102 |

MAST floor (F100, mean_current_v0 on Codabench): S 0.4833, R2 psi 0.7447, D_LCFS 0.263 m.
