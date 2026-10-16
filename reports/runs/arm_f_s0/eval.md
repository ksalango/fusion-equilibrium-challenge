# Evaluation of run arm_f_s0

Validation list `validation` (870 shots, 48186 frames at stride 4); chain: checkpoints arm_f_s0, mirror averaging on, q95 from the map, betaN from the DIII-D raw-input ridge (fitted on `train` at stride 4). Corner: q_cyl < 1.780 and ts_beta_p < 0.1088, 2356 frames.

MAST floor (F100, mean_current_v0 on Codabench): S 0.4833, R2 psi 0.7447, D_LCFS 0.263 m.

## Checks

| Id | Check | Status |
| --- | --- | --- |
| Z4 | Warm start | review: trained at or above the warm start (all 0.9754 against 0.9302, no near-repeat 0.9674 against 0.9166) |
| Z7 | Slices present | pass |
| Z8 | Clip rates | review: largest share 0.0733 |
| Z11 | Mirror averaging | review: on 0.9754, off 0.9750 |
| Z12 | MAST sign | pass (asserted before the MAST maps) |

## Score terms by slice

Identity: `0.55 max(0, R2 psi) + 0.15 max(0, R2 scalars) + 0.10 (1 - D_LCFS) + 0.20 consistency`, beside the scorer's S.

| Slice | Shots | S | Identity | R2 psi | R2 psi per shot | R2 q95 | R2 betaN | 1 - D_LCFS | Consistency | R_axis | Z_axis | kappa | tri_top | tri_bot | volume | li |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 870 | 0.9754 | 0.9754 | 0.9995 | 0.9991 | 0.9346 | 0.9338 | 0.9827 | 0.9366 | 0.9329 | 0.9490 | 0.8913 | 0.9779 | 0.9726 | 0.9527 | 0.8796 |
| no_near_repeat | 286 | 0.9674 | 0.9674 | 0.9993 | 0.9984 | 0.9143 | 0.9294 | 0.9788 | 0.9083 | 0.8988 | 0.9193 | 0.8414 | 0.9765 | 0.9616 | 0.9411 | 0.8196 |
| ip_negative | 74 | 0.9533 | 0.9533 | 0.9995 | 0.9994 | 0.9139 | 0.8258 | 0.9798 | 0.8754 | 0.7899 | 0.9040 | 0.8155 | 0.9425 | 0.9475 | 0.9207 | 0.8074 |
| corner | 151 | 0.9633 | 0.9633 | 0.9994 | 0.9994 | 0.9563 | 0.8038 | 0.9867 | 0.9150 | 0.9001 | 0.9784 | 0.8057 | 0.9783 | 0.9729 | 0.9102 | 0.8592 |

MAST floor (F100, mean_current_v0 on Codabench): S 0.4833, R2 psi 0.7447, D_LCFS 0.263 m.

## Score terms by phase

| Slice | Shots | S | Identity | R2 psi | R2 psi per shot | R2 q95 | R2 betaN | 1 - D_LCFS | Consistency | R_axis | Z_axis | kappa | tri_top | tri_bot | volume | li |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase_rising | 858 | 0.9658 | 0.9658 | 0.9990 | 0.9982 | 0.9058 | 0.9474 | 0.9819 | 0.8959 | 0.8965 | 0.8754 | 0.8466 | 0.9779 | 0.9742 | 0.8961 | 0.8045 |
| phase_flat_top | 862 | 0.9804 | 0.9804 | 0.9995 | 0.9993 | 0.9768 | 0.9297 | 0.9840 | 0.9464 | 0.9396 | 0.9688 | 0.9009 | 0.9794 | 0.9735 | 0.9639 | 0.8985 |
| phase_late | 427 | 0.9224 | 0.9224 | 0.9988 | 0.9970 | 0.5894 | 0.7360 | 0.9707 | 0.8833 | 0.8371 | 0.9113 | 0.8234 | 0.9472 | 0.9497 | 0.9387 | 0.7754 |
| no_near_repeat_phase_rising | 281 | 0.9620 | 0.9620 | 0.9988 | 0.9981 | 0.8990 | 0.9540 | 0.9785 | 0.8792 | 0.8919 | 0.8118 | 0.8109 | 0.9783 | 0.9669 | 0.8856 | 0.8089 |
| no_near_repeat_phase_flat_top | 279 | 0.9714 | 0.9714 | 0.9992 | 0.9989 | 0.9620 | 0.9114 | 0.9820 | 0.9156 | 0.8970 | 0.9559 | 0.8538 | 0.9781 | 0.9610 | 0.9503 | 0.8128 |
| no_near_repeat_phase_late | 151 | 0.9221 | 0.9221 | 0.9982 | 0.9969 | 0.5220 | 0.8642 | 0.9673 | 0.8619 | 0.8267 | 0.8969 | 0.7400 | 0.9439 | 0.9482 | 0.9403 | 0.7375 |

MAST floor (F100, mean_current_v0 on Codabench): S 0.4833, R2 psi 0.7447, D_LCFS 0.263 m.

## Worst 20 validation shots by R2 psi

| Shot | R2 psi | D_LCFS | Boundary failures |
| --- | --- | --- | --- |
| d42a0ed0c0 | 0.8213 | 0.5233 | 0 of 1 |
| b6f9f193bf | 0.9884 | 0.0553 | 0 of 11 |
| 269e33625f | 0.9896 | 0.0172 | 0 of 22 |
| 316f541fd6 | 0.9896 | 0.0382 | 0 of 1 |
| ec216f0d01 | 0.9917 | 0.0335 | 0 of 12 |
| f4889b292e | 0.9944 | 0.0313 | 0 of 62 |
| 2427c80a5d | 0.9951 | 0.0604 | 0 of 65 |
| b2df6a7008 | 0.9951 | 0.0345 | 0 of 17 |
| 533cba65c5 | 0.9953 | 0.0344 | 0 of 53 |
| 8496e7cfe6 | 0.9959 | 0.0267 | 0 of 48 |
| ab04959aa2 | 0.9962 | 0.0202 | 0 of 81 |
| 2b9b50b9b4 | 0.9962 | 0.0239 | 0 of 75 |
| e365322b54 | 0.9968 | 0.0399 | 0 of 76 |
| ced938a97c | 0.9970 | 0.0261 | 0 of 37 |
| 8448e4c175 | 0.9970 | 0.0182 | 0 of 48 |
| 4274201181 | 0.9971 | 0.0190 | 0 of 19 |
| e2932f4823 | 0.9972 | 0.0402 | 0 of 2 |
| 939fa018a8 | 0.9972 | 0.0188 | 0 of 67 |
| 4c146d6cd7 | 0.9973 | 0.0176 | 0 of 28 |
| 1b1a5c5531 | 0.9974 | 0.0200 | 0 of 38 |

## Overfit panel

| Epoch | Train loss | Validation loss | No-near-repeat loss |
| --- | --- | --- | --- |
| 178 (best epoch) | 0.04394 | 0.04067 | 0.06475 |
| 193 (stopped epoch) | 0.04389 | 0.04066 | 0.06486 |

Trained: S all 0.9754, no near-repeat 0.9674, gap 0.0080.

Warm start (ridge): S all 0.9302, no near-repeat 0.9166, gap 0.0136.

Seed spread of S over 3 run(s): mean 0.9755, standard deviation 0.0003, range 0.9751 to 0.9758.
- arm_f_s0 (seed 0): S 0.9754, no near-repeat 0.9674
- arm_f_s1 (seed 1): S 0.9758, no near-repeat 0.9678
- arm_f_s2 (seed 2): S 0.9751, no near-repeat 0.9665

## Warm start beside the trained model

Prediction chain (warm-start checkpoint `warm.pt` against the best checkpoint; q95 from the map, betaN from the DIII-D head, mirror averaging on):

| Slice | Warm start | Trained | Difference |
| --- | --- | --- | --- |
| all | 0.9302 | 0.9754 | 0.0452 |
| no_near_repeat | 0.9166 | 0.9674 | 0.0508 |
| ip_negative | 0.8915 | 0.9533 | 0.0617 |
| corner | 0.8799 | 0.9633 | 0.0834 |

The run record holds no training-time warm-start or best score (trained with --no-score).

Chain score, mirror averaging on: S 0.9754 (no near-repeat 0.9674); off: S 0.9750 (no near-repeat 0.9666).

## Clip rates

Share of scaled values beyond 10 per column.

DIII-D validation (191411 frames): `ip_rate_rel` 0.0061, `resistive_age` 0.0016, `q_cyl` 0.0006, `ts_beta_p` 0.0000, `coil_pt_00` 0.0000, `coil_pt_04` 0.0000, `coil_pt_05` 0.0000, `coil_pt_19` 0.0000, `coil_pt_20` 0.0000, `coil_pt_21` 0.0000.

DIII-D test (193359 frames): `ip_rate_rel` 0.0059, `q_cyl` 0.0004, `resistive_age` 0.0000.

MAST test (69075 frames): `ip_rate_rel` 0.0733, `q_cyl` 0.0002, `ts_beta_p` 0.0000.

## Grouped permutation importance (validation, all shots)

| Group | Columns | S | drop S | drop r2_psi | drop r2_qb | drop one_minus_dlcfs | drop consistency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| coil_pattern | 22 | 0.7967 | 0.1788 | 0.0120 | 0.0641 | 0.0693 | 0.7781 |
| plasma_content | 3 | 0.9220 | 0.0535 | 0.0029 | 0.0038 | 0.0111 | 0.2512 |
| field_current | 1 | 0.9514 | 0.0241 | 0.0007 | 0.0145 | 0.0040 | 0.1056 |
| shot_position | 3 | 0.9469 | 0.0286 | 0.0008 | 0.0089 | 0.0045 | 0.1317 |

Drop of each consistency value:

| Group | R_axis | Z_axis | kappa | tri_top | tri_bot | volume | li |
| --- | --- | --- | --- | --- | --- | --- | --- |
| coil_pattern | 1.0923 | 2.1757 | 1.2288 | 0.4172 | 0.4403 | 0.9963 | 0.8632 |
| plasma_content | 1.1707 | 0.0555 | 0.3131 | 0.0289 | 0.0308 | 0.2001 | 0.1968 |
| field_current | 0.1753 | 0.0535 | 0.1677 | 0.0090 | 0.0069 | 0.0621 | 0.2651 |
| shot_position | 0.0950 | 0.0645 | 0.3530 | 0.0103 | 0.0086 | 0.0454 | 0.3454 |

## MAST sensitivity (label-free)

61 MAST test shots, 904 frames; unpermuted maps: boundary on 1.000 of frames, map RMS 0.07884 Wb/rad; medians R_axis 0.946, Z_axis -0.000, kappa 1.846, R_geo 0.887, a 0.558. No MAST label is read.

| Group | Map change / RMS | Boundary share | Frames with both | median abs change R_axis | median abs change Z_axis | median abs change kappa | median abs change R_geo | median abs change a |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| coil_pattern | 0.057 | 1.000 | 904 | 0.0196 | 0.0049 | 0.0491 | 0.0141 | 0.0132 |
| plasma_content | 0.036 | 1.000 | 904 | 0.0139 | 0.0001 | 0.0165 | 0.0151 | 0.0054 |
| field_current | 0.014 | 1.000 | 904 | 0.0039 | 0.0002 | 0.0173 | 0.0025 | 0.0037 |
| shot_position | 0.022 | 1.000 | 904 | 0.0055 | 0.0002 | 0.0314 | 0.0049 | 0.0066 |

MAST floor (F100, mean_current_v0 on Codabench): S 0.4833, R2 psi 0.7447, D_LCFS 0.263 m.
