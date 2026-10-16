# Evaluation of run arm_g_s0

Validation list `validation` (870 shots, 48186 frames at stride 4); chain: checkpoints arm_g_s0, mirror averaging on, q95 from the map, betaN from the DIII-D raw-input ridge (fitted on `train` at stride 4). Corner: q_cyl < 1.780 and ts_beta_p < 0.1088, 2356 frames.

MAST floor (F100, mean_current_v0 on Codabench): S 0.4833, R2 psi 0.7447, D_LCFS 0.263 m.

## Checks

| Id | Check | Status |
| --- | --- | --- |
| Z4 | Warm start | review: trained at or above the warm start (all 0.9784 against 0.9302, no near-repeat 0.9705 against 0.9166) |
| Z7 | Slices present | pass |
| Z8 | Clip rates | review: largest share 0.0733 |
| Z11 | Mirror averaging | review: on 0.9784, off 0.9779 |
| Z12 | MAST sign | pass (asserted before the MAST maps) |

## Score terms by slice

Identity: `0.55 max(0, R2 psi) + 0.15 max(0, R2 scalars) + 0.10 (1 - D_LCFS) + 0.20 consistency`, beside the scorer's S.

| Slice | Shots | S | Identity | R2 psi | R2 psi per shot | R2 q95 | R2 betaN | 1 - D_LCFS | Consistency | R_axis | Z_axis | kappa | tri_top | tri_bot | volume | li |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 870 | 0.9784 | 0.9784 | 0.9996 | 0.9992 | 0.9468 | 0.9338 | 0.9843 | 0.9459 | 0.9373 | 0.9530 | 0.9011 | 0.9824 | 0.9770 | 0.9626 | 0.9079 |
| no_near_repeat | 286 | 0.9705 | 0.9705 | 0.9994 | 0.9985 | 0.9271 | 0.9294 | 0.9805 | 0.9177 | 0.9009 | 0.9249 | 0.8520 | 0.9813 | 0.9694 | 0.9511 | 0.8444 |
| ip_negative | 74 | 0.9568 | 0.9568 | 0.9995 | 0.9995 | 0.9213 | 0.8258 | 0.9814 | 0.8895 | 0.8101 | 0.9122 | 0.8259 | 0.9504 | 0.9581 | 0.9405 | 0.8292 |
| corner | 151 | 0.9685 | 0.9685 | 0.9995 | 0.9995 | 0.9719 | 0.8038 | 0.9882 | 0.9339 | 0.9065 | 0.9803 | 0.8394 | 0.9844 | 0.9783 | 0.9279 | 0.9207 |

MAST floor (F100, mean_current_v0 on Codabench): S 0.4833, R2 psi 0.7447, D_LCFS 0.263 m.

## Score terms by phase

| Slice | Shots | S | Identity | R2 psi | R2 psi per shot | R2 q95 | R2 betaN | 1 - D_LCFS | Consistency | R_axis | Z_axis | kappa | tri_top | tri_bot | volume | li |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase_rising | 858 | 0.9702 | 0.9702 | 0.9991 | 0.9983 | 0.9285 | 0.9474 | 0.9829 | 0.9085 | 0.8972 | 0.8774 | 0.8539 | 0.9803 | 0.9770 | 0.9175 | 0.8559 |
| phase_flat_top | 862 | 0.9832 | 0.9832 | 0.9996 | 0.9994 | 0.9845 | 0.9297 | 0.9856 | 0.9568 | 0.9448 | 0.9724 | 0.9104 | 0.9843 | 0.9782 | 0.9715 | 0.9356 |
| phase_late | 427 | 0.9278 | 0.9278 | 0.9989 | 0.9975 | 0.6203 | 0.7360 | 0.9734 | 0.8967 | 0.8527 | 0.9227 | 0.8493 | 0.9525 | 0.9567 | 0.9516 | 0.7917 |
| no_near_repeat_phase_rising | 281 | 0.9656 | 0.9656 | 0.9989 | 0.9982 | 0.9230 | 0.9540 | 0.9796 | 0.8873 | 0.8900 | 0.8191 | 0.8167 | 0.9810 | 0.9710 | 0.9034 | 0.8300 |
| no_near_repeat_phase_flat_top | 279 | 0.9751 | 0.9751 | 0.9993 | 0.9990 | 0.9744 | 0.9114 | 0.9836 | 0.9287 | 0.8993 | 0.9593 | 0.8641 | 0.9835 | 0.9699 | 0.9586 | 0.8659 |
| no_near_repeat_phase_late | 151 | 0.9260 | 0.9260 | 0.9984 | 0.9973 | 0.5337 | 0.8642 | 0.9701 | 0.8749 | 0.8443 | 0.9128 | 0.7692 | 0.9501 | 0.9571 | 0.9532 | 0.7377 |

MAST floor (F100, mean_current_v0 on Codabench): S 0.4833, R2 psi 0.7447, D_LCFS 0.263 m.

## Worst 20 validation shots by R2 psi

| Shot | R2 psi | D_LCFS | Boundary failures |
| --- | --- | --- | --- |
| d42a0ed0c0 | 0.8309 | 0.5132 | 0 of 1 |
| b6f9f193bf | 0.9884 | 0.0475 | 0 of 11 |
| 269e33625f | 0.9912 | 0.0183 | 0 of 22 |
| 316f541fd6 | 0.9912 | 0.0297 | 0 of 1 |
| ec216f0d01 | 0.9925 | 0.0310 | 0 of 12 |
| f4889b292e | 0.9941 | 0.0307 | 0 of 62 |
| b2df6a7008 | 0.9950 | 0.0337 | 0 of 17 |
| 533cba65c5 | 0.9953 | 0.0314 | 0 of 53 |
| 2427c80a5d | 0.9957 | 0.0459 | 0 of 65 |
| e2932f4823 | 0.9965 | 0.0515 | 0 of 2 |
| e365322b54 | 0.9966 | 0.0340 | 0 of 76 |
| ab04959aa2 | 0.9972 | 0.0182 | 0 of 81 |
| ced938a97c | 0.9972 | 0.0262 | 0 of 37 |
| 2b9b50b9b4 | 0.9972 | 0.0170 | 0 of 75 |
| 8496e7cfe6 | 0.9972 | 0.0266 | 0 of 48 |
| 4c146d6cd7 | 0.9974 | 0.0165 | 0 of 28 |
| 1b1a5c5531 | 0.9974 | 0.0200 | 0 of 38 |
| 4274201181 | 0.9975 | 0.0189 | 0 of 19 |
| a2ae4f3c2b | 0.9975 | 0.0241 | 0 of 71 |
| 6208fa6b91 | 0.9976 | 0.0162 | 0 of 26 |

## Overfit panel

| Epoch | Train loss | Validation loss | No-near-repeat loss |
| --- | --- | --- | --- |
| 182 (best epoch) | 0.13521 | 0.12642 | 0.19824 |
| 197 (stopped epoch) | 0.13507 | 0.12643 | 0.19837 |

Trained: S all 0.9784, no near-repeat 0.9705, gap 0.0079.

Warm start (ridge): S all 0.9302, no near-repeat 0.9166, gap 0.0136.

Seed spread of S over 3 run(s): mean 0.9784, standard deviation 0.0001, range 0.9783 to 0.9784.
- arm_g_s0 (seed 0): S 0.9784, no near-repeat 0.9705
- arm_g_s1 (seed 1): S 0.9783, no near-repeat 0.9701
- arm_g_s2 (seed 2): S 0.9784, no near-repeat 0.9705

## Warm start beside the trained model

Prediction chain (warm-start checkpoint `warm.pt` against the best checkpoint; q95 from the map, betaN from the DIII-D head, mirror averaging on):

| Slice | Warm start | Trained | Difference |
| --- | --- | --- | --- |
| all | 0.9302 | 0.9784 | 0.0482 |
| no_near_repeat | 0.9166 | 0.9705 | 0.0539 |
| ip_negative | 0.8915 | 0.9568 | 0.0653 |
| corner | 0.8799 | 0.9685 | 0.0886 |

The run record holds no training-time warm-start or best score (trained with --no-score).

Chain score, mirror averaging on: S 0.9784 (no near-repeat 0.9705); off: S 0.9779 (no near-repeat 0.9701).

## Clip rates

Share of scaled values beyond 10 per column.

DIII-D validation (191411 frames): `ip_rate_rel` 0.0061, `resistive_age` 0.0016, `q_cyl` 0.0006, `ts_beta_p` 0.0000, `coil_pt_00` 0.0000, `coil_pt_04` 0.0000, `coil_pt_05` 0.0000, `coil_pt_19` 0.0000, `coil_pt_20` 0.0000, `coil_pt_21` 0.0000.

DIII-D test (193359 frames): `ip_rate_rel` 0.0059, `q_cyl` 0.0004, `resistive_age` 0.0000.

MAST test (69075 frames): `ip_rate_rel` 0.0733, `q_cyl` 0.0002, `ts_beta_p` 0.0000.

## Grouped permutation importance (validation, all shots)

| Group | Columns | S | drop S | drop r2_psi | drop r2_qb | drop one_minus_dlcfs | drop consistency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| coil_pattern | 22 | 0.7972 | 0.1812 | 0.0122 | 0.0625 | 0.0707 | 0.7901 |
| plasma_content | 3 | 0.9251 | 0.0533 | 0.0028 | 0.0039 | 0.0115 | 0.2500 |
| field_current | 1 | 0.9538 | 0.0246 | 0.0007 | 0.0174 | 0.0047 | 0.1058 |
| shot_position | 3 | 0.9482 | 0.0302 | 0.0009 | 0.0127 | 0.0052 | 0.1365 |

Drop of each consistency value:

| Group | R_axis | Z_axis | kappa | tri_top | tri_bot | volume | li |
| --- | --- | --- | --- | --- | --- | --- | --- |
| coil_pattern | 1.0719 | 2.1661 | 1.2303 | 0.4333 | 0.4551 | 0.9434 | 0.9260 |
| plasma_content | 1.0863 | 0.0489 | 0.3252 | 0.0305 | 0.0307 | 0.1799 | 0.1977 |
| field_current | 0.1675 | 0.0509 | 0.1674 | 0.0093 | 0.0094 | 0.0612 | 0.2746 |
| shot_position | 0.0972 | 0.0593 | 0.3679 | 0.0102 | 0.0094 | 0.0458 | 0.3659 |

## MAST sensitivity (label-free)

61 MAST test shots, 904 frames; unpermuted maps: boundary on 1.000 of frames, map RMS 0.07927 Wb/rad; medians R_axis 0.941, Z_axis -0.000, kappa 1.835, R_geo 0.886, a 0.563. No MAST label is read.

| Group | Map change / RMS | Boundary share | Frames with both | median abs change R_axis | median abs change Z_axis | median abs change kappa | median abs change R_geo | median abs change a |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| coil_pattern | 0.061 | 1.000 | 904 | 0.0222 | 0.0039 | 0.0546 | 0.0193 | 0.0149 |
| plasma_content | 0.039 | 1.000 | 904 | 0.0101 | 0.0001 | 0.0143 | 0.0173 | 0.0055 |
| field_current | 0.016 | 1.000 | 904 | 0.0035 | 0.0001 | 0.0259 | 0.0031 | 0.0054 |
| shot_position | 0.021 | 1.000 | 904 | 0.0033 | 0.0001 | 0.0319 | 0.0029 | 0.0056 |

MAST floor (F100, mean_current_v0 on Codabench): S 0.4833, R2 psi 0.7447, D_LCFS 0.263 m.
