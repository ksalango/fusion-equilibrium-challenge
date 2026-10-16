# Evaluation of run arm_d8_g_s0

Validation list `validation` (870 shots, 48186 frames at stride 4); chain: checkpoints arm_d8_g_s0, mirror averaging on, q95 from the map, betaN from the DIII-D raw-input ridge (fitted on `train` at stride 4). Corner: q_cyl < 1.780 and ts_beta_p < 0.1088, 2356 frames.

MAST floor (F100, mean_current_v0 on Codabench): S 0.4833, R2 psi 0.7447, D_LCFS 0.263 m.

## Checks

| Id | Check | Status |
| --- | --- | --- |
| Z4 | Warm start | review: trained at or above the warm start (all 0.9765 against 0.9190, no near-repeat 0.9692 against 0.9130) |
| Z7 | Slices present | pass |
| Z8 | Clip rates | review: largest share 0.0006 |
| Z11 | Mirror averaging | review: on 0.9765, off 0.9758 |
| Z12 | MAST sign | pass (asserted before the MAST maps) |

## Score terms by slice

Identity: `0.55 max(0, R2 psi) + 0.15 max(0, R2 scalars) + 0.10 (1 - D_LCFS) + 0.20 consistency`, beside the scorer's S.

| Slice | Shots | S | Identity | R2 psi | R2 psi per shot | R2 q95 | R2 betaN | 1 - D_LCFS | Consistency | R_axis | Z_axis | kappa | tri_top | tri_bot | volume | li |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 870 | 0.9765 | 0.9765 | 0.9995 | 0.9992 | 0.9447 | 0.9338 | 0.9838 | 0.9373 | 0.9333 | 0.9497 | 0.8818 | 0.9817 | 0.9769 | 0.9605 | 0.8774 |
| no_near_repeat | 286 | 0.9692 | 0.9692 | 0.9994 | 0.9986 | 0.9305 | 0.9294 | 0.9801 | 0.9103 | 0.9024 | 0.9248 | 0.8361 | 0.9809 | 0.9696 | 0.9517 | 0.8066 |
| ip_negative | 74 | 0.9568 | 0.9568 | 0.9995 | 0.9994 | 0.9525 | 0.8258 | 0.9804 | 0.8782 | 0.7918 | 0.9131 | 0.7856 | 0.9476 | 0.9555 | 0.9293 | 0.8246 |
| corner | 151 | 0.9668 | 0.9668 | 0.9994 | 0.9994 | 0.9717 | 0.8038 | 0.9881 | 0.9259 | 0.8852 | 0.9794 | 0.8201 | 0.9837 | 0.9781 | 0.9253 | 0.9094 |

MAST floor (F100, mean_current_v0 on Codabench): S 0.4833, R2 psi 0.7447, D_LCFS 0.263 m.

## Score terms by phase

| Slice | Shots | S | Identity | R2 psi | R2 psi per shot | R2 q95 | R2 betaN | 1 - D_LCFS | Consistency | R_axis | Z_axis | kappa | tri_top | tri_bot | volume | li |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase_rising | 858 | 0.9658 | 0.9658 | 0.9990 | 0.9981 | 0.9278 | 0.9474 | 0.9820 | 0.8876 | 0.8918 | 0.8590 | 0.8148 | 0.9798 | 0.9745 | 0.9053 | 0.7882 |
| phase_flat_top | 862 | 0.9827 | 0.9827 | 0.9996 | 0.9993 | 0.9839 | 0.9297 | 0.9853 | 0.9544 | 0.9425 | 0.9715 | 0.9027 | 0.9839 | 0.9790 | 0.9720 | 0.9294 |
| phase_late | 427 | 0.9180 | 0.9180 | 0.9986 | 0.9963 | 0.5940 | 0.7360 | 0.9686 | 0.8611 | 0.8187 | 0.9261 | 0.7514 | 0.9454 | 0.9502 | 0.9440 | 0.6917 |
| no_near_repeat_phase_rising | 281 | 0.9632 | 0.9632 | 0.9989 | 0.9981 | 0.9220 | 0.9540 | 0.9789 | 0.8760 | 0.8929 | 0.8127 | 0.7919 | 0.9807 | 0.9674 | 0.8917 | 0.7948 |
| no_near_repeat_phase_flat_top | 279 | 0.9752 | 0.9752 | 0.9993 | 0.9990 | 0.9742 | 0.9114 | 0.9832 | 0.9292 | 0.9022 | 0.9605 | 0.8605 | 0.9833 | 0.9719 | 0.9645 | 0.8617 |
| no_near_repeat_phase_late | 151 | 0.9227 | 0.9227 | 0.9981 | 0.9963 | 0.5800 | 0.8642 | 0.9668 | 0.8438 | 0.8313 | 0.9162 | 0.6855 | 0.9473 | 0.9507 | 0.9443 | 0.6313 |

MAST floor (F100, mean_current_v0 on Codabench): S 0.4833, R2 psi 0.7447, D_LCFS 0.263 m.

## Worst 20 validation shots by R2 psi

| Shot | R2 psi | D_LCFS | Boundary failures |
| --- | --- | --- | --- |
| d42a0ed0c0 | 0.8484 | 0.4802 | 0 of 1 |
| ec216f0d01 | 0.9900 | 0.0346 | 0 of 12 |
| 316f541fd6 | 0.9907 | 0.0308 | 0 of 1 |
| 269e33625f | 0.9912 | 0.0183 | 0 of 22 |
| b6f9f193bf | 0.9922 | 0.0575 | 0 of 11 |
| b2df6a7008 | 0.9949 | 0.0341 | 0 of 17 |
| e2932f4823 | 0.9962 | 0.0510 | 0 of 2 |
| 2427c80a5d | 0.9965 | 0.0355 | 0 of 65 |
| 533cba65c5 | 0.9965 | 0.0286 | 0 of 53 |
| 1b1a5c5531 | 0.9967 | 0.0227 | 0 of 38 |
| 2b9b50b9b4 | 0.9970 | 0.0176 | 0 of 75 |
| 4c146d6cd7 | 0.9970 | 0.0172 | 0 of 28 |
| 4274201181 | 0.9971 | 0.0185 | 0 of 19 |
| ab04959aa2 | 0.9972 | 0.0187 | 0 of 81 |
| ced938a97c | 0.9972 | 0.0254 | 0 of 37 |
| a2ae4f3c2b | 0.9972 | 0.0233 | 0 of 71 |
| 6208fa6b91 | 0.9973 | 0.0169 | 0 of 26 |
| 8496e7cfe6 | 0.9973 | 0.0282 | 0 of 48 |
| 59fabf2dd5 | 0.9973 | 0.0154 | 0 of 30 |
| e365322b54 | 0.9974 | 0.0344 | 0 of 76 |

## Overfit panel

| Epoch | Train loss | Validation loss | No-near-repeat loss |
| --- | --- | --- | --- |
| 177 (best epoch) | 0.16341 | 0.14117 | 0.20558 |
| 192 (stopped epoch) | 0.16291 | 0.14114 | 0.20572 |

Trained: S all 0.9765, no near-repeat 0.9692, gap 0.0073.

Warm start (ridge): S all 0.9190, no near-repeat 0.9130, gap 0.0060.

Seed spread of S over 2 run(s): mean 0.9763, standard deviation 0.0002, range 0.9762 to 0.9765.
- arm_d8_g_s0 (seed 0): S 0.9765, no near-repeat 0.9692
- arm_d8_g_s1 (seed 1): S 0.9762, no near-repeat 0.9688

## Warm start beside the trained model

Prediction chain (warm-start checkpoint `warm.pt` against the best checkpoint; q95 from the map, betaN from the DIII-D head, mirror averaging on):

| Slice | Warm start | Trained | Difference |
| --- | --- | --- | --- |
| all | 0.9190 | 0.9765 | 0.0574 |
| no_near_repeat | 0.9130 | 0.9692 | 0.0562 |
| ip_negative | 0.8859 | 0.9568 | 0.0709 |
| corner | 0.8749 | 0.9668 | 0.0919 |

The run record holds no training-time warm-start or best score (trained with --no-score).

Chain score, mirror averaging on: S 0.9765 (no near-repeat 0.9692); off: S 0.9758 (no near-repeat 0.9681).

## Clip rates

Share of scaled values beyond 10 per column.

DIII-D validation (191411 frames): `q_cyl` 0.0006, `ts_beta_p` 0.0000, `coil_pt_00` 0.0000, `coil_pt_04` 0.0000, `coil_pt_05` 0.0000, `coil_pt_19` 0.0000, `coil_pt_20` 0.0000, `coil_pt_21` 0.0000.

DIII-D test (193359 frames): `q_cyl` 0.0004.

MAST test (69075 frames): `q_cyl` 0.0002, `ts_beta_p` 0.0000.

## Grouped permutation importance (validation, all shots)

| Group | Columns | S | drop S | drop r2_psi | drop r2_qb | drop one_minus_dlcfs | drop consistency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| coil_pattern | 22 | 0.7927 | 0.1838 | 0.0138 | 0.0789 | 0.0712 | 0.7863 |
| plasma_content | 3 | 0.9222 | 0.0543 | 0.0031 | 0.0039 | 0.0122 | 0.2538 |
| field_current | 1 | 0.9470 | 0.0295 | 0.0009 | 0.0154 | 0.0049 | 0.1310 |

Drop of each consistency value:

| Group | R_axis | Z_axis | kappa | tri_top | tri_bot | volume | li |
| --- | --- | --- | --- | --- | --- | --- | --- |
| coil_pattern | 1.1201 | 2.1623 | 1.7057 | 0.4399 | 0.4615 | 0.9925 | 1.3928 |
| plasma_content | 1.3172 | 0.0591 | 0.2922 | 0.0301 | 0.0290 | 0.2181 | 0.2148 |
| field_current | 0.1832 | 0.0518 | 0.2361 | 0.0108 | 0.0089 | 0.0569 | 0.3693 |

## MAST sensitivity (label-free)

61 MAST test shots, 904 frames; unpermuted maps: boundary on 1.000 of frames, map RMS 0.07857 Wb/rad; medians R_axis 0.954, Z_axis -0.000, kappa 1.756, R_geo 0.896, a 0.580. No MAST label is read.

| Group | Map change / RMS | Boundary share | Frames with both | median abs change R_axis | median abs change Z_axis | median abs change kappa | median abs change R_geo | median abs change a |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| coil_pattern | 0.053 | 1.000 | 904 | 0.0186 | 0.0026 | 0.0587 | 0.0157 | 0.0128 |
| plasma_content | 0.034 | 1.000 | 904 | 0.0065 | 0.0001 | 0.0200 | 0.0118 | 0.0071 |
| field_current | 0.017 | 1.000 | 904 | 0.0024 | 0.0002 | 0.0275 | 0.0025 | 0.0061 |

MAST floor (F100, mean_current_v0 on Codabench): S 0.4833, R2 psi 0.7447, D_LCFS 0.263 m.
