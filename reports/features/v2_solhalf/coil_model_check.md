# MAST features under the coil model `sol_half`, beside `unit`

Diagnostic, label-free. MAST public test features of `../data/processed/features/v2` (MAST coil model `unit`) and of `../data/processed/features/v2_solhalf` (`sol_half`), same configuration (`01263fe0c4f1`), against the `train` list of `splits/split_v1.json` (DIII-D, read only). The version's scaler is used for X10 on both.

## 1. Summary: does MAST move toward DIII-D?

`toward` when the distance from DIII-D (median shift, share outside the band, slope ratio minus 1, distance) shrinks under the variant's coil model.

| check | base | variant | moves |
| --- | --- | --- | --- |
| X8 coil_c10: median shift (training IQR) | -1.258 | -1.768 | away |
| X8 coil_c10: share outside the training band | 0.03107 | 0.1298 | away |
| X8 coil_c20: median shift (training IQR) | -3.974 | -3.966 | toward |
| X8 coil_c20: share outside the training band | 0.8742 | 0.874 | unchanged |
| X8 coil_c02: median shift (training IQR) | -2.279 | -3.119 | away |
| X8 coil_c02: share outside the training band | 0.6925 | 0.9521 | away |
| X8 coil_c30: median shift (training IQR) | 1.417 | 1.27 | toward |
| X8 coil_c30: share outside the training band | 0.2718 | 0.2162 | toward |
| X8 coil_c12: median shift (training IQR) | 1.764 | 1.486 | toward |
| X8 coil_c12: share outside the training band | 0.005241 | 0.01325 | away |
| X8 shaping: median shift (training IQR) | 1.586 | 1.112 | toward |
| X8 shaping: share outside the training band | 0.3877 | 0.2429 | toward |
| X8 shafranov_lambda: median shift (training IQR) | -0.2524 | 0.01533 | toward |
| X8 shafranov_lambda: share outside the training band | 0.001607 | 0.001303 | toward |
| X8 vacuum_residual: median shift (training IQR) | -4.199 | -4.352 | away |
| X8 vacuum_residual: share outside the training band | 0.9703 | 0.9742 | away |
| X8 ip_rate_rel: median shift (training IQR) | -0.08656 | -0.08656 | unchanged |
| X8 ip_rate_rel: share outside the training band | 0.1906 | 0.1906 | unchanged |
| X14 slope over DIII-D training (all) | 0.5181 | 0.7487 | toward |
| X14 median offset from the training line (all) | -0.1744 | -0.06732 | toward |
| X14 slope over DIII-D training (flat_top) | 0.6442 | 0.7004 | toward |
| X14 median offset from the training line (flat_top) | -0.1507 | -0.04083 | toward |
| vacuum residual over coil_c20 (DIII-D training 0.028) | 0.08081 | 0.0832 | away |
| X10 set_a: median distance | 0.9607 | 1.084 | away |
| X10 set_b: median distance | 2.713 | 2.377 | toward |

## 2. Columns that do not read the coil currents

Status **pass**: every feature without a coil read is identical in the two MAST tables (8 columns, `ip_rate_rel` among them). Coil features, largest absolute change:

| feature | reads coil currents | identical | largest abs change |
| --- | --- | --- | --- |
| shafranov_lambda | True | False | 1.546 |
| ohmic_flux_used | True | False | 1.159 |
| coil_pt_21 | True | False | 0.1006 |
| coil_pt_19 | True | False | 0.09963 |
| coil_c10 | True | False | 0.09407 |
| coil_pt_13 | True | False | 0.09088 |
| coil_pt_18 | True | False | 0.08861 |
| coil_pt_10 | True | False | 0.08829 |
| coil_pt_14 | True | False | 0.08702 |
| coil_pt_20 | True | False | 0.08049 |
| coil_pt_02 | True | False | 0.06688 |
| coil_pt_17 | True | False | 0.06101 |

## 3. Ranges against DIII-D training (X8), all frames

| feature | train 1st | train 99th | base median | variant median | base outside band | variant outside band | base shift (IQR) | variant shift (IQR) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| coil_pt_00 | 0.03591 | 0.1045 | 0.05236 | 0.05417 | 0.0004922 | 0.0007962 | -2.584 | -2.398 |
| coil_pt_01 | 0.03459 | 0.0715 | 0.0446 | 0.04952 | 0.01181 | 0.003807 | -0.3409 | 0.2183 |
| coil_pt_02 | 0.0261 | 0.06381 | 0.04361 | 0.04886 | 0.005632 | 0.01002 | 0.4012 | 0.8805 |
| coil_pt_03 | 0.03722 | 0.0709 | 0.04465 | 0.04953 | 0.04685 | 0.007195 | -0.6003 | 0.002798 |
| coil_pt_04 | 0.0283 | 0.1037 | 0.05258 | 0.05422 | 0.0002606 | 0.0004922 | -2.231 | -2.096 |
| coil_pt_05 | 0.03553 | 0.07102 | 0.04143 | 0.03864 | 0.2738 | 0.3293 | -1.636 | -1.978 |
| coil_pt_06 | 0.02175 | 0.0444 | 0.02851 | 0.0305 | 0.005646 | 0.004358 | -0.4198 | -0.01314 |
| coil_pt_07 | 0.01735 | 0.03598 | 0.02646 | 0.02946 | 0.008759 | 0.02112 | 0.6056 | 1.166 |
| coil_pt_08 | 0.02289 | 0.04435 | 0.02862 | 0.0305 | 0.02031 | 0.01317 | -0.7116 | -0.3242 |
| coil_pt_09 | 0.01856 | 0.07542 | 0.04164 | 0.03865 | 0.008599 | 0.001853 | -1.397 | -1.698 |
| coil_pt_10 | 0.01086 | 0.04004 | 0.01414 | 0.00818 | 0.3355 | 0.7047 | -1.257 | -2.034 |
| coil_pt_11 | 0.002288 | 0.009791 | 0.001158 | 8.093e-05 | 0.7573 | 0.9676 | -2.046 | -2.58 |
| coil_pt_12 | 0.002906 | 0.01174 | 0.001199 | 6.514e-05 | 0.8004 | 0.984 | -2.321 | -2.809 |
| coil_pt_13 | 0.01097 | 0.04069 | 0.01413 | 0.008061 | 0.3491 | 0.7 | -1.307 | -2.05 |
| coil_pt_14 | -0.03807 | -0.004854 | -0.03724 | -0.04285 | 0.4436 | 0.851 | -1.726 | -2.397 |
| coil_pt_15 | -0.04181 | -0.01905 | -0.03821 | -0.04274 | 0.2137 | 0.585 | -1.719 | -2.453 |
| coil_pt_16 | -0.04465 | -0.02431 | -0.03538 | -0.03865 | 0.01902 | 0.06163 | -0.8962 | -1.456 |
| coil_pt_17 | -0.04166 | -0.01701 | -0.03798 | -0.04271 | 0.2078 | 0.5985 | -1.911 | -2.634 |
| coil_pt_18 | -0.03758 | -0.00261 | -0.03712 | -0.04287 | 0.4677 | 0.8712 | -2.1 | -2.803 |
| coil_pt_19 | -0.09954 | -0.05211 | -0.09157 | -0.09966 | 0.1956 | 0.5044 | -1.924 | -2.572 |
| coil_pt_20 | -0.09668 | -0.05355 | -0.07625 | -0.08262 | 0.01879 | 0.05083 | -0.8133 | -1.314 |
| coil_pt_21 | -0.09938 | -0.04952 | -0.09125 | -0.09955 | 0.1865 | 0.5071 | -2.001 | -2.632 |
| coil_c10 | -0.1087 | -0.05532 | -0.09013 | -0.09765 | 0.03107 | 0.1298 | -1.258 | -1.768 |
| coil_c01 | -0.003477 | 0.005633 | -5.601e-06 | -4.26e-05 | 0.0003474 | 0.0003764 | -0.5576 | -0.5753 |
| coil_c20 | -0.03578 | -0.01967 | -0.0427 | -0.04267 | 0.8742 | 0.874 | -3.974 | -3.966 |
| coil_c11 | -0.007081 | 0.01479 | -5.174e-05 | -6.681e-05 | 5.791e-05 | 5.791e-05 | 0.05408 | 0.0504 |
| coil_c02 | 0.005897 | 0.01585 | 0.004203 | 0.001748 | 0.6925 | 0.9521 | -2.279 | -3.119 |
| coil_c30 | -0.01334 | 0.00979 | 0.007997 | 0.007299 | 0.2718 | 0.2162 | 1.417 | 1.27 |
| coil_c21 | -0.006408 | 0.01198 | -4.808e-06 | 1.511e-05 | 1.448e-05 | 1.448e-05 | -0.02046 | -0.01456 |
| coil_c12 | -0.01456 | 0.02515 | -0.002351 | -0.003388 | 0.005241 | 0.01325 | 1.764 | 1.486 |
| coil_c03 | -0.004281 | 0.002789 | 2.346e-05 | 9.68e-06 | 0 | 0 | 0.3773 | 0.3672 |
| shaping | 0.01295 | 0.02495 | 0.0235 | 0.02206 | 0.3877 | 0.2429 | 1.586 | 1.112 |
| ts_beta_p | 0.01681 | 0.3651 | 0.1023 | 0.1023 | 0.011 | 0.011 | -0.04556 | -0.04556 |
| ts_pe_peaking | 1.789 | 3.738 | 2.092 | 2.092 | 0.08518 | 0.08518 | -0.8879 | -0.8879 |
| ts_valid | 0 | 1 | 1 | 1 | 0 | 0 | nan | nan |
| q_cyl | 1.588 | 5.926 | 1.807 | 1.807 | 0.2826 | 0.2826 | -1.015 | -1.015 |
| ip_frac | 0.3638 | 0.9981 | 0.9732 | 0.9732 | 0.1259 | 0.1259 | -0.1701 | -0.1701 |
| ip_rate_rel | -9.287 | 7.319 | -0.0181 | -0.0181 | 0.1906 | 0.1906 | -0.08656 | -0.08656 |
| resistive_age | 0.1526 | 2.804 | 0.1808 | 0.1808 | 0.3726 | 0.3726 | -0.9462 | -0.9462 |
| shafranov_lambda | 0.24 | 1.913 | 0.6333 | 0.7569 | 0.001607 | 0.001303 | -0.2524 | 0.01533 |
| ohmic_flux_used | -0.5294 | -0.1001 | -0.2111 | -0.1056 | 0.03296 | 0.4271 | -0.2528 | 0.9411 |
| greenwald_frac | 0.1198 | 1.283 | 0.6066 | 0.6066 | 0.03826 | 0.03826 | 0.1466 | 0.1466 |
| vacuum_residual | -0.002066 | 0.0003948 | -0.003451 | -0.00355 | 0.9703 | 0.9742 | -4.199 | -4.352 |

### By phase (X9 rule), even coil terms and the vacuum residual

| feature | phase | MAST frames | base outside band | variant outside band | base shift (IQR) | variant shift (IQR) |
| --- | --- | --- | --- | --- | --- | --- |
| coil_c10 | rising | 8971 | 0.01984 | 0.01583 | -1.674 | -1.638 |
| coil_c10 | flat_top | 51915 | 0.0393 | 0.1601 | -1.228 | -1.738 |
| coil_c10 | late | 8189 | 0.01368 | 0.06509 | -0.7291 | -1.6 |
| coil_c20 | rising | 8971 | 0.5274 | 0.5274 | -2.826 | -2.829 |
| coil_c20 | flat_top | 51915 | 0.9382 | 0.9378 | -4.086 | -4.077 |
| coil_c20 | late | 8189 | 0.8123 | 0.8085 | -3.313 | -3.297 |
| coil_c02 | rising | 8971 | 0.992 | 0.9991 | -3.884 | -3.654 |
| coil_c02 | flat_top | 51915 | 0.7092 | 0.9795 | -2.329 | -3.13 |
| coil_c02 | late | 8189 | 0.1038 | 0.6569 | -0.3627 | -2.36 |
| coil_c30 | rising | 8971 | 0.2814 | 0.2859 | 1.058 | 1.076 |
| coil_c30 | flat_top | 51915 | 0.2113 | 0.1482 | 1.404 | 1.261 |
| coil_c30 | late | 8189 | 0.4397 | 0.3506 | 1.98 | 1.719 |
| coil_c12 | rising | 8971 | 0.001115 | 0.0008918 | 1.514 | 1.661 |
| coil_c12 | flat_top | 51915 | 0 | 0 | 1.769 | 1.478 |
| coil_c12 | late | 8189 | 0.01246 | 0.05849 | 1.203 | 0.7474 |
| shaping | rising | 8971 | 0.03077 | 0.02307 | -0.08405 | -0.005382 |
| shaping | flat_top | 51915 | 0.4596 | 0.3183 | 1.658 | 1.203 |
| shaping | late | 8189 | 0.3703 | 0.1272 | 2.163 | 1.436 |
| shafranov_lambda | rising | 8971 | 0.0002229 | 0.0005574 | -0.1809 | -0.1998 |
| shafranov_lambda | flat_top | 51915 | 0.0004238 | 9.631e-05 | -0.2761 | -0.008148 |
| shafranov_lambda | late | 8189 | 0.00635 | 0.002442 | -0.3376 | 0.1193 |
| vacuum_residual | rising | 8971 | 0.9605 | 0.9609 | -4.222 | -4.176 |
| vacuum_residual | flat_top | 51915 | 0.9978 | 0.9989 | -4.21 | -4.375 |
| vacuum_residual | late | 8189 | 0.7494 | 0.7622 | -3.811 | -4.143 |

## 4. Stretch terms and the vacuum relation (X9, F82)

| quantity | split | n | median | IQR | outside band | median shift (IQR) |
| --- | --- | --- | --- | --- | --- | --- |
| coil_c20 | DIII-D train | 1289467 | -0.02619 | 0.004155 | nan | nan |
| coil_c20 | MAST, unit | 69075 | -0.0427 | 0.008262 | 0.8742 | -3.974 |
| coil_c20 | MAST, sol_half | 69075 | -0.04267 | 0.008235 | 0.874 | -3.966 |
| coil_c02 | DIII-D train | 1289467 | 0.01086 | 0.002921 | nan | nan |
| coil_c02 | MAST, unit | 69075 | 0.004203 | 0.004593 | 0.6925 | -2.279 |
| coil_c02 | MAST, sol_half | 69075 | 0.001748 | 0.002738 | 0.9521 | -3.119 |
| shaping | DIII-D train | 1289467 | 0.0187 | 0.003027 | nan | nan |
| shaping | MAST, unit | 69075 | 0.0235 | 0.006335 | 0.3877 | 1.586 |
| shaping | MAST, sol_half | 69075 | 0.02206 | 0.005308 | 0.2429 | 1.112 |
| vacuum_residual | DIII-D train | 1289467 | -0.0007309 | 0.0006477 | nan | nan |
| vacuum_residual | MAST, unit | 69075 | -0.003451 | 0.0007356 | 0.9703 | -4.199 |
| vacuum_residual | MAST, sol_half | 69075 | -0.00355 | 0.000851 | 0.9742 | -4.352 |

Median abs(`vacuum_residual`) over median abs(`coil_c20`), beside the current-free field R^4 - 4 R^2 Z^2 fitted the same way at the same radii:

| split | phase | n | ratio | ratio for R^4 - 4 R^2 Z^2 |
| --- | --- | --- | --- | --- |
| DIII-D train | all | 1289467 | 0.02795 | 0.0258 |
| DIII-D train | rising | 234424 | 0.03074 | 0.0258 |
| DIII-D train | flat_top | 1002719 | 0.02726 | 0.0258 |
| DIII-D train | late | 52324 | 0.02674 | 0.0258 |
| MAST, base | all | 69075 | 0.08081 | 0.0606 |
| MAST, base | rising | 8971 | 0.08412 | 0.0606 |
| MAST, base | flat_top | 51915 | 0.08147 | 0.0606 |
| MAST, base | late | 8189 | 0.07418 | 0.0606 |
| MAST, variant | all | 69075 | 0.0832 | 0.0606 |
| MAST, variant | rising | 8971 | 0.08342 | 0.0606 |
| MAST, variant | flat_top | 51915 | 0.08407 | 0.0606 |
| MAST, variant | late | 8189 | 0.07955 | 0.0606 |

## 5. Thomson against force balance (X14, F85)

Line `shafranov_lambda = intercept + slope ts_beta_p` over frames inside both 1st to 99th percentile bands of each split; median offset from the DIII-D training line of the same frames.

| frames | split | n | slope | slope over training | intercept | r | residual std | median offset from training line |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | DIII-D train | 1225650 | 2.964 | 1 | 0.4735 | 0.7355 | 0.2285 | -0.0063 |
| all | MAST, unit | 65972 | 1.536 | 0.5181 | 0.4655 | 0.6066 | 0.1091 | -0.1744 |
| all | MAST, sol_half | 66045 | 2.219 | 0.7487 | 0.5026 | 0.7193 | 0.1157 | -0.06732 |
| flat_top | DIII-D train | 956349 | 3.043 | 1 | 0.4363 | 0.7462 | 0.2217 | -0.0003344 |
| flat_top | MAST, unit | 49590 | 1.96 | 0.6442 | 0.4177 | 0.7149 | 0.09228 | -0.1507 |
| flat_top | MAST, sol_half | 49567 | 2.131 | 0.7004 | 0.5086 | 0.6969 | 0.1059 | -0.04083 |

## 6. Combined distance (X10)

200000 DIII-D training frames (seed 0); DIII-D validation 5000, MAST, unit 5000, MAST, sol_half 5000 frames (both MAST tables on the same frames).

| set | split | n | median | 90th | 99th | max | beyond validation 99th |
| --- | --- | --- | --- | --- | --- | --- | --- |
| set_a | DIII-D validation | 5000 | 0.1941 | 0.3739 | 0.8065 | 19.38 | 0.01 |
| set_a | MAST, unit | 5000 | 0.9607 | 1.276 | 2.228 | 6.773 | 0.8642 |
| set_a | MAST, sol_half | 5000 | 1.084 | 1.318 | 2.37 | 6.726 | 0.9982 |
| set_b | DIII-D validation | 5000 | 0.3709 | 0.809 | 1.841 | 19.91 | 0.01 |
| set_b | MAST, unit | 5000 | 2.713 | 4.161 | 8.005 | 21.14 | 0.9584 |
| set_b | MAST, sol_half | 5000 | 2.377 | 3.264 | 6.849 | 19.63 | 0.9538 |

Median share of the squared distance per feature group:

| set | split | coil_pattern | plasma_content | field_current | shot_position | coil_terms |
| --- | --- | --- | --- | --- | --- | --- |
| set_a | DIII-D validation | 0.2081 | 0.2355 | 0.02231 | 0.3106 | nan |
| set_a | MAST, unit | 0.7345 | 0.08167 | 0.02224 | 0.1128 | nan |
| set_a | MAST, sol_half | 0.7483 | 0.09171 | 0.02001 | 0.09595 | nan |
| set_b | DIII-D validation | nan | 0.1554 | 0.004894 | 0.2145 | 0.4552 |
| set_b | MAST, unit | nan | 0.04256 | 0.05601 | 0.07633 | 0.7619 |
| set_b | MAST, sol_half | nan | 0.05118 | 0.03862 | 0.08647 | 0.7625 |

Runtime 10 s.
