# MAST features at backup reference radii, version 1

Diagnostic. MAST public test features built with R0 0.7 m and a0 0.5 m (Morris 1998) passed to the build as an override, beside the version-1 MAST features (R0 0.85 m, a0 0.65 m, unverified in `machines.py`). Version 1 is unchanged: `machines.py`, the configuration, the version-1 feature tables and `scaler.json`; DIII-D is not rebuilt. Reference of every comparison: the `train` list of `splits/split_v1.json`.

Features that change with the radii: the coil points (moved, and divided by mu0 R0 Ip) and the named terms; `ts_beta_p`, `greenwald_frac` and the resistive time (`ip_rate_rel`, `resistive_age`) through a0^2; `q_cyl` through a0^2 / R0^2; `shafranov_lambda` through R0 / a0; `ohmic_flux_used` through R0 and the reference point. `ip_frac`, `ts_pe_peaking` and `ts_valid` do not depend on them.

## 1. Layout clearance of MAST at the backup radii (X4)

Status **pass**: smallest clearance 0.162 m at the backup radii, 0.199 m at the version-1 radii (rule 0.15 m); layout symmetric True; failing points 0. Geometry from `../data/raw/data/mast_public_test/mast_shot_0035568738.parquet`. Smallest clearances:

| column | x | y | R_m | Z_m | clearance_m | nearest_column | in_grid | ok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| coil_pt_00 | -0.75 | -1.5 | 0.325 | -0.75 | 0.1618 | magnetics_sol_current | True | True |
| coil_pt_01 | -0.75 | -0.75 | 0.325 | -0.375 | 0.1618 | magnetics_sol_current | True | True |
| coil_pt_02 | -0.75 | 0 | 0.325 | 0 | 0.1618 | magnetics_sol_current | True | True |
| coil_pt_03 | -0.75 | 0.75 | 0.325 | 0.375 | 0.1618 | magnetics_sol_current | True | True |
| coil_pt_04 | -0.75 | 1.5 | 0.325 | 0.75 | 0.1618 | magnetics_sol_current | True | True |
| coil_pt_05 | -0.375 | -1.5 | 0.5125 | -0.75 | 0.3493 | magnetics_sol_current | True | True |

## 2. Build

1206 MAST shots, 69075 rows, keys equal the frame table: True. Features in `../data/processed/features/scratch/mast_r0.70_a0.50/mast_public_test.parquet` (float32, as version 1). Point table `points_mast_e9ad71ba8c61a335.npz` (built), points hash `5ddbe94efba0`.

## 3. Ranges against DIII-D training (X8), all frames

Share of frames outside the training 1st to 99th percentile band and outside the training range, and the median shift in training interquartile ranges, for MAST at both radii. `vacuum_residual` uses each build's own radii.

| feature | train 1st | train 99th | v1 median | backup median | v1 outside band | backup outside band | v1 outside range | backup outside range | v1 shift (IQR) | backup shift (IQR) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| coil_pt_00 | 0.03591 | 0.1045 | 0.05236 | 0.03762 | 0.0004922 | 0.3092 | 0 | 0 | -2.584 | -4.102 |
| coil_pt_01 | 0.03459 | 0.0715 | 0.0446 | 0.0346 | 0.01181 | 0.4989 | 0.0003185 | 0.0004922 | -0.3409 | -1.478 |
| coil_pt_02 | 0.0261 | 0.06381 | 0.04361 | 0.03407 | 0.005632 | 0.02952 | 0.0006949 | 0.003779 | 0.4012 | -0.4686 |
| coil_pt_03 | 0.03722 | 0.0709 | 0.04465 | 0.03462 | 0.04685 | 0.7509 | 0.000333 | 0.0005646 | -0.6003 | -1.838 |
| coil_pt_04 | 0.0283 | 0.1037 | 0.05258 | 0.03774 | 0.0002606 | 0.001795 | 0 | 0 | -2.231 | -3.447 |
| coil_pt_05 | 0.03553 | 0.07102 | 0.04143 | 0.02728 | 0.2738 | 0.9666 | 0 | 0 | -1.636 | -3.371 |
| coil_pt_06 | 0.02175 | 0.0444 | 0.02851 | 0.0214 | 0.005646 | 0.5655 | 8.686e-05 | 0.0001737 | -0.4198 | -1.874 |
| coil_pt_07 | 0.01735 | 0.03598 | 0.02646 | 0.02034 | 0.008759 | 0.08712 | 0.0006949 | 0.005038 | 0.6056 | -0.5386 |
| coil_pt_08 | 0.02289 | 0.04435 | 0.02862 | 0.02146 | 0.02031 | 0.7437 | 0.0001303 | 0.0002316 | -0.7116 | -2.184 |
| coil_pt_09 | 0.01856 | 0.07542 | 0.04164 | 0.02738 | 0.008599 | 0.002591 | 0 | 0 | -1.397 | -2.832 |
| coil_pt_10 | 0.01086 | 0.04004 | 0.01414 | 0.007663 | 0.3355 | 0.7394 | 0 | 0 | -1.257 | -2.101 |
| coil_pt_11 | 0.002288 | 0.009791 | 0.001158 | 0.001095 | 0.7573 | 0.8489 | 0.000304 | 0 | -2.046 | -2.078 |
| coil_pt_12 | 0.002906 | 0.01174 | 0.001199 | 0.001127 | 0.8004 | 0.8828 | 0.0001448 | 0 | -2.321 | -2.352 |
| coil_pt_13 | 0.01097 | 0.04069 | 0.01413 | 0.00772 | 0.3491 | 0.7254 | 0.0006804 | 0.0002895 | -1.307 | -2.092 |
| coil_pt_14 | -0.03807 | -0.004854 | -0.03724 | -0.02533 | 0.4436 | 0.007673 | 0.0001448 | 0.0001737 | -1.726 | -0.3003 |
| coil_pt_15 | -0.04181 | -0.01905 | -0.03821 | -0.02686 | 0.2137 | 0.02366 | 0.0002461 | 0.0003619 | -1.719 | 0.1172 |
| coil_pt_16 | -0.04465 | -0.02431 | -0.03538 | -0.02707 | 0.01902 | 0.1608 | 0.0005791 | 0.004792 | -0.8962 | 0.5215 |
| coil_pt_17 | -0.04166 | -0.01701 | -0.03798 | -0.02674 | 0.2078 | 0.01041 | 0.0002172 | 0.0003474 | -1.911 | -0.1923 |
| coil_pt_18 | -0.03758 | -0.00261 | -0.03712 | -0.02522 | 0.4677 | 0.008281 | 7.239e-05 | 5.791e-05 | -2.1 | -0.6468 |
| coil_pt_19 | -0.09954 | -0.05211 | -0.09157 | -0.06276 | 0.1956 | 0.04608 | 0.0001737 | 0.0003764 | -1.924 | 0.3831 |
| coil_pt_20 | -0.09668 | -0.05355 | -0.07625 | -0.06041 | 0.01879 | 0.1285 | 0.0004777 | 0.00194 | -0.8133 | 0.4309 |
| coil_pt_21 | -0.09938 | -0.04952 | -0.09125 | -0.06252 | 0.1865 | 0.01814 | 0.0001882 | 0.0003909 | -2.001 | 0.1824 |
| coil_c10 | -0.1087 | -0.05532 | -0.09013 | -0.06545 | 0.03107 | 0.05497 | 0.0002606 | 0.000637 | -1.258 | 0.4169 |
| coil_c01 | -0.003477 | 0.005633 | -5.601e-06 | 1.891e-05 | 0.0003474 | 4.343e-05 | 0 | 0 | -0.5576 | -0.546 |
| coil_c20 | -0.03578 | -0.01967 | -0.0427 | -0.02738 | 0.8742 | 0.01207 | 0 | 2.895e-05 | -3.974 | -0.2857 |
| coil_c11 | -0.007081 | 0.01479 | -5.174e-05 | 6.061e-06 | 5.791e-05 | 1.448e-05 | 0 | 0 | 0.05408 | 0.06823 |
| coil_c02 | 0.005897 | 0.01585 | 0.004203 | 0.00275 | 0.6925 | 0.874 | 0.0003764 | 0.0001448 | -2.279 | -2.776 |
| coil_c30 | -0.01334 | 0.00979 | 0.007997 | 0.002045 | 0.2718 | 0.01244 | 0.01328 | 0 | 1.417 | 0.1654 |
| coil_c21 | -0.006408 | 0.01198 | -4.808e-06 | -1.599e-05 | 1.448e-05 | 0 | 0 | 0 | -0.02046 | -0.02377 |
| coil_c12 | -0.01456 | 0.02515 | -0.002351 | 0.0006755 | 0.005241 | 1.448e-05 | 0 | 0 | 1.764 | 2.578 |
| coil_c03 | -0.004281 | 0.002789 | 2.346e-05 | 1.041e-05 | 0 | 0 | 0 | 0 | 0.3773 | 0.3677 |
| ts_beta_p | 0.01681 | 0.3651 | 0.1023 | 0.06051 | 0.011 | 0.01395 | 0 | 0 | -0.04556 | -0.3352 |
| ts_pe_peaking | 1.789 | 3.738 | 2.092 | 2.092 | 0.08518 | 0.08518 | 0 | 0 | -0.8879 | -0.8879 |
| ts_valid | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | nan | nan |
| q_cyl | 1.588 | 5.926 | 1.807 | 1.577 | 0.2826 | 0.5164 | 0 | 0 | -1.015 | -1.286 |
| ip_frac | 0.3638 | 0.9981 | 0.9732 | 0.9732 | 0.1259 | 0.1259 | 2.895e-05 | 2.895e-05 | -0.1701 | -0.1701 |
| ip_rate_rel | -9.287 | 7.319 | -0.0181 | -0.01071 | 0.1906 | 0.08785 | 0 | 0 | -0.08656 | -0.08136 |
| resistive_age | 0.1526 | 2.804 | 0.1808 | 0.3056 | 0.3726 | 0.08123 | 1.449e-05 | 1.449e-05 | -0.9462 | -0.7571 |
| shafranov_lambda | 0.24 | 1.913 | 0.6333 | 0.2356 | 0.001607 | 0.5142 | 0.0001882 | 0.0004488 | -0.2524 | -1.114 |
| ohmic_flux_used | -0.5294 | -0.1001 | -0.2111 | -0.2673 | 0.03296 | 0.07945 | 0.0001303 | 0.0001158 | -0.2528 | -0.888 |
| greenwald_frac | 0.1198 | 1.283 | 0.6066 | 0.359 | 0.03826 | 0.01735 | 0 | 0.001154 | 0.1466 | -0.4828 |
| vacuum_residual | -0.002066 | 0.0003948 | -0.003451 | -0.0007046 | 0.9703 | 0 | 2.895e-05 | 0 | -4.199 | 0.04059 |
| shaping | 0.01295 | 0.02495 | 0.0235 | 0.01514 | 0.3877 | 0.199 | 0 | 0 | 1.586 | -1.174 |

### By phase

Reference restricted to the same phase (X9 rule).

| feature | phase | MAST frames | v1 outside band | backup outside band | v1 outside range | backup outside range | v1 shift (IQR) | backup shift (IQR) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| coil_pt_00 | rising | 8971 | 0.8132 | 0.9955 | 0 | 0 | -3.184 | -4.336 |
| coil_pt_00 | flat_top | 51915 | 5.779e-05 | 0.2324 | 0 | 0 | -2.578 | -4.104 |
| coil_pt_00 | late | 8189 | 0 | 0.0006106 | 0 | 0 | -1.653 | -3.203 |
| coil_pt_01 | rising | 8971 | 0.01315 | 0.5217 | 0.0001115 | 0 | -0.06053 | -1.304 |
| coil_pt_01 | flat_top | 51915 | 0.001772 | 0.4402 | 0 | 0 | -0.3381 | -1.493 |
| coil_pt_01 | late | 8189 | 0.0635 | 0.7672 | 0.002687 | 0.004152 | -0.7102 | -1.586 |
| coil_pt_02 | rising | 8971 | 0.004905 | 0.002006 | 0.0002229 | 0 | 0.9607 | -0.1902 |
| coil_pt_02 | flat_top | 51915 | 0.001368 | 0.003583 | 3.852e-05 | 0 | 0.3753 | -0.4784 |
| coil_pt_02 | late | 8189 | 0.02393 | 0.2054 | 0.005862 | 0.03187 | -0.1349 | -0.7684 |
| coil_pt_03 | rising | 8971 | 0.06588 | 0.7389 | 0.0002229 | 0 | -0.3847 | -1.723 |
| coil_pt_03 | flat_top | 51915 | 0.02498 | 0.7553 | 0 | 9.631e-05 | -0.5911 | -1.849 |
| coil_pt_03 | late | 8189 | 0.0762 | 0.809 | 0.002809 | 0.004762 | -0.8204 | -1.633 |
| coil_pt_04 | rising | 8971 | 0 | 0.04214 | 0 | 0 | -2.595 | -3.46 |
| coil_pt_04 | flat_top | 51915 | 0 | 0.0005779 | 0 | 0 | -2.241 | -3.474 |
| coil_pt_04 | late | 8189 | 0 | 0.001465 | 0 | 0 | -1.529 | -2.817 |
| coil_pt_05 | rising | 8971 | 0.9671 | 1 | 0 | 0 | -3.137 | -4.257 |
| coil_pt_05 | flat_top | 51915 | 0.2388 | 0.998 | 0 | 0 | -1.603 | -3.312 |
| coil_pt_05 | late | 8189 | 0.03138 | 0 | 0 | 0 | -0.2139 | -2.448 |
| coil_pt_06 | rising | 8971 | 0.05886 | 0.8088 | 0 | 0 | -0.4984 | -1.785 |
| coil_pt_06 | flat_top | 51915 | 0.001214 | 0.4565 | 0 | 0 | -0.4188 | -1.916 |
| coil_pt_06 | late | 8189 | 0.007327 | 0.5664 | 0.0007327 | 0.001465 | -0.4935 | -1.742 |
| coil_pt_07 | rising | 8971 | 0.008583 | 0.03679 | 0.001226 | 0 | 1.187 | -0.2562 |
| coil_pt_07 | flat_top | 51915 | 0.003294 | 0.04636 | 0.0002504 | 0.000443 | 0.591 | -0.5456 |
| coil_pt_07 | late | 8189 | 0.01942 | 0.275 | 0.005862 | 0.04237 | -0.01743 | -0.8325 |
| coil_pt_08 | rising | 8971 | 0.1106 | 0.8667 | 0 | 0 | -0.8627 | -2.123 |
| coil_pt_08 | flat_top | 51915 | 0.01304 | 0.7025 | 0 | 0 | -0.7124 | -2.225 |
| coil_pt_08 | late | 8189 | 0.01478 | 0.7594 | 0.001099 | 0.001954 | -0.5821 | -1.595 |
| coil_pt_09 | rising | 8971 | 0.009921 | 0.4255 | 0 | 0 | -2.559 | -3.41 |
| coil_pt_09 | flat_top | 51915 | 1.926e-05 | 0.0004045 | 0 | 0 | -1.379 | -2.822 |
| coil_pt_09 | late | 8189 | 0.01917 | 0.0001221 | 0 | 0 | -0.2404 | -1.838 |
| coil_pt_10 | rising | 8971 | 0.9765 | 0.9964 | 0.03266 | 0.00836 | -3.179 | -3.354 |
| coil_pt_10 | flat_top | 51915 | 0.23 | 0.7804 | 0 | 0 | -1.22 | -2.043 |
| coil_pt_10 | late | 8189 | 0.1103 | 0.02516 | 0 | 0 | 0.5768 | -1.198 |
| coil_pt_11 | rising | 8971 | 0.9893 | 0.9939 | 0.01784 | 0.002006 | -3.315 | -2.842 |
| coil_pt_11 | flat_top | 51915 | 0.8213 | 0.9208 | 0 | 0 | -2.023 | -2.073 |
| coil_pt_11 | late | 8189 | 0.1518 | 0.107 | 0 | 0 | -1.05 | -1.398 |
| coil_pt_12 | rising | 8971 | 0.9951 | 0.9979 | 0.02954 | 0.002452 | -2.91 | -2.57 |
| coil_pt_12 | flat_top | 51915 | 0.8695 | 0.9501 | 0 | 0 | -2.406 | -2.456 |
| coil_pt_12 | late | 8189 | 0.1774 | 0.1956 | 0 | 0 | -1.056 | -1.36 |
| coil_pt_13 | rising | 8971 | 0.9744 | 0.9949 | 0.03779 | 0.009252 | -2.918 | -3.06 |
| coil_pt_13 | flat_top | 51915 | 0.2567 | 0.7726 | 0 | 0 | -1.292 | -2.087 |
| coil_pt_13 | late | 8189 | 0.05727 | 0.05165 | 0 | 0 | 0.3028 | -0.967 |
| coil_pt_14 | rising | 8971 | 0.7458 | 0.02073 | 0.02742 | 0.002675 | -2.602 | -0.9056 |
| coil_pt_14 | flat_top | 51915 | 0.4699 | 0.001214 | 0.0007127 | 0 | -1.651 | -0.2792 |
| coil_pt_14 | late | 8189 | 0.0436 | 0.006961 | 0.001221 | 0.001465 | -1.037 | 0.2673 |
| coil_pt_15 | rising | 8971 | 0.3276 | 0.004793 | 0.004124 | 0.0007803 | -2.417 | -0.3636 |
| coil_pt_15 | flat_top | 51915 | 0.2552 | 0.003564 | 0.001541 | 0 | -1.7 | 0.1323 |
| coil_pt_15 | late | 8189 | 0.02894 | 0.0983 | 0.002076 | 0.003053 | -0.8976 | 0.5819 |
| coil_pt_16 | rising | 8971 | 0.01137 | 0.1503 | 0.001561 | 0.03623 | -1.29 | 0.3057 |
| coil_pt_16 | flat_top | 51915 | 0.01743 | 0.1295 | 0.000732 | 0.0002889 | -0.9006 | 0.5237 |
| coil_pt_16 | late | 8189 | 0.01636 | 0.1882 | 0.004885 | 0.04079 | -0.2173 | 0.7711 |
| coil_pt_17 | rising | 8971 | 0.34 | 0.005239 | 0.004124 | 0.0007803 | -2.379 | -0.5743 |
| coil_pt_17 | flat_top | 51915 | 0.2327 | 0.0009053 | 0.001714 | 0 | -1.918 | -0.1955 |
| coil_pt_17 | late | 8189 | 0.02088 | 0.05178 | 0.001832 | 0.002931 | -1.163 | 0.3417 |
| coil_pt_18 | rising | 8971 | 0.8453 | 0.03355 | 0.01505 | 0.002341 | -2.626 | -1.085 |
| coil_pt_18 | flat_top | 51915 | 0.4541 | 0.001868 | 0.006761 | 0 | -2.085 | -0.6528 |
| coil_pt_18 | late | 8189 | 0.04079 | 0.00867 | 0.0006106 | 0.0004885 | -1.245 | 0.02311 |
| coil_pt_19 | rising | 8971 | 0.2016 | 0.02352 | 0.00301 | 0.0002229 | -2.258 | 0.123 |
| coil_pt_19 | flat_top | 51915 | 0.2542 | 0.02566 | 0.003198 | 0 | -1.929 | 0.3781 |
| coil_pt_19 | late | 8189 | 0.04103 | 0.09183 | 0.001465 | 0.003175 | -1.088 | 0.6747 |
| coil_pt_20 | rising | 8971 | 0.009364 | 0.1415 | 0.001226 | 0.02185 | -1.103 | 0.2574 |
| coil_pt_20 | flat_top | 51915 | 0.01589 | 0.1003 | 0.0006549 | 0 | -0.8185 | 0.4335 |
| coil_pt_20 | late | 8189 | 0.02418 | 0.1093 | 0.005251 | 0.02919 | -0.205 | 0.6652 |
| coil_pt_21 | rising | 8971 | 0.1626 | 0.002898 | 0.00301 | 0.0002229 | -2.255 | -0.055 |
| coil_pt_21 | flat_top | 51915 | 0.2528 | 0.004912 | 0.002774 | 0 | -2.043 | 0.1699 |
| coil_pt_21 | late | 8189 | 0.0309 | 0.07828 | 0.001587 | 0.003297 | -1.112 | 0.6039 |
| coil_c10 | rising | 8971 | 0.01984 | 0.02909 | 0.001115 | 0 | -1.674 | 0.2038 |
| coil_c10 | flat_top | 51915 | 0.0393 | 0.03879 | 0.0006934 | 0.0001541 | -1.228 | 0.4187 |
| coil_c10 | late | 8189 | 0.01368 | 0.1071 | 0.002198 | 0.005373 | -0.7291 | 0.6579 |
| coil_c01 | rising | 8971 | 0 | 0 | 0 | 0 | -0.3635 | -0.3706 |
| coil_c01 | flat_top | 51915 | 0.0005201 | 0 | 0 | 0 | -0.6163 | -0.6002 |
| coil_c01 | late | 8189 | 0.0003663 | 0 | 0 | 0 | -0.3705 | -0.3552 |
| coil_c20 | rising | 8971 | 0.5274 | 0.01059 | 0.01092 | 0 | -2.826 | 0.2035 |
| coil_c20 | flat_top | 51915 | 0.9382 | 0.007474 | 0.503 | 0.000443 | -4.086 | -0.2921 |
| coil_c20 | late | 8189 | 0.8123 | 0.01026 | 0 | 0.0002442 | -3.313 | -0.3932 |
| coil_c11 | rising | 8971 | 0 | 0 | 0 | 0 | 0.1739 | 0.1586 |
| coil_c11 | flat_top | 51915 | 1.926e-05 | 0 | 0 | 0 | 0.02519 | 0.0407 |
| coil_c11 | late | 8189 | 0.0003663 | 0 | 0 | 0 | 0.138 | 0.1706 |
| coil_c02 | rising | 8971 | 0.992 | 0.9982 | 0.1613 | 0.01516 | -3.884 | -3.639 |
| coil_c02 | flat_top | 51915 | 0.7092 | 0.9316 | 0.001695 | 7.705e-05 | -2.329 | -2.841 |
| coil_c02 | late | 8189 | 0.1038 | 0.2165 | 0 | 0 | -0.3627 | -1.504 |
| coil_c30 | rising | 8971 | 0.2814 | 0 | 0.0004459 | 0 | 1.058 | 0.06187 |
| coil_c30 | flat_top | 51915 | 0.2113 | 0 | 0.05769 | 0 | 1.404 | 0.1787 |
| coil_c30 | late | 8189 | 0.4397 | 0.04824 | 0.112 | 0 | 1.98 | 0.3256 |
| coil_c21 | rising | 8971 | 0 | 0 | 0 | 0 | 0.0792 | 0.05755 |
| coil_c21 | flat_top | 51915 | 0 | 0 | 0 | 0 | -0.03543 | -0.03702 |
| coil_c21 | late | 8189 | 0 | 0 | 0 | 0 | 0.05874 | 0.05786 |
| coil_c12 | rising | 8971 | 0.001115 | 0 | 0 | 0 | 1.514 | 2.377 |
| coil_c12 | flat_top | 51915 | 0 | 0 | 0 | 0 | 1.769 | 2.585 |
| coil_c12 | late | 8189 | 0.01246 | 0 | 0 | 0 | 1.203 | 1.873 |
| coil_c03 | rising | 8971 | 0 | 0 | 0 | 0 | 0.1951 | 0.2033 |
| coil_c03 | flat_top | 51915 | 0 | 0 | 0 | 0 | 0.4192 | 0.4098 |
| coil_c03 | late | 8189 | 0 | 0 | 0 | 0 | 0.3129 | 0.2808 |
| ts_beta_p | rising | 8917 | 0.005719 | 0.01054 | 0.0001121 | 0.0001121 | 1.167 | 0.3078 |
| ts_beta_p | flat_top | 51471 | 0.006023 | 0.01399 | 0 | 0 | -0.1766 | -0.4862 |
| ts_beta_p | late | 8066 | 0.02765 | 0.01165 | 0 | 0 | 0.452 | -0.02018 |
| ts_pe_peaking | rising | 8917 | 0.02265 | 0.02265 | 0 | 0 | -0.7146 | -0.7146 |
| ts_pe_peaking | flat_top | 51471 | 0.09069 | 0.09069 | 0 | 0 | -0.8453 | -0.8453 |
| ts_pe_peaking | late | 8066 | 0.06596 | 0.06596 | 0 | 0 | -1.157 | -1.157 |
| ts_valid | rising | 8971 | 0 | 0 | 0 | 0 | nan | nan |
| ts_valid | flat_top | 51915 | 0 | 0 | 0 | 0 | nan | nan |
| ts_valid | late | 8189 | 0 | 0 | 0 | 0 | nan | nan |
| q_cyl | rising | 8971 | 0.2725 | 0.6044 | 0 | 0 | -0.99 | -1.164 |
| q_cyl | flat_top | 51915 | 0.2499 | 0.5367 | 0 | 0 | -1.3 | -1.619 |
| q_cyl | late | 8189 | 0.04677 | 0.1541 | 0 | 0 | -0.2567 | -0.4233 |
| ip_frac | rising | 8971 | 0.02174 | 0.02174 | 0 | 0 | 0.3304 | 0.3304 |
| ip_frac | flat_top | 51915 | 0.1597 | 0.1597 | 3.852e-05 | 3.852e-05 | -0.1958 | -0.1958 |
| ip_frac | late | 8189 | 0.01246 | 0.01246 | 0 | 0 | -0.288 | -0.288 |
| ip_rate_rel | rising | 8962 | 0.4225 | 0.0337 | 0 | 0 | 3.484 | 1.717 |
| ip_rate_rel | flat_top | 51876 | 0.1051 | 0.06088 | 0 | 0 | -0.05371 | -0.03352 |
| ip_rate_rel | late | 8185 | 0.05058 | 0.01063 | 0 | 0 | -1.159 | -0.5874 |
| resistive_age | rising | 8962 | 0.2966 | 0.02209 | 0.0001116 | 0.0001116 | -0.8962 | -0.677 |
| resistive_age | flat_top | 51876 | 0.7921 | 0.3845 | 0.08023 | 0.002988 | -1.103 | -0.9125 |
| resistive_age | late | 8185 | 0.5206 | 0.04643 | 0.00171 | 0.0001222 | -0.8911 | -0.6791 |
| shafranov_lambda | rising | 8971 | 0.0002229 | 0.4588 | 0 | 0.1429 | -0.1809 | -1.14 |
| shafranov_lambda | flat_top | 51915 | 0.0004238 | 0.5017 | 0 | 0.0003467 | -0.2761 | -1.122 |
| shafranov_lambda | late | 8189 | 0.00635 | 0.4821 | 0.001587 | 0.003786 | -0.3376 | -1.061 |
| ohmic_flux_used | rising | 8971 | 0.0068 | 0.0214 | 0.001003 | 0.0008918 | -0.1238 | -1.139 |
| ohmic_flux_used | flat_top | 51915 | 0.003525 | 0.06312 | 0 | 0 | -0.134 | -0.7802 |
| ohmic_flux_used | late | 8189 | 0.02357 | 0.06069 | 0 | 0 | -0.5026 | -0.9565 |
| greenwald_frac | rising | 8917 | 0.009196 | 0.005944 | 0.0001121 | 0.001234 | 0.9691 | -0.09612 |
| greenwald_frac | flat_top | 51471 | 0.01638 | 0.01496 | 0 | 0.001418 | 0.02826 | -0.6261 |
| greenwald_frac | late | 8066 | 0.07339 | 0.0181 | 0 | 0.002727 | 0.6065 | -0.2787 |
| vacuum_residual | rising | 8971 | 0.9605 | 0 | 0.005462 | 0 | -4.222 | 0.2078 |
| vacuum_residual | flat_top | 51915 | 0.9978 | 0 | 0.3005 | 0 | -4.21 | 0.01771 |
| vacuum_residual | late | 8189 | 0.7494 | 0 | 0 | 0 | -3.811 | -0.05085 |
| shaping | rising | 8971 | 0.03077 | 0.7802 | 0 | 0 | -0.08405 | -1.88 |
| shaping | flat_top | 51915 | 0.4596 | 0.1349 | 0.07327 | 0 | 1.658 | -1.227 |
| shaping | late | 8189 | 0.3703 | 0.006716 | 0 | 0 | 2.163 | -0.2681 |

## 4. Stretch terms and vacuum relation (6b quantities)

Median abs(`vacuum_residual`) over median abs(`coil_c20`), beside the same ratio for the current-free field R^4 - 4 R^2 Z^2 sampled at the layout points and fitted the same way at the same radii (the part of the residual that comes from the fit alone, for that field).

| split | R0 (m) | a0 (m) | phase | n | ratio | ratio for R^4 - 4 R^2 Z^2 |
| --- | --- | --- | --- | --- | --- | --- |
| DIII-D train | 1.67 | 0.67 | all | 1289467 | 0.02795 | 0.0258 |
| DIII-D train | 1.67 | 0.67 | rising | 234424 | 0.03074 | 0.0258 |
| DIII-D train | 1.67 | 0.67 | flat_top | 1002719 | 0.02726 | 0.0258 |
| DIII-D train | 1.67 | 0.67 | late | 52324 | 0.02674 | 0.0258 |
| MAST, v1 radii | 0.85 | 0.65 | all | 69075 | 0.08081 | 0.0606 |
| MAST, v1 radii | 0.85 | 0.65 | rising | 8971 | 0.08412 | 0.0606 |
| MAST, v1 radii | 0.85 | 0.65 | flat_top | 51915 | 0.08147 | 0.0606 |
| MAST, v1 radii | 0.85 | 0.65 | late | 8189 | 0.07418 | 0.0606 |
| MAST, backup radii | 0.7 | 0.5 | all | 69075 | 0.02573 | 0.057 |
| MAST, backup radii | 0.7 | 0.5 | rising | 8971 | 0.02714 | 0.057 |
| MAST, backup radii | 0.7 | 0.5 | flat_top | 51915 | 0.02562 | 0.057 |
| MAST, backup radii | 0.7 | 0.5 | late | 8189 | 0.02524 | 0.057 |

Median and interquartile range:

| quantity | split | n | median | IQR |
| --- | --- | --- | --- | --- |
| vacuum_residual | DIII-D train | 1289467 | -0.0007309 | 0.0006477 |
| vacuum_residual | MAST, v1 radii | 69075 | -0.003451 | 0.0007356 |
| vacuum_residual | MAST, backup radii | 69075 | -0.0007046 | 0.0001 |
| shaping | DIII-D train | 1289467 | 0.0187 | 0.003027 |
| shaping | MAST, v1 radii | 69075 | 0.0235 | 0.006335 |
| shaping | MAST, backup radii | 69075 | 0.01514 | 0.003399 |
| coil_c20 | DIII-D train | 1289467 | -0.02619 | 0.004155 |
| coil_c20 | MAST, v1 radii | 69075 | -0.0427 | 0.008262 |
| coil_c20 | MAST, backup radii | 69075 | -0.02738 | 0.004317 |
| coil_c02 | DIII-D train | 1289467 | 0.01086 | 0.002921 |
| coil_c02 | MAST, v1 radii | 69075 | 0.004203 | 0.004593 |
| coil_c02 | MAST, backup radii | 69075 | 0.00275 | 0.002623 |

## 5. Thomson against force balance (X14)

Line `shafranov_lambda = intercept + slope ts_beta_p` over frames inside both 1st to 99th percentile bands of each split; median offset from the DIII-D training line of the same frames (all or flat-top).

| frames | split | n | slope | intercept | r | residual std | median offset from training line |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | DIII-D train | 1225650 | 2.964 | 0.4735 | 0.7355 | 0.2285 | -0.0063 |
| all | MAST, v1 radii | 65972 | 1.536 | 0.4655 | 0.6066 | 0.1091 | -0.1744 |
| all | MAST, backup radii | 65976 | 1.73 | 0.1278 | 0.4851 | 0.09978 | -0.4293 |
| flat_top | DIII-D train | 956349 | 3.043 | 0.4363 | 0.7462 | 0.2217 | -0.0003344 |
| flat_top | MAST, v1 radii | 49590 | 1.96 | 0.4177 | 0.7149 | 0.09228 | -0.1507 |
| flat_top | MAST, backup radii | 49614 | 2.556 | 0.07505 | 0.6914 | 0.07596 | -0.399 |

## 6. Combined distance (X10 method), ip_frac unscaled

200000 DIII-D training frames (seed 0); DIII-D validation 5000, DIII-D test 5000, MAST, v1 radii 5000, MAST, backup radii 5000 frames (MAST at both radii on the same frames). Version-1 statistics for every column but `ip_frac` (identity) and `shaping` (DIII-D training, originals and mirrors: median 0.018697, interquartile range 0.0030271).

| variant | split | n | median | 90th | 99th | max | beyond validation 99th |
| --- | --- | --- | --- | --- | --- | --- | --- |
| set_a, ip_frac unscaled | DIII-D validation | 5000 | 0.1941 | 0.3739 | 0.8065 | 19.38 | 0.01 |
| set_a, ip_frac unscaled | DIII-D test | 5000 | 0.202 | 0.3807 | 0.8492 | 3.177 | 0.0126 |
| set_a, ip_frac unscaled | MAST, v1 radii | 5000 | 0.9561 | 1.295 | 2.295 | 7.669 | 0.8586 |
| set_a, ip_frac unscaled | MAST, backup radii | 5000 | 0.9823 | 1.2 | 1.845 | 3.998 | 0.8448 |
| set_b, ip_frac unscaled | DIII-D validation | 5000 | 0.3818 | 0.8342 | 1.897 | 20 | 0.01 |
| set_b, ip_frac unscaled | DIII-D test | 5000 | 0.3926 | 0.8407 | 2.02 | 9.544 | 0.0112 |
| set_b, ip_frac unscaled | MAST, v1 radii | 5000 | 4.348 | 5.258 | 8.732 | 28.08 | 1 |
| set_b, ip_frac unscaled | MAST, backup radii | 5000 | 3.132 | 3.429 | 4.278 | 7.105 | 1 |
| set_b, shaping for coil_c20 and coil_c02, ip_frac unscaled | DIII-D validation | 5000 | 0.3709 | 0.809 | 1.841 | 19.91 | 0.01 |
| set_b, shaping for coil_c20 and coil_c02, ip_frac unscaled | DIII-D test | 5000 | 0.3809 | 0.817 | 1.928 | 9.43 | 0.0108 |
| set_b, shaping for coil_c20 and coil_c02, ip_frac unscaled | MAST, v1 radii | 5000 | 2.582 | 3.929 | 7.949 | 27.25 | 0.932 |
| set_b, shaping for coil_c20 and coil_c02, ip_frac unscaled | MAST, backup radii | 5000 | 2.209 | 2.72 | 3.979 | 6.859 | 0.891 |

Median share of the squared distance per feature group:

| variant | split | coil_pattern | plasma_content | field_current | shot_position | coil_terms |
| --- | --- | --- | --- | --- | --- | --- |
| set_a, ip_frac unscaled | DIII-D validation | 0.2081 | 0.2355 | 0.02231 | 0.3106 | nan |
| set_a, ip_frac unscaled | DIII-D test | 0.2131 | 0.2327 | 0.02317 | 0.3024 | nan |
| set_a, ip_frac unscaled | MAST, v1 radii | 0.739 | 0.0831 | 0.02227 | 0.1108 | nan |
| set_a, ip_frac unscaled | MAST, backup radii | 0.7127 | 0.06724 | 0.03763 | 0.1216 | nan |
| set_b, ip_frac unscaled | DIII-D validation | nan | 0.1509 | 0.004642 | 0.2132 | 0.467 |
| set_b, ip_frac unscaled | DIII-D test | nan | 0.1536 | 0.004494 | 0.2109 | 0.4635 |
| set_b, ip_frac unscaled | MAST, v1 radii | nan | 0.06401 | 0.04894 | 0.03161 | 0.8443 |
| set_b, ip_frac unscaled | MAST, backup radii | nan | 0.04931 | 0.05205 | 0.04508 | 0.8209 |
| set_b, shaping for coil_c20 and coil_c02, ip_frac unscaled | DIII-D validation | nan | 0.1554 | 0.004894 | 0.2145 | 0.4552 |
| set_b, shaping for coil_c20 and coil_c02, ip_frac unscaled | DIII-D test | nan | 0.1572 | 0.00479 | 0.2169 | 0.4551 |
| set_b, shaping for coil_c20 and coil_c02, ip_frac unscaled | MAST, v1 radii | nan | 0.09188 | 0.05504 | 0.07984 | 0.7238 |
| set_b, shaping for coil_c20 and coil_c02, ip_frac unscaled | MAST, backup radii | nan | 0.05965 | 0.06807 | 0.08853 | 0.7095 |

Runtime 19 s.
