# Feature profile, version 2

Feature tables `../data/processed/features/v2`, configuration hash `01263fe0c4f1`. Reference of every comparison: the 1289474 frames of the 5821 shots of the `train` list of `split_v1.json`. Nothing is fitted on test inputs.

| Id | Check | Status | Summary |
| --- | --- | --- | --- |
| X1 | Rows and keys | pass | rows and keys equal the frame table on every split (diii_d_train 1559340, diii_d_public_test 193359, mast_public_test 69075); build record matches; configuration unchanged since the build |
| X2 | Inputs only | pass | 41 distinct columns read; every one is in input_columns(machine, include_flags=True) or is time_ms |
| X3 | Completeness | review | lowest finite share: diii_d_train ts_beta_p 0.9852; diii_d_public_test ts_beta_p 0.9869; mast_public_test ts_beta_p 0.9910. Frames with Ip missing or below 20 kA: diii_d_train 7, diii_d_public_test 0, mast_public_test 0 |
| X4 | Layout | pass | 22 points, symmetric True; smallest clearance DIII-D 0.279 m, MAST 0.199 m (rule 0.15 m); failing points: 0 |
| X5 | Mirror | pass | mirroring twice returns the input on 3 of 3 splits; mirrored points against points from A and B exchanged, 4522 frames of 20 training shots: median 0.0021, 95th percentile 0.0041, max 0.0051 of the pattern's range (F51 expects about 0.02; pass at or below 0.05 at the 95th percentile) |
| X6 | Current direction | review | 1182807 positive-Ip and 106066 negative-Ip training frames; share of negative-Ip values inside the positive band from 0.960 to 0.999; largest median shift coil_c11 -0.54 interquartile ranges |
| X7 | Force-balance sign | pass | median 4 pi (R0/a0) coil_c10 on flat-top frames: DIII-D -2.259 (K9 -2.096), MAST -1.485 (K9 -1.351); signs match, order matches |
| X8 | Per-feature ranges | review | largest MAST share outside the training band: coil_c20 0.874, coil_pt_12 0.800, coil_pt_11 0.757 |
| X9 | Stretch terms | review | MAST share outside the training band by phase: coil_c20 all 0.874; coil_c02 all 0.693; shaping all 0.388; coil_c20 rising 0.527; coil_c02 rising 0.992; shaping rising 0.031; coil_c20 flat_top 0.938; coil_c02 flat_top 0.709; shaping flat_top 0.460; coil_c20 late 0.812; coil_c02 late 0.104; shaping late 0.370 |
| X10 | Combined distance | review | set_a: MAST median 0.96 against validation 0.19, 0.859 of MAST frames beyond the validation 99th percentile; set_b: MAST median 2.72 against validation 0.37, 0.962 of MAST frames beyond the validation 99th percentile |
| X11 | Extremes after scaling | review | largest share of scaled values beyond 10: train coil_c12 0.0089; validation coil_c12 0.0104; diii_d_public_test coil_c12 0.0162; mast_public_test ip_rate_rel 0.0733 |
| X12 | Shared factor removed | review | 200000 training frames. Largest /r/ with Ip (signed, magnitude): coil features 0.154, 0.430; raw currents 0.942, 0.517. Median /r/ between columns: coil features 0.346, raw currents 0.532 |
| X13 | Pattern preserved | review | shown in notebook 06, section 3 |
| X14 | Thomson against force balance | review | shafranov_lambda against ts_beta_p, slope (r): train 2.964 (0.736), diii_d_public_test 2.879 (0.730), mast_public_test 1.536 (0.607); median offset from the training line: diii_d_public_test -0.014, mast_public_test -0.174 |
| X15 | Timescales | review | MAST band width over DIII-D training: ip__rate50 9.75, ip_rate_rel 4.30; time since rise 0.07, resistive_age 0.21. Frames off the flat-top: DIII-D train 0.222, DIII-D test 0.227, MAST 0.248. Median ip_rise_ms: DIII-D train 14.9, DIII-D test 14.9, MAST 2.2 |

X13 (the pattern before and after scaling) is shown in notebook 06, section 3; X16 is written by `scripts/feature_linear_check.py` to `linear_check.md`.

## X3. Completeness

Share of finite values per feature and split. Coil features are NaN where a coil current is missing or Ip is below the threshold; Thomson features where `ts_valid` is 0.

| Feature | diii_d_train | diii_d_public_test | mast_public_test |
| --- | --- | --- | --- |
| coil_pt_00 | 1.0000 | 1.0000 | 1.0000 |
| coil_pt_01 | 1.0000 | 1.0000 | 1.0000 |
| coil_pt_02 | 1.0000 | 1.0000 | 1.0000 |
| coil_pt_03 | 1.0000 | 1.0000 | 1.0000 |
| coil_pt_04 | 1.0000 | 1.0000 | 1.0000 |
| coil_pt_05 | 1.0000 | 1.0000 | 1.0000 |
| coil_pt_06 | 1.0000 | 1.0000 | 1.0000 |
| coil_pt_07 | 1.0000 | 1.0000 | 1.0000 |
| coil_pt_08 | 1.0000 | 1.0000 | 1.0000 |
| coil_pt_09 | 1.0000 | 1.0000 | 1.0000 |
| coil_pt_10 | 1.0000 | 1.0000 | 1.0000 |
| coil_pt_11 | 1.0000 | 1.0000 | 1.0000 |
| coil_pt_12 | 1.0000 | 1.0000 | 1.0000 |
| coil_pt_13 | 1.0000 | 1.0000 | 1.0000 |
| coil_pt_14 | 1.0000 | 1.0000 | 1.0000 |
| coil_pt_15 | 1.0000 | 1.0000 | 1.0000 |
| coil_pt_16 | 1.0000 | 1.0000 | 1.0000 |
| coil_pt_17 | 1.0000 | 1.0000 | 1.0000 |
| coil_pt_18 | 1.0000 | 1.0000 | 1.0000 |
| coil_pt_19 | 1.0000 | 1.0000 | 1.0000 |
| coil_pt_20 | 1.0000 | 1.0000 | 1.0000 |
| coil_pt_21 | 1.0000 | 1.0000 | 1.0000 |
| coil_c10 | 1.0000 | 1.0000 | 1.0000 |
| coil_c01 | 1.0000 | 1.0000 | 1.0000 |
| coil_c20 | 1.0000 | 1.0000 | 1.0000 |
| coil_c11 | 1.0000 | 1.0000 | 1.0000 |
| coil_c02 | 1.0000 | 1.0000 | 1.0000 |
| coil_c30 | 1.0000 | 1.0000 | 1.0000 |
| coil_c21 | 1.0000 | 1.0000 | 1.0000 |
| coil_c12 | 1.0000 | 1.0000 | 1.0000 |
| coil_c03 | 1.0000 | 1.0000 | 1.0000 |
| shaping | 1.0000 | 1.0000 | 1.0000 |
| ts_beta_p | 0.9852 | 0.9869 | 0.9910 |
| ts_pe_peaking | 0.9852 | 0.9869 | 0.9910 |
| ts_valid | 1.0000 | 1.0000 | 1.0000 |
| q_cyl | 1.0000 | 1.0000 | 1.0000 |
| ip_frac | 1.0000 | 1.0000 | 1.0000 |
| ip_rate_rel | 0.9955 | 0.9965 | 0.9992 |
| resistive_age | 0.9955 | 0.9965 | 0.9992 |
| shafranov_lambda | 1.0000 | 1.0000 | 1.0000 |
| ohmic_flux_used | 1.0000 | 1.0000 | 1.0000 |
| greenwald_frac | 0.9852 | 0.9869 | 0.9910 |

Frames with Ip missing or below 20 kA in magnitude:

| Split | Frames | Ip missing | Below threshold | Share affected |
| --- | --- | --- | --- | --- |
| diii_d_train | 1559340 | 0 | 7 | 0.000004 |
| diii_d_public_test | 193359 | 0 | 0 | 0.000000 |
| mast_public_test | 69075 | 0 | 0 | 0.000000 |

## X5. Mirror

Mirroring twice: diii_d_train identity, diii_d_public_test identity, mast_public_test identity.
Training shots of the A/B exchange: 0448d620e0, 0a672c5a6f, 12e65e23a9, 2abef28a77, 41fe301a86, 4bac1949c7, 7f654c231d, 8147e11f7c, 890a49db4a, 8d51a69eaa, 990d86f2c3, a076601c87, a152882371, a4bc596f12, ba12fb7876, cedcbf7905, d7aec3eacb, e8d670db78, eebfc7d7c4, f86d785364.

## X6. Coil terms by current direction (DIII-D training)

| Term | Median + | Median - | IQR + | IQR - | Shift (IQR +) | Share - inside + band |
| --- | --- | --- | --- | --- | --- | --- |
| coil_c10 | -0.0722 | -0.0679 | 0.0155 | 0.0078 | 0.27 | 0.982 |
| coil_c01 | 0.0011 | 0.0019 | 0.0021 | 0.0017 | 0.37 | 0.986 |
| coil_c20 | -0.0263 | -0.0252 | 0.0042 | 0.0032 | 0.25 | 0.982 |
| coil_c11 | -0.0001 | -0.0023 | 0.0040 | 0.0025 | -0.54 | 0.999 |
| coil_c02 | 0.0109 | 0.0105 | 0.0029 | 0.0031 | -0.12 | 0.982 |
| coil_c30 | 0.0013 | 0.0000 | 0.0047 | 0.0049 | -0.28 | 0.968 |
| coil_c21 | 0.0001 | -0.0002 | 0.0036 | 0.0012 | -0.09 | 0.996 |
| coil_c12 | -0.0089 | -0.0090 | 0.0038 | 0.0031 | -0.02 | 0.960 |
| coil_c03 | -0.0005 | -0.0005 | 0.0014 | 0.0012 | -0.02 | 0.999 |

## X7. Force-balance sign

K9 medians from reports/coil_flux/reference_ratios.parquet. Flat-top frames: DIII-D 1002719, MAST 51915.

## X8. Per-feature ranges (all frames)

Share outside the training 1st to 99th percentile band and outside the training range; median shift in training interquartile ranges.

| Feature | Train band | Val band / range / shift | DIII-D test band / range / shift | MAST band / range / shift |
| --- | --- | --- | --- | --- |
| coil_pt_00 | 0.036 to 0.104 | 0.020 / 0.000 / 0.01 | 0.026 / 0.000 / -0.03 | 0.000 / 0.000 / -2.58 |
| coil_pt_01 | 0.035 to 0.072 | 0.023 / 0.000 / -0.01 | 0.025 / 0.000 / -0.00 | 0.012 / 0.000 / -0.34 |
| coil_pt_02 | 0.026 to 0.064 | 0.016 / 0.000 / -0.01 | 0.018 / 0.000 / 0.03 | 0.006 / 0.001 / 0.40 |
| coil_pt_03 | 0.037 to 0.071 | 0.022 / 0.000 / -0.03 | 0.022 / 0.000 / 0.00 | 0.047 / 0.000 / -0.60 |
| coil_pt_04 | 0.028 to 0.104 | 0.020 / 0.000 / -0.01 | 0.027 / 0.000 / -0.02 | 0.000 / 0.000 / -2.23 |
| coil_pt_05 | 0.036 to 0.071 | 0.019 / 0.000 / -0.01 | 0.028 / 0.000 / -0.04 | 0.274 / 0.000 / -1.64 |
| coil_pt_06 | 0.022 to 0.044 | 0.021 / 0.000 / -0.00 | 0.025 / 0.000 / -0.01 | 0.006 / 0.000 / -0.42 |
| coil_pt_07 | 0.017 to 0.036 | 0.017 / 0.000 / -0.01 | 0.018 / 0.000 / 0.02 | 0.009 / 0.001 / 0.61 |
| coil_pt_08 | 0.023 to 0.044 | 0.021 / 0.000 / -0.02 | 0.025 / 0.000 / -0.01 | 0.020 / 0.000 / -0.71 |
| coil_pt_09 | 0.019 to 0.075 | 0.019 / 0.000 / 0.00 | 0.026 / 0.000 / -0.03 | 0.009 / 0.000 / -1.40 |
| coil_pt_10 | 0.011 to 0.040 | 0.020 / 0.000 / 0.00 | 0.029 / 0.000 / -0.02 | 0.335 / 0.000 / -1.26 |
| coil_pt_11 | 0.002 to 0.010 | 0.019 / 0.000 / 0.01 | 0.025 / 0.000 / -0.01 | 0.757 / 0.000 / -2.05 |
| coil_pt_12 | 0.003 to 0.012 | 0.017 / 0.000 / 0.03 | 0.023 / 0.000 / 0.00 | 0.800 / 0.000 / -2.32 |
| coil_pt_13 | 0.011 to 0.041 | 0.022 / 0.000 / 0.03 | 0.024 / 0.000 / 0.00 | 0.349 / 0.001 / -1.31 |
| coil_pt_14 | -0.038 to -0.005 | 0.018 / 0.000 / 0.02 | 0.027 / 0.000 / 0.00 | 0.444 / 0.000 / -1.73 |
| coil_pt_15 | -0.042 to -0.019 | 0.021 / 0.000 / 0.02 | 0.020 / 0.000 / 0.01 | 0.214 / 0.000 / -1.72 |
| coil_pt_16 | -0.045 to -0.024 | 0.018 / 0.000 / 0.01 | 0.017 / 0.000 / -0.01 | 0.019 / 0.001 / -0.90 |
| coil_pt_17 | -0.042 to -0.017 | 0.016 / 0.000 / 0.02 | 0.023 / 0.000 / -0.01 | 0.208 / 0.000 / -1.91 |
| coil_pt_18 | -0.038 to -0.003 | 0.017 / 0.000 / 0.02 | 0.026 / 0.000 / 0.00 | 0.468 / 0.000 / -2.10 |
| coil_pt_19 | -0.100 to -0.052 | 0.021 / 0.000 / 0.02 | 0.020 / 0.000 / 0.00 | 0.196 / 0.000 / -1.92 |
| coil_pt_20 | -0.097 to -0.054 | 0.019 / 0.000 / 0.01 | 0.017 / 0.000 / -0.02 | 0.019 / 0.000 / -0.81 |
| coil_pt_21 | -0.099 to -0.050 | 0.018 / 0.000 / 0.02 | 0.021 / 0.000 / -0.01 | 0.186 / 0.000 / -2.00 |
| coil_c10 | -0.109 to -0.055 | 0.018 / 0.000 / 0.03 | 0.017 / 0.000 / -0.01 | 0.031 / 0.000 / -1.26 |
| coil_c01 | -0.003 to 0.006 | 0.016 / 0.000 / -0.03 | 0.023 / 0.000 / 0.02 | 0.000 / 0.000 / -0.56 |
| coil_c20 | -0.036 to -0.020 | 0.019 / 0.000 / 0.01 | 0.024 / 0.000 / 0.00 | 0.874 / 0.000 / -3.97 |
| coil_c11 | -0.007 to 0.015 | 0.017 / 0.000 / 0.01 | 0.021 / 0.000 / 0.01 | 0.000 / 0.000 / 0.05 |
| coil_c02 | 0.006 to 0.016 | 0.020 / 0.000 / 0.03 | 0.025 / 0.000 / -0.02 | 0.693 / 0.000 / -2.28 |
| coil_c30 | -0.013 to 0.010 | 0.019 / 0.000 / -0.03 | 0.031 / 0.000 / -0.02 | 0.272 / 0.013 / 1.42 |
| coil_c21 | -0.006 to 0.012 | 0.020 / 0.000 / -0.00 | 0.017 / 0.000 / 0.02 | 0.000 / 0.000 / -0.02 |
| coil_c12 | -0.015 to 0.025 | 0.021 / 0.000 / -0.02 | 0.028 / 0.000 / 0.01 | 0.005 / 0.000 / 1.76 |
| coil_c03 | -0.004 to 0.003 | 0.024 / 0.000 / 0.05 | 0.021 / 0.000 / 0.01 | 0.000 / 0.000 / 0.38 |
| shaping | 0.013 to 0.025 | 0.018 / 0.000 / -0.00 | 0.025 / 0.000 / -0.02 | 0.388 / 0.000 / 1.59 |
| ts_beta_p | 0.017 to 0.365 | 0.019 / 0.000 / -0.02 | 0.019 / 0.000 / 0.04 | 0.011 / 0.000 / -0.05 |
| ts_pe_peaking | 1.789 to 3.738 | 0.020 / 0.000 / 0.02 | 0.022 / 0.000 / 0.01 | 0.085 / 0.000 / -0.89 |
| ts_valid | 0.000 to 1.000 | 0.000 / 0.000 / nan | 0.000 / 0.000 / nan | 0.000 / 0.000 / nan |
| q_cyl | 1.588 to 5.926 | 0.022 / 0.000 / 0.02 | 0.022 / 0.000 / 0.04 | 0.283 / 0.000 / -1.02 |
| ip_frac | 0.364 to 0.998 | 0.020 / 0.000 / 0.00 | 0.019 / 0.000 / 0.00 | 0.126 / 0.000 / -0.17 |
| ip_rate_rel | -9.287 to 7.319 | 0.019 / 0.000 / 0.00 | 0.018 / 0.000 / 0.01 | 0.191 / 0.000 / -0.09 |
| resistive_age | 0.153 to 2.804 | 0.017 / 0.000 / 0.00 | 0.015 / 0.000 / -0.01 | 0.373 / 0.000 / -0.95 |
| shafranov_lambda | 0.240 to 1.913 | 0.018 / 0.000 / -0.03 | 0.017 / 0.000 / 0.01 | 0.002 / 0.000 / -0.25 |
| ohmic_flux_used | -0.529 to -0.100 | 0.019 / 0.000 / -0.02 | 0.018 / 0.000 / -0.01 | 0.033 / 0.000 / -0.25 |
| greenwald_frac | 0.120 to 1.283 | 0.019 / 0.000 / -0.03 | 0.016 / 0.000 / 0.04 | 0.038 / 0.000 / 0.15 |

## X9. Stretch terms by phase

| Term | Phase | Train band | Train median | MAST median | MAST outside band | MAST outside range | DIII-D test outside band |
| --- | --- | --- | --- | --- | --- | --- | --- |
| coil_c20 | all | -0.036 to -0.020 | -0.026 | -0.043 | 0.874 | 0.000 | 0.024 |
| coil_c20 | rising | -0.036 to -0.020 | -0.025 | -0.036 | 0.527 | 0.011 | 0.023 |
| coil_c20 | flat_top | -0.035 to -0.020 | -0.026 | -0.043 | 0.938 | 0.503 | 0.025 |
| coil_c20 | late | -0.044 to -0.019 | -0.027 | -0.049 | 0.812 | 0.000 | 0.023 |
| coil_c02 | all | 0.006 to 0.016 | 0.011 | 0.004 | 0.693 | 0.000 | 0.025 |
| coil_c02 | rising | 0.006 to 0.015 | 0.010 | -0.001 | 0.992 | 0.161 | 0.025 |
| coil_c02 | flat_top | 0.006 to 0.015 | 0.011 | 0.004 | 0.709 | 0.002 | 0.028 |
| coil_c02 | late | 0.005 to 0.020 | 0.012 | 0.011 | 0.104 | 0.000 | 0.027 |
| shaping | all | 0.013 to 0.025 | 0.019 | 0.023 | 0.388 | 0.000 | 0.025 |
| shaping | rising | 0.013 to 0.025 | 0.018 | 0.018 | 0.031 | 0.000 | 0.023 |
| shaping | flat_top | 0.013 to 0.024 | 0.019 | 0.024 | 0.460 | 0.073 | 0.026 |
| shaping | late | 0.013 to 0.032 | 0.020 | 0.030 | 0.370 | 0.000 | 0.029 |

## X10. Combined distance

Scaled features, 200000 training frames (seed 0), up to 5000 frames per compared split. With mirrors: the nearest of the training frames and their mirrors.

Version 1 beside this version: `reports/features/v1/profile.json`, same split file, samples and seed, so the same sampled frames; only the sets and the scaling differ.

| Set | Split | Median | 90th | 99th | Max | Beyond validation 99th | Median with mirrors | v1 median | v1 99th | v1 beyond validation 99th |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| set_a | validation | 0.194 | 0.374 | 0.807 | 19.385 | 0.010 | 0.192 | 0.230 | 1.060 | 0.010 |
| set_a | diii_d_public_test | 0.202 | 0.381 | 0.849 | 3.177 | 0.013 | 0.200 | 0.236 | 1.092 | 0.012 |
| set_a | mast_public_test | 0.956 | 1.295 | 2.295 | 7.669 | 0.859 | 0.956 | 1.027 | 2.851 | 0.435 |
| set_b | validation | 0.371 | 0.809 | 1.841 | 19.908 | 0.010 | 0.371 | 0.421 | 2.149 | 0.010 |
| set_b | diii_d_public_test | 0.381 | 0.817 | 1.928 | 9.430 | 0.011 | 0.381 | 0.426 | 2.192 | 0.011 |
| set_b | mast_public_test | 2.717 | 4.149 | 8.345 | 27.205 | 0.962 | 2.700 | 4.431 | 9.325 | 1.000 |

Median share of the squared distance carried by each feature group:

- set_a, validation: coil_pattern 0.208, plasma_content 0.236, field_current 0.022, shot_position 0.311
- set_a, diii_d_public_test: coil_pattern 0.213, plasma_content 0.233, field_current 0.023, shot_position 0.302
- set_a, mast_public_test: coil_pattern 0.739, plasma_content 0.083, field_current 0.022, shot_position 0.111
- set_b, validation: plasma_content 0.155, field_current 0.005, shot_position 0.214, coil_terms 0.455
- set_b, diii_d_public_test: plasma_content 0.157, field_current 0.005, shot_position 0.217, coil_terms 0.455
- set_b, mast_public_test: plasma_content 0.043, field_current 0.054, shot_position 0.074, coil_terms 0.762

## X11. Scaled values beyond 10 in magnitude

Features with a non-zero share on any split in either version; each cell is this version's share, then version 1's in brackets (a dash where that version has no such feature).

| Feature | train | validation | diii_d_public_test | mast_public_test |
| --- | --- | --- | --- | --- |
| coil_pt_00 | 0.00000 (0.00000) | 0.00001 (0.00001) | 0.00000 (0.00000) | 0.00000 (0.00000) |
| coil_pt_01 | 0.00000 (0.00000) | 0.00000 (0.00000) | 0.00000 (0.00000) | 0.00000 (0.00000) |
| coil_pt_02 | 0.00000 (0.00000) | 0.00000 (0.00000) | 0.00000 (0.00000) | 0.00000 (0.00000) |
| coil_pt_03 | 0.00000 (0.00000) | 0.00000 (0.00000) | 0.00000 (0.00000) | 0.00000 (0.00000) |
| coil_pt_04 | 0.00000 (0.00000) | 0.00001 (0.00001) | 0.00000 (0.00000) | 0.00000 (0.00000) |
| coil_pt_05 | 0.00000 (0.00000) | 0.00001 (0.00001) | 0.00000 (0.00000) | 0.00000 (0.00000) |
| coil_pt_08 | 0.00000 (0.00000) | 0.00000 (0.00000) | 0.00000 (0.00000) | 0.00000 (0.00000) |
| coil_pt_09 | 0.00000 (0.00000) | 0.00000 (0.00000) | 0.00000 (0.00000) | 0.00000 (0.00000) |
| coil_pt_16 | 0.00000 (0.00000) | 0.00000 (0.00000) | 0.00000 (0.00000) | 0.00000 (0.00000) |
| coil_pt_19 | 0.00000 (0.00000) | 0.00001 (0.00001) | 0.00000 (0.00000) | 0.00000 (0.00000) |
| coil_pt_20 | 0.00000 (0.00000) | 0.00001 (0.00001) | 0.00000 (0.00000) | 0.00000 (0.00000) |
| coil_pt_21 | 0.00000 (0.00000) | 0.00001 (0.00001) | 0.00000 (0.00000) | 0.00000 (0.00000) |
| coil_c10 | 0.00001 (0.00001) | 0.00001 (0.00001) | 0.00000 (0.00000) | 0.00000 (0.00000) |
| coil_c01 | 0.00000 (0.00000) | 0.00000 (0.00000) | 0.00000 (0.00000) | 0.00000 (0.00000) |
| coil_c20 | 0.00001 (0.00001) | 0.00001 (0.00001) | 0.00001 (0.00001) | 0.00187 (0.00187) |
| coil_c11 | 0.00000 (0.00000) | 0.00000 (0.00000) | 0.00005 (0.00005) | 0.00000 (0.00000) |
| coil_c02 | 0.00000 (0.00000) | 0.00001 (0.00001) | 0.00000 (0.00000) | 0.00007 (0.00007) |
| coil_c30 | 0.00000 (0.00000) | 0.00000 (0.00000) | 0.00000 (0.00000) | 0.00812 (0.00812) |
| coil_c21 | 0.00000 (0.00000) | 0.00001 (0.00001) | 0.00000 (0.00000) | 0.00000 (0.00000) |
| coil_c12 | 0.00889 (0.00889) | 0.01045 (0.01045) | 0.01622 (0.01622) | 0.00000 (0.00000) |
| shaping | 0.00001 (-) | 0.00001 (-) | 0.00001 (-) | 0.00064 (-) |
| ts_beta_p | 0.00001 (0.00001) | 0.00001 (0.00001) | 0.00000 (0.00000) | 0.00001 (0.00001) |
| q_cyl | 0.00057 (0.00057) | 0.00060 (0.00060) | 0.00044 (0.00044) | 0.00019 (0.00019) |
| ip_frac | 0.00000 (0.02154) | 0.00000 (0.02186) | 0.00000 (0.02077) | 0.00000 (0.02051) |
| ip_rate_rel | 0.00616 (0.00616) | 0.00612 (0.00612) | 0.00586 (0.00586) | 0.07327 (0.07327) |
| resistive_age | 0.00045 (0.00045) | 0.00161 (0.00161) | 0.00002 (0.00002) | 0.00000 (0.00000) |
| shafranov_lambda | 0.00001 (0.00001) | 0.00001 (0.00001) | 0.00000 (0.00000) | 0.00000 (0.00000) |
| ohmic_flux_used | 0.00034 (0.00034) | 0.00028 (0.00028) | 0.00035 (0.00035) | 0.00236 (0.00236) |
| greenwald_frac | 0.00007 (0.00007) | 0.00002 (0.00002) | 0.00001 (0.00001) | 0.00017 (0.00017) |

## X12. Correlations (DIII-D training)

- raw_currents: median |r| 0.532, largest 0.961 (magnetics_F1A__mean and magnetics_F1B__mean); with Ip: ip__mean largest 0.942 (magnetics_F6A__mean), median 0.795, abs_ip largest 0.517 (sol__mean), median 0.273
- coil_features: median |r| 0.346, largest 0.995 (coil_pt_16 and coil_pt_20); with Ip: ip__mean largest 0.154 (coil_c11), median 0.045, abs_ip largest 0.430 (coil_pt_20), median 0.177

## X14. Thomson against force balance

Line `shafranov_lambda = intercept + slope ts_beta_p` over frames inside both 1st to 99th percentile bands of the split.

| Frames | Split | n | Slope | Intercept | r | Residual std | Median offset from training line |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | train | 1225650 | 2.964 | 0.473 | 0.736 | 0.229 | -0.006 |
| all | diii_d_public_test | 184050 | 2.879 | 0.473 | 0.730 | 0.223 | -0.014 |
| all | mast_public_test | 65972 | 1.536 | 0.466 | 0.607 | 0.109 | -0.174 |
| flat_top | train | 956349 | 3.043 | 0.436 | 0.746 | 0.222 | -0.000 |
| flat_top | diii_d_public_test | 142764 | 2.973 | 0.433 | 0.738 | 0.214 | -0.011 |
| flat_top | mast_public_test | 49590 | 1.960 | 0.418 | 0.715 | 0.092 | -0.151 |

## X15. Timescales

| Quantity | Machine | 1st | Median | 99th | Width over DIII-D training |
| --- | --- | --- | --- | --- | --- |
| ip__rate50 (kA/s) | DIII-D train | -1333.032 | 16.827 | 1243.548 | 1.00 |
| ip__rate50 (kA/s) | DIII-D test | -1273.385 | 17.248 | 1217.691 | 0.97 |
| ip__rate50 (kA/s) | MAST | -19076.343 | -5.527 | 6057.824 | 9.75 |
| ip_rate_rel | DIII-D train | -9.287 | 0.105 | 7.319 | 1.00 |
| ip_rate_rel | DIII-D test | -8.872 | 0.113 | 7.043 | 0.96 |
| ip_rate_rel | MAST | -56.418 | -0.018 | 14.989 | 4.30 |
| time since the Ip rise (s) | DIII-D train | 0.249 | 2.632 | 6.547 | 1.00 |
| time since the Ip rise (s) | DIII-D test | 0.257 | 2.625 | 6.582 | 1.00 |
| time since the Ip rise (s) | MAST | 0.073 | 0.248 | 0.529 | 0.07 |
| resistive_age | DIII-D train | 0.153 | 0.805 | 2.804 | 1.00 |
| resistive_age | DIII-D test | 0.153 | 0.799 | 2.472 | 0.87 |
| resistive_age | MAST | 0.060 | 0.181 | 0.627 | 0.21 |

Changing current (open item 6): share of frames off the flat-top, and share with abs(`ip_rate_rel`) above each threshold (Ip changing by that fraction of itself per resistive time).

- DIII-D train: off the flat-top 0.222; above 0.1 0.870, above 0.3 0.687, above 1 0.392
- DIII-D test: off the flat-top 0.227; above 0.1 0.874, above 0.3 0.693, above 1 0.389
- MAST: off the flat-top 0.248; above 0.1 0.939, above 0.3 0.824, above 1 0.595

`ip_rise_ms` per shot (10 percent crossing):

- DIII-D train: 5821 shots, 1st 8.1, median 14.9, 99th 44.8 ms
- DIII-D test: 874 shots, 1st 8.6, median 14.9, 99th 35.1 ms
- MAST: 1206 shots, 1st 1.4, median 2.2, 99th 13.5 ms

Runtime 30 s.
