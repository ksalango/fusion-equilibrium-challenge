# Frame table report

Build at 2026-10-06T18:41:50+00:00 on commit 12da37629d4baa6b091a03dbc2cb6adf63c98a1c (starter kit 5b467988d2e5f28e20be490a926d0a71649de28d), runtime 329 s.
Arguments: ``.

| Id | Check | Status | Headline |
| --- | --- | --- | --- |
| T1 | Row and shot counts | pass | 9121 shots, 1821774 frames |
| T2 | Keys | pass | unique keys, increasing time |
| T3 | Labels untouched | pass | exact copies |
| T4 | Input completeness | review | see per split |
| T5 | Shot-level flags against QA | pass | rates match, directions match, NaN count matches |
| T6 | Ip at the first frame | pass | minimum 0.172 |
| T7 | Thomson coverage against QA | review | QA: about 97% covered; 53 / 6 / 1 shots without a profile |
| T8 | Resampling forms agree | review | |nearest - mean| / std(mean) |

## T1 Row and shot counts (pass)

- diii_d_train: 7041 shots, 1559340 frames, 0 errors (expected 7041 / 1559340)
- diii_d_public_test: 874 shots, 193359 frames, 0 errors (expected 874 / 193359)
- mast_public_test: 1206 shots, 69075 frames, 0 errors (expected 1206 / 69075)

## T2 Keys (pass)

- diii_d_train: 0 duplicate keys, 0 non-increasing time steps
- diii_d_public_test: 0 duplicate keys, 0 non-increasing time steps
- mast_public_test: 0 duplicate keys, 0 non-increasing time steps

## T3 Labels untouched (pass)

- diii_d_train: 50 shots compared, 0 mismatches
- diii_d_public_test: 0 label columns
- mast_public_test: 0 label columns

## T4 Input completeness (review)

- diii_d_train: 1.0000 of frames complete; 0 shots with an incomplete frame
- diii_d_public_test: 1.0000 of frames complete; 0 shots with an incomplete frame
- mast_public_test: 1.0000 of frames complete; 0 shots with an incomplete frame

## T5 Shot-level flags against QA (pass)

- DIII-D recording rates: {2000: 2397, 20000: 5518}
- MAST recording rates: {1900: 114, 5000: 1092}
- DIII-D Ip direction: {'diii_d_train': {np.int64(1): 6420, np.int64(-1): 603, np.int64(0): 18}, 'diii_d_public_test': {np.int64(1): 798, np.int64(-1): 76}}
- MAST shots with NaN currents: 114
- reference (QA table /Users/kamia/Desktop/PROJECTS/fusion-challenge/fusion-equilibrium-challenge/reports/qa/qa_per_shot.parquet): rates {20000: 5518, 2000: 2397}, directions {'diii_d_train': {-1: 603, 0: 18, 1: 6420}, 'diii_d_public_test': {-1: 76, 1: 798}}, MAST NaN shots 114

## T6 Ip at the first frame (pass)

- diii_d_train: minimum 0.172; 0 shots below 0.15
- diii_d_public_test: minimum 0.194; 0 shots below 0.15

## T7 Thomson coverage against QA (review)

- diii_d_train: 0.9672 of shots with ts_core_valid on at least 90% of frames; 62 shots with no valid core profile
- diii_d_public_test: 0.9703 of shots with ts_core_valid on at least 90% of frames; 6 shots with no valid core profile
- mast_public_test: 0.9726 of shots with ts_core_valid on at least 90% of frames; 1 shots with no valid core profile

## T8 Resampling forms agree (review)

- diii_d_train at 2000 Hz (495981 frames): ip p50 0.0072 p99 0.0345; magnetics_F1A p50 0.0087 p99 0.0364; magnetics_F1B p50 0.0075 p99 0.0320; magnetics_F2A p50 0.0130 p99 0.0668; magnetics_F2B p50 0.0130 p99 0.0680; magnetics_F3A p50 0.0073 p99 0.0373; magnetics_F3B p50 0.0049 p99 0.0289; magnetics_F4A p50 0.0077 p99 0.0331; magnetics_F4B p50 0.0062 p99 0.0353; magnetics_F5A p50 0.0054 p99 0.0240; magnetics_F5B p50 0.0043 p99 0.0190; magnetics_F6A p50 0.0050 p99 0.0278; magnetics_F6B p50 0.0040 p99 0.0265; magnetics_F7A p50 0.0037 p99 0.0207; magnetics_F7B p50 0.0025 p99 0.0168; magnetics_F8A p50 0.0050 p99 0.0244; magnetics_F8B p50 0.0032 p99 0.0181; magnetics_F9A p50 0.0039 p99 0.0223; magnetics_F9B p50 0.0020 p99 0.0121; sol p50 0.0015 p99 0.0071; tf p50 0.0038 p99 0.0210
- diii_d_train at 20000 Hz (1063359 frames): ip p50 0.0037 p99 0.0172; magnetics_F1A p50 0.0086 p99 0.0307; magnetics_F1B p50 0.0079 p99 0.0270; magnetics_F2A p50 0.0109 p99 0.0433; magnetics_F2B p50 0.0081 p99 0.0323; magnetics_F3A p50 0.0056 p99 0.0297; magnetics_F3B p50 0.0056 p99 0.0278; magnetics_F4A p50 0.0048 p99 0.0250; magnetics_F4B p50 0.0060 p99 0.0257; magnetics_F5A p50 0.0038 p99 0.0173; magnetics_F5B p50 0.0039 p99 0.0162; magnetics_F6A p50 0.0030 p99 0.0142; magnetics_F6B p50 0.0029 p99 0.0136; magnetics_F7A p50 0.0014 p99 0.0106; magnetics_F7B p50 0.0020 p99 0.0106; magnetics_F8A p50 0.0036 p99 0.0165; magnetics_F8B p50 0.0023 p99 0.0115; magnetics_F9A p50 0.0020 p99 0.0150; magnetics_F9B p50 0.0011 p99 0.0092; sol p50 0.0013 p99 0.0052; tf p50 0.0025 p99 0.0175
- diii_d_public_test at 2000 Hz (68268 frames): ip p50 0.0060 p99 0.0289; magnetics_F1A p50 0.0082 p99 0.0346; magnetics_F1B p50 0.0069 p99 0.0315; magnetics_F2A p50 0.0110 p99 0.0571; magnetics_F2B p50 0.0107 p99 0.0579; magnetics_F3A p50 0.0069 p99 0.0347; magnetics_F3B p50 0.0047 p99 0.0261; magnetics_F4A p50 0.0071 p99 0.0298; magnetics_F4B p50 0.0057 p99 0.0331; magnetics_F5A p50 0.0048 p99 0.0211; magnetics_F5B p50 0.0038 p99 0.0163; magnetics_F6A p50 0.0043 p99 0.0237; magnetics_F6B p50 0.0035 p99 0.0228; magnetics_F7A p50 0.0032 p99 0.0182; magnetics_F7B p50 0.0023 p99 0.0155; magnetics_F8A p50 0.0043 p99 0.0212; magnetics_F8B p50 0.0030 p99 0.0170; magnetics_F9A p50 0.0037 p99 0.0216; magnetics_F9B p50 0.0020 p99 0.0122; sol p50 0.0015 p99 0.0074; tf p50 0.0038 p99 0.0210
- diii_d_public_test at 20000 Hz (125091 frames): ip p50 0.0037 p99 0.0172; magnetics_F1A p50 0.0082 p99 0.0299; magnetics_F1B p50 0.0076 p99 0.0264; magnetics_F2A p50 0.0101 p99 0.0408; magnetics_F2B p50 0.0076 p99 0.0307; magnetics_F3A p50 0.0055 p99 0.0304; magnetics_F3B p50 0.0056 p99 0.0274; magnetics_F4A p50 0.0047 p99 0.0239; magnetics_F4B p50 0.0058 p99 0.0254; magnetics_F5A p50 0.0036 p99 0.0170; magnetics_F5B p50 0.0037 p99 0.0158; magnetics_F6A p50 0.0030 p99 0.0143; magnetics_F6B p50 0.0028 p99 0.0134; magnetics_F7A p50 0.0013 p99 0.0101; magnetics_F7B p50 0.0020 p99 0.0102; magnetics_F8A p50 0.0035 p99 0.0161; magnetics_F8B p50 0.0023 p99 0.0111; magnetics_F9A p50 0.0021 p99 0.0152; magnetics_F9B p50 0.0011 p99 0.0090; sol p50 0.0013 p99 0.0052; tf p50 0.0026 p99 0.0179
- mast_public_test at 1900 Hz (6136 frames): ip p50 0.0044 p99 0.0238; magnetics_efps_current p50 0.0106 p99 0.0582; magnetics_p2l_current p50 0.0036 p99 0.0141; magnetics_p2u_current p50 0.0018 p99 0.0075; magnetics_p3l_current p50 0.1518 p99 1.4302; magnetics_p3u_current p50 0.1581 p99 1.8455; magnetics_p4l_current p50 0.0044 p99 0.0139; magnetics_p4u_current p50 0.0045 p99 0.0138; magnetics_p5l_current p50 0.0069 p99 0.0430; magnetics_p5u_current p50 0.0055 p99 0.0239; magnetics_p6l_current p50 0.0273 p99 0.1492; magnetics_p6u_current p50 0.0272 p99 0.1579; sol p50 0.0012 p99 0.0049; tf p50 0.0000 p99 0.0140
- mast_public_test at 5000 Hz (62939 frames): ip p50 0.0041 p99 0.0293; magnetics_efps_current p50 0.0085 p99 0.0329; magnetics_p2l_current p50 0.0034 p99 0.0105; magnetics_p2u_current p50 0.0031 p99 0.0096; magnetics_p3l_current p50 0.1633 p99 2.2454; magnetics_p3u_current p50 0.1138 p99 1.8498; magnetics_p4l_current p50 0.0030 p99 0.0101; magnetics_p4u_current p50 0.0030 p99 0.0105; magnetics_p5l_current p50 0.0059 p99 0.0644; magnetics_p5u_current p50 0.0053 p99 0.0463; magnetics_p6l_current p50 0.0358 p99 0.2574; magnetics_p6u_current p50 0.0384 p99 0.2692; sol p50 0.0011 p99 0.0047; tf p50 0.0047 p99 0.0193

## Missing values per column

### diii_d_train

| Column | Role | Missing | Fraction | Zeros |
| --- | --- | --- | --- | --- |
| machine | key | 0 | 0.0000 |  |
| split | key | 0 | 0.0000 |  |
| shot | key | 0 | 0.0000 |  |
| frame | key | 0 | 0.0000 | 7041 |
| time_ms | key | 0 | 0.0000 | 0 |
| n_frames | key | 0 | 0.0000 | 0 |
| dt_prev_ms | key | 7041 | 0.0045 | 0 |
| ip__nearest | input | 0 | 0.0000 | 0 |
| ip__mean | input | 0 | 0.0000 | 0 |
| ip__rate | input | 0 | 0.0000 | 0 |
| sol__nearest | input | 0 | 0.0000 | 184 |
| sol__mean | input | 0 | 0.0000 | 12 |
| sol__rate | input | 0 | 0.0000 | 3 |
| tf__nearest | input | 0 | 0.0000 | 0 |
| tf__mean | input | 0 | 0.0000 | 0 |
| tf__rate | input | 0 | 0.0000 | 0 |
| magnetics_F1A__nearest | input | 0 | 0.0000 | 134 |
| magnetics_F1A__mean | input | 0 | 0.0000 | 3 |
| magnetics_F1A__rate | input | 0 | 0.0000 | 0 |
| magnetics_F1B__nearest | input | 0 | 0.0000 | 132 |
| magnetics_F1B__mean | input | 0 | 0.0000 | 1 |
| magnetics_F1B__rate | input | 0 | 0.0000 | 1 |
| magnetics_F2A__nearest | input | 0 | 0.0000 | 379 |
| magnetics_F2A__mean | input | 0 | 0.0000 | 5 |
| magnetics_F2A__rate | input | 0 | 0.0000 | 1 |
| magnetics_F2B__nearest | input | 0 | 0.0000 | 575 |
| magnetics_F2B__mean | input | 0 | 0.0000 | 16 |
| magnetics_F2B__rate | input | 0 | 0.0000 | 2 |
| magnetics_F3A__nearest | input | 0 | 0.0000 | 3222 |
| magnetics_F3A__mean | input | 0 | 0.0000 | 97 |
| magnetics_F3A__rate | input | 0 | 0.0000 | 2 |
| magnetics_F3B__nearest | input | 0 | 0.0000 | 2776 |
| magnetics_F3B__mean | input | 0 | 0.0000 | 255 |
| magnetics_F3B__rate | input | 0 | 0.0000 | 2 |
| magnetics_F4A__nearest | input | 0 | 0.0000 | 845 |
| magnetics_F4A__mean | input | 0 | 0.0000 | 16 |
| magnetics_F4A__rate | input | 0 | 0.0000 | 0 |
| magnetics_F4B__nearest | input | 0 | 0.0000 | 872 |
| magnetics_F4B__mean | input | 0 | 0.0000 | 40 |
| magnetics_F4B__rate | input | 0 | 0.0000 | 2 |
| magnetics_F5A__nearest | input | 0 | 0.0000 | 990 |
| magnetics_F5A__mean | input | 0 | 0.0000 | 5 |
| magnetics_F5A__rate | input | 0 | 0.0000 | 0 |
| magnetics_F5B__nearest | input | 0 | 0.0000 | 1145 |
| magnetics_F5B__mean | input | 0 | 0.0000 | 6 |
| magnetics_F5B__rate | input | 0 | 0.0000 | 0 |
| magnetics_F6A__nearest | input | 0 | 0.0000 | 0 |
| magnetics_F6A__mean | input | 0 | 0.0000 | 0 |
| magnetics_F6A__rate | input | 0 | 0.0000 | 0 |
| magnetics_F6B__nearest | input | 0 | 0.0000 | 2 |
| magnetics_F6B__mean | input | 0 | 0.0000 | 0 |
| magnetics_F6B__rate | input | 0 | 0.0000 | 0 |
| magnetics_F7A__nearest | input | 0 | 0.0000 | 12 |
| magnetics_F7A__mean | input | 0 | 0.0000 | 0 |
| magnetics_F7A__rate | input | 0 | 0.0000 | 0 |
| magnetics_F7B__nearest | input | 0 | 0.0000 | 0 |
| magnetics_F7B__mean | input | 0 | 0.0000 | 0 |
| magnetics_F7B__rate | input | 0 | 0.0000 | 1 |
| magnetics_F8A__nearest | input | 0 | 0.0000 | 1456 |
| magnetics_F8A__mean | input | 0 | 0.0000 | 55 |
| magnetics_F8A__rate | input | 0 | 0.0000 | 2 |
| magnetics_F8B__nearest | input | 0 | 0.0000 | 367 |
| magnetics_F8B__mean | input | 0 | 0.0000 | 32 |
| magnetics_F8B__rate | input | 0 | 0.0000 | 2 |
| magnetics_F9A__nearest | input | 0 | 0.0000 | 492 |
| magnetics_F9A__mean | input | 0 | 0.0000 | 43 |
| magnetics_F9A__rate | input | 0 | 0.0000 | 4 |
| magnetics_F9B__nearest | input | 0 | 0.0000 | 1276 |
| magnetics_F9B__mean | input | 0 | 0.0000 | 36 |
| magnetics_F9B__rate | input | 0 | 0.0000 | 6 |
| ts_core_te_peak__interp | input | 6842 | 0.0044 | 0 |
| ts_core_te_peak__nearest | input | 6842 | 0.0044 | 0 |
| ts_core_ne_peak__interp | input | 6842 | 0.0044 | 0 |
| ts_core_ne_peak__nearest | input | 6842 | 0.0044 | 0 |
| ts_core_pe_peak__interp | input | 6842 | 0.0044 | 0 |
| ts_core_pe_peak__nearest | input | 6842 | 0.0044 | 0 |
| ts_core_pe_mean__interp | input | 6842 | 0.0044 | 0 |
| ts_core_pe_mean__nearest | input | 6842 | 0.0044 | 0 |
| ts_core_n_valid | input | 6842 | 0.0044 | 0 |
| ts_core_dt_ms | input | 6842 | 0.0044 | 15212 |
| ts_core_gap_ms | input | 13019 | 0.0083 | 0 |
| ts_edge_te_mean__interp | input | 10897 | 0.0070 | 0 |
| ts_edge_te_mean__nearest | input | 10897 | 0.0070 | 0 |
| ts_edge_ne_mean__interp | input | 10897 | 0.0070 | 0 |
| ts_edge_ne_mean__nearest | input | 10897 | 0.0070 | 0 |
| ts_edge_pe_mean__interp | input | 10897 | 0.0070 | 0 |
| ts_edge_pe_mean__nearest | input | 10897 | 0.0070 | 0 |
| ts_edge_n_valid | input | 10897 | 0.0070 | 0 |
| ts_edge_dt_ms | input | 10897 | 0.0070 | 43 |
| ts_edge_gap_ms | input | 12245 | 0.0079 | 0 |
| mag_rate_hz | flag | 0 | 0.0000 | 0 |
| mag_has_nan | flag | 0 | 0.0000 | 1559340 |
| ip_direction | flag | 0 | 0.0000 | 604 |
| tf_direction | flag | 0 | 0.0000 | 14 |
| ip_fix | flag | 0 | 0.0000 |  |
| ip_shift_ms | flag | 0 | 0.0000 | 0 |
| ip_over_peak | flag | 0 | 0.0000 | 0 |
| inputs_complete | flag | 0 | 0.0000 | 0 |
| ts_core_valid | flag | 0 | 0.0000 | 23066 |
| ts_edge_valid | flag | 0 | 0.0000 | 30613 |
| efit_q95 | label | 0 | 0.0000 | 0 |
| efit_beta_n | label | 0 | 0.0000 | 2 |
| efit_li | label | 0 | 0.0000 | 2 |
| efit_r_axis | label | 0 | 0.0000 | 0 |
| efit_z_axis | label | 0 | 0.0000 | 5 |
| efit_lcfs_n | label | 0 | 0.0000 | 0 |
| psi_min | label | 0 | 0.0000 | 0 |
| psi_max | label | 0 | 0.0000 | 0 |
| psi_mean | label | 0 | 0.0000 | 0 |

### diii_d_public_test

| Column | Role | Missing | Fraction | Zeros |
| --- | --- | --- | --- | --- |
| machine | key | 0 | 0.0000 |  |
| split | key | 0 | 0.0000 |  |
| shot | key | 0 | 0.0000 |  |
| frame | key | 0 | 0.0000 | 874 |
| time_ms | key | 0 | 0.0000 | 0 |
| n_frames | key | 0 | 0.0000 | 0 |
| dt_prev_ms | key | 874 | 0.0045 | 0 |
| ip__nearest | input | 0 | 0.0000 | 0 |
| ip__mean | input | 0 | 0.0000 | 0 |
| ip__rate | input | 0 | 0.0000 | 0 |
| sol__nearest | input | 0 | 0.0000 | 25 |
| sol__mean | input | 0 | 0.0000 | 1 |
| sol__rate | input | 0 | 0.0000 | 0 |
| tf__nearest | input | 0 | 0.0000 | 0 |
| tf__mean | input | 0 | 0.0000 | 0 |
| tf__rate | input | 0 | 0.0000 | 0 |
| magnetics_F1A__nearest | input | 0 | 0.0000 | 16 |
| magnetics_F1A__mean | input | 0 | 0.0000 | 0 |
| magnetics_F1A__rate | input | 0 | 0.0000 | 0 |
| magnetics_F1B__nearest | input | 0 | 0.0000 | 37 |
| magnetics_F1B__mean | input | 0 | 0.0000 | 0 |
| magnetics_F1B__rate | input | 0 | 0.0000 | 0 |
| magnetics_F2A__nearest | input | 0 | 0.0000 | 58 |
| magnetics_F2A__mean | input | 0 | 0.0000 | 0 |
| magnetics_F2A__rate | input | 0 | 0.0000 | 0 |
| magnetics_F2B__nearest | input | 0 | 0.0000 | 70 |
| magnetics_F2B__mean | input | 0 | 0.0000 | 5 |
| magnetics_F2B__rate | input | 0 | 0.0000 | 0 |
| magnetics_F3A__nearest | input | 0 | 0.0000 | 370 |
| magnetics_F3A__mean | input | 0 | 0.0000 | 19 |
| magnetics_F3A__rate | input | 0 | 0.0000 | 0 |
| magnetics_F3B__nearest | input | 0 | 0.0000 | 297 |
| magnetics_F3B__mean | input | 0 | 0.0000 | 36 |
| magnetics_F3B__rate | input | 0 | 0.0000 | 0 |
| magnetics_F4A__nearest | input | 0 | 0.0000 | 160 |
| magnetics_F4A__mean | input | 0 | 0.0000 | 0 |
| magnetics_F4A__rate | input | 0 | 0.0000 | 1 |
| magnetics_F4B__nearest | input | 0 | 0.0000 | 146 |
| magnetics_F4B__mean | input | 0 | 0.0000 | 2 |
| magnetics_F4B__rate | input | 0 | 0.0000 | 0 |
| magnetics_F5A__nearest | input | 0 | 0.0000 | 204 |
| magnetics_F5A__mean | input | 0 | 0.0000 | 5 |
| magnetics_F5A__rate | input | 0 | 0.0000 | 0 |
| magnetics_F5B__nearest | input | 0 | 0.0000 | 233 |
| magnetics_F5B__mean | input | 0 | 0.0000 | 2 |
| magnetics_F5B__rate | input | 0 | 0.0000 | 0 |
| magnetics_F6A__nearest | input | 0 | 0.0000 | 0 |
| magnetics_F6A__mean | input | 0 | 0.0000 | 0 |
| magnetics_F6A__rate | input | 0 | 0.0000 | 0 |
| magnetics_F6B__nearest | input | 0 | 0.0000 | 0 |
| magnetics_F6B__mean | input | 0 | 0.0000 | 0 |
| magnetics_F6B__rate | input | 0 | 0.0000 | 0 |
| magnetics_F7A__nearest | input | 0 | 0.0000 | 0 |
| magnetics_F7A__mean | input | 0 | 0.0000 | 0 |
| magnetics_F7A__rate | input | 0 | 0.0000 | 0 |
| magnetics_F7B__nearest | input | 0 | 0.0000 | 0 |
| magnetics_F7B__mean | input | 0 | 0.0000 | 0 |
| magnetics_F7B__rate | input | 0 | 0.0000 | 0 |
| magnetics_F8A__nearest | input | 0 | 0.0000 | 183 |
| magnetics_F8A__mean | input | 0 | 0.0000 | 3 |
| magnetics_F8A__rate | input | 0 | 0.0000 | 2 |
| magnetics_F8B__nearest | input | 0 | 0.0000 | 26 |
| magnetics_F8B__mean | input | 0 | 0.0000 | 1 |
| magnetics_F8B__rate | input | 0 | 0.0000 | 1 |
| magnetics_F9A__nearest | input | 0 | 0.0000 | 49 |
| magnetics_F9A__mean | input | 0 | 0.0000 | 1 |
| magnetics_F9A__rate | input | 0 | 0.0000 | 0 |
| magnetics_F9B__nearest | input | 0 | 0.0000 | 126 |
| magnetics_F9B__mean | input | 0 | 0.0000 | 6 |
| magnetics_F9B__rate | input | 0 | 0.0000 | 1 |
| ts_core_te_peak__interp | input | 657 | 0.0034 | 0 |
| ts_core_te_peak__nearest | input | 657 | 0.0034 | 0 |
| ts_core_ne_peak__interp | input | 657 | 0.0034 | 0 |
| ts_core_ne_peak__nearest | input | 657 | 0.0034 | 0 |
| ts_core_pe_peak__interp | input | 657 | 0.0034 | 0 |
| ts_core_pe_peak__nearest | input | 657 | 0.0034 | 0 |
| ts_core_pe_mean__interp | input | 657 | 0.0034 | 0 |
| ts_core_pe_mean__nearest | input | 657 | 0.0034 | 0 |
| ts_core_n_valid | input | 657 | 0.0034 | 0 |
| ts_core_dt_ms | input | 657 | 0.0034 | 1933 |
| ts_core_gap_ms | input | 1550 | 0.0080 | 0 |
| ts_edge_te_mean__interp | input | 934 | 0.0048 | 0 |
| ts_edge_te_mean__nearest | input | 934 | 0.0048 | 0 |
| ts_edge_ne_mean__interp | input | 934 | 0.0048 | 0 |
| ts_edge_ne_mean__nearest | input | 934 | 0.0048 | 0 |
| ts_edge_pe_mean__interp | input | 934 | 0.0048 | 0 |
| ts_edge_pe_mean__nearest | input | 934 | 0.0048 | 0 |
| ts_edge_n_valid | input | 934 | 0.0048 | 0 |
| ts_edge_dt_ms | input | 934 | 0.0048 | 2 |
| ts_edge_gap_ms | input | 1123 | 0.0058 | 0 |
| mag_rate_hz | flag | 0 | 0.0000 | 0 |
| mag_has_nan | flag | 0 | 0.0000 | 193359 |
| ip_direction | flag | 0 | 0.0000 | 0 |
| tf_direction | flag | 0 | 0.0000 | 0 |
| ip_fix | flag | 0 | 0.0000 |  |
| ip_shift_ms | flag | 0 | 0.0000 | 0 |
| ip_over_peak | flag | 0 | 0.0000 | 0 |
| inputs_complete | flag | 0 | 0.0000 | 0 |
| ts_core_valid | flag | 0 | 0.0000 | 2540 |
| ts_edge_valid | flag | 0 | 0.0000 | 4466 |

### mast_public_test

| Column | Role | Missing | Fraction | Zeros |
| --- | --- | --- | --- | --- |
| machine | key | 0 | 0.0000 |  |
| split | key | 0 | 0.0000 |  |
| shot | key | 0 | 0.0000 |  |
| frame | key | 0 | 0.0000 | 1206 |
| time_ms | key | 0 | 0.0000 | 0 |
| n_frames | key | 0 | 0.0000 | 0 |
| dt_prev_ms | key | 1206 | 0.0175 | 0 |
| ip__nearest | input | 0 | 0.0000 | 0 |
| ip__mean | input | 0 | 0.0000 | 0 |
| ip__rate | input | 0 | 0.0000 | 0 |
| sol__nearest | input | 0 | 0.0000 | 0 |
| sol__mean | input | 0 | 0.0000 | 0 |
| sol__rate | input | 0 | 0.0000 | 0 |
| tf__nearest | input | 0 | 0.0000 | 0 |
| tf__mean | input | 0 | 0.0000 | 0 |
| tf__rate | input | 0 | 0.0000 | 35 |
| magnetics_p2l_current__nearest | input | 0 | 0.0000 | 0 |
| magnetics_p2l_current__mean | input | 0 | 0.0000 | 0 |
| magnetics_p2l_current__rate | input | 0 | 0.0000 | 0 |
| magnetics_p2u_current__nearest | input | 0 | 0.0000 | 0 |
| magnetics_p2u_current__mean | input | 0 | 0.0000 | 0 |
| magnetics_p2u_current__rate | input | 0 | 0.0000 | 0 |
| magnetics_p3l_current__nearest | input | 0 | 0.0000 | 0 |
| magnetics_p3l_current__mean | input | 0 | 0.0000 | 0 |
| magnetics_p3l_current__rate | input | 0 | 0.0000 | 0 |
| magnetics_p3u_current__nearest | input | 0 | 0.0000 | 0 |
| magnetics_p3u_current__mean | input | 0 | 0.0000 | 0 |
| magnetics_p3u_current__rate | input | 0 | 0.0000 | 0 |
| magnetics_p4l_current__nearest | input | 0 | 0.0000 | 0 |
| magnetics_p4l_current__mean | input | 0 | 0.0000 | 0 |
| magnetics_p4l_current__rate | input | 0 | 0.0000 | 0 |
| magnetics_p4u_current__nearest | input | 0 | 0.0000 | 0 |
| magnetics_p4u_current__mean | input | 0 | 0.0000 | 0 |
| magnetics_p4u_current__rate | input | 0 | 0.0000 | 0 |
| magnetics_p5l_current__nearest | input | 0 | 0.0000 | 0 |
| magnetics_p5l_current__mean | input | 0 | 0.0000 | 0 |
| magnetics_p5l_current__rate | input | 0 | 0.0000 | 0 |
| magnetics_p5u_current__nearest | input | 0 | 0.0000 | 0 |
| magnetics_p5u_current__mean | input | 0 | 0.0000 | 0 |
| magnetics_p5u_current__rate | input | 0 | 0.0000 | 0 |
| magnetics_p6l_current__nearest | input | 0 | 0.0000 | 0 |
| magnetics_p6l_current__mean | input | 0 | 0.0000 | 0 |
| magnetics_p6l_current__rate | input | 0 | 0.0000 | 0 |
| magnetics_p6u_current__nearest | input | 0 | 0.0000 | 0 |
| magnetics_p6u_current__mean | input | 0 | 0.0000 | 0 |
| magnetics_p6u_current__rate | input | 0 | 0.0000 | 0 |
| magnetics_efps_current__nearest | input | 0 | 0.0000 | 0 |
| magnetics_efps_current__mean | input | 0 | 0.0000 | 0 |
| magnetics_efps_current__rate | input | 0 | 0.0000 | 0 |
| ts_core_te_peak__interp | input | 52 | 0.0008 | 0 |
| ts_core_te_peak__nearest | input | 52 | 0.0008 | 0 |
| ts_core_ne_peak__interp | input | 52 | 0.0008 | 0 |
| ts_core_ne_peak__nearest | input | 52 | 0.0008 | 0 |
| ts_core_pe_peak__interp | input | 52 | 0.0008 | 0 |
| ts_core_pe_peak__nearest | input | 52 | 0.0008 | 0 |
| ts_core_pe_mean__interp | input | 52 | 0.0008 | 0 |
| ts_core_pe_mean__nearest | input | 52 | 0.0008 | 0 |
| ts_core_n_valid | input | 52 | 0.0008 | 0 |
| ts_core_dt_ms | input | 52 | 0.0008 | 662 |
| ts_core_gap_ms | input | 812 | 0.0118 | 0 |
| ts_edge_te_mean__interp | input | 0 | 0.0000 | 0 |
| ts_edge_te_mean__nearest | input | 0 | 0.0000 | 0 |
| ts_edge_ne_mean__interp | input | 0 | 0.0000 | 85 |
| ts_edge_ne_mean__nearest | input | 0 | 0.0000 | 85 |
| ts_edge_pe_mean__interp | input | 0 | 0.0000 | 85 |
| ts_edge_pe_mean__nearest | input | 0 | 0.0000 | 85 |
| ts_edge_n_valid | input | 0 | 0.0000 | 0 |
| ts_edge_dt_ms | input | 0 | 0.0000 | 660 |
| ts_edge_gap_ms | input | 1731 | 0.0251 | 0 |
| mag_rate_hz | flag | 0 | 0.0000 | 0 |
| mag_has_nan | flag | 0 | 0.0000 | 62939 |
| ip_direction | flag | 0 | 0.0000 | 0 |
| tf_direction | flag | 0 | 0.0000 | 0 |
| ip_fix | flag | 0 | 0.0000 |  |
| ip_shift_ms | flag | 0 | 0.0000 | 69075 |
| ip_over_peak | flag | 0 | 0.0000 | 0 |
| inputs_complete | flag | 0 | 0.0000 | 0 |
| ts_core_valid | flag | 0 | 0.0000 | 621 |
| ts_edge_valid | flag | 0 | 0.0000 | 1275 |

