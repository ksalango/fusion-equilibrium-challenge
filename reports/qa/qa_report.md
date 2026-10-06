# QA report

Run at 2026-10-06T00:00:55+00:00 on commit cd19432ce6f2c3f84b3329777e047a9ab9360663 (starter kit 5b467988d2e5f28e20be490a926d0a71649de28d), runtime 102 s.
Arguments: `--out reports/qa`.
Shots per split: diii_d_public_test: 874, diii_d_train: 7041, mast_public_test: 1206, mast_samples: 3.
Shots with at least one error: 0.

## Summary

| Id | Check | Status | Headline | Note |
|---|---|---|---|---|
| 0.1 | File inventory | pass | diii_d_public_test: 874, diii_d_train: 7041, mast_public_test: 1206, mast_samples: 3 |  |
| 0.12 | Split integrity | pass | 0 shot ids shared between splits |  |
| 0.14 | Column sets | review | DIII-D label set matches the expected 11 columns |  |
| 0.2 | Ip time axis | pass | 69.7% of 7915 DIII-D shots corrected by the organizers' fix, 30.3% aligned by rule B; 0 shot(s) fail; 135 for review |  |
| 0.3 | MAST Thomson alignment | review | fix applied on 100.0% of 1209 MAST shots |  |
| 0.4 | Current units | review | DIII-D Ip p99 |value| median 1060; MAST Ip p99 |value| median 751.4 |  |
| 0.5 | Flux map indexing | pass | 0 shape failures; 0 shots with the scorer O-point more than one cell from the axis; 23 shots with a border extremum for review |  |
| 0.10 | Grid | pass | one grid per machine |  |
| 0.9 | Coil geometry | pass | one geometry per machine |  |
| 0.6 | Boundary contour | review | 0 shots flagged |  |
| 0.7 | Missing Thomson | review | DIII-D core profile within 20 ms: median 1.00 of frames; MAST core profile within 20 ms: median 1.00 of frames |  |
| 0.8 | magnetics_dsep | review | DIII-D: 0 sentinel values; MAST: 21 sentinel values |  |
| 0.11 | Labels and time bases | pass | 0 shot(s) fail; 49 shots with fewer than 10 frames |  |
| 0.11b | Test frame totals | pass | totals match |  |
| 0.15 | Scorer comparison | review | DIII-D: Hausdorff median 0.05 m, li ratio median 1.011; MAST: Hausdorff median 0.052 m, li ratio median 0.964 |  |
| 0.13 | README against data | review | 9 of 9 contradictions assessed |  |

## Checks

### 0.1 File inventory (pass)

diii_d_public_test: 874, diii_d_train: 7041, mast_public_test: 1206, mast_samples: 3

| split | files | open | one row | size GB | expected |
|---|---|---|---|---|---|
| diii_d_public_test | 874 | 874 | 874 | 7.19 | 874 |
| diii_d_train | 7041 | 7041 | 7041 | 93.43 | 7041 |
| mast_public_test | 1206 | 1206 | 1206 | 3.24 | 1206 |
| mast_samples | 3 | 3 | 3 | 0.01 | 3 |

### 0.12 Split integrity (pass)

0 shot ids shared between splits

Shot ids per split: diii_d_public_test: 874, diii_d_train: 7041, mast_public_test: 1206, mast_samples: 3
- diii_d_public_test and diii_d_train: 0 shared shot ids
- diii_d_public_test and mast_public_test: 0 shared shot ids
- diii_d_public_test and mast_samples: 0 shared shot ids
- diii_d_train and mast_public_test: 0 shared shot ids
- diii_d_train and mast_samples: 0 shared shot ids
- mast_public_test and mast_samples: 0 shared shot ids

### 0.14 Column sets (review)

DIII-D label set matches the expected 11 columns

- diii_d_public_test: 1 distinct column set(s), 46 columns
- diii_d_train: 1 distinct column set(s), 57 columns
- mast_public_test: 1 distinct column set(s), 38 columns
- mast_samples: 1 distinct column set(s), 55 columns
- DIII-D label set (train minus test): efit_beta_n, efit_lcfs_n, efit_lcfs_r, efit_lcfs_z, efit_li, efit_psirz, efit_q95, efit_r_axis, efit_z_axis, magnetics_dsep, magnetics_dsep_times
- Columns in DIII-D test but not train: none
- MAST label set (samples minus test): efit_beta_n, efit_beta_p, efit_elongation, efit_geom_axis_r, efit_geom_axis_z, efit_lcfs_n, efit_lcfs_r, efit_lcfs_z, efit_li, efit_psi_axis, efit_psi_boundary, efit_psirz, efit_q95, efit_r_axis, efit_z_axis, magnetics_dsep, magnetics_dsep_times
- Columns in MAST test but not in the samples: none
- Inputs only on DIII-D: ['magnetics_ECOILA', 'magnetics_F1A', 'magnetics_F1B', 'magnetics_F2A', 'magnetics_F2B', 'magnetics_F3A', 'magnetics_F3B', 'magnetics_F4A', 'magnetics_F4B', 'magnetics_F5A', 'magnetics_F5B', 'magnetics_F6A', 'magnetics_F6B', 'magnetics_F7A', 'magnetics_F7B', 'magnetics_F8A', 'magnetics_F8B', 'magnetics_F9A', 'magnetics_F9B', 'magnetics_bcoil', 'magnetics_plasma_current_times']
- Inputs only on MAST: ['magnetics_efps_current', 'magnetics_p2l_current', 'magnetics_p2u_current', 'magnetics_p3l_current', 'magnetics_p3u_current', 'magnetics_p4l_current', 'magnetics_p4u_current', 'magnetics_p5l_current', 'magnetics_p5u_current', 'magnetics_p6l_current', 'magnetics_p6u_current', 'magnetics_sol_current', 'magnetics_tf_current']
- diii_d_public_test columns: coil_R, coil_Z, coil_angle1, coil_angle2, coil_height, coil_input_column, coil_name, coil_width, efit_grid_R, efit_grid_Z, efit_times, magnetics_ECOILA, magnetics_F1A, magnetics_F1B, magnetics_F2A, magnetics_F2B, magnetics_F3A, magnetics_F3B, magnetics_F4A, magnetics_F4B, magnetics_F5A, magnetics_F5B, magnetics_F6A, magnetics_F6B, magnetics_F7A, magnetics_F7B, magnetics_F8A, magnetics_F8B, magnetics_F9A, magnetics_F9B, magnetics_bcoil, magnetics_plasma_current, magnetics_plasma_current_times, magnetics_time, source, thomson_chord_R, thomson_chord_Z, thomson_chord_name, thomson_core_R, thomson_core_Te, thomson_core_ne, thomson_core_times, thomson_edge_Te, thomson_edge_ne, thomson_edge_spatial, thomson_edge_times
- diii_d_train columns: coil_R, coil_Z, coil_angle1, coil_angle2, coil_height, coil_input_column, coil_name, coil_width, efit_beta_n, efit_grid_R, efit_grid_Z, efit_lcfs_n, efit_lcfs_r, efit_lcfs_z, efit_li, efit_psirz, efit_q95, efit_r_axis, efit_times, efit_z_axis, magnetics_ECOILA, magnetics_F1A, magnetics_F1B, magnetics_F2A, magnetics_F2B, magnetics_F3A, magnetics_F3B, magnetics_F4A, magnetics_F4B, magnetics_F5A, magnetics_F5B, magnetics_F6A, magnetics_F6B, magnetics_F7A, magnetics_F7B, magnetics_F8A, magnetics_F8B, magnetics_F9A, magnetics_F9B, magnetics_bcoil, magnetics_dsep, magnetics_dsep_times, magnetics_plasma_current, magnetics_plasma_current_times, magnetics_time, source, thomson_chord_R, thomson_chord_Z, thomson_chord_name, thomson_core_R, thomson_core_Te, thomson_core_ne, thomson_core_times, thomson_edge_Te, thomson_edge_ne, thomson_edge_spatial, thomson_edge_times
- mast_public_test columns: coil_R, coil_Z, coil_angle1, coil_angle2, coil_height, coil_input_column, coil_name, coil_width, efit_grid_R, efit_grid_Z, efit_times, magnetics_efps_current, magnetics_p2l_current, magnetics_p2u_current, magnetics_p3l_current, magnetics_p3u_current, magnetics_p4l_current, magnetics_p4u_current, magnetics_p5l_current, magnetics_p5u_current, magnetics_p6l_current, magnetics_p6u_current, magnetics_plasma_current, magnetics_sol_current, magnetics_tf_current, magnetics_time, source, thomson_chord_R, thomson_chord_Z, thomson_chord_name, thomson_core_R, thomson_core_Te, thomson_core_ne, thomson_core_times, thomson_edge_Te, thomson_edge_ne, thomson_edge_spatial, thomson_edge_times
- mast_samples columns: coil_R, coil_Z, coil_angle1, coil_angle2, coil_height, coil_input_column, coil_name, coil_width, efit_beta_n, efit_beta_p, efit_elongation, efit_geom_axis_r, efit_geom_axis_z, efit_grid_R, efit_grid_Z, efit_lcfs_n, efit_lcfs_r, efit_lcfs_z, efit_li, efit_psi_axis, efit_psi_boundary, efit_psirz, efit_q95, efit_r_axis, efit_times, efit_z_axis, magnetics_dsep, magnetics_dsep_times, magnetics_efps_current, magnetics_p2l_current, magnetics_p2u_current, magnetics_p3l_current, magnetics_p3u_current, magnetics_p4l_current, magnetics_p4u_current, magnetics_p5l_current, magnetics_p5u_current, magnetics_p6l_current, magnetics_p6u_current, magnetics_plasma_current, magnetics_sol_current, magnetics_tf_current, magnetics_time, source, thomson_chord_R, thomson_chord_Z, thomson_chord_name, thomson_core_R, thomson_core_Te, thomson_core_ne, thomson_core_times, thomson_edge_Te, thomson_edge_ne, thomson_edge_spatial, thomson_edge_times

### 0.2 Ip time axis (pass)

69.7% of 7915 DIII-D shots corrected by the organizers' fix, 30.3% aligned by rule B; 0 shot(s) fail; 135 for review

- diii_d_public_test: organizers' fix corrected 67.0% of 874 shots; rule B alignment applied to 33.0%
- diii_d_train: organizers' fix corrected 70.0% of 7041 shots; rule B alignment applied to 30.0%
- Organizers' shift on corrected shots (ms): -3169.74: 1031, -3204.43: 603, -3275.21: 406, -3135.51: 372, -3239.58: 291, -3118.56: 246, -3257.34: 234, -3311.33: 193, ... 63 more
- Rule B shift on aligned shots (ms): min -332.9, median -173.4, max 55.59; positive on 2.2%
- Breakdown after the organizers' fix (ms), corrected shots: min -118.3, median 14.87, max 130.7; untouched shots: min -40.69, median 188.3, max 347.8
- Breakdown on the final axis (ms): min -118.3, median 14.9, max 130.7
- Breakdown on the shipped axis (ms), corrected shots: min 3080, median 3217, max 3456
- Median |Ip| at EFIT frames, final axis (kA): min 197.3, median 995, max 2123; shipped axis: min 0.1901, median 5.073, max 2072
- Ratio median |Ip| over p99.5 |Ip|: min 0.193, median 0.9824, max 0.9986
- |Ip| at the first EFIT frame over p99.5 |Ip|: min 0.172, median 0.463, max 1.027
- First EFIT frame minus breakdown (ms): min 49.27, median 228.7, max 5584; negative on 0 shots
- 99.5th percentile |Ip| (kA): min 498.5, median 1068, max 2205
- Final axis covers the EFIT window: 100.0%
- Rule A diagnostic: solenoid swing-down anchor (ms) min -519.6, median 1.322, max 6530; lag to fixed breakdown min -6315, median 15.9, max 530.9; A minus B shift (ms) min -521.1, median -0.1778, max 6529; |A minus B| above 10 ms on 88 shots; no anchor on 2 shots
- MAST (1209 shots): Ip covers EFIT window 100.0%; median |Ip| at EFIT frames (kA): min 400.9, median 692.7, max 929.8; first-frame ratio min 0.192, median 0.727, max 1.154; Ip NaN count: min 0, median 0, max 7982

Shots for review (135, showing up to 20):
- diii_d_public_test/17deff9e15: rule A and rule B shifts differ by more than 10 ms
- diii_d_public_test/2a8627f026: rule A and rule B shifts differ by more than 10 ms
- diii_d_public_test/3806f9a549: rule A and rule B shifts differ by more than 10 ms
- diii_d_public_test/6a7e6a0e38: rule A and rule B shifts differ by more than 10 ms
- diii_d_public_test/73e56964b1: rule A and rule B shifts differ by more than 10 ms
- diii_d_public_test/883a99778c: rule A and rule B shifts differ by more than 10 ms
- diii_d_public_test/a2d4b93190: rule A and rule B shifts differ by more than 10 ms
- diii_d_public_test/f529ba7944: rule A and rule B shifts differ by more than 10 ms
- diii_d_train/06450fdad2: rule A and rule B shifts differ by more than 10 ms
- diii_d_train/065af5741d: rule A and rule B shifts differ by more than 10 ms
- diii_d_train/08d0ee1ffa: rule A and rule B shifts differ by more than 10 ms
- diii_d_train/08ffcbc9bb: rule A and rule B shifts differ by more than 10 ms
- diii_d_train/0c7dfda639: rule A and rule B shifts differ by more than 10 ms
- diii_d_train/0ee67bbb0e: rule A and rule B shifts differ by more than 10 ms
- diii_d_train/179019cb31: rule A and rule B shifts differ by more than 10 ms
- diii_d_train/23bc448cdf: rule A and rule B shifts differ by more than 10 ms
- diii_d_train/2816ad1386: rule A and rule B shifts differ by more than 10 ms
- diii_d_train/331a7543a3: rule A and rule B shifts differ by more than 10 ms
- diii_d_train/3544120460: rule A and rule B shifts differ by more than 10 ms
- diii_d_train/355f5534b6: rule A and rule B shifts differ by more than 10 ms

### 0.3 MAST Thomson alignment (review)

fix applied on 100.0% of 1209 MAST shots

- Fix changed the shot: 100.0% of 1209 shots
- Layouts (before -> after, positions): 131 -> 130 channels, 130 positions: 1209
- Aligned after fix: 100.0%
- NaN fraction of the dropped channel, Te: min 1, median 1, max 1; ne: min 1, median 1, max 1
- DIII-D pass-through: fix changed 0 of 7915 shots; core channels 44: 7117, 42: 478, 40: 249, 54: 70, 43: 1

### 0.4 Current units (review)

DIII-D Ip p99 |value| median 1060; MAST Ip p99 |value| median 751.4

**DIII-D** (values as stored, inside the EFIT window)

| column | shots | min of min | max of max | median p99 abs | max p99 abs | max NaN frac |
|---|---|---|---|---|---|---|
| ECOILA | 7901 | -69.13 | 66.33 | 32.61 | 65.98 | 0.000 |
| F1A | 7901 | -344.2 | 306.4 | 133.2 | 336.4 | 0.000 |
| F1B | 7901 | -346.9 | 293.8 | 141.9 | 339.8 | 0.000 |
| F2A | 7901 | -306.3 | 254.5 | 115 | 289.9 | 0.000 |
| F2B | 7901 | -313 | 296.7 | 114.7 | 302.8 | 0.000 |
| F3A | 7901 | -324.2 | 291.9 | 57.7 | 317.8 | 0.000 |
| F3B | 7901 | -327.1 | 288.3 | 104.6 | 323.2 | 0.000 |
| F4A | 7901 | -335.4 | 343.9 | 132.6 | 339.5 | 0.000 |
| F4B | 7901 | -343 | 342.4 | 132.1 | 338.2 | 0.000 |
| F5A | 7901 | -265.9 | 314.4 | 123.8 | 311 | 0.000 |
| F5B | 7901 | -301.5 | 315.6 | 155.9 | 313.8 | 0.000 |
| F6A | 7901 | -433.2 | 311.7 | 206 | 430.6 | 0.000 |
| F6B | 7901 | -427.8 | 433.3 | 203.4 | 428.6 | 0.000 |
| F7A | 7901 | -537.7 | 481 | 206.9 | 534.3 | 0.000 |
| F7B | 7901 | -576 | 521.7 | 219.7 | 569.9 | 0.000 |
| F8A | 7901 | -292.5 | 351.8 | 118.2 | 351 | 0.000 |
| F8B | 7901 | -343.8 | 352.7 | 151.6 | 348 | 0.000 |
| F9A | 7901 | -398.5 | 288 | 84.66 | 396.6 | 0.000 |
| F9B | 7901 | -403.5 | 414.8 | 97.08 | 404.1 | 0.000 |
| bcoil | 7901 | -135.2 | 134.4 | 119.5 | 129.6 | 0.000 |
| plasma_current | 7901 | -1704 | 2268 | 1060 | 2209 | 0.000 |

**MAST** (values as stored, inside the EFIT window)

| column | shots | min of min | max of max | median p99 abs | max p99 abs | max NaN frac |
|---|---|---|---|---|---|---|
| efps_current | 1209 | -20.84 | 18.16 | 8.921 | 20.7 | 0.167 |
| p2l_current | 1209 | -412.3 | 602.2 | 179.9 | 409.6 | 0.167 |
| p2u_current | 1209 | -593.5 | 365.6 | 184.8 | 418.1 | 0.167 |
| p3l_current | 1209 | -45.8 | 509.1 | 1.705 | 11.05 | 0.167 |
| p3u_current | 1209 | -58.14 | 37.98 | 1.639 | 13.22 | 0.167 |
| p4l_current | 1209 | -587 | -1.315 | 169.8 | 285.7 | 0.167 |
| p4u_current | 1209 | -288.1 | -1.162 | 170.2 | 286.8 | 0.167 |
| p5l_current | 1209 | -176.6 | 0.3432 | 113.6 | 173.3 | 0.167 |
| p5u_current | 1209 | -235.1 | 0.2882 | 113.5 | 172.6 | 0.167 |
| p6l_current | 1209 | -479.6 | 20.26 | 3.027 | 19.05 | 0.167 |
| p6u_current | 1209 | -98.84 | 20.36 | 3.068 | 19.55 | 0.167 |
| plasma_current | 1209 | 81.32 | 1366 | 751.4 | 969 | 0.167 |
| sol_current | 1209 | -66.49 | 43.3 | 41.43 | 48.25 | 0.167 |
| tf_current | 1209 | -178.4 | -52.38 | 84.85 | 87.12 | 0.834 |


### 0.5 Flux map indexing (pass)

0 shape failures; 0 shots with the scorer O-point more than one cell from the axis; 23 shots with a border extremum for review

- Shapes: (T, 65, 65): 7044
- DIII-D (7041 shots): scorer O-point cell distance, per-shot max: min 2.693e-05, median 0.4828, max 0.9298; interior local maximum found on min 1.00, median 1.00, max 1.00 of frames; global extremum cell distance, per-shot max: min 2.693e-05, median 0.4829, max 36.38; global extremum on the border: 23 shots; transposed reading median: min 2.693e-05, median 2.331, max 32.46; opposite extremum median: min 20.95, median 26.15, max 35.68
- MAST (3 shots): scorer O-point cell distance, per-shot max: min 0.4035, median 0.4462, max 0.4913; interior local maximum found on min 1.00, median 1.00, max 1.00 of frames; global extremum cell distance, per-shot max: min 0.4035, median 0.4462, max 0.4913; global extremum on the border: 0 shots; transposed reading median: min 4.349, median 4.769, max 4.771; opposite extremum median: min 20.35, median 20.5, max 20.53

Shots for review (23, showing up to 20):
- diii_d_train/1491b89cb3: global extremum on the grid border on 10% of sampled frames (O-point max distance 0.48 cells)
- diii_d_train/20aaf33cdd: global extremum on the grid border on 10% of sampled frames (O-point max distance 0.41 cells)
- diii_d_train/27b661cf0d: global extremum on the grid border on 10% of sampled frames (O-point max distance 0.50 cells)
- diii_d_train/36bc6cf20f: global extremum on the grid border on 50% of sampled frames (O-point max distance 0.51 cells)
- diii_d_train/5ffff526a8: global extremum on the grid border on 10% of sampled frames (O-point max distance 0.39 cells)
- diii_d_train/695c3c9001: global extremum on the grid border on 50% of sampled frames (O-point max distance 0.43 cells)
- diii_d_train/698f1cc29c: global extremum on the grid border on 20% of sampled frames (O-point max distance 0.49 cells)
- diii_d_train/7455726fc9: global extremum on the grid border on 10% of sampled frames (O-point max distance 0.45 cells)
- diii_d_train/806b526bd2: global extremum on the grid border on 10% of sampled frames (O-point max distance 0.49 cells)
- diii_d_train/9739dfba14: global extremum on the grid border on 10% of sampled frames (O-point max distance 0.51 cells)
- diii_d_train/973d6add2d: global extremum on the grid border on 20% of sampled frames (O-point max distance 0.44 cells)
- diii_d_train/9eb36ef610: global extremum on the grid border on 10% of sampled frames (O-point max distance 0.48 cells)
- diii_d_train/a2c4656611: global extremum on the grid border on 10% of sampled frames (O-point max distance 0.49 cells)
- diii_d_train/b1d10a768b: global extremum on the grid border on 10% of sampled frames (O-point max distance 0.48 cells)
- diii_d_train/b2ce477f64: global extremum on the grid border on 10% of sampled frames (O-point max distance 0.45 cells)
- diii_d_train/bda1f9217d: global extremum on the grid border on 10% of sampled frames (O-point max distance 0.40 cells)
- diii_d_train/c52ef2655b: global extremum on the grid border on 10% of sampled frames (O-point max distance 0.53 cells)
- diii_d_train/ced938a97c: global extremum on the grid border on 20% of sampled frames (O-point max distance 0.46 cells)
- diii_d_train/dbcb12171a: global extremum on the grid border on 10% of sampled frames (O-point max distance 0.49 cells)
- diii_d_train/e6df3967cb: global extremum on the grid border on 30% of sampled frames (O-point max distance 0.45 cells)

### 0.10 Grid (pass)

one grid per machine

- DIII-D: 1 grid configuration(s) over 7915 shots; R 0.84 to 2.54 m step 0.02656; Z -1.6 to 1.6 m step 0.05; strictly increasing and length 65: 100.0%
- MAST: 1 grid configuration(s) over 1209 shots; R 0.06 to 2 m step 0.03031; Z -2 to 2 m step 0.0625; strictly increasing and length 65: 100.0%

### 0.9 Coil geometry (pass)

one geometry per machine

- DIII-D: 1 geometry configuration(s) over 7915 shots; elements 19: 7915; input columns mapped to a current column: 100.0%; solenoid rows present: 100.0% (elements 1: 7915)
  - elements per column: {"magnetics_ECOILA": 1, "magnetics_F1A": 1, "magnetics_F1B": 1, "magnetics_F2A": 1, "magnetics_F2B": 1, "magnetics_F3A": 1, "magnetics_F3B": 1, "magnetics_F4A": 1, "magnetics_F4B": 1, "magnetics_F5A": 1, "magnetics_F5B": 1, "magnetics_F6A": 1, "magnetics_F6B": 1, "magnetics_F7A": 1, "magnetics_F7B": 1, "magnetics_F8A": 1, "magnetics_F8B": 1, "magnetics_F9A": 1, "magnetics_F9B": 1}: 7915
  - R 0.7225 to 2.612 m, Z -1.603 to 1.602 m; all angles zero: 0.0%
- MAST: 1 geometry configuration(s) over 1209 shots; elements 812: 1209; input columns mapped to a current column: 100.0%; solenoid rows present: 100.0% (elements 656: 1209)
  - elements per column: {"magnetics_p2l_current": 20, "magnetics_p2u_current": 20, "magnetics_p3l_current": 8, "magnetics_p3u_current": 8, "magnetics_p4l_current": 23, "magnetics_p4u_current": 23, "magnetics_p5l_current": 23, "magnetics_p5u_current": 23, "magnetics_p6l_current": 4, "magnetics_p6u_current": 4, "magnetics_sol_current": 656}: 1209
  - R 0.119 to 1.715 m, Z -1.844 to 1.849 m; all angles zero: 100.0%

### 0.6 Boundary contour (review)

0 shots flagged

- DIII-D (7041 shots): efit_lcfs_n min 37, median 83, max 93 (per-shot min) to min 57, median 92, max 110 (per-shot max); padded width 93: 1344, 92: 1321, 91: 1081, 94: 854, 90: 631, ... 22 more; padding zero: 7039, none: 2; NaN inside valid points: max 0; closed contours: min 0.833, median 1.000, max 1.000; R 1.018 to 2.35 m, Z -1.358 to 1.317 m
- MAST (3 shots): efit_lcfs_n min 125, median 125, max 129 (per-shot min) to min 155, median 155, max 155 (per-shot max); padded width 155: 3; padding nan: 3; NaN inside valid points: max 0; closed contours: min 0.972, median 1.000, max 1.000; R 0.1963 to 1.425 m, Z -1.202 to 1.297 m

### 0.7 Missing Thomson (review)

DIII-D core profile within 20 ms: median 1.00 of frames; MAST core profile within 20 ms: median 1.00 of frames

**DIII-D core** (7915 shots)

| metric | over shots |
|---|---|
| channel count | 44: 7117, 42: 478, 40: 249, 54: 70, 43: 1 |
| Thomson samples | min 32, median 2305, max 3287 |
| samples inside EFIT window | min 0, median 782, max 1759 |
| median sampling interval (ms) | min 0.3, median 5.872, max 20 |
| (time, channel) orientation | 100.0% of shots |
| mean missing fraction over channels | min 0.000, median 0.135, max 1.000 |
| max missing fraction over channels | min 0.002, median 1.000, max 1.000 |
| slices with no valid channel | min 0.000, median 0.000, max 1.000 |
| slices that are valid profiles | min 0.000, median 0.999, max 1.000 |
| EFIT frames with a valid profile within 10 ms | min 0.000, median 1.000, max 1.000 |
| EFIT frames with a valid profile within 20 ms | min 0.000, median 1.000, max 1.000 |
| EFIT frames with any valid channel within 10 ms | min 0.000, median 1.000, max 1.000 |
| EFIT frames with any valid channel within 20 ms | min 0.000, median 1.000, max 1.000 |

**DIII-D edge** (7915 shots)

| metric | over shots |
|---|---|
| channel count | 10: 7188, 14: 727 |
| Thomson samples | min 1, median 578, max 1456 |
| samples inside EFIT window | min 0, median 199, max 680 |
| median sampling interval (ms) | min 0.1943, median 25, max 4900 |
| (time, channel) orientation | 100.0% of shots |
| mean missing fraction over channels | min 0.000, median 0.000, max 1.000 |
| max missing fraction over channels | min 0.000, median 0.000, max 1.000 |
| slices with no valid channel | min 0.000, median 0.000, max 1.000 |
| slices that are valid profiles | min 0.000, median 1.000, max 1.000 |
| EFIT frames with a valid profile within 10 ms | min 0.000, median 0.795, max 1.000 |
| EFIT frames with a valid profile within 20 ms | min 0.000, median 1.000, max 1.000 |
| EFIT frames with any valid channel within 10 ms | min 0.000, median 0.795, max 1.000 |
| EFIT frames with any valid channel within 20 ms | min 0.000, median 1.000, max 1.000 |

**MAST core** (1209 shots)

| metric | over shots |
|---|---|
| channel count | 130: 1209 |
| Thomson samples | min 29, median 97, max 171 |
| samples inside EFIT window | min 5, median 72, max 126 |
| median sampling interval (ms) | min 0.007568, median 4.1, max 8.23 |
| (time, channel) orientation | 100.0% of shots |
| mean missing fraction over channels | min 0.005, median 0.064, max 0.945 |
| max missing fraction over channels | min 0.091, median 0.867, max 1.000 |
| slices with no valid channel | min 0.000, median 0.000, max 0.000 |
| slices that are valid profiles | min 0.000, median 1.000, max 1.000 |
| EFIT frames with a valid profile within 10 ms | min 0.000, median 1.000, max 1.000 |
| EFIT frames with a valid profile within 20 ms | min 0.000, median 1.000, max 1.000 |
| EFIT frames with any valid channel within 10 ms | min 0.364, median 1.000, max 1.000 |
| EFIT frames with any valid channel within 20 ms | min 0.636, median 1.000, max 1.000 |

**MAST edge** (1209 shots)

| metric | over shots |
|---|---|
| channel count | 16: 1209 |
| Thomson samples | min 12, median 98, max 157 |
| samples inside EFIT window | min 0, median 71, max 126 |
| median sampling interval (ms) | min 0.007538, median 4.1, max 8.23 |
| (time, channel) orientation | 100.0% of shots |
| mean missing fraction over channels | min 0.000, median 0.034, max 0.550 |
| max missing fraction over channels | min 0.000, median 0.172, max 0.800 |
| slices with no valid channel | min 0.000, median 0.000, max 0.000 |
| slices that are valid profiles | min 0.426, median 1.000, max 1.000 |
| EFIT frames with a valid profile within 10 ms | min 0.364, median 1.000, max 1.000 |
| EFIT frames with a valid profile within 20 ms | min 0.469, median 1.000, max 1.000 |
| EFIT frames with any valid channel within 10 ms | min 0.364, median 1.000, max 1.000 |
| EFIT frames with any valid channel within 20 ms | min 0.469, median 1.000, max 1.000 |


### 0.8 magnetics_dsep (review)

DIII-D: 0 sentinel values; MAST: 21 sentinel values

- DIII-D (7041 shots): values 0.000192 to 0.2391 m excluding the sentinel; per-shot median min 0.0009642, median 0.05929, max 0.1267; negative fraction min 0.000, median 0.000, max 0.000; -1 sentinel: 0 values on 0 shots; NaN: 0; dsep_times equals efit_times: 100.0%
- MAST (3 shots): values 0.0004488 to 0.005749 m excluding the sentinel; per-shot median min 0.001109, median 0.001121, max 0.001274; negative fraction min 0.000, median 0.000, max 0.000; -1 sentinel: 21 values on 3 shots; NaN: 1; dsep_times equals efit_times: 100.0%

### 0.11 Labels and time bases (pass)

0 shot(s) fail; 49 shots with fewer than 10 frames

Frames per shot
| split | shots | min | median | max | total frames | shots < 10 frames | shots with 0 frames |
|---|---|---|---|---|---|---|---|
| diii_d_public_test | 874 | 3 | 240 | 444 | 193359 | 4 | 0 |
| diii_d_train | 7041 | 1 | 241 | 445 | 1559340 | 42 | 0 |
| mast_public_test | 1206 | 5 | 58 | 98 | 69075 | 3 | 0 |
| mast_samples | 3 | 36 | 37 | 42 | 115 | 0 | 0 |

- DIII-D: efit_times step median min 20, median 20, max 2580 ms; largest gap per shot min 20, median 60, max 5160 ms; window 100 to 9020 ms
  - time bases: magnetics: dt 0.05: 5518, 0.5: 2397, length 480256: 5511, 49152: 1297, 50176: 1099, 327680: 5, ... 2 more; ip: dt 0.5: 7915, length 30719: 7915; ts_core: dt 5.992: 724, 4.999: 393, 6.251: 360, 6.814: 288, ... 442 more, length 2317: 804, 3047: 741, 3029: 276, 3020: 163, ... 814 more; ts_edge: dt 25.0: 2629, 24.504: 613, 10.0: 477, 18.879: 285, ... 250 more, length 580: 1591, 579: 1131, 421: 643, 578: 556, ... 129 more
  - q95 0.166 to 23.89; betaN -14.53 to 16.65; psi -0.5748 to 1.108; li 0 to 20.83; non-finite psi values 0; non-finite q95 0; non-finite betaN 0
- MAST: efit_times step median min 5, median 5, max 15 ms; largest gap per shot min 5, median 5, max 185 ms; window 5 to 635 ms
  - time bases: magnetics: dt 0.2: 1209, length 30000: 1095, 15482: 114; ts_core: dt 4.16: 313, 4.0: 208, 4.125: 105, 4.1: 81, ... 44 more, length 80: 54, 96: 51, 88: 49, 112: 37, ... 102 more; ts_edge: dt 4.16: 313, 4.0: 208, 4.1: 81, 4.125: 73, ... 43 more, length 99: 39, 95: 36, 87: 33, 112: 33, ... 101 more
  - q95 5.58 to 11.91; betaN 0.5014 to 1.337; psi -0.1961 to 0.06497; li 0.6114 to 1.672; non-finite psi values 0; non-finite q95 0; non-finite betaN 0

### 0.11b Test frame totals (pass)

totals match

- diii_d_public_test: 193359 frames over 874 shots (expected 193359)
- mast_public_test: 69075 frames over 1206 shots (expected 69075)

### 0.15 Scorer comparison (review)

DIII-D: Hausdorff median 0.05 m, li ratio median 1.011; MAST: Hausdorff median 0.052 m, li ratio median 0.964

**DIII-D**: 200 shots, 993 frames, 0 frames without a scorer contour; mask grid vs shot grid max difference 0 m

| quantity | over shots |
|---|---|
| Hausdorff distance, stored vs extracted contour (m), per-shot median | min 0.04809, median 0.05, max 0.1653 |
| Hausdorff distance (m), per-shot max | min 0.04964, median 0.05127, max 0.3679 |
| mean distance of stored points to extracted contour (m), per-shot median | min 0.002156, median 0.002657, max 0.1091 |
| |R_axis difference| (m), per-shot median | min 5.963e-06, median 0.0001499, max 0.001139 |
| |R_axis difference| (m), per-shot max | min 3.196e-05, median 0.0003559, max 0.003447 |
| |Z_axis difference| (m), per-shot median | min 7.03e-06, median 8.983e-05, max 0.0008359 |
| |Z_axis difference| (m), per-shot max | min 3.05e-05, median 0.0003281, max 0.002042 |
| li, scorer minus stored, per-shot median | min -0.05701, median 0.01056, max 0.2593 |
| li, scorer over stored, per-shot median | min 0.9823, median 1.011, max 1.308 |
| |li difference|, per-shot max | min 0.002709, median 0.03789, max 0.6141 |

**MAST**: 3 shots, 15 frames, 0 frames without a scorer contour; mask grid vs shot grid max difference 0 m

| quantity | over shots |
|---|---|
| Hausdorff distance, stored vs extracted contour (m), per-shot median | min 0.0488, median 0.052, max 0.05379 |
| Hausdorff distance (m), per-shot max | min 0.129, median 0.1302, max 0.1323 |
| mean distance of stored points to extracted contour (m), per-shot median | min 0.003404, median 0.003537, max 0.003539 |
| |R_axis difference| (m), per-shot median | min 6.852e-05, median 7.041e-05, max 8.431e-05 |
| |R_axis difference| (m), per-shot max | min 9.739e-05, median 0.0001036, max 0.0001083 |
| |Z_axis difference| (m), per-shot median | min 1.765e-05, median 2.826e-05, max 2.867e-05 |
| |Z_axis difference| (m), per-shot max | min 3.865e-05, median 4.927e-05, max 5.002e-05 |
| li, scorer minus stored, per-shot median | min -0.05779, median -0.03764, max -0.03353 |
| li, scorer over stored, per-shot median | min 0.953, median 0.9637, max 0.9678 |
| |li difference|, per-shot max | min 0.4043, median 0.4408, max 0.4523 |
| |kappa minus efit_elongation|, per-shot median | min 0.04825, median 0.06084, max 0.06428 |


### 0.13 README against data (review)

9 of 9 contradictions assessed

- D1 (DIII-D current units): median p99 |Ip| 1060, largest F-coil p99 569.9, ECOILA 32.61, bcoil 119.5; MAST Ip 751.4. Magnitudes in the hundreds to about a thousand read as kA and kA-turn, consistent with the dictionary and against the README's 'A' table. Final decision stays with the Green's-function fit in Phase C.
- D2 (DIII-D grid 'normalized 0-1'): efit_grid_R 0.84 to 2.54 m, efit_grid_Z -1.6 to 1.6 m on 7915 shots. Physical metres; dictionary agrees.
- D3 (Thomson orientation), DIII-D: core Te rows equal the time axis on 100.0% of 7915 shots, so storage is (time, channel). Dictionary agrees.
- D3 (Thomson orientation), MAST: core Te rows equal the time axis on 100.0% of 1209 shots, so storage is (time, channel). Dictionary agrees.
- D4 (MAST dtypes): magnetics_time float: 1209; efit_times float: 1209; Ip double: 1209. README says float64 for the time axes. Data is float32; dictionary agrees.
- D5 (DIII-D Thomson sample counts): core min 32, median 2305, max 3287, edge min 1, median 578, max 1456 over 7915 shots. README says about 1300 to 1900 core and 200 to 500 edge.
- D6 (MAST dsep sentinel): 21 values of -1 and 1 NaN over 3 labelled MAST shots. Only the three sample shots carry this label, so no new evidence beyond the dictionary.
- D7 (extra MAST EFIT columns): efit_beta_p, efit_elongation, efit_geom_axis_r, efit_geom_axis_z, efit_psi_axis, efit_psi_boundary in the MAST samples and absent from DIII-D train. Dictionary agrees.
- D7 continued, test inputs: MAST-only inputs ['magnetics_efps_current', 'magnetics_p2l_current', 'magnetics_p2u_current', 'magnetics_p3l_current', 'magnetics_p3u_current', 'magnetics_p4l_current', 'magnetics_p4u_current', 'magnetics_p5l_current', 'magnetics_p5u_current', 'magnetics_p6l_current', 'magnetics_p6u_current', 'magnetics_sol_current', 'magnetics_tf_current']; DIII-D-only inputs ['magnetics_ECOILA', 'magnetics_F1A', 'magnetics_F1B', 'magnetics_F2A', 'magnetics_F2B', 'magnetics_F3A', 'magnetics_F3B', 'magnetics_F4A', 'magnetics_F4B', 'magnetics_F5A', 'magnetics_F5B', 'magnetics_F6A', 'magnetics_F6B', 'magnetics_F7A', 'magnetics_F7B', 'magnetics_F8A', 'magnetics_F8B', 'magnetics_F9A', 'magnetics_F9B', 'magnetics_bcoil', 'magnetics_plasma_current_times'].
- D8 (no Parquet nulls): 0 null values over 9124 shots. README and dictionary agree.
- D9 (DIII-D Thomson channel counts): core 44: 7117, 42: 478, 40: 249, 54: 70, 43: 1; edge 10: 7188, 14: 727 over 7915 shots. README says 40, 42, 43, 44 or 54 core and 10 or 14 edge.

Proposed data-dictionary updates (not applied in this phase):
- D5: replace the single-layout statement (3029 core, 1443 edge) with the observed ranges core min 32, median 2305, max 3287, edge min 1, median 578, max 1456.
- D9: record the observed channel-count distribution and confirm the per-shot layout warning.
