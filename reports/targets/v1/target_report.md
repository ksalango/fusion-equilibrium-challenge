# Target profile, version 1

Target build `../data/processed/targets/v1` (configuration hash `b2192c3a01a9`, lists train, validation, local_test, stride 1). No MAST equilibrium label is read.

| Id | Check | Status | Summary |
| --- | --- | --- | --- |
| Y1 | Rows and keys | pass | train 1289474 rows, keys match; validation 191411 rows, keys match; local_test 78455 rows, keys match |
| Y2 | Bookkeeping | pass | 1000 frames of 50 training shots: coil + Ip G s + r against psi_phys 4.64e-15, compose_map against the stored map 3.10e-16 of the frame's flux range (rule 1e-10) |
| Y3 | Share total | pass | train median 0.9976 (5th to 95th 0.9886 to 1.0064), 0 frames without a boundary; validation median 0.9976 (5th to 95th 0.9887 to 1.0067), 0 frames without a boundary; local_test median 0.9977 (5th to 95th 0.9891 to 1.0062), 0 frames without a boundary (rule: median within 0.99 to 1.005) |
| Y4 | Parity | pass | 30 even and 20 odd components; largest departure 0.0e+00, mean 0.0e+00 (rule 1e-8) |
| Y5 | Mirror | pass | 1000 frames: coefficients of the flipped share against c times p 3.01e-15 of the largest coefficient (rule 1e-8); mirroring twice is the identity |
| Y6 | Reconstruction | review | relative error of the share through the basis: train median 0.0333 (mirrors 0.0333); validation median 0.0323 (mirrors 0.0323); local_test median 0.0336 (mirrors 0.0336) |
| Y7 | Flux basis | pass | 1000 reconstructed training shares: phi_mean + Phi c against G P s, DIII-D 2.47e-15, MAST 1.92e-15 of the largest value (rule 1e-10) |
| Y8 | Conservation | pass | placement on MAST: placed total plus the weight outside the grid against the total, mean share and 1000 reconstructed shares, worst 7.53e-16 (rule 1e-10); weight outside the MAST grid, median 4.34e-18, 95th 5.27e-18; mean share on MAST border cells 0.00e+00 |
| Y9 | Response | pass | DIII-D: beyond 5 cells 1.05e-02 (rule 0.02), beyond 10 cells 3.90e-03 (rule 0.005), symmetry outside the first 3 interior rings 1.70e-04 (rule 0.001; all sources 4.19e-03), Z flip 2.4e-15; share of the fit sample's current in the first 3 interior rings, largest 0.0e+00 (rule 0.0001, 100000 frames); MAST: beyond 5 cells 3.81e-02 (rule 0.1), beyond 10 cells 2.02e-02 (rule 0.03), symmetry outside the first 3 interior rings 3.09e-04 (rule 0.001; all sources 1.64e-02), Z flip 3.2e-15; share of the fit sample's current in the first 3 interior rings, largest 2.2e-06 (rule 0.0001, 100000 frames) |
| Y10 | Where the current lands | review | MAST constants: centroid R 0.865 m, Z -0.000 m; envelope R 0.257 to 1.470 m, height 2.250 m; nearest coil 0.109 m, grid edge 0.212 m; coil elements inside 0 |
| Y11 | Sensitivity to the constants | review | R0 +5%: centroid R 0.907 m, envelope R 0.287 to 1.530 m, nearest coil 0.101 m; R0 -5%: centroid R 0.822 m, envelope R 0.196 to 1.439 m, nearest coil 0.048 m; a0 +5%: centroid R 0.865 m, envelope R 0.227 to 1.500 m, nearest coil 0.076 m; a0 -5%: centroid R 0.864 m, envelope R 0.287 to 1.439 m, nearest coil 0.139 m; Morris 1998 (0.70, 0.50): centroid R 0.711 m, envelope R 0.227 to 1.197 m, nearest coil 0.079 m |
| Y12 | Remainder size | review | median RMS over the flux range: train 0.0095 (rising 0.0194, flat_top 0.0084, late 0.0116); validation 0.0095 (rising 0.0199, flat_top 0.0084, late 0.0109); local_test 0.0093 (rising 0.0191, flat_top 0.0081, late 0.0121); F79 0.0105 (rising 0.0193, flat-top 0.0093, late 0.0143) |
| Y13 | Remainder components | review | variance held: 5 0.9976 (F79 0.9978), 10 0.9998 (F79 0.9998), 20 1.0000 (F79 1.0) |
| Y14 | Remainder ridge | review | 10ms: training 0.8831, validation 0.8862; 50ms: training 0.9041, validation 0.9057 (F79: 0.8885, 0.8842); default window 50ms; the 50 ms ridge keeps the toroidal field's 10 ms rate |
| Y15 | Ceiling variants h to k | review | S: f_remainder_pca_5 0.9978, f_remainder_pca_10 0.9982, f_remainder_pca_20 0.9983, g_remainder_ridge_10 0.9973, h_symmetric_50 0.9981, h_symmetric_60 0.9982, h_symmetric_80 0.9986, i_placed_mast_50 0.9979, j_rescaled_50 0.9971, k_ridge_10ms_50 0.9971, k_ridge_50ms_50 0.9972 (F79: f 0.9982, g 0.9973) |
| Y16 | Mean current | review | validation with remainder S 0.7924, R2 psi 0.9913; validation without remainder S 0.7911, R2 psi 0.9903; DIII-D test (2000 frames): boundary on 1.000, median R_axis 1.737 m, Z_axis +0.003 m, kappa 1.778, boundary inner R 1.091 m, outer R 2.266 m, width 1.184 m, height 2.107 m; MAST test (2000 frames): boundary on 1.000, median R_axis 0.968 m, Z_axis +0.000 m, kappa 1.873, boundary inner R 0.338 m, outer R 1.465 m, width 1.125 m, height 2.102 m |

## Y10 and Y11. Where the current lands on MAST

Envelope: the cells inside the 99 percent support (largest cells first) of at least 1 percent of the 20000 reconstructed training shares, placed on MAST. Centroid of the placed mean share.

- nominal: centroid R 0.865 m, Z -0.000 m; envelope R 0.257 to 1.470 m, height 2.250 m; nearest coil 0.109 m, grid edge 0.212 m; coil elements inside 0
- R0 +5%: centroid R 0.907 m, Z -0.000 m; envelope R 0.287 to 1.530 m, height 2.250 m; nearest coil 0.101 m, grid edge 0.243 m; coil elements inside 0
- R0 -5%: centroid R 0.822 m, Z -0.000 m; envelope R 0.196 to 1.439 m, height 2.250 m; nearest coil 0.048 m, grid edge 0.152 m; coil elements inside 0
- a0 +5%: centroid R 0.865 m, Z +0.000 m; envelope R 0.227 to 1.500 m, height 2.375 m; nearest coil 0.076 m, grid edge 0.182 m; coil elements inside 0
- a0 -5%: centroid R 0.864 m, Z +0.000 m; envelope R 0.287 to 1.439 m, height 2.188 m; nearest coil 0.139 m, grid edge 0.243 m; coil elements inside 0
- Morris 1998 (0.70, 0.50): centroid R 0.711 m, Z +0.000 m; envelope R 0.227 to 1.197 m, height 1.750 m; nearest coil 0.079 m, grid edge 0.182 m; coil elements inside 0

## Y9. Response against the filament Green's function (F89)

Symmetry is ruled between source cells outside the first 3 interior rings; the value over all source cells is shown beside it. The asymmetry of cells in those rings is a property of the border treatment (F89); the ring guard is the share of the absolute current of the fit sample in those rings (the share as it is on DIII-D, placed on MAST), largest over the frames.

| Machine | Beyond 5 cells | Rule | Beyond 10 cells | Rule | Symmetry | Rule | Symmetry, all sources | Z flip | Rule | Ring share, largest | 99th percentile | Pooled | Rule |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DIII-D | 1.05e-02 | 0.02 | 3.90e-03 | 0.005 | 1.70e-04 (81 sources) | 0.001 | 4.19e-03 (121 sources) | 2.4e-15 | 1e-12 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0001 |
| MAST | 3.81e-02 | 0.1 | 2.02e-02 | 0.03 | 3.09e-04 (81 sources) | 0.001 | 1.64e-02 (121 sources) | 3.2e-15 | 1e-12 | 2.2e-06 | 0.0e+00 | 1.9e-09 | 0.0001 |

## Y15. Ceiling variants

| Variant | S | r2_psi | r2_qb | one_minus_dlcfs | consistency | r2_R_axis | r2_Z_axis | r2_kappa | r2_tri_top | r2_tri_bot | r2_volume | r2_li |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f_remainder_pca_5 | 0.9978 | 1.0000 | 1.0000 | 0.9924 | 0.9927 | 0.9999 | 0.9995 | 0.9853 | 0.9875 | 0.9826 | 0.9959 | 0.9984 |
| f_remainder_pca_10 | 0.9982 | 1.0000 | 1.0000 | 0.9934 | 0.9944 | 0.9999 | 0.9996 | 0.9886 | 0.9900 | 0.9862 | 0.9980 | 0.9988 |
| f_remainder_pca_20 | 0.9983 | 1.0000 | 1.0000 | 0.9936 | 0.9947 | 0.9999 | 0.9996 | 0.9890 | 0.9908 | 0.9866 | 0.9981 | 0.9989 |
| g_remainder_ridge_10 | 0.9973 | 0.9998 | 1.0000 | 0.9916 | 0.9911 | 0.9994 | 0.9992 | 0.9853 | 0.9842 | 0.9800 | 0.9910 | 0.9983 |
| h_symmetric_50 | 0.9981 | 1.0000 | 1.0000 | 0.9930 | 0.9939 | 0.9999 | 0.9996 | 0.9879 | 0.9899 | 0.9837 | 0.9974 | 0.9987 |
| h_symmetric_60 | 0.9982 | 1.0000 | 1.0000 | 0.9934 | 0.9943 | 0.9999 | 0.9997 | 0.9887 | 0.9898 | 0.9854 | 0.9978 | 0.9989 |
| h_symmetric_80 | 0.9986 | 1.0000 | 1.0000 | 0.9945 | 0.9957 | 1.0000 | 0.9999 | 0.9906 | 0.9921 | 0.9896 | 0.9985 | 0.9991 |
| i_placed_mast_50 | 0.9979 | 1.0000 | 1.0000 | 0.9926 | 0.9935 | 0.9999 | 0.9997 | 0.9864 | 0.9898 | 0.9831 | 0.9975 | 0.9978 |
| j_rescaled_50 | 0.9971 | 0.9999 | 1.0000 | 0.9919 | 0.9899 | 0.9990 | 0.9996 | 0.9854 | 0.9896 | 0.9831 | 0.9748 | 0.9976 |
| k_ridge_10ms_50 | 0.9971 | 0.9998 | 1.0000 | 0.9911 | 0.9904 | 0.9994 | 0.9993 | 0.9852 | 0.9843 | 0.9765 | 0.9900 | 0.9981 |
| k_ridge_50ms_50 | 0.9972 | 0.9999 | 1.0000 | 0.9912 | 0.9907 | 0.9994 | 0.9993 | 0.9858 | 0.9847 | 0.9769 | 0.9906 | 0.9982 |

## Y16. Mean current on DIII-D validation

| Arm | S | r2_psi | r2_qb | one_minus_dlcfs | consistency | r2_R_axis | r2_Z_axis | r2_kappa | r2_tri_top | r2_tri_bot | r2_volume | r2_li |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| validation_with_remainder | 0.7924 | 0.9913 | 0.8183 | 0.9161 | 0.1642 | -0.7432 | -0.4383 | -0.3845 | 0.6094 | 0.5400 | -0.0008 | -0.1965 |
| validation_without_remainder | 0.7911 | 0.9903 | 0.8183 | 0.9151 | 0.1608 | -0.7622 | -0.4461 | -0.3065 | 0.5990 | 0.5269 | -0.1708 | -0.1324 |

## Y16. Mean current on the test frames

Inputs only. Over the frames on which the scorer finds a closed boundary in the composed map: the derived R_axis, Z_axis and kappa and the boundary's inner and outer radius, width and height, as median (5th to 95th percentile).

| Machine | Frames | Boundary found | R_axis (m) | Z_axis (m) | kappa | Inner R (m) | Outer R (m) | Width (m) | Height (m) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DIII-D | 2000 | 1.000 | 1.737 (1.704 to 1.757) | 0.003 (-0.003 to 0.008) | 1.778 (1.650 to 1.921) | 1.091 (0.994 to 1.160) | 2.266 (2.185 to 2.346) | 1.184 (1.067 to 1.271) | 2.107 (1.863 to 2.256) |
| MAST | 2000 | 1.000 | 0.968 (0.944 to 0.989) | 0.000 (-0.001 to 0.003) | 1.873 (1.607 to 1.997) | 0.338 (0.153 to 0.480) | 1.465 (1.384 to 1.595) | 1.125 (0.936 to 1.389) | 2.102 (1.815 to 2.425) |

Runtime 328 s.
