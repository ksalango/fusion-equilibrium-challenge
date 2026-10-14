# Target profile, version 1

Target build `../data/processed/targets/v1` (configuration hash `b2192c3a01a9`, lists working_train, working_validation, stride 1). No MAST equilibrium label is read.

| Id | Check | Status | Summary |
| --- | --- | --- | --- |
| Y1 | Rows and keys | pass | working_train 65262 rows, keys match; working_validation 22052 rows, keys match |
| Y2 | Bookkeeping | pass | 1000 frames of 50 training shots: coil + Ip G s + r against psi_phys 4.73e-15, compose_map against the stored map 2.64e-16 of the frame's flux range (rule 1e-10) |
| Y3 | Share total | pass | working_train median 0.9977 (5th to 95th 0.9884 to 1.0065), 0 frames without a boundary; working_validation median 0.9974 (5th to 95th 0.9891 to 1.0063), 0 frames without a boundary (rule: median within 0.99 to 1.005) |
| Y4 | Parity | pass | 30 even and 20 odd components; largest departure 0.0e+00, mean 0.0e+00 (rule 1e-8) |
| Y5 | Mirror | pass | 1000 frames: coefficients of the flipped share against c times p 1.47e-15 of the largest coefficient (rule 1e-8); mirroring twice is the identity |
| Y6 | Reconstruction | review | relative error of the share through the basis: working_train median 0.0332 (mirrors 0.0332); working_validation median 0.0376 (mirrors 0.0376) |
| Y7 | Flux basis | pass | 1000 reconstructed training shares: phi_mean + Phi c against G P s, DIII-D 1.99e-15, MAST 1.60e-15 of the largest value (rule 1e-10) |
| Y8 | Conservation | pass | placement on MAST: placed total plus the weight outside the grid against the total, mean share and 1000 reconstructed shares, worst 6.36e-16 (rule 1e-10); weight outside the MAST grid, median 4.42e-18, 95th 5.27e-18; mean share on MAST border cells 0.00e+00 |
| Y9 | Response | pass | DIII-D: beyond 5 cells 1.05e-02 (rule 0.02), beyond 10 cells 3.90e-03 (rule 0.005), symmetry outside the first 3 interior rings 1.70e-04 (rule 0.001; all sources 4.19e-03), Z flip 2.4e-15; share of the fit sample's current in the first 3 interior rings, largest 0.0e+00 (rule 0.0001, 65162 frames); MAST: beyond 5 cells 3.81e-02 (rule 0.1), beyond 10 cells 2.02e-02 (rule 0.03), symmetry outside the first 3 interior rings 3.09e-04 (rule 0.001; all sources 1.64e-02), Z flip 3.2e-15; share of the fit sample's current in the first 3 interior rings, largest 8.4e-07 (rule 0.0001, 65162 frames) |
| Y10 | Where the current lands | review | MAST constants: centroid R 0.863 m, Z -0.000 m; envelope R 0.257 to 1.500 m, height 2.250 m; nearest coil 0.109 m, grid edge 0.212 m; coil elements inside 0 |
| Y11 | Sensitivity to the constants | review | R0 +5%: centroid R 0.906 m, envelope R 0.287 to 1.530 m, nearest coil 0.104 m; R0 -5%: centroid R 0.821 m, envelope R 0.196 to 1.439 m, nearest coil 0.048 m; a0 +5%: centroid R 0.864 m, envelope R 0.227 to 1.530 m, nearest coil 0.076 m; a0 -5%: centroid R 0.863 m, envelope R 0.287 to 1.439 m, nearest coil 0.139 m; Morris 1998 (0.70, 0.50): centroid R 0.710 m, envelope R 0.227 to 1.197 m, nearest coil 0.079 m |
| Y12 | Remainder size | review | median RMS over the flux range: working_train 0.0097 (rising 0.0206, flat_top 0.0085, late 0.0130); working_validation 0.0104 (rising 0.0187, flat_top 0.0092, late 0.0148); F79 0.0105 (rising 0.0193, flat-top 0.0093, late 0.0143) |
| Y13 | Remainder components | review | variance held: 5 0.9978 (F79 0.9978), 10 0.9998 (F79 0.9998), 20 1.0000 (F79 1.0) |
| Y14 | Remainder ridge | review | 10ms: training 0.8900, validation 0.8888; 50ms: training 0.9095, validation 0.9077 (F79: 0.8885, 0.8842); default window 50ms; the 50 ms ridge keeps the toroidal field's 10 ms rate |
| Y15 | Ceiling variants h to k | review | S: f_remainder_pca_5 0.9978, f_remainder_pca_10 0.9982, f_remainder_pca_20 0.9983, g_remainder_ridge_10 0.9973, h_symmetric_50 0.9981, h_symmetric_60 0.9982, h_symmetric_80 0.9986, i_placed_mast_50 0.9979, j_rescaled_50 0.9971, k_ridge_10ms_50 0.9971, k_ridge_50ms_50 0.9972 (F79: f 0.9982, g 0.9973) |
| Y16 | Mean current | review | validation with remainder S 0.7830, R2 psi 0.9918; validation without remainder S 0.7812, R2 psi 0.9908; DIII-D test (2000 frames): boundary on 1.000, median R_axis 1.736 m, Z_axis +0.003 m, kappa 1.777; MAST test (2000 frames): boundary on 1.000, median R_axis 0.967 m, Z_axis +0.000 m, kappa 1.872 |

## Y10 and Y11. Where the current lands on MAST

Envelope: the cells inside the 99 percent support (largest cells first) of at least 1 percent of the 20000 reconstructed training shares, placed on MAST. Centroid of the placed mean share.

- nominal: centroid R 0.863 m, Z -0.000 m; envelope R 0.257 to 1.500 m, height 2.250 m; nearest coil 0.109 m, grid edge 0.212 m; coil elements inside 0
- R0 +5%: centroid R 0.906 m, Z -0.000 m; envelope R 0.287 to 1.530 m, height 2.250 m; nearest coil 0.104 m, grid edge 0.243 m; coil elements inside 0
- R0 -5%: centroid R 0.821 m, Z -0.000 m; envelope R 0.196 to 1.439 m, height 2.250 m; nearest coil 0.048 m, grid edge 0.152 m; coil elements inside 0
- a0 +5%: centroid R 0.864 m, Z +0.000 m; envelope R 0.227 to 1.530 m, height 2.375 m; nearest coil 0.076 m, grid edge 0.182 m; coil elements inside 0
- a0 -5%: centroid R 0.863 m, Z +0.000 m; envelope R 0.287 to 1.439 m, height 2.125 m; nearest coil 0.139 m, grid edge 0.243 m; coil elements inside 0
- Morris 1998 (0.70, 0.50): centroid R 0.710 m, Z +0.000 m; envelope R 0.227 to 1.197 m, height 1.750 m; nearest coil 0.079 m, grid edge 0.182 m; coil elements inside 0

## Y9. Response against the filament Green's function (F89)

Symmetry is ruled between source cells outside the first 3 interior rings; the value over all source cells is shown beside it. The asymmetry of cells in those rings is a property of the border treatment (F89); the ring guard is the share of the absolute current of the fit sample in those rings (the share as it is on DIII-D, placed on MAST), largest over the frames.

| Machine | Beyond 5 cells | Rule | Beyond 10 cells | Rule | Symmetry | Rule | Symmetry, all sources | Z flip | Rule | Ring share, largest | 99th percentile | Pooled | Rule |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DIII-D | 1.05e-02 | 0.02 | 3.90e-03 | 0.005 | 1.70e-04 (81 sources) | 0.001 | 4.19e-03 (121 sources) | 2.4e-15 | 1e-12 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0001 |
| MAST | 3.81e-02 | 0.1 | 2.02e-02 | 0.03 | 3.09e-04 (81 sources) | 0.001 | 1.64e-02 (121 sources) | 3.2e-15 | 1e-12 | 8.4e-07 | 0.0e+00 | 1.1e-09 | 0.0001 |

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
| validation_with_remainder | 0.7830 | 0.9918 | 0.8321 | 0.9125 | 0.1073 | -0.7748 | -0.5150 | -0.5571 | 0.4893 | 0.2621 | -0.2043 | -0.2871 |
| validation_without_remainder | 0.7812 | 0.9908 | 0.8321 | 0.9109 | 0.1018 | -0.8075 | -0.5221 | -0.5307 | 0.4852 | 0.2274 | -0.4274 | -0.1876 |

Runtime 181 s.
