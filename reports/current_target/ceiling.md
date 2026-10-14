# Current-distribution target ceiling, split `split_v1.json`

300 training shots (cached frames for the PCA), 100 validation shots (5551 frames) scored at frame stride 4 with the true q95 and betaN; coil model solenoid_flat, fit of 2026-10-10T18:45:26+00:00; 8 workers; references from the cache.

| Variant | S | R2 psi | R2 q95,betaN | 1 - D_LCFS | Consistency | R_axis | Z_axis | kappa | tri_top | tri_bot | volume | li |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a_masked | 0.9784 | 0.9984 | 1.0000 | 0.9844 | 0.9041 | 0.9343 | 0.9994 | 0.9550 | 0.9870 | 0.9830 | 0.4846 | 0.9853 |
| b_unmasked | 0.9913 | 0.9945 | 1.0000 | 0.9905 | 0.9763 | 0.9974 | 0.9995 | 0.9015 | 0.9900 | 0.9826 | 0.9667 | 0.9961 |
| c_pca_20 | 0.9755 | 0.9984 | 1.0000 | 0.9790 | 0.8924 | 0.9333 | 0.9964 | 0.9390 | 0.9647 | 0.9394 | 0.4915 | 0.9824 |
| c_pca_50 | 0.9780 | 0.9984 | 1.0000 | 0.9826 | 0.9029 | 0.9357 | 0.9991 | 0.9527 | 0.9814 | 0.9736 | 0.4931 | 0.9849 |
| c_pca_100 | 0.9780 | 0.9984 | 1.0000 | 0.9832 | 0.9026 | 0.9341 | 0.9993 | 0.9530 | 0.9824 | 0.9768 | 0.4880 | 0.9849 |
| d_plasma_basis_50 | 0.9990 | 1.0000 | 1.0000 | 0.9953 | 0.9971 | 0.9999 | 0.9997 | 0.9911 | 0.9959 | 0.9953 | 0.9998 | 0.9983 |
| e_remainder_exact | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
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

## Mask statistics (validation frames)

Boundary found on 5551 of 5551 frames; frames without Ip 0.

- `dropped_abs_share`: median 0.0630, mean 0.0685, 5th to 95th percentile 0.0383 to 0.1144 (5551 frames)
- `masked_total_over_ip`: median 0.9975, mean 0.9979, 5th to 95th percentile 0.9890 to 1.0070 (5551 frames)

## Vacuum remainder (plasma target minus the flux of the masked current)

RMS of the remainder over the frame's flux range (max minus min of the true map), validation frames:

- all: median 0.0105, mean 0.0131, 5th to 95th percentile 0.0040 to 0.0305 (5551 frames)
- rising: median 0.0193, mean 0.0233, 5th to 95th percentile 0.0068 to 0.0518 (1103 frames)
- flat_top: median 0.0093, mean 0.0103, 5th to 95th percentile 0.0038 to 0.0206 (4308 frames)
- late: median 0.0143, mean 0.0162, 5th to 95th percentile 0.0054 to 0.0322 (140 frames)

Variance of the training remainder held by the leading components: 5: 0.9978, 10: 0.9998, 20: 1.0000.

Bookkeeping check (variant e against the true map): largest absolute error over the frame's flux range 1.33e-16.

Ridge (variant g): 10 coefficients from 42 inputs (currents and rates, Ip-signed), alpha 1.0, 16423 of 16423 training frames complete; pooled coefficient R2 training 0.8885, validation 0.8842.

## Target-phase variants h to k (`docs/TARGET.md`, Part 5)

Symmetric basis of the share: 80 components over 16397 training frames (passing the lighter label filter, with a boundary) and their mirrors; even and odd components among the first 50: 30 and 20, the first 60: 35 and 25, the first 80: 44 and 36; variance held 50: 0.985178, 60: 0.987160, 80: 0.989834.

Placement (variant i): MAST R0 0.85 m, a0 0.65 m; share of the reconstructed total kept through placement and back:
- retained: median 1.0000, mean 1.0000, 5th to 95th percentile 1.0000 to 1.0000 (5551 frames)
- reconstructed share total (variant j rescales it to one): median 0.9976, mean 0.9980, 5th to 95th percentile 0.9890 to 1.0071 (5551 frames)

Remainder ridges (variant k), pooled coefficient R2:

- 10ms: training 0.8885, validation 0.8842 (42 inputs, 16423 of 16423 training frames complete)
- 50ms: training 0.9084, validation 0.9055 (42 inputs, 16423 of 16423 training frames complete)

Against the references of the specification (S, consistency, volume):

| Variant | Reference | S - reference | Consistency - reference | Volume - reference |
| --- | --- | --- | --- | --- |
| h_symmetric_50 | f_remainder_pca_10 | -0.0002 | -0.0006 | -0.0006 |
| h_symmetric_60 | f_remainder_pca_10 | -0.0000 | -0.0001 | -0.0002 |
| h_symmetric_80 | f_remainder_pca_10 | +0.0004 | +0.0012 | +0.0005 |
| i_placed_mast_50 | h_symmetric_50 | -0.0001 | -0.0004 | +0.0002 |
| j_rescaled_50 | h_symmetric_50 | -0.0010 | -0.0040 | -0.0226 |
| k_ridge_10ms_50 | g_remainder_ridge_10 | -0.0002 | -0.0006 | -0.0009 |
| k_ridge_50ms_50 | g_remainder_ridge_10 | -0.0001 | -0.0004 | -0.0004 |

Reading rule for h: the base count is kept if S is within 0.0003 of f and consistency within 0.001.

Runtime: 230 s (frames_s 0, bases_s 0, remainder_ridge_s 20, plasma_basis_s 0, target_objects_s 0, validation_pass_s 10, references_s 0, score_a_masked_s 10, score_b_unmasked_s 10, score_c_pca_20_s 10, score_c_pca_50_s 11, score_c_pca_100_s 11, score_d_plasma_basis_50_s 11, score_e_remainder_exact_s 12, score_f_remainder_pca_5_s 12, score_f_remainder_pca_10_s 11, score_f_remainder_pca_20_s 11, score_g_remainder_ridge_10_s 12, score_h_symmetric_50_s 12, score_h_symmetric_60_s 12, score_h_symmetric_80_s 12, score_i_placed_mast_50_s 11, score_j_rescaled_50_s 11, score_k_ridge_10ms_50_s 10, score_k_ridge_50ms_50_s 10)
