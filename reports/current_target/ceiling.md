# Current-distribution target ceiling, split `split_v1.json`

300 training shots (16423 frames for the PCA), 100 validation shots (5551 frames) scored at frame stride 4 with the true q95 and betaN; coil model solenoid_flat, fit of 2026-10-10T18:45:26+00:00; 8 workers; references from the cache.

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

Runtime: 155 s (frames_s 0, bases_s 23, remainder_ridge_s 18, plasma_basis_s 0, validation_pass_s 8, references_s 0, score_a_masked_s 9, score_b_unmasked_s 9, score_c_pca_20_s 9, score_c_pca_50_s 9, score_c_pca_100_s 9, score_d_plasma_basis_50_s 9, score_e_remainder_exact_s 10, score_f_remainder_pca_5_s 10, score_f_remainder_pca_10_s 10, score_f_remainder_pca_20_s 10, score_g_remainder_ridge_10_s 10)
