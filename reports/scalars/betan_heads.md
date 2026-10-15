# betaN heads, working lists of `split_v1.json`, stride 4

300 training shots (16397 of 16423 frames kept by the lighter label filter), 100 validation shots (5551 frames with both labels finite). Features: version 2 (`../data/processed/features/v2`). Ridge alpha 1; the GBM correction uses the F75 settings. Stage 1 takes `a` and `kappa` from the true maps (label-side bound), stage 2 from the `hybrid_ridge` maps. Diagnostic only.

Base: `betaN_base = 100 (2 mu0 p_e / Bt0^2) a Bt0 / abs(Ip)`, with `p_e = ts_core_pe_mean__interp` (Pa), `Bt0 = 2e-7 N_tf abs(I_tf) / R0` (N_tf 144, R0 1.67 m, machines.py), `a` the stage's minor radius (m), Ip in MA; elongation enters the corrections, not the base. Predictions of the formula arms: `c * betaN_base * correction`, `c` fitted per stage by least squares on the training frames (stage 1: c = 3.7139), (stage 2: c = 3.7093).

Transfer proxy (corner): q_cyl < 1.780 (training 10th percentile) and ts_beta_p < 0.1088 (training 50th percentile); 302 validation frames.

| Status | Arm | Inputs | Stage 1 R2 all | Stage 2 R2 all | Stage 2 R2 negative Ip | Stage 2 R2 corner | Stage 2 R2 all - ridge_raw |
| --- | --- | --- | --- | --- | --- | --- | --- |
| reference | ridge_raw | 25 | 0.9408 | 0.9408 | 0.8926 | 0.8740 | +0.0000 |
| review | ridge_features | 10 | 0.8549 | 0.8524 | 0.7024 | 0.6801 | -0.0884 |
| review | formula_ridge | 10 | 0.7709 | 0.8017 | 0.8510 | 0.8558 | -0.1391 |
| review | formula_gbm | 10 | 0.9004 | 0.9108 | 0.9133 | 0.8729 | -0.0300 |

ridge_raw is expected near F75's 0.9409 (ridge on inputs, refit on the training frames); the lighter filter here removes more training frames than F75's finite-label rule.

## Stage 1

Base constant c = 3.7139. Frames: all 5551, negative Ip 423, corner 302.

| Arm | Inputs | R2 all | R2 negative Ip | R2 corner | R2 train | Fit frames | Missing pred | R2 all - ridge_raw |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ridge_raw | 25 | 0.9408 | 0.8926 | 0.8740 | 0.9297 | 16397 | 0 | +0.0000 |
| ridge_features | 10 | 0.8549 | 0.7187 | 0.6994 | 0.8342 | 16397 | 0 | -0.0859 |
| formula_ridge | 10 | 0.7709 | 0.8434 | 0.8577 | 0.8673 | 16397 | 0 | -0.1699 |
| formula_gbm | 10 | 0.9004 | 0.9089 | 0.8874 | 0.9466 | 16397 | 0 | -0.0404 |
| base x constant (context) | 0 | 0.7996 | 0.6059 | 0.8065 | 0.8172 | 16397 | 0 | -0.1412 |

Shapes: training shapes_true_train_2d3a250c8f13_s4.parquet (reused, 0 frames without a boundary, 0 without a row); validation shapes_true_val_4ae30c3fe5c4_s4.parquet (reused, 0 without a boundary, 0 without a row). A missing base counts as the pooled mean in R2.

## Stage 2

Base constant c = 3.7093. Frames: all 5551, negative Ip 423, corner 302.

| Arm | Inputs | R2 all | R2 negative Ip | R2 corner | R2 train | Fit frames | Missing pred | R2 all - ridge_raw |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ridge_raw | 25 | 0.9408 | 0.8926 | 0.8740 | 0.9297 | 16397 | 0 | +0.0000 |
| ridge_features | 10 | 0.8524 | 0.7024 | 0.6801 | 0.8277 | 16397 | 0 | -0.0884 |
| formula_ridge | 10 | 0.8017 | 0.8510 | 0.8558 | 0.8770 | 16397 | 0 | -0.1391 |
| formula_gbm | 10 | 0.9108 | 0.9133 | 0.8729 | 0.9524 | 16397 | 0 | -0.0300 |
| base x constant (context) | 0 | 0.8130 | 0.6062 | 0.8147 | 0.8232 | 16397 | 0 | -0.1278 |

Shapes: training shapes_hybrid2d3a250c8f13_train_2d3a250c8f13_s4.parquet (reused, 0 frames without a boundary, 0 without a row); validation shapes_hybrid2d3a250c8f13_val_4ae30c3fe5c4_s4.parquet (reused, 0 without a boundary, 0 without a row). A missing base counts as the pooled mean in R2.

## Frames and inputs

Lighter label filter on the training frames (each rule alone, then together):

| Rule | Frames | Shots |
| --- | --- | --- |
| impossible_label | 15 | 11 |
| tf_under_threshold | 10 | 1 |
| ip_under_fraction_of_peak | 1 | 1 |
| flagged_shot | 0 | 0 |
| removed (union) | 26 | 13 |
| kept | 16397 | 300 |

Training statistics, every frame of the 5821 training shots of the v2 feature table: q_cyl band (1st to 99th percentile) 1.588 to 5.926, 10th percentile 1.780 (1289467 frames); ts_beta_p median 0.1088, 75th percentile 0.1801 (1270471 frames).

Inputs of ridge_features and the corrections: `ts_pe_peaking`, `ts_valid`, `shafranov_lambda`, `greenwald_frac`, `ip_frac`, `ohmic_flux_used`, `shaping`, `q_cyl_clipped`, `kappa`, `a_over_a0`. Every frame-table input passes `frame_table.assert_input_columns` and every feature's reads pass `features.check_reads`; labels enter only as the regression target and the label filter.

Runtime 1 s.
