# Split report

Written by `scripts/find_near_repeats.py`; `make_split.py` and `leak_test.py` append.

## S1 Fingerprint coverage (review)

| Split | Shots | Fingerprinted | Too short |
| --- | --- | --- | --- |
| diii_d_train | 7041 | 6976 | 65 |
| diii_d_public_test | 874 | 870 | 4 |

## S2 Input distributions, training against test (review)

Per-shot medians over flat-top frames; largest Kolmogorov-Smirnov statistic first.

| Quantity | KS | p-value | Median difference (train IQR) | n train | n test |
| --- | --- | --- | --- | --- | --- |
| magnetics_F4B__mean | 0.041 | 1.55e-01 | -0.044 | 6959 | 865 |
| magnetics_F7B__mean | 0.040 | 1.72e-01 | +0.045 | 6959 | 865 |
| tf__mean | 0.040 | 1.73e-01 | -0.037 | 6959 | 865 |
| magnetics_F2A__mean | 0.038 | 2.12e-01 | +0.019 | 6959 | 865 |
| magnetics_F2B__mean | 0.037 | 2.26e-01 | +0.026 | 6959 | 865 |
| magnetics_F4A__mean | 0.036 | 2.79e-01 | +0.003 | 6959 | 865 |
| magnetics_F8A__mean | 0.034 | 3.28e-01 | -0.054 | 6959 | 865 |
| magnetics_F1A__mean | 0.033 | 3.48e-01 | +0.038 | 6959 | 865 |
| magnetics_F9A__mean | 0.033 | 3.73e-01 | +0.051 | 6959 | 865 |
| magnetics_F7A__mean | 0.033 | 3.74e-01 | +0.005 | 6959 | 865 |
| magnetics_F6A__mean | 0.031 | 4.58e-01 | -0.025 | 6959 | 865 |
| magnetics_F3B__mean | 0.030 | 4.89e-01 | +0.026 | 6959 | 865 |
| ts_core_ne_peak__interp | 0.030 | 5.03e-01 | +0.009 | 6927 | 861 |
| magnetics_F5A__mean | 0.029 | 5.13e-01 | -0.010 | 6959 | 865 |
| magnetics_F1B__mean | 0.028 | 5.59e-01 | +0.027 | 6959 | 865 |
| ts_core_te_peak__interp | 0.028 | 5.84e-01 | +0.017 | 6927 | 861 |
| magnetics_F5B__mean | 0.024 | 7.40e-01 | -0.016 | 6959 | 865 |
| magnetics_F8B__mean | 0.023 | 8.21e-01 | -0.003 | 6959 | 865 |
| ts_core_pe_mean__interp | 0.022 | 8.60e-01 | +0.006 | 6927 | 861 |
| sol__mean | 0.021 | 8.88e-01 | +0.018 | 6959 | 865 |
| magnetics_F6B__mean | 0.021 | 8.89e-01 | -0.006 | 6959 | 865 |
| magnetics_F3A__mean | 0.018 | 9.51e-01 | -0.035 | 6959 | 865 |
| ip__mean | 0.018 | 9.68e-01 | -0.001 | 6959 | 865 |
| magnetics_F9B__mean | 0.016 | 9.90e-01 | +0.031 | 6959 | 865 |

## S3 Nearest-neighbour distance among training shots (review)

| Shots | p05 | p25 | p50 | p75 | p95 | undefined |
| --- | --- | --- | --- | --- | --- | --- |
| training (other training shots) | 0.0186 | 0.0425 | 0.0659 | 0.1082 | 0.2248 | 65 |
| test (training shots) | 0.0203 | 0.0442 | 0.0694 | 0.1102 | 0.2303 | 4 |

Kolmogorov-Smirnov statistic between the two: 0.0328 (p = 3.66e-01).

## S4 Near-repeat prevalence (review)

| Threshold | Training shots with a near-repeat in training | Test shots with a near-repeat in training | Groups (all shots) | Shots in a group of 2 or more | Largest group |
| --- | --- | --- | --- | --- | --- |
| 0.05 | 33.930% | 31.236% | 6313 | 2459 | 29 |
| 0.10 | 70.998% | 70.709% | 4156 | 5258 | 35 |
| 0.20 | 92.118% | 91.876% | 2166 | 7040 | 73 |

Shots in the index: 7915; thresholds reported: [0.05, 0.1, 0.2]; provisional 0.1.

## Split `reports/split/split_provisional.json`

Mode `random`, threshold 0.1, seed 0; drawn 2026-10-10.

| S5 Split integrity | pass | train 5821, validation 870, local_test 350, working_train 300, working_validation 100 |

Training shots with a near-repeat in `train` (threshold 0.1): 4801 of 7041.

### S6 Split balance (review)

| List | Shots | Share 20 kHz | Share negative Ip | Median frames | Share no core Thomson |
| --- | --- | --- | --- | --- | --- |
| train | 5821 | 0.700 | 0.086 | 241 | 0.009 |
| validation | 870 | 0.701 | 0.085 | 237 | 0.007 |
| local_test | 350 | 0.700 | 0.086 | 243 | 0.017 |
| working_train | 300 | 0.703 | 0.087 | 240 | 0.003 |
| working_validation | 100 | 0.700 | 0.090 | 234 | 0.010 |
| test | 874 | 0.670 | 0.087 | 240 | 0.007 |

## S7 Leak test (review), split `reports/split/split_provisional.json`

Threshold 0.1; working subsets; validation frame stride 1; 300 training shots, 100 validation shots of which 17 have a near-repeat in training; arm B removes 22 training shots.

Runtime: basis 15 s, coefficients 4 s, references 116 s, per arm fit 0 s and scoring 124 s.

| Arm | Seed | Subset | Shots | S | R2 psi | R2 q95,betaN | 1 - D_LCFS | Consistency | psi sign |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 0 | all | 100 | 0.7656 | 0.8252 | 0.7519 | 0.9558 | 0.5169 | +1 |
| A | 0 | with_near_repeat | 17 | 0.8655 | 0.9401 | 0.8138 | 0.9656 | 0.6491 | +1 |
| A | 0 | without_near_repeat | 83 | 0.7394 | 0.7863 | 0.7343 | 0.9538 | 0.5073 | +1 |
| B | 0 | all | 100 | 0.7638 | 0.8238 | 0.7482 | 0.9563 | 0.5144 | +1 |
| B | 0 | with_near_repeat | 17 | 0.8358 | 0.9269 | 0.7881 | 0.9620 | 0.5579 | +1 |
| B | 0 | without_near_repeat | 83 | 0.7426 | 0.7887 | 0.7341 | 0.9551 | 0.5162 | +1 |
| C | 0 | all | 100 | 0.7660 | 0.8276 | 0.7515 | 0.9549 | 0.5132 | +1 |
| C | 0 | with_near_repeat | 17 | 0.8633 | 0.9388 | 0.8142 | 0.9652 | 0.6418 | +1 |
| C | 0 | without_near_repeat | 83 | 0.7403 | 0.7899 | 0.7337 | 0.9528 | 0.5029 | +1 |
| C | 1 | all | 100 | 0.7655 | 0.8229 | 0.7515 | 0.9562 | 0.5229 | +1 |
| C | 1 | with_near_repeat | 17 | 0.8703 | 0.9427 | 0.8163 | 0.9665 | 0.6638 | +1 |
| C | 1 | without_near_repeat | 83 | 0.7381 | 0.7824 | 0.7333 | 0.9541 | 0.5119 | +1 |
| C | 2 | all | 100 | 0.7673 | 0.8269 | 0.7535 | 0.9559 | 0.5197 | +1 |
| C | 2 | with_near_repeat | 17 | 0.8623 | 0.9397 | 0.8096 | 0.9652 | 0.6377 | +1 |
| C | 2 | without_near_repeat | 83 | 0.7421 | 0.7886 | 0.7369 | 0.9539 | 0.5120 | +1 |

Leak A minus B on all validation shots: +0.0018; spread of the C arms (max minus min): 0.0018, A minus mean C: -0.0007.

## S7 Leak test (review), split `reports/split/split_provisional.json`

Threshold 0.1; full parts; validation frame stride 4; 5821 training shots, 870 validation shots of which 584 have a near-repeat in training; arm B removes 1974 training shots.

Runtime: basis 20 s, coefficients 86 s, references 281 s, per arm fit 0 s and scoring 288 s.

| Arm | Seed | Subset | Shots | S | R2 psi | R2 q95,betaN | 1 - D_LCFS | Consistency | psi sign |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 0 | all | 870 | 0.7838 | 0.8326 | 0.7445 | 0.9583 | 0.5921 | +1 |
| A | 0 | with_near_repeat | 584 | 0.8075 | 0.8628 | 0.7571 | 0.9618 | 0.6161 | +1 |
| A | 0 | without_near_repeat | 286 | 0.7262 | 0.7591 | 0.7147 | 0.9511 | 0.5321 | +1 |
| B | 0 | all | 870 | 0.7801 | 0.8283 | 0.7414 | 0.9574 | 0.5879 | +1 |
| B | 0 | with_near_repeat | 584 | 0.7991 | 0.8546 | 0.7506 | 0.9605 | 0.6021 | +1 |
| B | 0 | without_near_repeat | 286 | 0.7337 | 0.7643 | 0.7195 | 0.9512 | 0.5517 | +1 |
| C | 0 | all | 870 | 0.7834 | 0.8324 | 0.7448 | 0.9585 | 0.5901 | +1 |
| C | 0 | with_near_repeat | 584 | 0.8066 | 0.8623 | 0.7572 | 0.9620 | 0.6129 | +1 |
| C | 0 | without_near_repeat | 286 | 0.7266 | 0.7595 | 0.7152 | 0.9515 | 0.5325 | +1 |
| C | 1 | all | 870 | 0.7840 | 0.8324 | 0.7438 | 0.9582 | 0.5937 | +1 |
| C | 1 | with_near_repeat | 584 | 0.8082 | 0.8630 | 0.7564 | 0.9617 | 0.6194 | +1 |
| C | 1 | without_near_repeat | 286 | 0.7250 | 0.7580 | 0.7136 | 0.9509 | 0.5297 | +1 |
| C | 2 | all | 870 | 0.7835 | 0.8326 | 0.7442 | 0.9581 | 0.5909 | +1 |
| C | 2 | with_near_repeat | 584 | 0.8072 | 0.8627 | 0.7567 | 0.9617 | 0.6152 | +1 |
| C | 2 | without_near_repeat | 286 | 0.7259 | 0.7593 | 0.7146 | 0.9507 | 0.5301 | +1 |

Leak A minus B on all validation shots: +0.0038; spread of the C arms (max minus min): 0.0006, A minus mean C: +0.0002.

## Split `/Users/kamia/Desktop/PROJECTS/fusion-challenge/fusion-equilibrium-challenge/splits/split_v1.json`

Mode `random`, threshold 0.1, seed 0; drawn 2026-10-10.

| S5 Split integrity | pass | train 5821, validation 870, local_test 350, working_train 300, working_validation 100 |

Training shots with a near-repeat in `train` (threshold 0.1): 4801 of 7041.

### S6 Split balance (review)

| List | Shots | Share 20 kHz | Share negative Ip | Median frames | Share no core Thomson |
| --- | --- | --- | --- | --- | --- |
| train | 5821 | 0.700 | 0.086 | 241 | 0.009 |
| validation | 870 | 0.701 | 0.085 | 237 | 0.007 |
| local_test | 350 | 0.700 | 0.086 | 243 | 0.017 |
| working_train | 300 | 0.703 | 0.087 | 240 | 0.003 |
| working_validation | 100 | 0.700 | 0.090 | 234 | 0.010 |
| test | 874 | 0.670 | 0.087 | 240 | 0.007 |
