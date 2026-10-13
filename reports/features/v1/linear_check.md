# Linear check of the features (X16), version 1

300 training shots (65262 frames), 100 validation shots scored at frame stride 4 (working lists). Ridge alpha 1.0, 50 components of the plasma target over mu0 R0 abs(Ip) (R0 1.67 m; frames below 20 kA left out of the fit). Every arm takes q95 and betaN from the scalar ridge of `ridge_signed`. Reference: `hybrid_ridge`, scored in this run (F61: 0.9186 on the working lists).

| Status | Arm | Inputs | S - hybrid_ridge |
| --- | --- | --- | --- |
| review | features_set_a | 29 | -0.0136 |
| review | features_set_b | 16 | -0.0150 |
| reference | hybrid_ridge | 25 | +0.0000 |

## all (100 shots)

| Arm | S | R2 psi | R2 q95,betaN | 1 - D_LCFS | Consistency | R_axis | Z_axis | kappa | tri_top | tri_bot | volume | li |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_ridge | 0.9186 | 0.9985 | 0.8375 | 0.9721 | 0.7331 | 0.7414 | 0.5450 | 0.5567 | 0.9213 | 0.9034 | 0.8122 | 0.6517 |
| features_set_a | 0.9050 | 0.9953 | 0.8375 | 0.9677 | 0.6759 | 0.6618 | 0.5282 | 0.5039 | 0.9087 | 0.8854 | 0.6729 | 0.5702 |
| features_set_b | 0.9037 | 0.9951 | 0.8375 | 0.9674 | 0.6699 | 0.6561 | 0.5457 | 0.4900 | 0.9100 | 0.8934 | 0.6473 | 0.5470 |

## ip_positive (91 shots)

| Arm | S | R2 psi | R2 q95,betaN | 1 - D_LCFS | Consistency | R_axis | Z_axis | kappa | tri_top | tri_bot | volume | li |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_ridge | 0.9209 | 0.9985 | 0.8370 | 0.9722 | 0.7447 | 0.7464 | 0.6163 | 0.5550 | 0.9246 | 0.9066 | 0.8161 | 0.6478 |
| features_set_a | 0.9073 | 0.9951 | 0.8370 | 0.9679 | 0.6884 | 0.6661 | 0.6310 | 0.4849 | 0.9115 | 0.8875 | 0.6825 | 0.5556 |
| features_set_b | 0.9058 | 0.9949 | 0.8370 | 0.9678 | 0.6812 | 0.6603 | 0.6339 | 0.4700 | 0.9130 | 0.8966 | 0.6631 | 0.5313 |

## ip_negative (9 shots)

| Arm | S | R2 psi | R2 q95,betaN | 1 - D_LCFS | Consistency | R_axis | Z_axis | kappa | tri_top | tri_bot | volume | li |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_ridge | 0.8858 | 0.9987 | 0.8172 | 0.9703 | 0.5847 | 0.4305 | -0.4857 | 0.5698 | 0.8807 | 0.8389 | 0.6983 | 0.6746 |
| features_set_a | 0.8778 | 0.9975 | 0.8172 | 0.9654 | 0.5506 | 0.3337 | -0.9556 | 0.6669 | 0.8714 | 0.8424 | 0.4057 | 0.7340 |
| features_set_b | 0.8712 | 0.9972 | 0.8172 | 0.9636 | 0.5191 | 0.3253 | -0.7292 | 0.6616 | 0.8716 | 0.8285 | 0.2213 | 0.7250 |

## Coefficient R2 (validation frames, no scoring)

Pooled R2 of the predicted coefficients of the per-Ip plasma target, frames above the Ip threshold.

| Inputs | Inputs count | Training | Validation |
| --- | --- | --- | --- |
| features_set_a | 29 | 0.7093 | 0.4953 |
| features_set_a without coil_pattern | 7 | 0.4462 | 0.0458 |
| features_set_a without plasma_content | 26 | 0.6923 | 0.4621 |
| features_set_a without field_current | 28 | 0.7070 | 0.4879 |
| features_set_a without shot_position | 26 | 0.6727 | 0.5226 |
| features_set_b | 16 | 0.6998 | 0.4730 |
| features_set_b without coil_terms | 7 | 0.4462 | 0.0458 |
| features_set_b without plasma_content | 13 | 0.6818 | 0.4385 |
| features_set_b without field_current | 15 | 0.6973 | 0.4646 |
| features_set_b without shot_position | 13 | 0.6590 | 0.4970 |
| raw inputs of hybrid_ridge | 25 | 0.3904 | 0.7122 |

Runtime: 50 s (reference_fit_s 0, per_ip_basis_s 20, coefficient_r2_s 2, references_s 0, score_hybrid_ridge_s 9, score_features_set_a_s 9, score_features_set_b_s 9)
