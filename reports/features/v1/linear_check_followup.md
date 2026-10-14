# Linear check follow-up (X16), version 1

300 training shots (65262 frames), 100 validation shots scored at frame stride 4 (working lists); ridge alpha 1.0, 50 components; frames below 20 kA left out of every ridge fit. Every arm takes q95 and betaN from the scalar ridge of `ridge_signed`. Variants beside version 1: `masked_current_flux`, `ip_frac_unscaled`; by phase: yes. Diagnostic only: version 1 and its outputs are unchanged.

- `version 1`: the plasma target over mu0 R0 abs(Ip), as in `linear_check.md`; `hybrid_ridge` predicts the plasma target itself from its 25 inputs.
- `masked_current_flux`: the flux of the current inside the scorer's boundary (plasma target minus the vacuum remainder, variant e of F79) over mu0 R0 abs(Ip), its own basis; `hybrid_ridge` refitted on this target, basis and frames from its own inputs; the true remainder added back before scoring.
- `ip_frac_unscaled`: the version-1 target; `ip_frac` with identity scaling. `hybrid_ridge` does not read `ip_frac`; its version-1 score is the reference.
- Masked-current basis: 50 components, 1.0000 of the variance over 65262 frames; frames without a boundary (target zero): training 0.0000, validation 0.0000.

S - reference: against `hybrid_ridge` of the same variant (of version 1 for `ip_frac_unscaled`).

## Scores, all (100 shots)

| Variant | Arm | S | S - v1 same arm | S - reference | R2 psi | R2 q95,betaN | 1 - D_LCFS | Consistency | R_axis | Z_axis | kappa | tri_top | tri_bot | volume | li |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| version 1 | hybrid_ridge | 0.9186 | +0.0000 | +0.0000 | 0.9985 | 0.8375 | 0.9721 | 0.7331 | 0.7414 | 0.5450 | 0.5567 | 0.9213 | 0.9034 | 0.8122 | 0.6517 |
| version 1 | features_set_a | 0.9050 | +0.0000 | -0.0136 | 0.9953 | 0.8375 | 0.9677 | 0.6759 | 0.6618 | 0.5282 | 0.5039 | 0.9087 | 0.8854 | 0.6729 | 0.5702 |
| version 1 | features_set_b | 0.9037 | +0.0000 | -0.0150 | 0.9951 | 0.8375 | 0.9674 | 0.6699 | 0.6561 | 0.5457 | 0.4900 | 0.9100 | 0.8934 | 0.6473 | 0.5470 |
| masked_current_flux | hybrid_ridge | 0.8947 | -0.0239 | +0.0000 | 0.9978 | 0.8375 | 0.9670 | 0.6181 | 0.5883 | 0.2399 | 0.4789 | 0.8798 | 0.8954 | 0.7204 | 0.5239 |
| masked_current_flux | features_set_a | 0.9055 | +0.0005 | +0.0108 | 0.9959 | 0.8375 | 0.9677 | 0.6770 | 0.6653 | 0.5236 | 0.4871 | 0.9071 | 0.8778 | 0.7437 | 0.5344 |
| masked_current_flux | features_set_b | 0.9048 | +0.0011 | +0.0101 | 0.9957 | 0.8375 | 0.9677 | 0.6737 | 0.6638 | 0.5394 | 0.4764 | 0.9099 | 0.8858 | 0.7317 | 0.5090 |
| ip_frac_unscaled | features_set_a | 0.9050 | +0.0000 | -0.0136 | 0.9953 | 0.8375 | 0.9677 | 0.6759 | 0.6619 | 0.5282 | 0.5039 | 0.9087 | 0.8854 | 0.6729 | 0.5703 |
| ip_frac_unscaled | features_set_b | 0.9037 | +0.0000 | -0.0150 | 0.9951 | 0.8375 | 0.9674 | 0.6699 | 0.6561 | 0.5457 | 0.4900 | 0.9100 | 0.8934 | 0.6473 | 0.5470 |

## Scores, ip_positive (91 shots)

| Variant | Arm | S | S - v1 same arm | S - reference | R2 psi | R2 q95,betaN | 1 - D_LCFS | Consistency | R_axis | Z_axis | kappa | tri_top | tri_bot | volume | li |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| version 1 | hybrid_ridge | 0.9209 | +0.0000 | +0.0000 | 0.9985 | 0.8370 | 0.9722 | 0.7447 | 0.7464 | 0.6163 | 0.5550 | 0.9246 | 0.9066 | 0.8161 | 0.6478 |
| version 1 | features_set_a | 0.9073 | +0.0000 | -0.0135 | 0.9951 | 0.8370 | 0.9679 | 0.6884 | 0.6661 | 0.6310 | 0.4849 | 0.9115 | 0.8875 | 0.6825 | 0.5556 |
| version 1 | features_set_b | 0.9058 | +0.0000 | -0.0151 | 0.9949 | 0.8370 | 0.9678 | 0.6812 | 0.6603 | 0.6339 | 0.4700 | 0.9130 | 0.8966 | 0.6631 | 0.5313 |
| masked_current_flux | hybrid_ridge | 0.8959 | -0.0250 | +0.0000 | 0.9977 | 0.8370 | 0.9672 | 0.6243 | 0.5977 | 0.2841 | 0.4665 | 0.8776 | 0.9009 | 0.7319 | 0.5117 |
| masked_current_flux | features_set_a | 0.9079 | +0.0005 | +0.0120 | 0.9957 | 0.8370 | 0.9679 | 0.6895 | 0.6715 | 0.6277 | 0.4679 | 0.9090 | 0.8805 | 0.7514 | 0.5182 |
| masked_current_flux | features_set_b | 0.9068 | +0.0010 | +0.0109 | 0.9955 | 0.8370 | 0.9680 | 0.6844 | 0.6689 | 0.6297 | 0.4557 | 0.9122 | 0.8886 | 0.7441 | 0.4916 |
| ip_frac_unscaled | features_set_a | 0.9073 | +0.0000 | -0.0135 | 0.9951 | 0.8370 | 0.9679 | 0.6885 | 0.6661 | 0.6310 | 0.4850 | 0.9115 | 0.8875 | 0.6825 | 0.5557 |
| ip_frac_unscaled | features_set_b | 0.9058 | +0.0000 | -0.0151 | 0.9949 | 0.8370 | 0.9678 | 0.6812 | 0.6603 | 0.6340 | 0.4700 | 0.9130 | 0.8966 | 0.6631 | 0.5313 |

## Scores, ip_negative (9 shots)

| Variant | Arm | S | S - v1 same arm | S - reference | R2 psi | R2 q95,betaN | 1 - D_LCFS | Consistency | R_axis | Z_axis | kappa | tri_top | tri_bot | volume | li |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| version 1 | hybrid_ridge | 0.8858 | +0.0000 | +0.0000 | 0.9987 | 0.8172 | 0.9703 | 0.5847 | 0.4305 | -0.4857 | 0.5698 | 0.8807 | 0.8389 | 0.6983 | 0.6746 |
| version 1 | features_set_a | 0.8778 | +0.0000 | -0.0080 | 0.9975 | 0.8172 | 0.9654 | 0.5506 | 0.3337 | -0.9556 | 0.6669 | 0.8714 | 0.8424 | 0.4057 | 0.7340 |
| version 1 | features_set_b | 0.8712 | +0.0000 | -0.0146 | 0.9972 | 0.8172 | 0.9636 | 0.5191 | 0.3253 | -0.7292 | 0.6616 | 0.8716 | 0.8285 | 0.2213 | 0.7250 |
| masked_current_flux | hybrid_ridge | 0.8644 | -0.0214 | +0.0000 | 0.9987 | 0.8172 | 0.9653 | 0.4799 | 0.0426 | -0.4030 | 0.5848 | 0.8887 | 0.7847 | 0.4088 | 0.6497 |
| masked_current_flux | features_set_a | 0.8787 | +0.0008 | +0.0143 | 0.9982 | 0.8172 | 0.9657 | 0.5525 | 0.2698 | -0.9789 | 0.6518 | 0.8779 | 0.8226 | 0.5289 | 0.7169 |
| masked_current_flux | features_set_b | 0.8758 | +0.0046 | +0.0114 | 0.9980 | 0.8172 | 0.9644 | 0.5394 | 0.3099 | -0.7650 | 0.6540 | 0.8784 | 0.8289 | 0.3979 | 0.7064 |
| ip_frac_unscaled | features_set_a | 0.8778 | +0.0000 | -0.0080 | 0.9975 | 0.8172 | 0.9654 | 0.5506 | 0.3337 | -0.9559 | 0.6669 | 0.8714 | 0.8427 | 0.4058 | 0.7339 |
| ip_frac_unscaled | features_set_b | 0.8712 | -0.0000 | -0.0146 | 0.9972 | 0.8172 | 0.9636 | 0.5190 | 0.3253 | -0.7295 | 0.6615 | 0.8716 | 0.8285 | 0.2214 | 0.7249 |

## Scores, phase rising (1103 validation frames of 100 shots)

| Variant | Arm | S | S - v1 same arm | S - reference | R2 psi | R2 q95,betaN | 1 - D_LCFS | Consistency | R_axis | Z_axis | kappa | tri_top | tri_bot | volume | li |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| version 1 | hybrid_ridge | 0.8731 | +0.0000 | +0.0000 | 0.9973 | 0.7759 | 0.9715 | 0.5552 | 0.6779 | -0.7586 | 0.1935 | 0.8974 | 0.9387 | 0.7490 | 0.4301 |
| version 1 | features_set_a | 0.8731 | +0.0000 | -0.0000 | 0.9959 | 0.7759 | 0.9676 | 0.5610 | 0.7138 | -0.0165 | 0.3244 | 0.9197 | 0.9199 | 0.6186 | 0.4306 |
| version 1 | features_set_b | 0.8726 | +0.0000 | -0.0005 | 0.9957 | 0.7759 | 0.9675 | 0.5590 | 0.7085 | 0.0355 | 0.3243 | 0.9246 | 0.9314 | 0.5990 | 0.3894 |
| masked_current_flux | hybrid_ridge | 0.8502 | -0.0229 | +0.0000 | 0.9968 | 0.7759 | 0.9630 | 0.4464 | 0.5115 | -0.0432 | 0.0510 | 0.8357 | 0.9095 | 0.5315 | 0.2856 |
| masked_current_flux | features_set_a | 0.8746 | +0.0016 | +0.0245 | 0.9969 | 0.7759 | 0.9684 | 0.5655 | 0.6920 | -0.0194 | 0.3149 | 0.9249 | 0.9214 | 0.7051 | 0.4001 |
| masked_current_flux | features_set_b | 0.8744 | +0.0018 | +0.0242 | 0.9969 | 0.7759 | 0.9686 | 0.5643 | 0.6917 | 0.0311 | 0.3140 | 0.9311 | 0.9273 | 0.7013 | 0.3535 |
| ip_frac_unscaled | features_set_a | 0.8731 | -0.0000 | -0.0000 | 0.9959 | 0.7759 | 0.9676 | 0.5609 | 0.7138 | -0.0161 | 0.3243 | 0.9196 | 0.9198 | 0.6186 | 0.4305 |
| ip_frac_unscaled | features_set_b | 0.8726 | +0.0000 | -0.0005 | 0.9957 | 0.7759 | 0.9675 | 0.5590 | 0.7085 | 0.0360 | 0.3242 | 0.9246 | 0.9313 | 0.5990 | 0.3893 |

## Scores, phase flat_top (4308 validation frames of 99 shots)

| Variant | Arm | S | S - v1 same arm | S - reference | R2 psi | R2 q95,betaN | 1 - D_LCFS | Consistency | R_axis | Z_axis | kappa | tri_top | tri_bot | volume | li |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| version 1 | hybrid_ridge | 0.9246 | +0.0000 | +0.0000 | 0.9983 | 0.8095 | 0.9728 | 0.7842 | 0.7408 | 0.8007 | 0.6112 | 0.9296 | 0.8970 | 0.8187 | 0.6918 |
| version 1 | features_set_a | 0.9042 | +0.0000 | -0.0204 | 0.9940 | 0.8095 | 0.9685 | 0.6959 | 0.6082 | 0.6393 | 0.5482 | 0.9069 | 0.8774 | 0.6892 | 0.6022 |
| version 1 | features_set_b | 0.9024 | +0.0000 | -0.0222 | 0.9938 | 0.8095 | 0.9682 | 0.6880 | 0.6014 | 0.6529 | 0.5312 | 0.9073 | 0.8847 | 0.6642 | 0.5741 |
| masked_current_flux | hybrid_ridge | 0.8965 | -0.0281 | +0.0000 | 0.9973 | 0.8095 | 0.9690 | 0.6485 | 0.5708 | 0.2852 | 0.5522 | 0.8935 | 0.8961 | 0.7524 | 0.5891 |
| masked_current_flux | features_set_a | 0.9043 | +0.0001 | +0.0077 | 0.9947 | 0.8095 | 0.9682 | 0.6949 | 0.6205 | 0.6343 | 0.5321 | 0.9043 | 0.8665 | 0.7359 | 0.5705 |
| masked_current_flux | features_set_b | 0.9031 | +0.0007 | +0.0066 | 0.9944 | 0.8095 | 0.9681 | 0.6897 | 0.6181 | 0.6461 | 0.5199 | 0.9063 | 0.8751 | 0.7223 | 0.5398 |
| ip_frac_unscaled | features_set_a | 0.9042 | -0.0000 | -0.0204 | 0.9940 | 0.8095 | 0.9685 | 0.6959 | 0.6082 | 0.6392 | 0.5483 | 0.9070 | 0.8774 | 0.6891 | 0.6022 |
| ip_frac_unscaled | features_set_b | 0.9024 | -0.0000 | -0.0222 | 0.9938 | 0.8095 | 0.9682 | 0.6879 | 0.6014 | 0.6528 | 0.5312 | 0.9074 | 0.8846 | 0.6642 | 0.5741 |

## Scores, phase late (140 validation frames of 49 shots)

| Variant | Arm | S | S - v1 same arm | S - reference | R2 psi | R2 q95,betaN | 1 - D_LCFS | Consistency | R_axis | Z_axis | kappa | tri_top | tri_bot | volume | li |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| version 1 | hybrid_ridge | 0.8942 | +0.0000 | +0.0000 | 0.9980 | 0.7422 | 0.9569 | 0.6915 | 0.6064 | 0.7880 | 0.5131 | 0.8539 | 0.7796 | 0.8400 | 0.4598 |
| version 1 | features_set_a | 0.8583 | +0.0000 | -0.0359 | 0.9948 | 0.7422 | 0.9453 | 0.5265 | 0.6128 | 0.5306 | 0.0555 | 0.8832 | 0.8048 | 0.5064 | 0.2925 |
| version 1 | features_set_b | 0.8545 | +0.0000 | -0.0398 | 0.9945 | 0.7422 | 0.9458 | 0.5079 | 0.6120 | 0.4834 | 0.0381 | 0.8872 | 0.8055 | 0.4437 | 0.2851 |
| masked_current_flux | hybrid_ridge | 0.8635 | -0.0307 | +0.0000 | 0.9980 | 0.7422 | 0.9367 | 0.5482 | 0.5425 | 0.4573 | 0.2984 | 0.8004 | 0.7569 | 0.8052 | 0.1763 |
| masked_current_flux | features_set_a | 0.8643 | +0.0060 | +0.0007 | 0.9962 | 0.7422 | 0.9499 | 0.5501 | 0.6211 | 0.5255 | -0.0011 | 0.8649 | 0.8085 | 0.8152 | 0.2154 |
| masked_current_flux | features_set_b | 0.8626 | +0.0082 | -0.0009 | 0.9960 | 0.7422 | 0.9511 | 0.5418 | 0.6271 | 0.4788 | -0.0267 | 0.8703 | 0.8176 | 0.7905 | 0.2085 |
| ip_frac_unscaled | features_set_a | 0.8583 | +0.0000 | -0.0359 | 0.9948 | 0.7422 | 0.9453 | 0.5267 | 0.6128 | 0.5310 | 0.0561 | 0.8832 | 0.8048 | 0.5066 | 0.2927 |
| ip_frac_unscaled | features_set_b | 0.8545 | +0.0000 | -0.0398 | 0.9945 | 0.7422 | 0.9458 | 0.5079 | 0.6120 | 0.4839 | 0.0388 | 0.8865 | 0.8052 | 0.4438 | 0.2853 |

## Coefficient R2, version 1

Pooled R2 of the predicted coefficients of the variant's per-Ip target, frames above the Ip threshold (validation: the strided frames).

| Inputs | Count | Training | Validation | rising training | flat_top training | late training | rising validation | flat_top validation | late validation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| features_set_a | 29 | 0.7093 | 0.4953 | 0.5674 | 0.6063 | 0.7796 | 0.6034 | 0.4369 | 0.3581 |
| features_set_a without coil_pattern | 7 | 0.4462 | 0.0458 | 0.2848 | 0.2998 | 0.5311 | 0.1830 | -0.0606 | -0.0087 |
| features_set_a without plasma_content | 26 | 0.6923 | 0.4621 | 0.5343 | 0.5675 | 0.7777 | 0.5751 | 0.4007 | 0.3151 |
| features_set_a without field_current | 28 | 0.7070 | 0.4879 | 0.5701 | 0.6006 | 0.7784 | 0.6039 | 0.4277 | 0.3365 |
| features_set_a without shot_position | 26 | 0.6727 | 0.5226 | 0.4503 | 0.5983 | 0.7374 | 0.4985 | 0.5118 | 0.3486 |
| features_set_b | 16 | 0.6998 | 0.4730 | 0.5562 | 0.5882 | 0.7751 | 0.5885 | 0.4111 | 0.3311 |
| features_set_b without coil_terms | 7 | 0.4462 | 0.0458 | 0.2848 | 0.2998 | 0.5311 | 0.1830 | -0.0606 | -0.0087 |
| features_set_b without plasma_content | 13 | 0.6818 | 0.4385 | 0.5207 | 0.5471 | 0.7730 | 0.5601 | 0.3736 | 0.2801 |
| features_set_b without field_current | 15 | 0.6973 | 0.4646 | 0.5587 | 0.5818 | 0.7738 | 0.5878 | 0.4013 | 0.3031 |
| features_set_b without shot_position | 13 | 0.6590 | 0.4970 | 0.4217 | 0.5770 | 0.7301 | 0.4618 | 0.4876 | 0.3282 |
| raw inputs of hybrid_ridge | 25 | 0.3904 | 0.7122 | 0.6563 | 0.7577 | 0.0860 | 0.6244 | 0.7324 | 0.5712 |

Share of the target's variance (sum of squared deviations from the overall mean) by phase: train rising 0.114, flat_top 0.333, late 0.553; validation rising 0.249, flat_top 0.683, late 0.067.

## Coefficient R2, masked_current_flux

Pooled R2 of the predicted coefficients of the variant's per-Ip target, frames above the Ip threshold (validation: the strided frames).

| Inputs | Count | Training | Validation | rising training | flat_top training | late training | rising validation | flat_top validation | late validation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| features_set_a | 29 | 0.7198 | 0.4894 | 0.5645 | 0.5773 | 0.7919 | 0.6352 | 0.3878 | 0.4042 |
| features_set_a without coil_pattern | 7 | 0.4503 | 0.0147 | 0.2734 | 0.2306 | 0.5428 | 0.2163 | -0.1705 | 0.0097 |
| features_set_a without plasma_content | 26 | 0.6910 | 0.4397 | 0.4918 | 0.5062 | 0.7891 | 0.5820 | 0.3342 | 0.3352 |
| features_set_a without field_current | 28 | 0.7172 | 0.4810 | 0.5688 | 0.5705 | 0.7905 | 0.6357 | 0.3777 | 0.3696 |
| features_set_a without shot_position | 26 | 0.6839 | 0.5108 | 0.4286 | 0.5725 | 0.7493 | 0.5221 | 0.4497 | 0.4437 |
| features_set_b | 16 | 0.7102 | 0.4681 | 0.5567 | 0.5574 | 0.7867 | 0.6221 | 0.3621 | 0.3719 |
| features_set_b without coil_terms | 7 | 0.4503 | 0.0147 | 0.2734 | 0.2306 | 0.5428 | 0.2163 | -0.1705 | 0.0097 |
| features_set_b without plasma_content | 13 | 0.6806 | 0.4189 | 0.4819 | 0.4846 | 0.7837 | 0.5693 | 0.3095 | 0.2989 |
| features_set_b without field_current | 15 | 0.7072 | 0.4580 | 0.5600 | 0.5495 | 0.7853 | 0.6204 | 0.3506 | 0.3317 |
| features_set_b without shot_position | 13 | 0.6703 | 0.4859 | 0.3996 | 0.5499 | 0.7414 | 0.4809 | 0.4265 | 0.4164 |
| raw inputs of hybrid_ridge | 25 | 0.3707 | 0.7203 | 0.6783 | 0.7300 | 0.0815 | 0.6522 | 0.7179 | 0.5419 |

Share of the target's variance (sum of squared deviations from the overall mean) by phase: train rising 0.109, flat_top 0.311, late 0.580; validation rising 0.256, flat_top 0.682, late 0.061.

## Coefficient R2, ip_frac_unscaled

Pooled R2 of the predicted coefficients of the variant's per-Ip target, frames above the Ip threshold (validation: the strided frames).

| Inputs | Count | Training | Validation | rising training | flat_top training | late training | rising validation | flat_top validation | late validation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| features_set_a | 29 | 0.7093 | 0.4953 | 0.5674 | 0.6063 | 0.7796 | 0.6034 | 0.4369 | 0.3581 |
| features_set_a without coil_pattern | 7 | 0.4462 | 0.0458 | 0.2848 | 0.2997 | 0.5311 | 0.1831 | -0.0606 | -0.0086 |
| features_set_a without plasma_content | 26 | 0.6923 | 0.4621 | 0.5343 | 0.5675 | 0.7777 | 0.5751 | 0.4007 | 0.3152 |
| features_set_a without field_current | 28 | 0.7070 | 0.4879 | 0.5700 | 0.6006 | 0.7784 | 0.6039 | 0.4277 | 0.3365 |
| features_set_a without shot_position | 26 | 0.6727 | 0.5226 | 0.4503 | 0.5983 | 0.7374 | 0.4985 | 0.5118 | 0.3486 |
| features_set_b | 16 | 0.6998 | 0.4730 | 0.5562 | 0.5882 | 0.7751 | 0.5885 | 0.4110 | 0.3312 |
| features_set_b without coil_terms | 7 | 0.4462 | 0.0458 | 0.2848 | 0.2997 | 0.5311 | 0.1831 | -0.0606 | -0.0086 |
| features_set_b without plasma_content | 13 | 0.6818 | 0.4385 | 0.5207 | 0.5471 | 0.7730 | 0.5601 | 0.3736 | 0.2801 |
| features_set_b without field_current | 15 | 0.6973 | 0.4646 | 0.5586 | 0.5818 | 0.7738 | 0.5878 | 0.4013 | 0.3032 |
| features_set_b without shot_position | 13 | 0.6590 | 0.4970 | 0.4217 | 0.5770 | 0.7301 | 0.4618 | 0.4876 | 0.3282 |
| raw inputs of hybrid_ridge | 25 | 0.3904 | 0.7122 | 0.6563 | 0.7577 | 0.0860 | 0.6244 | 0.7324 | 0.5712 |

Share of the target's variance (sum of squared deviations from the overall mean) by phase: train rising 0.114, flat_top 0.333, late 0.553; validation rising 0.249, flat_top 0.683, late 0.067.

Runtime: 281 s (reference_fit_s 0, per_ip_basis_s 0, masked_target_s 106, coefficient_r2_s 1, references_s 0, phase_references_s 3, score_hybrid_ridge_s 20, score_features_set_a_s 20, score_features_set_b_s 22, score_masked_current_flux: hybrid_ridge_s 21, score_masked_current_flux: features_set_a_s 22, score_masked_current_flux: features_set_b_s 22, score_ip_frac_unscaled: features_set_a_s 22, score_ip_frac_unscaled: features_set_b_s 22)
