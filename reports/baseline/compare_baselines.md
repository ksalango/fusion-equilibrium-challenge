# Baseline comparison, split `/Users/kamia/Desktop/PROJECTS/fusion-challenge/fusion-equilibrium-challenge/splits/split_v1.json`

5821 training shots (1289474 frames), 870 validation shots scored on every 4th frame (full lists); 8 workers; references built.

Models: `ridge_raw`, `ridge_signed`, `physics_mean`, `hybrid_ridge` (`fusion_eq.models`), and two ceilings with no learned flux model: `ceiling_psi_basis` (the true validation maps encoded and decoded with the training basis of the stored map) and `ceiling_plasma_basis` (the same with the basis of the plasma target, the coil flux added back). `physics_mean`, `hybrid_ridge` and the two ceilings take q95 and betaN from the scalar ridge of `ridge_signed`, so their differences from `ridge_signed` come from the flux map alone. Coil model: solenoid_flat, fit of 2026-10-10T18:45:26+00:00; flat solenoid term 0.00897 Wb/rad per kA; mean plasma shape over 1289472 frames; 50 components.

## all (870 shots)

| Model | S | R2 psi | R2 q95,betaN | 1 - D_LCFS | Consistency | psi sign |
| --- | --- | --- | --- | --- | --- | --- |
| ridge_raw | 0.7838 | 0.8326 | 0.7445 | 0.9583 | 0.5921 | +1 |
| ridge_signed | 0.9219 | 0.9982 | 0.8183 | 0.9719 | 0.7645 | +1 |
| physics_mean | 0.8000 | 0.9910 | 0.8183 | 0.9285 | 0.1971 | +1 |
| hybrid_ridge | 0.9219 | 0.9982 | 0.8183 | 0.9719 | 0.7646 | +1 |
| ceiling_psi_basis | 0.9717 | 1.0000 | 0.8183 | 0.9950 | 0.9972 | +1 |
| ceiling_plasma_basis | 0.9716 | 1.0000 | 0.8183 | 0.9953 | 0.9967 | +1 |

## with_near_repeat (584 shots)

| Model | S | R2 psi | R2 q95,betaN | 1 - D_LCFS | Consistency | psi sign |
| --- | --- | --- | --- | --- | --- | --- |
| ridge_raw | 0.8075 | 0.8628 | 0.7571 | 0.9618 | 0.6161 | +1 |
| ridge_signed | 0.9265 | 0.9983 | 0.8267 | 0.9737 | 0.7802 | +1 |
| physics_mean | 0.8019 | 0.9915 | 0.8267 | 0.9308 | 0.1976 | +1 |
| hybrid_ridge | 0.9265 | 0.9983 | 0.8267 | 0.9737 | 0.7803 | +1 |
| ceiling_psi_basis | 0.9730 | 1.0000 | 0.8267 | 0.9953 | 0.9973 | +1 |
| ceiling_plasma_basis | 0.9731 | 1.0000 | 0.8267 | 0.9956 | 0.9975 | +1 |

## without_near_repeat (286 shots)

| Model | S | R2 psi | R2 q95,betaN | 1 - D_LCFS | Consistency | psi sign |
| --- | --- | --- | --- | --- | --- | --- |
| ridge_raw | 0.7262 | 0.7591 | 0.7147 | 0.9511 | 0.5321 | +1 |
| ridge_signed | 0.9104 | 0.9979 | 0.7986 | 0.9680 | 0.7249 | +1 |
| physics_mean | 0.7961 | 0.9898 | 0.7986 | 0.9238 | 0.1977 | +1 |
| hybrid_ridge | 0.9104 | 0.9979 | 0.7986 | 0.9680 | 0.7249 | +1 |
| ceiling_psi_basis | 0.9686 | 1.0000 | 0.7986 | 0.9943 | 0.9969 | +1 |
| ceiling_plasma_basis | 0.9683 | 1.0000 | 0.7986 | 0.9949 | 0.9950 | +1 |

## ip_positive (794 shots)

| Model | S | R2 psi | R2 q95,betaN | 1 - D_LCFS | Consistency | psi sign |
| --- | --- | --- | --- | --- | --- | --- |
| ridge_raw | 0.8946 | 0.9686 | 0.7831 | 0.9672 | 0.7383 | +1 |
| ridge_signed | 0.9239 | 0.9982 | 0.8224 | 0.9720 | 0.7716 | +1 |
| physics_mean | 0.8015 | 0.9909 | 0.8224 | 0.9286 | 0.2013 | +1 |
| hybrid_ridge | 0.9239 | 0.9982 | 0.8224 | 0.9720 | 0.7717 | +1 |
| ceiling_psi_basis | 0.9723 | 1.0000 | 0.8224 | 0.9950 | 0.9972 | +1 |
| ceiling_plasma_basis | 0.9722 | 1.0000 | 0.8224 | 0.9954 | 0.9968 | +1 |

## ip_negative (74 shots)

| Model | S | R2 psi | R2 q95,betaN | 1 - D_LCFS | Consistency | psi sign |
| --- | --- | --- | --- | --- | --- | --- |
| ridge_raw | 0.0862 | -0.9159 | -0.0038 | 0.8620 | 0.0000 | +1 |
| ridge_signed | 0.8725 | 0.9982 | 0.7133 | 0.9700 | 0.5975 | +1 |
| physics_mean | 0.7706 | 0.9916 | 0.7133 | 0.9276 | 0.1274 | +1 |
| hybrid_ridge | 0.8724 | 0.9982 | 0.7133 | 0.9700 | 0.5973 | +1 |
| ceiling_psi_basis | 0.9556 | 1.0000 | 0.7133 | 0.9944 | 0.9957 | +1 |
| ceiling_plasma_basis | 0.9556 | 1.0000 | 0.7133 | 0.9949 | 0.9958 | +1 |

Runtime (s): frames_s 0, fit_s 208, references_s 60, score_ridge_raw_s 80, residuals_ridge_raw_s 14, score_ridge_signed_s 81, residuals_ridge_signed_s 14, score_physics_mean_s 84, residuals_physics_mean_s 14, score_hybrid_ridge_s 89, residuals_hybrid_ridge_s 15, score_ceiling_psi_basis_s 100, residuals_ceiling_psi_basis_s 25, score_ceiling_plasma_basis_s 97, residuals_ceiling_plasma_basis_s 25
