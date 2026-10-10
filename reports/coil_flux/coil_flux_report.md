# Coil flux report

| Id | Check | Status | Headline |
| --- | --- | --- | --- |
| K3 | Total current | review | ratio p05/p50/p95 on reference frames: [-1.004, -0.994, 1.005] |
| K4 | Sign rule | pass | psi_physical = -1 x sign(Ip) x psi_stored over 8 shots |
| K5 | Agreement outside the plasma (away from coils), solenoid_flat | review | median rms / flux range by phase on fit frames: {'rising': 0.0226, 'flat_top': 0.0089, 'late': 0.009} |
| K5 | Agreement outside the plasma (near coils), solenoid_flat | review | median rms / flux range by phase on fit frames: {'rising': 0.026, 'flat_top': 0.0096, 'late': 0.01} |
| K5 | Flat-top agreement away from coils (solenoid_flat, fit frames) | pass | median 0.0089 of flux range |
| K5 | Validation shots not used in the fit (solenoid_flat, 50 shots) | review | median rms / flux range by phase: {'rising': 0.0235, 'flat_top': 0.009, 'late': 0.0133}; near coils {'rising': 0.0285, 'flat_top': 0.0098, 'late': 0.0153} |
| K6 | Coil scales | review | adopted solenoid_flat: shaping coils at 1, flat solenoid term 0.00897 Wb/rad per kA (0.00951 with the pattern at -9.264); per-coil scales with a per-frame constant 0.924 to 1.095; per_coil without constant 0.412 to 149.058, per-shot IQR up to 3.43 |
| K7 | Remainder | review | axis distance median 0.1288 m; amplitude/Ip spread (IQR/median) on flat-top 0.106 |
| K8 | Mirror symmetry | fail | max difference 1.79e-02 of flux range |
| K9 | MAST checks | review | medians by machine {'bz_ratio': {'DIII-D': -2.096, 'MAST': -1.351}, 'sol_flux_ratio': {'DIII-D': -0.137, 'MAST': -0.091}} (each machine under its adopted coil model, signed Ip) |

## Reference shots

`{"positive_20khz": ["0011c2afc9", "001270afa9", "001cbcc9e6"], "negative_ip": ["00000a10ac", "001bc5a4ae", "003ac617f1"], "rate_2khz": ["00000c4a7b", "0003ff34e7"]}`

## K5 by variant, median rms / flux range

| Variant | Fit flat-top, away | Fit flat-top, near | Fit rising, away | Fit late, away | Validation flat-top, away | Validation rising, away | Validation late, away |
| --- | --- | --- | --- | --- | --- | --- | --- |
| unscaled | 0.1941 | 0.1941 | 0.2827 | 0.1155 | 0.1726 | 0.3481 | 0.2592 |
| per_coil | 0.0279 | 0.0392 | 0.0849 | 0.0305 | 0.0306 | 0.0998 | 0.0468 |
| per_coil_constant | 0.0063 | 0.0069 | 0.0167 | 0.0067 | 0.0066 | 0.0194 | 0.0089 |
| solenoid_only | 0.0369 | 0.0533 | 0.0626 | 0.0240 | 0.0337 | 0.0744 | 0.0486 |
| solenoid_only_constant | 0.0066 | 0.0068 | 0.0159 | 0.0054 | 0.0067 | 0.0180 | 0.0090 |
| shared_plus_solenoid | 0.0355 | 0.0511 | 0.0687 | 0.0240 | 0.0334 | 0.0808 | 0.0489 |
| shared_plus_solenoid_constant | 0.0065 | 0.0068 | 0.0162 | 0.0055 | 0.0067 | 0.0183 | 0.0088 |
| solenoid_flat | 0.0089 | 0.0096 | 0.0226 | 0.0090 | 0.0090 | 0.0235 | 0.0133 |
| solenoid_flat_plus_pattern | 0.0088 | 0.0085 | 0.0218 | 0.0088 | 0.0090 | 0.0222 | 0.0129 |

## Coefficients

| Variant | Parameter | Kind | Value | Per-shot median | Per-shot IQR |
| --- | --- | --- | --- | --- | --- |
| per_coil | magnetics_F1A | free | 0.41200 | 0.6409 | 0.6969 |
| per_coil | magnetics_F2A | free | 0.56480 | 0.6278 | 1.5823 |
| per_coil | magnetics_F3A | free | 0.75931 | 0.7030 | 3.4294 |
| per_coil | magnetics_F4A | free | 1.14706 | 1.2893 | 0.9868 |
| per_coil | magnetics_F5A | free | 0.68884 | 0.8948 | 0.9683 |
| per_coil | magnetics_F6A | free | 1.06127 | 1.0629 | 0.1382 |
| per_coil | magnetics_F7A | free | 1.07156 | 0.9899 | 0.2546 |
| per_coil | magnetics_F8A | free | 1.02702 | 1.1929 | 1.7264 |
| per_coil | magnetics_F9A | free | 1.01163 | 1.1930 | 0.7708 |
| per_coil | magnetics_F1B | free | 0.68037 | 0.8865 | 0.5337 |
| per_coil | magnetics_F2B | free | 0.43999 | 0.6005 | 1.2300 |
| per_coil | magnetics_F3B | free | 0.74831 | 0.7679 | 1.2533 |
| per_coil | magnetics_F4B | free | 0.96668 | 1.1465 | 1.1628 |
| per_coil | magnetics_F5B | free | 0.92448 | 1.1747 | 0.7261 |
| per_coil | magnetics_F6B | free | 1.03544 | 0.9993 | 0.1205 |
| per_coil | magnetics_F7B | free | 1.06703 | 1.0148 | 0.1947 |
| per_coil | magnetics_F8B | free | 0.90941 | 0.8431 | 0.8484 |
| per_coil | magnetics_F9B | free | 0.97423 | 1.0642 | 0.7554 |
| per_coil | magnetics_ECOILA | free | 149.05833 | 145.6544 | 6.1105 |
| per_coil_constant | magnetics_F1A | free | 1.05246 | nan | nan |
| per_coil_constant | magnetics_F2A | free | 1.09515 | nan | nan |
| per_coil_constant | magnetics_F3A | free | 1.00632 | nan | nan |
| per_coil_constant | magnetics_F4A | free | 0.92786 | nan | nan |
| per_coil_constant | magnetics_F5A | free | 0.98155 | nan | nan |
| per_coil_constant | magnetics_F6A | free | 1.02917 | nan | nan |
| per_coil_constant | magnetics_F7A | free | 0.97809 | nan | nan |
| per_coil_constant | magnetics_F8A | free | 0.94190 | nan | nan |
| per_coil_constant | magnetics_F9A | free | 0.98065 | nan | nan |
| per_coil_constant | magnetics_F1B | free | 1.06734 | nan | nan |
| per_coil_constant | magnetics_F2B | free | 1.07288 | nan | nan |
| per_coil_constant | magnetics_F3B | free | 1.03446 | nan | nan |
| per_coil_constant | magnetics_F4B | free | 0.92352 | nan | nan |
| per_coil_constant | magnetics_F5B | free | 0.96554 | nan | nan |
| per_coil_constant | magnetics_F6B | free | 1.01526 | nan | nan |
| per_coil_constant | magnetics_F7B | free | 0.98356 | nan | nan |
| per_coil_constant | magnetics_F8B | free | 0.99195 | nan | nan |
| per_coil_constant | magnetics_F9B | free | 0.99524 | nan | nan |
| per_coil_constant | magnetics_ECOILA | free | -11.55165 | nan | nan |
| solenoid_only | magnetics_ECOILA | free | 147.50686 | nan | nan |
| solenoid_only_constant | magnetics_ECOILA | free | -9.22355 | nan | nan |
| shared_plus_solenoid | magnetics_F1A | shared | 1.01681 | nan | nan |
| shared_plus_solenoid | magnetics_F2A | shared | 1.01681 | nan | nan |
| shared_plus_solenoid | magnetics_F3A | shared | 1.01681 | nan | nan |
| shared_plus_solenoid | magnetics_F4A | shared | 1.01681 | nan | nan |
| shared_plus_solenoid | magnetics_F5A | shared | 1.01681 | nan | nan |
| shared_plus_solenoid | magnetics_F6A | shared | 1.01681 | nan | nan |
| shared_plus_solenoid | magnetics_F7A | shared | 1.01681 | nan | nan |
| shared_plus_solenoid | magnetics_F8A | shared | 1.01681 | nan | nan |
| shared_plus_solenoid | magnetics_F9A | shared | 1.01681 | nan | nan |
| shared_plus_solenoid | magnetics_F1B | shared | 1.01681 | nan | nan |
| shared_plus_solenoid | magnetics_F2B | shared | 1.01681 | nan | nan |
| shared_plus_solenoid | magnetics_F3B | shared | 1.01681 | nan | nan |
| shared_plus_solenoid | magnetics_F4B | shared | 1.01681 | nan | nan |
| shared_plus_solenoid | magnetics_F5B | shared | 1.01681 | nan | nan |
| shared_plus_solenoid | magnetics_F6B | shared | 1.01681 | nan | nan |
| shared_plus_solenoid | magnetics_F7B | shared | 1.01681 | nan | nan |
| shared_plus_solenoid | magnetics_F8B | shared | 1.01681 | nan | nan |
| shared_plus_solenoid | magnetics_F9B | shared | 1.01681 | nan | nan |
| shared_plus_solenoid | magnetics_ECOILA | free | 145.99611 | nan | nan |
| shared_plus_solenoid_constant | magnetics_F1A | shared | 1.00125 | nan | nan |
| shared_plus_solenoid_constant | magnetics_F2A | shared | 1.00125 | nan | nan |
| shared_plus_solenoid_constant | magnetics_F3A | shared | 1.00125 | nan | nan |
| shared_plus_solenoid_constant | magnetics_F4A | shared | 1.00125 | nan | nan |
| shared_plus_solenoid_constant | magnetics_F5A | shared | 1.00125 | nan | nan |
| shared_plus_solenoid_constant | magnetics_F6A | shared | 1.00125 | nan | nan |
| shared_plus_solenoid_constant | magnetics_F7A | shared | 1.00125 | nan | nan |
| shared_plus_solenoid_constant | magnetics_F8A | shared | 1.00125 | nan | nan |
| shared_plus_solenoid_constant | magnetics_F9A | shared | 1.00125 | nan | nan |
| shared_plus_solenoid_constant | magnetics_F1B | shared | 1.00125 | nan | nan |
| shared_plus_solenoid_constant | magnetics_F2B | shared | 1.00125 | nan | nan |
| shared_plus_solenoid_constant | magnetics_F3B | shared | 1.00125 | nan | nan |
| shared_plus_solenoid_constant | magnetics_F4B | shared | 1.00125 | nan | nan |
| shared_plus_solenoid_constant | magnetics_F5B | shared | 1.00125 | nan | nan |
| shared_plus_solenoid_constant | magnetics_F6B | shared | 1.00125 | nan | nan |
| shared_plus_solenoid_constant | magnetics_F7B | shared | 1.00125 | nan | nan |
| shared_plus_solenoid_constant | magnetics_F8B | shared | 1.00125 | nan | nan |
| shared_plus_solenoid_constant | magnetics_F9B | shared | 1.00125 | nan | nan |
| shared_plus_solenoid_constant | magnetics_ECOILA | free | -9.01331 | nan | nan |
| solenoid_flat | flat | flat_wb_per_ka | 0.00897 | nan | nan |
| solenoid_flat_plus_pattern | magnetics_ECOILA | free | -9.26449 | nan | nan |
| solenoid_flat_plus_pattern | flat | flat_wb_per_ka | 0.00951 | nan | nan |

## Frames

3412 frames analysed; boundary found on 100.000%.

