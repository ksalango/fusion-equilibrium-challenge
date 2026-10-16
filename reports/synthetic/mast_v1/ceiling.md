# Ceiling of the current basis on synthetic MAST equilibria (mast_v1)

Run of 2026-10-16T05:35:10+00:00, repository commit 64d128ebfbd029545619ea24f935ce16ac87af20. 329 converged FreeGS equilibria of 384 targets ({'LSN': {'targets': 192, 'converged': 165}, 'DN': {'targets': 192, 'converged': 164}}); FreeGS {'version': '0.8.3.dev54+g9fbec2056', 'url': 'https://github.com/freegs-plasma/freegs', 'commit': '9fbec2056b4322ea253bb5614896d549365163f3'}; coil model sol_half; basis {'n_components': 50, 'file_sha256': '2724d7063045ab1fbea9c9dc96be613609dafecaf571c80b6ff0881848d60382'}. No MAST equilibrium label read. True q95 and betaN in every variant.

## Score terms

One global flux sign per variant. DIII-D rows from `reports/current_target/ceiling.json` (validation frames of the working lists, true q95 and betaN).

| Variant | S | r2_psi | r2_qb | one_minus_dlcfs | consistency | R_axis | Z_axis | kappa | tri_top | tri_bot | volume | li |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DIII-D F90 (h at 50: basis, 10-component remainder) | 0.9981 | 1.0000 | 1.0000 | 0.9930 | 0.9939 | 0.9999 | 0.9996 | 0.9879 | 0.9899 | 0.9837 | 0.9974 | 0.9987 |
| DIII-D a_masked (exact share, no remainder) | 0.9784 | 0.9984 | 1.0000 | 0.9844 | 0.9041 | 0.9343 | 0.9994 | 0.9550 | 0.9870 | 0.9830 | 0.4846 | 0.9853 |
| MAST synthetic truth | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| MAST synthetic exact_share | 0.9991 | 1.0000 | 1.0000 | 0.9955 | 0.9976 | 0.9994 | 0.9998 | 0.9977 | 0.9950 | 0.9920 | 0.9999 | 0.9992 |
| MAST synthetic basis_share | 0.9932 | 0.9999 | 1.0000 | 0.9781 | 0.9769 | 0.9982 | 0.9983 | 0.9646 | 0.9534 | 0.9443 | 0.9932 | 0.9863 |
| MAST synthetic basis_flux | 0.9959 | 1.0000 | 1.0000 | 0.9859 | 0.9865 | 0.9996 | 0.9981 | 0.9915 | 0.9604 | 0.9608 | 0.9992 | 0.9957 |
| MAST synthetic mean_share | 0.7811 | 0.9465 | 1.0000 | 0.8175 | 0.1440 | -1.9263 | 0.1029 | -0.6324 | 0.7126 | 0.1928 | -1.6938 | -3.7771 |

## Per-frame boundary distance and share diagnostics

| Variant | D_LCFS median (cm) | p95 (cm) | predicted boundary missing |
| --- | --- | --- | --- |
| truth | 0.00 | 0.00 | 0 |
| exact_share | 0.23 | 0.71 | 0 |
| basis_share | 0.76 | 6.30 | 0 |
| basis_flux | 0.65 | 5.79 | 0 |
| mean_share | 13.80 | 35.67 | 0 |

| Quantity | p5 | median | p95 |
| --- | --- | --- | --- |
| Share total (masked current over Ip) | 0.9901 | 0.9973 | 0.9994 |
| Absolute current outside the boundary, share of the total | 0.0660 | 0.0846 | 0.1231 |
| The same without nodes within one cell of a coil | 0.0101 | 0.0126 | 0.0196 |
| Absolute share outside the placed reference lattice | 0.0000 | 0.0000 | 0.0000 |
| Relative error of the projected share (native grid) | 0.0174 | 0.0465 | 0.1548 |

## By shape and profile (basis_share unless stated)

Pooled R2 within a bin uses that bin's own spread, so a term whose quantity varies little inside the bin (kappa inside a kappa bin) reads lower than overall; the boundary distance in cm compares bins directly.

### topology

| Bin | Variant | n | S | r2_psi | 1 - D_LCFS | consistency | D_LCFS median (cm) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DN | basis_share | 164 | 0.9936 | 1.0000 | 0.9798 | 0.9781 | 0.71 |
| DN | exact_share | 164 | 0.9989 | 1.0000 | 0.9951 | 0.9968 | 0.24 |
| DN | basis_flux | 164 | 0.9955 | 1.0000 | 0.9872 | 0.9837 | 0.61 |
| LSN | basis_share | 165 | 0.9926 | 0.9999 | 0.9765 | 0.9750 | 0.83 |
| LSN | exact_share | 165 | 0.9992 | 1.0000 | 0.9958 | 0.9980 | 0.23 |
| LSN | basis_flux | 165 | 0.9961 | 1.0000 | 0.9847 | 0.9881 | 0.75 |

### kappa

| Bin | Variant | n | S | r2_psi | 1 - D_LCFS | consistency | D_LCFS median (cm) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.4-1.7 | basis_share | 4 | 0.9698 | 1.0000 | 0.9938 | 0.8523 | 0.28 |
| 1.4-1.7 | exact_share | 4 | 0.9973 | 1.0000 | 0.9974 | 0.9878 | 0.17 |
| 1.4-1.7 | basis_flux | 4 | 0.9881 | 1.0000 | 0.9913 | 0.9448 | 0.77 |
| 1.7-1.85 | basis_share | 53 | 0.9964 | 1.0000 | 0.9934 | 0.9851 | 0.39 |
| 1.7-1.85 | exact_share | 53 | 0.9978 | 1.0000 | 0.9958 | 0.9910 | 0.20 |
| 1.7-1.85 | basis_flux | 53 | 0.9950 | 1.0000 | 0.9914 | 0.9791 | 0.51 |
| 1.85-2 | basis_share | 97 | 0.9952 | 1.0000 | 0.9918 | 0.9802 | 0.56 |
| 1.85-2 | exact_share | 97 | 0.9981 | 1.0000 | 0.9960 | 0.9925 | 0.20 |
| 1.85-2 | basis_flux | 97 | 0.9952 | 1.0000 | 0.9917 | 0.9800 | 0.49 |
| 2-2.3 | basis_share | 163 | 0.9867 | 0.9999 | 0.9697 | 0.9490 | 1.54 |
| 2-2.3 | exact_share | 163 | 0.9987 | 1.0000 | 0.9950 | 0.9959 | 0.28 |
| 2-2.3 | basis_flux | 163 | 0.9942 | 1.0000 | 0.9825 | 0.9799 | 1.04 |
| >=2.3 | basis_share | 12 | 0.9452 | 0.9992 | 0.9089 | 0.7738 | 6.54 |
| >=2.3 | exact_share | 12 | 0.9994 | 1.0000 | 0.9963 | 0.9989 | 0.25 |
| >=2.3 | basis_flux | 12 | 0.9815 | 1.0000 | 0.9588 | 0.9284 | 3.15 |

### aspect_ratio

| Bin | Variant | n | S | r2_psi | 1 - D_LCFS | consistency | D_LCFS median (cm) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.2-1.45 | basis_share | 62 | 0.9925 | 1.0000 | 0.9849 | 0.9700 | 0.65 |
| 1.2-1.45 | exact_share | 62 | 0.9991 | 1.0000 | 0.9969 | 0.9969 | 0.20 |
| 1.2-1.45 | basis_flux | 62 | 0.9945 | 1.0000 | 0.9873 | 0.9788 | 0.55 |
| 1.45-1.6 | basis_share | 124 | 0.9931 | 1.0000 | 0.9830 | 0.9739 | 0.64 |
| 1.45-1.6 | exact_share | 124 | 0.9988 | 1.0000 | 0.9957 | 0.9962 | 0.19 |
| 1.45-1.6 | basis_flux | 124 | 0.9948 | 1.0000 | 0.9885 | 0.9798 | 0.56 |
| 1.6-1.75 | basis_share | 100 | 0.9904 | 0.9999 | 0.9745 | 0.9653 | 0.92 |
| 1.6-1.75 | exact_share | 100 | 0.9989 | 1.0000 | 0.9945 | 0.9971 | 0.26 |
| 1.6-1.75 | basis_flux | 100 | 0.9951 | 1.0000 | 0.9854 | 0.9830 | 0.76 |
| 1.75-2.1 | basis_share | 43 | 0.9861 | 0.9998 | 0.9626 | 0.9496 | 1.81 |
| 1.75-2.1 | exact_share | 43 | 0.9985 | 1.0000 | 0.9949 | 0.9954 | 0.39 |
| 1.75-2.1 | basis_flux | 43 | 0.9939 | 1.0000 | 0.9778 | 0.9808 | 1.55 |

### beta_p

| Bin | Variant | n | S | r2_psi | 1 - D_LCFS | consistency | D_LCFS median (cm) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0-0.2 | basis_share | 69 | 0.9936 | 0.9999 | 0.9828 | 0.9769 | 0.56 |
| 0-0.2 | exact_share | 69 | 0.9995 | 1.0000 | 0.9967 | 0.9991 | 0.20 |
| 0-0.2 | basis_flux | 69 | 0.9967 | 1.0000 | 0.9874 | 0.9897 | 0.52 |
| 0.2-0.35 | basis_share | 98 | 0.9934 | 1.0000 | 0.9798 | 0.9773 | 0.75 |
| 0.2-0.35 | exact_share | 98 | 0.9990 | 1.0000 | 0.9964 | 0.9968 | 0.26 |
| 0.2-0.35 | basis_flux | 98 | 0.9957 | 1.0000 | 0.9871 | 0.9852 | 0.73 |
| 0.35-0.5 | basis_share | 101 | 0.9930 | 0.9999 | 0.9757 | 0.9776 | 0.92 |
| 0.35-0.5 | exact_share | 101 | 0.9989 | 1.0000 | 0.9945 | 0.9974 | 0.24 |
| 0.35-0.5 | basis_flux | 101 | 0.9955 | 1.0000 | 0.9846 | 0.9850 | 0.73 |
| 0.5-0.8 | basis_share | 61 | 0.9918 | 1.0000 | 0.9741 | 0.9719 | 0.89 |
| 0.5-0.8 | exact_share | 61 | 0.9988 | 1.0000 | 0.9943 | 0.9968 | 0.22 |
| 0.5-0.8 | basis_flux | 61 | 0.9957 | 1.0000 | 0.9846 | 0.9863 | 0.73 |

### li

| Bin | Variant | n | S | r2_psi | 1 - D_LCFS | consistency | D_LCFS median (cm) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3-0.6 | basis_share | 54 | 0.9603 | 0.9998 | 0.9628 | 0.8208 | 1.85 |
| 0.3-0.6 | exact_share | 54 | 0.9963 | 1.0000 | 0.9912 | 0.9862 | 0.51 |
| 0.3-0.6 | basis_flux | 54 | 0.9894 | 1.0000 | 0.9792 | 0.9573 | 1.50 |
| 0.6-0.8 | basis_share | 141 | 0.9903 | 1.0000 | 0.9739 | 0.9645 | 0.96 |
| 0.6-0.8 | exact_share | 141 | 0.9989 | 1.0000 | 0.9950 | 0.9969 | 0.30 |
| 0.6-0.8 | basis_flux | 141 | 0.9951 | 1.0000 | 0.9837 | 0.9836 | 0.82 |
| 0.8-1 | basis_share | 108 | 0.9956 | 1.0000 | 0.9890 | 0.9834 | 0.42 |
| 0.8-1 | exact_share | 108 | 0.9994 | 1.0000 | 0.9975 | 0.9983 | 0.14 |
| 0.8-1 | basis_flux | 108 | 0.9959 | 1.0000 | 0.9906 | 0.9842 | 0.45 |
| 1-1.4 | basis_share | 26 | 0.9921 | 1.0000 | 0.9874 | 0.9671 | 0.54 |
| 1-1.4 | exact_share | 26 | 0.9997 | 1.0000 | 0.9985 | 0.9993 | 0.13 |
| 1-1.4 | basis_flux | 26 | 0.9917 | 1.0000 | 0.9923 | 0.9624 | 0.53 |

## Coil-model check (FreeGS vacuum flux against the coil model, same currents)

RMS of the difference over the frame's flux range. Near: nodes within one grid cell of a coil element (FreeGS has no self-distance floor there).

| Nodes | p5 | median | p95 | max |
| --- | --- | --- | --- | --- |
| all | 0.000040 | 0.000083 | 0.000101 | 0.000106 |
| away from coils | 0.000000 | 0.000000 | 0.000000 | 0.000001 |
| near coils | 0.000122 | 0.000255 | 0.000309 | 0.000323 |
| P2, away from coils | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| P3, away from coils | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| P4, away from coils | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| P5, away from coils | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| P6, away from coils | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| solenoid, away from coils | 0.000000 | 0.000000 | 0.000000 | 0.000001 |
| FreeGS record vs its own column patterns | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Coverage of the target ranges (achieved)

| Quantity | requested | p5 | median | p95 | min | max |
| --- | --- | --- | --- | --- | --- | --- |
| ip_ka | 400-900 | 427.406 | 638.662 | 869.215 | 401.608 | 898.365 |
| kappa | 1.6-2.2 | 1.755 | 2.013 | 2.267 | 1.675 | 2.574 |
| R_axis | 0.85-1.0 | 0.867 | 0.931 | 0.986 | 0.826 | 1.017 |
| beta_p | 0.1-0.6 | 0.126 | 0.345 | 0.572 | 0.100 | 0.599 |
| li | 0.5-1.0 | 0.549 | 0.763 | 1.030 | 0.503 | 1.116 |
| aspect_ratio | a 0.5-0.62 | 1.384 | 1.582 | 1.826 | 1.273 | 2.051 |
| tri_top |  | 0.155 | 0.356 | 0.484 | 0.066 | 0.587 |
| tri_bot |  | 0.170 | 0.334 | 0.460 | 0.055 | 0.509 |
| volume |  | 6.452 | 7.975 | 9.747 | 5.550 | 11.713 |
| q95 |  | 3.432 | 5.497 | 9.184 | 2.597 | 15.817 |
| beta_n |  | 0.306 | 0.906 | 2.159 | 0.135 | 2.956 |

Coil currents found (kA-turn; solenoid kA, fixed):

| Column | p5 | median | p95 |
| --- | --- | --- | --- |
| p2l | -107.1 | 198.7 | 366.9 |
| p2u | -103.0 | 155.4 | 366.9 |
| p3l | 0.0 | 0.0 | 0.0 |
| p3u | 0.0 | 0.0 | 0.0 |
| p4l | -247.8 | -111.3 | -40.2 |
| p4u | -245.7 | -98.0 | -22.2 |
| p5l | -157.4 | -88.0 | -9.0 |
| p5u | -177.2 | -103.2 | -31.4 |
| p6l | 0.0 | 0.0 | 0.0 |
| p6u | 0.0 | 0.0 | 0.0 |
| sol | -41.5 | -19.2 | 3.2 |

## Method notes and flags

1. Magnetic axis range. FreeGS cannot impose R_axis; the design samples the boundary centre (0.80 to 0.92 m, clipped so the inner edge stays at least 0.22 m and the outer midplane at most 1.48 m out) and the achieved R_axis is reported under Coverage against the requested 0.85 to 1.0 m.
2. Lower single null. A single tied lower X-point with the upper tip as an ordinary boundary point gave elongations near 1.5 and negative upper triangularity, or no convergence, within MAST's coil currents. LSN targets are therefore lower-dominant near-double-nulls: the upper X-point is constrained 2 to 10 cm above the mirror of the lower one and not tied to the boundary flux. Truly single-null shapes are not covered.
3. Coil currents. P3 and P6 are held at zero and P2, P4 and P5 are bounded to the range of the MAST test inputs; without the bounds the first double null used P4 at -970 kA-turn (test range -288 to -1) and P6 at +550. The set describes plasmas MAST's coils can make, not every shape in the requested ranges; achieved shapes are reported and binned.
4. Current profiles. Two-parameter FreeGS profiles (ConstrainBetapIp, alpha_m = 1, alpha_n 0.15 to 1.0, calibrated against the scorer's li, which is about 0.9 times FreeGS's li1 and none of FreeGS's li1, li2, li3; the scorer's docstring calls its li ITER li(2) while its formula normalises by the poloidal perimeter). No pedestal or edge current, no hollow ramp-up profile, no limited (inner-wall) plasma. Each of these would test the basis harder; a profile family through ProfilesPprimeFfprime with an edge term is the next step if the ceiling here is high.
5. No passive currents. The synthetic truth is coils plus plasma exactly, so the chain without a remainder is complete here; real MAST maps carry vessel currents (largest in the ramp-up) that the chain sets to zero. This ceiling is an upper bound in that respect and says nothing about the size of MAST's remainder.
6. Coil-model check. FreeGS's machine is built from the same geometry and the same convention (sol_half), so the check confirms the implementation (turns, sharing, signs, the solenoid factor applied once) and the discretisation; it cannot say whether half the feed current is the right physics.
7. Projection. basis_share fits the placed basis to the native share by least squares, the analogue of the DIII-D target. The model predicts coefficients from inputs, so a high ceiling here places the MAST shortfall of the submissions in the input-to-coefficient map (transfer), not in the basis.
8. Software. FreeGS 0.8.2 (PyPI) fails under NumPy 2.5 in find_critical; FreeGS is pinned to commit 9fbec20 (30 December 2025), which carries the upstream fix. FreeGS's own q fails under the same NumPy, so q95 comes from scalars.q95_from_map on the native map.

## Reading

Left for the review of the full run.
