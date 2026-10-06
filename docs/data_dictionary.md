# Data dictionary

Column dictionary for the Fusion Equilibrium Challenge Parquet files, built from the six
sample shots in the starter kit (`parquet_data/`: DIII-D 203702, 203703, 203704; MAST 28348,
28350, 28351) and cross-checked against the starter README sections "Complete Signal
Dictionary", "Machine geometry", "Data errata (v1.1.0)" and "Cross-machine convention notes".

Every statement about shapes, ranges and conventions below was measured on the sample files
with `scripts/summarize_shot.py` (raw files, no fixes applied) plus a one-off check script whose
output is kept under `internal/shot_summaries/`. Anything that can vary per shot has been seen
on three shots per machine only and must be re-verified by the QA pass on the full dataset.

Conventions used in this document:

- One Parquet file is one shot and holds exactly one row. Every cell is a nested array. `T` is
  the number of EFIT timeslices of the shot, `M` the number of magnetics samples.
- All time arrays are in milliseconds on both machines.
- Role: **input** (present in test files, may feed a model), **label** (EFIT-derived, withheld
  on test, target side only), **fixed** (machine description that does not change within a
  machine, present on every split).
- Shapes are given as measured, with the per-shot range where the three sample shots differ.

## 1. DIII-D columns (57)

Sample shots: 203702 (T = 166), 203703 (T = 166), 203704 (T = 168). All three have identical
column lists, dtypes, time-base lengths, Thomson layouts and coil geometry.

### 1.1 Time bases

| Column | Group | Role | Shape | dtype | Units | Sample interval (median) | Span on samples (ms) | Notes |
|---|---|---|---|---|---|---|---|---|
| `efit_times` | time base | input | `(T,)` | float64 | ms | 20 ms | 260 to 3700 | Prediction timestamps. Uniform 20 ms except for one gap of 120 to 160 ms near 3500 ms on all three sample shots. Identical to `magnetics_dsep_times`. |
| `magnetics_time` | time base | input | `(M,)`, M = 480256 | float32 | ms | 0.05 ms (20 kHz) | -4011 to 20002 | Shared by ECOILA, bcoil and all 18 F-coils. README: M varies by shot, six lengths occur, 2 kHz shots exist. Not seen in samples. |
| `magnetics_plasma_current_times` | time base | input | `(30719,)` | float32 | ms | 0.5 ms (2 kHz) | -858 to 14501 as shipped | Ip only. Shipped axis is wrong on about 69 percent of shots. See section 4.1. All three sample shots are affected. |
| `thomson_core_times` | time base | input | `(3029,)` | float64 | ms | 4.0 ms, irregular (230 to 247 distinct intervals between 1 and 10 ms) | -1497 to 12919 | Vertical core Thomson system. |
| `thomson_edge_times` | time base | input | `(1443,)` | float64 | ms | 10 ms | -1500 to 12920 | Tangential (edge) Thomson system. |
| `magnetics_dsep_times` | time base | label | `(T,)` | float32 | ms | 20 ms | 260 to 3700 | Byte-equal to `efit_times` on all samples (float32 copy). Withheld on test together with `magnetics_dsep`. |

### 1.2 Inputs

| Column | Group | Role | Shape | dtype | Units | Time base | Missing convention | Notes |
|---|---|---|---|---|---|---|---|---|
| `source` | metadata | input | scalar | string | | | | `"DIII-D"`. |
| `magnetics_F1A` to `magnetics_F9B` (18 columns) | coil current | input | `(M,)` | float64 | kA·turn, see D1 | `magnetics_time` | exact 0 outside the shot | Shaping coils, A upper, B lower. Sample range -403 to +209. Exact zeros are 1 to 7 percent of samples, almost all before t = 0, none inside the EFIT window. One geometry rectangle each. |
| `magnetics_ECOILA` | coil current | input | `(M,)` | float64 | kA, see D1 | `magnetics_time` | exact 0 outside the shot | Ohmic coil (central solenoid). Sample range -37 to +42. Turn convention unresolved (README). One geometry rectangle, R = 0.7225 m, Z = 0, width 0.129 m, height 3.506 m. |
| `magnetics_bcoil` | coil current | input | `(M,)` | float64 | kA, see D1 | `magnetics_time` | exact 0 outside the shot | Toroidal field coil. Sample range -118 to +0.8. No geometry row (does not contribute to poloidal flux). |
| `magnetics_plasma_current` | plasma current | input | `(30719,)` | float32 | kA, see D1 | `magnetics_plasma_current_times` after fix | exact 0 late in the record | Ip. Sample peak 1014 to 1383. 177 to 296 exact zeros per shot, all after the plasma. |
| `thomson_core_Te` | Thomson | input | `(3029, 42)` | float64 | eV | `thomson_core_times` | 0, never NaN | 42 channels on all samples (README: 40, 42, 43, 44 or 54 by shot). 28 to 30 percent of time slices are all-zero; inside the remaining slices 64 to 66 percent of values are zero. No negative values. |
| `thomson_core_ne` | Thomson | input | `(3029, 42)` | float64 | m^-3 | `thomson_core_times` | 0, never NaN | Same zero pattern as Te. Sample max 4.2e19. |
| `thomson_core_R` | Thomson | fixed per shot | `(42,)` | float64 | m | none | | Constant 1.94 on every channel (vertical laser). The informative coordinate is Z, available only through `thomson_chord_Z` for the `TS_core_*` chords. |
| `thomson_edge_Te` | Thomson | input | `(1443, 14)` | float64 | eV | `thomson_edge_times` | 0, never NaN | 14 channels on all samples (README: 10 or 14 by shot). 55 to 62 percent of time slices all-zero; inside the rest 26 to 32 percent zeros. |
| `thomson_edge_ne` | Thomson | input | `(1443, 14)` | float64 | m^-3 | `thomson_edge_times` | 0, never NaN | Same zero pattern as Te. |
| `thomson_edge_spatial` | Thomson | fixed per shot | `(14,)` | float64 | m | none | | Z positions of the tangential channels, -0.089 to -0.041. Equal to `thomson_chord_Z` of the `TS_tangential_*` chords. On MAST the same column holds R, see section 3. |
| `efit_grid_R` | grid | input | `(65,)` | float64 | m | none | | 0.84 to 2.54, uniform step 0.0266. Same on every shot. |
| `efit_grid_Z` | grid | input | `(65,)` | float64 | m | none | | -1.6 to 1.6, uniform step 0.05. Contains one exact zero (Z = 0 grid line). |
| `coil_name` | geometry | fixed | `(19,)` | string | | none | | F1A to F9B and ECOILA. |
| `coil_input_column` | geometry | fixed | `(19,)` | string | | none | | Join key to the current columns. One row per column. `magnetics_bcoil` and `magnetics_plasma_current` have no row. |
| `coil_R`, `coil_Z` | geometry | fixed | `(19,)` | float64 | m | none | | Rectangle centres. R 0.72 to 2.61, Z -1.60 to 1.60. |
| `coil_width`, `coil_height` | geometry | fixed | `(19,)` | float64 | m | none | | Width 0.051 to 0.235, height 0.085 to 3.506 (ECOILA). |
| `coil_angle1`, `coil_angle2` | geometry | fixed | `(19,)` | float64 | degrees | none | | Parallelogram skew. Non-zero only for F5A/F5B (angle1 = plus or minus 45), F6A/F6B (angle2 = plus or minus 92.4), F7A/F7B (angle2 = plus or minus 108.06). 0 means an axis-aligned rectangle. |
| `thomson_chord_name` | geometry | fixed per shot | `(70,)` | string | | none | | 42 `TS_core_*`, 14 `TS_tangential_*`, 14 `TS_divertor_*` on all samples. README: layouts vary per shot (59 to 138 chords, 22 layouts in train). |
| `thomson_chord_R`, `thomson_chord_Z` | geometry | fixed per shot | `(70,)` | float64 | m | none | | Core chords: R = 1.94, Z 0.026 to 0.825. Tangential chords: R 1.673 to 2.265, Z -0.089 to -0.041. Divertor chords reach Z = -1.243. |

### 1.3 Labels

All label columns are on `efit_times`. They are absent from the DIII-D public test files
(assumed from the README; to be verified when the test files are downloaded).

| Column | Group | Role | Shape | dtype | Units | Missing convention | Notes |
|---|---|---|---|---|---|---|---|
| `efit_psirz` | label | label (primary target) | `(T, 65, 65)` | float64 | V·s/rad | none, no NaN | Index order is `[t, iZ, iR]` (row = Z, column = R), verified against the EFIT axis on every frame and against the scorer, which indexes `phi[iz, ir]`. Magnetic axis is the **minimum** of ψ on DIII-D. Sample range -0.33 to +0.82. |
| `efit_q95` | label | label (scored scalar) | `(T,)` | float64 | | | Sample range 3.6 to 9.8. |
| `efit_beta_n` | label | label (scored scalar) | `(T,)` | float64 | | | Sample range 0.085 to 0.58. |
| `efit_li` | label | label (auxiliary, in Consistency) | `(T,)` | float64 | | | Sample range 0.80 to 1.76. |
| `efit_r_axis`, `efit_z_axis` | label | label (auxiliary, in Consistency) | `(T,)` | float64 | m | | R 1.66 to 1.71, Z -0.058 to 0.043. |
| `efit_lcfs_n` | label | label | `(T,)` | float64 | count | | Valid point count of the boundary contour, 86 to 90 on samples. |
| `efit_lcfs_r`, `efit_lcfs_z` | label | label | `(T, 90)` | float64 | m | zero-padded beyond `efit_lcfs_n` | Padded width 90 on all samples. Closed contour (first point equals last point). The valid region contains real Z = 0 points (about 2 per frame), so `efit_lcfs_n` is the only safe way to find the padding. |
| `magnetics_dsep` | label | label (not scored) | `(T,)` | float32 | m | none on samples | EFIT a-file DSEP, separatrix to limiter clearance, positive means diverted. Sample range 0.001 to 0.050 (all diverted). A label despite the `magnetics_` prefix. |

## 2. MAST columns (55)

Sample shots: 28348 (T = 36), 28350 (T = 42), 28351 (T = 37). All three have the 30000-sample
magnetics time base; the early-campaign 15482-sample union grid described in the README (114 of
1206 test shots) is not represented in the samples. Test files carry inputs only, so the EFIT
columns below exist only in the three demo shots.

### 2.1 Time bases

| Column | Group | Role | Shape | dtype | Units | Sample interval (median) | Span on samples (ms) | Notes |
|---|---|---|---|---|---|---|---|---|
| `efit_times` | time base | input | `(T,)` | float32 | ms | 5 ms | 65 to 270 | Uniform 5 ms on 28350 and 28351; one 10 ms gap on 28348. README says float64 (D4). |
| `magnetics_time` | time base | input | `(30000,)` | float32 | ms | 0.2 ms (5 kHz) | -2000 to 4000 | Shared by every MAST magnetics column including Ip. README says float64 (D4). Second population of length 15482 with per-column NaN holes exists in the test set (README). |
| `thomson_core_times` | time base | input | `(64,)` | float64 | ms | 2 ms within bursts, 19.33 ms between bursts | 1.7 to 264 | Burst-mode laser: short trains of 2 ms pulses separated by 19.33 ms. The median interval understates the mean spacing (about 4 ms). |
| `thomson_edge_times` | time base | input | `(63,)` | float64 | ms | same burst pattern | 1.7 to 262 | One sample fewer than the core system on every sample shot. |
| `magnetics_dsep_times` | time base | label | `(T,)` | float32 | ms | 5 ms | 65 to 270 | Byte-equal to `efit_times`. Demo shots only. |

### 2.2 Inputs

| Column | Group | Role | Shape | dtype | Units | Time base | Missing convention | Notes |
|---|---|---|---|---|---|---|---|---|
| `source` | metadata | input | scalar | string | | | | `"MAST"`. |
| `magnetics_p2l_current` to `magnetics_p6u_current` (10 columns) | coil current | input | `(30000,)` | float64 | kA·turn | `magnetics_time` | none on samples (NaN holes on the 15482-sample population) | P2 to P6, l lower, u upper. Sample range -165 to +246. No exact zeros. Turns per coil from the geometry row count: P2 = 20, P3 = 8, P4 = 23, P5 = 23, P6 = 4. |
| `magnetics_sol_current` | coil current | input | `(30000,)` | float64 | kA | `magnetics_time` | as above | Central solenoid. Sample range -42 to +46. 656 geometry elements share this column. |
| `magnetics_tf_current` | coil current | input | `(30000,)` | float64 | kA | `magnetics_time` | as above | Toroidal field. Sample range -85 to 0.05. No geometry row. |
| `magnetics_efps_current` | coil current | input | `(30000,)` | float64 | kA | `magnetics_time` | as above | Error field protection system. Sample range -0.08 to 12.2. No geometry row, no DIII-D counterpart. |
| `magnetics_plasma_current` | plasma current | input | `(30000,)` | float64 | kA | `magnetics_time` | none | Ip. Sample peak 785 to 801. No time-axis fix needed. |
| `thomson_core_Te` | Thomson | input | `(64, 131)` | float64 | eV | `thomson_core_times` | NaN | Raw data channel 0 is NaN on every time slice and has no coordinate. See section 4.2. 9 to 12 percent NaN overall, scattered per point (59 to 125 of 131 channels have at least one NaN). No all-NaN time slices. No zeros. |
| `thomson_core_ne` | Thomson | input | `(64, 131)` | float64 | m^-3 | `thomson_core_times` | NaN | Same NaN pattern as Te. Sample max 2.3e20. |
| `thomson_core_R` | Thomson | fixed per shot | `(130,)` | float64 | m | none | | Real radii 0.234 to 1.517, monotone increasing (horizontal midplane laser). 130 entries versus 131 data channels. Equal to `thomson_chord_R` of the `TS_core_*` chords. |
| `thomson_edge_Te` | Thomson | input | `(63, 16)` | float64 | eV | `thomson_edge_times` | NaN | 16 channels. 5 to 8 percent NaN. |
| `thomson_edge_ne` | Thomson | input | `(63, 16)` | float64 | m^-3 | `thomson_edge_times` | NaN | Same pattern as Te. |
| `thomson_edge_spatial` | Thomson | fixed per shot | `(16,)` | float64 | m | none | | R positions 1.328 to 1.485 (outboard edge). On DIII-D the same column holds Z. |
| `efit_grid_R` | grid | input | `(65,)` | float64 | m | none | | 0.06 to 2.00, uniform step 0.0303. |
| `efit_grid_Z` | grid | input | `(65,)` | float64 | m | none | | -2.0 to 2.0, uniform step 0.0625. Contains one exact zero. |
| `coil_name` | geometry | fixed | `(812,)` | string | | none | | 812 distinct element names: `sol_*` (656), `p2_inner_lower_*` (12), `p2_outer_lower_*` (8), same for upper, `p3_lower_*` (8), `p4_*` (23), `p5_*` (23), `p6_*` (4) per side. |
| `coil_input_column` | geometry | fixed | `(812,)` | string | | none | | 11 distinct values. Row counts per column: p2l/p2u 20, p3l/p3u 8, p4l/p4u 23, p5l/p5u 23, p6l/p6u 4, sol 656. `magnetics_tf_current`, `magnetics_efps_current`, `magnetics_plasma_current` have no rows. |
| `coil_R`, `coil_Z` | geometry | fixed | `(812,)` | float64 | m | none | | R 0.119 to 1.715, Z -1.844 to 1.849. |
| `coil_width`, `coil_height` | geometry | fixed | `(812,)` | float64 | m | none | | Width 0.0115 to 0.024, height 0.017 to 0.052 (single conductor turns). |
| `coil_angle1`, `coil_angle2` | geometry | fixed | `(812,)` | float64 | degrees | none | | All exactly 0 (axis-aligned rectangles). |
| `thomson_chord_name` | geometry | fixed per shot | `(146,)` | string | | none | | 130 `TS_core_*` plus 16 `TS_edge_*` on all samples. No divertor subsystem. |
| `thomson_chord_R`, `thomson_chord_Z` | geometry | fixed per shot | `(146,)` | float64 | m | none | | R equals `thomson_core_R` and `thomson_edge_spatial` respectively. Z is exactly 0 on every chord (midplane laser), which is a real coordinate, not a placeholder. |

### 2.3 Labels (demo shots only)

| Column | Group | Role | Shape | dtype | Units | Missing convention | Notes |
|---|---|---|---|---|---|---|---|
| `efit_psirz` | label | label (primary target) | `(T, 65, 65)` | float64 | V·s/rad | none | Index order `[t, iZ, iR]`, verified on every frame. Magnetic axis is the **maximum** of ψ on MAST (opposite to DIII-D). Sample range -0.20 to +0.065. `efit_psi_axis` equals the frame maximum to within 1e-4. |
| `efit_q95` | label | label (scored scalar) | `(T,)` | float64 | | | 5.6 to 11.9. |
| `efit_beta_n` | label | label (scored scalar) | `(T,)` | float64 | | | 0.50 to 1.34. Higher than DIII-D samples (0.09 to 0.58). |
| `efit_li` | label | label (auxiliary) | `(T,)` | float64 | | | 0.61 to 1.67. |
| `efit_r_axis`, `efit_z_axis` | label | label (auxiliary) | `(T,)` | float64 | m | | R 0.79 to 0.92, Z 0.003 to 0.012. |
| `efit_beta_p` | label | label (MAST demo only) | `(T,)` | float64 | | | Poloidal beta, 0.14 to 0.29. No DIII-D counterpart. |
| `efit_elongation` | label | label (MAST demo only) | `(T,)` | float64 | | | 1.67 to 2.16. No DIII-D counterpart, but kappa is one of the seven Consistency scalars the scorer derives. |
| `efit_psi_axis`, `efit_psi_boundary` | label | label (MAST demo only) | `(T,)` | float64 | V·s/rad | | Axis 0.045 to 0.065, boundary -0.045 to 0.030. No DIII-D counterpart. |
| `efit_geom_axis_r`, `efit_geom_axis_z` | label | label (MAST demo only) | `(T,)` | float64 | m | | Geometric axis, R 0.716 to 0.833, Z 0.003 to 0.069. No DIII-D counterpart. |
| `efit_lcfs_n` | label | label | `(T,)` | float64 | count | | 125 to 155 on samples. |
| `efit_lcfs_r`, `efit_lcfs_z` | label | label | `(T, 155)` | float64 | m | NaN-padded beyond `efit_lcfs_n` | Padded width 155 on all samples. Closed contour, starting at the inboard midplane (R = 0.196, Z = 0). The valid region contains real Z = 0 points (about 3 per frame). |
| `magnetics_dsep` | label | label (not scored) | `(T,)` | float32 | m | -1 sentinel and NaN | δR_sep (upper minus lower separatrix radial gap at the outboard midplane), a different quantity from the DIII-D column. Sample range -0.0004 to 0.006 excluding 6 to 7 frames per shot that hold exactly -1, plus one NaN on 28350. The -1 values are undocumented (D7). |

## 3. Cross-machine mapping

| Physical concept | DIII-D column(s) | MAST column(s) | Unit or convention difference |
|---|---|---|---|
| Machine identifier | `source` = `"DIII-D"` | `source` = `"MAST"` | |
| Prediction time base | `efit_times`, float64, 20 ms cadence, 166 to 168 frames on samples | `efit_times`, float32, 5 ms cadence, 36 to 42 frames on samples | Cadence and dtype differ. README: T ranges 1 to 445 (DIII-D) and 5 to 98 (MAST). |
| Flux map | `efit_psirz` `[t, iZ, iR]`, axis at the minimum | `efit_psirz` `[t, iZ, iR]`, axis at the maximum | Opposite sign convention. The scorer picks one global sign per machine. Amplitude is not normalized: DIII-D spans about 1.1 V·s/rad, MAST about 0.26. |
| Flux grid | `efit_grid_R` 0.84 to 2.54 m, `efit_grid_Z` -1.6 to 1.6 m | `efit_grid_R` 0.06 to 2.00 m, `efit_grid_Z` -2.0 to 2.0 m | Different extents and steps (0.0266 by 0.05 m versus 0.0303 by 0.0625 m). Both uniform, both 65 by 65. |
| Plasma current | `magnetics_plasma_current`, float32, kA (see D1), own time base that needs `fix_d3d_ip_times` | `magnetics_plasma_current`, float64, kA, on `magnetics_time` | DIII-D peaks 1.0 to 1.4 MA on samples, MAST 0.8 MA. |
| Central solenoid | `magnetics_ECOILA`, kA (see D1), one lumped rectangle, turn count ambiguous | `magnetics_sol_current`, kA, 656 single-turn elements | Turn handling differs: MAST turns come from element count, DIII-D ECOILA scale is unresolved. |
| Toroidal field | `magnetics_bcoil` | `magnetics_tf_current` | No geometry on either machine. Does not contribute to poloidal flux. |
| Shaping (poloidal field) coils | `magnetics_F{1-9}{A,B}`, 18 columns, kA·turn (see D1), one rectangle each, six are parallelograms | `magnetics_p{2-6}{l,u}_current`, 10 columns, kA·turn, 4 to 23 elements each, all rectangles | Same quantity (ampere-turns) once D1 is settled. P2 has inner and outer sub-coils fed in parallel under one column. |
| Error field coil | none | `magnetics_efps_current` | MAST only, no geometry. |
| Magnetics time base | `magnetics_time`, float32, 0.05 ms (20 kHz), -4.0 to 20.0 s, M varies by shot | `magnetics_time`, float32, 0.2 ms (5 kHz), -2.0 to 4.0 s, two populations | DIII-D Ip is on a separate axis. MAST early-campaign shots have per-column NaN holes. |
| Thomson core Te, ne | `(T_c, C_c)`, C_c = 42 on samples, 0 means missing, vertical laser at R = 1.94 m | `(T_c, 131)`, NaN means missing, drop data channel 0, horizontal midplane laser | Orientation of the chord and missing convention both differ. Row order is (time, channel) on both. |
| Thomson core coordinate | `thomson_core_R` constant 1.94; Z from `thomson_chord_Z` of `TS_core_*` | `thomson_core_R` real radii 0.23 to 1.52 m (130 values) | The shipped coordinate means different things. Use `thomson_chord_R/Z` for a uniform (R, Z) per channel on both machines. |
| Thomson edge Te, ne | `(T_e, C_e)`, C_e = 14 on samples, 0 means missing | `(T_e, 16)`, NaN means missing | |
| Thomson edge coordinate | `thomson_edge_spatial` = Z (-0.09 to -0.04 m) | `thomson_edge_spatial` = R (1.33 to 1.49 m) | Same column name, different axis. |
| Thomson timing | core 4 ms irregular, edge 10 ms, both cover -1.5 to 12.9 s | 2 ms bursts every 19.33 ms, cover 0 to 0.26 s | DIII-D has many more slices than EFIT frames (3029 versus 166); MAST has about 1.7 per EFIT frame. |
| Thomson chord geometry | `thomson_chord_*`, 70 on samples: core, tangential, divertor | `thomson_chord_*`, 146: core, edge; Z = 0 | Name prefixes differ (`TS_tangential_*` versus `TS_edge_*`). |
| Coil geometry | 19 rows, one per current column | 812 rows, many per column | Turn count folded into currents on both. |
| Separatrix gap | `magnetics_dsep`, separatrix to limiter clearance, positive means diverted | `magnetics_dsep`, upper minus lower separatrix radial gap, straddles zero, -1 sentinel | Different physical quantity under the same name. Both are labels. |
| Boundary contour | `efit_lcfs_r/z` `(T, 90)`, zero-padded | `efit_lcfs_r/z` `(T, 155)`, NaN-padded | Use `efit_lcfs_n` on both. |
| Scored scalars | `efit_q95`, `efit_beta_n` | same | |
| Auxiliary scalars | `efit_li`, `efit_r_axis`, `efit_z_axis` | same, plus `efit_beta_p`, `efit_elongation`, `efit_psi_axis`, `efit_psi_boundary`, `efit_geom_axis_r/z` | The six extra MAST columns are demo-only and have no DIII-D training counterpart. |

## 4. Known errata and traps

### 4.1 DIII-D Ip time axis (`magnetics_plasma_current_times`)

Every DIII-D file ships the same Ip time axis (-858.19 to 14500.81 ms, 0.5 ms step). On about
69 percent of shots the Ip trace actually starts at `magnetics_time[0]`. The starter helper
`data_fixes.fix_d3d_ip_times(row)` decides per shot by locating plasma breakdown (first sample
where |Ip| exceeds 10 percent of its 99.5th percentile) and keeping the shipped axis only if
breakdown falls in [-500, 1000] ms. Never blanket-shift.

Result on the three sample shots (helper imported read-only from the starter kit):

| Shot | Affected | Shift applied (ms) | Breakdown, shipped axis (ms) | Breakdown, fixed axis (ms) | Median abs Ip on `efit_times`, shipped | Median abs Ip, fixed |
|---|---|---|---|---|---|---|
| 203702 | yes | -3152.57 | 3165.8 | 13.2 | 1.9 | 1357.2 |
| 203703 | yes | -3152.57 | 3165.8 | 13.2 | 1.5 | 1357.2 |
| 203704 | yes | -3152.57 | 3165.8 | 13.2 | 1.9 | 993.9 |

All three samples are in the affected group, so the samples give no example of an unaffected
shot. The shift equals `magnetics_time[0] - ipt[0]` = -4010.75 - (-858.19). The corrected axis
then spans -4011 to 11348 ms and still covers the full EFIT window (260 to 3700 ms).

Full-dataset QA (October 4, 2026, 7,915 DIII-D shots): the shots the helper leaves untouched
are the 2 kHz `magnetics_time` population (about 2,400 shots, 0.5 ms step; the remaining shots
record at 20 kHz, 0.05 ms step). After the helper, Ip breakdown sits at a median of 14.9 ms on
the 20 kHz shots and around 190 ms on the 2 kHz shots, so the two populations disagree by an
axis offset, not physics. One 2 kHz shot (21f23b2392) failed check 0.2 with the plasma current
reaching the EFIT frames at under 10 percent of its peak.

Project rule (applied by `fusion_eq.io.load_shot` after the helper, `fusion_eq/ip_alignment.py`):
for a DIII-D shot the helper leaves untouched, shift the Ip time axis so that breakdown, under
the helper's own definition (first sample with |Ip| above 10 percent of the shot's 99.5th
percentile), falls at 14.9 ms, the 20 kHz median. The target is a hard-coded constant. The shift
and the breakdown before and after are recorded in `Shot.fixes`; `align_ip=False` disables the
rule. A diagnostic anchor on the central-solenoid swing-down (the time `magnetics_ECOILA`
leaves its pre-charge plateau, near 0 ms; on the sample shots 0.6 to 1.2 ms with a lag of 9 to
13 ms to breakdown) is reported by QA check 0.2 against the breakdown rule. The reference
anchor is 1.5 ms, the 20 kHz median from the full run.

### 4.2 MAST Thomson core channel offset

`thomson_core_R` has 130 entries, `thomson_core_Te` and `thomson_core_ne` have 131 channels.
Raw data channel 0 is NaN on every time slice of every sample shot (Te and ne), and it is the only
channel that is fully NaN. `data_fixes.align_mast_thomson_core(row)` returns `(R, Te, ne)` with
shapes `(130,)`, `(T_c, 130)`, `(T_c, 130)` on all three sample shots, and R is monotone
increasing. After the fix, `Te[:, j]` sits at `R[j]`, which also equals `thomson_chord_R` of chord
`TS_core_j`. The helper passes DIII-D rows through unchanged.

### 4.3 `magnetics_dsep` is a label

Despite the `magnetics_` prefix, `magnetics_dsep` and `magnetics_dsep_times` are EFIT-derived
and withheld on test. `magnetics_dsep_times` is byte-equal to `efit_times` on all six samples.
The column must never be used as an input, and on MAST it is a different quantity (section 3).

### 4.4 Zero versus NaN missing conventions

- DIII-D Thomson uses exact 0 for missing values and contains no NaN. Both the core and the
  edge system have whole time slices of zeros (28 to 30 percent of core slices, 55 to 62 percent
  of edge slices) and scattered per-channel zeros inside the remaining slices. Zero is a
  physically impossible temperature or density, so the data-prep step converts exact zeros to
  NaN before any interpolation onto `efit_times`.
- DIII-D magnetics columns contain exact zeros (1 to 7 percent of samples) but only outside the
  plasma window (before t = 0 or after the shot), so no conversion is needed there. Ip has a
  few hundred exact zeros after the plasma ends.
- MAST Thomson uses NaN for missing values and contains no zeros. NaN is scattered per point,
  not per slice: 59 to 125 of the 131 core channels have at least one NaN, and no time slice is
  fully NaN on the samples.
- MAST magnetics on the samples are complete. The early-campaign population has NaN holes on
  the union time grid; mask per column (README).
- MAST `magnetics_dsep` uses -1 as a sentinel (6 to 7 frames per shot) and contains one NaN.

### 4.5 Boundary contour padding

| Machine | Padded width | `efit_lcfs_n` range on samples | Frames with n equal to width | Padding value | Real zeros inside the valid region |
|---|---|---|---|---|---|
| DIII-D | 90 | 86 to 90 | 1 to 2 per shot | 0 (r and z) | none in r; about 2 per frame in z |
| MAST | 155 | 125 to 155 | 1 to 2 per shot | NaN (r and z) | none in r; about 3 per frame in z |

The padded width is the maximum `efit_lcfs_n` of the shot, so it varies per shot. Because valid
DIII-D contours contain genuine Z = 0 coordinates, testing for zeros would drop real points; use
`efit_lcfs_n` on both machines. Both machines store a closed contour whose first and last points
coincide.

### 4.6 Per-shot Thomson layouts on DIII-D

The three DIII-D samples share one layout (42 core, 14 tangential, 14 divertor chords; 42-column
`thomson_core_R`; 14-entry `thomson_edge_spatial`). The README states that the channel counts
(40 to 54 core, 10 or 14 edge) and the chord layout (22 layouts in train, 59 to 138 chords) vary
per shot. Never cache one shot's layout. Any feature that needs a fixed-length Thomson vector must
interpolate profiles onto a fixed set of positions per shot, using `thomson_chord_R/Z`.

### 4.7 Other traps seen on the samples

- `efit_psirz` is indexed `[t, iZ, iR]`. Transposing it silently breaks the O-point and every
  derived Consistency scalar.
- DIII-D `efit_times` is not perfectly uniform: all three samples have one 120 to 160 ms gap
  near 3500 ms. Frame-count arithmetic must use the array, not the cadence.
- MAST Thomson runs in bursts (2 ms pulses, 19.33 ms gaps), so nearest-neighbour matching to a
  5 ms EFIT frame can land up to about 10 ms away.
- `thomson_edge_spatial` is a Z coordinate on DIII-D and an R coordinate on MAST.
- `magnetics_time` on both machines spans far beyond the plasma window. Interpolation onto
  `efit_times` never extrapolates on the samples, but pre-shot zeros (DIII-D) sit inside the array.
- MAST `efit_times` and `magnetics_time` are float32. Casting to float64 before any arithmetic
  avoids precision surprises when comparing time axes.

## 5. Where the README and the data disagree

Contradictions are recorded with evidence and left unresolved.

| ID | README statement | Observed on the samples | Evidence |
|---|---|---|---|
| D1 | DIII-D current units. The signal dictionary says F-coils are kA·turn, ECOILA, bcoil and Ip are kA. The cross-machine units table says all DIII-D `magnetics_F*`, `ECOILA`, `bcoil`, `plasma_current` are **A** ("already A"). The F-coil section says "about plus or minus 600 kA·turn in-window". | Sample magnitudes: Ip peak 1014 to 1383; F-coils up to 403 (F7B); ECOILA plus or minus 42; bcoil -118. | The errata text describes the Ip noise floor as "about 2 kA" and the real current as "about 1 MA"; the helper demo prints the shipped-axis median as 1.9 and the fixed median as 1357, which only reads as kA. F-coil values of hundreds of amperes would be physically implausible for a shaping coil, and the README's own range quotes kA·turn. The working assumption in this project is kA and kA·turn, but the README's "A" claim is not explained and must be checked against the Green's-function fit in Phase C. |
| D2 | "Key Differences" table: DIII-D R and Z coordinates are "Normalized 0-1". | `efit_grid_R` 0.84 to 2.54 m, `efit_grid_Z` -1.6 to 1.6 m, physical metres on all three shots. | Same table gives MAST R as 0.12 to 2.0 m while the grid section says 0.06 to 2.0; the data shows 0.06 to 2.00. |
| D3 | "Key Differences" table: DIII-D Thomson data orientation is (spatial, time), MAST is (time, spatial). | Both machines store `(time, channel)`: DIII-D core `(3029, 42)`, MAST core `(64, 131)`. | Outer length matches `thomson_*_times` on every sample. |
| D4 | MAST `magnetics_time` is float64; MAST `efit_times` is float64. | Both are float32 in the files. The ten P-coil, solenoid, TF, EFPS and Ip columns are float64. | Parquet schema. DIII-D dtypes match the README (`magnetics_time` float32, currents float64, Ip and its times float32, `efit_times` float64). |
| D5 | DIII-D `thomson_core_times` has about 1300 to 1900 samples; `thomson_edge_times` about 200 to 500. | 3029 core and 1443 edge samples on all three shots. | Spans -1.5 to 12.9 s at 4 ms and 10 ms. |
| D6 | `magnetics_dsep` on MAST "straddles zero on ordinary diverted plasmas"; no sentinel is mentioned. | 6 to 7 frames per shot hold exactly -1, and 28350 has one NaN. | The -1 value is orders of magnitude outside the rest of the range (-0.0004 to 0.006 m) and appears to be a fill value. |
| D7 | The signal dictionary lists five DIII-D EFIT scalars and the MAST table lists none beyond `efit_psirz`. | The MAST demo shots carry six additional EFIT columns (`efit_beta_p`, `efit_elongation`, `efit_psi_axis`, `efit_psi_boundary`, `efit_geom_axis_r`, `efit_geom_axis_z`) that DIII-D lacks. | Parquet schema: 55 MAST columns, 57 DIII-D columns. |
| D8 | Cross-machine note: "no parquet nulls anywhere in this release", gaps are plain NaN. | Consistent on the samples: no null cells, all missing values are NaN or 0. | No disagreement, recorded for completeness. |
| D9 | DIII-D `thomson_core_R` "varies by shot: 40, 42, 43, 44 or 54"; edge 10 or 14. | 42 and 14 on all three samples. | Consistent, but the samples cover a single layout only. |

## 6. Open questions

1. DIII-D current units (D1). Decide with the Green's-function fit in Phase C: the fitted scale
   between computed coil flux and ψ outside the plasma should be 1 under the right unit.
2. ECOILA turn convention. README states the EFIT model uses 48 single-turn elements over the
   same envelope and a second unshipped group ECOILB. Treat its scale as one number to fit.
3. Resolved in part by the QA run: the unaffected 31 percent of DIII-D shots are the 2 kHz
   magnetics population and their Ip axis is late by about 175 ms against the 20 kHz shots.
   The project aligns them to a breakdown at 14.9 ms (section 4.1). Still open: whether the
   solenoid swing-down anchor agrees with that rule across the full set, and the per-shot
   spread of breakdown within the 20 kHz group.
4. Meaning of the -1 sentinel and the NaN in MAST `magnetics_dsep` (D6). Only relevant for
   analysis, since the column is a label.
5. Whether the DIII-D all-zero Thomson time slices are absent laser pulses (regular pattern) or
   dropped data, and how many EFIT frames have a valid Thomson slice within a chosen tolerance.
6. The 120 to 160 ms gap in DIII-D `efit_times` near 3500 ms on all three samples. Whether it is
   common, and whether any shots have gaps large enough to matter for input resampling.
7. The MAST early-campaign population (15482-sample union time grid with per-column NaN holes)
   is not in the samples. The loader must be tested on it once the test files are downloaded.
8. The exact column list of the released test files for both machines, to confirm which EFIT
   columns are dropped and that `magnetics_dsep*` is absent.
9. Variation of DIII-D magnetics length M and rate (20 kHz versus 2 kHz shots) and of Thomson
   chord layouts across the full dataset, both stated in the README and not seen in the samples.
10. Which MAST scalar conventions (for example `efit_elongation` versus the scorer's kappa, and
    `efit_li` normalization) match the seven quantities the scorer derives, so that the demo
    shots can be used as a sanity check of the scorer on MAST.
