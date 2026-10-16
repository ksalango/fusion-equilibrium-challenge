# Feature specification: transferable inputs and their analysis

Specification for the feature phase. The repository's data rules and writing standard
apply. Finding ids (F01 to F79) refer to `docs/findings.md`. Check ids in this document are
X1 to X16.

## Purpose

Build the inputs of the plasma model in forms that mean the same on DIII-D and MAST, from
the frame table, driven by a configuration file so that a feature is added by writing one
function and one configuration entry. Then compare the features on the two machines.

The phase settles five things:

1. Whether the features can be built on every frame of both machines (completeness).
2. Whether MAST lies inside the DIII-D range, feature by feature and in combination, and
   where the model will extrapolate.
3. Which features, if any, differ between machines for an instrument reason.
4. Whether time features scaled by the plasma's own timescale are comparable across
   machines (F33).
5. Whether the features carry the information a linear model takes from the raw inputs
   (linear check against the hybrid ridge).

Out of scope: the plasma model, the loss, the target basis, Thomson profile descriptors on
a normalized radius (they need the raw channel positions and a flux map), and any use of
MAST equilibrium labels.

## Decisions this phase implements

Taken on 11 and 12 October 2026 and recorded under "Carried into modelling".

- Inputs: four groups (coil pattern, plasma content, field against current, position in
  the shot). No raw coil current, no Ip in kA, no raw Thomson unit, no absolute time.
- Scaling: each feature is made unitless by physics first. A robust scaler (median and
  interquartile range) is then fitted on DIII-D training frames, originals and mirrors
  together, and applied unchanged to MAST. Features that form one pattern share one scale.
  A feature may be scaled with the target machine's own input statistics only where the
  evidence points to an instrument difference; no feature does so in version 1.
- Mirroring: up-down mirroring is applied to the feature matrix, not to the coil currents.
  Because the coil pattern is itself the input, the mirror of a frame is the pattern
  flipped in Z together with the flipped target, and this is exact. Swapping A and B coil
  currents is not needed and is not used, since the shipped coils are not exact mirrors
  (F51). The entry under "Carried into modelling" that describes mirroring as a current
  swap with recalculated coil flux is corrected as part of this phase.

## Context

- Input: the frame table (`read_frame_table`), with the additions of Part 0. Nothing in
  this phase reads a shot file directly.
- Coil flux: the existing coil-flux code with each machine's adopted coil model (DIII-D:
  shaping coils as stored plus the flat solenoid term, F48; MAST: every scale 1, no flat
  term) and the filament counts already adopted for the Green's tables.
- Split: `splits/split_v1.json`. Statistics are fitted on the `train` list only.
- Only columns returned by `input_columns(machine)` may enter a feature. The existing label
  guard is applied to every column a feature reads.

## Machine constants

Held in one place (`fusion_eq/machines.py`), each with its source and a `verified` field,
and written to every build record.

| Constant | DIII-D | MAST | Status |
| --- | --- | --- | --- |
| Reference major radius R0 (m) | 1.67 | 0.85 | Verified. DIII-D: National Academies 2019, Final Report of the Committee on a Strategic Plan for U.S. Burning Plasma Research, Appendix G. MAST: Sykes et al. 2001, Nucl. Fusion 41 1423, Table 1 |
| Reference minor radius a0 (m) | 0.67 | 0.65 | Verified, same sources as R0 |
| Toroidal-field turns N_tf | 144 | 24 | Verified. DIII-D from the data (F74). MAST: Morris 1998, Europhysics News 29(6) 233 (24 centre-column conductors), consistent with Sykes et al. 2001 (2.1 MA rod current) and Berkery et al. 2021 (-86 kA per conductor) |

`q_cyl` on MAST is linear in N_tf, so an error in the MAST value shifts that input. The
build record states the verified status of each constant at build time; every constant was
verified on 13 October 2026, with the values unchanged. The alternative published MAST set,
R0 0.70 m and a0 0.50 m (Morris 1998), is a sensitivity case
(`reports/features/v1/backup_radii_check.md`, F88), not a constant.

Normalized coordinates on each machine: x = (R - R0) / a0, y = Z / a0.

## Part 0: additions to the frame table

A rebuild of the frame table with two additions. Existing columns are unchanged.

| Addition | Where | Content |
| --- | --- | --- |
| `c__rate50` | Frame rows, for `ip`, `sol` and every shaping coil | Least-squares slope over a centred 50 ms window, same rule as `c__rate` (F32). kA/s |
| `ip_rise_ms` | Shot index | Time at which Ip begins to rise, by the detection rule the Ip alignment uses, applied to both machines after any alignment |
| `sol_at_rise` | Shot index | Solenoid current at `ip_rise_ms` (windowed mean, 2 ms). kA |

`--rate-window-ms` accepts a list (default `10 50`).

## Features

Units in the formulas: currents in A, pressure in Pa, flux in Wb/rad, lengths in m, with
mu0 = 4 pi 1e-7. Ip is the signed `ip__mean`; magnitudes are used where stated.

### Group 1: coil pattern (what the coils are doing)

The calculated coil flux is sampled at a fixed set of points in normalized coordinates,
taken relative to its value at the reference point (0, 0), and divided by mu0 R0 Ip:

`coil_pt_i = (psi_coil(P_i) - psi_coil(P_ref)) / (mu0 R0 Ip)`

- psi_coil is in the physical frame, and Ip is signed, so the ratio has the same sign for
  both current directions. The DIII-D flat solenoid term cancels in the difference.
- The flux at the points comes from a Green's table evaluated at the points themselves
  (points by coil elements), built with the same code and filament counts as the grid
  tables, so that a frame's values are one matrix product.
- Layout. Candidate: the 5 by 5 lattice x in {-0.75, -0.375, 0, 0.375, 0.75},
  y in {-1.5, -0.75, 0, 0.75, 1.5}, less the reference point (24 points). The layout is
  identical on both machines in normalized coordinates and symmetric about y = 0.
- Clearance rule. Every point must lie inside the flux grid and at least 0.15 m from the
  nearest coil element on both machines. A point that fails on either machine is removed
  on both, together with its mirror partner. MAST has coils inside the vessel close to
  the plasma, so some candidate points are expected to fail. The final layout is written
  into the configuration file and the clearances into the report.

Named terms (set B). A polynomial in x and y of total order 1 to 3 is fitted to the point
values by least squares, with a fixed matrix that is the same on both machines:

`coil_pt(x, y) ~ sum over 1 <= m + n <= 3 of c_mn x^m y^n`

| Term | Meaning |
| --- | --- |
| `coil_c10` | Vertical field at the centre (holds the plasma against its outward push) |
| `coil_c01` | Radial field at the centre (pushes the plasma up or down) |
| `coil_c20`, `coil_c02` | Curvature of the field: the stretch that sets elongation |
| `coil_c11` | Tilt |
| `coil_c30`, `coil_c21`, `coil_c12`, `coil_c03` | Third order: triangularity and up-down asymmetry |

### Group 2: plasma content (Thomson)

Built from the core summaries in their time-interpolated form. Edge summaries are not
used (F38).

| Feature | Formula | Notes |
| --- | --- | --- |
| `ts_beta_p` | 8 pi^2 a0^2 p_e / (mu0 Ip^2), p_e = `ts_core_pe_mean__interp` | Electron pressure against the poloidal field pressure |
| `ts_pe_peaking` | `ts_core_pe_peak__interp` / `ts_core_pe_mean__interp` | Peak-to-average ratio |
| `ts_valid` | `ts_core_valid` as 0 or 1 | Flag; not scaled |

When `ts_valid` is false the two Thomson features are NaN in the feature table.

### Group 3: field against current

| Feature | Formula | Notes |
| --- | --- | --- |
| `q_cyl` | a0^2 N_tf abs(I_tf) / (R0^2 abs(Ip)) | The cylindrical safety factor, from Bt = mu0 N_tf I_tf / (2 pi R0). Magnitudes (F14) |

### Group 4: position in the shot

The plasma's current-spreading time is estimated per frame from size and temperature:

`tau_R = 7.8 a0^2 Te^1.5` seconds, Te = `ts_core_te_peak__interp` in keV.

The coefficient is the Spitzer estimate (Coulomb logarithm 17, effective charge 1, first
radial mode). Its absolute value is not relied on; the dependence on size and temperature
is what makes times comparable across machines. Where Thomson is invalid on a frame, the
shot's median Te over its valid frames is used; a shot with no valid frame gives NaN.

| Feature | Formula | Notes |
| --- | --- | --- |
| `ip_frac` | `ip_over_peak` | Uses the whole shot's Ip trace |
| `ip_rate_rel` | (`ip__rate50` / `ip__mean`) tau_R | Relative rate of change in the plasma's own time units |
| `resistive_age` | Integral of dt / tau_R from `ip_rise_ms` to the frame time | Trapezoid over the shot's frames, starting from (first frame time - `ip_rise_ms`) / tau_R at the first frame |

### Candidates for ablation

Built and stored with the others, outside the default sets.

| Feature | Formula | What it gives |
| --- | --- | --- |
| `shafranov_lambda` | -4 pi (R0 / a0) `coil_c10` - ln(8 R0 / a0) + 1.5 | Force balance: an estimate of beta_p + li / 2 from the coils' vertical field. Large-aspect-ratio form, inexact on MAST; a handle, not a measurement. The sign convention is confirmed against the Bz ratio of check K9 (medians -2.09 on DIII-D, -1.35 on MAST) |
| `ohmic_flux_used` | (psi_sol(P_ref, t) - psi_sol(P_ref, t_rise)) / (mu0 R0 Ip) | Solenoid flux spent since the rise, from `sol__mean`, `sol_at_rise` and each machine's solenoid model. A history handle on li |
| `greenwald_frac` | n_e pi a0^2 / Ip with n_e = `ts_core_ne_peak__interp` in 1e20 m^-3 and Ip in MA | Density relative to current and size |

Not built in this phase: the position of the Thomson temperature peak (it needs raw
channel positions, and helps DIII-D only).

### Feature sets

| Set | Content | Count |
| --- | --- | --- |
| `set_a` | Coil points, Thomson pair and flag, `q_cyl`, the three shot-position features | Points + 7 |
| `set_b` | The nine named coil terms in place of the points | 16 |
| `set_c` (version 2) | `set_a` plus `shaping` after the coil points | Points + 8 (30) |
| `set_b10` (version 2) | Version-2 `set_b` with `coil_c10` in place of `shafranov_lambda` | 15 |

Each candidate is added to a set singly in later ablations. Sets are read by the model phase
only; adding a set leaves the build's inputs unchanged, and the build checks (X1, D7) compare
the stored configuration without its sets (`features.same_build_inputs`).

## Configuration

One YAML file per feature version (`configs/features_v1.yaml`), read by the build. It
holds the machine-independent choices only: the sample-point layout, the list of
features, and the sets.

Each feature entry declares:

| Field | Content |
| --- | --- |
| `name`, `function`, `params` | The registered function that computes it and its arguments |
| `group` | Group label used in reports and ablations |
| `mirror` | `even` (unchanged), `odd` (sign changes) or `permute` (points exchanged with their mirror partners) |
| `scale` | `robust` or `none` |
| `scale_group` | Optional. Features sharing a group share one median and one interquartile range, computed over all their values |
| `scale_with` | `source` (DIII-D training statistics; default) or `target` (the target machine's own input statistics; requires disclosure) |
| `transform` | Optional, applied before scaling (for example `log1p`) |
| `uses_future` | True when the value at a frame uses samples later than that frame |

Mirror rules in version 1: coil points `permute`; `coil_c01`, `coil_c11`, `coil_c21`,
`coil_c03` are `odd`; every other feature is `even`. The coil points share the scale
group `coil_pattern`.

Feature functions are registered by name. Adding a feature takes one registered function
and one configuration entry; nothing else in the build changes.

## Deliverables

1. Package code with tests. The layout is a proposal, adjusted to the existing modules and
   agreed before implementation.
   - Feature registry and the feature functions above.
   - `build_features(frames, machine, config)`: unitless, unscaled features for a set of
     frame-table rows, keyed on `shot` and `frame`.
   - `mirror_features(X, config)`: the mirrored feature matrix, by sign changes and
     permutation only.
   - A scaler with `fit`, `transform`, `save` and `load`: robust statistics per feature or
     per scale group, fitted on originals and mirrors together; a zero interquartile range
     raises; NaN becomes 0 after scaling.
   - `feature_matrix(split, shots, set_name, mirror=False)`: the scaled matrix a model
     reads, with the column list.
   - Point Green's tables per machine, cached like the grid tables.
2. `scripts/build_frame_table.py`: extended for Part 0.
3. `scripts/build_features.py --config configs/features_v1.yaml`: builds all three splits.
   Outputs under `../data/processed/features/v1/`: one Parquet file per split (unscaled),
   `build.json` (commits, arguments, machine constants with their verified status,
   runtime). The scaler statistics are written to `reports/features/v1/scaler.json`,
   which is committed.
4. `scripts/profile_features.py`: writes `reports/features/v1/feature_report.md` with the
   checks below, and the tables `range_comparison.parquet`, `nn_distance.parquet` and
   `correlations.parquet`.
5. `scripts/feature_linear_check.py`: writes `reports/features/v1/linear_check.md`.
6. `notebooks/06_features.ipynb`: the review notebook described below.
7. Tests for every module added.
8. A "Features" section in `docs/data_dictionary.md`, and the corrected mirroring entry in
   `docs/findings.md`.

## Checks

Written to `feature_report.md` with status `pass`, `fail` or `review`.

### Build

| Id | Check | Status rule |
| --- | --- | --- |
| X1 | Rows and keys | Pass if every split has the frame table's rows and keys |
| X2 | Inputs only | Pass if every column read by a feature is in `input_columns(machine)` |
| X3 | Completeness | Review. Share of finite values per feature and split; frames where abs(Ip) is too small to divide by |
| X4 | Layout | Pass if every sample point meets the clearance rule on both machines and the layout is symmetric |
| X5 | Mirror | Pass if mirroring twice returns the input. Review: on 20 DIII-D shots, the mirrored coil points against the points calculated from A and B currents exchanged; a difference of about 2 percent of the pattern's range is expected (F51), a large one means a wrong permutation or sign |
| X6 | Current direction | Review. Distributions of the coil terms on positive-Ip against negative-Ip DIII-D shots; they are expected to overlap |
| X7 | Force-balance sign | Pass if 4 pi (R0 / a0) `coil_c10` reproduces the sign and order of the K9 medians |

### Ranges across machines

Nothing is fitted on test inputs. MAST test and DIII-D test are compared with DIII-D
training.

| Id | Check | Content |
| --- | --- | --- |
| X8 | Per-feature ranges | For each feature: the training 1st to 99th percentile band and full range; the share of MAST frames and of DIII-D test frames outside each; medians in units of the training interquartile range |
| X9 | Stretch terms | The same for `coil_c20` and `coil_c02` in detail, by phase. These decide how far the model must extrapolate in elongation |
| X10 | Combined distance | Distance in scaled feature space from each frame to its nearest DIII-D training frame (a fixed-seed sample of 200,000 training frames; 5,000 frames each of validation, DIII-D test and MAST), per set. Reported: percentiles per group and the share of MAST frames beyond the validation 99th percentile |
| X11 | Extremes after scaling | Share of scaled values beyond 10 in magnitude, per feature and split |

### Design rules and instrument differences

| Id | Check | Content |
| --- | --- | --- |
| X12 | Shared factor removed | Correlation matrix of the coil features on DIII-D training, beside the raw-current correlations of F42 |
| X13 | Pattern preserved | For a few frames, the coil points before and after scaling with the shared scale (notebook) |
| X14 | Thomson against force balance | `ts_beta_p` against `shafranov_lambda` on DIII-D training, DIII-D test and MAST: the fitted line and its scatter per machine. Agreement on DIII-D with a shift on MAST alone points to an instrument or aspect-ratio effect in one of the two routes; it is recorded as a reading |
| X15 | Timescales | `ip__rate50` raw against `ip_rate_rel`, and frame time against `resistive_age`, on both machines. Also the share of frames with changing current on MAST (open item 6) and the distribution of `ip_rise_ms` per machine |

### Linear check

| Id | Check | Content |
| --- | --- | --- |
| X16 | Information kept | On the working lists at stride 4: a ridge (alpha 1) on the scaled features of `set_a`, and of `set_b`, predicting 50 principal components of the plasma target divided by mu0 R0 abs(Ip); the map is rebuilt, the coil flux added back, and the result scored with the organizers' scorer using the scalars of the existing scalar ridge. `hybrid_ridge` is scored on the same frames in the same run as the reference (0.9186 on these lists in F61) |

A score at or above the reference means the features keep what a linear model uses in the
raw inputs. A score well below it means a group has lost information, and is examined
before any network is trained.

## Review notebook

`notebooks/06_features.ipynb` calls the package and the outputs above.

1. Build summary and the check table.
2. The sample points drawn over the coils and the grid of each machine, with clearances.
3. Coil pattern: example frames, original and mirrored; the named terms by current
   direction; correlations (X12); the scaled pattern (X13).
4. Thomson and field features on both machines.
5. Time features on both machines (X15).
6. Ranges: the per-feature table and overlaid histograms (X8, X9, X11).
7. Combined distance (X10).
8. Thomson against force balance (X14).
9. Linear check (X16).
10. Findings.

## Decisions this phase feeds

| Decision | Evidence |
| --- | --- |
| Which coil form the first model uses (points or named terms) | X10, X16 |
| Whether the plasma model must extrapolate in elongation, and so which model families are admissible | X9, X10 |
| Whether any feature is scaled with MAST's own statistics | X8, X14 |
| Whether time features are kept for transfer or for DIII-D only | X15 |
| Which candidates enter the first ablations | X8, X14 |

## Tests

On synthetic inputs:

- A coil pattern that is an exact polynomial of order 3 is recovered by the named terms.
- A uniform flux offset leaves every coil feature unchanged.
- Reversing every current together with Ip leaves every feature unchanged.
- Mirroring twice is the identity; a pattern symmetric in y has zero odd terms; the
  mirror of the points equals the points of the flipped field.
- Features have no units: scaling R0, a0 and the coil geometry together, with currents
  scaled to match, leaves the coil features unchanged.
- The scaler fitted on originals and mirrors gives a zero centre for odd features and one
  shared scale for a scale group; NaN becomes 0; a constant feature raises.
- A label column requested by a feature raises.
- Every registered feature declares `mirror`, `scale` and `uses_future`, and every
  feature in a set is registered.
- `resistive_age` increases along a shot and is the same for two shots that differ only
  by a shift of the time axis.

## Procedure

1. Layout review: the existing modules this phase touches, the function that detects the
   Ip rise, the definition of the K9 Bz ratio, and the clearance of the candidate sample
   points on both machines. The module layout and the final point layout are agreed
   before implementation.
2. Implement Part 0, the features, the mirror, the scaler and the tests. Run the tests.
3. Frame table rebuild, launched manually: `python scripts/build_frame_table.py`.
4. Feature build, launched manually:
   `python scripts/build_features.py --config configs/features_v1.yaml`.
5. Profile, launched manually: `python scripts/profile_features.py`.
6. Review notebook sections 1 to 8.
7. Linear check, launched manually: `python scripts/feature_linear_check.py`.
8. Findings added to `docs/findings.md`; the mirroring entry corrected.

## Constraints

- Nothing is written under `../data/raw/`.
- No label column is used to compute any feature, flag or statistic.
- No statistic is fitted on test inputs in version 1. The `target` scaling option exists
  in the code and is not used by any feature.
- MAST equilibrium labels are not read.
- Every run against the data is launched manually.
- Built tables are not committed. Code, tests, the configuration, the notebook and
  `reports/features/` are.
- New packages are added to `requirements.txt` the same day they are installed.
