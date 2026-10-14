# Target specification: current share, remainder and the map they rebuild

Specification for the target phase. The repository's data rules and writing standard
apply. Finding ids (F01 to F79) refer to `docs/findings.md`. Check ids in this document are
Y1 to Y16. "Remainder" is the vacuum remainder of F79 (the leftover term of the 12 October
decision).

## Purpose

Build what the plasma model is trained to predict, and the physics that turns a prediction
back into a flux map on either machine: the plasma current inside the boundary as a share
of Ip, a basis for it that is symmetric up-down and can be placed on another machine's
grid, the flux that current produces through each machine's Green's function, and the
remainder learned on DIII-D.

The phase settles six things:

1. Whether the three-part map (coil flux, flux of the current, remainder) keeps its ceiling
   when the current basis is fitted on originals and mirrors (reference: variant f of F79,
   S 0.9982, consistency 0.9944).
2. How many components the symmetric basis needs.
3. What placing the current onto another machine's grid costs.
4. Whether the predicted share should be rescaled to a total of one.
5. Whether the remainder ridge holds on the full lists and with the 50 ms rates.
6. Whether a DIII-D-shaped current, placed on MAST with the working machine constants,
   lies inside the MAST grid and clear of its coils. This uses no MAST label.

Out of scope: the plasma model, its loss and its loader (they call the functions built
here), q95 and betaN, the plasma-flux target held as the fallback, a model of the MAST
remainder, and any use of MAST equilibrium labels.

## Decisions this phase implements

Taken on 12 October 2026 and recorded under "Carried into modelling".

- Target: the toroidal current density inside the plasma boundary, as a share of Ip, on a
  lattice scaled to each machine. Each machine's Green's function turns it into flux.
- Final map: coil flux, plus the flux of the predicted current, plus the remainder. The
  remainder is learned on DIII-D (a ridge on Ip-signed currents and their rates) and is
  zero on MAST.
- Mirroring: the mirror of a frame is the flip of its target in Z. Z_axis changes sign,
  tri_top and tri_bot exchange, every other label is unchanged. Bases for the current are
  fitted on originals and mirrors together.
- The remainder is hardware of DIII-D (coil-model error and vessel currents, with the flux
  of the current the mask drops; F79, a reading). Its basis and ridge are fitted on
  originals only, on DIII-D's native grid.

## Context

- Existing code: `fusion_eq/current_target.py` (current density from the map, mask by the
  scorer's boundary, plasma response, remainder) and `scripts/current_ceiling.py`
  (variants a to g). This phase extends them. Functions already validated by variant e
  (bookkeeping exact to 1e-16) are reused, not rewritten.
- Coil flux: the existing code with each machine's adopted coil model (F48).
- Sign rule: the per-machine constant already in the package (F45, F71).
- Input: the frame table, for Ip at each frame and for the remainder ridge. Flux maps are
  read through the existing loader.
- Split: `splits/split_v1.json`. Everything fitted is fitted on the `train` list.
- Machine constants: `fusion_eq/machines.py` (see `docs/FEATURES.md`). On MAST the reference
  radii decide where the predicted current is put: an error of 5 cm in the MAST major radius
  moves the plasma by 1.6 grid cells. Every constant is verified against a published source
  (13 October 2026); the values are unchanged and the sources are recorded in the file:

  | Constant | Value | Source |
  | --- | --- | --- |
  | MAST R0, a0 | 0.85 m, 0.65 m | Sykes et al. 2001, Nucl. Fusion 41 1423, Table 1 |
  | MAST toroidal-field turns | 24 | Morris 1998, Europhysics News 29(6) 233: 24 centre-column conductors; consistent with the 2.1 MA rod current of Sykes et al. 2001 and the -86 kA per conductor of Berkery et al. 2021 |
  | DIII-D R0, a0 | 1.67 m, 0.67 m | National Academies 2019, Final Report of the Committee on a Strategic Plan for U.S. Burning Plasma Research, Appendix G |
  | DIII-D toroidal-field turns | 144 | F74 |

  The alternative published nominal set for MAST, R0 0.70 m and a0 0.50 m (Morris 1998), is
  a sensitivity case (Y11), not a constant.

## Definitions

Units: flux in Wb/rad, currents in A (Ip = 1e3 times the signed `ip__mean`), lengths in m.
Subscript m is the machine.

| Symbol | Definition |
| --- | --- |
| psi_phys | k_m sign(Ip) psi_stored, with k = -1 on DIII-D and +1 on MAST |
| psi_coil | Calculated coil flux in the physical frame, adopted coil model |
| psi_plasma | psi_phys - psi_coil |
| I_cell | Toroidal current density of psi_phys times the cell area dR dZ, inside the scorer's boundary of the true map; zero outside |
| s | I_cell / Ip. The current share: no units, positive where the current flows with Ip |
| G_m | Plasma response of machine m: flux at every grid node per ampere in every grid cell (4,225 by 4,225) |
| P_m | Placement of the reference lattice onto the native grid of machine m (identity on DIII-D) |
| psi_J | Ip G_m P_m s, the flux of the current |
| r | psi_plasma - psi_J, the remainder |

The map in the stored frame is then

`psi_stored = k_m sign(Ip) (psi_coil + Ip G_m P_m s + r)`

exactly, by the definition of r. A prediction replaces s by its basis form and r by the
ridge's output (DIII-D) or zero (MAST).

The share is divided by the measured Ip, an input available on every test frame, and is
not rescaled. Its total over the grid is the masked current over the measured Ip: median
0.9975, 5th to 95th percentile 0.989 to 1.007 (F78). Forcing the total to one is priced as
a ceiling variant and left to the model phase.

## Reference lattice and placement

Scaled coordinates on each machine: x = (R - R0) / a0, y = Z / a0.

The reference lattice is the DIII-D flux grid in DIII-D's scaled coordinates: 65 by 65
nodes, x from -1.239 to 1.299 in steps of 0.0396, y from -2.388 to 2.388 in steps of
0.0746, each node owning one cell. The share and its basis live on this lattice.

| | DIII-D | MAST |
| --- | --- | --- |
| Reference lattice in metres | The native grid | R 0.045 to 1.694, Z -1.552 to 1.552; cell 25.8 by 48.5 mm |
| Native grid in scaled coordinates | The reference lattice | x -1.215 to 1.769, y -3.077 to 3.077; cell 0.0466 by 0.0962 |
| Placement | Identity | Conservative remap |

Consequences of this choice:

- On DIII-D nothing is resampled. The targets, the ceiling of F79 and the mirror (an exact
  flip of the Z index, since the grid is symmetric about Z = 0) are unaffected by the
  machine constants.
- The machine constants enter only the placement on MAST. A corrected constant changes
  P_mast and nothing else; no target is rebuilt.

Placement on MAST. P_m maps a share on the reference lattice to a share on the native
cells by overlap of cell areas in scaled coordinates: each reference cell gives each
native cell the fraction of its own area that the native cell covers. The operator is
linear, separable in x and y, and conserves the total of any share lying inside the native
grid. Native cells outside the reference lattice receive nothing. The first reference
column (R = 0.045 m on MAST) lies inside the centre column and outside the MAST grid; the
weight lost there is reported and expected to be zero.

MAST's cells are larger than the reference cells in scaled units (by 1.18 in x and 1.29 in
y), so placement averages; it does not interpolate between coarse values.

## Part 1: current share and its basis

For every labelled frame: psi_phys, the current density, the scorer's boundary, the mask,
the share s on the native grid, and its diagnostics (total, share of the absolute current
outside the boundary, whether a boundary was found).

Basis. Principal components of s:

- Fitted on a fixed-seed sample of 100,000 frames of the `train` list that pass the
  lighter label filter and have a boundary, together with their mirrors.
- Parity is exact: every component is even or odd in y and the mean is even. One way is to
  fit the even part (s + flip s) / 2 and the odd part (s - flip s) / 2 separately and merge
  by variance; fitting on the stacked data is acceptable if the parity test passes.
- 50 components by default, orthonormal, ordered by variance. The parity of each component
  is stored as a vector p of +1 and -1.

Coefficients: `c = B^T (s - s_mean)`. These 50 numbers per frame are the plasma model's
target. Component variances are stored so the model phase can standardize them.

The lighter label filter (impossible label values, toroidal field under 50 kA, Ip under
0.1 of the shot's peak, the two flagged shots; F77) becomes one function in the package,
used here for the fit samples and by the model phase for training frames. Targets are
built for every labelled frame; the filter result is stored as a flag.

## Part 2: flux of the current on each machine

- Plasma response G_m for both machines, built by the existing code on each native grid
  and cached like the coil Green's tables.
- Placement P_m for both machines.
- Flux basis per machine: `phi_mean_m = G_m P_m s_mean` and `Phi_m = G_m P_m B`
  (4,225 by 50), in Wb/rad per ampere of Ip. The flux of a predicted current is then
  `Ip (phi_mean_m + Phi_m c)`: one matrix product, which is what a loss computed on the
  flux of the predicted current needs.
- `compose_map(machine, psi_coil, ip, c, remainder=None)`: the stored-frame map from the
  parts, with the sign rule applied. The inverse of the target build.

Known limit, recorded and not addressed here: each component carries the dependence of
current density on major radius that DIII-D plasmas have. On MAST, where R varies across
the plasma by a much larger factor than on DIII-D, that dependence differs. It is judged
by MAST submissions.

## Part 3: remainder

On DIII-D, originals only, native grid.

- `r = psi_plasma - Ip G s` with the exact masked share (not its 50-component form), as in
  F79.
- Basis: principal components of sign(Ip) r in Wb/rad, 10 components, fitted on the same
  sample of training frames.
- Ridge (alpha 1) from the 42 inputs of variant g (Ip, solenoid, toroidal field and the 18
  F-coils, mean and rate, currents Ip-signed) to the 10 coefficients, fitted on all `train`
  frames that pass the filter. Where the frame table carries the 50 ms rates, the ridge is
  fitted with each rate window and both are reported; the better one on validation is
  saved as the default.
- The mirror of a frame's remainder is the flip of the original's. The ridge is not
  retrained on mirrors.

## Part 4: mirror on the target side

| Quantity | Mirror |
| --- | --- |
| Share s, any map on the native DIII-D grid | Flip of the Z index |
| Current coefficients c | c times p (odd components change sign) |
| Z_axis | Sign changes |
| tri_top, tri_bot | Exchanged |
| R_axis, kappa, volume, li, q95, betaN | Unchanged |
| Remainder, coil flux | Flip of the original's |

`mirror_coeffs` and `mirror_labels` are the functions the loader calls beside
`mirror_features`. Because the basis has exact parity, mirroring a target in training is a
multiplication by p; no map is rebuilt.

The seven shape numbers the scorer derives from the true map (R_axis, Z_axis, kappa,
tri_top, tri_bot, volume, li) are stored per frame, computed with the scorer's functions
(F23) and taken from the cached scorer references where they exist. They serve loss terms
and diagnostics in the model phase.

## Part 5: ceiling rerun

`scripts/current_ceiling.py` gains four variants, scored like variants a to g on the
working lists (100 validation shots, stride 4, true q95 and betaN, bases fitted on the
working training list).

| Variant | Content | Compared with |
| --- | --- | --- |
| h | As f at 10 remainder components, with the current basis fitted on originals and mirrors, at 50, 60 and 80 components | f: S 0.9982, consistency 0.9944, volume 0.998 |
| i | h at 50, with the reconstructed share placed onto the lattice of MAST's nodes and back before the flux is calculated | h |
| j | h at 50, with the reconstructed share rescaled to a total of one | h |
| k | h at 50, with the remainder from the ridge, for each rate window available | g: S 0.9973, consistency 0.9911, volume 0.991 |

Reading:

- h. 50 components are kept if S is within 0.0003 of f and consistency within 0.001 (the
  cost of float16 storage, F69, sets the scale). Otherwise the smallest count that meets
  it is adopted.
- i. A pessimistic estimate of the placement cost on MAST: DIII-D's share is coarsened to
  MAST's cell size in scaled units and brought back, two remaps where MAST has one, and
  judged against a truth at DIII-D's resolution.
- j. Decides whether the model phase rescales the predicted share.
- k. The ceiling of the whole target chain on DIII-D with nothing taken from the labels
  but the current coefficients.

## Part 6: mean-current predictor

A predictor `mean_current_v0` in the registry: coil flux plus `Ip phi_mean_m`, with the
ridge remainder on DIII-D and zero on MAST, and the scalars of `baseline_v0`. It contains
no model. It exercises the MAST path end to end (sign constant, placement, machine
constants, MAST response) before a network exists.

- Local reference on DIII-D validation, with and without the remainder. `physics_mean`
  (coil flux plus the mean plasma shape, F59: R2 psi 0.991, S 0.800) is the comparable
  earlier result.
- Label-free check on MAST test frames (Y16).
- Submission is optional and launched manually. Its MAST R2 psi is read against the DIII-D
  reference without the remainder; `coil_only_v0` gave -1.357 on MAST (F72).

## Configuration

`configs/target_v1.yaml`, read by the build and the ceiling:

| Field | Default |
| --- | --- |
| `current.n_components` | 50 |
| `current.fit_frames`, `current.seed` | 100000, 0 |
| `current.symmetric` | true |
| `remainder.n_components` | 10 |
| `remainder.ridge_alpha` | 1.0 |
| `remainder.inputs`, `remainder.rate_windows_ms` | The 42 inputs of variant g; [10, 50] |
| `mask` | `scorer_boundary` |
| `placement` | `conservative` |
| `shape_numbers` | true |

Machine constants are not in this file; they are read from `fusion_eq/machines.py` and
written to the build record with their verified status.

## Deliverables

1. Package code with tests. The layout is a proposal, adjusted to the existing modules and
   agreed before implementation.
   - `reference_lattice()`, `native_lattice(machine)` and `placement(machine)`.
   - `current_share(...)`: the share and its diagnostics for a frame or a shot, wrapping
     the existing masked-current function.
   - A current basis with `fit`, `transform`, `inverse`, `save` and `load`, holding the
     mean, the components, their variances and the parity vector.
   - `flux_basis(machine, basis)` and `compose_map(...)`.
   - A remainder model with `fit`, `predict`, `save` and `load` (basis and ridge).
   - `mirror_share`, `mirror_coeffs`, `mirror_labels`.
   - The lighter label filter as one function.
   - Plasma response tables for both machines, cached.
2. `scripts/current_ceiling.py`: variants h to k added to `reports/current_target/`.
3. `scripts/build_targets.py --config configs/target_v1.yaml`: fits the bases and the
   ridge on `train`, then builds the tables. Options `--lists` (default: train, validation
   and local_test; `working` for the two working lists) and `--stride`. Outputs under
   `../data/processed/targets/v1/`: one Parquet file per list keyed on `shot` and `frame`
   (current coefficients, share total, share outside the boundary, reconstruction error,
   remainder coefficients, remainder RMS over the frame's flux range, shape numbers,
   boundary and filter flags), the fitted objects (current basis, flux bases and placement
   per machine, remainder model), and `build.json` (commits, arguments, machine constants
   with their verified status, checksums of the fitted objects, runtime).
4. `scripts/profile_targets.py`: writes `reports/targets/v1/target_report.md` with the
   checks below, and `target_summary.json`.
5. `mean_current_v0` in the predictor registry.
6. `notebooks/07_targets.ipynb`: the review notebook described below.
7. Tests for every module added.
8. A "Targets" section in `docs/data_dictionary.md`.

## Checks

Written to `target_report.md` with status `pass`, `fail` or `review`.

### Build

| Id | Check | Status rule |
| --- | --- | --- |
| Y1 | Rows and keys | Pass if each table has one row per labelled frame of its list (at the stride used) and the frame table's keys |
| Y2 | Bookkeeping | Pass if, on 1,000 frames, coil flux plus Ip G s plus r equals psi_phys to 1e-10 of the frame's flux range, and `compose_map` returns the stored map |
| Y3 | Share total | Pass if the median total is within 0.99 to 1.005. Reported: percentiles by list and by phase; frames with no boundary |
| Y4 | Parity | Pass if every component is even or odd to 1e-8 and the mean is even. Reported: count of each |
| Y5 | Mirror | Pass if, on 1,000 frames, the coefficients of the flipped share equal c times p to 1e-8, and mirroring twice is the identity |
| Y6 | Reconstruction | Review. Relative error of the 50-component share by list and phase, training against validation, originals against mirrors |
| Y7 | Flux basis | Pass if `phi_mean + Phi c` equals G P applied to the reconstructed share to 1e-10, on both machines |

### Placement and MAST

No MAST label is read.

| Id | Check | Content |
| --- | --- | --- |
| Y8 | Conservation | Pass if placement on MAST conserves the total of the mean share and of 1,000 reconstructed training shares to 1e-10, after the weight outside the MAST grid, which is reported |
| Y9 | Response | Pass if, on both machines, the response to a unit current in one cell agrees with the coil Green's function for a filament at that node to 1e-3 at nodes more than five cells away, and the response is symmetric under a flip in Z |
| Y10 | Where the current lands | Review. The mean share and the envelope holding 99 percent of the training shares, placed on MAST and reported in metres: inner and outer radius, height, centroid; distance to the nearest MAST coil element and to the grid edge; coil elements inside the envelope |
| Y11 | Sensitivity to the constants | Review. Y10's outputs with the MAST R0 and a0 each moved by plus and minus 5 percent, and with the alternative published nominal set R0 0.70 m, a0 0.50 m (Morris 1998) |

### Remainder

| Id | Check | Content |
| --- | --- | --- |
| Y12 | Size | Remainder RMS over the frame's flux range by phase, full lists, beside F79 (median 0.0105; rising 0.0193, flat-top 0.0093, late 0.0143) |
| Y13 | Components | Variance held by 5, 10 and 20 components, beside F79 (0.9978, 0.9998, 1.0000) |
| Y14 | Ridge | R2 of the 10 coefficients on training and validation for each rate window, beside F79 (0.8885, 0.8842) |

### Ceiling and the mean-current predictor

| Id | Check | Content |
| --- | --- | --- |
| Y15 | Variants h to k | The table of Part 5 with every score term, beside f and g |
| Y16 | Mean current | `mean_current_v0` on DIII-D validation with and without the remainder (every score term). On 2,000 MAST test frames and 2,000 DIII-D test frames: share of frames on which the scorer finds a closed boundary in the composed map, and the medians of the derived R_axis, Z_axis and kappa. Review |

## Review notebook

`notebooks/07_targets.ipynb` calls the package and the outputs above.

1. Build summary and the check table.
2. One frame through the chain: stored map, coil flux, current share, flux of the current,
   remainder, and the recomposed map.
3. The mean share and the leading components, with their parity; a frame and its mirror.
4. The same mean and components placed on MAST, drawn in metres over the MAST coils and
   grid (Y10, Y11).
5. Remainder: leading components, size by phase, ridge prediction against truth (Y12 to
   Y14).
6. Ceiling variants (Y15).
7. Mean-current maps on both machines (Y16).
8. Findings.

## Decisions this phase feeds

| Decision | Evidence |
| --- | --- |
| Number of current components | Y15 (h), Y6 |
| Whether the model phase rescales the predicted share to a total of one | Y15 (j), Y3 |
| Whether placement resolution needs more than the conservative remap | Y15 (i) |
| Which rate window the remainder ridge uses | Y14, Y15 (k) |
| Whether the MAST reference radii must be verified before the first model submission | Y10, Y11, Y16 |
| Whether target B stands or the fallback is prepared | Y15 (h, k) against f and g |

## Tests

On synthetic inputs:

- A map built from a coil flux, a known share and a known remainder returns that share and
  that remainder, and `compose_map` returns the map.
- A uniform current density inside a circle gives a share whose total is the masked
  current over Ip.
- Reversing every current together with Ip leaves the share and the coefficients
  unchanged.
- The basis has exact parity; the coefficients of a flipped share are c times p; mirroring
  twice is the identity; `mirror_labels` changes Z_axis and the triangularities only.
- Placement is the identity on DIII-D. On MAST it conserves the total, leaves a share
  outside the reference lattice at zero, and keeps the centroid and widths of a smooth blob
  in scaled coordinates to within a tenth of a cell.
- The response of each machine is symmetric in its two arguments and under a flip in Z.
- No units: scaling a machine's grid and radii by one factor leaves
  psi_J / (mu0 R0 Ip) unchanged for the same share.
- The remainder model recovers a planted linear relation and survives save and load.
- A list without labels, and the labelled MAST sample shots, are refused by the target
  build with a clear error.

## Procedure

1. Layout review: the functions of `current_target.py` this phase reuses, how the existing
   response treats the cell that holds the current and the cells at the grid edge, where
   the Green's tables and scorer references are cached, and whether a label filter
   function exists. The module layout and the storage of the fitted objects are agreed
   before implementation.
2. Implement Parts 1 to 6 and the tests. Run the tests.
3. Ceiling rerun, launched manually: `python scripts/current_ceiling.py`.
4. Target build on the working lists, launched manually:
   `python scripts/build_targets.py --config configs/target_v1.yaml --lists working`.
5. Profile, launched manually: `python scripts/profile_targets.py`. Review notebook
   sections 1 to 7.
6. Full target build, launched manually:
   `python scripts/build_targets.py --config configs/target_v1.yaml`, then the profile
   again.
7. Findings added to `docs/findings.md` (F80 onward) and "Carried into modelling" updated.

Steps 3 to 5 do not need the frame table rebuild of `docs/FEATURES.md`; without it the
remainder ridge is fitted with the 10 ms rates only.

## Constraints

- Nothing is written under `../data/raw/`.
- MAST equilibrium labels are not read. Every MAST quantity in this phase comes from
  geometry, machine constants and test inputs.
- Nothing is fitted on test inputs.
- Bases and the ridge are fitted on the `train` list only (the working training list in
  the ceiling script).
- Every run against the data is launched manually.
- Built tables and fitted objects are not committed. Code, tests, the configuration, the
  notebook and `reports/targets/` are.
- New packages are added to `requirements.txt` the same day they are installed.
