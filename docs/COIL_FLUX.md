# Coil flux specification: testing the physics-first structure

Specification for the coil-flux phase. The repository's data rules and writing standard
apply. Item ids (D1 to D5, D8, D9) match the pre-modelling analysis plan. Finding ids (F01
to F25) refer to `docs/findings.md`.

## Purpose

Calculate the flux that the coils produce on the EFIT grid from coil geometry and coil
currents, and test whether subtracting it from a true DIII-D flux map leaves the flux of
the plasma alone. If it does, the physics-first structure is viable: the coil part is
calculated on any machine and a model learns only the plasma part.

The phase settles five things:

1. Whether the calculated coil flux matches the data, and to what accuracy (D3).
2. The rule relating the stored sign of the flux map to the direction of the plasma
   current, which runs both ways on DIII-D (D3, F14).
3. The units and turn conventions of the current columns, and the scale of the solenoid
   (D4, open questions 1 and 2 of the data dictionary).
4. Whether a frame mirrored top to bottom is a valid training example (D9).
5. Whether the same code gives a physically sensible coil flux on MAST, using inputs and
   geometry only (D5).

Out of scope: the remainder across the working subset (D6), wall currents (D7), any model,
and any feature built from the coil flux (Part E).

## Context

- Coil geometry is identical in every shot of a machine (F06). DIII-D has 19 elements,
  one per current column, including the solenoid `magnetics_ECOILA`. MAST has 812
  elements: 156 for the ten P-coils and 656 for the solenoid. The toroidal field coils
  and `magnetics_efps_current` have no geometry and produce no flux on the map.
- Geometry columns: `coil_name`, `coil_input_column`, `coil_R`, `coil_Z`, `coil_width`,
  `coil_height`, `coil_angle1`, `coil_angle2`. Six DIII-D coils are parallelograms
  (F5A/B, F6A/B, F7A/B); every MAST element is a rectangle.
- The flux map `efit_psirz` is in V·s/rad, indexed [frame, Z index, R index], on one
  fixed 65 by 65 grid per machine (F04).
- Current magnitudes are consistent with kA and kA-turn (F07), but the dataset README
  contradicts itself on DIII-D units (data dictionary, D1), and the solenoid's turn
  convention on DIII-D is unresolved. This phase resolves both.
- The magnetic axis is at the flux minimum on every DIII-D shot, for both directions of
  the plasma current (F14). The scorer's `AXIS_SIGN` gives the convention per machine:
  minimum on DIII-D, maximum on MAST.
- Currents at a frame time come from `fusion_eq.resample.window_mean`, built in the
  frame-table phase. This phase is implemented after that code is in place.
- The scorer's boundary extraction and axis finder are used through
  `fusion_eq/starter_kit.py`.

### MAST and the challenge rules

No MAST equilibrium label is read in this phase. The three labelled MAST sample shots
are not loaded. MAST conventions are set from the dataset documentation, published
machine parameters and label-free physics checks. Everything that needs a true flux map
runs on DIII-D training shots.

## The calculation

The flux per radian at (R, Z) from a circular filament at (Rc, Zc) carrying current I,
with lengths in metres and current in amperes:

```
psi = (mu0 * I / (2 * pi)) * sqrt(R * Rc) / k * ((2 - k^2) * K(k^2) - 2 * E(k^2))
k^2 = 4 * R * Rc / ((R + Rc)^2 + (Z - Zc)^2)
```

K and E are the complete elliptic integrals (`scipy.special.ellipk`, `ellipe`, which
take k^2). A positive current gives positive flux with its maximum at the filament.

Each coil element is divided into a grid of filaments that share its current. The
parallelogram convention for `coil_angle1` and `coil_angle2` is confirmed from the
starter kit's documentation before implementation and recorded in the data dictionary.
It is expected to follow EFIT's: `angle1` tilts the top and bottom sides from the
horizontal, `angle2` is the angle of the left and right sides from the horizontal, and 0
means no skew.

Two further operators are needed for the tests, and are reused by later phases:

```
delta_star(psi) = R * d/dR((1/R) * dpsi/dR) + d2psi/dZ2 = -mu0 * R * J
Bz = (1/R) * dpsi/dR
```

J is the toroidal current density. Its integral over the plasma cross-section is the
plasma current.

## Deliverables

1. Package functions with tests. The layout is a proposal, agreed before implementation.
   - `fusion_eq/coil_flux.py`: the filament Green's function; division of an element
     into filaments, including parallelograms; `build_green_table(machine, geometry,
     grid)` returning one 65 by 65 pattern per current column for one unit of that
     column's current; `coil_flux(table, currents)` returning the map for one or many
     frames as a weighted sum.
   - `fusion_eq/grad_shafranov.py`: `delta_star`, `current_density(psi, grid, mask)`,
     `vertical_field(psi, grid)` and `flux_from_current(J, grid, targets)`, which sums
     the Green's function over source cells for target points that are not source
     cells.
   - `fusion_eq/machines.py`: for each current column with geometry, how its current is
     divided among its elements (`shared`: the column holds the total, split equally;
     `per_element`: every element carries the column's value) and a scale factor. The
     initial values follow the data dictionary: DIII-D F-coils and MAST P-coils
     `shared` in kA-turn; MAST solenoid `per_element` in kA; DIII-D solenoid scale left
     as the quantity this phase fits.
2. `scripts/build_green_tables.py`: writes one table per machine to
   `../data/processed/green/` with a `build.json`. Geometry and grid are read from one
   shot per machine.
3. `scripts/fit_coil_scales.py`: the fit described under D4, over a sample of DIII-D
   training shots. Writes `reports/coil_flux/coil_scales.parquet` and
   `coil_flux_report.md`.
4. `notebooks/03_coil_flux.ipynb`: the review notebook described below.
5. `tests/test_coil_flux.py`, `tests/test_grad_shafranov.py`.
6. The confirmed conventions (units, turn handling, skew angles, sign rule) added to
   `docs/data_dictionary.md`.

## Analyses

| Id | Analysis | Method | Decision it feeds |
| --- | --- | --- | --- |
| D1 | Coils on the grid | Draw every coil element over the grid outline for both machines, coloured by current column. Confirm every `coil_input_column` value is a current column present in the files. | The geometry and the join are understood; a wrong skew convention shows as misplaced F5 to F7 coils |
| D2 | Green's function | Implement and test as below. Compare the coil flux at 1 by 1, 4 by 4 and 8 by 8 filaments per element. | The subdivision used from here on |
| D3a | Single-frame subtraction | For 8 DIII-D training shots (3 with positive Ip at 20 kHz, 3 with negative Ip, 2 at 2 kHz), at three frames each (current rising, mid flat-top, late): true map, calculated coil flux and their difference, drawn side by side with the stored axis and the scorer's boundary. | Whether a single smooth hill centred on the axis remains, with no structure localized at a coil |
| D3b | Current consistency | On the same frames: current density from the true map with `delta_star`, kept inside the scorer's boundary. Compare its integral with the measured Ip. Then calculate the plasma's own flux from that current at the grid points outside the boundary, and compare true map minus plasma flux with the calculated coil flux there. | A test with no free parameter: outside the plasma, what is left of the true map after removing the plasma's own flux must be the coil flux. Gives the sign rule and the accuracy. |
| D4 | Coil scales and the solenoid | Over 50 DIII-D training shots, all frames: least-squares fit of (true map minus plasma flux), at points outside the boundary, to the sum of each column's pattern times its current times an unknown scale, one scale per current column, with the sign rule from D3b applied. Repeat with a per-frame constant added. Then, through each of the 8 shots, track the mean of the unexplained part against the solenoid current. | The scale of every current column. Shaping coils are expected at 1. The solenoid scale is the open turn convention. A constant that is needed, or a drift with the solenoid current, means flux is missing from the calculation. |
| D8 | Green's tables | One stored table per machine; coil flux for any frame is a weighted sum. Time the calculation for 100,000 frames. | Fast enough to run on every frame in training and at prediction |
| D9 | Up-down symmetry | Confirm that DIII-D's A and B coils mirror each other in Z and that the grid is symmetric about Z = 0. Check that the coil flux from swapped A and B currents equals the original flipped in Z. Count frames by the position of the scorer's X-point: lower, upper or both. | Whether a mirrored frame is a valid extra training example and how much it adds |
| D5 | MAST, label-free | Build the MAST table. For 20 MAST test shots, calculate the coil flux through the shot from inputs only. Three checks: (1) the vertical field at the reference point divided by mu0 x Ip / (4 pi R0), compared with the same ratio on DIII-D; (2) the change in solenoid flux at the reference point from the start of the Ip rise to flat-top, divided by mu0 x R0 x Ip, compared with DIII-D; (3) maps of the coil flux with field nulls marked, at flat-top. | Whether the MAST turn conventions are right to within their large possible errors (a factor of 4 to 23 on a P-coil, 656 on the solenoid), and that one code path serves both machines |

The reference point for D5 is provisional: the published major radius of each machine at
Z = 0 (DIII-D 1.67 m, MAST 0.85 m). The source is recorded in the data dictionary. The
definition is settled in Part E.

## Acceptance checks

Written to `coil_flux_report.md` with status `pass`, `fail` or `review`.

| Id | Check | Status rule |
| --- | --- | --- |
| K1 | Geometry join | Pass if every element maps to a current column present in the files and the counts are 19 on DIII-D and 812 (156 and 656) on MAST |
| K2 | Filament convergence | Pass if the coil flux on the grid changes by less than 0.1% of its range between the chosen subdivision and one twice as fine |
| K3 | Total current | Review. Distribution over frames of the integrated current density over the measured Ip. Expected magnitude close to 1; the sign gives the convention. |
| K4 | Sign rule | Pass if one rule, stated in terms of the direction of Ip, holds on all 8 shots |
| K5 | Agreement outside the plasma | Review. Root-mean-square difference between (true map minus plasma flux) and the scaled coil flux, as a fraction of the flux range of the frame, by phase of the shot |
| K6 | Coil scales | Review. Fitted scale and its spread across shots, per current column |
| K7 | Remainder | Review. Distance from the extremum of (true map minus coil flux) to the stored axis; spread of its amplitude divided by Ip across flat-top frames |
| K8 | Mirror symmetry | Pass if the flipped and the swapped-current coil flux agree to within 0.1% of the flux range |
| K9 | MAST checks | Review. The two ratios of D5 on MAST against their DIII-D distributions |

## Review notebook

`notebooks/03_coil_flux.ipynb`. Sections:

1. Coils on the grid, both machines (D1).
2. One frame in detail: true map, coil flux, difference (D3a). Repeated for one shot of
   each Ip direction.
3. Current density from the true map, its total against Ip, and the comparison outside
   the plasma (D3b).
4. Coil scales: the fitted values, with and without the per-frame constant; the
   unexplained part against solenoid current through a shot (D4).
5. Mirror check and X-point counts (D9).
6. MAST: coil flux maps and the two ratios against DIII-D (D5).
7. Findings: the conventions adopted, the accuracy reached, and whether the structure is
   viable.

## Decisions this phase feeds

| Decision | Options | Evidence |
| --- | --- | --- |
| Overall structure | Physics-first is viable; viable with a fitted solenoid scale; not viable | K5 and K7, notebook sections 2 to 4 |
| Sign rule | The rule that puts every shot into one standard direction | K3, K4 |
| Units and turns | Per current column on both machines | K6, K9 |
| Solenoid | Calculated outright; calculated with a fitted scale; a learned offset | K6, notebook section 4 |
| Data augmentation | Mirroring valid or not | K8, X-point counts |

The structure is treated as viable if, with at most the solenoid scale fitted, the
agreement outside the plasma on flat-top frames is within a few percent of the frame's
flux range and the difference map shows no structure localized at a coil. The threshold
is provisional and is revisited when the numbers are in.

## Tests

- The filament formula agrees with direct numerical integration of the loop's vector
  potential at a set of points, to a relative error below 1e-6.
- Near the axis of a single loop, the field from the flux matches mu0 x I / (2a).
- The formula is symmetric under exchange of source and target.
- A rectangle divided into more filaments converges; a parallelogram's filaments lie
  inside its outline and preserve its area.
- `coil_flux` with a table equals the direct sum over elements.
- `shared` and `per_element` conventions give total currents that differ by exactly the
  element count.
- `delta_star` applied to the flux of a filament is zero away from the filament, and the
  integrated current density of a synthetic plasma made of filaments recovers their
  total current.
- `flux_from_current` reproduces the flux of that synthetic plasma outside it.
- Mirroring: flipping a synthetic up-down symmetric coil set leaves the flux unchanged.

## Procedure

1. Review the starter kit's geometry documentation and `fusion_scoring/`. Confirm the
   skew-angle convention. Agree the module layout.
2. Implement the functions and tests. Run the tests.
3. Build the tables, launched manually: `python scripts/build_green_tables.py`.
4. Review notebook sections 1 to 3. The sign rule and the convergence choice are fixed
   here before the fit.
5. Fit, launched manually: `python scripts/fit_coil_scales.py`.
6. Review notebook sections 4 to 7. Add the findings to `docs/findings.md` and the
   conventions to `docs/data_dictionary.md`.

## Constraints

- Nothing is written under `../data/raw/`.
- No MAST equilibrium label is read.
- Labels are used on the target side only: true maps, the scorer's boundary and the
  stored axis serve the tests, and never enter the coil-flux calculation.
- Every run against the data is launched manually.
- Scales fitted in this phase are fitted on DIII-D training shots only.
- New packages are added to `requirements.txt` the same day they are installed.
