# MAST solenoid scale check (label-free)

Status: script written, full run pending. Numbers marked "smoke" come from the first 40 shots
of each test split (`--max-shots 40`) and are replaced by the full run.

Script: `scripts/mast_solenoid_check.py`. Outputs: `reports/coil_flux/mast_solenoid_check/`
(`shots.parquet`, `impact_features.parquet`, `summary.json`).

```
python scripts/mast_solenoid_check.py
```

## Question

The first model submission (`2026-10-15_arm_g_v1`) scores DIII-D S 0.979 but MAST R2 psi
0.72, below the frozen-shape floor (0.745), while the MAST shape terms improved. A smooth,
frame-varying flux error outside the plasma model fits that pattern. The solenoid is the
prime suspect: the DIII-D solenoid needed a fitted flat term (F48), and the MAST solenoid
was only checked for gross errors (F53, F72). The question is whether the MAST solenoid flux
of the coil model is off by a constant factor of about 2.

## Rules followed

- No MAST equilibrium label is read. The shots come from the two public test splits, whose
  files carry no label. Every raw column requested is checked against
  `frame_table.is_label_column`. The three labelled MAST sample shots are not opened.
- Inputs, coil geometry, the repository's documentation and published machine numbers only.
- The coil model is not changed. The factor `f` multiplies the MAST solenoid column inside
  the script only.
- DIII-D serves as the calibrated reference. Its adopted coil model (shaping coils at
  scale 1 plus the flat solenoid term, F47 and F48) was fitted on DIII-D training labels,
  which the challenge allows.

## Summary (smoke, to be confirmed by the full run)

1. **Code path.** Each of the 656 MAST solenoid elements carries the full value of
   `magnetics_sol_current` (`solenoid_turns="per_element"`). The README calls that column the
   solenoid "feed" current in kA. If the winding has two conductors in parallel per turn
   (two-in-hand), each element carries half the feed current, and the model overstates the
   solenoid's ampere-turns by exactly 2.
2. **Hardware.** The geometry has 4 layers of 164 conductors (656, no duplicate positions).
   The MAST Upgrade solenoid is documented as 324 turns wound two-in-hand in four layers,
   which is 648 conductors, built by the MAST methods. 656 conductors in four layers is
   consistent with 328 two-in-hand turns. The original MAST turn count was not confirmed
   from a primary source (see Sources).
3. **Flux swing.** Over the data's solenoid range (+46 to -47 kA), the model's flux swing is
   1.38 Wb at R0 and 1.57 Wb at R0 - a0 with `f` = 1. With `f` = 0.5 it is 0.69 and
   0.79 Wb. The published MAST swing is 0.8 to 0.9 V s, which corresponds to a factor of 0.51
   to 0.65. The precharge sits at 43 to 46 kA (5th to 95th percentile), so the observed range
   is the operating range, not a subset of it.
4. **Flux consumption.** The calibrated case uses the total coil flux, the Hirshman-Neilson
   inductance and the start just before breakdown. In that case DIII-D's implied Ejima
   coefficient is 0.52 (inside the typical 0.4 to 0.5 band), and MAST's is 0.97 with `f` = 1.
   The ratio is 1.86 (90 percent bootstrap 1.63 to 2.00). The solenoid factor that gives MAST
   the DIII-D coefficient is 0.55 (shot IQR 0.52 to 0.62). It ranges from 0.52 to 0.58 across
   the Hirshman-Neilson cases (start or 10 percent crossing, R0 or edge mean).
5. **P-coils.** The vertical-field balance at R0 puts the MAST P-coils inside the
   force-balance band at scale 1 (ratio -1.37 against a band of -1.30 to -2.20). That rules
   out the factor 4 to 23 errors that per-turn currents would give, and it is insensitive to
   the solenoid factor. The breakdown field-null fit is inconclusive (section 3).
6. **Impact.** With `f` = 0.5 the MAST map changes by 0.023 Wb/rad rms. Its sum of squares is
   0.19 of the pooled sum of squares of the submitted MAST maps, the same order as the gap
   between R2 psi 0.72 and DIII-D-level values. The arm_g model's inputs (`set_a`) change
   little: the `coil_pt` points shift by a median of 0.02 to 0.10 interquartile ranges.
   DIII-D training data, targets, the scaler and the model weights are untouched. A fix
   needs the MAST coil-flux model, a MAST-only feature rebuild and a re-prediction, and no
   retraining.

The evidence points to a factor of 0.5 on the MAST solenoid, with a likely hardware cause.
The physics checks bound the factor to about 0.45 to 0.70 once the inductance assumptions
are varied, and they exclude 1.0 unless the MAST Ejima coefficient is about twice DIII-D's.

## 1. Units and turns

### Code path

- `machines.MAST.solenoid_turns = "per_element"`. `coil_flux.element_current_fraction`
  returns the scale (1.0) for every element, so each of the 656 `sol_*` elements carries the
  full column value: the solenoid carries 656 times the column value in ampere-turns.
- The MAST P-coils are `"shared"`. The column is the coil's total kA-turn, split equally over
  its elements (P2 20, P3 8, P4 23, P5 23, P6 4).
- MAST has no adopted coil-model file (`coil_flux.ADOPTED_FILES` has DIII-D only), so every
  MAST scale is 1 with no flat term.
- DIII-D for comparison: `magnetics_ECOILA` enters only as the flat term
  0.00897 Wb/rad per kA. The rectangle's scale is 0 (`resources/coil_scales_d3d.json`).

### Documentation

- README, units table: MAST `magnetics_sol_current` is "Central solenoid feed - kA". P-coils
  are "kA-turn (natively)". The README also says the four kA channels "stay in plain kA
  because no trustworthy turn count exists for them". It names `ECOILA` as ambiguous and says
  nothing further about the MAST solenoid.
- README, geometry section: MAST ships "one row per conductor turn, the solenoid alone being
  656 of them", and "all 656 solenoid elements share `magnetics_sol_current`". That treats one
  row as one turn. It is right for the P-coils, where the column is in kA-turn and the split
  is shared. For the solenoid, whose column is a feed current, the README never says whether
  one row is one turn or one conductor of a multi-conductor turn.
- `docs/data_dictionary.md` records the same "656 single-turn elements" reading. It was the
  initial convention of the coil-flux phase (`docs/COIL_FLUX.md`, Deliverables 1) and was
  never tested beyond the gross-error checks of F53.

### Geometry

| Quantity | MAST solenoid |
| --- | --- |
| Elements | 656, no duplicate (R, Z) |
| Layers | 4, at R = 0.1190, 0.1318, 0.1446, 0.1574 m; 164 elements each |
| Axial extent | Z = -1.471 to 1.459 m, pitch 0.0180 m |
| Element | 0.0115 m x 0.017 m |
| Model flux per kA, (R0, 0) | 2.342e-3 Wb/rad |
| Model flux per kA, (R0 - a0, 0) | 2.677e-3 Wb/rad |
| Long-solenoid formula, per kA per element | 2.700e-3 Wb/rad |

The filament model agrees with the long-solenoid formula, so the calculation is correct for
the convention it is given. With no duplicate positions, no element is counted twice. A
two-in-hand winding is invisible in the geometry: its two parallel conductors appear as two
ordinary elements. The question is the current per element, not the element count.

The flux is not flat over the plasma region. At R0 it falls from 2.34e-3 at Z = 0 to
1.94e-3 at Z = 1.0 m and 1.25e-3 at Z = 1.5 m (per kA). A solenoid error therefore changes
the shape of the coil flux, not only its level.

### Published MAST solenoid

| Source | Statement |
| --- | --- |
| MAST Upgrade centre column design and analysis (Fusion Eng. Des. 86, 2011), as summarized by search | Solenoid flux swing raised from about 0.9 V s on MAST to 1.7 V s on MAST-U. MAST-U solenoid: 324 turns of 12 mm x 18 mm copper, wound two-in-hand in four layers |
| SOFT 2010 abstract P2-005 (MAST-U centre column) | Swing 0.9 to 1.7 V s; magnet construction follows the methods used on MAST |
| FreeGS `machine.MASTU` | `Solenoid(0.19475, -1.581, 1.581, 324)`: 324 turns |
| FreeGS `machine.MAST` | `Solenoid(0.15, -1.4, 1.4, 100)`: a placeholder turn count, not usable |
| FreeGSNKE validation on MAST-U (arXiv 2407.12432, section 4.1.1) | Each coil has a "current multiplier factor (used for the solenoid only)". The value is not given in the text |
| MAST start-up study (search summary) | "The total presently available from the solenoid is ~0.8 Vs" |
| Science Museum Group, MAST solenoid half-section (1999) | "4 layers wound around 1000 times" (a museum description, not a specification) |
| Board entry quoted in the brief | A MAST Grad-Shafranov solver needed a "half-current solenoid" |
| Walkthrough log of 12 October, quoting Berkery et al. (2021) | Solenoid swing +22 to -17 kA, against +46 to -47 kA in the data |

The MAST-U figures (324 turns, two-in-hand, 648 conductors, 12 x 18 mm) match the shipped
MAST pattern closely: 656 conductors, 11.5 x 17 mm elements, 4 layers. FreeGSNKE's
solenoid-only multiplier and the board's half-current solenoid both point to one
per-conductor factor. The Berkery range is about half the data's range, consistent with a
per-conductor current, but how that range is defined could not be checked here.

### Flux swing against the published value

| | Current span (1st pct of record minimum to 99th pct of record maximum) | Swing at (R0, 0) | Swing at (R0 - a0, 0) |
| --- | --- | --- | --- |
| `f` = 1 | 93.6 kA | 1.38 Wb | 1.57 Wb |
| `f` = 0.5 | 93.6 kA | 0.69 Wb | 0.79 Wb |
| Published | | 0.8 to 0.9 V s | |

(Smoke. The swing is `2 pi psi` per kA times the span.) The solenoid precharge is at 43.4 to
46.0 kA (5th to 95th percentile) on every smoke shot, and the extremes are +46.3 and
-47.3 kA, which looks like a supply limit. The published swing therefore compares with the
observed span directly. It sits between the two factors, nearer 0.5: the published values
correspond to a factor of 0.51 to 0.65, depending on the published figure and on where the
swing is taken.

### P-coils

The P-coil columns are in kA-turn per the README, and the shared convention divides each
column by its element count. That is right if the column is the coil's total ampere-turns.
If it were a per-turn current (the IMAS `pf_active` convention), the P-coil flux would be 4
to 23 times larger, which the vertical-field balance of section 3 excludes. P2 is parallel-fed
(12 inner and 8 outer elements under one column). Splitting the column equally over 20
elements assumes the same current per turn in both sub-coils, which cannot be checked without
labels. P2 is at Z = +-1.66 to 1.85 m, far from the plasma, so the effect on the plasma region
is second order.

## 2. Flux consumption

### Method

The script works per shot from the raw magnetics (`shot_balance`). The currents are 2 ms
window means, as in the frame table.

- **Start.** The last Ip sample within 2 percent of the peak above the pre-shot baseline,
  before the organizers' 10 percent crossing, less 1 ms. On MAST this is before the
  merging-compression P3 pulse ends. Every smoke MAST shot is a merging-compression start, with
  P3 at about 120 kA-turn just before breakdown. The P3 swing is therefore inside the
  shaping-coil part of the balance.
- **End.** The first time the 2 ms mean of Ip reaches 90 percent of the peak.
- **Consumed flux.** `-2 pi sign(Ip) (psi(end) - psi(start))` from the coil model at (R0, 0),
  and as the mean of (R0 - a0, 0) and (R0 + a0, 0). It is divided by `mu0 R0 Ip(end)` and
  split into a solenoid part `c_sol` and a shaping-coil part `c_pf`.
- **Inductive need.** `ell (1 - Ip(start)/Ip(end))`, with `ell` in units of `mu0 R0`.
  - LA, the brief's formula: `ln(8 R0 / a0) + li/2 - 2`.
  - HN: the Hirshman-Neilson external inductance at nominal elongation 1.8, plus `li/2`.
  - li is 1.0 on DIII-D and 0.7 on MAST.
  - LA gives 1.49 (DIII-D) and 0.70 (MAST); HN gives 1.13 and 0.46.
- **Implied Ejima coefficient.** `C_E = c - need`, on two bases: solenoid only, as the brief
  asks, and total coil flux.
- **Solenoid factor.** Per MAST shot, the `f` that gives MAST the DIII-D median `C_E` of the
  same case: `f* = (need + C_E,DIII-D - c_pf) / c_sol` on the total basis, and
  `(need + C_E,DIII-D) / c_sol` on the solenoid-only basis.

### Results (smoke, 39 DIII-D and 40 MAST shots)

Case: start before breakdown, (R0, 0).

| | DIII-D | MAST, `f` = 1 |
| --- | --- | --- |
| `c_sol` median (IQR) | 1.06 (1.00 to 1.26) | 1.01 (0.93 to 1.14) |
| `c_pf` median | 0.61 | 0.41 |
| Need, LA / HN | 1.47 / 1.12 | 0.70 / 0.47 |
| `C_E` total, LA | 0.16 | 0.73 |
| `C_E` total, HN | **0.52** | **0.97** |
| `C_E` solenoid only, LA / HN | -0.42 / -0.06 | 0.30 / 0.54 |

| Case (start / point / inductance) | `C_E` total ratio MAST / DIII-D (90 pct) | `f*` total, median (IQR) | `f*` solenoid only |
| --- | --- | --- | --- |
| start / R0 / HN | 1.86 (1.63 to 2.00) | 0.55 (0.52 to 0.62) | 0.41 |
| start / edge / HN | 1.64 (1.52 to 1.78) | 0.58 (0.54 to 0.67) | 0.42 |
| t10 / R0 / HN | 1.97 (1.76 to 2.11) | 0.52 (0.48 to 0.61) | 0.38 |
| t10 / edge / HN | 1.71 (1.52 to 1.87) | 0.56 (0.52 to 0.68) | 0.38 |
| start / R0 / LA | 4.45 (3.14 to 5.63) | 0.43 (0.41 to 0.50) | 0.28 |
| start / edge / LA | 2.83 (2.45 to 3.38) | 0.46 (0.43 to 0.53) | 0.29 |
| t10 / R0 / LA | 4.49 (3.50 to 5.55) | 0.41 (0.38 to 0.48) | 0.26 |
| t10 / edge / LA | 2.87 (2.41 to 3.52) | 0.45 (0.41 to 0.55) | 0.27 |

### Reading

- **The total basis is the meaningful one.** On DIII-D the shaping coils supply about 35
  percent of the consumed flux (`c_pf` 0.61 against `c_sol` 1.06). The solenoid-only
  coefficient is then negative on DIII-D, which cannot be physical, so its MAST/DIII-D ratio
  has no meaning. The solenoid-only `f*` (0.26 to 0.42) inherits that bias and is not used.
- **The HN inductance is the calibrated choice.** With it, DIII-D's total-basis `C_E` is
  0.49 to 0.65 across the four HN cases, in the typical band. With the LA formula it is 0.16
  to 0.29. The circular large-aspect formula overstates the inductance of a shaped plasma,
  so the LA ratios (2.8 to 4.5) exaggerate the discrepancy.
- **The reading under `f` = 1.** MAST's coefficient would be about 0.97, twice DIII-D's on
  the same method. Under `f` = 0.5 it is 0.5 x 1.01 + 0.41 - 0.47 = 0.45, matching DIII-D.
- **Sensitivity.** `f*` moves by `delta(need) / c_sol`, about 0.5 per unit of MAST li.
  - MAST li from 0.5 to 1.0 gives `f*` from about 0.45 to 0.70.
  - A lower DIII-D li (0.8) raises DIII-D's `C_E` by 0.1, and `f*` with it, by about 0.1.
  - Reaching `f*` = 1 needs MAST li near 1.6, or a MAST Ejima coefficient twice DIII-D's.
    Neither is a typical value.
- **Not modelled.** Vessel and passive-structure currents at the flat-top start. The flux
  linked at the current centroid rather than at R0 (the edge-mean cases bracket it).
  Merging compression, where the plasma forms around P3 and moves, so the flux balance at R0
  is approximate. The full run splits MAST into merging-compression and solenoid starts
  (`mast_f_star_total [solenoid_start]`).

## 3. Other label-free checks on the P-coil scales

### Vertical field at (R0, 0), flat-top start, over `mu0 Ip / (4 pi R0)` (smoke)

| | Solenoid | Shaping coils | Total | Force balance, large aspect, `beta_p` 0.1 to 1.0 |
| --- | --- | --- | --- | --- |
| DIII-D | 0 (flat term) | -2.01 (IQR -2.13 to -1.94) | -2.01 | -2.09 to -2.99 |
| MAST, `f` = 1 | +0.04 | -1.38 (IQR -1.47 to -1.21) | -1.37 | -1.30 to -2.20 |
| MAST, `f` = 0.5 | +0.02 | -1.38 | -1.36 | -1.30 to -2.20 |

The MAST P-coils at scale 1 give a vertical field inside the force-balance band, at its
low-`beta_p` end. DIII-D, whose shaping coils are confirmed at scale 1 (F47), sits just
outside the same end (-2.01 against -2.09), so the formula reads slightly low on both
machines. A per-turn reading of the P-coil columns (factor 4 to 23)
is excluded. The large-aspect formula is coarse at aspect ratio 1.3, so the check bounds a
common P-coil scale to about +-30 percent and no tighter. The solenoid's stray field at R0
is 3 percent of the total, so this check says nothing about `f`.

### Breakdown field null (MAST, exploratory)

The script fits the `f` that minimizes the coil poloidal field over R 0.30 to 1.30 m,
Z -0.80 to 0.80 m. Two times are used: at the start, and 10 ms earlier, before the P3 pulse.

- At the start (P3 active, merging compression): `f*` median 0.56 (IQR 0.52 to 0.65).
- 10 ms earlier (P3 idle): `f*` median 0.07, with the rms field 436 G at `f` = 1 against
  462 G for the solenoid alone. The P-coils are not nulling the solenoid's stray field at
  that time, so the fit has nothing to fit.

With no solenoid-start shot in the smoke sample, the null test is inconclusive. The
start-time value agrees with 0.5, but on merging-compression shots it is not a designed
null, and it is not taken as evidence.

### Not done

- An up-down current balance (pu / pl) would check the pairs but not their scale.
- A P2 inner/outer split cannot be checked without labels.

## 4. What a factor changes

The factor enters only where the adopted coil model is applied to MAST. DIII-D's coil
flux, the DIII-D plasma targets (`target_build`, `targets.coil_flux_of_frames` with DIII-D
scales), the loss inputs, the feature scaler (fitted on DIII-D training) and the trained
weights are untouched.

### Composed MAST map (`prediction.MapComposer`, adopted MAST scales)

The coil flux added to the model's plasma flux changes by `(f - 1) sol pattern_sol` on every
frame. Smoke values over all 69,075 MAST frames, at `f` = 0.5:

| Quantity | Value |
| --- | --- |
| rms change per element | 0.0235 Wb/rad |
| Sum of squares of the change | 1.61e5 (6.98e4 about its own mean) |
| Pooled sum of squares of the submitted MAST maps (`arm_g_v1`) | 8.47e5 |
| Ratio | 0.19 |

R2 psi pools every finite pixel of every frame (`local_score.py`). A coil-flux error of
this size is therefore worth up to about 0.19 of R2 psi, the same order as the gap between
0.72 and DIII-D-level values. Whether correcting it recovers that much depends on how the
remaining error correlates with it, which only a submission can show. The change is
smooth and nearly flat over the plasma, so the shape terms barely move. That matches the
observed pattern: R2 psi poor while the shape terms improved.

### Features (MAST only; shift in DIII-D scaler interquartile ranges, `f` = 0.5, smoke)

| Feature | In arm_g (`set_a`) | Median abs shift | 95th pct abs shift | MAST scaled median `f` = 1 / 0.5 | DIII-D test scaled median |
| --- | --- | --- | --- | --- | --- |
| `coil_pt_00` to `coil_pt_21` | yes | 0.017 to 0.097 | 0.05 to 0.29 | pointwise; on most points 0.02 to 0.11 further from the DIII-D median | |
| `coil_c10` | no | 0.44 | 1.30 | -1.26 / -1.77 | -0.01 |
| `coil_c02` | no | 0.90 | 2.65 | -2.28 / -3.12 | -0.02 |
| `shaping` (set_b) | no | 0.44 | 1.30 | 1.59 / 1.11 | -0.02 |
| `coil_c30` (set_b) | no | 0.13 | 0.39 | 1.42 / 1.27 | -0.02 |
| `coil_c12` (set_b) | no | 0.33 | 0.96 | 1.76 / 1.49 | 0.01 |
| `shafranov_lambda` (set_b) | no | 0.23 | 0.68 | -0.25 / 0.02 | 0.01 |
| `ohmic_flux_used` (candidate) | no | 1.19 | 2.61 | -0.25 / 0.94 | -0.01 |
| `coil_c01`, `c11`, `c21`, `c03`, `c20` | no | 0.003 to 0.011 | under 0.04 | unchanged | |

The full table, including the share of MAST frames outside the DIII-D test 1st to 99th
percentile before and after, is `impact_features.parquet`.

- **arm_g (set_a).** The only coil-derived inputs are the `coil_pt` points. Their reference
  point (R0, 0) removes the flat part of the solenoid flux, so the shift is small. On the
  points at y = +-1.5 and x = 0.375 (`coil_pt_10`, `13`, `14`, `18`, `19`, `21`) the median
  shift is 0.08 to 0.10 IQR, and the share of MAST frames outside the DIII-D range grows
  (for example from 0.23 to 0.50 on `coil_pt_10`). On `coil_pt_11` and `12` it goes from
  0.64 and 0.78 to 0.91 and 0.98. A solenoid fix is mainly a map fix for
  this model, with a small second-order change in its inputs.
- **set_b models (arm_setb_g).** `shaping`, `shafranov_lambda`, `coil_c12` and `coil_c30`
  move by 0.1 to 0.4 IQR, and move toward DIII-D: `shaping` 1.59 to 1.11,
  `shafranov_lambda` -0.25 to 0.02. Under `f` = 1 the MAST set_b inputs carry an error from
  the solenoid itself.
- **`ohmic_flux_used`.** It is proportional to `f` on MAST and moves by 1.2 IQR. It is not
  in either set.

### What a fix requires (proposal, not implemented)

1. Adopt a MAST coil model with the solenoid at 0.5: a
   `fusion_eq/resources/coil_scales_mast.json` record (`magnetics_sol_current` 0.5, the ten
   P-coils 1.0, no flat term, source naming this check) and an entry in
   `coil_flux.ADOPTED_FILES`.
   - This applies the factor through `adopted_scales`, as on DIII-D. The Green and point
     tables, their hashes and the point-table cache key do not change.
   - Changing `machines.MAST.solenoid_turns` instead would rebuild the MAST Green table and
     change its hash.
   - A data-dictionary entry and a finding would record the convention: one element is one
     conductor carrying half the feed current.
2. Rebuild the MAST features only: `scripts/build_features.py` for `mast_public_test`.
   DIII-D features, the scaler and every DIII-D determinism hash stay as they are. The MAST
   feature hash changes, by design.
3. Re-predict MAST with the existing checkpoints (`scripts/predict.py`), no retraining.
   - The composed maps pick up the new coil flux through `MapComposer`.
   - q95 and the shape scalars follow from the maps.
   - The MAST betaN head is applied to the new shapes. Its stage-2 refit uses DIII-D training
     frames, so it does not change.
4. Optional, cheapest to isolate the effect: compose with the new MAST coil flux while
   keeping the old MAST features. That separates the map effect from the input effect.
   Both variants are submissions, so the choice and the submission budget are Kamia's.

Other consumers of `adopted_scales("MAST")` follow automatically: the submission fallbacks
(`make_submission.py`, `predict.py`), `profile_targets.py`, `worst_mast_shots.py`,
`fit_coil_scales.py` K9 and the predictors. Earlier MAST diagnostics (F53, F72, F100) were
computed under `f` = 1 and would need a note.

The factor 0.5 rests on the winding (two-in-hand) and on DIII-D-calibrated physics, not on
MAST labels or on leaderboard feedback.

## Sources

- MAST Upgrade centre column design and analysis, Fusion Engineering and Design 86 (2011);
  https://www.sciencedirect.com/science/article/abs/pii/S0920379611003875 and
  https://ui.adsabs.harvard.edu/abs/2011FusED..86.1398T/abstract. The page returned 403. The
  figures (324 turns, two-in-hand, four layers, 12 x 18 mm, 0.9 to 1.7 V s) come from search
  summaries and are to be confirmed against the full text.
- SOFT 2010 abstract P2-005: https://www.ipfn.tecnico.ulisboa.pt/soft2010_abstractbook/pdf/P2-005.pdf
- FreeGS machine definitions: https://raw.githubusercontent.com/freegs-plasma/freegs/master/freegs/machine.py
- FreeGSNKE validation against EFIT++ on MAST-U, arXiv 2407.12432: https://arxiv.org/html/2407.12432
- FreeGSNKE repository (multiplier field of active coils):
  https://github.com/FusionComputingLab/freegsnke
- Science Museum Group, half section of the MAST central solenoid:
  https://collection.sciencemuseumgroup.org.uk/objects/co8928682/half-section-of-the-central-solenoid-from-the-mast-nuclear-fusion-experiment
- Wikipedia, Mega Ampere Spherical Tokamak: https://en.wikipedia.org/wiki/Mega_Ampere_Spherical_Tokamak
- S. P. Hirshman and G. H. Neilson, Phys. Fluids 29, 790 (1986): external inductance of a
  shaped plasma (formula in `hirshman_neilson_inductance`).
- R0 and a0: `machines.py` (DIII-D 1.67 / 0.67 m; MAST 0.85 / 0.65 m, Sykes et al. 2001).

## Full-run results

Pending. To be filled from the printed summary and `summary.json`:

- Shot counts, errors, and the MAST start-up classes.
- The section 2 tables for all eight cases, with `f*` for the solenoid-start and
  merging-compression classes separately.
- The section 1 current ranges and swings over all shots.
- The section 3 vertical-field and null results.
- The section 4 tables (unchanged from the smoke run: they already use every MAST frame).
