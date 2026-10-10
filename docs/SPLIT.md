# Split specification: near-repeat shots and the fixed validation split

Specification for the second analysis phase. The repository's data rules and writing
standard apply. Item ids (A5, B8) match the pre-modelling analysis plan. Finding ids (F01
to F44) refer to `docs/findings.md`.

## Purpose

Fix one split of the DIII-D training shots into training, validation and local test
parts, by whole shot, and save it. Every later analysis and every model uses this split,
so the way it is drawn is decided from evidence first.

The phase settles four things:

1. Whether DIII-D test inputs are distributed like training inputs, beyond the hardware
   markers already compared (B8, F08).
2. Whether near-repeat shots exist, how many, and whether test shots have near-repeats
   in training as often as training shots do (B8, F09).
3. How much of a locally measured score comes from near-repeat shots sitting on both
   sides of a split (B8).
4. The split itself, and the working subset used for fast iteration (A5).

Out of scope: the baseline diagnostics (C1 to C9), the label-quality flag (B5), any frame
filter, and any model other than the plain PCA and ridge fit used for item 3.

## Context

- Shot ids are anonymized codes, so shots cannot be put in date order (F03). Similarity
  between shots is measured from their inputs instead.
- DIII-D test shots match training shots on every hardware marker examined: recording
  rate, Thomson layouts, frames per shot, Ip direction, Thomson coverage (F08).
- Several short shots share nearly the same EFIT time window across training and test.
  This is the repeat-shot hypothesis, not yet tested (F09).
- The frame table (`../data/processed/frame_table`, 9,121 shots, 1,821,774 frames) holds
  every input on frame times. Current signals are read in the windowed-mean form
  (`__mean`, F28). Flux maps are read with `fusion_eq.frame_table.read_flux_maps`.
- Counts: 7,041 DIII-D training shots (1,559,340 frames), 874 DIII-D test shots (193,359
  frames).
- The scorer is used through `fusion_eq/starter_kit.py`.

### Two readings of a validation score

If the organizers drew the test shots at random from one pool, a near-repeat of a test
shot is as likely to be in training as a near-repeat of a validation shot is under a
random split. A random split by shot then represents the leaderboard. A split that keeps
near-repeats together is the stricter reading: it measures accuracy on plasmas the model
has not seen a copy of, which is closer to what transfer to MAST demands. Both readings
are wanted, and the analyses below establish whether they differ.

## Shot similarity

Similarity is computed from inputs only, so it is defined for test shots as well.

1. Signals: the 21 DIII-D current signals in the frame table (`ip__mean`, `sol__mean`,
   `tf__mean` and the 18 shaping-coil `__mean` columns).
2. Fingerprint: each signal's value at fixed times, every 100 ms from 100 ms to
   9,000 ms, by linear interpolation in time between frames. Times outside the shot's
   EFIT window, or inside a gap between frames longer than 200 ms, are NaN.
3. Scale: each signal is divided by its interquartile range over DIII-D training frames.
   For the toroidal field current, which is constant within a shot and takes both signs
   across shots, the scale is the interquartile range of its absolute value. Scales come
   from training frames only.
4. Distance between two shots: the root mean square of the scaled differences over the
   fingerprint entries present in both. It is defined when the two shots share at least
   5 grid times and at least half of the grid times of the shorter shot; otherwise the
   pair has no distance. A distance of 0.1 means the currents differ by a tenth of each
   signal's typical range, over the time the two shots overlap.
5. Near-repeat: a pair of shots with a distance below a threshold. The threshold is set
   in review from the distribution of nearest-neighbour distances and from plots of
   paired shots. Counts are reported at 0.05, 0.10 and 0.20; 0.10 is the provisional
   value.
6. Groups: agglomerative clustering with complete linkage at the threshold, so that
   every pair inside a group is a near-repeat pair and long chains cannot form.

Shots with fewer than 5 grid times inside their EFIT window cannot be fingerprinted.
They are counted and each forms its own group.

A shot with the plasma current reversed is far from its mirror under this measure. That
is intended here: the measure describes raw inputs, as a model without sign handling sees
them.

## Deliverables

1. Package functions with tests. The layout is a proposal, agreed before implementation.
   - `fusion_eq/similarity.py`: `shot_fingerprints(frames, signals, times_ms)`;
     `pairwise_distance(fingerprints, scales, min_common)`, which handles NaN entries
     with matrix operations and does not loop over pairs; `nearest_neighbours(distance,
     k)`; `group_shots(distance, threshold)`.
   - `fusion_eq/split.py`: `draw_split(shot_table, sizes, seed, groups=None)`;
     `read_split(path)`; `check_split(split, shot_index)`, which raises if a shot is in
     two parts, a training shot is unassigned, or a test shot is present.
   - `fusion_eq/baseline.py`: `fit_pca_ridge(train_shots, params)`, `predict(model,
     shots)` and `score_local(predictions, shots)`. This is the plain baseline, built
     here because item 3 needs it and reused unchanged by the baseline phase (C1, C2).
2. `scripts/find_near_repeats.py`: fingerprints, distances, neighbours and groups for
   all 7,915 DIII-D shots. Writes to `reports/split/`:
   - `shot_neighbours.parquet`: one row per shot with its split, its 10 nearest
     neighbours (id, split, distance) and the number of grid times in its window. The
     threshold can then be varied in the notebook without a rerun.
   - `shot_groups.parquet`: group id per shot at each of the three thresholds.
   - `split_report.md`: the acceptance checks below.
3. `scripts/make_split.py`: arguments `--mode` (`random` or `grouped`), `--threshold`,
   `--seed`, `--n-validation` (default 870), `--n-local-test` (default 350),
   `--n-working-train` (default 300), `--n-working-validation` (default 100), `--out`
   (default `splits/split_v1.json`). Appends its checks to `split_report.md`.
4. `scripts/leak_test.py`: the fits and scores described under B8c. Arguments `--split`
   (the split file to test), `--threshold`, `--working` (use the working subsets) and
   `--val-frame-stride`. Writes `reports/split/leak_test.parquet` and a section of
   `split_report.md`.
5. `splits/split_v1.json`, committed: the lists `train`, `validation`, `local_test`,
   `working_train` and `working_validation`; the mode, threshold and seed; the
   repository commit; and for every shot its group id and whether it has a near-repeat
   in the `train` list.
6. `notebooks/04_split.ipynb`: the review notebook described below.
7. `tests/test_similarity.py`, `tests/test_split.py`, `tests/test_baseline.py`.

## Analyses

| Id | Analysis | Method | Decision it feeds |
| --- | --- | --- | --- |
| B8a | Test inputs against training inputs | Per shot: the median over flat-top frames (`ip_over_peak` above 0.9) of each of the 21 current signals and of `ts_core_te_peak__interp`, `ts_core_ne_peak__interp` and `ts_core_pe_mean__interp`. Per quantity: the two-sample Kolmogorov-Smirnov statistic between training and test shots, and the difference in medians in units of the training interquartile range. | Whether any input is distributed differently on the test shots |
| B8b | Near-repeat shots | Fingerprints and distances as defined above. Nearest-neighbour distance for every training shot among the other training shots, and for every test shot among the training shots; the two distributions compared. Share of shots with a near-repeat and the distribution of group sizes, at the three thresholds. | Whether test shots are a random draw from the same pool as training shots, and how common near-repeats are |
| B8c | Leak test | With a provisional random split drawn first, the PCA and ridge baseline is fitted three ways and scored on the same validation shots: (A) on the whole training part; (B) on the training part with every near-repeat of a validation shot removed; (C) on the training part with the same number of shots removed at random, three seeds. Scores are reported for all validation shots, for those with a near-repeat in training and for those without. | How much of a local score depends on near-repeats. A minus B is the leak; the spread of C is the noise level. |
| A5 | The split | Drawn by `make_split.py` in the chosen mode, stratified by recording rate and Ip direction so that each part mirrors the test shots (F08). In `grouped` mode whole groups are assigned, so no group spans two parts. The working subsets are drawn inside `train` and `validation`. | One split used by every later phase |

### The baseline used in B8c

The starter kit's PCA and ridge baseline is reviewed first. Where its choices differ
from those below, the starter kit's are used, so that the result is comparable with the
published baseline, and the difference is recorded.

- Inputs: the 21 current signals in `__mean` form and the three core Thomson summaries
  in `__interp` form. Missing Thomson is set to the training median, with
  `ts_core_valid` added as a 0 or 1 column. Inputs are standardized with training
  statistics.
- Targets: 50 principal components of the flux map, and `efit_q95` and `efit_beta_n`.
  The PCA basis and mean are fitted once, on a fixed-seed sample of 100,000 frames from
  the training part, and shared by the three fits. The arms then differ only in the rows
  given to the ridge. The effect of the removed shots on a 50-component basis is taken
  as negligible; this is an assumption of the test.
- Regularization: one fixed value, not tuned in this phase.
- No frame filter. The same validation frames are scored in every arm.
- The full set of flux maps is never held in memory (about 26 GB as float32).
  Coefficients are computed shot by shot and cached under
  `../data/processed/baseline/` for reuse by the baseline phase.
- Scoring uses the challenge scorer and reports S and its four terms.

A smoke run on the working subset records the runtime of the fit and of the scorer. If
scoring all validation frames is slow, a frame stride is applied to the validation
frames, the same in every arm, and recorded.

## Acceptance checks

Written to `split_report.md` with status `pass`, `fail` or `review`.

| Id | Check | Status rule |
| --- | --- | --- |
| S1 | Fingerprint coverage | Review. Shots fingerprinted and shots too short, per split |
| S2 | Input distributions | Review. Kolmogorov-Smirnov statistic and median difference per quantity, largest first |
| S3 | Nearest-neighbour distance | Review. Quantiles of the distance for training shots among training shots and for test shots among training shots, and the Kolmogorov-Smirnov statistic between the two |
| S4 | Near-repeat prevalence | Review. At each threshold: share of training shots with a near-repeat in training, share of test shots with one in training, number of groups, largest group |
| S5 | Split integrity | Pass if every training shot is in exactly one of `train`, `validation` and `local_test`; no test shot appears; the counts match the arguments; the working subsets lie inside their parent lists; and in `grouped` mode no group spans two parts |
| S6 | Split balance | Review. Per part and for the test shots: share recorded at 20 kHz, share with negative Ip, median frames per shot, share with no valid core Thomson profile |
| S7 | Leak test | Review. S and its four terms per arm and per subset of validation shots; A minus B against the spread of C |

## Review notebook

`notebooks/04_split.ipynb`. Sections:

1. Input distributions, training against test (B8a): the quantities with the largest
   differences, drawn as overlaid histograms.
2. Nearest-neighbour distances (B8b): the two distributions on a logarithmic axis, with
   the three thresholds marked.
3. What a near-repeat looks like: for pairs at distances near 0.02, 0.05, 0.10, 0.20
   and 0.50, the plasma current and three shaping-coil currents of both shots through
   time. For training pairs, `efit_q95` and `efit_beta_n` through time as well. The
   threshold is chosen here.
4. Groups: sizes at the chosen threshold; the share of test shots with a near-repeat in
   training against the share for training shots.
5. Leak test (B8c): the scores by arm and subset.
6. The split: counts and balance against the test shots (S5, S6).
7. Findings: the threshold, the mode, and what a validation score represents.

## Decisions this phase feeds

| Decision | Options | Evidence |
| --- | --- | --- |
| Near-repeat threshold | A distance read off the distribution and the paired plots | Notebook sections 2 and 3 |
| Whether validation represents the leaderboard | Yes, if test shots have near-repeats in training as often as training shots do; otherwise the difference is stated | S3, S4 |
| Mode of the split | Random by shot, with the group recorded, so that scores are reported for all validation shots (the leaderboard reading) and for those without a near-repeat in training (the stricter reading); or grouped, which gives the stricter reading only | S3, S4, S7 |
| Any input needing attention | Per quantity that differs between training and test | S2 |

The random mode with groups recorded is preferred when the test shots have near-repeats
in training about as often as training shots do and at least 200 validation shots have
no near-repeat in training, since it gives both readings from one split. The leak is
treated as material when A minus B exceeds three times the spread of the C arms and
0.002 in S. Both thresholds are provisional and are revisited when the numbers are in.

## Tests

On small synthetic tables:

- Two shots built from the same signals have distance 0; scaling one signal's difference
  by its scale gives the expected distance.
- A shot and a truncated copy of it have distance 0 over their overlap, and no distance
  when the overlap is below the minimum.
- `pairwise_distance` agrees with a direct loop over pairs on a small set containing NaN.
- Complete linkage: three shots where A is near B and B is near C but A is far from C
  do not form one group.
- `draw_split` is reproducible for a seed, returns the requested counts, and in grouped
  mode never divides a group.
- `check_split` raises on a duplicated shot, an unassigned shot and a test shot.
- `fit_pca_ridge` recovers a synthetic linear map, and `predict` returns one map and two
  scalars per frame in frame-table order.
- A perfect prediction scores 1 through `score_local`.

## Procedure

1. Review the starter kit's ridge baseline and the scorer's local entry point. Agree the
   module layout.
2. Implement the functions and tests. Run the tests.
3. Near-repeats, launched manually: `python scripts/find_near_repeats.py`.
4. Review notebook sections 1 to 4. The threshold is fixed here.
5. Provisional split, launched manually:
   `python scripts/make_split.py --mode random --out reports/split/split_provisional.json`.
6. Leak test, launched manually: first with `--working` to record the runtime, then in
   full: `python scripts/leak_test.py --split reports/split/split_provisional.json`.
7. Review notebook section 5 and choose the mode.
8. Final split, launched manually: `python scripts/make_split.py --mode <mode>`. With
   the same seed, the random mode reproduces the provisional split.
9. Review notebook sections 6 and 7. Add the findings to `docs/findings.md`.

## Constraints

- Nothing is written under `../data/raw/`.
- Test shots are compared and never fitted: scales come from training frames, and no
  test shot enters the split file or any fit.
- The split assigns all 7,041 training shots. Shots and frames that are candidates for
  exclusion (F21, F44, open item 8) are handled by training filters in later phases, on
  top of the split.
- The split is by whole shot in every mode.
- Once `splits/split_v1.json` is committed it is not redrawn. A change of split is a
  new file with a new version number.
- Every run against the data is launched manually.
- New packages are added to `requirements.txt` the same day they are installed.
