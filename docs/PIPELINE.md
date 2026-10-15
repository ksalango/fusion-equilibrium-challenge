# Pipeline specification: model, training and prediction

Specification for the model pipeline phase. The repository's data rules and writing
standard apply. Finding ids (F01 to F100) refer to `docs/findings.md`; decision, proposal
and reading ids (D, P, R) to `claude/model-walkthrough-log-2026-10-12.md`. Check ids in
this document are Z1 to Z14.

## Purpose

Build the plasma model and everything around it: the loader, the config-driven loss, the
training and evaluation harness, and the end-to-end prediction path that turns stored
tables or raw shot files into a scored submission on either machine. The feature tables
(version 2), the target tables and fitted objects (version 1) and the coil flux are
consumed, not rebuilt.

The phase settles six things, by training runs judged with the organizers' scorer on
validation and by submission:

1. What a network adds over the linear path on target B (arm F against the warm-start
   ridge; R1 is the untested reading this answers).
2. What the geometry loss terms add (arm G against arm F; P1).
3. How much of the MAST floor (S 0.483, R2 psi 0.745, D_LCFS 0.263 m; F100) the planned
   inputs close (R32's test).
4. Whether set_b with `shaping` and `shafranov_lambda` transfers better than set_a (F98's
   open reading), by submission.
5. Whether the shot-position features help or hurt transfer (D8), by submission.
6. What sequence information adds (rung 2 against rung 1).

Out of scope: the feature and target builds, a physics model of the MAST remainder,
synthetic MAST equilibria, changes to the scalar-head study (its result is implemented as
decided), and any use of MAST equilibrium labels.

## Decisions this phase implements

From the walkthrough log (D1 to D14, the independent review) and the decisions of
14 October (D15 to D19, logged in this phase's consolidation step):

- **Loss target (D1).** The flux of the predicted current is compared with the flux of
  the true current inside the scorer's boundary of the true map, never with the full map
  and never with the remainder. Masks and weights come from the true map (P3).
- **Normalization (D2) and registry (D3).** Every loss term is normalized by its value
  for the mean-current prediction; terms are registered and switched by configuration.
- **Arms and terms (D16).** Two loss configurations: arm F, the flux term alone; arm G,
  the flux term plus a gradient term and a boundary-band term. The contour term, the li
  term and the component-handle terms are registered and off (the last pending the F96
  test). The warm start of the linear path (P2) is on in both arms.
- **Frame weighting (D17).** The loss is computed in flux units, so frames carry the
  natural weight of their plasma flux (which scales with Ip). No per-frame unitless
  normalization (R25). A frame-weight exponent and per-phase multipliers exist as
  configuration, at their neutral defaults. Reopening condition: if the per-phase
  evaluation of the first runs shows consistency lagging on late or low-Ip frames while
  flat-top holds, the exponent is the first configuration rerun.
- **Rung 2 (D18, item A, P5).** A sequence model (GRU default, temporal convolution as a
  configuration option) over a shot's frames, causal by default, with the inter-frame
  time step as an input. Built in this phase, staged after the rung-1 path so it cannot
  delay the first submission.
- **Working rules (D19).** No auxiliary shape outputs in version 1 (the F96 correlation
  test runs instead); input clipping on at a generous bound (P11) with the bounded path
  registered and off (item 4); random Thomson masking on at a low rate (item 6); early
  stopping and Optuna select on the no-near-repeat validation subset (item 7); weight
  decay and dropout as the regularizers, with the MAST-like corner slice reported at
  every selection (item 8); negative-Ip shots stay in training and are a reported slice
  (item 9); three seeds per arm, mirror averaging on by default at prediction, seed
  averaging as an option (item 10).
- **Scalars (D15, F99, F74).** q95 is calculated from the predicted map (no fitted
  constant; N_tf 144 and 24). betaN takes one head per machine: on DIII-D the ridge on
  the raw inputs; on MAST the geometry-explicit formula times the gradient-boosted
  correction on the transferable inputs. Before any submission that uses the MAST head,
  stage 2 is rerun with the shapes of the maps that submission carries.
- **Determinism and raw-file prediction (D7).** Prediction regenerates features from raw
  shot files when asked, and a clean rebuild on a subset is byte-identical.
- **Standing decisions.** Mirroring on in training, on the fly in the loader (decision 4
  of 12 October); the lighter label filter; the stored DIII-D scaler applied unchanged to
  MAST (scaling decision of 12 October); the MAST sign constant +1 asserted on every MAST
  prediction (F71, F100); float16 submissions (F69); run comparisons judged by the
  organizers' scorer on validation, reported for all shots and for shots without a
  near-repeat (F57, F58).

One refinement of D1's letter, flagged in the walkthrough before this document was
written: the default loss reference is the flux of the true current's 50-component form
(`Phi c_true`), not the exact `Ip G s`. The two differ by the basis truncation (median
3.3 percent of the share, F92/Y6), which no choice of c can reduce, so training against
the exact flux adds a constant noise floor and an 11 GB per-frame flux cache. The exact
reference remains available as `loss.reference: exact` with its own build step. Masks and
band weights still come from the true map.

## Context

- Inputs: the version 2 feature tables and the stored scaler
  (`../data/processed/features/v2`, `reports/features/v2/scaler.json`); the version 1
  target tables and fitted objects (`../data/processed/targets/v1`: current basis with
  parity vector and component variances, flux bases `phi_mean_m` and `Phi_m` per machine,
  placement, remainder model, per-frame shape numbers, boundary and filter flags); the
  frame table; the coil flux code with the adopted coil models; the scorer and its cached
  references; `splits/split_v1.json`; `fusion_eq/machines.py`.
- Existing code reused, not rewritten: `compose_map`, `q95_from_map`, the predictor
  registry and submission writer (`baseline_v0`, `coil_only_v0`, `mean_current_v0`), the
  lighter-label-filter function, `mirror_features`, `mirror_coeffs`, `mirror_labels`.
- The betaN head study's code (`scripts/betan_heads.py`) is the basis of the scalar
  heads; this phase productionizes the two chosen heads and the stage-2 refit.
- Development runs on the working lists; full training runs on RunPod, launched by Kamia.

## Definitions

Units as in `docs/TARGET.md`. Subscript m is the machine; f indexes frames.

| Symbol | Definition |
| --- | --- |
| x | The scaled feature vector of the configured set (set_a, 29, default) |
| c | The 50 current coefficients of the target build, standardized by the stored component standard deviations. The model's output space |
| c_hat | The model's prediction of c |
| psi_hat_pl | `Ip (phi_mean_m + Phi_m c_hat)` after unstandardizing, the flux of the predicted current |
| psi_ref | The loss reference: `Ip (phi_mean_m + Phi_m c_true)` by default; `Ip G s` under `loss.reference: exact` |
| M_f | Inside mask of frame f: the pixels inside the scorer's boundary of the true map |
| B_f | Band mask of frame f: the pixels within `loss.band_cells` grid cells of that boundary, both sides |
| w_f | Frame weight: `(abs(Ip_f) / Ip_med) ** (gamma - 2)` times the phase multiplier; gamma 2 and multipliers 1 are the neutral defaults (the flux-unit loss then carries its natural Ip-squared weight) |

The gradient of the plasma-flux error equals the gradient of the full-map error, because
the true coil flux is common to both sides of the comparison; the gradient and band terms
therefore act on the quantities the scorer derives from the full map (checked by Z3).

## Part 1: loss module

A registry of terms. Each term has a configuration entry with `enabled`, `weight` and a
normalization constant computed once on a fixed sample of 10,000 filter-passing training
frames as the term's value for the mean-current prediction (c_hat = 0), stored in the run
record. The total loss is the weighted sum of normalized terms times w_f, averaged over
the batch.

| Term | Definition | Default |
| --- | --- | --- |
| `flux` | Mean squared error of psi_hat_pl against psi_ref over M_f | On |
| `grad` | Mean squared error of the finite-difference gradient (central differences on the native grid) of psi_hat_pl against psi_ref, over M_f | Off; on in arm G |
| `band` | The flux error over B_f | Off; on in arm G |
| `contour` | Distance between the scorer's boundary of the composed predicted map and of the true map | Registered, off |
| `li_term` | Error in li derived from the composed maps with the scorer's functions | Registered, off |
| `coeff` | Weighted squared error on named components of c (the F96 handles) | Registered, off |

Loss inputs (P3). A build step `scripts/build_loss_inputs.py` writes, per labelled frame:
the packed inside mask, the packed band mask, and (only under `loss.reference: exact`)
psi_ref in float16. Masks derive from the cached scorer boundaries of the true maps, the
same boundaries the target build masked with; the stored `efit_lcfs` columns are not used
(F23). The packed masks are about 1 GB for all lists; the exact-reference cache about
11 GB and not built by default.

Terms that compose the full map (`contour`, `li_term`) are evaluated through
`compose_map` with the frame's true coil flux and, on DIII-D, the ridge remainder; they
are differentiable only where their scorer functions are, and run at a configured stride
if enabled. They are not needed for arms F and G.

## Part 2: loader

`make_loader(lists, set_name, config, mode)` over the feature and target tables.

- Frames: those passing the lighter label filter (the stored flag).
- Mirror (on): each sampled frame is flipped with probability 0.5 by `mirror_features`
  and `mirror_coeffs` (c times p), with the masks flipped in the Z index. Exact by
  construction (F80, review of 14 October).
- Thomson masking (on, rate 0.03): the frame's Thomson features are set to their
  scaled-missing value and `ts_valid` to 0.
- Input clipping (on): scaled features clipped to `loader.clip_bound` (default 10);
  the clip rate per feature and machine is reported by the evaluation.
- Tail weights (off): optional sample weights emphasizing strong shaping and high
  `ts_beta_p` DIII-D frames, from DIII-D training statistics only (P12).
- Batching: by frame for rung 1 (large batches; the data fits in device memory); by shot
  with padding for rung 2, each frame carrying the time step to the previous frame as an
  additional input.

## Part 3: models

Registered model classes sharing one interface: x in, standardized c_hat out.

**Rung 1, `linear_mlp`.** A linear path plus an MLP, summed:
`c_hat = W x + b + MLP(x)`. The warm start (P2) fits a ridge from the scaled features to
the standardized c on the training frames before training, initializes W and b with it,
and records its own validation score through the scorer (the linear check on target B,
beside F98's 0.9050 on the working lists). MLP defaults: 3 hidden layers of 512, GELU,
dropout 0.1, zero-initialized output layer so training starts at the ridge. The bounded
variant (a tanh saturation on the MLP output, item 4) is a registered option, off.

**Rung 2, `gru_seq`.** A GRU over a shot's frames (hidden 256, 2 layers, causal), the
per-frame input x plus the time step, a linear skip per frame warm-started as in rung 1,
and the same output head. Bidirectional is a configuration option, off; if ever enabled
it is disclosed in the run record (the `uses_future` convention). `tcn_seq` (temporal
convolution) is the registered alternative.

## Part 4: training

`scripts/train.py --config configs/model/<arm>.yaml [--lists working]`.

- AdamW, cosine schedule with warmup, float32, gradient clipping. Batch size, learning
  rate, depth, width, dropout and weight decay in the configuration with the defaults
  above.
- Early stopping: validation loss on the no-near-repeat subset of the validation list,
  patience 15 epochs, maximum epochs bounded; best and last checkpoints saved outside
  git.
- Seeds: `train.seed` covers initialization, shuffling, dropout and Thomson masking.
- The organizers' scorer runs at the best checkpoint (full validation at stride 4), not
  every epoch.
- Optuna harness: `scripts/optuna_study.py --config ... --trials N`, trials on the
  working lists, objective = the scorer on working validation with the scalar term held
  at the baseline value (so the objective moves with the map alone), study stored under
  `reports/optuna/<study>.db`, every trial's parameters and slice scores logged.
  `local_test` is read by nothing in this phase until the final pre-submission check.
- Run record `reports/runs/<run_id>/run.json`: configuration and its hash, git commit,
  data build hashes (frame table, features v2, targets v1, loss inputs), seed, device,
  wall time per stage, environment freeze, early-stop epoch, normalization constants,
  warm-start score. Metrics per epoch (each term, train and validation) in
  `metrics.csv`.

## Part 5: prediction

`scripts/predict.py --predictor <name> --machines d3d mast --source tables|raw`, ending
in the existing submission writer.

1. Features: from the stored tables, or, under `--source raw`, rebuilt from raw shot
   files through the frame-table and feature code with the stored scaler (D7). The two
   sources agree byte-identically on the determinism subset (Z5).
2. Model forward: c_hat per frame. Mirror averaging (on): the prediction is averaged
   with the flipped-back prediction of the mirrored features. Seed averaging
   (`predict.seed_average`, off): c_hat averaged over the arm's seeds.
3. Map: c_hat to the share, placed on the machine's grid, to flux through `Phi_m`; the
   remainder from the stored ridge on DIII-D and zero on MAST; `compose_map` with the
   machine's coil flux and sign constant. The MAST sign constant +1 is asserted (F71).
4. Scalars: `q95_from_map` on every predicted map. betaN by the per-machine heads (F99);
   before any submission using the MAST head, the stage-2 refit runs: the model predicts
   the training list (stride 4), shapes are derived from those maps with the scorer's
   functions, the gradient-boosted correction is refitted on them, and the refitted head
   is stored with the submission's run record.
5. Output: float16 maps and the scalars in the submission format, plus `run.json` with
   the model checkpoint hash, the head versions and the local validation score.

Predictors registered: `rung1_v1` (arm and seed set by configuration), later `rung2_v1`.

## Part 6: evaluation

`scripts/evaluate.py --run <run_id>` writes `reports/runs/<run_id>/eval.md`:

- The organizers' scorer on composed validation maps: every term (R2 psi pooled and
  per-shot mean, scalars, D_LCFS, each consistency term), by phase, for all shots and
  the no-near-repeat subset, the negative-Ip slice, and the MAST-like corner slice
  (F99's proxy).
- The score identity beside the terms: S = 0.55 R2 psi + 0.15 scalars +
  0.10 (1 - D_LCFS) + 0.20 consistency, negative terms clipped (a reading, reproduced on
  both scored submissions), with the MAST floor of F100 as the comparison line in every
  report.
- Worst 20 validation shots by R2 psi, with shot ids and D_LCFS.
- Overfit panel: train against validation loss at the stopping epoch, the all-shots
  against no-near-repeat gap beside the ridge's (F58), and the seed spread of S.
- Clip rates per feature and machine; the warm-start score beside the trained score.
- Grouped permutation importance on DIII-D validation per score term, and the label-free
  MAST sensitivity check: each feature group permuted on MAST test inputs, reporting the
  change in predicted maps and in the derived boundary and axis, with no label read
  (P15).

## Part 7: run matrix and configurations

`configs/model/base.yaml` holds the defaults; one small file per arm overrides it. The
matrix, in launch priority:

| # | Arm | Override | Seeds | When |
| --- | --- | --- | --- | --- |
| 1 | F | loss: flux only | 3 | First night |
| 2 | G | loss: flux + grad + band | 3 | First night |
| 3 | set_b | As the F/G winner, set_b | 2 | First night if built in time |
| 4 | rung 2 | `gru_seq`, F and G | 2 each | After stage 4, no gate on the first submission |
| 5 | D8 | Winner, shot-position group out | 2 | Day 2 |
| 6 | P12 | Winner, tail weights on | 2 | Day 2 to 3 if time |

Free comparisons without training: the warm-start ridge, mirror averaging on against
off, seed averaging on against off. Submission budget: 5 per day to 17 October 5 PM PDT;
the F/G winner is the first model submission; D8 and set_b are submitted for their MAST
scores. Challenge 2 counts entries with DIII-D S at 0.85 or above.

## Configuration

`configs/model/base.yaml`, principal fields and defaults:

| Field | Default |
| --- | --- |
| `features.set` | `set_a` (v2) |
| `loader.mirror`, `loader.thomson_mask_rate` | true, 0.03 |
| `loader.clip_bound`, `loader.tail_weights` | 10, false |
| `loss.reference` | `basis` (`exact` optional) |
| `loss.terms.flux` / `grad` / `band` / `contour` / `li_term` / `coeff` | on / off / off / off / off / off; weight 1 each |
| `loss.band_cells` | 2 |
| `loss.frame_gamma`, `loss.phase_weights` | 2, [1, 1, 1] |
| `model.name` | `linear_mlp` (hidden [512, 512, 512], gelu, dropout 0.1) |
| `model.warm_start`, `model.bounded` | true, false |
| `rung2` | `gru_seq`, hidden 256, layers 2, causal |
| `train.seed`, `train.batch_frames`, `train.max_epochs`, `train.patience` | 0, 16384, 200, 15 |
| `train.lr`, `train.weight_decay` | 3e-4, 1e-4 |
| `predict.mirror_average`, `predict.seed_average` | true, false |
| `scalars` | q95 from map; betaN d3d `ridge_raw`, mast `formula_gbm` with stage-2 refit |

Machine constants are read from `fusion_eq/machines.py` and written to every run record
with their verified status.

## Deliverables

1. Package code with tests. The layout is a proposal reported with stage 1; departures
   from the existing modules are flagged, not silent.
   - The loss registry, its terms and normalization; `build_loss_inputs`.
   - `make_loader` with mirror, masking, clipping, tail weights and both batching modes.
   - `linear_mlp`, `gru_seq`, `tcn_seq` and the warm start.
   - The trainer, early stopping, run records and the Optuna harness.
   - The prediction chain with mirror and seed averaging, the stage-2 betaN refit, and
     the `rung1_v1` / `rung2_v1` predictors in the existing registry.
   - The evaluation report.
2. Scripts: `build_loss_inputs.py`, `train.py`, `evaluate.py`, `predict.py`,
   `optuna_study.py`.
3. `configs/model/base.yaml` and the arm files of Part 7.
4. `notebooks/08_model.ipynb`: the review notebook below.
5. Tests for every module added.
6. A short RunPod procedure note (`docs/RUNPOD.md`): clone, environment from the pinned
   requirements, one-time copy of `../data/processed` (and loss inputs) to the pod or a
   network volume, runs writing `reports/runs/`, results pulled back by rsync and
   committed from the local machine. Pods never commit.

## Checks

Written to each run's `eval.md`, or to the stage test output where marked (T).

| Id | Check | Status rule |
| --- | --- | --- |
| Z1 | Loss bookkeeping (T) | Pass if every enabled term is at most 1e-10 of its normalization for c_hat = c_true, and equals 1.0 for the mean-current prediction after normalization |
| Z2 | Mirror equivariance (T) | Pass if every term is unchanged to 1e-6 when the frame and the prediction are mirrored together, and the loader's double mirror is the identity |
| Z3 | Gradient identity (T) | Pass if, on 100 frames, the gradient of the plasma-flux error equals the gradient of the full-map error to 1e-10 |
| Z4 | Warm start | Review. The ridge path's validation score through the scorer, beside F98's linear check; the trained model at or above it |
| Z5 | Determinism | Pass if prediction from a saved checkpoint is byte-identical across two runs, and `--source raw` matches `--source tables` byte-identically on the D7 subset |
| Z6 | Smoke run | Review. On the working lists: training loss decreases, early stopping engages, evaluation and prediction complete, every report section present |
| Z7 | Slices present | Pass if every evaluation carries the phase, negative-Ip, no-near-repeat and corner slices, the worst-20 table, the overfit panel and the MAST floor line |
| Z8 | Clip rates | Review. Share of clipped values per feature and machine |
| Z9 | Rung 2 batching (T) | Pass if padded batching equals a per-shot loop to 1e-6, and a causal model's output at frame t is unchanged by any change to later frames |
| Z10 | Stage-2 refit | Review. The MAST betaN head before and after the refit on the submission model's shapes (validation R2 and the corner), stored with the submission record |
| Z11 | Mirror averaging | Pass if the averaged prediction of an exactly symmetric synthetic frame equals the single prediction; review: on against off on validation |
| Z12 | MAST sign | Pass if every MAST prediction run asserts the +1 constant before writing |
| Z13 | Run record | Pass if the run record carries the configuration hash, commit, data build hashes, seed, environment and normalization constants |
| Z14 | Optuna harness (T) | Pass if a 2-trial study on the working lists completes, logs both trials and leaves `local_test` unread |

## Review notebook

`notebooks/08_model.ipynb`:

1. Run summary and the check table for the first arms.
2. The warm-start ridge against the trained model: score chain (ceiling, ridge, F, G).
3. Loss curves per term; the overfit panel.
4. Predicted against true maps and boundaries on example validation frames, best and
   worst.
5. Per-term, per-phase score tables; the slices; F against G.
6. Permutation importance and the MAST sensitivity check.
7. MAST predictions: example maps, boundary statistics beside F100's, the floor line.
8. Findings.

## Decisions this phase feeds

| Decision | Evidence |
| --- | --- |
| Whether the geometry terms stay on (and their weights) | Z6 runs of F and G through the scorer |
| The frame-weight exponent (D17's reopening condition) | The per-phase consistency slices |
| set_a against set_b; shot-position in or out | The MAST scores of arms 3 and 5 |
| Whether rung 2 replaces or joins rung 1 | Arm 4 against the winner |
| Whether the contour, li or coeff terms are tried | F against G; the F96 correlation test |
| Whether a rough MAST remainder moves up the list (R32) | The first model submission's MAST R2 psi |

## Tests

On synthetic inputs (no real data read by the test suite):

- A planted linear relation from features to coefficients is recovered by the warm-start
  ridge and preserved by a zero-initialized MLP at step 0.
- The loss terms on a synthetic basis and known maps: Z1 to Z3; the frame-weight
  exponent reproduces hand-computed weights; a disabled term contributes nothing.
- The loader: double mirror identity, Thomson masking consistent with the flag, clipping
  at the bound, tail weights touching only the configured tail, batch-by-shot padding
  reversible.
- The models: output shape and standardization round trip; the bounded variant saturates;
  GRU causality (Z9); saving and loading a checkpoint reproduces outputs exactly.
- Prediction: mirror averaging on a symmetric frame (Z11); the chain refuses a MAST run
  without the sign assertion; a predictor asked for MAST labels raises.
- The stage-2 refit changes only the MAST correction and leaves the DIII-D head
  untouched.
- Run records: hashes present and stable across a repeated dry run.

## Procedure

Staged for Claude Code; Kamia launches every run against real data and makes every
commit.

1. Stage 1: loss module, loss-input builder, loader, tests. The module layout is
   reported with this stage (proposal, not a gate); departures from existing modules are
   flagged.
2. Stage 2: rung 1, warm start, trainer, Optuna harness, tests.
3. Kamia: `build_loss_inputs.py`, then the smoke run on the working lists
   (`train.py --lists working`), errors pasted back.
4. Stage 3: prediction, evaluation, stage-2 refit, submission path, determinism test.
5. Kamia: smoke predict and evaluate on the working lists; then RunPod setup and the
   Part 7 matrix, priorities 1 to 3 first; the F/G winner is submitted the next morning
   after `local_test` is checked once.
6. Stage 4: rung 2 and its tests; arm 4 launched when ready.
7. Findings F101 onward added to `docs/findings.md`. Consolidation: D15 to D19, R29 to
   R32, the score-weight reading, the worst-MAST-shots reading and the F100 scalar note
   merged into `claude/model-walkthrough-log-2026-10-12.md` (merge, never overwrite: the
   project-knowledge copy carries the independent-review section).

## Constraints

- Nothing is written under `../data/raw/`.
- MAST equilibrium labels are not read; nothing is fitted, calibrated, validated or
  selected on them. Nothing is fitted on test inputs.
- `local_test` is read only at the final pre-submission check of a candidate.
- Everything learned is fitted on the `train` list (working training list in studies).
- Every run against the data is launched manually by Kamia; Claude Code runs tests on
  synthetic data only.
- Checkpoints and built tables are not committed. Code, tests, configurations, the
  notebook and `reports/` are, committed from the local machine only.
- New packages are added to `requirements.txt` the same day they are installed.
