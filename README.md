# Fusion Equilibrium Challenge — NeurIPS 2026

Reconstructing tokamak magnetic equilibria — flux map ψ(R,Z), q95, βN — from
non-magnetic diagnostics (coil currents + Thomson scattering). Trained on DIII-D,
generalizing zero-shot to MAST.

Competition: https://www.codabench.org/competitions/17739/
Challenge site: https://fusion-equilibrium-challenge.sophelio.io/
Codabench user: ksalango

**Status:** competition closed Oct 2026. Public test score S 0.9786 on DIII-D
(trained) and 0.668 on MAST (zero-shot, no MAST labels used).

## Approach
A physics-plus-ML hybrid. The flux from the coils is calculated from each machine's
coil geometry with a Green's function, and a PyTorch network (a linear path plus an
MLP, started from a ridge fit) predicts only the plasma's current distribution, as 50
basis coefficients, from inputs made unitless so they mean the same on both machines.
Training uses up-down mirror augmentation and a loss on the flux, its gradient and a
band around the plasma boundary. Predicting current instead of flux keeps machine
geometry out of what the network learns; a ceiling test before any training showed
the choice costs almost nothing on DIII-D (S 0.9982 against 0.9990 for a flux target,
with perfect coefficients). Details in `docs/TARGET.md` and `docs/findings.md`.

The zero-shot inference pipeline takes the DIII-D-trained network and the DIII-D
scaler to MAST unchanged. It places the predicted current on MAST's grid, converts it
to flux with MAST's own Green's function and coil model, and calculates q95 from the
predicted map and βN from a physics formula. No MAST equilibrium label is read at any
stage.

## Evaluation
Every run was scored locally with the organizers' scorer before submission, on 870
DIII-D shots held out by whole shot (`splits/split_v1.json`, with a further 350 shots
kept as a local test set), and reported separately for the 286 validation shots with
no near-repeat in training. Seven experiment arms (loss terms, feature sets, a GRU
sequence model, mirroring off) were trained as 15 runs over one to three seeds each;
the submitted arm scores S 0.9784 on validation with a seed standard deviation of
0.0001. MAST has no labels to validate on, so transfer was checked with label-free
diagnostics on the predicted MAST maps (boundary and topology flags over all 1,206
test shots, coil-model checks against published hardware data) and then by
leaderboard submission.

Public test set, scored on Codabench:

| Metric | DIII-D (trained) | MAST (zero-shot) |
| --- | --- | --- |
| Score S | 0.9786 | 0.668 |
| R² of ψ | 0.9996 | 0.981 |
| Boundary error D_LCFS | 0.016 m | 0.179 m |

G_ratio (MAST S / DIII-D S): 0.683.

For reference on MAST: the physics-only version of the pipeline (no learned model)
scores 0.483 and the first learned model scored 0.496. Correcting the MAST solenoid
coil model from published winding data raised that to 0.653, and wall-mask q95 with
physics βN gave 0.668. The remaining gap is in shape detail (vertical position,
triangularity) and βN.

## Setup
Requires Python >= 3.11 (built on 3.13).

    python3.13 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    pip install -e .

Data lives OUTSIDE this repo at `../data/` (98 GB, from Hugging Face:
Sophelio/fusion-equilibrium-challenge). Starter kit expected at
`../fusion-equilibrium-challenge-starter/` (upstream, read-only, pull for updates).

## Reproduce
Training and evaluation read processed tables built from the raw data by the
`scripts/build_*.py` scripts (frame table, Green's tables, features, targets, loss
inputs; specifications in `docs/`).

    # one training run (arm G, seed 0); writes reports/runs/arm_g_s0/
    python scripts/train.py --config configs/model/arm_g.yaml --seed 0 --run-id arm_g_s0

    # score it with the organizers' scorer; writes reports/runs/arm_g_s0/eval.md
    python scripts/evaluate.py --run arm_g_s0

    # tests (synthetic inputs)
    pytest -q

Each run record (`run.json`) holds the configuration and its hash, the git commit,
the data build hashes, the seed and the environment freeze. `scripts/predict.py`
builds the test-set predictions for both machines (`docs/PIPELINE.md`, Part 5).

## Layout
- `src/fusion_eq/` — the package (data loading, features, models, evaluation)
- `scripts/` — command-line entry points: data builds, training, evaluation, prediction, diagnostics
- `configs/` — feature, target and model configurations (one file per experiment arm in `configs/model/`)
- `tests/` — pytest suite
- `docs/` — phase specifications, data dictionary and the findings register (`findings.md`)
- `reports/` — per-run metrics and diagnostics (run records, evaluation reports, QA, determinism checks)
- `notebooks/` — exploration; numbered, kept small — logic goes in the package
- `notes/` — domain study
- `splits/` — the fixed train / validation / local-test split
