# Fusion Equilibrium Challenge — NeurIPS 2026

Reconstructing tokamak magnetic equilibria — flux map ψ(R,Z), q95, βN — from
non-magnetic diagnostics (coil currents + Thomson scattering). Trained on DIII-D,
generalizing zero-shot to MAST.

Competition: https://www.codabench.org/competitions/17739/
Challenge site: https://fusion-equilibrium-challenge.sophelio.io/
Codabench user: ksalango

## Layout
- `src/fusion_eq/` — the package (data loading, features, models, evaluation)
- `notebooks/` — exploration; numbered, kept small — logic goes in the package
- `experiments/` — one folder per experiment: config + results + notes
- `notes/` — domain study, decisions log, submission log

## Setup
Requires Python >= 3.11 (built on 3.13).

    python3.13 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    pip install -e .

Data lives OUTSIDE this repo at `../data/` (98 GB, from Hugging Face:
Sophelio/fusion-equilibrium-challenge). Starter kit expected at
`../fusion-equilibrium-challenge-starter/` (upstream, read-only, pull for updates).

## Key dates
- Development phase ends: **Oct 17, 2026, 5:00 PM PDT** (5 subs/day, 100 total)
- Final phase: Oct 19–26 (3 submissions, blind)
