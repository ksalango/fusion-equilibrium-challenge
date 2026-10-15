# RunPod procedure

How the Part 7 training runs of `docs/PIPELINE.md` run on a rented GPU pod. The pod is a
compute target only: it reads a copy of the processed data, writes run records and
checkpoints, and everything comes back to the local machine by rsync. Commits are made from the
local machine only; a pod never commits, pushes or holds a token.

Paths below assume the pod's persistent storage is `/workspace` (a network volume when one is
attached) and that the layout mirrors the local one:

```
/workspace/fusion-challenge/
    fusion-equilibrium-challenge/            the repository
    fusion-equilibrium-challenge-starter/    the organizers' starter kit (read-only)
    data/processed/                          the processed builds
    data/runs/                               checkpoints written by the runs
```

The pod holds no raw shot file. Training runs with `--no-score`; the organizers' scorer, which
reads the true maps from the raw files, runs locally afterwards through `scripts/evaluate.py`.

With this layout every script's defaults resolve as they do locally, so the commands need no
path arguments.

## 1. Clone

```
mkdir -p /workspace/fusion-challenge && cd /workspace/fusion-challenge
git clone https://github.com/ksalango/fusion-equilibrium-challenge.git
git clone https://github.com/Sophelio/fusion-equilibrium-challenge-starter.git
git -C fusion-equilibrium-challenge checkout <commit>
git -C fusion-equilibrium-challenge-starter checkout <starter commit>
```

`<commit>` is the local commit the runs are meant to reproduce (`git rev-parse HEAD` locally);
`<starter commit>` is the starter-kit commit recorded in any local `run.json` (`starter_commit`).
A private repository needs a read-only deploy key or a fine-grained read token for the clone;
the token is typed at the prompt, never stored in the repository or the pod's shell history.

## 2. Environment

Python 3.13, as locally. If the pod image lacks it, any 3.13 interpreter works (for example
`uv venv --python 3.13 .venv`).

```
cd /workspace/fusion-challenge/fusion-equilibrium-challenge
python3.13 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
grep -v '^-e ' requirements.txt > /tmp/requirements-pod.txt
pip install -r /tmp/requirements-pod.txt
pip install -e . --no-deps
python -c "import torch; print(torch.cuda.is_available())"
```

The last line must print `True`. The CUDA wheel is installed first so that
`pip install -r` finds `torch==2.14.1` already satisfied (`2.14.1+cu126` matches the pin)
and keeps it; every other package comes from the pinned `requirements.txt`. The `-e` line of
`requirements.txt` points at the local editable install and is replaced by `pip install -e .`.
The cu126 build runs on any driver of the CUDA 12 series or later; torch 2.14.1 is also
published for cu130 and cu132 if the pod's driver calls for them (`nvidia-smi`).

Quick check that the environment matches before any run:

```
pytest -q tests/test_model_config.py tests/test_loss.py tests/test_plasma_models.py
```

## 3. Data, once per pod or volume

From the local machine, with the pod's SSH address (RunPod's "SSH over exposed TCP" gives
`root@<ip>` and `-p <port>`):

```
cd ~/Desktop/PROJECTS/fusion-challenge
rsync -avP -e "ssh -p <port>" data/processed/ root@<ip>:/workspace/fusion-challenge/data/processed/
```

This copies every processed build, about 5 GB: frame table (rung 2 reads its frame times),
features v2, targets v1, loss inputs (`data/processed/loss_inputs/v1`, built by
`scripts/build_loss_inputs.py`), Green's tables and the caches. The scaler is committed
(`reports/features/v2/scaler.json`) and arrives with the clone. No raw shot file and no
`local_test` data is copied.

## 4. Launch sheet

One tmux session runs the whole matrix of Part 7, one run after another, every run on the full
`train` and `validation` lists (the default `--lists full`), on the GPU (selected automatically),
without the scorer (`--no-score`). Each run is named by `--run-id`, so the local evaluation
commands of step 5 are known in advance. `--seed` overrides `train.seed` (initialization,
dropout, shuffling, mirroring and Thomson masking); the run record holds the seed and the
configuration hash including it.

Open the session:

```
tmux new-session -s matrix
cd /workspace/fusion-challenge/fusion-equilibrium-challenge
source .venv/bin/activate
mkdir -p /workspace/logs
```

Then paste this block into the session; the shell runs the lines in order, each after the
previous has finished, and a failed run does not stop the ones after it:

```
python scripts/train.py --config configs/model/arm_f.yaml --seed 0 --run-id arm_f_s0 --no-score 2>&1 | tee /workspace/logs/arm_f_s0.log
python scripts/train.py --config configs/model/arm_f.yaml --seed 1 --run-id arm_f_s1 --no-score 2>&1 | tee /workspace/logs/arm_f_s1.log
python scripts/train.py --config configs/model/arm_f.yaml --seed 2 --run-id arm_f_s2 --no-score 2>&1 | tee /workspace/logs/arm_f_s2.log
python scripts/train.py --config configs/model/arm_g.yaml --seed 0 --run-id arm_g_s0 --no-score 2>&1 | tee /workspace/logs/arm_g_s0.log
python scripts/train.py --config configs/model/arm_g.yaml --seed 1 --run-id arm_g_s1 --no-score 2>&1 | tee /workspace/logs/arm_g_s1.log
python scripts/train.py --config configs/model/arm_g.yaml --seed 2 --run-id arm_g_s2 --no-score 2>&1 | tee /workspace/logs/arm_g_s2.log
python scripts/train.py --config configs/model/arm_setb_g.yaml --seed 0 --run-id arm_setb_g_s0 --no-score 2>&1 | tee /workspace/logs/arm_setb_g_s0.log
python scripts/train.py --config configs/model/arm_setb_g.yaml --seed 1 --run-id arm_setb_g_s1 --no-score 2>&1 | tee /workspace/logs/arm_setb_g_s1.log
python scripts/train.py --config configs/model/arm_d8_g.yaml --seed 0 --run-id arm_d8_g_s0 --no-score 2>&1 | tee /workspace/logs/arm_d8_g_s0.log
python scripts/train.py --config configs/model/arm_d8_g.yaml --seed 1 --run-id arm_d8_g_s1 --no-score 2>&1 | tee /workspace/logs/arm_d8_g_s1.log
python scripts/train.py --config configs/model/rung2_f.yaml --seed 0 --run-id rung2_f_s0 --no-score 2>&1 | tee /workspace/logs/rung2_f_s0.log
python scripts/train.py --config configs/model/rung2_f.yaml --seed 1 --run-id rung2_f_s1 --no-score 2>&1 | tee /workspace/logs/rung2_f_s1.log
python scripts/train.py --config configs/model/rung2_g.yaml --seed 0 --run-id rung2_g_s0 --no-score 2>&1 | tee /workspace/logs/rung2_g_s0.log
python scripts/train.py --config configs/model/rung2_g.yaml --seed 1 --run-id rung2_g_s1 --no-score 2>&1 | tee /workspace/logs/rung2_g_s1.log
```

Detach with `Ctrl-b d`; the runs continue. Reattach with `tmux attach -t matrix`; follow one run
with `tail -f /workspace/logs/<run_id>.log`.

Idle time only, when the matrix has finished and the pod would otherwise sit unused (a filler
run: arm G without the mirror augmentation; not part of the matrix):

```
python scripts/train.py --config configs/model/arm_g_nomirror.yaml --seed 0 --run-id arm_g_nomirror_s0 --no-score 2>&1 | tee /workspace/logs/arm_g_nomirror_s0.log
```

Each run writes `reports/runs/<run_id>/` (`run.json`, `metrics.csv`) inside the repository and
its checkpoints to `/workspace/fusion-challenge/data/runs/<run_id>/` (`best.pt`, `last.pt` and
`warm.pt`, the model at the warm start). Rerunning a run id replaces its record and checkpoints.

## 5. Results back and local evaluation

From the local machine:

```
cd ~/Desktop/PROJECTS/fusion-challenge
rsync -avP -e "ssh -p <port>" root@<ip>:/workspace/fusion-challenge/fusion-equilibrium-challenge/reports/runs/ \
    fusion-equilibrium-challenge/reports/runs/
rsync -avP -e "ssh -p <port>" root@<ip>:/workspace/fusion-challenge/data/runs/ data/runs/
rsync -avP -e "ssh -p <port>" root@<ip>:/workspace/logs/ fusion-equilibrium-challenge/internal/pod_logs/
```

Run records land in the local `reports/` (committed); checkpoints in `data/runs/` (outside the
repository, never committed); logs in the gitignored `internal/`. The scoring then runs locally,
one evaluation per arm with the other seeds in the seed spread (the warm start is scored from
`warm.pt`):

```
cd fusion-equilibrium-challenge
python scripts/evaluate.py --run arm_f_s0 --seed-runs arm_f_s1 arm_f_s2
python scripts/evaluate.py --run arm_g_s0 --seed-runs arm_g_s1 arm_g_s2
python scripts/evaluate.py --run arm_setb_g_s0 --seed-runs arm_setb_g_s1
python scripts/evaluate.py --run arm_d8_g_s0 --seed-runs arm_d8_g_s1
python scripts/evaluate.py --run rung2_f_s0 --seed-runs rung2_f_s1
python scripts/evaluate.py --run rung2_g_s0 --seed-runs rung2_g_s1
python scripts/evaluate.py --run arm_g_nomirror_s0
```

Each evaluation scores the other seeds' best checkpoints through the same chain for the seed
spread, and the warm start from `warm.pt`, so nothing depends on the training-time scorer.
Prediction and submission run locally as well (`scripts/predict.py`, with `--predictor rung2_v1`
for a rung-2 run).

## 6. Commits

Commits are made on the local machine, after the results are pulled back and reviewed. The pod
never runs `git commit` or `git push`, holds no Hugging Face or GitHub write token, and its
clone is discarded with the pod. A run is reproducible from its `run.json`: the commit, the
configuration and its hash, the data build hashes, the seed and the environment freeze.
