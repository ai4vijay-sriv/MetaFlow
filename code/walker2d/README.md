# MetaFlow - Walker2D

Same SF-MAML pipeline as HalfCheetah and Ant, run on Walker2d-v4. Steps 1
through 7 are the same scripts as those two. Step 8 uses `ddpgtesting19.py`
same as Ant did (not `ddpgtest.py`, which was HalfCheetah's version).

## Folder layout

```
walker2d/
  cleanrl/                 (keep this name, imports depend on it)
    pyproject.toml, poetry.lock
    cleanrl/                pipeline scripts
      dqn2_cont.py
      q_online_cont.py
      data3_cont.py
      sf_maml_cont.py
      task2_cont.py
      ddpg_continuous_action.py
      ddpgtesting19.py       <- downstream eval, see step 8
    diayn/                  shared code used by the scripts above
      models_cont.py
      utils_cont.py
      evaluate_diayn_cont.py
  pretrained/              small checkpoints for the quick-start below
```

## Quick start: reproduce the result without running the full pipeline

This folder includes a `pretrained/` directory with the two small checkpoint
files needed for the last pipeline step - the meta-learned successor-feature
network (`pretrained/maml/latest.pth`, from step 5) and the fitted task
vector `w` (`pretrained/env_phi_task/latest.pth`, from step 7). Together
they're under 100KB. Skip steps 1-7 entirely and just run:

```
cd cleanrl
poetry install
source "$(poetry env list --full-path)/bin/activate"
cd ..
python -m cleanrl.cleanrl.ddpgtesting19 \
    --env-id Walker2d-v4 \
    --model-path pretrained/maml/latest.pth \
    --w-path pretrained/env_phi_task/latest.pth
```

This reproduces the same MetaFlow trend reported in the paper for this
environment. If you want to see how those two files are actually produced,
or reproduce the whole pipeline from scratch, follow the steps below.

## Setup

```
cd cleanrl
poetry install
source "$(poetry env list --full-path)/bin/activate"
cd ..
```

This activates the virtual environment poetry just created, so the `python`
you run from here on is the right one with everything installed. If you open
a new terminal later, just re-run the `source` line (no need to `poetry
install` again).

## Running it

One thing before you start: every script here logs to Weights & Biases by default (`track` defaults to `True`). Either run `wandb login` once (free account, one-time), or add `--no-track` to any command below to skip cloud logging - everything still gets written to TensorBoard logs under `runs/` either way.

Run everything from the `walker2d` folder (one level above `cleanrl/`),
using `python -m`. Checkpoints and data land under `runs/`, each script
prints where it saved to.

### 1. Train DIAYN

```
python -m cleanrl.cleanrl.dqn2_cont --env-id Walker2d-v4
```

### 2. Watch the skills, pick the ones you want

```
python -m cleanrl.diayn.evaluate_diayn_cont \
    --env-id Walker2d-v4 \
    --model-path runs/checkpoints/diayn/<run_from_step_1>/latest.pth
```

Go through the per-skill videos, note down 6 skills that look misaligned
with normal forward locomotion (falling, rolling, barely moving, etc).

### 3. Train the per-skill Q-functions (`q_online`)

The 6 skill indices from step 2 need to go into `allowed_skills` in three
places: `cleanrl/q_online_cont.py`, `cleanrl/data3_cont.py` (step 4) and
`cleanrl/sf_maml_cont.py` (step 5). Each one hardcodes its own copy near the
top of the script. Update all three to match before running anything.

Then:

```
python -m cleanrl.cleanrl.q_online_cont \
    --env-id Walker2d-v4 \
    --model-path-disc runs/checkpoints/diayn/<run_from_step_1>/latest.pth
```

### 4. Generate data for meta-training (`data3`)

```
python -m cleanrl.cleanrl.data3_cont \
    --env-id Walker2d-v4 \
    --model-path-disc runs/checkpoints/qtargetmaml/<run_from_step_3>/latest.pth \
    --model-path-qnet runs/checkpoints/qtargetmaml/<run_from_step_3>/latest.pth
```

### 5. Meta-learn the successor features (`sf_maml`)

```
python -m cleanrl.cleanrl.sf_maml_cont \
    --env-id Walker2d-v4 \
    --data-path runs/data/<run_from_step_4>/task_regression_data.pkl \
    --disc-path runs/checkpoints/qtargetmaml/<run_from_step_3>/latest.pth \
    --qnet-path runs/checkpoints/qtargetmaml/<run_from_step_3>/latest.pth
```

### 6. Generate data for the downstream task

Same as before - open `cleanrl/ddpg_continuous_action.py`, find the block
marked "comment theses lines when not required", uncomment those 5 lines,
run:

```
python -m cleanrl.cleanrl.ddpg_continuous_action --env-id Walker2d-v4
```

then comment them back out.

### 7. Fit the task vector `w` (`task2`)

```
python -m cleanrl.cleanrl.task2_cont \
    --env-id Walker2d-v4 \
    --env-data-path runs/data/<run_from_step_6>/task_regression_data.pkl \
    --model-path2 runs/checkpoints/maml/<run_from_step_5>/latest.pth \
    --qnet-path runs/checkpoints/qtargetmaml/<run_from_step_3>/latest.pth
```

### 8. Downstream adaptation - `ddpgtesting19.py`

```
python -m cleanrl.cleanrl.ddpgtesting19 \
    --env-id Walker2d-v4 \
    --model-path runs/checkpoints/maml/<run_from_step_5>/latest.pth \
    --w-path runs/checkpoints/env_phi_task/<run_from_step_7>/latest.pth
```

Like Ant, this version puts the successor features into both the critic and
the actor (copies the SF network's first layer into the actor and freezes
it), not just the critic like HalfCheetah's `ddpgtest.py` did.

`--no-pretrained` runs it as the Base comparison. `--disc-path` is in the
args but unused.

## A few things worth knowing

- Walker2d's action space is 6-dimensional (`n_actions = 6` in
  sf_maml_cont.py).
