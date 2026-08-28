# MetaFlow - Acrobot

Same DQN-based pipeline as CartPole and LunarLander, run on Acrobot-v1.

## Folder layout

```
acrobot/
  cleanrl/                 (keep this name, imports depend on it)
    pyproject.toml, poetry.lock
    cleanrl/                pipeline scripts
      dqn2.py
      q_online.py
      data3.py
      sf_maml.py
      task2.py
      dqn.py
      dqntest.py
    diayn/                  shared code used by the scripts above
      models.py
      utils.py
      evaluate_diayn.py
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
python -m cleanrl.cleanrl.dqntest \
    --env-id Acrobot-v1 \
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

Run everything from the `acrobot` folder (one level above `cleanrl/`), using
`python -m`. Checkpoints and data land under `runs/`, each script prints
where it saved to.

### 1. Train DIAYN

```
python -m cleanrl.cleanrl.dqn2 --env-id Acrobot-v1
```

### 2. Watch the skills, pick the ones you want

```
python -m cleanrl.diayn.evaluate_diayn \
    --env-id Acrobot-v1 \
    --model-path runs/checkpoints/diayn/<run_from_step_1>/latest.pth
```

Go through the per-skill videos, note down 6 skills that look off from a
normal Acrobot swing-up policy.

### 3. Train the per-skill Q-functions (`q_online`)

The 6 skill indices from step 2 go into `allowed_skills` in three places:
`cleanrl/q_online.py`, `cleanrl/data3.py` (step 4), and `cleanrl/sf_maml.py`
(step 5). Update all three to match before running anything. The current
value in all three, `[2, 6, 8, 10, 21, 23]`, is the original selection this
environment was actually run with.

```
python -m cleanrl.cleanrl.q_online \
    --env-id Acrobot-v1 \
    --disc-path runs/checkpoints/diayn/<run_from_step_1>/latest.pth
```

### 4. Generate data for meta-training (`data3`)

```
python -m cleanrl.cleanrl.data3 \
    --env-id Acrobot-v1 \
    --model-path-disc runs/checkpoints/diayn/<run_from_step_1>/latest.pth \
    --model-path-qnet runs/checkpoints/qtargetmaml/<run_from_step_3>/latest.pth
```

### 5. Meta-learn the successor features (`sf_maml`)

```
python -m cleanrl.cleanrl.sf_maml \
    --env-id Acrobot-v1 \
    --data-path runs/data/<run_from_step_4>/maml_training_data.pkl \
    --disc-path runs/checkpoints/diayn/<run_from_step_1>/latest.pth
```

### 6. Generate data for the downstream task

`dqn.py` can dump the data task2.py needs, but the lines that do it are
commented out by default. Look for the block near the end of the training
loop (search for `reward_data`), uncomment it, run:

```
python -m cleanrl.cleanrl.dqn --env-id Acrobot-v1
```

comment it back out once it's done.

### 7. Fit the task vector `w` (`task2`)

```
python -m cleanrl.cleanrl.task2 \
    --env-id Acrobot-v1 \
    --env-data-path runs/data/<run_from_step_6>/task_regression_data.pkl \
    --model-path2 runs/checkpoints/maml/<run_from_step_5>/latest.pth \
    --qnet-path runs/checkpoints/qtargetmaml/<run_from_step_3>/latest.pth
```

### 8. Downstream adaptation (`dqntest`)

```
python -m cleanrl.cleanrl.dqntest \
    --env-id Acrobot-v1 \
    --model-path runs/checkpoints/maml/<run_from_step_5>/latest.pth \
    --w-path runs/checkpoints/env_phi_task/<run_from_step_7>/latest.pth
```

`--no-pretrained` runs it as the Base comparison instead.

## A few things worth knowing

- `sf_maml.py`'s `n_actions` is 3 (Acrobot has 3 discrete actions - apply
  negative torque, no torque, or positive torque).
