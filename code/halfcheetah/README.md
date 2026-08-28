# MetaFlow - HalfCheetah

This is the SF-MAML pipeline for HalfCheetah-v4. It runs in stages, each one
producing a checkpoint or data file that the next stage needs.

## Folder layout

```
halfcheetah/
  cleanrl/                 (keep this name, imports depend on it)
    pyproject.toml, poetry.lock
    cleanrl/                pipeline scripts
      dqn2_cont.py
      q_online_cont.py
      data3_cont.py
      sf_maml_cont.py
      task2_cont.py
      ddpg_continuous_action.py
      ddpgtest.py
    diayn/                  shared code used by the scripts above
      models_cont.py        (network classes: Discriminator, Actor, Critic, SFNetwork...)
      utils_cont.py          (training loop helpers)
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
python -m cleanrl.cleanrl.ddpgtest \
    --env-id HalfCheetah-v4 \
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

Run all commands from the `halfcheetah` folder (one level above `cleanrl/`),
using `python -m`, since the scripts import each other as `cleanrl.diayn.*`
and `cleanrl.cleanrl.*`.

Checkpoints and data get written under `runs/`. Each script prints the run
folder it saved to at the end - grab that path and pass it into the next
script.

### 1. Train DIAYN

```
python -m cleanrl.cleanrl.dqn2_cont --env-id HalfCheetah-v4
```

Saves a discriminator checkpoint to `runs/checkpoints/diayn/<run>/latest.pth`.

### 2. Watch the skills and pick the ones you want

```
python -m cleanrl.diayn.evaluate_diayn_cont \
    --env-id HalfCheetah-v4 \
    --model-path runs/checkpoints/diayn/<run_from_step_1>/latest.pth
```

This makes a video for each skill under `videos/`. Go through them and pick
6 skills (that's `n_skills_selected`, can be changed) that look off from
normal forward locomotion - falling, rolling, not really moving, that kind
of thing. Write down their indices.

### 3. Train the per-skill Q-functions (`q_online`)

The 6 skill indices from step 2 need to go into `allowed_skills` in three
places: `cleanrl/q_online_cont.py`, `cleanrl/data3_cont.py` (step 4) and
`cleanrl/sf_maml_cont.py` (step 5). Each one hardcodes its own copy of the
list near the top of the script, right now they're all set to the same
placeholder `[0, 3, 5, 16, 22, 23]`. Update all three to match before
running anything, otherwise the skill numbering won't line up across
stages.

Then run:

```
python -m cleanrl.cleanrl.q_online_cont \
    --env-id HalfCheetah-v4 \
    --model-path-disc runs/checkpoints/diayn/<run_from_step_1>/latest.pth
```

This loads the discriminator and trains a skill-conditioned actor/critic
against the DIAYN reward. Saves one checkpoint with the q-network, actor and
discriminator bundled together, to `runs/checkpoints/qtargetmaml/<run>/latest.pth`.
The next few scripts all read from this file.

### 4. Generate data for meta-training (`data3`)

```
python -m cleanrl.cleanrl.data3_cont \
    --env-id HalfCheetah-v4 \
    --model-path-disc runs/checkpoints/qtargetmaml/<run_from_step_3>/latest.pth \
    --model-path-qnet runs/checkpoints/qtargetmaml/<run_from_step_3>/latest.pth
```

Writes `runs/data/<run>/task_regression_data.pkl` - state/action pairs with
their Q-values, one set per skill, which is what the MAML step trains on.

### 5. Meta-learn the successor features (`sf_maml`)

```
python -m cleanrl.cleanrl.sf_maml_cont \
    --env-id HalfCheetah-v4 \
    --data-path runs/data/<run_from_step_4>/task_regression_data.pkl \
    --disc-path runs/checkpoints/qtargetmaml/<run_from_step_3>/latest.pth \
    --qnet-path runs/checkpoints/qtargetmaml/<run_from_step_3>/latest.pth
```

Saves the meta-trained successor feature network to
`runs/checkpoints/maml/<run>/latest.pth`.

### 6. Generate data for the downstream task

`task2_cont.py` needs its own data, this time from actually running DDPG on
the real HalfCheetah objective. `ddpg_continuous_action.py` can dump this
data, but the lines that do it are commented out by default (so normal DDPG
runs don't carry the overhead).

Open `cleanrl/ddpg_continuous_action.py` and look for this block (search for
"comment theses lines when not required"):

```python
    # #comment theses lines when not required
    # print(f"Saving reward {len(reward_data)} entries")
    # model_dir = f"runs/data/{run_name}"
    # os.makedirs(model_dir, exist_ok=True)
    # with open(os.path.join(model_dir, "task_regression_data.pkl"), "wb") as f:
    #     pickle.dump(reward_data, f)
```

Uncomment those 5 lines, run:

```
python -m cleanrl.cleanrl.ddpg_continuous_action --env-id HalfCheetah-v4
```

and comment them back out once it's done. This writes another
`task_regression_data.pkl`, but under a different run folder than step 4 -
don't mix the two up.

### 7. Fit the task vector `w` (`task2`)

```
python -m cleanrl.cleanrl.task2_cont \
    --env-id HalfCheetah-v4 \
    --env-data-path runs/data/<run_from_step_6>/task_regression_data.pkl \
    --model-path2 runs/checkpoints/maml/<run_from_step_5>/latest.pth \
    --qnet-path runs/checkpoints/qtargetmaml/<run_from_step_3>/latest.pth
```

Saves to `runs/checkpoints/env_phi_task/<run>/latest.pth`.

### 8. Downstream adaptation (`ddpgtest`)

```
python -m cleanrl.cleanrl.ddpgtest \
    --env-id HalfCheetah-v4 \
    --model-path runs/checkpoints/maml/<run_from_step_5>/latest.pth \
    --w-path runs/checkpoints/env_phi_task/<run_from_step_7>/latest.pth
```

This is the final step - initializes the critic from the meta-learned
successor features and `w`, then fine-tunes with regular DDPG. Set
`--no-pretrained` to run it as the Base comparison instead (plain DDPG
critic, no successor features, everything else the same).

`--disc-path` is in the args but not actually used - the discriminator
loading in this script is commented out and doesn't affect the run.

## A few things worth knowing

- `sf_dim` (32) and `n_skills_selected`/`n_skills_total` (6/25) are the same
  across all environments. Everything else can be tuned per environment
  through the command line flags.
