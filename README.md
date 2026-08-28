# MetaFlow

Code for the paper "Successor Features Enable Transfer from Behaviorally
Diverse Tasks" (MetaFlow).

- `code/` - one folder per environment used in the paper. Each folder is
  self-contained - its own copy of the code, its own README with exact
  commands, nothing shared between folders. Pick the environment you care
  about and just use that folder.
- `Training Tasks videos/` - example clips of the training-task behaviors
  referenced in the paper.

## Environments

| Folder | Environment | Type |
|---|---|---|
| `code/halfcheetah/` | HalfCheetah-v4 | continuous control |
| `code/ant/` | Ant-v4 | continuous control |
| `code/walker2d/` | Walker2d-v4 | continuous control |
| `code/cartpole/` | CartPole-v1 | discrete |
| `code/lunarlander/` | LunarLander-v2 | discrete |
| `code/acrobot/` | Acrobot-v1 | discrete |
| `code/windygrid/` | WindyGridWorld-v0 | discrete, grid world |
| `code/maze/` | maze-random-10x10-plus-v0 | discrete, grid world |

Each folder's README walks through regenerating everything from scratch.
Each folder also includes a small `pretrained/` directory with the two
files needed to run the final step directly and reproduce the reported
result. See "Quick start" near the top of each folder's own README.

## The pipeline, in short

Every environment follows the same 8-step pipeline:

1. **Train DIAYN** - unsupervised skill discovery, no task reward involved.
2. **Watch the skills** - render a video per skill, manually pick a handful
   that look misaligned with the real task.
3. **Train per-skill Q-functions** - proper Q-function + actor for just
   those picked skills, using the DIAYN discriminator as the reward.
4. **Collect meta-training data** - roll out the skill-conditioned policies,
   save (state, action, Q-value) pairs.
5. **Meta-learn successor features** - MAML across the picked skills, output
   is a shared successor-feature network, not tied to any one skill.
6. **Collect real task data** - run plain DDPG/DQN on the actual downstream
   objective just to gather transitions.
7. **Fit the task weight vector `w`** - using the successor features and
   the real transitions, solve for how much each feature matters for the
   real reward.
8. **Downstream adaptation** - initialize a critic from the successor
   features, plug in `w`, fine-tune with ordinary RL. This produces the
   numbers reported in the paper.

Continuous-control environments (HalfCheetah, Ant, Walker2D) use
DDPG-flavored scripts (`*_cont.py`, `ddpgtest*.py`). Discrete environments
use DQN-flavored scripts (no `_cont` suffix, `dqntest.py`). Same idea either
way. Each environment's own README has the exact commands and file names for
that environment.

## Setup

You need **Python 3.10** specifically (the project pins `>=3.10,<3.11`) and
[Poetry](https://python-poetry.org/) to install dependencies.

**Install Python 3.10** if you don't have it. On Ubuntu/Debian:
```
sudo apt install python3.10 python3.10-venv
```
On Mac (via Homebrew):
```
brew install python@3.10
```
Or use [pyenv](https://github.com/pyenv/pyenv) if you manage multiple Python
versions: `pyenv install 3.10.20`.

**Install Poetry** if you don't have it:
```
curl -sSL https://install.python-poetry.org | python3 -
```
Check it worked with `poetry --version`. Poetry will automatically find and
use Python 3.10 for these projects even if it's not your system default, as
long as it's installed somewhere on your machine.

**Then, for whichever environment you're working on:**
```
cd code/<environment_folder>/cleanrl
poetry install
source "$(poetry env list --full-path)/bin/activate"
cd ..
```
This creates a virtual environment with everything the scripts need (torch,
gymnasium with mujoco/box2d support, wandb, tyro, stable-baselines3, etc.)
and activates it in your current terminal. From here on, plain `python`
commands (run from the environment folder, one level above `cleanrl/`) use
this environment. If you open a new terminal later, you only need to re-run
the `source` line - no need to `poetry install` again.

Each of the 8 environment folders needs this setup done separately (they're
independent projects, not linked to each other).

### Two environments need one extra package

`code/windygrid/` and `code/maze/` use environments that aren't on PyPI - they're
installed straight from GitHub, after the steps above:
```
# windygrid only
pip install git+https://github.com/ibrahim-elshar/gym-windy-gridworlds

# maze only
pip install git+https://github.com/MattChanTK/gym-maze
```
This is also called out in those two folders' own READMEs.

### About Weights & Biases (wandb)

Every script logs to Weights & Biases by default. Either run `wandb login`
once (needs a free account), or add `--no-track` to any command to skip
cloud logging - everything is written to TensorBoard logs under `runs/`
either way, so `--no-track` doesn't lose you anything if you just want to
run things locally.

### If you're on a headless server (no display)

MuJoCo (used by HalfCheetah, Ant, Walker2D) can need a rendering backend
even for non-visual use. If you hit rendering errors, try:
```
export MUJOCO_GL=egl
```
before running anything.

## A note on GPUs

The scripts pick `cuda` automatically if a GPU is available, otherwise fall
back to `cpu`. No specific GPU index is required.
