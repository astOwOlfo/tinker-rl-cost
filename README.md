# Tinker RL Cost Calculator

Calculate costs for Tinker RL training runs based on token usage and pricing.

## Installation

# With uv

```bash
uv add git+https://github.com/astowolfo/tinker-rl-cost
```

# With pip

```bash
pip install git+https://github.com/astowolfo/tinker-rl-cost
```

## Usage

To print the cost of a run:

```bash
uv run tinker-rl-cost https://wandb.ai/entity/project/runs/run_id
```

(you will first need to run `export WANDB_API_KEY=...` or `wandb login`)

or

```bash
uv run tinker-rl-cost /path/to/run/directory
```

The directory should be the one you used as the `log_path` to `tinker_cookbook.rl.train.Config` when doing the RL run.
