# Tinker RL Cost Calculator

Calculate costs for Tinker RL training runs based on token usage and pricing.

## Installation

```bash
pip install /path/to/tinker-rl-cost
```

Or with uv:

```bash
uv pip install /path/to/tinker-rl-cost
```

## Usage

Calculate cost from a W&B run:

```bash
tinker-rl-cost https://wandb.ai/entity/project/runs/run_id
```

Calculate cost from a local directory:

```bash
tinker-rl-cost /path/to/run/directory
```

The directory must contain:
- `config.json` - with a `model_name` field
- `metrics.jsonl` - with metrics including `env/all/total_ac_tokens` and `env/all/total_ob_tokens`

## Cost Formula

**Training cost** (per step):
```
cost = (total_ob_tokens * input_price + total_ac_tokens * output_price + (total_ob_tokens + total_ac_tokens) * train_price) / 1,000,000
```

**Evaluation cost** (per step, if evaluation metrics exist):
```
cost = (total_ob_tokens * input_price + total_ac_tokens * output_price) / 1,000,000
```

## Requirements

- Python 3.8+
- wandb
- WANDB_API_KEY environment variable (for W&B runs)
