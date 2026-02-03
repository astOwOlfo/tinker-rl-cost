#!/usr/bin/env python3
"""
Calculate total cost for a W&B run or local directory based on token usage and Tinker pricing.

Cost formula:
- Training: sum over all RL steps of (total_ob_tokens * input_price + total_ac_tokens * output_price + total_train_tokens * train_price)
  where total_train_tokens = total_ac_tokens + total_ob_tokens
- Evaluation: sum over evaluation epochs of (total_ac_tokens * output_price + total_ob_tokens * input_price)
"""

import argparse
import json
import os
import sys
import wandb
from wandb.errors.errors import CommError


# Valid models from user's list
VALID_MODELS = {
    "Qwen/Qwen3-VL-235B-A22B-Instruct",
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-30B-A3B-Base",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-8B-Base",
    "Qwen/Qwen3-4B-Instruct-2507",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "deepseek-ai/DeepSeek-V3.1",
    "deepseek-ai/DeepSeek-V3.1-Base",
    "meta-llama/Llama-3.1-70B",
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.2-1B",
    "moonshotai/Kimi-K2-Thinking",
}

# Pricing in USD per million tokens (prefill=input, sample=output, train=training)
# Source: https://thinkingmachines.ai/tinker/
PRICING = {
    "Qwen/Qwen3-4B-Instruct-2507": {"input": 0.07, "output": 0.22, "train": 0.22},
    "Qwen/Qwen3-8B": {"input": 0.13, "output": 0.40, "train": 0.40},
    "Qwen/Qwen3-30B-A3B": {"input": 0.12, "output": 0.30, "train": 0.36},
    "Qwen/Qwen3-VL-30B-A3B-Instruct": {"input": 0.18, "output": 0.44, "train": 0.53},
    "Qwen/Qwen3-32B": {"input": 0.49, "output": 1.47, "train": 1.47},
    "Qwen/Qwen3-235B-A22B-Instruct-2507": {"input": 0.68, "output": 1.70, "train": 2.04},
    "Qwen/Qwen3-VL-235B-A22B-Instruct": {"input": 1.02, "output": 2.56, "train": 3.07},
    "meta-llama/Llama-3.2-1B": {"input": 0.03, "output": 0.09, "train": 0.09},
    "meta-llama/Llama-3.2-3B": {"input": 0.06, "output": 0.18, "train": 0.18},
    "meta-llama/Llama-3.1-8B": {"input": 0.13, "output": 0.40, "train": 0.40},
    "meta-llama/Llama-3.1-70B": {"input": 1.05, "output": 3.16, "train": 3.16},
    "deepseek-ai/DeepSeek-V3.1": {"input": 1.13, "output": 2.81, "train": 3.38},
    "openai/gpt-oss-120b": {"input": 0.18, "output": 0.44, "train": 0.52},
    "openai/gpt-oss-20b": {"input": 0.12, "output": 0.30, "train": 0.36},
    "moonshotai/Kimi-K2-Thinking": {"input": 0.98, "output": 2.44, "train": 2.93},
}


def parse_wandb_url(url):
    """Extract entity, project, and run_id from W&B URL."""
    # URL format: https://wandb.ai/entity/project/runs/run_id
    if not url.startswith("https://wandb.ai/"):
        raise ValueError(f"Invalid W&B URL: {url}. Must start with https://wandb.ai/")

    parts = url.replace("https://wandb.ai/", "").split("/")
    if len(parts) < 4 or parts[2] != "runs":
        raise ValueError(f"Invalid W&B URL format: {url}. Expected format: https://wandb.ai/entity/project/runs/run_id")

    entity = parts[0]
    project = parts[1]
    run_id = parts[3]

    return entity, project, run_id


def get_model_pricing(model_name):
    """Get pricing for a model, failing loudly if not available."""
    if model_name not in VALID_MODELS:
        raise ValueError(
            f"Model '{model_name}' is not in the valid models list. "
            f"Valid models are:\n" + "\n".join(f"  - {m}" for m in sorted(VALID_MODELS))
        )

    if model_name not in PRICING:
        raise ValueError(
            f"Model '{model_name}' is valid but has no pricing information available. "
            f"Models with pricing:\n" + "\n".join(f"  - {m}" for m in sorted(PRICING.keys()))
        )

    return PRICING[model_name]


def calculate_cost_from_directory(directory_path):
    """Calculate total cost from a local Tinker run log directory."""
    # Check for required files
    config_path = os.path.join(directory_path, "config.json")
    metrics_path = os.path.join(directory_path, "metrics.jsonl")

    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"Directory '{directory_path}' does not contain 'config.json'. "
            "This does not appear to be a Tinker run log directory."
        )

    if not os.path.isfile(metrics_path):
        raise FileNotFoundError(
            f"Directory '{directory_path}' does not contain 'metrics.jsonl'. "
            "This does not appear to be a Tinker run log directory."
        )

    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    if "model_name" not in config:
        raise KeyError(
            f"config.json does not contain 'model_name' field. Cannot determine model for pricing."
        )

    model_name = config["model_name"]

    # Get pricing
    pricing = get_model_pricing(model_name)

    # Load metrics
    metrics_entries = []
    with open(metrics_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                metrics_entries.append(entry)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_num} of metrics.jsonl: {e}"
                ) from e

    if not metrics_entries:
        raise ValueError("metrics.jsonl is empty or contains no valid entries.")

    # Check for required metrics
    required_metrics = ["env/all/total_ob_tokens", "env/all/total_ac_tokens"]

    # Collect all available metrics
    all_metrics = set()
    for entry in metrics_entries:
        all_metrics.update(entry.keys())

    # Verify required metrics exist
    for metric in required_metrics:
        if metric not in all_metrics:
            raise KeyError(
                f"Required metric '{metric}' not found in metrics.jsonl. "
                f"Available metrics: {sorted(all_metrics)}"
            )

    # Check evaluation metrics
    has_eval_ob = "test/env/all/total_ob_tokens" in all_metrics
    has_eval_ac = "test/env/all/total_ac_tokens" in all_metrics

    if has_eval_ob != has_eval_ac:
        raise ValueError(
            f"Evaluation metrics are inconsistent: "
            f"has test/env/all/total_ob_tokens={has_eval_ob}, "
            f"has test/env/all/total_ac_tokens={has_eval_ac}. "
            "Both must exist or neither must exist."
        )

    # Calculate training cost
    training_cost = 0.0

    for entry in metrics_entries:
        if "env/all/total_ob_tokens" in entry and "env/all/total_ac_tokens" in entry:
            ob_tokens = entry["env/all/total_ob_tokens"]
            ac_tokens = entry["env/all/total_ac_tokens"]

            # Verify tokens are valid numbers
            if ob_tokens is None or ac_tokens is None:
                continue

            # Cost for this step
            input_cost = (ob_tokens / 1_000_000) * pricing["input"]
            output_cost = (ac_tokens / 1_000_000) * pricing["output"]
            train_tokens = ob_tokens + ac_tokens
            train_cost = (train_tokens / 1_000_000) * pricing["train"]

            step_cost = input_cost + output_cost + train_cost
            training_cost += step_cost

    # Calculate evaluation cost if available
    eval_cost = 0.0

    if has_eval_ob:
        for entry in metrics_entries:
            if "test/env/all/total_ob_tokens" in entry and "test/env/all/total_ac_tokens" in entry:
                ob_tokens = entry["test/env/all/total_ob_tokens"]
                ac_tokens = entry["test/env/all/total_ac_tokens"]

                # Verify tokens are valid numbers
                if ob_tokens is None or ac_tokens is None:
                    continue

                # Evaluation cost (no training cost)
                input_cost = (ob_tokens / 1_000_000) * pricing["input"]
                output_cost = (ac_tokens / 1_000_000) * pricing["output"]

                step_cost = input_cost + output_cost
                eval_cost += step_cost

    # Total cost
    total_cost = training_cost + eval_cost

    # Print results
    if not has_eval_ob:
        print("\033[93mWARNING: No evaluation metrics found (test/env/all/total_ob_tokens, test/env/all/total_ac_tokens)\033[0m")
        print()
        print("WARNING: If collating multi-turn rollouts, this cost is an overestimate.")
        print()
        print(f"Cost: ${total_cost:.4f}")
    else:
        print(f"Cost: ${total_cost:.4f} (train: ${training_cost:.4f}, evaluation: ${eval_cost:.4f})")

    return total_cost


def calculate_cost(run_url):
    """Calculate total cost for a W&B run."""
    # Parse URL
    entity, project, run_id = parse_wandb_url(run_url)
    run_path = f"{entity}/{project}/{run_id}"

    # Initialize W&B API and fetch run
    api = wandb.Api()
    try:
        run = api.run(run_path)
    except (CommError, ValueError) as e:
        if "Could not find run" in str(e):
            print(f"\033[91mERROR: Could not find W&B run '{run_path}'.\033[0m")
            print(f"\033[91mMaybe you forgot to run `export WANDB_API_KEY=...` or `wandb login`?\033[0m")
            print(f"\033[91mOr the run doesn't exist / you don't have access to it.\033[0m")
            sys.exit(1)
        else:
            raise

    # Get model name from config
    config = run.config
    if "model_name" not in config:
        raise KeyError("Config does not contain 'model_name' field. Cannot determine model for pricing.")

    model_name = config["model_name"]

    # Get pricing
    pricing = get_model_pricing(model_name)

    # Get history
    history = run.history(pandas=False)
    if not history:
        raise ValueError("Run history is empty. Cannot calculate cost.")

    # Check for required metrics
    required_metrics = ["env/all/total_ob_tokens", "env/all/total_ac_tokens"]
    eval_metrics = ["test/env/all/total_ob_tokens", "test/env/all/total_ac_tokens"]

    # Collect all available metrics
    all_metrics = set()
    for entry in history:
        all_metrics.update(entry.keys())

    # Verify required metrics exist
    for metric in required_metrics:
        if metric not in all_metrics:
            raise KeyError(
                f"Required metric '{metric}' not found in run history. "
                f"Available metrics: {sorted(all_metrics)}"
            )

    # Check evaluation metrics
    has_eval_ob = "test/env/all/total_ob_tokens" in all_metrics
    has_eval_ac = "test/env/all/total_ac_tokens" in all_metrics

    if has_eval_ob != has_eval_ac:
        raise ValueError(
            f"Evaluation metrics are inconsistent: "
            f"has test/env/all/total_ob_tokens={has_eval_ob}, "
            f"has test/env/all/total_ac_tokens={has_eval_ac}. "
            "Both must exist or neither must exist."
        )

    # Calculate training cost
    training_cost = 0.0

    for entry in history:
        if "env/all/total_ob_tokens" in entry and "env/all/total_ac_tokens" in entry:
            ob_tokens = entry["env/all/total_ob_tokens"]
            ac_tokens = entry["env/all/total_ac_tokens"]

            # Verify tokens are valid numbers
            if ob_tokens is None or ac_tokens is None:
                continue

            # Cost for this step
            input_cost = (ob_tokens / 1_000_000) * pricing["input"]
            output_cost = (ac_tokens / 1_000_000) * pricing["output"]
            train_tokens = ob_tokens + ac_tokens
            train_cost = (train_tokens / 1_000_000) * pricing["train"]

            step_cost = input_cost + output_cost + train_cost
            training_cost += step_cost

    # Calculate evaluation cost if available
    eval_cost = 0.0

    if has_eval_ob:
        for entry in history:
            if "test/env/all/total_ob_tokens" in entry and "test/env/all/total_ac_tokens" in entry:
                ob_tokens = entry["test/env/all/total_ob_tokens"]
                ac_tokens = entry["test/env/all/total_ac_tokens"]

                # Verify tokens are valid numbers
                if ob_tokens is None or ac_tokens is None:
                    continue

                # Evaluation cost (no training cost)
                input_cost = (ob_tokens / 1_000_000) * pricing["input"]
                output_cost = (ac_tokens / 1_000_000) * pricing["output"]

                step_cost = input_cost + output_cost
                eval_cost += step_cost

    # Total cost
    total_cost = training_cost + eval_cost

    # Print results
    if not has_eval_ob:
        print("\033[93mWARNING: No evaluation metrics found (test/env/all/total_ob_tokens, test/env/all/total_ac_tokens)\033[0m")
        print()
        print(f"Cost: ${total_cost:.4f}")
    else:
        print(f"Cost: ${total_cost:.4f} (train: ${training_cost:.4f}, evaluation: ${eval_cost:.4f})")

    return total_cost


def main():
    parser = argparse.ArgumentParser(
        description="Calculate total cost for a W&B run or local directory based on token usage and Tinker pricing."
    )
    parser.add_argument(
        "path_or_url",
        type=str,
        help="Path to Tinker run log directory or W&B run URL (e.g., https://wandb.ai/entity/project/runs/run_id)"
    )

    args = parser.parse_args()
    path_or_url = args.path_or_url

    # Check if it's a directory
    if os.path.isdir(path_or_url):
        calculate_cost_from_directory(path_or_url)
    elif path_or_url.startswith("https://wandb.ai/"):
        # Try as W&B URL
        calculate_cost(path_or_url)
    else:
        # Check if it could be a path that doesn't exist
        if "/" in path_or_url or "\\" in path_or_url or not path_or_url.startswith("http"):
            # Looks like a path but doesn't exist
            print(f"\033[91mERROR: no such directory or wandb run: {path_or_url}\033[0m")
            sys.exit(1)
        else:
            # Might be a malformed URL, try it anyway
            try:
                calculate_cost(path_or_url)
            except Exception:
                print(f"\033[91mERROR: no such directory or wandb run: {path_or_url}\033[0m")
                sys.exit(1)


if __name__ == "__main__":
    main()
