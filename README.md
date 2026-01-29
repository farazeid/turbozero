# FastSVERL Implementation

This repository contains an implementation of **FastSVERL** (Fast Shapley Value Estimation using Reinforcement Learning) applied to the game of Go, specifically targeting the analysis of AlphaGo's Move 37.

It leverages the **KataGo** architecture and extends it to learn Shapley value explanations for agent behaviour, outcome prediction, and value estimation.

## Features

### 1. FastSVERL Parallel Training

Train all three Shapley models (Behaviour, Outcome, Prediction) simultaneously in a single highly-efficient parallel training loop. This maximizes GPU utilization and VRAM occupancy.

**Script:** `scripts/train_parallel.py`

**Usage:**

```bash
# Basic usage (training all 3 types)
hare run --rm -it -v .:/app -e WANDB_API_KEY=$WANDB_API_KEY --gpus device=0 $USER/turbozero uv run scripts/train_parallel.py --data-dir data/katago/2026-01-25npzs

# Interact/Invoke with custom parameters
hare run --rm -it -v .:/app -e WANDB_API_KEY=$WANDB_API_KEY --gpus device=0 $USER/turbozero uv run scripts/train_parallel.py \
    --data-dir data/katago \
    --batch-size 128 \
    --max-steps 10000 \
    --types behaviour prediction outcome
```

**Adjustable Parameters:**

- `--data-dir`: Path to directory containing `.npz` trajectory files (recursive search supported).
- `--batch-size`: Training batch size. Increase to max out GPU utilization. (Default: 128)
- `--lr`: Learning rate for Adam optimizer. (Default: 3e-4)
- `--max-steps`: Total training steps. (Default: 10000)
- `--eval-every`: Frequency of evaluation steps. (Default: 500)
- `--log-every`: Frequency of W&B logging. (Default: 100)
- `--types`: List of Shapley models to train. Options: `behaviour`, `outcome`, `prediction`. (Default: all three)
- `--shapley-blocks-ratio`: Ratio of residual blocks in Shapley model relative to the agent. (Default: 0.5)

### 2. Dataset Preparation

Tools to download KataGo training data and prepare specific game scenarios (like AlphaGo Game 2).

#### Download KataGo Data

Downloads high-ELO self-play trajectories from the KataGo public archive.

**Script:** `scripts/download_katago_data.sh`

**Usage:**

```bash
# Download latest 1 day of data
./scripts/download_katago_data.sh --latest-n-days 1 --output-dir data/katago

# Interactive flags
./scripts/download_katago_data.sh --latest-n-days 5 --output-dir my_data_folder
```

#### SGF to Trajectory (Move 37)

Converts an SGF game record into the KataGo-compatible `.npz` trajectory format, including all features (binaries + global). This is used to generate the evaluation dataset for Move 37.

**Script:** `scripts/sgf_to_trajectory.py`

**Usage:**

```bash
# Convert Game 2 SGF to trajectory
python scripts/sgf_to_trajectory.py \
    --sgf-path data/alphago_game2.sgf \
    --output-path data/alphago_game2_move37.npz \
    --start-move 37 \
    --end-move 38
```

### 3. Evaluation & Metrics

The training loop automatically performs evaluations to verify the faithfulness and quality of the Shapley estimates.

**Telemetry Logged to W&B:**

- `train/loss`, `train/shapley_loss`, `train/l2_reg`
- `train/efficiency_gap`: Distance from satisfied efficiency axiom (monitor for "shrinking gap")
- `train/grad_norm`, `train/mask_coverage`
- `eval/[type]_move37_nullity`: Validation of Nullity axiom (lower is better)
- `eval/[type]_move37_symmetry`: Validation of Symmetry axiom (lower is better)
- `eval/[type]_move37_image`: 19x19 heatmap with Board/Stone overlay
- `eval/spearman_behaviour_prediction`: Spearman rank correlation between Behaviour and Prediction attributions.
- `telemetry/eval_cycle_duration_sec`: Time spent on full evaluation cycle
- `telemetry/training_iterations_per_sec`, `telemetry/seconds_per_step`
- `telemetry/total_iterations`, `telemetry/elapsed_time_sec`
- `final/total_time_sec`, `final/avg_training_iterations_per_sec`

### 4. KataGo Architecture Integration

The implementation faithfully reproduces the KataGo architecture, including:

- **Global Input Features**: Komi, game rules, and other global state variables are processed and injected into the network.
- **Global Pooling**: The global pooling layer in the residual tower is aware of the Shapley masks, ensuring correct normalization during explanation generation.

## 5. Tests

All tests use `tyro` for configuration and log to W&B (`fastsverl-tests` project).

```bash
# Run all tests
for test in tests/test_*.py; do uv run python $test; done

# Individual tests
uv run python tests/test_katago_shapes.py      # Network output shapes
uv run python tests/test_katago_params.py      # Parameter counts
uv run python tests/test_katago_loss.py        # Loss function
uv run python tests/test_katago_offline.py     # End-to-end with real data

uv run python tests/test_shapley_models.py     # Shapley model shapes
uv run python tests/test_shapley_alignment.py  # Efficiency axiom
uv run python tests/test_shapley_loss.py       # Shapley loss functions
uv run python tests/test_shapley_masking.py    # Feature masking
uv run python tests/test_shapley_mask_sampling.py  # Mask sampling distribution
```

## 6. Reproducibility Guide

#### Step 1: Prepare Environment

```bash
hare run --rm -it -v .:/app -e WANDB_API_KEY=$WANDB_API_KEY --gpus device=0 $USER/turbozero bash
```

#### Step 2: Download Data & Checkpoint

```bash
bash scripts/download_katago_data.sh --latest-n-days 1
# Note: Checkpoint download script usage if applicable
# bash scripts/download_katago_checkpoint.sh
```

#### Step 3: Train Shapley Models

**Option A: Parallel Training (Recommended)**

```bash
uv run scripts/train_parallel.py --data-dir data/katago/2026-01-25npzs
```

**Option B: Comparison / Individual Training**

**Main training entry point:** `scripts/train_shapley.py`

**Usage:**

```bash
# Behaviour Shapley
uv run python scripts/train_shapley.py --shapley_type behaviour

# Outcome Shapley
uv run python scripts/train_shapley.py --shapley_type outcome

# Prediction Shapley
uv run python scripts/train_shapley.py --shapley_type prediction

# Resume from local checkpoint
uv run python scripts/train_shapley.py --resume_from ./checkpoints/shapley/5000

# Resume from W&B Run ID
uv run python scripts/train_shapley.py --resume_from entity/project/run_id
```

**Key Arguments (`train_shapley.py`):**

| Argument                 | Default                                    | Description                            |
| ------------------------ | ------------------------------------------ | -------------------------------------- |
| `--data_dir`             | `./data/katago_trajectories`               | Directory with .npz files              |
| `--agent_checkpoint`     | `./checkpoints/katago_frozen/model.bin.gz` | KataGo checkpoint                      |
| `--shapley_type`         | `behaviour`                                | One of: behaviour, outcome, prediction |
| `--batch_size`           | 64                                         | Training batch size                    |
| `--lr`                   | 3e-4                                       | Learning rate                          |
| `--max_steps`            | 10000                                      | Maximum training steps                 |
| `--log_every`            | 100                                        | W&B logging interval                   |
| `--shapley_blocks_ratio` | 0.5                                        | Shapley model depth ratio              |
| `--save_dir`             | `./checkpoints/shapley`                    | Directory to save checkpoints          |
| `--resume_from`          | `None`                                     | Local path or W&B Run ID to resume     |
| `--save_every`           | 5000                                       | Steps between checkpoints              |
| `--eval_every`           | 500                                        | Eval cycle interval (Move 37 + Axioms) |
| `--move37_path`          | `data/alphago_game2_move37.npz`            | Path to special Move 37 observation    |

#### Step 4: Analyze AlphaGo Move 37

```bash
# Prepare Game 2 data (using script)
./scripts/prepare_alphago_game2.sh

# Create Move 37 subset
uv run python scripts/sgf_to_trajectory.py \
  --sgf-path data/alphago_game2.sgf \
  --output-path data/alphago_game2_move37.npz \
  --start-move 37 \
  --end-move 38

# Log dataset statistics to W&B
uv run python scripts/log_dataset_stats.py --data-path data/alphago_game2_move37.npz
```

#### Step 5: Comparative Analysis

To compute the **Spearman Rank Correlation** between separately trained models (e.g., Behaviour vs Prediction) on Move 37:

```bash
uv run python scripts/compare_shapley_models.py \
  --behaviour_ckpt ./checkpoints/behaviour_run/checkpoint_10000 \
  --prediction_ckpt ./checkpoints/prediction_run/checkpoint_10000
```

## Running Tests

To verify the installation and functionality, run the test suite. You can disable W&B logging for tests using `--wandb-mode disabled`.

```bash
# Run all tests (disabling W&B logging)
for test in tests/test_*.py; do
    echo Running $test...
    uv run python $test --wandb-mode disabled || exit 1
done
```

Or run individual tests:

```bash
uv run python tests/test_checkpoint_resume.py --wandb-mode online
```

## Environment Variables

- `WANDB_API_KEY`: **Required**. The script will raise an error if this is not set. Pass it to the container using `-e WANDB_API_KEY=$WANDB_API_KEY`.
