# FastSVERL: Shapley Value Estimation for Go 19x19

This repository implements FastSVERL for estimating Shapley values on Go 19x19, using KataGo as the frozen agent.

## Quick Start

```bash
# Inside Docker container
hare run --rm -it -v .:/app -e WANDB_API_KEY=$WANDB_API_KEY --gpus device=0 $USER/turbozero bash

# Run training
uv run python scripts/train_shapley.py \
  --data_dir ./data/katago/2026-01-25npzs/kata1-b28c512nbt-s12313658112-d5687582971 \
  --max_steps 1000
```

---

## Components & Invocation

### 1. KataGo Neural Network Architecture

The KataGo b28c512nbt architecture is implemented in `core/networks/katago.py`.

**Architecture Features:**

- Nested bottleneck residual blocks with global pooling
- Multi-head outputs: Policy (362), Value (3: win/loss/draw), Ownership (19×19), Misc (6)
- Exact match with official KataGo checkpoint structure

**Adjustable Parameters (`KataGoConfig`):**

| Parameter          | Default | Description                                      |
| ------------------ | ------- | ------------------------------------------------ |
| `num_blocks`       | 28      | Number of nested bottleneck blocks               |
| `num_channels`     | 512     | Trunk channel width (c_trunk)                    |
| `num_mid_channels` | 256     | Bottleneck internal width (c_mid)                |
| `c_gpool`          | 64      | Global pooling channels                          |
| `gpool_start_idx`  | 2       | First block with global pooling                  |
| `gpool_interval`   | 3       | Interval between gpool blocks (2, 5, 8, 11, ...) |
| `internal_length`  | 2       | ResBlocks per nested block                       |

### 2. Shapley Models (FastSVERL)

Three explanation models in `core/networks/shapley.py`:

| Model               | Output                | Explains                      |
| ------------------- | --------------------- | ----------------------------- |
| `BehaviourShapley`  | (B, 19, 19, 1 or 362) | Agent's action selection      |
| `OutcomeShapley`    | (B, 19, 19, 1)        | True game outcome             |
| `PredictionShapley` | (B, 19, 19, 1)        | Agent's internal win estimate |

**Adjustable Parameters (`ShapleyConfig`):**

| Parameter      | Default | Description                                 |
| -------------- | ------- | ------------------------------------------- |
| `blocks_ratio` | 0.5     | Fraction of agent's blocks to use           |
| `multi_action` | False   | If True, explain all 362 actions separately |
| `num_blocks`   | 28      | Base block count (same as agent)            |
| `num_channels` | 512     | Channel width (same as agent)               |

### 3. Training Script

Main training entry point: `scripts/train_shapley.py`

**Usage:**

```bash
uv run python scripts/train_shapley.py [OPTIONS]
```

**Key Arguments:**

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

**Telemetry Logged to W&B:**

- `train/loss`, `train/shapley_loss`, `train/l2_reg`
- `train/efficiency_gap`: Distance from satisfied efficiency axiom
- `train/grad_norm`, `train/mask_coverage`
- `telemetry/training_iterations_per_sec`, `telemetry/seconds_per_step`
- `telemetry/total_iterations`, `telemetry/elapsed_time_sec`
- `final/total_time_sec`, `final/avg_training_iterations_per_sec`

### 4. Data Preparation

**Download KataGo Training Data:**

```bash
bash scripts/download_katago_data.sh --latest-n-days 1
```

**Download KataGo Checkpoint:**

```bash
bash scripts/download_katago_checkpoint.sh
```

**Convert SGF to Trajectory:**

```bash
uv run python scripts/sgf_to_trajectory.py \
  --sgf data/alphago_lee_sedol_game2.sgf \
  --output data/alphago_game2.npz
```

### 5. Tests

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

### 6. Reproducibility Guide

#### Step 1: Prepare Environment

```bash
hare run --rm -it -v .:/app -e WANDB_API_KEY=$WANDB_API_KEY --gpus device=0 $USER/turbozero bash
```

#### Step 2: Download Data & Checkpoint

```bash
bash scripts/download_katago_data.sh --latest-n-days 1
bash scripts/download_katago_checkpoint.sh
```

#### Step 3: Train Shapley Models

````bash
# Behaviour Shapley
uv run python scripts/train_shapley.py --shapley_type behaviour

# Outcome Shapley
uv run python scripts/train_shapley.py --shapley_type outcome

# Prediction Shapley
uv run python scripts/train_shapley.py --shapley_type prediction

# Resume from local checkpoint
uv run python scripts/train_shapley.py --resume_from ./checkpoints/shapley/5000

# Resume from W&B run
uv run python scripts/train_shapley.py --resume_from entity/project/run_id

#### Step 4: Analyze AlphaGo Move 37

```bash
# Prepare Game 2 data
./scripts/prepare_alphago_game2.sh

# Create Move 37 subset
uv run python scripts/sgf_to_trajectory.py \
  --sgf data/alphago_lee_sedol_game2.sgf \
  --output data/alphago_game2_move37.npz \
  --limit-moves 37

# Log dataset statistics
uv run python scripts/log_dataset_stats.py --data-path data/alphago_game2_move37.npz
````
