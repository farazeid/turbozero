# KataGo Implementation in JAX/Flax

I have implemented the KataGo neural network architecture and data preparation pipeline. This allows for high-performance training and evaluation on Go 19x19, specifically designed for FastSVERL Shapley value estimation.

## Components & Invocation

### 1. Neural Network Architecture

The KataGo architecture is implemented in `core/networks/katago.py`. It includes bottleneck blocks, global pooling, and multi-head outputs (Policy, Value, Ownership, Score).

**Adjustable Parameters (`KataGoConfig`):**

- `num_blocks`: Number of residual blocks in the trunk (default: 20).
- `num_channels`: Number of filters in the main trunk (default: 128).
- `num_mid_channels`: Number of filters in the bottleneck middle layers (default: 128).
- `bnorm_momentum`: Batch normalization momentum (default: 0.99).

### 2. Data Preparation Pipeline

Official KataGo training data can be downloaded and processed for JAX.

- **Download**: `bash scripts/download_katago_data.sh --latest-n-days <N>`
  - Adjustable: `--latest-n-days` (Number of latest days to download).
- **Processing**: `scripts/prepare_katago_npz.py` handles JIT-compatible bit-unpacking and tensor transposition to NHWC.
  - Adjustable: `batch_size` and `pos_len` (Board size, default 19).

### 3. Shapley Models (FastSVERL)

The Shapley value estimation models are implemented in `core/networks/shapley.py`.

**Adjustable Parameters (`ShapleyConfig`):**

- `blocks_ratio`: Ratio of residual blocks compared to the trunk (default: 1.0).
- `multi_action`: Boolean toggle for behavior model. `False` (default) explains a single scalar output; `True` explains all 362 actions.

### 4. Verification (KataGo)

- `uv run python tests/test_katago_shapes.py`: Verify network output shapes.
- `uv run python tests/test_katago_params.py`: Check parameter counts.
- `uv run python tests/test_katago_loss.py`: Verify loss function logic.
- `uv run python tests/test_katago_offline.py`: End-to-end integration test with real KataGo data.

- `uv run python tests/test_shapley_models.py`: Basic shape and W&B logging check.
- `uv run python tests/test_shapley_alignment.py`: Rigorous check for Efficiency Axiom and spatial shapes in both Scalar/Multi modes.
- `uv run python tests/test_shapley_loss.py`: End-to-end loss function verification for all model types.

---

# Run

```bash
uv sync
```

```bash
uv run src/train.py  # --help
```

<br>
<br>
<br>
<br>
<br>
<br>

# _turbozero_ 🏎️ 🏎️ 🏎️ 🏎️

📣 If you're looking for the old PyTorch version of turbozero, it's been moved here: [turbozero_torch](https://github.com/lowrollr/turbozero_torch) 📣

#### _`turbozero`_ is a vectorized implementation of [AlphaZero](https://deepmind.google/discover/blog/alphazero-shedding-new-light-on-chess-shogi-and-go/) written in JAX

It contains:

- Monte Carlo Tree Search with subtree persistence
- Batched Replay Memory
- A complete, customizable training/evaluation loop

#### _`turbozero`_ is _*fast*_ and _*parallelized*_:

- every consequential part of the training loop is JIT-compiled
- parititions across multiple GPUs by default when available 🚀 NEW! 🚀
- self-play and evaluation episodes are batched/vmapped with hardware-acceleration in mind

#### _`turbozero`_ is _*extendable*_:

- see an [idea on twitter](https://twitter.com/ptrschmdtnlsn/status/1748800529608888362) for a simple tweak to MCTS?
  - [implement it](https://github.com/lowrollr/turbozero/blob/main/core/evaluators/mcts/weighted_mcts.py) then [test it](https://github.com/lowrollr/turbozero/blob/main/notebooks/weighted_mcts.ipynb) by extending core components

#### _`turbozero`_ is _*flexible*_:

- easy to integrate with you custom JAX environment or neural network architecture.
- Use the provided training and evaluation utilities, or pick and choose the components that you need.

To get started, check out the [Hello World Notebook](https://github.com/lowrollr/turbozero/blob/main/notebooks/hello_world.ipynb)

## Installation

`turbozero` uses `poetry` for dependency management, you can install it with:

```
pip install poetry==1.7.1
```

Then, to install dependencies:

```
poetry install
```

If you're using a GPU/TPU/etc., after running the previous command you'll need to install the device-specific version of JAX.

For a GPU w/ CUDA 12:

```
poetry source add jax https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

to point poetry towards JAX cuda releases, then use

```
poetry add jax[cuda12_pip]==0.4.35
```

to install the CUDA 12 release for JAX. See https://jax.readthedocs.io/en/latest/installation.html for other devices/cuda versions.

I have tested this project with CUDA 11 and CUDA 12.

To launch an ipython kernel, run:

```
poetry run python -m ipykernel install --user --name turbozero
```

## Issues

If you use this project and encounter an issue, error, or undesired behavior, please submit a [GitHub Issue](https://github.com/lowrollr/turbozero/issues) and I will do my best to resolve it as soon as I can. You may also contact me directly via `hello@jacob.land`.

## Contributing

Contributions, improvements, and fixes are more than welcome! For now I don't have a formal process for this, other than creating a [Pull Request](https://github.com/lowrollr/turbozero/pulls). For large changes, consider creating an [Issue](https://github.com/lowrollr/turbozero/issues) beforehand.

If you are interested in contributing but don't know what to work on, please reach out. I have plenty of things you could do.

## References

Papers/Repos I found helpful.

Repositories:

- [google-deepmind/mctx](https://github.com/google-deepmind/mctx): Monte Carlo tree search in JAX
- [sotetsuk/pgx](https://github.com/sotetsuk/pgx): Vectorized RL game environments in JAX
- [instadeepai/flashbax](https://github.com/instadeepai/flashbax): Accelerated Replay Buffers in JAX
- [google-deepmind/open_spiel](https://github.com/google-deepmind/open_spiel): RL algorithms

Papers:

- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815)
- [Revisiting Fundamentals of Experience Replay](https://arxiv.org/abs/2007.06700)

## Cite This Work

If you found this work useful, please cite it with:

```
@software{turbozero,
  author = {Marshall, Jacob},
  title = {{turbozero: fast + parallel AlphaZero}},
  url = {https://github.com/lowrollr/turbozero}
}
```
