import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import optax
import tyro
from flax.training.train_state import TrainState

import wandb
from core.networks.katago import KataGoConfig, KataGoNetwork
from core.training.loss_fns import katago_loss_fn
from scripts.prepare_katago_npz import KataGoDataLoader


@dataclass
class Args:
    num_blocks: int = 6
    num_channels: int = 64
    num_mid_channels: int = 64
    batch_size: int = 4
    pos_len: int = 19
    lr: float = 1e-3
    wandb_project: str = "fastsverl-tests"
    wandb_entity: str = "fastsverl"
    wandb_name: str = Path(__file__).stem


def test_katago_offline(args: Args):
    # Initialize W&B
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY not found in environment.")

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        mode="online",
    )

    # Configure model
    config = KataGoConfig(
        num_blocks=args.num_blocks,
        num_channels=args.num_channels,
        num_mid_channels=args.num_mid_channels,
    )
    model = KataGoNetwork(config=config)

    # Find NPZ files
    npz_files = glob.glob("data/katago/**/*.npz", recursive=True)
    if not npz_files:
        raise FileNotFoundError("No NPZ files found in data/katago")

    print(f"Found {len(npz_files)} NPZ files. Using the first one for testing.")
    loader = KataGoDataLoader(
        [npz_files[0]], batch_size=args.batch_size, pos_len=args.pos_len
    )

    # Get a batch
    batch = next(iter(loader))
    for k, v in batch.items():
        print(f"Batch {k} shape: {v.shape}")
    x = batch["binaryInputNCHW"]

    # Initialize parameters
    key = jax.random.PRNGKey(0)
    variables = model.init(key, x, train=False)
    params = variables["params"]
    batch_stats = variables.get("batch_stats", {})

    # Create TrainState
    class KataGoTrainState(TrainState):
        batch_stats: Any

    tx = optax.adam(args.lr)
    train_state = KataGoTrainState.create(
        apply_fn=model.apply, params=params, tx=tx, batch_stats=batch_stats
    )

    # Define grad function
    def compute_loss(params, state, batch):
        loss, (aux, updates) = katago_loss_fn(params, state, batch)
        return loss, aux

    grad_fn = jax.value_and_grad(compute_loss, has_aux=True)

    # Compute gradients
    (loss, aux), grads = grad_fn(params, train_state, batch)

    print(f"Offline Loss: {loss}")
    print(f"Offline Aux metrics: {aux}")

    # Verify gradients are non-zero
    grad_norm = jnp.sqrt(
        sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads))
    )
    print(f"Gradient norm: {grad_norm}")

    assert loss > 0
    assert grad_norm > 0

    # Log to W&B
    wandb.log(aux)
    wandb.log({"grad_norm": grad_norm, "success": True})

    print("Offline integration test passed!")
    wandb.finish()


if __name__ == "__main__":
    args = tyro.cli(Args)
    test_katago_offline(args)
