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


@dataclass
class Args:
    num_blocks: int = 2
    num_channels: int = 32
    num_mid_channels: int = 32
    batch_size: int = 4
    pos_len: int = 19
    num_features: int = 17
    global_targets_nc: int = 64
    lr: float = 1e-3
    wandb_project: str = "fastsverl-tests"
    wandb_entity: str = "fastsverl"
    wandb_name: str = Path(__file__).stem


def test_katago_loss(args: Args):
    # Initialize W&B
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY not found in environment.")

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        mode="online",
    )

    config = KataGoConfig(
        num_blocks=args.num_blocks,
        num_channels=args.num_channels,
        num_mid_channels=args.num_mid_channels,
    )
    model = KataGoNetwork(config=config)

    x = jnp.zeros((args.batch_size, args.pos_len, args.pos_len, args.num_features))

    # Initialize variables
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

    # Create dummy batch
    batch = {
        "binaryInputNCHW": x,
        "policyTargetsNCMove": jnp.zeros(
            (args.batch_size, args.pos_len * args.pos_len + 1)
        ),
        "globalTargetsNC": jnp.zeros((args.batch_size, args.global_targets_nc)),
        "valueTargetsNCHW": jnp.zeros((args.batch_size, args.pos_len, args.pos_len, 1)),
        "scoreDistrN": jnp.zeros((args.batch_size, 1)),
    }

    # Compute loss
    loss, (aux_metrics, updates) = katago_loss_fn(params, train_state, batch)

    print(f"Loss: {loss}")
    print(f"Aux metrics: {aux_metrics}")

    assert loss > 0
    assert "policy_loss" in aux_metrics
    assert "value_loss" in aux_metrics
    assert "ownership_loss" in aux_metrics
    assert "score_loss" in aux_metrics

    # Log to W&B
    wandb.log(aux_metrics)
    wandb.log({"success": True})

    print("Loss test passed!")
    wandb.finish()


if __name__ == "__main__":
    args = tyro.cli(Args)
    test_katago_loss(args)
