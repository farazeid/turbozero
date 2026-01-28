import os
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import tyro

import wandb
from core.networks.katago import KataGoConfig, KataGoNetwork


@dataclass
class Args:
    num_blocks: int = 20
    num_channels: int = 128
    num_mid_channels: int = 128
    batch_size: int = 1
    pos_len: int = 19
    num_features: int = 17
    wandb_project: str = "fastsverl-tests"
    wandb_entity: str = "fastsverl"
    wandb_name: str = Path(__file__).stem


def count_params(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def test_katago_params(args: Args):
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

    key = jax.random.PRNGKey(0)
    params = model.init(key, x, train=False)

    num_params = count_params(params)
    print(f"Total parameters: {num_params:,}")

    # Log to W&B
    wandb.log(
        {
            "num_params": num_params,
            "num_blocks": config.num_blocks,
            "num_channels": config.num_channels,
            "success": True,
        }
    )

    wandb.finish()


if __name__ == "__main__":
    args = tyro.cli(Args)
    test_katago_params(args)
