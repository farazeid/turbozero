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
    num_blocks: int = 2
    num_channels: int = 64
    num_mid_channels: int = 64
    batch_size: int = 4
    pos_len: int = 19
    num_features: int = 17
    wandb_project: str = "fastsverl-tests"
    wandb_entity: str = "fastsverl"
    wandb_name: str = Path(__file__).stem


def test_katago_shapes(args: Args):
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

    # Dummy input: (batch, H, W, C)
    x = jnp.zeros((args.batch_size, args.pos_len, args.pos_len, args.num_features))

    # Initialize parameters
    key = jax.random.PRNGKey(0)
    params = model.init(key, x, train=False)

    # Forward pass
    policy, value, ownership, score = model.apply(params, x, train=False)

    # Verify shapes
    print(f"Policy shape: {policy.shape}")
    print(f"Value shape: {value.shape}")
    print(f"Ownership shape: {ownership.shape}")
    print(f"Score shape: {score.shape}")

    assert policy.shape == (args.batch_size, args.pos_len * args.pos_len + 1)
    assert value.shape == (args.batch_size, 1)
    assert ownership.shape == (args.batch_size, args.pos_len, args.pos_len, 1)
    assert score.shape == (args.batch_size, 1)

    # Log to W&B
    wandb.log(
        {
            "policy_shape": policy.shape[1],
            "value_shape": value.shape[1],
            "ownership_shape": args.pos_len,
            "score_shape": score.shape[1],
            "success": True,
        }
    )

    print("Shape test passed!")
    wandb.finish()


if __name__ == "__main__":
    args = tyro.cli(Args)
    test_katago_shapes(args)
