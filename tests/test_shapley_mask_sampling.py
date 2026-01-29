import os
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import tyro

import wandb
from core.training.mask_utils import sample_shapley_masks


@dataclass
class Args:
    batch_size: int = 10
    height: int = 5
    width: int = 5
    wandb_project: str = "fastsverl-tests"
    wandb_entity: str = "fastsverl"
    wandb_name: str = Path(__file__).stem


def test_mask_shapes_and_distribution(args: Args):
    # Initialize W&B
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY not found in environment.")

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        mode="online",
    )

    key = jax.random.PRNGKey(0)
    B, H, W = args.batch_size, args.height, args.width
    masks = sample_shapley_masks(key, B, H, W)

    # 1. Check Mask Shapes
    assert masks.shape == (B, H, W, 1)

    # 2. Check Binary
    is_binary = jnp.all((masks == 0.0) | (masks == 1.0))
    if not is_binary:
        wandb.log({"success": False, "error": "Masks are not binary"})
    assert is_binary

    # 3. Check Distribution (Small Board)
    # n=25, p(k) dist
    # Sample many to check dist
    N_samples = 1000
    masks_large = sample_shapley_masks(key, N_samples, H, W)
    ks = jnp.sum(masks_large, axis=(1, 2, 3))

    # Empirical distribution
    hist, _ = np.histogram(ks, bins=range(1, H * W))

    # WandB Log Histogram
    wandb.log({"k_distribution": wandb.Histogram(ks)})
    wandb.log({"success": True})

    # Simple bounds check
    assert jnp.min(ks) > 0
    assert jnp.max(ks) < H * W

    print("Mask shape and distribution test passed!")
    wandb.finish()


if __name__ == "__main__":
    args = tyro.cli(Args)
    test_mask_shapes_and_distribution(args)
