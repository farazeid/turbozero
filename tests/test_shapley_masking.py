import os
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import tyro

import wandb
from core.networks.shapley import (
    BehaviourShapley,
    ShapleyConfig,
)


@dataclass
class Args:
    num_blocks: int = 1
    num_channels: int = 16
    num_mid_channels: int = 16
    batch_size: int = 2
    pos_len: int = 19
    num_features: int = 17
    num_actions: int = 362
    wandb_project: str = "fastsverl-tests"
    wandb_entity: str = "fastsverl"
    wandb_name: str = Path(__file__).stem


def test_shapley_masking(args: Args):
    # Initialize W&B
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY not found in environment.")

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        mode="online",
    )

    config = ShapleyConfig(
        num_blocks=args.num_blocks,
        num_channels=args.num_channels,
        num_mid_channels=args.num_mid_channels,
        multi_action=False,  # Single action for simplicity
    )

    x = jax.random.normal(
        jax.random.PRNGKey(0),
        (args.batch_size, args.pos_len, args.pos_len, args.num_features),
    )
    key = jax.random.PRNGKey(1)

    # Create a mask with specific zeros
    mask = jnp.ones((args.batch_size, args.pos_len, args.pos_len, 1))
    # Mask out the center 5x5 area
    mask = mask.at[:, 7:12, 7:12, :].set(0.0)

    model = BehaviourShapley(config=config, num_actions=args.num_actions)
    variables = model.init(key, x, mask=mask, train=False)

    # 1. Verify that masked positions have ZERO attribution
    phi = model.apply(variables, x, mask=mask, train=False)

    masked_vals = phi[:, 7:12, 7:12, :]
    max_masked_val = jnp.abs(masked_vals).max()
    print(f"Max absolute value in masked area: {max_masked_val}")
    assert max_masked_val == 0.0, (
        f"Masked values should be 0, but got max {max_masked_val}"
    )

    # 2. Verify efficiency with masking
    grand_val = jnp.array([[1.0], [-1.0]])
    null_val = jnp.array([[0.0], [0.0]])

    eff_phi = model.apply(
        variables, x, mask=mask, train=False, grand_val=grand_val, null_val=null_val
    )

    # Masked values should still be zero even after efficiency correction
    masked_eff_vals = eff_phi[:, 7:12, 7:12, :]
    max_masked_eff_val = jnp.abs(masked_eff_vals).max()
    print(f"Max absolute value in masked area (after efficiency): {max_masked_eff_val}")
    assert max_masked_eff_val == 0.0

    # Sum of attributions should match grand_val - null_val
    phi_sum = eff_phi.sum(axis=(1, 2))
    print(f"Phi sum: {phi_sum.flatten()}")
    print(f"Target sum: {(grand_val - null_val).flatten()}")
    assert jnp.allclose(phi_sum, grand_val - null_val, atol=1e-5)

    # 3. Nullity Axiom: Changing masked features should not affect unmasked attributions
    # Create x2 by changing masked area of x
    x2 = x.at[:, 7:12, 7:12, :].add(10.0)
    phi2 = model.apply(variables, x2, mask=mask, train=False)

    # Difference in unmasked area should be zero
    diff = jnp.abs(phi - phi2)
    max_unmasked_diff = diff.max()
    print(
        f"Max difference in output when changing masked features: {max_unmasked_diff}"
    )
    # Note: Because of Global Pooling and BatchNorm, small numerical differences might exist,
    # but the architectural masking in NormMask and KataGPool should minimize this.
    assert max_unmasked_diff < 1e-5

    wandb.log({"max_masked_val": float(max_masked_val), "success": True})
    wandb.finish()
    print("Masking verification tests passed!")


if __name__ == "__main__":
    args = tyro.cli(Args)
    test_shapley_masking(args)
