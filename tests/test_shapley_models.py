import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import tyro

import wandb
from core.networks.shapley import (
    BehaviourShapley,
    OutcomeShapley,
    PredictionShapley,
    ShapleyConfig,
)


@dataclass
class Args:
    model_type: Literal["behaviour", "outcome", "prediction", "all"] = "all"
    num_blocks: int = 2
    num_channels: int = 32
    num_mid_channels: int = 32
    blocks_ratio: float = 0.5
    batch_size: int = 2
    pos_len: int = 19
    num_features: int = 17
    num_actions: int = 362
    wandb_project: str = "fastsverl-tests"
    wandb_entity: str = "fastsverl"
    wandb_name: str = Path(__file__).stem


def test_behaviour_shapley(args: Args, config: ShapleyConfig, key):
    """Test BehaviourShapley model."""
    print("\n=== Testing BehaviourShapley ===")
    x = jnp.zeros((args.batch_size, args.pos_len, args.pos_len, args.num_features))

    model = BehaviourShapley(config=config, num_actions=args.num_actions)
    variables = model.init(key, x, train=False)
    phi = model.apply(variables, x, train=False)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {phi.shape}")

    expected_shape = (args.batch_size, args.pos_len, args.pos_len, args.num_actions)
    assert phi.shape == expected_shape, f"Expected {expected_shape}, got {phi.shape}"
    print("  ✓ Shape test passed")

    # Test single-action mode
    single_config = ShapleyConfig(
        num_blocks=args.num_blocks,
        num_channels=args.num_channels,
        num_mid_channels=args.num_mid_channels,
        blocks_ratio=args.blocks_ratio,
        multi_action=False,
    )
    single_model = BehaviourShapley(config=single_config, num_actions=args.num_actions)
    single_vars = single_model.init(key, x, train=False)
    single_phi = single_model.apply(single_vars, x, train=False)
    assert single_phi.shape == (args.batch_size, args.pos_len, args.pos_len, 1)
    print("  ✓ Single-action mode test passed")

    return {"behaviour_shape": phi.shape[3], "behaviour_single_passes": True}


def test_outcome_shapley(args: Args, config: ShapleyConfig, key):
    """Test OutcomeShapley model."""
    print("\n=== Testing OutcomeShapley ===")
    x = jnp.zeros((args.batch_size, args.pos_len, args.pos_len, args.num_features))

    model = OutcomeShapley(config=config)
    variables = model.init(key, x, train=False)
    phi = model.apply(variables, x, train=False)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {phi.shape}")

    expected_shape = (args.batch_size, args.pos_len, args.pos_len, 1)
    assert phi.shape == expected_shape, f"Expected {expected_shape}, got {phi.shape}"
    print("  ✓ Shape test passed")

    # Test efficiency constraint
    grand_val = jnp.array([0.8, -0.4])[:, None]  # (B, 1)
    null_val = jnp.array([0.1, 0.1])[:, None]  # (B, 1)

    eff_phi = model.apply(
        variables, x, train=False, grand_val=grand_val, null_val=null_val
    )
    phi_sum = eff_phi.sum(axis=(1, 2))
    target_sum = grand_val - null_val

    efficiency_gap = jnp.abs(phi_sum - target_sum).mean()
    print(f"  phi_sum: {phi_sum.squeeze()}")
    print(f"  target_sum: {target_sum.squeeze()}")
    print(f"  Efficiency gap: {efficiency_gap}")
    assert efficiency_gap < 1e-5, f"Efficiency gap too large: {efficiency_gap}"
    print("  ✓ Efficiency constraint passed")

    return {"outcome_efficiency_gap": float(efficiency_gap)}


def test_prediction_shapley(args: Args, config: ShapleyConfig, key):
    """Test PredictionShapley model."""
    print("\n=== Testing PredictionShapley ===")
    x = jnp.zeros((args.batch_size, args.pos_len, args.pos_len, args.num_features))

    model = PredictionShapley(config=config)
    variables = model.init(key, x, train=False)
    phi = model.apply(variables, x, train=False)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {phi.shape}")

    expected_shape = (args.batch_size, args.pos_len, args.pos_len, 1)
    assert phi.shape == expected_shape, f"Expected {expected_shape}, got {phi.shape}"
    print("  ✓ Shape test passed")

    # Test efficiency constraint
    grand_val = jnp.array([0.5, 0.3])[:, None]  # (B, 1)
    null_val = jnp.array([0.0, 0.0])[:, None]  # (B, 1)

    eff_phi = model.apply(
        variables, x, train=False, grand_val=grand_val, null_val=null_val
    )
    phi_sum = eff_phi.sum(axis=(1, 2))
    target_sum = grand_val - null_val

    efficiency_gap = jnp.abs(phi_sum - target_sum).mean()
    print(f"  Efficiency gap: {efficiency_gap}")
    assert efficiency_gap < 1e-5, f"Efficiency gap too large: {efficiency_gap}"
    print("  ✓ Efficiency constraint passed")

    return {"prediction_efficiency_gap": float(efficiency_gap)}


def main(args: Args):
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY not found in environment.")

    # Create config with multi_action=True for behaviour test
    config = ShapleyConfig(
        num_blocks=args.num_blocks,
        num_channels=args.num_channels,
        num_mid_channels=args.num_mid_channels,
        blocks_ratio=args.blocks_ratio,
        multi_action=True,
    )

    key = jax.random.PRNGKey(0)
    all_metrics = {}

    if args.model_type in ["behaviour", "all"]:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"{args.wandb_name}_behaviour",
            mode="online",
            reinit=True,
        )
        metrics = test_behaviour_shapley(args, config, key)
        all_metrics.update(metrics)
        wandb.log({**metrics, "success": True})
        wandb.finish()

    if args.model_type in ["outcome", "all"]:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"{args.wandb_name}_outcome",
            mode="online",
            reinit=True,
        )
        metrics = test_outcome_shapley(args, config, key)
        all_metrics.update(metrics)
        wandb.log({**metrics, "success": True})
        wandb.finish()

    if args.model_type in ["prediction", "all"]:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"{args.wandb_name}_prediction",
            mode="online",
            reinit=True,
        )
        metrics = test_prediction_shapley(args, config, key)
        all_metrics.update(metrics)
        wandb.log({**metrics, "success": True})
        wandb.finish()

    print("\n" + "=" * 40)
    print(f"All {args.model_type} tests passed!")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
