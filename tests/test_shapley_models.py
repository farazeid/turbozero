import os
from dataclasses import dataclass
from pathlib import Path

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


def test_shapley_models(args: Args):
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
        blocks_ratio=args.blocks_ratio,
    )

    x = jnp.zeros((args.batch_size, args.pos_len, args.pos_len, args.num_features))
    key = jax.random.PRNGKey(0)

    # 1. Behaviour Shapley
    b_model = BehaviourShapley(config=config, num_actions=args.num_actions)
    b_vars = b_model.init(key, x, train=False)
    b_phi = b_model.apply(b_vars, x, train=False)
    print(f"Behaviour phi shape: {b_phi.shape}")
    assert b_phi.shape == (
        args.batch_size,
        args.pos_len,
        args.pos_len,
        args.num_actions,
    )

    # 2. Outcome Shapley
    o_model = OutcomeShapley(config=config)
    o_vars = o_model.init(key, x, train=False)
    o_phi = o_model.apply(o_vars, x, train=False)
    print(f"Outcome phi shape: {o_phi.shape}")
    assert o_phi.shape == (args.batch_size, args.pos_len, args.pos_len, 1)

    # 3. Prediction Shapley
    p_model = PredictionShapley(config=config)
    p_vars = p_model.init(key, x, train=False)
    p_phi = p_model.apply(p_vars, x, train=False)
    print(f"Prediction phi shape: {p_phi.shape}")
    assert p_phi.shape == (args.batch_size, args.pos_len, args.pos_len, 1)

    # 4. Efficiency Constraint Verification
    grand_val = jnp.array([0.8, -0.4])[:, None]  # (B, 1)
    null_val = jnp.array([0.1, 0.1])[:, None]  # (B, 1)

    # Eval mode with efficiency on Outcome model
    eff_phi = o_model.apply(
        o_vars, x, train=False, grand_val=grand_val, null_val=null_val
    )
    phi_sum = eff_phi.sum(axis=(1, 2))
    target_sum = grand_val - null_val

    print(f"phi_sum: {phi_sum.squeeze()}")
    print(f"target_sum: {target_sum.squeeze()}")

    efficiency_gap = jnp.abs(phi_sum - target_sum).mean()
    assert efficiency_gap < 1e-5
    print("Efficiency constraint verified.")

    # Log to W&B
    wandb.log(
        {
            "efficiency_gap": float(efficiency_gap),
            "num_blocks": config.num_blocks,
            "blocks_ratio": config.blocks_ratio,
            "num_channels": config.num_channels,
            "success": True,
        }
    )

    wandb.finish()
    print("All tests passed!")


if __name__ == "__main__":
    args = tyro.cli(Args)
    test_shapley_models(args)
