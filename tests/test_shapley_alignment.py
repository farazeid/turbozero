import os
from dataclasses import dataclass, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import tyro

import wandb
from core.networks.shapley import (
    BehaviourShapley,
    OutcomeShapley,
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
    multi_action: bool = False
    wandb_project: str = "fastsverl-tests"
    wandb_entity: str = "fastsverl"
    wandb_name: str = Path(__file__).stem


def test_shapley_alignment(args: Args):
    # Initialize W&B
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY not found in environment.")

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        mode="online",
    )

    base_config = ShapleyConfig(
        num_blocks=args.num_blocks,
        num_channels=args.num_channels,
        num_mid_channels=args.num_mid_channels,
        blocks_ratio=args.blocks_ratio,
        multi_action=args.multi_action,
    )

    x = jnp.zeros((args.batch_size, args.pos_len, args.pos_len, args.num_features))
    key = jax.random.PRNGKey(0)

    # 1. Behaviour Shapley Alignment Check
    mode_str = "Multi" if args.multi_action else "Scalar"
    print(f"\n--- Checking BehaviourShapley ({mode_str} Mode) ---")

    b_model = BehaviourShapley(config=base_config, num_actions=args.num_actions)
    b_vars = b_model.init(key, x, train=False)
    b_phi = b_model.apply(b_vars, x, train=False)

    expected_outputs = args.num_actions if args.multi_action else 1
    assert b_phi.shape == (
        args.batch_size,
        args.pos_len,
        args.pos_len,
        expected_outputs,
    )
    print(f"Shape verified: {b_phi.shape}")

    # Efficiency Verification
    grand_val = jax.random.normal(key, (args.batch_size, expected_outputs))
    eff_phi = b_model.apply(b_vars, x, train=False, grand_val=grand_val)
    phi_sum = eff_phi.sum(axis=(1, 2))

    gap = jnp.abs(phi_sum - grand_val).mean()
    assert gap < 1e-4
    print(f"Efficiency verified. Gap: {gap}")

    # 2. Outcome/Prediction Scalar Verification
    print("\n--- Checking Outcome/Prediction Models ---")
    o_model = OutcomeShapley(config=base_config)
    o_vars = o_model.init(key, x, train=False)
    o_phi = o_model.apply(o_vars, x, train=False)
    assert o_phi.shape == (args.batch_size, args.pos_len, args.pos_len, 1)

    # Scalar efficiency check
    o_grand = jnp.array([1.0, -1.0])[:, None]
    o_eff_phi = o_model.apply(o_vars, x, train=False, grand_val=o_grand)
    o_phi_sum = o_eff_phi.sum(axis=(1, 2))
    assert jnp.allclose(o_phi_sum, o_grand, atol=1e-4)
    print("Scalar efficiency verified.")

    # Log to W&B
    wandb.log(
        {
            "success": True,
            "multi_action": args.multi_action,
            "efficiency_gap": float(gap),
        }
    )

    wandb.finish()


if __name__ == "__main__":
    base_args = tyro.cli(Args)

    # Run for both multi_action permutations
    for multi_action in [False, True]:
        args = replace(base_args, multi_action=multi_action)
        if multi_action:
            args = replace(args, wandb_name=f"{args.wandb_name}--multi-action")

        test_shapley_alignment(args)

    print("\nAll alignment tests passed for both modes!")
