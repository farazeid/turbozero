import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import optax
import tyro
from flax.training.train_state import TrainState

import wandb
from core.networks.shapley import (
    BehaviourShapley,
    OutcomeShapley,
    PredictionShapley,
    ShapleyConfig,
)
from core.training.loss_fns import shapley_loss_fn


class TrainStateWithStats(TrainState):
    batch_stats: Any


@dataclass
class Args:
    num_blocks: int = 1
    num_channels: int = 16
    num_mid_channels: int = 16
    batch_size: int = 4
    pos_len: int = 19
    num_features: int = 17
    num_actions: int = 362
    wandb_project: str = "fastsverl-tests"
    wandb_entity: str = "fastsverl"
    wandb_name: str = Path(__file__).stem


def run_test_for_model(
    name: str, model, config: ShapleyConfig, args: Args, key: jax.random.PRNGKey
):
    print(f"\n--- Testing Loss for {name} ---")

    # Initialize model
    x = jnp.zeros((args.batch_size, args.pos_len, args.pos_len, args.num_features))
    variables = model.init(key, x, train=False)

    # Create TrainState
    tx = optax.adam(1e-3)
    train_state = TrainStateWithStats.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
        batch_stats=variables.get("batch_stats", {}),
    )

    # Prepare dummy batch
    # num_outputs logic
    num_outputs = (
        args.num_actions
        if (isinstance(model, BehaviourShapley) and config.multi_action)
        else 1
    )

    coalition_mask = jax.random.bernoulli(
        key, 0.5, (args.batch_size, args.pos_len, args.pos_len, 1)
    ).astype(jnp.float32)
    target_char_vals = jax.random.normal(key, (args.batch_size, num_outputs))
    null_char_vals = jax.random.normal(key, (args.batch_size, num_outputs))

    batch = {
        "observation": x,
        "coalition_mask": coalition_mask,
        "target_char_vals": target_char_vals,
        "null_char_vals": null_char_vals,
    }

    # Compute loss
    loss, (aux, updates) = shapley_loss_fn(train_state.params, train_state, batch)

    print(f"Loss: {loss:.4f}")
    print(f"Aux: {aux}")

    assert loss > 0
    assert "shapley_loss" in aux

    # Log to W&B
    wandb.log(
        {
            f"{name}_loss": float(loss),
            f"{name}_shapley_loss": float(aux["shapley_loss"]),
        }
    )


def test_shapley_losses(args: Args):
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
    )

    key = jax.random.PRNGKey(0)

    # 1. Behaviour (Scalar - Default)
    conf_b_scalar = replace(base_config, multi_action=False)
    model_b_scalar = BehaviourShapley(
        config=conf_b_scalar, num_actions=args.num_actions
    )
    run_test_for_model("Behaviour_Scalar", model_b_scalar, conf_b_scalar, args, key)

    # 2. Behaviour (Multi-Action)
    conf_b_multi = replace(base_config, multi_action=True)
    model_b_multi = BehaviourShapley(config=conf_b_multi, num_actions=args.num_actions)
    run_test_for_model("Behaviour_Multi", model_b_multi, conf_b_multi, args, key)

    # 3. Outcome
    model_o = OutcomeShapley(config=base_config)
    run_test_for_model("Outcome", model_o, base_config, args, key)

    # 4. Prediction
    model_p = PredictionShapley(config=base_config)
    run_test_for_model("Prediction", model_p, base_config, args, key)

    wandb.finish()
    print("\nAll loss tests passed!")


if __name__ == "__main__":
    args = tyro.cli(Args)
    test_shapley_losses(args)
