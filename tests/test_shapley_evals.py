import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro

import wandb
from core.evaluators.shapley_eval import ShapleyEvaluator
from core.networks.katago import KataGoConfig, KataGoNetwork
from core.networks.shapley import (
    BehaviourShapley,
    OutcomeShapley,
    PredictionShapley,
    ShapleyConfig,
)
from core.training.shapley_trainer import ShapleyTrainState


@dataclass
class Args:
    model_type: Literal["behaviour", "outcome", "prediction", "all"] = "all"
    move37_path: str = "data/alphago_game2_move37.npz"
    wandb_project: str = "fastsverl-tests"
    wandb_entity: str = "fastsverl"
    wandb_name: str = Path(__file__).stem


def load_move37(path):
    data = np.load(path)
    # Move 37 is index 36 (0-indexed)
    idx = 36
    batch = {
        "binaryInputNCHW": jnp.array(data["binaryInputNCHW"][idx : idx + 1]),
        "action_taken": jnp.array(
            np.argmax(data["policyTargetsNCMove"][idx : idx + 1, 0, :], axis=-1)
        ),
    }
    return batch


def run_test(args: Args, shapley_type: str, key):
    print(f"=== Testing {shapley_type} Evaluation ===")

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"{args.wandb_name}_{shapley_type}",
        reinit=True,
    )

    # 1. Setup Models (Small versions for testing)
    agent_config = KataGoConfig(num_blocks=2, num_channels=32, num_mid_channels=32)
    agent = KataGoNetwork(config=agent_config)

    shapley_config = ShapleyConfig(
        num_blocks=1,
        num_channels=32,
        num_mid_channels=32,
        multi_action=(shapley_type == "behaviour"),
    )

    if shapley_type == "behaviour":
        shapley_model = BehaviourShapley(config=shapley_config, num_actions=362)
    elif shapley_type == "outcome":
        shapley_model = OutcomeShapley(config=shapley_config)
    else:  # prediction
        shapley_model = PredictionShapley(config=shapley_config)

    # Initialize variables
    dummy_x = jnp.zeros((1, 19, 19, 22))
    key, a_key, s_key = jax.random.split(key, 3)
    agent_variables = agent.init(a_key, dummy_x, train=False)

    dummy_global = jnp.zeros((1, 19))
    shapley_variables = shapley_model.init(
        s_key, dummy_x, global_input=dummy_global, mask=None, train=False
    )
    train_state = ShapleyTrainState.create(
        apply_fn=shapley_model.apply,
        params=shapley_variables["params"],
        tx=optax.identity(),  # No optimizer needed for eval
        batch_stats=shapley_variables.get("batch_stats"),
    )

    # 2. Load Move 37
    move37_batch = load_move37(args.move37_path)

    # 3. Evaluate
    evaluator = ShapleyEvaluator()
    image, metrics = evaluator.evaluate_move37(
        train_state=train_state,
        agent_apply_fn=agent.apply,
        agent_variables=agent_variables,
        move37_batch=move37_batch,
        shapley_type=shapley_type,
    )

    # 4. Log
    wandb.log({**metrics, f"eval/{shapley_type}_image": image})
    print(f"Logged {shapley_type} metrics and image to WandB.")

    wandb.finish()


def main(args: Args):
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY not found in environment.")

    key = jax.random.PRNGKey(0)

    types = (
        ["behaviour", "outcome", "prediction"]
        if args.model_type == "all"
        else [args.model_type]
    )

    for t in types:
        key, step_key = jax.random.split(key)
        run_test(args, t, step_key)


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
