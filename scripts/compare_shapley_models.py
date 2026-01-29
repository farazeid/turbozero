import os
import time
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import tyro
from flax.core import FrozenDict

import wandb
from core.evaluators.shapley_eval import ShapleyEvaluator
from core.networks.katago import KataGoConfig, KataGoNetwork
from core.networks.shapley import BehaviourShapley, OutcomeShapley, ShapleyConfig
from core.training.checkpointing import get_checkpoint_manager, load_checkpoint
from core.training.katago_loader import copy_params, load_katago_weights


@dataclass
class Args:
    # Checkpoints
    behaviour_ckpt: str
    prediction_ckpt: str
    agent_checkpoint: str = "./checkpoints/katago_frozen/model.bin.gz"

    # Data
    move37_path: str = "data/alphago_game2_move37.npz"
    m37_idx: int = 36

    # Agent Config (Must match the checkpoint)
    agent_blocks: int = 28
    agent_channels: int = 512
    agent_mid_channels: int = 256

    # Shapley Config
    shapley_blocks_ratio: float = 0.5

    # WandB
    wandb_project: str = "fastsverl-tests"
    wandb_entity: str = "fastsverl"
    wandb_name: str = "shapley_comparison"


def load_model(
    ckpt_path, shapley_type, agent_config, shapley_blocks_ratio, dummy_input, key
):
    shapley_config = ShapleyConfig(
        num_blocks=agent_config.num_blocks,
        blocks_ratio=shapley_blocks_ratio,
        num_channels=agent_config.num_channels,
        num_mid_channels=agent_config.num_mid_channels,
        multi_action=(shapley_type == "behaviour"),
    )

    if shapley_type == "behaviour":
        model = BehaviourShapley(config=shapley_config, num_actions=362)
    else:
        model = OutcomeShapley(config=shapley_config)

    variables = model.init(key, dummy_input, mask=None, train=False)

    # Load from checkpoint
    ckpt_manager = get_checkpoint_manager(ckpt_path)
    # We need a dummy train state to load into
    import optax
    from flax.training.train_state import TrainState

    state = TrainState.create(
        apply_fn=model.apply, params=variables["params"], tx=optax.identity()
    )

    loaded = load_checkpoint(ckpt_manager, target_state=state)
    if loaded:
        return loaded["state"]
    else:
        print(f"Warning: Could not load checkpoint from {ckpt_path}")
        return state


def main(args: Args):
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        config=vars(args),
    )

    key = jax.random.PRNGKey(42)

    # Load Agent config only
    _, loaded_config = load_katago_weights(args.agent_checkpoint)
    agent_config = KataGoConfig(
        num_blocks=loaded_config.get("num_blocks", args.agent_blocks),
        num_channels=loaded_config.get("c_trunk", args.agent_channels),
        num_mid_channels=loaded_config.get("c_mid", args.agent_mid_channels),
    )

    # 2. Load Data
    move37_data = np.load(args.move37_path)
    x = jnp.array(move37_data["binaryInputNCHW"][args.m37_idx : args.m37_idx + 1])
    action_taken = int(
        np.argmax(move37_data["policyTargetsNCMove"][args.m37_idx, 0, :])
    )

    # 3. Load Shapley Models
    dummy_input = jnp.zeros((1, 19, 19, 22))
    key, b_key, p_key = jax.random.split(key, 3)

    state_b = load_model(
        args.behaviour_ckpt,
        "behaviour",
        agent_config,
        args.shapley_blocks_ratio,
        dummy_input,
        b_key,
    )
    state_p = load_model(
        args.prediction_ckpt,
        "prediction",
        agent_config,
        args.shapley_blocks_ratio,
        dummy_input,
        p_key,
    )

    # 4. Compute Correlation
    evaluator = ShapleyEvaluator()
    correlation = evaluator.compute_correlations(state_b, state_p, x, action_taken)

    print(
        f"Spearman Rank Correlation (Behaviour vs Prediction) on Move 37: {correlation:.4f}"
    )

    wandb.log(
        {
            "comparison/move37_spearman_correlation": correlation,
        }
    )

    wandb.finish()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
