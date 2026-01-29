import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from tqdm import tqdm

import wandb
from core.evaluators.shapley_eval import ShapleyEvaluator
from core.networks.katago import KataGoConfig, KataGoNetwork
from core.networks.shapley import (
    BehaviourShapley,
    OutcomeShapley,
    PredictionShapley,
    ShapleyConfig,
)
from core.training.data import KataGoDataLoader
from core.training.katago_loader import copy_params, load_katago_weights
from core.training.shapley_trainer import ShapleyTrainer


@dataclass
class Args:
    # Data
    data_dir: str = "./data/katago_trajectories"
    agent_checkpoint: Optional[str] = (
        "./checkpoints/katago_frozen/model.bin.gz"  # Path to KataGo checkpoint
    )

    # Training
    batch_size: int = 128  # Increased to improve GPU utilization
    lr: float = 3e-4
    max_steps: int = 10000
    log_every: int = 100
    save_every: int = 5000
    seed: int = 42
    use_importance_sampling: bool = True
    save_dir: str = "./checkpoints/shapley_parallel"
    resume_from: Optional[str] = None

    # Models to train
    types: List[str] = field(
        default_factory=lambda: ["behaviour", "prediction", "outcome"]
    )

    # Agent Config
    agent_blocks: int = 28
    agent_channels: int = 512
    agent_mid_channels: int = 256

    # Shapley Config
    shapley_blocks_ratio: float = 0.5

    # WandB
    wandb_project: str = "fastsverl"
    wandb_entity: str = "fastsverl"
    wandb_name: str = "shapley_parallel"
    eval_every: int = 500
    move37_path: str = "data/alphago_game2_move37.npz"


def main(args: Args):
    # 0. Enforce WANDB_API_KEY
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError(
            "WANDB_API_KEY not found in environment. "
            "Please set it or run with -e WANDB_API_KEY=$WANDB_API_KEY"
        )

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        config=vars(args),
    )

    key = jax.random.PRNGKey(args.seed)

    # 1. Initialize Dataset
    print(f"Loading data from {args.data_dir}...")
    data_path = Path(args.data_dir)
    if data_path.is_file():
        npz_files = [str(data_path)]
    else:
        npz_files = sorted([str(p) for p in data_path.rglob("*.npz")])

    if not npz_files:
        print("No .npz files found!")

    dataloader = KataGoDataLoader(npz_files, batch_size=args.batch_size)

    # 2. Initialize Agent (Frozen)
    print("Initializing Agent...")
    agent_weights, loaded_config = load_katago_weights(args.agent_checkpoint)
    agent_config = KataGoConfig(
        num_blocks=loaded_config.get("num_blocks", args.agent_blocks),
        num_channels=loaded_config.get("c_trunk", args.agent_channels),
        num_mid_channels=loaded_config.get("c_mid", args.agent_mid_channels),
    )
    agent = KataGoNetwork(config=agent_config)

    key, a_key = jax.random.split(key)
    dummy_input = jnp.zeros((1, 19, 19, 22))
    dummy_global = jnp.zeros((1, 19))
    agent_variables = agent.init(
        a_key, dummy_input, global_input=dummy_global, train=False
    )
    agent_variables = copy_params(agent_variables, agent_weights)

    # 3. Initialize Multiple Shapley Models
    trainers = {}
    train_states = {}

    shapley_config = ShapleyConfig(
        num_blocks=agent_config.num_blocks,
        blocks_ratio=args.shapley_blocks_ratio,
        num_channels=agent_config.num_channels,
        num_mid_channels=agent_config.num_mid_channels,
    )

    for stype in args.types:
        print(f"Initializing {stype} Shapley Model...")
        if stype == "behaviour":
            model = BehaviourShapley(
                config=replace(shapley_config, multi_action=True), num_actions=362
            )
        elif stype == "outcome":
            model = OutcomeShapley(config=shapley_config)
        elif stype == "prediction":
            model = PredictionShapley(config=shapley_config)
        else:
            continue

        optimizer = optax.adam(args.lr)
        trainer = ShapleyTrainer(shapley_type=stype, optimizer=optimizer)

        key, s_key = jax.random.split(key)
        train_state = trainer.create_train_state(
            s_key, model, dummy_input, sample_global=dummy_global
        )

        trainers[stype] = trainer
        train_states[stype] = train_state

    # 4. Evaluator
    evaluator = ShapleyEvaluator()
    move37_data = np.load(args.move37_path)
    m37_idx = 36
    move37_batch = {
        "binaryInputNCHW": jnp.array(
            move37_data["binaryInputNCHW"][m37_idx : m37_idx + 1]
        ),
        "globalInputNC": jnp.array(move37_data["globalInputNC"][m37_idx : m37_idx + 1]),
        "action_taken": jnp.array(
            np.argmax(
                move37_data["policyTargetsNCMove"][m37_idx : m37_idx + 1, 0, :], axis=-1
            )
        ),
    }

    # 5. Training Loop
    print("Starting parallel training...")
    step = 0
    iterator = iter(dataloader)
    epoch_start_time = time.time()

    with tqdm(total=args.max_steps) as pbar:
        while step < args.max_steps:
            try:
                batch = next(iterator)
            except StopIteration:
                dataloader.reset()
                iterator = iter(dataloader)
                batch = next(iterator)

            step_metrics = {}
            for stype in args.types:
                key, step_key = jax.random.split(key)
                train_states[stype], metrics = trainers[stype].train_step(
                    train_state=train_states[stype],
                    agent_apply_fn=agent.apply,
                    agent_variables=agent_variables,
                    batch=batch,
                    key=step_key,
                    use_importance_sampling=args.use_importance_sampling,
                )
                for k, v in metrics.items():
                    step_metrics[f"{stype}/{k}"] = v

            if step % args.log_every == 0:
                wandb.log({"step": step, **step_metrics})
                pbar.set_postfix(step=step)

            if step % args.eval_every == 0:
                print(f"Evaluating at step {step}...")
                eval_metrics_combined = {}

                # Plot Move 37 image for each type
                for stype in args.types:
                    image, eval_m = evaluator.evaluate_move37(
                        train_state=train_states[stype],
                        agent_apply_fn=agent.apply,
                        agent_variables=agent_variables,
                        move37_batch=move37_batch,
                        shapley_type=stype,
                    )
                    eval_metrics_combined[f"eval/{stype}_move37_image"] = image
                    for k, v in eval_m.items():
                        eval_metrics_combined[k] = v

                # Spearman Rank Correlation (Behaviour vs Prediction)
                if "behaviour" in train_states and "prediction" in train_states:
                    corr = evaluator.compute_correlations(
                        train_states["behaviour"],
                        train_states["prediction"],
                        move37_batch["binaryInputNCHW"],
                        move37_batch["globalInputNC"],
                        move37_batch["action_taken"][0],
                    )
                    eval_metrics_combined["eval/spearman_behaviour_prediction"] = corr
                    print(f"Spearman Correlation: {corr:.4f}")

                wandb.log({"step": step, **eval_metrics_combined})

            step += 1
            pbar.update(1)

    wandb.finish()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
