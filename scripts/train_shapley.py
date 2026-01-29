import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
from core.training.checkpointing import (
    download_wandb_checkpoint,
    get_checkpoint_manager,
    load_checkpoint,
    save_checkpoint,
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
    batch_size: int = 64
    lr: float = 3e-4
    max_steps: int = 10000
    log_every: int = 100
    save_every: int = 5000
    seed: int = 42
    use_importance_sampling: bool = True  # Enable off-policy importance sampling
    save_dir: str = "./checkpoints/shapley"
    resume_from: Optional[str] = (
        None  # Local path or W&B run ID (entity/project/run_id)
    )

    # Networks
    shapley_type: str = "behaviour"  # behaviour, outcome, prediction
    # Agent Config (Must match the checkpoint if not loaded)
    agent_blocks: int = 28
    agent_channels: int = 512
    agent_mid_channels: int = 256

    # Shapley Config
    shapley_blocks_ratio: float = 0.5

    # WandB
    wandb_project: str = "fastsverl"
    wandb_entity: str = "fastsverl"
    wandb_name: str = "shapley_train"
    eval_every: int = 500
    move37_path: str = "data/alphago_game2_move37.npz"


def main(args: Args):
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"{args.wandb_name}_{args.shapley_type}",
        config=vars(args),
    )

    key = jax.random.PRNGKey(args.seed)

    # 1. Initialize Dataset
    print(f"Loading data from {args.data_dir}...")
    data_path = Path(args.data_dir)
    if data_path.is_file():
        npz_files = [str(data_path)]
    else:
        npz_files = sorted([str(p) for p in data_path.glob("*.npz")])

    if not npz_files:
        print("No .npz files found! Please verify data directory.")
        # We can continue for testing if needed, but warnings will be printed by loader

    dataloader = KataGoDataLoader(npz_files, batch_size=args.batch_size)

    # 2. Initialize Agent (Frozen)
    print("Initializing Agent...")

    agent_weights = None
    loaded_config = None

    if args.agent_checkpoint and os.path.exists(args.agent_checkpoint):
        print(f"Loading agent weights from {args.agent_checkpoint}...")
        agent_weights, loaded_config = load_katago_weights(args.agent_checkpoint)
        print(f"Loaded config: {loaded_config}")

        # Override args with loaded config
        # Mapping:
        # num_blocks -> agent_blocks
        # c_trunk -> agent_channels
        # c_mid -> agent_mid_channels
        # c_gpool -> ??? num_score_channels?

        agent_config = KataGoConfig(
            num_blocks=loaded_config.get("num_blocks", args.agent_blocks),
            num_channels=loaded_config.get("c_trunk", args.agent_channels),
            num_mid_channels=loaded_config.get("c_mid", args.agent_mid_channels),
            c_gpool=loaded_config.get("c_gpool", 64),
        )
    else:
        print(
            "WARNING: No agent checkpoint provided (or file not found). Using random agent weights."
        )
        agent_config = KataGoConfig(
            num_blocks=args.agent_blocks,
            num_channels=args.agent_channels,
            num_mid_channels=args.agent_mid_channels,
        )

    agent = KataGoNetwork(config=agent_config)

    key, a_key = jax.random.split(key)
    # Dummy input for init
    dummy_input = jnp.zeros((1, 19, 19, 22))
    agent_variables = agent.init(a_key, dummy_input, train=False)

    # Copy weights if loaded
    if agent_weights:
        print("Copying loaded weights to agent variables...")
        agent_variables = copy_params(agent_variables, agent_weights)
        print("Weights copied successfully.")

    # 3. Initialize Shapley Model
    print(f"Initializing {args.shapley_type} Shapley Model...")
    shapley_config = ShapleyConfig(
        num_blocks=min(10, agent_config.num_blocks // 2),  # Example scaling
        blocks_ratio=args.shapley_blocks_ratio,
        num_channels=agent_config.num_channels,
        num_mid_channels=agent_config.num_mid_channels,
        multi_action=(args.shapley_type == "behaviour"),
    )

    if args.shapley_type == "behaviour":
        shapley_model = BehaviourShapley(config=shapley_config, num_actions=362)
    elif args.shapley_type in ["outcome", "prediction"]:
        shapley_model = OutcomeShapley(config=shapley_config)
    else:
        raise ValueError(f"Unsupported shapley type: {args.shapley_type}")

    # 4. Initialize Trainer
    optimizer = optax.adam(args.lr)
    trainer = ShapleyTrainer(shapley_type=args.shapley_type, optimizer=optimizer)

    key, s_key = jax.random.split(key)
    # Create dummy global input for init
    dummy_global = jnp.zeros((1, 19))  # 19 global features
    train_state = trainer.create_train_state(
        s_key, shapley_model, dummy_input, sample_global=dummy_global
    )

    # 4.2 Initialize Evaluator and Load Move 37 data
    print(f"Initializing Evaluator and loading Move 37 data from {args.move37_path}...")
    evaluator = ShapleyEvaluator()
    move37_data = np.load(args.move37_path)
    # Move 37 is index 36 (0-indexed)
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

    # 4.5. Checkpointing Setup
    ckpt_manager = get_checkpoint_manager(args.save_dir)
    start_step = 0

    if args.resume_from:
        resume_path = args.resume_from
        if not os.path.exists(resume_path) and "/" in resume_path:
            # Try downloading from W&B
            print(f"Attempting to download checkpoint from W&B run: {resume_path}")
            resume_path = download_wandb_checkpoint(resume_path)

        print(f"Resuming from checkpoint: {resume_path}")
        # Need a manager for the resume path specifically if it's different
        resume_manager = get_checkpoint_manager(resume_path)
        checkpoint_data = load_checkpoint(resume_manager, target_state=train_state)

        if checkpoint_data:
            train_state = checkpoint_data["state"]
            dataloader.load_state(checkpoint_data["dataloader_state"])
            start_step = checkpoint_data["step"]
            print(f"Resumed at step {start_step}")
        else:
            print(
                f"Warning: No checkpoint found at {resume_path}. Starting from scratch."
            )

    # Check if we have data
    if not npz_files:
        print("Skipping training loop due to missing data.")
        wandb.finish()
        return

    # 5. Training Loop
    print("Starting training...")
    step = start_step
    iterator = iter(dataloader)

    # Telemetry tracking
    step_times = []
    epoch_start_time = time.time()
    log_window_start_time = time.time()
    log_window_steps = 0

    with tqdm(total=args.max_steps, initial=start_step) as pbar:
        while step < args.max_steps:
            step_start_time = time.time()

            try:
                batch = next(iterator)
            except StopIteration:
                dataloader.reset()
                iterator = iter(dataloader)
                batch = next(iterator)

            key, step_key = jax.random.split(key)

            train_state, metrics = trainer.train_step(
                train_state=train_state,
                agent_apply_fn=agent.apply,
                agent_variables=agent_variables,
                batch=batch,
                key=step_key,
                use_importance_sampling=args.use_importance_sampling,
            )

            # Telemetry: step timing
            step_end_time = time.time()
            step_duration = step_end_time - step_start_time
            step_times.append(step_duration)
            log_window_steps += 1

            # Logging every log_every steps
            if step % args.log_every == 0:
                # Calculate iterations/sec over the logging window
                log_window_duration = step_end_time - log_window_start_time
                if log_window_duration > 0:
                    iterations_per_sec = log_window_steps / log_window_duration
                else:
                    iterations_per_sec = 0.0

                # Calculate average step time over recent steps
                recent_step_times = step_times[-min(100, len(step_times)) :]
                avg_step_time = sum(recent_step_times) / len(recent_step_times)

                # Total elapsed time
                elapsed_time = step_end_time - epoch_start_time

                wandb.log(
                    {
                        # Training metrics
                        "train/step": step,
                        "train/loss": metrics["loss"],
                        "train/shapley_loss": metrics["shapley_loss"],
                        "train/l2_reg": metrics.get("l2_reg", 0.0),
                        "train/efficiency_gap": metrics.get("efficiency_gap", 0.0),
                        "train/grad_norm": metrics.get("grad_norm", 0.0),
                        "train/learning_rate": args.lr,
                        "train/mask_coverage": metrics.get("mask_coverage", 0.0),
                        # Telemetry
                        "telemetry/training_iterations_per_sec": iterations_per_sec,
                        "telemetry/seconds_per_step": avg_step_time,
                        "telemetry/total_iterations": step + 1,
                        "telemetry/elapsed_time_sec": elapsed_time,
                    }
                )
                pbar.set_postfix(
                    loss=f"{metrics['loss']:.2f}", ips=f"{iterations_per_sec:.1f}"
                )

                # Reset logging window
                log_window_start_time = step_end_time
                log_window_steps = 0

            # Evaluations every eval_every steps
            if step % args.eval_every == 0:
                eval_start_time = time.time()
                print(f"Running evaluation at step {step}...")

                # We reuse evaluate_move37 which returns image and metrics (axiom tests inside)
                eval_image, eval_metrics = evaluator.evaluate_move37(
                    train_state=train_state,
                    agent_apply_fn=agent.apply,
                    agent_variables=agent_variables,
                    move37_batch=move37_batch,
                    shapley_type=args.shapley_type,
                )

                eval_cycle_duration = time.time() - eval_start_time
                eval_metrics["telemetry/eval_cycle_duration_sec"] = eval_cycle_duration

                # Log to wandb
                wandb.log(
                    {
                        **eval_metrics,
                        f"eval/{args.shapley_type}_move37_image": eval_image,
                        "train/step": step,
                    }
                )
                print(f"Evaluation complete in {eval_cycle_duration:.2f}s")

            # Checkpointing
            if step % args.save_every == 0 and step > start_step:
                dataloader_state = dataloader.get_state()
                save_checkpoint(
                    ckpt_manager, train_state, dataloader_state, step, args, key
                )
                # Log checkpoint to W&B
                # wandb.save(f"{args.save_dir}/checkpoint_{step}/*")
                pass

            step += 1
            pbar.update(1)

    # Final telemetry
    total_time = time.time() - epoch_start_time
    final_iterations_per_sec = step / total_time if total_time > 0 else 0.0
    wandb.log(
        {
            "final/total_time_sec": total_time,
            "final/total_iterations": step,
            "final/avg_training_iterations_per_sec": final_iterations_per_sec,
        }
    )

    print(f"Training complete. Total time: {total_time:.2f}s, Iterations: {step}")
    wandb.finish()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
