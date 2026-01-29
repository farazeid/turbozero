"""Verification test for checkpointing and resumption consistency."""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro

import wandb
from core.networks.katago import KataGoConfig
from core.networks.shapley import BehaviourShapley, ShapleyConfig
from core.training.checkpointing import (
    get_checkpoint_manager,
    load_checkpoint,
    save_checkpoint,
)
from core.training.data import KataGoDataLoader
from core.training.shapley_trainer import ShapleyTrainer


@dataclass
class Args:
    batch_size: int = 2
    lr: float = 1e-3
    save_dir: str = "./tmp_ckpt_test"
    shapley_type: str = "behaviour"
    wandb_project: str = "fastsverl-tests"
    wandb_entity: str = "fastsverl"
    wandb_name: str = Path(__file__).stem


def run_test(args: Args):
    if os.path.exists(args.save_dir):
        shutil.rmtree(args.save_dir)

    # Initialize W&B
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY not found in environment.")

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        mode="online",
    )

    # 1. Setup
    key = jax.random.PRNGKey(42)
    s_config = ShapleyConfig(num_blocks=2, num_channels=32, num_mid_channels=16)
    model = BehaviourShapley(config=s_config, num_actions=362)
    optimizer = optax.adam(args.lr)
    trainer = ShapleyTrainer(shapley_type=args.shapley_type, optimizer=optimizer)

    dummy_input = jnp.zeros((1, 19, 19, 22))
    dummy_global = jnp.zeros((1, 19))  # Assuming batch size 1 for init
    key, subkey = jax.random.split(key)
    initial_state = trainer.create_train_state(
        subkey, model, dummy_input, sample_global=dummy_global
    )

    # Use dummy data
    import glob

    data_dir = "./tests/data"
    npz_files = sorted(glob.glob(os.path.join(data_dir, "**/*.npz"), recursive=True))
    dataloader = KataGoDataLoader(npz_files, batch_size=args.batch_size)
    iterator = iter(dataloader)

    # 2. Train for 2 steps and save
    ckpt_manager = get_checkpoint_manager(args.save_dir)
    train_state = initial_state

    # Mock agent components
    def dummy_agent_fn(variables, x, train=False, **kwargs):
        # policy (362), value (3), ownership (19,19,1), misc (6)
        return (
            jnp.zeros((x.shape[0], 362)),
            jnp.zeros((x.shape[0], 3)),
            jnp.zeros((x.shape[0], 19, 19, 1)),
            jnp.zeros((x.shape[0], 6)),
        )

    agent_vars = {"params": {}}

    print("Training Phase 1 (Steps 1-2)...")
    for step in range(1, 3):
        batch = next(iterator)
        key, subkey = jax.random.split(key)
        train_state, metrics = trainer.train_step(
            train_state, dummy_agent_fn, agent_vars, batch, subkey
        )
        print(f"Step {step} loss: {metrics['loss']:.4f}")

    # Save at step 2
    save_checkpoint(ckpt_manager, train_state, dataloader.get_state(), 2, args, key)

    # 3. Continue to step 4
    print("Training Phase 1 Continue (Steps 3-4)...")
    for step in range(3, 5):
        batch = next(iterator)
        key, subkey = jax.random.split(key)
        train_state, metrics = trainer.train_step(
            train_state, dummy_agent_fn, agent_vars, batch, subkey
        )
        print(f"Step {step} loss: {metrics['loss']:.4f}")

    final_loss_continuous = metrics["loss"]

    # 4. Resume from step 2
    print("\nResuming from Step 2...")
    resume_manager = get_checkpoint_manager(args.save_dir)
    checkpoint_data = load_checkpoint(
        resume_manager, step=2, target_state=initial_state
    )

    # Re-init fresh components
    resume_dataloader = KataGoDataLoader(npz_files, batch_size=args.batch_size)
    resume_dataloader.load_state(checkpoint_data["dataloader_state"])
    resume_iterator = iter(resume_dataloader)
    resume_train_state = checkpoint_data["state"]
    resume_key = checkpoint_data["prng_key"]

    print("Training Phase 2 (Steps 3-4)...")
    for step in range(3, 5):
        batch = next(resume_iterator)
        resume_key, subkey = jax.random.split(resume_key)
        resume_train_state, metrics = trainer.train_step(
            resume_train_state, dummy_agent_fn, agent_vars, batch, subkey
        )
        print(f"Step {step} loss: {metrics['loss']:.4f}")

    final_loss_resumed = metrics["loss"]

    import sys

    print(f"\nFinal Continuous Loss: {final_loss_continuous:f}")
    print(f"Final Resumed Loss:    {final_loss_resumed:f}")

    diff = jnp.abs(final_loss_continuous - final_loss_resumed)
    rel_diff = diff / jnp.maximum(jnp.abs(final_loss_continuous), 1e-8)
    print(f"Difference: {diff:f} (Relative: {rel_diff:e})")

    # Tolerant check for GPU non-determinism
    success = rel_diff < 1e-4

    # Log to W&B
    wandb.log(
        {
            "final_loss_continuous": float(final_loss_continuous),
            "final_loss_resumed": float(final_loss_resumed),
            "loss_diff": float(diff),
            "loss_rel_diff": float(rel_diff),
            "success": success,
        }
    )

    if success:
        print("SUCCESS: Checkpoint/Resume logic verified.")
    else:
        print("FAILURE: Checkpoint/Resume resulted in significantly different loss.")
        wandb.finish()
        sys.exit(1)

    wandb.finish()


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_test(args)
