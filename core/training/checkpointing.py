"""Robust checkpointing for Shapley training using Orbax for state and JSON for metadata."""

import json
import os
from typing import Any, Dict, Optional

import jax
import orbax.checkpoint as ocp
from flax.training import train_state

import wandb


def get_checkpoint_manager(
    checkpoint_dir: str, max_to_keep: int = 3
) -> ocp.CheckpointManager:
    """Returns an Orbax CheckpointManager for the TrainState."""
    options = ocp.CheckpointManagerOptions(max_to_keep=max_to_keep, create=True)
    return ocp.CheckpointManager(
        os.path.abspath(checkpoint_dir),
        ocp.StandardCheckpointer(),  # Handles the TrainState PyTree
        options=options,
    )


def save_checkpoint(
    ckpt_manager: ocp.CheckpointManager,
    state: train_state.TrainState,
    dataloader_state: Dict[str, Any],
    step: int,
    args: Any,
    prng_key: Any,
):
    """Saves a training checkpoint (State + Metadata)."""
    # 1. Save TrainState via Orbax
    ckpt_manager.save(step, state)
    ckpt_manager.wait_until_finished()

    # 2. Save Metadata via JSON in the checkpoint folder
    from dataclasses import asdict, is_dataclass

    args_dict = asdict(args) if is_dataclass(args) else args
    filtered_args = {
        k: v for k, v in args_dict.items() if isinstance(v, (int, float, bool, str))
    }

    metadata = {
        "dataloader_state": dataloader_state,
        "step": step,
        "args": filtered_args,
        "prng_key": jax.device_get(prng_key).tolist()
        if hasattr(prng_key, "device")
        else prng_key.tolist()
        if hasattr(prng_key, "tolist")
        else prng_key,
    }

    metadata_path = os.path.join(ckpt_manager.directory, str(step), "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Checkpoint and metadata saved at step {step}")


def load_checkpoint(
    ckpt_manager: ocp.CheckpointManager,
    step: Optional[int] = None,
    target_state: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """Loads a training checkpoint and its metadata."""
    if step is None:
        step = ckpt_manager.latest_step()

    if step is None:
        return None

    # 1. Restore TrainState
    # If target_state is provided, restore into it to get proper types
    if target_state:
        restored_state = ckpt_manager.restore(
            step, args=ocp.args.StandardRestore(target_state)
        )
    else:
        restored_state = ckpt_manager.restore(step)

    # 2. Restore Metadata
    metadata_path = os.path.join(ckpt_manager.directory, str(step), "metadata.json")
    if not os.path.exists(metadata_path):
        print(f"Warning: Metadata not found at {metadata_path}")
        return {"state": restored_state, "step": step}

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Convert prng_key back to jax array with correct dtype (uint32)
    if "prng_key" in metadata:
        metadata["prng_key"] = jax.numpy.array(
            metadata["prng_key"], dtype=jax.numpy.uint32
        )

    return {
        "state": restored_state,
        "dataloader_state": metadata.get("dataloader_state"),
        "step": metadata.get("step", step),
        "args": metadata.get("args"),
        "prng_key": metadata.get("prng_key"),
    }


def download_wandb_checkpoint(
    run_path: str,
    artifact_name: str = "checkpoint",
    download_dir: str = "./checkpoints/wandb",
) -> str:
    """Downloads a checkpoint artifact from W&B."""
    api = wandb.Api()
    if ":" not in artifact_name:
        artifact_name += ":latest"

    artifact = api.artifact(f"{run_path}/{artifact_name}")
    path = artifact.download(root=download_dir)
    print(f"Downloaded W&B artifact to {path}")
    return path
