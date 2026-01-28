import glob
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from einops import rearrange
from tqdm import tqdm

import wandb


@dataclass
class Args:
    data_path: str
    wandb_project: str = "fastsverl-tests"
    wandb_entity: str = "fastsverl"
    wandb_name: str = "dataset_analysis"
    max_files: int = 100  # Limit for large datasets


def unpack_binary_input(packed_input, pos_len=19):
    # packed_input: (N, C, packed_bytes)
    # Reimplement unpacking logic from prepare_katago_npz.py or similar
    # bits = jnp.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=jnp.uint8)
    # Using numpy for simple stats
    bits = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.uint8)

    # We only care about Stone positions for heatmap?
    # Stone positions are usually first channels?
    # Features.py: 0=1.0, 1=pla, 2=opp.
    # We can try to sum packed bits if we just want "activity", but for "Stone Position Heatmap", we need unpacked.

    # Let's fully unpack using numpy
    # (N, C, P) -> (N, C, P, 1) & (8) -> (N, C, P, 8)
    unpacked = (packed_input[..., None] & bits) > 0
    unpacked = unpacked.astype(np.float32)
    unpacked = rearrange(unpacked, "b c p b8 -> b c (p b8)")
    unpacked = unpacked[:, :, : pos_len * pos_len]
    unpacked = rearrange(unpacked, "b c (h w) -> b h w c", h=pos_len, w=pos_len)
    return unpacked


def process_file(path, accumulators, lengths, steps_proxy):
    try:
        data = np.load(path)
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return

    # Check format
    if "binaryInputNCHWPacked" in data:
        # KataGo Shuffled
        packed = data["binaryInputNCHWPacked"]
        # Shape (N, C, P)
        # Unpack
        unpacked = unpack_binary_input(packed)
        # Channels 0 (ones), 1 (pla), 2 (opp) usually?
        # features.features.py line 109: 0=1.0 constant.
        # 1=pla, 2=opp.
        # Heatmap of stones: sum(channel 1 + channel 2)
        # unpacked shape: (N, H, W, C)

        stones = unpacked[..., 1] + unpacked[..., 2]  # (N, H, W)
        accumulators["stones"] += np.sum(stones, axis=0)

        # Liberties
        # Ch 3: 1 liberty, Ch 4: 2 liberties
        accumulators["liberty_1"] += np.sum(unpacked[..., 3], axis=0)
        accumulators["liberty_2"] += np.sum(unpacked[..., 4], axis=0)

        # Ladders
        # Ch 14: Ladder capture, Ch 17: Ladder work
        accumulators["ladder_capture"] += np.sum(unpacked[..., 14], axis=0)
        accumulators["ladder_work"] += np.sum(unpacked[..., 17], axis=0)

        # History
        # Ch 9-13: Previous moves placeholders (just summing generic history presence)
        # Actually Ch 9 is prev1, Ch 10 is prev2, etc.
        history_sum = np.sum(unpacked[..., 9:14], axis=-1)  # (N, H, W)
        accumulators["history"] += np.sum(history_sum, axis=0)

        # Proxy for step: Number of stones
        stones_count = np.sum(stones, axis=(1, 2))  # (N,)
        steps_proxy.extend(stones_count.tolist())

        # Lengths: Not applicable for shuffled data
        # Unless we group by GameID if available?
        # Assuming shuffled -> No trajectory length info.
        pass

    elif "binaryInputNCHW" in data:
        # Our Trajectory format (converted SGF)
        unpacked = data["binaryInputNCHW"]
        # Shape (T, H, W, C)
        # Length
        lengths.append(unpacked.shape[0])

        stones = unpacked[..., 1] + unpacked[..., 2]
        accumulators["stones"] += np.sum(stones, axis=0)

        accumulators["liberty_1"] += np.sum(unpacked[..., 3], axis=0)
        accumulators["liberty_2"] += np.sum(unpacked[..., 4], axis=0)
        accumulators["ladder_capture"] += np.sum(unpacked[..., 14], axis=0)
        accumulators["ladder_work"] += np.sum(unpacked[..., 17], axis=0)

        history_sum = np.sum(unpacked[..., 9:14], axis=-1)
        accumulators["history"] += np.sum(history_sum, axis=0)

        # Proxy for step: Number of stones
        stones_count = np.sum(stones, axis=(1, 2))  # (N,)
        steps_proxy.extend(stones_count.tolist())

    else:
        print(f"Unknown format in {path}")


def main(args: Args):
    full_name = args.wandb_name

    # Derive suffix from data_path
    path_obj = Path(args.data_path)
    suffix = path_obj.stem

    if suffix:
        full_name = f"{full_name}--{suffix}"

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=full_name,
        config=vars(args),
    )

    # Find files
    if os.path.isfile(args.data_path):
        files = [args.data_path]
    elif os.path.isdir(args.data_path):
        # Recursive glob
        files = glob.glob(os.path.join(args.data_path, "**/*.npz"), recursive=True)
    else:
        print(f"Path not found: {args.data_path}")
        return

    files.sort()
    if args.max_files > 0:
        files = files[: args.max_files]

    print(f"Processing {len(files)} files...")

    accumulators = {
        "stones": np.zeros((19, 19), dtype=np.float32),
        "liberty_1": np.zeros((19, 19), dtype=np.float32),
        "liberty_2": np.zeros((19, 19), dtype=np.float32),
        "ladder_capture": np.zeros((19, 19), dtype=np.float32),
        "ladder_work": np.zeros((19, 19), dtype=np.float32),
        "history": np.zeros((19, 19), dtype=np.float32),
    }

    lengths = []
    steps_proxy = []

    for f in tqdm(files):
        process_file(f, accumulators, lengths, steps_proxy)

    # Log Heatmaps
    images = []
    for name, accum in accumulators.items():
        total = np.sum(accum)
        if total > 0:
            norm = accum / np.max(accum)
        else:
            norm = accum

        images.append(wandb.Image(norm, caption=f"{name} (N={len(files)})"))

    wandb.log({"heatmaps": images})
    print(f"Logged {len(images)} heatmaps: {list(accumulators.keys())}")

    # Log Histogram
    if lengths:
        wandb.log({"trajectory_lengths": wandb.Histogram(lengths)})
        print(f"Logged {len(lengths)} trajectories (length).")

    if steps_proxy:
        wandb.log({"step_proxy_stones": wandb.Histogram(steps_proxy)})
        print(f"Logged {len(steps_proxy)} steps (stone count proxy).")

    if not lengths and not steps_proxy:
        print("No stats found.")

    wandb.finish()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
