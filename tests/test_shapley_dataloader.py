import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

import wandb
from core.training.data import KataGoDataLoader


@dataclass
class Args:
    batch_size: int = 4
    data_size: int = 20
    wandb_project: str = "fastsverl-tests"
    wandb_entity: str = "fastsverl"
    wandb_name: str = Path(__file__).stem


# Create a dummy npz file for testing
def create_dummy_npz(path, size=10):
    data = {
        "binaryInputNCHWPacked": np.random.randint(
            0, 255, (size, 22, 46), dtype=np.uint8
        ),  # 19x19 packed
        "globalInputNC": np.random.randn(size, 19),
        "policyTargetsNCMove": np.random.randn(size, 2, 362),
        "globalTargetsNC": np.random.randn(size, 60),
        "scoreDistrN": np.random.randn(size, 400),
        "valueTargetsNCHW": np.random.randn(size, 1, 19, 19),
    }
    np.savez(path, **data)


def test_dataloader(args: Args):
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY not found in environment.")

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        mode="online",
    )

    # Setup temp dir
    tmp_path = Path("./tests/temp_data")
    tmp_path.mkdir(parents=True, exist_ok=True)

    try:
        dummy_path = tmp_path / "test_katago.npz"
        create_dummy_npz(dummy_path, size=args.data_size)

        loader = KataGoDataLoader([str(dummy_path)], batch_size=args.batch_size)

        # Iterate
        count = 0
        for batch in loader:
            count += 1
            assert "binaryInputNCHW" in batch
            assert batch["binaryInputNCHW"].shape == (args.batch_size, 19, 19, 22)
            assert batch["globalTargetsNC"].shape == (args.batch_size, 60)

            # Simple stats check
            mean_val = np.mean(batch["binaryInputNCHW"])
            wandb.log({"batch_mean": mean_val, "batch_idx": count})

        expected_batches = args.data_size // args.batch_size
        assert count == expected_batches

        wandb.log({"success": True, "batches_processed": count})
        print(f"Propcessed {count} batches.")

    finally:
        # Cleanup
        if tmp_path.exists():
            shutil.rmtree(tmp_path)

    wandb.finish()


if __name__ == "__main__":
    args = tyro.cli(Args)
    test_dataloader(args)
