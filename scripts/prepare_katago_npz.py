import os

import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange


def unpack_binary_input(packed_input, pos_len=19):
    """
    Unpacks binaryInputNCHWPacked into (N, H, W, C).
    packed_input: (N, C, packed_bytes) where packed_bytes is 46 for 19x19.
    """
    bits = jnp.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=jnp.uint8)
    # Unpack bits: (N, C, packed_bytes, 8)
    unpacked = (rearrange(packed_input, "b c p -> b c p 1") & bits) > 0
    unpacked = unpacked.astype(jnp.float32)

    # Flatten bits and reshape to (N, H, W, C)
    unpacked = rearrange(unpacked, "b c p b8 -> b c (p b8)")
    unpacked = unpacked[:, :, : pos_len * pos_len]
    unpacked = rearrange(unpacked, "b c (h w) -> b h w c", h=pos_len, w=pos_len)

    return unpacked


def process_katago_batch(batch, include_meta=False, include_qvalues=False):
    """
    Processes a raw batch from np.load into JAX arrays.
    """
    jax_batch = {
        "binaryInputNCHW": unpack_binary_input(
            jnp.array(batch["binaryInputNCHWPacked"])
        ),
        "globalInputNC": jnp.array(batch["globalInputNC"]),
        "policyTargetsNCMove": jnp.array(batch["policyTargetsNCMove"]),
        "globalTargetsNC": jnp.array(batch["globalTargetsNC"]),
        "scoreDistrN": jnp.array(batch["scoreDistrN"]),
        "valueTargetsNCHW": rearrange(
            jnp.array(batch["valueTargetsNCHW"]), "b c h w -> b h w c"
        ),
    }

    if include_meta and "metadataInputNC" in batch:
        jax_batch["metadataInputNC"] = jnp.array(batch["metadataInputNC"])
    if include_qvalues and "qValueTargetsNCMove" in batch:
        jax_batch["qValueTargetsNCMove"] = jnp.array(batch["qValueTargetsNCMove"])

    return jax_batch


class KataGoDataLoader:
    def __init__(self, npz_files, batch_size, pos_len=19):
        self.npz_files = npz_files
        self.batch_size = batch_size
        self.pos_len = pos_len
        self.current_file_idx = 0
        self.current_npz_data = None
        self.current_row_idx = 0
        self.num_rows = 0

    def _load_next_file(self):
        if self.current_file_idx >= len(self.npz_files):
            return False

        file_path = self.npz_files[self.current_file_idx]
        print(f"Loading {file_path}...")
        self.current_npz_data = np.load(file_path)
        self.num_rows = self.current_npz_data["binaryInputNCHWPacked"].shape[0]
        self.current_row_idx = 0
        self.current_file_idx += 1
        return True

    def __iter__(self):
        self.current_file_idx = 0
        self.current_npz_data = None
        return self

    def __next__(self):
        if (
            self.current_npz_data is None
            or self.current_row_idx + self.batch_size > self.num_rows
        ):
            if not self._load_next_file():
                raise StopIteration

        start = self.current_row_idx
        end = start + self.batch_size
        self.current_row_idx = end

        batch = {
            key: self.current_npz_data[key][start:end]
            for key in [
                "binaryInputNCHWPacked",
                "globalInputNC",
                "policyTargetsNCMove",
                "globalTargetsNC",
                "scoreDistrN",
                "valueTargetsNCHW",
            ]
        }

        # Transfer and process in JAX
        return process_katago_batch(batch)


if __name__ == "__main__":
    # Small test if run directly
    import sys

    if len(sys.argv) > 1:
        loader = KataGoDataLoader([sys.argv[1]], batch_size=4)
        for batch in loader:
            print("Batch keys:", batch.keys())
            print("Binary input shape:", batch["binaryInputNCHW"].shape)
            break
