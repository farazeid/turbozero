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
        if "binaryInputNCHWPacked" in self.current_npz_data:
            self.num_rows = self.current_npz_data["binaryInputNCHWPacked"].shape[0]
            self.packed = True
        else:
            self.num_rows = self.current_npz_data["binaryInputNCHW"].shape[0]
            self.packed = False
        self.current_row_idx = 0
        self.current_file_idx += 1
        return True

    def reset(self):
        """Resets the dataloader to the beginning of the file list."""
        self.current_file_idx = 0
        self.current_npz_data = None
        self.current_row_idx = 0
        self.num_rows = 0

    def __iter__(self):
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

        all_possible_keys = [
            "globalInputNC",
            "policyTargetsNCMove",
            "globalTargetsNC",
            "scoreDistrN",
            "valueTargetsNCHW",
            "binaryInputNCHWPacked",
            "binaryInputNCHW",
        ]
        keys = [k for k in all_possible_keys if k in self.current_npz_data]

        batch = {key: self.current_npz_data[key][start:end] for key in keys}

        # Transfer and process in JAX
        if self.packed:
            processed = process_katago_batch(batch)
        else:
            # Already unpacked, just rename and convert
            processed = {
                "binaryInputNCHW": jnp.array(batch["binaryInputNCHW"]),
            }
            if "globalInputNC" in batch:
                processed["globalInputNC"] = jnp.array(batch["globalInputNC"])
            if "policyTargetsNCMove" in batch:
                processed["policyTargetsNCMove"] = jnp.array(
                    batch["policyTargetsNCMove"]
                )
            if "globalTargetsNC" in batch:
                processed["globalTargetsNC"] = jnp.array(batch["globalTargetsNC"])
            if "scoreDistrN" in batch:
                processed["scoreDistrN"] = jnp.array(batch["scoreDistrN"])
            if "valueTargetsNCHW" in batch:
                processed["valueTargetsNCHW"] = rearrange(
                    jnp.array(batch["valueTargetsNCHW"]), "b c h w -> b h w c"
                )

        # Extract action and behaviour logprob for importance sampling
        # policyTargetsNCMove: (B, C, num_actions) where C typically = 2
        # C=0 is typically the policy target, C=1 is often unused
        policy_targets = batch["policyTargetsNCMove"][:, 0, :]  # (B, num_actions)

        # Find the action taken (argmax of policy target)
        action_taken = np.argmax(policy_targets, axis=-1)  # (B,)

        # Get behaviour log-probability (log of probability at that action)
        # Handle potential issues with zero probabilities
        policy_probs = policy_targets / (
            np.sum(policy_targets, axis=-1, keepdims=True) + 1e-8
        )
        behaviour_logprob = np.log(
            policy_probs[np.arange(len(action_taken)), action_taken] + 1e-8
        )

        processed["action_taken"] = jnp.asarray(action_taken)
        processed["behaviour_logprob"] = jnp.asarray(behaviour_logprob)

        return processed

    def get_state(self):
        """Returns the current state of the dataloader for checkpointing."""
        return {
            "current_file_idx": self.current_file_idx,
            "current_row_idx": self.current_row_idx,
        }

    def load_state(self, state):
        """Restores the dataloader state from a checkpoint."""
        # Note: We subtract 1 from current_file_idx because _load_next_file increments it
        # but the current_npz_data belongs to the file at current_file_idx - 1.
        # Actually, it's simpler to just store the actual values and reload them.
        self.current_file_idx = state.get("current_file_idx", 0)
        self.current_row_idx = state.get("current_row_idx", 0)

        # Force reload of the correct file on next __next__ call
        self.current_npz_data = None
        if self.current_file_idx > 0:
            # Re-load the previous file so that current_row_idx is valid within it
            self.current_file_idx -= 1
            self._load_next_file()
            self.current_row_idx = state.get("current_row_idx", 0)
