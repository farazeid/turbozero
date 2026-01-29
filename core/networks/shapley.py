"""
Shapley Network for FastSVERL.
Uses the same architecture as KataGo but with configurable depth.
"""

from dataclasses import dataclass
from typing import Optional

import flax.linen as nn
import jax.numpy as jnp
from einops import reduce

from core.networks.katago import (
    KataGoConfig,
    NestedBottleneckBlock,
    NormActConv,
)


@dataclass
class ShapleyConfig(KataGoConfig):
    """
    Configuration for the Shapley model.
    Inherits from KataGoConfig and adds a ratio to control the depth.
    """

    blocks_ratio: float = 0.5  # Use fewer blocks than agent
    multi_action: bool = False


class ShapleyHead(nn.Module):
    """
    Head for the Shapley model.
    Outputs a value for each feature (spatial) and each target.
    """

    num_outputs: int  # e.g., 1 for value/outcome, 362 for behaviour
    num_channels: int = 256

    @nn.compact
    def __call__(self, x, mask, mask_sum_hw, train: bool = True):
        # x is (B, H, W, C)
        # We want to output (B, H, W, num_outputs)

        # Reduced dimension spatial features
        h = nn.Conv(
            features=self.num_channels // 2,
            kernel_size=(1, 1),
            padding="SAME",
            use_bias=False,
        )(x)
        h = nn.BatchNorm(use_running_average=not train)(h)
        h = nn.relu(h)

        # Spatial Shapley values
        phi = nn.Conv(
            features=self.num_outputs, kernel_size=(1, 1), padding="SAME", use_bias=True
        )(h)

        return phi


class ShapleyNetwork(nn.Module):
    """
    Shapley model architecture based on KataGo.
    Uses NestedBottleneckBlock for the trunk.
    """

    config: ShapleyConfig
    num_outputs: int

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        global_input: Optional[jnp.ndarray] = None,
        mask: Optional[jnp.ndarray] = None,
        train: bool = True,
        grand_val: Optional[jnp.ndarray] = None,
        null_val: Optional[jnp.ndarray] = None,
    ):
        # x: (B, H, W, C)
        # global_input: (B, Cg)
        # mask: (B, H, W, 1) - If None, defaults to all ones.

        if mask is None:
            mask = jnp.ones((x.shape[0], x.shape[1], x.shape[2], 1))

        mask_sum_hw = reduce(mask, "b h w 1 -> b 1 1 1", "sum")
        mask_sum = jnp.sum(mask)

        # Initial Conv - Mask input so unknown features are invisible
        out = nn.Conv(
            features=self.config.num_channels,
            kernel_size=(3, 3),
            padding="SAME",
            use_bias=False,
        )(x * mask)

        # Global input processing (matched to KataGo)
        if global_input is not None:
            global_out = nn.Dense(
                self.config.num_channels, use_bias=False, name="linear_global"
            )(global_input)
            out = out + global_out[:, None, None, :]

        # Residual Trunk (fewer blocks than agent)
        num_blocks = max(1, int(self.config.num_blocks * self.config.blocks_ratio))
        for i in range(num_blocks):
            # GPool for blocks: gpool_start_idx, gpool_start_idx + interval, ...
            has_gpool = (
                i >= self.config.gpool_start_idx
                and (i - self.config.gpool_start_idx) % self.config.gpool_interval == 0
            )
            cg = self.config.c_gpool if has_gpool else None

            out = NestedBottleneckBlock(
                c_trunk=self.config.num_channels,
                c_mid=self.config.num_mid_channels,
                c_gpool=cg,
                internal_length=self.config.internal_length,
                name=f"blocks.{i}",
            )(out, mask, mask_sum_hw, mask_sum, train=train)

        # Final norm
        out = nn.BatchNorm(use_running_average=not train, name="norm_final")(out)
        out = nn.relu(out)

        # Head
        phi = ShapleyHead(
            num_outputs=self.num_outputs, num_channels=self.config.num_mid_channels
        )(out, mask, mask_sum_hw, train=train)

        # Explicitly mask the output: unknown features get zero attribution
        phi = phi * mask

        # Evaluation mode: apply efficiency constraint if grand_val/null_val provided
        if not train and grand_val is not None:
            # Efficiency constraint: sum(phi) = grand_val - null_val
            if null_val is None:
                null_val = jnp.zeros_like(grand_val)

            # Sum over spatial dimensions (features)
            phi_sum = reduce(phi, "b h w c -> b c", "sum")

            # Target sum
            target_sum = grand_val - null_val
            if target_sum.ndim == 1:
                target_sum = target_sum[:, None]

            # Correction factor
            correction = (target_sum - phi_sum) / jnp.maximum(
                reduce(mask, "b h w 1 -> b 1", "sum"), 1e-5
            )

            # Apply correction uniformly to masked positions
            phi = phi + correction[:, None, None, :] * mask

        return phi


# Convenience wrappers for specific Shapley types
class BehaviourShapley(nn.Module):
    """Shapley model for explaining agent behaviour (policy)."""

    config: ShapleyConfig
    num_actions: int = 362  # 19*19 + 1

    @nn.compact
    def __call__(
        self, x, global_input=None, mask=None, train=True, grand_val=None, null_val=None
    ):
        num_outputs = self.num_actions if self.config.multi_action else 1
        return ShapleyNetwork(config=self.config, num_outputs=num_outputs)(
            x, global_input, mask, train, grand_val, null_val
        )


class OutcomeShapley(nn.Module):
    """Shapley model for explaining game outcome."""

    config: ShapleyConfig

    @nn.compact
    def __call__(
        self, x, global_input=None, mask=None, train=True, grand_val=None, null_val=None
    ):
        return ShapleyNetwork(config=self.config, num_outputs=1)(
            x, global_input, mask, train, grand_val, null_val
        )


class PredictionShapley(nn.Module):
    """Shapley model for explaining agent's value prediction."""

    config: ShapleyConfig

    @nn.compact
    def __call__(
        self, x, global_input=None, mask=None, train=True, grand_val=None, null_val=None
    ):
        return ShapleyNetwork(config=self.config, num_outputs=1)(
            x, global_input, mask, train, grand_val, null_val
        )
