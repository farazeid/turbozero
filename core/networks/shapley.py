from dataclasses import dataclass
from typing import Optional

import flax.linen as nn
import jax.numpy as jnp
from einops import reduce

from core.networks.katago import (
    BottleneckResBlock,
    KataGoConfig,
)


@dataclass
class ShapleyConfig(KataGoConfig):
    """
    Configuration for the Shapley model.
    Inherits from KataGoConfig and adds a ratio to control the depth.
    """

    blocks_ratio: float = 1.0
    multi_action: bool = False


class ShapleyHead(nn.Module):
    """
    Head for the Shapley model.
    Outputs a value for each feature (spatial) and each target.
    """

    num_outputs: int  # e.g., 1 for value/outcome, 362 for behaviour
    config: ShapleyConfig

    @nn.compact
    def __call__(self, x, mask, mask_sum_hw, train: bool = True):
        # x is (B, H, W, C)
        # We want to output (B, H, W, num_outputs)

        # Reduced dimension spatial features (similar to KataGo PolicyHead but more flexible)
        h = nn.Conv(
            features=self.config.num_channels // 2,
            kernel_size=(1, 1),
            padding="SAME",
            use_bias=False,
        )(x)
        h = nn.BatchNorm(use_running_average=not train)(h)
        h = nn.relu(h)

        # Spatial Shapley values
        # For each of the num_outputs, we want a spatial map.
        # So output is (B, H, W, num_outputs)
        phi = nn.Conv(
            features=self.num_outputs, kernel_size=(1, 1), padding="SAME", use_bias=True
        )(h)

        return phi


class ShapleyNetwork(nn.Module):
    """
    Shapley model architecture based on KataGo.
    """

    config: ShapleyConfig
    num_outputs: int

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        train: bool = True,
        grand_val: Optional[jnp.ndarray] = None,
        null_val: Optional[jnp.ndarray] = None,
    ):
        # x: (B, H, W, C)
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

        # Residual Trunk
        num_blocks = max(1, int(self.config.num_blocks * self.config.blocks_ratio))
        for i in range(num_blocks):
            # Global pooling every 2 blocks, same as KataGo
            cg = self.config.num_channels // 4 if i % 2 == 0 else None
            out = BottleneckResBlock(
                c_mid=self.config.num_mid_channels, c_gpool=cg, config=self.config
            )(out, mask, mask_sum_hw, mask_sum, train=train)

        # Head
        phi = ShapleyHead(num_outputs=self.num_outputs, config=self.config)(
            out, mask, mask_sum_hw, train=train
        )

        # Explicitly mask the output: unknown features get zero attribution
        phi = phi * mask

        # Evaluation mode: apply efficiency constraint if grand_val/null_val provided
        if not train and grand_val is not None:
            # Efficiency constraint: sum(phi) = grand_val - null_val
            # phi shape: (B, H, W, num_outputs)
            # grand_val shape: (B, num_outputs) or (B, 1) or (B,)
            # null_val defaults to 0 if not provided
            if null_val is None:
                null_val = jnp.zeros_like(grand_val)

            # Sum over spatial dimensions (features)
            phi_sum = reduce(phi, "b h w c -> b c", "sum")

            # Target total value
            target_sum = grand_val - null_val
            if len(target_sum.shape) == 1:
                target_sum = target_sum[:, None]

            # Difference to distribute only to unmasked (known) features
            # Avoid division by zero if mask is all zeros
            num_known_features = jnp.maximum(1.0, reduce(mask, "b h w 1 -> b 1", "sum"))
            diff = (target_sum - phi_sum) / num_known_features

            # Add diff only to unmasked spatial positions
            phi = phi + (diff[:, None, None, :] * mask)

        return phi


class BehaviourShapley(nn.Module):
    config: ShapleyConfig
    num_actions: int = 362

    @nn.compact
    def __call__(self, x, mask=None, train=True, grand_val=None, null_val=None):
        num_outputs = self.num_actions if self.config.multi_action else 1
        return ShapleyNetwork(config=self.config, num_outputs=num_outputs)(
            x, mask=mask, train=train, grand_val=grand_val, null_val=null_val
        )


class OutcomeShapley(nn.Module):
    config: ShapleyConfig

    @nn.compact
    def __call__(self, x, mask=None, train=True, grand_val=None, null_val=None):
        return ShapleyNetwork(config=self.config, num_outputs=1)(
            x, mask=mask, train=train, grand_val=grand_val, null_val=null_val
        )


class PredictionShapley(nn.Module):
    config: ShapleyConfig

    @nn.compact
    def __call__(self, x, mask=None, train=True, grand_val=None, null_val=None):
        return ShapleyNetwork(config=self.config, num_outputs=1)(
            x, mask=mask, train=train, grand_val=grand_val, null_val=null_val
        )
