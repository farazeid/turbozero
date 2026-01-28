from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange, reduce


@dataclass
class KataGoConfig:
    num_blocks: int = 20
    num_channels: int = 128
    num_mid_channels: int = 128
    num_policy_channels: int = 32
    num_value_channels: int = 32
    num_ownership_channels: int = 64
    num_score_channels: int = 64
    bnorm_epsilon: float = 1e-5
    bnorm_momentum: float = 0.99
    activation: str = "relu"


class KataGPool(nn.Module):
    """
    KataGo Global Pooling layer.
    Concatenates global mean, size-scaled mean, and global max.
    """

    @nn.compact
    def __call__(self, x, mask, mask_sum_hw):
        # x: (N, H, W, C)
        # mask: (N, H, W, 1)
        # mask_sum_hw: (N, 1, 1, 1)

        mask_sum_hw_sqrt_offset = jnp.sqrt(mask_sum_hw) - 14.0

        # mean: sum(x * mask) / mask_sum
        layer_mean = reduce(x * mask, "b h w c -> b 1 1 c", "sum") / mask_sum_hw

        # max: max(x + (mask - 1.0) * huge)
        layer_max = reduce(x + (mask - 1.0) * 1e5, "b h w c -> b 1 1 c", "max")

        out_pool1 = layer_mean
        out_pool2 = layer_mean * (mask_sum_hw_sqrt_offset / 10.0)
        out_pool3 = layer_max

        # Concatenate along channel axis
        out = jnp.concatenate([out_pool1, out_pool2, out_pool3], axis=-1)
        return out


class NormMask(nn.Module):
    """
    Simplified KataGo Normalization with masking.
    Currently implements standard Batch Norm with masking.
    """

    c_in: int
    epsilon: float = 1e-5
    momentum: float = 0.99

    @nn.compact
    def __call__(self, x, mask, mask_sum, train: bool = True):
        if train:
            # Compute mean and var over masked areas
            mean = reduce(x * mask, "b h w c -> 1 1 1 c", "sum") / mask_sum
            diff = (x - mean) * mask
            var = reduce(jnp.square(diff), "b h w c -> 1 1 1 c", "sum") / mask_sum
        else:
            mean = jnp.zeros((1, 1, 1, self.c_in))
            var = jnp.ones((1, 1, 1, self.c_in))

        x_norm = (x - mean) / jnp.sqrt(var + self.epsilon)

        gamma = self.param("gamma", nn.initializers.ones, (1, 1, 1, self.c_in))
        beta = self.param("beta", nn.initializers.zeros, (1, 1, 1, self.c_in))

        return x_norm * gamma + beta


class NormActConv(nn.Module):
    c_out: int
    kernel_size: int
    c_gpool: Optional[int] = None
    config: Optional[KataGoConfig] = None

    @nn.compact
    def __call__(self, x, mask, mask_sum_hw, mask_sum, train: bool = True):
        # 1. Norm
        x = NormMask(c_in=x.shape[-1])(x, mask, mask_sum, train=train)

        # 2. Act
        x = nn.relu(x)

        # 3. Conv / GPool mix
        if self.c_gpool is not None:
            h_spatial = nn.Conv(
                self.c_out,
                (self.kernel_size, self.kernel_size),
                padding="SAME",
                use_bias=False,
            )(x)

            g = KataGPool()(x, mask, mask_sum_hw)
            h_global = nn.Dense(self.c_out, use_bias=False)(g)

            x = h_spatial + h_global
        else:
            x = nn.Conv(
                self.c_out,
                (self.kernel_size, self.kernel_size),
                padding="SAME",
                use_bias=False,
            )(x)

        return x


class BottleneckResBlock(nn.Module):
    c_mid: int
    c_gpool: Optional[int] = None
    internal_length: int = 1
    config: Optional[KataGoConfig] = None

    @nn.compact
    def __call__(self, x, mask, mask_sum_hw, mask_sum, train: bool = True):
        c_main = x.shape[-1]

        out = x
        # 1x1 downsample
        out = NormActConv(c_out=self.c_mid, kernel_size=1)(
            out, mask, mask_sum_hw, mask_sum, train=train
        )

        # 3x3 mid stack
        for i in range(self.internal_length):
            cg = self.c_gpool if i == 0 else None
            out = NormActConv(c_out=self.c_mid, kernel_size=3, c_gpool=cg)(
                out, mask, mask_sum_hw, mask_sum, train=train
            )

        # 1x1 upsample
        out = NormActConv(c_out=c_main, kernel_size=1)(
            out, mask, mask_sum_hw, mask_sum, train=train
        )

        return x + out


class PolicyHead(nn.Module):
    num_actions: int  # e.g. 19*19+1
    config: Optional[KataGoConfig] = None

    @nn.compact
    def __call__(self, x, mask, mask_sum_hw, train: bool = True):
        # Reduced dimension spatial features
        h = nn.Conv(features=2, kernel_size=(1, 1), padding="SAME", use_bias=False)(x)
        h = nn.BatchNorm(use_running_average=not train)(h)
        h = nn.relu(h)

        # Spatial logits
        spatial_logits = nn.Conv(
            features=1, kernel_size=(1, 1), padding="SAME", use_bias=True
        )(h)
        spatial_logits = rearrange(spatial_logits, "b h w 1 -> b (h w)")

        # Pass logit pooled from trunk
        g = reduce(x, "b h w c -> b c", "mean")
        pass_logit = nn.Dense(features=1)(g)

        logits = jnp.concatenate([spatial_logits, pass_logit], axis=-1)
        return logits


class ValueHead(nn.Module):
    config: Optional[KataGoConfig] = None

    @nn.compact
    def __call__(self, x, mask, mask_sum_hw, train: bool = True):
        # Trunk pooled features
        g = reduce(x, "b h w c -> b c", "mean")

        # Value (win/loss/draw)
        v = nn.Dense(features=64)(g)
        v = nn.relu(v)
        value = nn.Dense(features=1)(v)

        # Ownership (spatial)
        ownership = nn.Conv(
            features=1, kernel_size=(1, 1), padding="SAME", use_bias=True
        )(x)
        ownership = jnp.tanh(ownership)

        # Score (expectation/distribution)
        score = nn.Dense(features=1)(v)

        return value, ownership, score


class KataGoNetwork(nn.Module):
    config: KataGoConfig

    @nn.compact
    def __call__(self, x, train: bool = True):
        # Precompute masks
        mask = jnp.ones((x.shape[0], x.shape[1], x.shape[2], 1))
        mask_sum_hw = reduce(mask, "b h w 1 -> b 1 1 1", "sum")
        mask_sum = jnp.sum(mask)

        # Initial Conv
        x = nn.Conv(
            features=self.config.num_channels,
            kernel_size=(3, 3),
            padding="SAME",
            use_bias=False,
        )(x)

        # Residual Trunk
        for i in range(self.config.num_blocks):
            cg = self.config.num_channels // 4 if i % 2 == 0 else None
            x = BottleneckResBlock(
                c_mid=self.config.num_mid_channels, c_gpool=cg, config=self.config
            )(x, mask, mask_sum_hw, mask_sum, train=train)

        # Heads
        policy_logits = PolicyHead(num_actions=19 * 19 + 1, config=self.config)(
            x, mask, mask_sum_hw, train=train
        )
        value, ownership, score = ValueHead(config=self.config)(
            x, mask, mask_sum_hw, train=train
        )

        return policy_logits, value, ownership, score
