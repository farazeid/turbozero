"""
KataGo Neural Network Architecture for JAX/Flax.
This implementation exactly matches the b28c512nbt model structure.
"""

from dataclasses import dataclass
from typing import Optional

import flax.linen as nn
import jax.numpy as jnp
from einops import rearrange, reduce


@dataclass
class KataGoConfig:
    """Configuration for KataGo b28 model."""

    num_blocks: int = 28
    num_channels: int = 512  # c_trunk
    num_mid_channels: int = 256  # c_mid
    c_gpool: int = 64
    internal_length: int = 2  # blocks in each nested block's blockstack
    # GPool block indices: 2, 5, 8, 11, 14, 17, 20, 23, 26 (start at 2, then every 3)
    gpool_start_idx: int = 2
    gpool_interval: int = 3
    # Input features
    num_bin_input: int = 22
    num_global_input: int = 19


class KataGPool(nn.Module):
    """
    KataGo Global Pooling layer.
    Returns (mean, scaled_mean, max) concatenated.
    Output shape: (B, 1, 1, 3*C)
    """

    @nn.compact
    def __call__(self, x, mask, mask_sum_hw):
        # x: (N, H, W, C)
        # mask: (N, H, W, 1)
        # mask_sum_hw: (N, 1, 1, 1)

        mask_sum_hw_sqrt_offset = jnp.sqrt(mask_sum_hw) - 14.0
        denom = jnp.maximum(mask_sum_hw, 1e-5)

        layer_mean = reduce(x * mask, "b h w c -> b 1 1 c", "sum") / denom
        layer_max = reduce(x + (mask - 1.0) * 1e5, "b h w c -> b 1 1 c", "max")

        out_pool1 = layer_mean
        out_pool2 = layer_mean * (mask_sum_hw_sqrt_offset / 10.0)
        out_pool3 = layer_max

        out = jnp.concatenate([out_pool1, out_pool2, out_pool3], axis=-1)
        return out


class NormMask(nn.Module):
    """BatchNorm with masking support."""

    c_in: int

    @nn.compact
    def __call__(self, x, mask, mask_sum, train: bool = True):
        # Use standard BatchNorm
        x = nn.BatchNorm(use_running_average=not train, momentum=0.99, epsilon=1e-5)(x)
        return x


class KataConvAndGPool(nn.Module):
    """
    KataGo Conv + Global Pooling module.
    GPool path is ADDED to the regular conv path.
    """

    c_out: int
    c_gpool: int

    @nn.compact
    def __call__(self, x, mask, mask_sum_hw, mask_sum, train: bool = True):
        # Regular path: conv1r
        outr = nn.Conv(
            self.c_out, (3, 3), padding="SAME", use_bias=False, name="conv1r"
        )(x)

        # Global path: conv1g -> norm -> act -> gpool -> linear
        outg = nn.Conv(
            self.c_gpool, (3, 3), padding="SAME", use_bias=False, name="conv1g"
        )(x)
        outg = nn.BatchNorm(
            use_running_average=not train, momentum=0.99, epsilon=1e-5, name="normg"
        )(outg)
        outg = nn.relu(outg)

        # Global pooling: (B, H, W, c_gpool) -> (B, 1, 1, 3*c_gpool)
        outg = KataGPool()(outg, mask, mask_sum_hw)

        # Linear: 3*c_gpool -> c_out
        outg = nn.Dense(self.c_out, use_bias=False, name="linear_g")(outg)

        # Add (with broadcasting from (B,1,1,c_out) to (B,H,W,c_out))
        out = outr + outg
        return out


class NormActConv(nn.Module):
    """
    Pre-activation Norm -> Act -> Conv module.
    Optionally includes global pooling path.
    """

    c_out: int
    kernel_size: int = 3
    c_gpool: Optional[int] = None

    @nn.compact
    def __call__(self, x, mask, mask_sum_hw, mask_sum, train: bool = True):
        # 1. Norm
        out = nn.BatchNorm(
            use_running_average=not train, momentum=0.99, epsilon=1e-5, name="norm"
        )(x)

        # 2. Activation
        out = nn.relu(out)

        # 3. Conv (with or without GPool)
        if self.c_gpool is not None:
            out = KataConvAndGPool(
                c_out=self.c_out, c_gpool=self.c_gpool, name="convpool"
            )(out, mask, mask_sum_hw, mask_sum, train=train)
        else:
            out = nn.Conv(
                self.c_out,
                (self.kernel_size, self.kernel_size),
                padding="SAME",
                use_bias=False,
                name="conv",
            )(out)

        return out


class ResBlock(nn.Module):
    """
    Standard residual block with 2 NormActConv layers.
    First layer can optionally have global pooling.

    When gpool is present:
    - normactconv1: c_in=c_main, c_out=c_mid-c_gpool, c_gpool=c_gpool
    - normactconv2: c_in=c_mid-c_gpool, c_out=c_main
    """

    c_main: int  # Input/output channels (c_mid of parent nested block)
    c_mid: int  # Internal width (usually same as c_main for non-nested)
    c_gpool: Optional[int] = None

    @nn.compact
    def __call__(self, x, mask, mask_sum_hw, mask_sum, train: bool = True):
        # Calculate internal width
        c_internal = self.c_mid - (self.c_gpool if self.c_gpool else 0)

        # normactconv1 (potentially with gpool)
        # Output: c_internal (even though gpool adds back to c_internal, the total is c_internal)
        out = NormActConv(
            c_internal, kernel_size=3, c_gpool=self.c_gpool, name="normactconv1"
        )(x, mask, mask_sum_hw, mask_sum, train=train)

        # normactconv2 (always standard conv, back to c_main)
        out = NormActConv(
            self.c_main, kernel_size=3, c_gpool=None, name="normactconv2"
        )(out, mask, mask_sum_hw, mask_sum, train=train)

        return x + out


class NestedBottleneckBlock(nn.Module):
    """
    Nested bottleneck residual block.
    Structure:
    - normactconvp: 1x1 projection down (c_trunk -> c_mid)
    - blockstack: list of ResBlocks at c_mid
    - normactconvq: 1x1 projection up (c_mid -> c_trunk)
    """

    c_trunk: int
    c_mid: int
    c_gpool: Optional[int] = None  # If set, first ResBlock in blockstack has gpool
    internal_length: int = 2

    @nn.compact
    def __call__(self, x, mask, mask_sum_hw, mask_sum, train: bool = True):
        # Project down: c_trunk -> c_mid
        out = NormActConv(self.c_mid, kernel_size=1, c_gpool=None, name="normactconvp")(
            x, mask, mask_sum_hw, mask_sum, train=train
        )

        # Block stack
        for i in range(self.internal_length):
            # First block in stack gets gpool (if specified for this outer block)
            cg = self.c_gpool if i == 0 else None
            out = ResBlock(
                c_main=self.c_mid, c_mid=self.c_mid, c_gpool=cg, name=f"blockstack.{i}"
            )(out, mask, mask_sum_hw, mask_sum, train=train)

        # Project up: c_mid -> c_trunk
        out = NormActConv(
            self.c_trunk, kernel_size=1, c_gpool=None, name="normactconvq"
        )(out, mask, mask_sum_hw, mask_sum, train=train)

        return x + out


class PolicyHead(nn.Module):
    """KataGo policy head."""

    config: KataGoConfig

    @nn.compact
    def __call__(self, x, mask, mask_sum_hw, train: bool = True):
        c_trunk = self.config.num_channels
        c_policy = 64

        # conv1p: spatial features
        p = nn.Conv(c_policy, (1, 1), padding="SAME", use_bias=False, name="conv1p")(x)

        # conv1g: global features for gpool
        g = nn.Conv(c_policy, (1, 1), padding="SAME", use_bias=False, name="conv1g")(x)

        # biasg (acts as bias/scale, using simplified implementation)
        g_bias = self.param("biasg.bias", nn.initializers.zeros, (c_policy,))
        g = g + g_bias.reshape(1, 1, 1, -1)
        g = nn.relu(g)

        # Global pooling
        g_pool = KataGPool()(g, mask, mask_sum_hw)  # (B, 1, 1, 3*c_policy)

        # Linear: 3*c_policy -> c_policy
        g_out = nn.Dense(c_policy, use_bias=False, name="linear_g")(g_pool)

        # Add global features to spatial
        p = p + g_out

        # bias2
        p_bias = self.param("bias2.bias", nn.initializers.zeros, (c_policy,))
        p = p + p_bias.reshape(1, 1, 1, -1)
        p = nn.relu(p)

        # conv2p: 2 channels (move + optimistic move?)
        spatial = nn.Conv(2, (1, 1), padding="SAME", use_bias=False, name="conv2p")(p)
        # Use first channel as main policy
        spatial_logits = rearrange(spatial[..., 0], "b h w -> b (h w)")

        # Pass logit
        g_squeeze = g_pool.squeeze(1).squeeze(1)  # (B, 3*c_policy)
        pass_hidden = nn.Dense(c_policy, name="linear_pass")(g_squeeze)
        pass_hidden = pass_hidden + self.param(
            "linear_pass_bias.bias", nn.initializers.zeros, (c_policy,)
        )
        pass_hidden = nn.relu(pass_hidden)
        pass_logit = nn.Dense(2, name="linear_pass2")(pass_hidden)  # 2 outputs
        pass_logit = pass_logit[:, 0:1]  # Use first as pass logit

        logits = jnp.concatenate([spatial_logits, pass_logit], axis=-1)
        return logits


class ValueHead(nn.Module):
    """KataGo value head."""

    config: KataGoConfig

    @nn.compact
    def __call__(self, x, mask, mask_sum_hw, train: bool = True):
        c_trunk = self.config.num_channels
        c_val = 128
        c_val2 = 144

        # conv1: spatial
        v = nn.Conv(c_val, (1, 1), padding="SAME", use_bias=False, name="conv1")(x)

        # bias1
        v_bias = self.param("bias1.bias", nn.initializers.zeros, (c_val,))
        v = v + v_bias.reshape(1, 1, 1, -1)
        v = nn.relu(v)

        # Global pooling
        v_pool = KataGPool()(v, mask, mask_sum_hw)  # (B, 1, 1, 3*c_val)
        v_flat = v_pool.squeeze(1).squeeze(1)  # (B, 3*c_val = 384)

        # linear2: 384 -> c_val2
        v2 = nn.Dense(c_val2, name="linear2")(v_flat)
        v2_bias = self.param("bias2.bias", nn.initializers.zeros, (c_val2,))
        v2 = v2 + v2_bias
        v2 = nn.relu(v2)

        # Value outputs
        value = nn.Dense(3, name="linear_valuehead")(v2)  # win/loss/draw
        value_bias = self.param("bias_valuehead.bias", nn.initializers.zeros, (3,))
        value = value + value_bias

        # Misc value (score, etc)
        misc = nn.Dense(6, name="linear_miscvaluehead")(v2)
        misc_bias = self.param("bias_miscvaluehead.bias", nn.initializers.zeros, (6,))
        misc = misc + misc_bias

        # Ownership
        ownership = nn.Conv(
            1, (1, 1), padding="SAME", use_bias=True, name="conv_ownership"
        )(v)
        ownership = jnp.tanh(ownership)

        return value, ownership, misc


class KataGoNetwork(nn.Module):
    """
    Full KataGo network matching b28c512nbt architecture.
    """

    config: KataGoConfig

    @nn.compact
    def __call__(self, x, mask=None, train: bool = True):
        # x: (B, H, W, C) - binary input features

        # Create mask if not provided
        if mask is None:
            mask = jnp.ones((x.shape[0], x.shape[1], x.shape[2], 1))

        mask_sum_hw = reduce(mask, "b h w 1 -> b 1 1 1", "sum")
        mask_sum = jnp.sum(mask)

        # Initial conv: conv_spatial (22 -> 512)
        trunk = nn.Conv(
            self.config.num_channels,
            (3, 3),
            padding="SAME",
            use_bias=False,
            name="conv_spatial",
        )(x)

        # Global input processing (linear_global)
        # For now, we skip global inputs - can be added later
        # global_out = nn.Dense(self.config.num_channels, use_bias=False, name="linear_global")(global_inputs)
        # trunk = trunk + global_out[:, None, None, :]

        # Trunk blocks
        for i in range(self.config.num_blocks):
            # GPool for blocks: gpool_start_idx, gpool_start_idx + interval, ...
            has_gpool = (
                i >= self.config.gpool_start_idx
                and (i - self.config.gpool_start_idx) % self.config.gpool_interval == 0
            )
            cg = self.config.c_gpool if has_gpool else None

            trunk = NestedBottleneckBlock(
                c_trunk=self.config.num_channels,
                c_mid=self.config.num_mid_channels,
                c_gpool=cg,
                internal_length=self.config.internal_length,
                name=f"blocks.{i}",
            )(trunk, mask, mask_sum_hw, mask_sum, train=train)

        # Trunk final norm
        trunk = nn.BatchNorm(
            use_running_average=not train,
            momentum=0.99,
            epsilon=1e-5,
            name="norm_trunkfinal",
        )(trunk)
        trunk = nn.relu(trunk)

        # Heads
        policy = PolicyHead(config=self.config, name="policy_head")(
            trunk, mask, mask_sum_hw, train=train
        )
        value, ownership, misc = ValueHead(config=self.config, name="value_head")(
            trunk, mask, mask_sum_hw, train=train
        )

        # Return in expected format
        # value[:, 0] is win probability (before softmax)
        return policy, value, ownership, misc
