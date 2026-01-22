from typing import Dict, Tuple

import chex
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState


def sample_coalition_mask(
    key: jax.Array, batch_size: int, height: int, width: int
) -> jax.Array:
    """Samples a random coalition mask (H, W) for each item in the batch.

    Each position is included in the coalition with probability p,
    where p is sampled uniformly from [0, 1] for each batch item.
    """
    key_size, key_mask = jax.random.split(key)
    # Sample p uniformly for each item in batch
    p = jax.random.uniform(key_size, (batch_size, 1, 1))
    # Sample inclusion mask: 1 if feature is in coalition, 0 otherwise
    # We use float32 for the mask so it can be used in multiplication directly
    mask = jax.random.bernoulli(key_mask, p, (batch_size, height, width)).astype(
        jnp.float32
    )
    return mask


def compute_prediction_shapley_loss(
    params: chex.ArrayTree,
    train_state: TrainState,
    observation: chex.Array,
    key: jax.Array,
) -> Tuple[chex.Array, Dict[str, chex.Array]]:
    """Computes FastSVERL surrogate loss for the Value head (Prediction Shapley)."""
    batch_size, h, w, _ = observation.shape
    key_coal, _ = jax.random.split(key)
    coal_mask = sample_coalition_mask(key_coal, batch_size, h, w)

    variables = (
        {"params": params, "batch_stats": train_state.batch_stats}
        if hasattr(train_state, "batch_stats")
        else {"params": params}
    )

    # 1. Predicted Shapley values from the specific head
    out_full = train_state.apply_fn(
        variables, x=observation, train=False, mutable=False
    )
    pred_shapley = out_full["prediction_shapley"]  # (batch, H, W, 1)

    # 2. Target values v(C) and v(null) from the Value head
    masked_obs = observation * coal_mask[..., None]
    out_C = train_state.apply_fn(variables, x=masked_obs, train=False, mutable=False)
    value_C = jax.lax.stop_gradient(out_C["value"])

    null_obs = jnp.zeros_like(observation)
    out_null = train_state.apply_fn(variables, x=null_obs, train=False, mutable=False)
    value_null = jax.lax.stop_gradient(out_null["value"])

    # 3. Surrogate loss: (sum_{i in C} g(o)_i - (v(C) - v(null)))^2
    predicted_sum = (pred_shapley.squeeze(-1) * coal_mask).sum(axis=(1, 2))
    target = (value_C - value_null).squeeze(-1)
    loss = jnp.mean((predicted_sum - target) ** 2)

    return loss, {"prediction_shapley_loss": loss}


def compute_behaviour_characteristic_loss(
    params: chex.ArrayTree,
    train_state: TrainState,
    observation: chex.Array,
    key: jax.Array,
    behaviour_shapley_approx: bool = True,
) -> Tuple[chex.Array, Dict[str, chex.Array]]:
    """Computes MSE loss for the Behaviour Characteristic head (Eq. 28)."""
    batch_size, h, w, _ = observation.shape
    key_coal, _ = jax.random.split(key)
    coal_mask = sample_coalition_mask(key_coal, batch_size, h, w)

    variables = (
        {"params": params, "batch_stats": train_state.batch_stats}
        if hasattr(train_state, "batch_stats")
        else {"params": params}
    )

    # Target: Original spatial policy logits pi(s, a) for the full observation
    out_full = train_state.apply_fn(
        variables, x=observation, train=False, mutable=False
    )

    if behaviour_shapley_approx:
        target = jax.lax.stop_gradient(out_full["spatial_policy_logits"])
    else:
        target = jax.lax.stop_gradient(out_full["policy"])

    # Prediction: bhvr_char(s, a | C)
    masked_obs = observation * coal_mask[..., None]
    out_masked = train_state.apply_fn(
        variables, x=masked_obs, train=False, mutable=False
    )
    pred_char = out_masked["behaviour_characteristic"]

    if not behaviour_shapley_approx:
        # Strict mode: Target is global policy vector (B, A)
        # Prediction is spatial map (B, H, W, A) -> Sum to get global vector
        pred_char = pred_char.sum(axis=(1, 2))

    loss = jnp.mean((target - pred_char) ** 2)

    return loss, {"behaviour_char_loss": loss}


def compute_behaviour_shapley_loss(
    params: chex.ArrayTree,
    train_state: TrainState,
    observation: chex.Array,
    key: jax.Array,
    behaviour_shapley_approx: bool = True,
) -> Tuple[chex.Array, Dict[str, chex.Array]]:
    """Computes surrogate loss for the Behaviour Shapley head (Eq. 19/20)."""
    batch_size, h, w, _ = observation.shape
    key_coal, _ = jax.random.split(key)
    coal_mask = sample_coalition_mask(key_coal, batch_size, h, w)

    variables = (
        {"params": params, "batch_stats": train_state.batch_stats}
        if hasattr(train_state, "batch_stats")
        else {"params": params}
    )

    # 1. Predicted Shapley values
    out_full = train_state.apply_fn(
        variables, x=observation, train=False, mutable=False
    )
    pred_shapley = out_full["behaviour_shapley"]  # (batch, H, W, OutDim)

    # 2. Target from Behaviour Characteristic head
    masked_obs = observation * coal_mask[..., None]
    out_C = train_state.apply_fn(variables, x=masked_obs, train=False, mutable=False)
    char_C = jax.lax.stop_gradient(out_C["behaviour_characteristic"])

    null_obs = jnp.zeros_like(observation)
    out_null = train_state.apply_fn(variables, x=null_obs, train=False, mutable=False)
    char_null = jax.lax.stop_gradient(out_null["behaviour_characteristic"])

    # 3. Surrogate loss
    # (sum_{i in C} phi_i - (char(C) - char(null)))^2
    predicted_sum = (pred_shapley * coal_mask[..., None]).sum(axis=(1, 2))

    # Target is ALWAYS the difference in the Characteristic Function
    # Since Char Head is spatial (Conv), we sum it to get the global/pooled value
    # (In Approx mode, this explains global feature sum. In Strict, it explains global policy).
    target = (char_C - char_null).sum(axis=(1, 2))

    loss = jnp.mean((predicted_sum - target) ** 2)

    return loss, {"behaviour_shapley_loss": loss}
