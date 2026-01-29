"""Importance sampling utilities for off-policy Shapley training.

Implements normalised importance sampling as described in FastSVERL Appendix D:
- Ratio: ρ = π(a|s) / π_b(a|s) = exp(log π(a|s) - log π_b(a|s))
- Normalised: w_i = ρ_i / Σ_j ρ_j
- No clipping (unclipped IS achieves best accuracy per paper)
"""

import jax.numpy as jnp
from jax import Array


def compute_importance_weights(
    current_policy: Array,
    action_taken: Array,
    behaviour_logprob: Array,
    normalize: bool = True,
    epsilon: float = 1e-8,
) -> Array:
    """Computes (normalised) importance sampling weights.

    Args:
        current_policy: (B, num_actions) softmax probabilities from current agent
        action_taken: (B,) action indices that were taken in the data
        behaviour_logprob: (B,) log-probability of action under behaviour policy
        normalize: If True, normalise weights to sum to 1 within batch
        epsilon: Small constant to prevent division by zero

    Returns:
        weights: (B,) importance sampling weights
    """
    batch_size = current_policy.shape[0]

    # Get current policy's log-probability for the taken action
    # current_policy: (B, A), action_taken: (B,) -> (B,)
    action_indices = jnp.arange(batch_size)
    current_probs = current_policy[action_indices, action_taken]
    current_logprob = jnp.log(current_probs + epsilon)

    # Compute importance ratio: π(a|s) / π_b(a|s)
    # = exp(log π(a|s) - log π_b(a|s))
    log_ratio = current_logprob - behaviour_logprob
    ratio = jnp.exp(log_ratio)

    # Normalise weights within batch (self-normalised IS)
    if normalize:
        weights = ratio / (jnp.sum(ratio) + epsilon)
    else:
        weights = ratio

    return weights


def compute_importance_weights_from_logits(
    current_logits: Array,
    action_taken: Array,
    behaviour_logprob: Array,
    normalize: bool = True,
    epsilon: float = 1e-8,
) -> Array:
    """Computes importance weights from logits (avoids double softmax).

    Args:
        current_logits: (B, num_actions) raw logits from current agent
        action_taken: (B,) action indices that were taken in the data
        behaviour_logprob: (B,) log-probability of action under behaviour policy
        normalize: If True, normalise weights to sum to 1 within batch
        epsilon: Small constant for numerical stability

    Returns:
        weights: (B,) importance sampling weights
    """
    import jax

    # Convert logits to log-probabilities via log_softmax
    current_log_probs = jax.nn.log_softmax(current_logits, axis=-1)

    # Get log-prob for the taken action
    batch_size = current_logits.shape[0]
    action_indices = jnp.arange(batch_size)
    current_logprob = current_log_probs[action_indices, action_taken]

    # Compute importance ratio
    log_ratio = current_logprob - behaviour_logprob
    ratio = jnp.exp(log_ratio)

    # Normalise
    if normalize:
        weights = ratio / (jnp.sum(ratio) + epsilon)
    else:
        weights = ratio

    return weights
