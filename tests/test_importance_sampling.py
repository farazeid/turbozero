"""Tests for importance sampling module."""

import os
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import tyro

import wandb
from core.training.importance_sampling import (
    compute_importance_weights,
    compute_importance_weights_from_logits,
)


@dataclass
class Args:
    batch_size: int = 8
    num_actions: int = 362
    wandb_project: str = "fastsverl-tests"
    wandb_entity: str = "fastsverl"
    wandb_name: str = Path(__file__).stem


def test_importance_sampling(args: Args):
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY not found in environment.")

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        mode="online",
    )

    key = jax.random.PRNGKey(42)
    B = args.batch_size
    A = args.num_actions

    # Test 1: Weights sum to 1 when normalised
    print("\n=== Test 1: Normalised weights sum to 1 ===")
    key, k1, k2 = jax.random.split(key, 3)
    current_policy = jax.nn.softmax(jax.random.normal(k1, (B, A)), axis=-1)
    action_taken = jax.random.randint(k2, (B,), 0, A)
    behaviour_logprob = jnp.log(current_policy[jnp.arange(B), action_taken])

    weights = compute_importance_weights(
        current_policy, action_taken, behaviour_logprob, normalize=True
    )

    weights_sum = jnp.sum(weights)
    print(f"  Weights sum: {weights_sum:.6f}")
    assert jnp.abs(weights_sum - 1.0) < 1e-5, (
        f"Weights should sum to 1, got {weights_sum}"
    )
    print("  ✓ Normalised weights sum to 1")

    # Test 2: Same policy gives uniform weights
    print("\n=== Test 2: Same policy gives uniform weights ===")
    # When current policy == behaviour policy, all ratios are 1, so uniform
    uniform_weights = compute_importance_weights(
        current_policy, action_taken, behaviour_logprob, normalize=True
    )
    expected_uniform = jnp.ones(B) / B
    weight_diff = jnp.max(jnp.abs(uniform_weights - expected_uniform))
    print(f"  Max weight diff from uniform: {weight_diff:.6f}")
    assert weight_diff < 1e-5, f"Weights should be uniform, max diff: {weight_diff}"
    print("  ✓ Same policy gives uniform weights")

    # Test 3: From logits gives same result
    print("\n=== Test 3: compute_importance_weights_from_logits gives same result ===")
    key, k3 = jax.random.split(key)
    logits = jax.random.normal(k3, (B, A))
    probs = jax.nn.softmax(logits, axis=-1)
    behaviour_lp = jnp.log(probs[jnp.arange(B), action_taken] + 1e-8)

    weights_from_probs = compute_importance_weights(
        probs, action_taken, behaviour_lp, normalize=True
    )
    weights_from_logits = compute_importance_weights_from_logits(
        logits, action_taken, behaviour_lp, normalize=True
    )

    diff = jnp.max(jnp.abs(weights_from_probs - weights_from_logits))
    print(f"  Max diff between methods: {diff:.6f}")
    assert diff < 1e-4, f"Methods should give same result, diff: {diff}"
    print("  ✓ Both methods give same result")

    # Test 4: Different policies give non-uniform weights
    print("\n=== Test 4: Different policies give non-uniform weights ===")
    key, k4 = jax.random.split(key)
    different_policy = jax.nn.softmax(jax.random.normal(k4, (B, A)), axis=-1)
    different_logprob = jnp.log(different_policy[jnp.arange(B), action_taken] + 1e-8)

    non_uniform_weights = compute_importance_weights(
        current_policy, action_taken, different_logprob, normalize=True
    )

    variance = jnp.var(non_uniform_weights)
    print(f"  Weight variance: {variance:.6f}")
    print(f"  Min weight: {jnp.min(non_uniform_weights):.6f}")
    print(f"  Max weight: {jnp.max(non_uniform_weights):.6f}")
    # Should still sum to 1
    assert jnp.abs(jnp.sum(non_uniform_weights) - 1.0) < 1e-5
    print("  ✓ Different policies produce non-uniform weights")

    # Log results
    wandb.log(
        {
            "test1_weights_sum": float(weights_sum),
            "test2_weight_diff": float(weight_diff),
            "test3_method_diff": float(diff),
            "test4_weight_variance": float(variance),
            "success": True,
        }
    )

    wandb.finish()
    print("\n" + "=" * 40)
    print("All importance sampling tests passed!")


if __name__ == "__main__":
    args = tyro.cli(Args)
    test_importance_sampling(args)
