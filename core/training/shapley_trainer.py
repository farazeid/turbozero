from functools import partial
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from core.training.loss_fns import shapley_loss_fn
from core.training.mask_utils import sample_shapley_masks


def get_agent_target(agent_output, shapley_type: str):
    """
    Extracts the target value from the agent output based on the Shapley type.

    Args:
        agent_output: Tuple (policy_logits, value, ownership, score)
        shapley_type: 'behaviour', 'outcome', 'prediction' (alias for outcome), 'score'

    Returns:
        Target tensor of shape (B, num_outputs)
    """
    policy_logits, value, ownership, score = agent_output

    if shapley_type == "behaviour":
        # Target is action probabilities
        return jax.nn.softmax(policy_logits)
    elif shapley_type in ["outcome", "prediction"]:
        # Target is value estimate (win probability/score expectation)
        return value
    elif shapley_type == "score":
        return score
    elif shapley_type == "ownership":
        # Ownership is spatial (B, 19, 19, 1)
        # We might need to flatten it if the Shapley model expects (B, num_outputs)
        # but ShapleyNetwork usually outputs spatial maps sum to (B, num_outputs).
        # IF num_outputs is 361, then ownership is fine.
        return jnp.reshape(ownership, (ownership.shape[0], -1))
    else:
        raise ValueError(f"Unknown shapley_type: {shapley_type}")


class ShapleyTrainState(TrainState):
    batch_stats: Any = None


class ShapleyTrainer:
    def __init__(self, shapley_type: str, optimizer: optax.GradientTransformation):
        self.shapley_type = shapley_type
        self.optimizer = optimizer

    def create_train_state(self, key, shapley_model, sample_input, sample_global=None):
        """Initializes the TrainState for the Shapley model."""
        variables = shapley_model.init(
            key, sample_input, global_input=sample_global, mask=None, train=False
        )
        params = variables["params"]

        if "batch_stats" in variables:
            return ShapleyTrainState.create(
                apply_fn=shapley_model.apply,
                params=params,
                tx=self.optimizer,
                batch_stats=variables["batch_stats"],
            )
        else:
            return TrainState.create(
                apply_fn=shapley_model.apply, params=params, tx=self.optimizer
            )

    @partial(jax.jit, static_argnums=(0, 2, 6))
    def train_step(
        self,
        train_state: TrainState,
        agent_apply_fn: Any,  # static
        agent_variables: Any,  # FrozenDict: {'params': ..., 'batch_stats': ...}
        batch: Dict[str, Any],
        key: jax.random.PRNGKey,
        use_importance_sampling: bool = True,  # static
    ) -> Tuple[TrainState, Dict[str, Any]]:
        """
        Performs a single training step.

        Args:
            train_state: Current Shapley model TrainState
            agent_apply_fn: Apply function for the frozen agent (static)
            agent_variables: Variables for the frozen agent (params + batch_stats)
            batch: Data batch (must contain 'binaryInputNCHW', 'globalInputNC', optionally
                   'action_taken' and 'behaviour_logprob' for importance sampling)
            key: PRNG key for mask sampling
            use_importance_sampling: If True, compute and apply importance weights

        Returns:
            Updated TrainState and metrics.
        """
        x = batch["binaryInputNCHW"]
        g = batch.get("globalInputNC")
        B, H, W, C = x.shape

        # 1. Sample Masks
        # Uses the specific FastSHAP distribution
        mask = sample_shapley_masks(key, B, H, W)  # (B, H, W, 1)

        # 2. Compute Targets (Frozen Agent)
        # We need v(S) = Agent(x * mask, mask=mask) and v(empty) = Agent(x * 0, mask=zeros)

        # Prepare batch for agent (Masked, Null, Full optional)
        # We always need Masked and Null. If IS is on, we need Full too.
        # Construct batch: [Masked, Null, (Full)]

        mask_null = jnp.zeros_like(mask)
        x_masked = x * mask
        x_null = x * 0.0

        inputs_list = [x_masked, x_null]
        masks_list = [mask, mask_null]

        need_full = (
            use_importance_sampling
            and "action_taken" in batch
            and "behaviour_logprob" in batch
        )

        if need_full:
            inputs_list.append(x)
            masks_list.append(jnp.ones_like(mask))

        # Concatenate
        x_combined = jnp.concatenate(inputs_list, axis=0)
        mask_combined = jnp.concatenate(masks_list, axis=0)
        g_combined = jnp.concatenate([g] * len(inputs_list), axis=0)

        # Single Agent Pass
        out_combined = agent_apply_fn(
            agent_variables,
            x_combined,
            global_input=g_combined,
            mask=mask_combined,
            train=False,
        )

        # Split outputs
        # out_combined is tuple (policy, value, ownership, score)
        # Each element has shape (K*B, ...)
        def split_head(head_out):
            return jnp.split(head_out, len(inputs_list), axis=0)

        split_outputs = [split_head(h) for h in out_combined]
        # Transpose to get [(p,v,o,s)_masked, (p,v,o,s)_null, ...]
        outputs_list = list(zip(*split_outputs))

        out_masked = outputs_list[0]
        target_char_vals = get_agent_target(out_masked, self.shapley_type)

        out_null = outputs_list[1]
        null_char_vals = get_agent_target(out_null, self.shapley_type)

        # Stop gradient on targets
        target_char_vals = jax.lax.stop_gradient(target_char_vals)
        null_char_vals = jax.lax.stop_gradient(null_char_vals)

        # 3. Compute Importance Weights (if enabled and data available)
        importance_weights = None
        if need_full:
            # Get current policy from agent on full observation
            out_full = outputs_list[2]
            policy_logits = out_full[0]  # (B, num_actions)

            # Compute importance weights
            from core.training.importance_sampling import (
                compute_importance_weights_from_logits,
            )

            importance_weights = compute_importance_weights_from_logits(
                current_logits=policy_logits,
                action_taken=batch["action_taken"],
                behaviour_logprob=batch["behaviour_logprob"],
                normalize=True,  # Self-normalised IS per FastSVERL
            )

        # 4. Compute Loss & Update
        # Prepare batch for loss function
        loss_batch = {
            "observation": x,
            "global_input": g,
            "coalition_mask": mask,
            "target_char_vals": target_char_vals,
            "null_char_vals": null_char_vals,
        }

        grad_fn = jax.value_and_grad(shapley_loss_fn, has_aux=True)
        (loss, (metrics, _)), grads = grad_fn(
            train_state.params, train_state, loss_batch, importance_weights
        )

        # Compute gradient norm for stability monitoring
        grad_norm = optax.global_norm(grads)
        metrics["grad_norm"] = grad_norm
        metrics["mask_coverage"] = jnp.mean(mask)

        train_state = train_state.apply_gradients(grads=grads)

        return train_state, metrics
