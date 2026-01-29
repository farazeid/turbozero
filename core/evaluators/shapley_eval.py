import time

import jax.numpy as jnp
from flax.core import FrozenDict

from core.evaluators.visualizer import GoVisualizer, spearman_correlation


class ShapleyEvaluator:
    def __init__(self, pos_len=19):
        self.pos_len = pos_len
        self.visualizer = GoVisualizer(pos_len=pos_len)

    def evaluate_move37(
        self, train_state, agent_apply_fn, agent_variables, move37_batch, shapley_type
    ):
        """
        Evaluates the model on Move 37 and generates visualizations.
        """
        start_time = time.time()

        x = move37_batch["binaryInputNCHW"]  # (1, 19, 19, C)
        g = move37_batch.get("globalInputNC")  # (1, Cg)
        # For visualization, we need stone positions (channels 1 and 2 usually)
        stones = x[0, :, :, 1:3]  # (19, 19, 2)

        # 1. Prediction/Explanation for the full board (mask=None)
        # We need to handle both multi-action and single-action
        variables = FrozenDict({"params": train_state.params})
        if hasattr(train_state, "batch_stats") and train_state.batch_stats is not None:
            variables = variables.copy({"batch_stats": train_state.batch_stats})

        phi = train_state.apply_fn(
            variables, x=x, global_input=g, mask=None, train=False
        )  # (1, 19, 19, num_outputs)

        prefix = "AlphaGo v Lee Sedol, Game 2, Move 37 — "
        caption_map = {
            "behaviour": f"{prefix}Behaviour Shapley model's values onto Go 19x19 board demonstrates each position's influence on the current/next move made by the agent policy",
            "outcome": f"{prefix}Outcome Shapley model's values onto Go 19x19 board demonstrates each position's influence on the agent's true win-likelihood estimate",
            "prediction": f"{prefix}Prediction Shapley model's values onto Go 19x19 board demonstrates each position's influence on the agent's internal win-likelihood estimate",
        }
        caption = caption_map.get(
            shapley_type, f"{shapley_type.capitalize()} Shapley values"
        )

        # If behaviour, phi shape is (1, 19, 19, 362)
        # We want to visualize the attribution for the action taken in Move 37
        if shapley_type == "behaviour":
            action_taken = move37_batch["action_taken"][0]
            phi_to_plot = phi[0, :, :, action_taken]
        else:
            phi_to_plot = phi[0, :, :, 0]

        image = self.visualizer.plot_shapley(
            phi_to_plot, stones=stones, title=shapley_type.capitalize(), caption=caption
        )

        # 2. Axiom Validations
        # Nullity Axiom: Mask a random stone position and check if phi changes insignificantly
        nullity_score = self.test_nullity(train_state, x, g)

        # Symmetry Axiom: Rotated/flipped versions should have zero difference
        symmetry_score = self.test_symmetry(train_state, x, g)

        eval_duration = time.time() - start_time

        metrics = {
            f"eval/{shapley_type}_move37_nullity": nullity_score,
            f"eval/{shapley_type}_move37_symmetry": symmetry_score,
            f"eval/{shapley_type}_eval_time": eval_duration,
        }

        return image, metrics

    def test_nullity(self, train_state, x, g):
        # Placeholder for nullity test as described by user
        # "At a random masked position, place a stone and calculate difference between before and after"
        B, H, W, C = x.shape
        mask = jnp.ones((B, H, W, 1))
        # Mask out a random position (0, r, c)
        r, c = 10, 10  # Random-ish
        mask = mask.at[0, r, c, 0].set(0.0)

        variables = FrozenDict({"params": train_state.params})
        if hasattr(train_state, "batch_stats") and train_state.batch_stats is not None:
            variables = variables.copy({"batch_stats": train_state.batch_stats})

        phi_before = train_state.apply_fn(
            variables, x=x, global_input=g, mask=mask, train=False
        )

        # Change something at (r, c) in x
        x_after = x.at[0, r, c, 1].set(1.0 - x[0, r, c, 1])  # Flip stone presence
        phi_after = train_state.apply_fn(
            variables, x=x_after, global_input=g, mask=mask, train=False
        )

        diff = jnp.abs(phi_before - phi_after).mean()
        return float(diff)

    def test_symmetry(self, train_state, x, g):
        # Symmetry: Rotate input and check if output is rotated accordingly
        variables = FrozenDict({"params": train_state.params})
        if hasattr(train_state, "batch_stats") and train_state.batch_stats is not None:
            variables = variables.copy({"batch_stats": train_state.batch_stats})

        phi_orig = train_state.apply_fn(
            variables, x=x, global_input=g, mask=None, train=False
        )

        # Rotate x (rot90 on H, W)
        x_rot = jnp.rot90(x, k=1, axes=(1, 2))
        # Note: global inputs are NOT rotated as they are not spatial
        phi_rot_pred = train_state.apply_fn(
            variables, x=x_rot, global_input=g, mask=None, train=False
        )

        # Rotate original phi
        phi_orig_rot = jnp.rot90(phi_orig, k=1, axes=(1, 2))

        diff = jnp.abs(phi_orig_rot - phi_rot_pred).mean()
        return float(diff)

    def compute_correlations(self, train_state_b, train_state_p, x, g, action_taken):
        # Spearman correlation between behavior and prediction
        variables_b = FrozenDict({"params": train_state_b.params})
        if (
            hasattr(train_state_b, "batch_stats")
            and train_state_b.batch_stats is not None
        ):
            variables_b = variables_b.copy({"batch_stats": train_state_b.batch_stats})

        variables_p = FrozenDict({"params": train_state_p.params})
        if (
            hasattr(train_state_p, "batch_stats")
            and train_state_p.batch_stats is not None
        ):
            variables_p = variables_p.copy({"batch_stats": train_state_p.batch_stats})

        phi_b = train_state_b.apply_fn(
            variables_b, x=x, global_input=g, mask=None, train=False
        )
        phi_p = train_state_p.apply_fn(
            variables_p, x=x, global_input=g, mask=None, train=False
        )

        vector_b = phi_b[0, :, :, action_taken].flatten()
        vector_p = phi_p[0, :, :, 0].flatten()

        corr = spearman_correlation(vector_b, vector_p)
        return float(corr)
