from typing import Callable, List

import chex
import jax
import numpy as np
import PIL.Image
import PIL.ImageDraw
from flax.training.train_state import TrainState

import wandb
from core.evaluators.evaluator import Evaluator


class ShapleyVisualizer:
    def __init__(
        self,
        env_step_fn: Callable,
        env_init_fn: Callable,
        state_to_nn_input_fn: Callable,
        evaluator: Evaluator,
    ):
        self.env_step_fn = env_step_fn
        self.env_init_fn = env_init_fn
        self.state_to_nn_input_fn = state_to_nn_input_fn
        self.evaluator = evaluator

    def generate_heatmap(
        self,
        shapley_values: np.ndarray,
        observation: np.ndarray,
        cur_player_id: int,
        board_size: int,
        img_size: int = 512,
    ) -> np.ndarray:
        """Generates a high-res board visualization with Shapley heatmap overlay."""
        # 1. Setup Image
        img = PIL.Image.new(
            "RGB", (img_size, img_size), (240, 217, 181)
        )  # Board wood color
        draw = PIL.ImageDraw.Draw(img)

        # 2. Draw Grid
        margin = img_size // (board_size + 1)
        grid_spacing = (img_size - 2 * margin) // (board_size - 1)

        # Adjust margin to center the grid exactly
        # grid extends from margin to margin + (board_size-1)*grid_spacing
        start = margin
        end = margin + (board_size - 1) * grid_spacing

        for i in range(board_size):
            # Vertical lines
            x = start + i * grid_spacing
            draw.line([(x, start), (x, end)], fill="black", width=2)
            # Horizontal lines
            y = start + i * grid_spacing
            draw.line([(start, y), (end, y)], fill="black", width=2)

        # 3. Draw Heatmap (squares under stones/intersections)
        # Normalize shapley values
        vmax = np.max(np.abs(shapley_values))
        if vmax < 1e-6:
            vmax = 1.0
        normalized = shapley_values / vmax

        # Square size for heatmap (slightly smaller than spacing to see grid?)
        # Or fill the cell. Let's make it cover the intersection area.
        half_spacing = grid_spacing // 2

        for r in range(board_size):
            for c in range(board_size):
                val = normalized[r, c]
                if abs(val) < 0.05:
                    continue  # Skip negligible values

                # Color: Red (negative) to Green (positive)
                # Using alpha blending manually or just solid blocks with opacity
                # PIL doesn't support alpha on RGB, need RGBA for transparency
                # But we can just draw solid rectangles if we want, or use RGBA image.

                # Let's use RGBA for the heatmap layer
                pass

        # Re-doing with RGBA for heatmap overlay
        heatmap_img = PIL.Image.new("RGBA", (img_size, img_size), (0, 0, 0, 0))
        heatmap_draw = PIL.ImageDraw.Draw(heatmap_img)

        for r in range(board_size):
            for c in range(board_size):
                val = normalized[r, c]
                if abs(val) < 0.01:
                    continue

                x_center = start + c * grid_spacing
                y_center = start + r * grid_spacing

                # Rectangle area
                rect = [
                    (x_center - half_spacing, y_center - half_spacing),
                    (x_center + half_spacing, y_center + half_spacing),
                ]

                # Calculate color
                alpha = int(min(abs(val), 1.0) * 180) + 50  # Base opacity
                if val > 0:
                    # Green
                    color = (0, 255, 0, alpha)
                else:
                    # Red
                    color = (255, 0, 0, alpha)

                heatmap_draw.rectangle(rect, fill=color)

        img = PIL.Image.alpha_composite(img.convert("RGBA"), heatmap_img)
        draw = PIL.ImageDraw.Draw(img)  # Get draw object for composite

        # 4. Draw Stones
        # Observation: [H, W, C]
        # Assuming channel 0 is self, channel 1 is opponent (standard AlphaZero/PGX)
        # Need to know which color 'Self' is.
        # cur_player_id: 0 = Black, 1 = White (usually)

        # Black stones
        black_stones_mask = (
            observation[..., 0] if cur_player_id == 0 else observation[..., 1]
        )
        # White stones
        white_stones_mask = (
            observation[..., 1] if cur_player_id == 0 else observation[..., 0]
        )

        stone_radius = int(grid_spacing * 0.4)

        for r in range(board_size):
            for c in range(board_size):
                x = start + c * grid_spacing
                y = start + r * grid_spacing

                bbox = [
                    (x - stone_radius, y - stone_radius),
                    (x + stone_radius, y + stone_radius),
                ]

                if black_stones_mask[r, c] > 0.5:
                    draw.ellipse(bbox, fill="black", outline="black")
                elif white_stones_mask[r, c] > 0.5:
                    draw.ellipse(bbox, fill="white", outline="black")

        return np.array(img.convert("RGB"))

    def visualize_episode(
        self,
        params: chex.ArrayTree,
        train_state: TrainState,
        key: jax.Array,
        max_steps: int = 100,
    ) -> List[wandb.Image]:
        """Plays one episode and generates explanations for each step."""
        key_init, key_episode = jax.random.split(key)
        state, metadata = self.env_init_fn(key_init)
        eval_state = self.evaluator.init(template_embedding=state)

        images = []
        step_count = 0

        h = self.state_to_nn_input_fn(state).shape[0]

        while not metadata.terminated.all() and step_count < max_steps:
            key_eval, key_episode = jax.random.split(key_episode)

            # 1. Get action from Evaluator
            eval_output = self.evaluator.evaluate(
                key=key_eval,
                eval_state=eval_state,
                env_state=state,
                root_metadata=metadata,
                params=params,
                env_step_fn=self.env_step_fn,
            )

            # 2. Get Shapley values for the current state
            nn_input = self.state_to_nn_input_fn(state)[None, ...]
            out = train_state.apply_fn(params, x=nn_input, train=False, mutable=False)
            if isinstance(out, tuple):
                out = out[0]

            obs_np = np.array(self.state_to_nn_input_fn(state))
            cur_pid = metadata.cur_player_id.item()

            # Prediction Shapley (Value)
            if "prediction_shapley" in out:
                s_vals = np.array(out["prediction_shapley"][0, ..., 0])
                heatmap = self.generate_heatmap(s_vals, obs_np, cur_pid, h)
                images.append(
                    wandb.Image(
                        heatmap,
                        caption=f"Step {step_count} (Player {cur_pid}) - Pred (Value)",
                    )
                )

            # Behaviour Shapley (Policy Features)
            if "behaviour_shapley" in out:
                # Shape is (1, H, W, C) -> (H, W, C)
                b_vals = np.array(out["behaviour_shapley"][0])

                num_channels = b_vals.shape[-1]

                if num_channels == 2:
                    # Approximation mode: 2 latent channels
                    heatmaps = []
                    for c in range(num_channels):
                        hm = self.generate_heatmap(b_vals[..., c], obs_np, cur_pid, h)
                        heatmaps.append(hm)

                    # Combine heatmaps side-by-side
                    combo_img = np.concatenate(
                        heatmaps, axis=1
                    )  # Width-wise concatenation
                    caption = (
                        f"Step {step_count} (Player {cur_pid}) - Bhvr (Feat 0 & 1)"
                    )

                    images.append(wandb.Image(combo_img, caption=caption))

                else:
                    # Strict mode: Full action space explanation
                    # Visualize explanation for the *selected action*
                    selected_action = eval_output.action.item()

                    # Ensure action index is valid for the output
                    if selected_action < num_channels:
                        hm = self.generate_heatmap(
                            b_vals[..., selected_action], obs_np, cur_pid, h
                        )
                        caption = f"Step {step_count} (Player {cur_pid}) - Bhvr (Action {selected_action})"
                        images.append(wandb.Image(hm, caption=caption))

            # 3. Step environment and evaluator state
            state, metadata = self.env_step_fn(state, eval_output.action)
            eval_state = self.evaluator.step(eval_output.eval_state, eval_output.action)
            step_count += 1

        return images
