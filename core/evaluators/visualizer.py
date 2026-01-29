import io

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

import wandb


class GoVisualizer:
    def __init__(self, pos_len=19):
        self.pos_len = pos_len
        # Custom colormap: Red (-1) -> White (0) -> Green (1)
        colors = [(1, 0, 0), (1, 1, 1), (0, 1, 0)]  # R, W, G
        self.cmap = LinearSegmentedColormap.from_list("RdWhGn", colors)

    def plot_shapley(self, shapley_values, stones=None, title=None, caption=None):
        """
        Plots 19x19 Shapley values with stone overlays.

        Args:
            shapley_values: (19, 19) array of Shapley values.
            stones: (19, 19, 2) array where [..., 0] is pla and [..., 1] is opp.
            title: Title for the plot.
            caption: Caption for the wandb.Image.

        Returns:
            wandb.Image
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        # Normalize shapley values to [-1, 1] if needed, or use a fixed scale
        # The user wants -1 to 1 gradient.
        vmax = max(np.abs(shapley_values).max(), 1e-8)
        # Use symmetric norm
        im = ax.imshow(
            shapley_values, cmap=self.cmap, vmin=-vmax, vmax=vmax, origin="upper"
        )

        # Draw 19x19 grid
        ax.set_xticks(np.arange(-0.5, self.pos_len, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.pos_len, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)

        # Remove major ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Overlay stones
        if stones is not None:
            # stones: (19, 19, 2) -> (H, W, C)
            # Ch 0: PLAYER (black stones usually?), Ch 1: OPPONENT (white stones?)
            # In KataGo features, Ch 1 is 'pla', Ch 2 is 'opp'.
            # We assume the input 'stones' follows this.
            for r in range(self.pos_len):
                for c in range(self.pos_len):
                    if stones[r, c, 0] > 0.5:  # Player
                        circle = plt.Circle(
                            (c, r), 0.35, color="black", fill=True, zorder=10
                        )
                        ax.add_patch(circle)
                    elif stones[r, c, 1] > 0.5:  # Opponent
                        circle = plt.Circle(
                            (c, r),
                            0.35,
                            color="white",
                            fill=True,
                            edgecolor="black",
                            zorder=10,
                        )
                        ax.add_patch(circle)

        if title:
            ax.set_title(title, fontsize=16)

        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Convert plt to image
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)

        return wandb.Image(Image.open(buf), caption=caption)


def spearman_correlation(x, y):
    """Computes Spearman rank correlation between two vectors."""
    import scipy.stats

    cor, _ = scipy.stats.spearmanr(x, y)
    return cor
