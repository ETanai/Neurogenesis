from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch

from models.ng_autoencoder import NGAutoEncoder
from training.neurogenesis_trainer import NeurogenesisTrainer


def plot_recon_error_history(
    trainer: NeurogenesisTrainer, class_id: Any, figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot mean reconstruction-error vs growth iteration for each hidden layer.

    Args:
        trainer: NeurogenesisTrainer with history recorded
        class_id: identifier for the class whose history to plot
        figsize: size of the matplotlib figure
    Returns:
        matplotlib Figure object
    """
    history: List[torch.Tensor] = trainer.history[class_id]["layer_errors"]
    fig, ax = plt.subplots(figsize=figsize)
    # For each growth snapshot, history[i] is error tensor for that iteration
    # Plot mean error over iterations
    means: List[float] = [errs[0].mean().item() for errs in history]
    ax.plot(range(len(means)), means, marker="o")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean Reconstruction Error")
    ax.set_title(f"Class {class_id} RE History")
    ax.grid(True)
    return fig


def plot_recon_grid(
    ae: NGAutoEncoder,
    x: torch.Tensor,
    view_shape: Optional[Tuple[int, ...]] = None,
    ncols: int = 8,
    figsize: Tuple[int, int] = (6, 6),
) -> plt.Figure:
    """
    Display original vs reconstructed images in a grid.

    Args:
        ae: trained NGAutoEncoder
        x: input batch tensor
        view_shape: shape to view each image as (e.g. (1,28,28))
        ncols: number of columns in the grid
        figsize: size of the matplotlib figure
    Returns:
        matplotlib Figure object
    """
    # get paired originals and reconstructions
    device = next(ae.parameters()).device
    x.to(device)
    paired = ae.grid_recon(x, view_shape=view_shape)  # shape (2B, C, H, W) or (2B, features)
    num = paired.shape[0]
    nrows = (num + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    for idx in range(num):
        img = paired[idx]
        # convert to 2D array
        if img.ndim == 1:
            side = int(img.numel() ** 0.5)
            arr = img.view(side, side).detach().numpy()
        else:
            arr = img.squeeze().detach().numpy()
        axes[idx].imshow(arr, interpolation="nearest")
        axes[idx].axis("off")
    # hide extra axes
    for ax in axes[num:]:
        ax.axis("off")
    fig.tight_layout()
    return fig
