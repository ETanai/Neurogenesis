import io
from typing import Any, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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


def plot_recon_grid_mlflow(
    ae,
    x: torch.Tensor,
    view_shape: Optional[Tuple[int, ...]] = None,
    ncols: int = 8,
    figsize: Tuple[int, int] = (6, 6),
    return_mlflow_artifact: bool = False,
    artifact_name: str = "reconstructions.png",
) -> Union[plt.Figure, Tuple[plt.Figure, bytes]]:
    """
    Display original vs reconstructed images in a grid, and optionally
    return the PNG bytes for MLflow logging.

    Args:
        ae: trained autoencoder with a .grid_recon(x, view_shape) method
        x: input batch tensor
        view_shape: shape to view each image as (e.g. (1,28,28))
        ncols: number of columns in the grid
        figsize: size of the matplotlib figure
        return_mlflow_artifact: if True, return (fig, png_bytes) instead of just fig
        artifact_name: suggested filename for the MLflow artifact

    Returns:
        If return_mlflow_artifact is False:
            plt.Figure
        Else:
            (plt.Figure, bytes)  – PNG data you can call, e.g.:
                mlflow.log_image(png_bytes, artifact_name)
    """
    # move to same device as model
    device = next(ae.parameters()).device
    x = x.to(device)

    # get originals + reconstructions: shape (2B, C, H, W) or (2B, features)
    paired = ae.grid_recon(x, view_shape=view_shape)
    num = paired.shape[0]
    nrows = (num + ncols - 1) // ncols

    # build figure
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for idx in range(num):
        img = paired[idx]
        if img.ndim == 1:
            side = int(img.numel() ** 0.5)
            arr = img.view(side, side).detach().cpu().numpy()
        else:
            arr = img.squeeze().detach().cpu().numpy()

        axes[idx].imshow(arr, interpolation="nearest")
        axes[idx].axis("off")

    for ax in axes[num:]:
        ax.axis("off")

    fig.tight_layout()

    if return_mlflow_artifact:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        png_bytes = buf.getvalue()
        buf.close()
        return fig, png_bytes

    return fig


def plot_partial_recon_grid_mlflow(
    ae,
    x: torch.Tensor,
    view_shape: Optional[Tuple[int, ...]] = None,
    levels: Optional[Sequence[int]] = None,
    ncols: Optional[int] = None,
    figsize: Tuple[int, int] = (10, 6),
    col_group_titles: Optional[List[str]] = None,  # e.g. ["Pre-trained", "Novel"]
    col_group_splits: Optional[List[int]] = None,  # cumulative column counts, e.g. [8, 16]
    return_mlflow_artifact: bool = False,
    artifact_name: str = "partial_reconstructions.png",
) -> Union[plt.Figure, Tuple[plt.Figure, bytes]]:
    """
    Make a grid with one row of inputs and one row per partial reconstruction level.
    Uses ae.forward_partial(x, layer_idx=k) to decode from the mirror of encoder layer k.

    Args:
        ae: model exposing .forward_partial(x, layer_idx, ret_lat=False)
            and attributes .hidden_sizes (used to infer default levels if None).
        x: input batch tensor [B, ...].
        view_shape: final HxW or CxHxW shape for imshow if x/recons are flat.
        levels: which encoder layer indices to visualize (0 = first hidden layer).
                Defaults to range(len(ae.hidden_sizes)).
        ncols: number of columns to show; defaults to min(B, 16).
        figsize: matplotlib figure size.
        col_group_titles: optional list of titles to annotate column groups.
        col_group_splits: optional cumulative column counts for groups (same length as titles).
        return_mlflow_artifact: if True, returns (fig, png_bytes).
        artifact_name: suggested filename for MLflow (not used by this function, just a hint).

    Returns:
        Figure or (Figure, png_bytes).
    """
    # --- Prep ---
    device = next(ae.parameters()).device
    x = x.to(device)
    B = x.shape[0]
    if ncols is None:
        ncols = min(B, 16)

    if levels is None:
        # assume one level per hidden size
        try:
            n_lvls = len(ae.hidden_sizes)
            levels = list(range(n_lvls))
        except Exception:
            # fallback: try a few layers
            levels = [0, 1, 2, 3]

    # restrict batch to ncols (visual clarity)
    x_vis = x[:ncols]

    # gather reconstructions
    # row 0 = inputs; rows 1.. = partial recon at each level
    rows = [x_vis]
    with torch.no_grad():
        for k in levels:
            rows.append(ae.forward_partial(x_vis, layer_idx=int(k)))  # [B, ...] each

    nrows = 1 + len(levels)

    # --- Build figure ---
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    if nrows == 1:
        axes = axes[None, :]  # ensure 2D indexing

    def _to_numpy_img(t: torch.Tensor):
        t = t.detach().float().cpu()
        # resolve shape
        if view_shape is not None:
            t = t.view((t.size(0),) + tuple(view_shape))
        else:
            if t.ndim == 2:  # [B, F] -> assume square grayscale
                side = int(t.size(1) ** 0.5)
                t = t.view(t.size(0), 1, side, side)
            elif t.ndim == 3:  # [B, H, W] -> add channel dim
                t = t.unsqueeze(1)
            # else [B, C, H, W] already
        return t

    # convert all rows to [B, C, H, W] numpy
    rows_np = [_to_numpy_img(r) for r in rows]

    # --- Plot ---
    for r in range(nrows):
        batch = rows_np[r]
        is_gray = batch.shape[1] == 1
        for c in range(ncols):
            ax = axes[r, c]
            img = batch[c]
            if is_gray:
                ax.imshow(img.squeeze(0), cmap="gray", interpolation="nearest")
            else:
                ax.imshow(img.transpose(0, 1).transpose(1, 2))  # C,H,W -> H,W,C
            ax.axis("off")

    # row labels on the left (similar to "Input / Level 1..")
    # place them on the first column's axes titles to avoid extra layout fuss
    axes[0, 0].set_ylabel("Input", rotation=0, labelpad=30, va="center")
    for i, k in enumerate(levels, start=1):
        axes[i, 0].set_ylabel(f"Level {k + 1}", rotation=0, labelpad=30, va="center")

    # optional column group headers (spanning subsets of columns) and separators
    if col_group_titles and col_group_splits:
        assert len(col_group_titles) == len(col_group_splits), (
            "col_group_titles and col_group_splits must have same length"
        )
        # normalize splits to number of columns
        splits = [min(max(s, 0), ncols) for s in col_group_splits]
        start = 0
        for title, end in zip(col_group_titles, splits):
            if end <= start:
                continue
            span_cols = end - start
            center_col = start + span_cols / 2.0 - 0.5
            x_frac = (center_col + 0.5) / ncols
            fig.text(x_frac, 0.99, title, ha="center", va="top")
            start = end

        # draw separators between column groups for clarity
        for boundary in splits[:-1]:
            if boundary <= 0 or boundary >= ncols:
                continue
            x_frac = boundary / ncols
            line = Line2D(
                [x_frac, x_frac],
                [0.02, 0.98],
                transform=fig.transFigure,
                color="tab:red",
                linewidth=2.0,
                linestyle="--",
                alpha=0.85,
            )
            fig.add_artist(line)

    # overall left margin so row labels aren't cut off
    fig.subplots_adjust(left=0.08)

    if return_mlflow_artifact:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        png_bytes = buf.getvalue()
        buf.close()
        return fig, png_bytes

    return fig
