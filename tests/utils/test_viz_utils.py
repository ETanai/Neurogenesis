import matplotlib
import pytest
import torch

matplotlib.use("Agg")  # no display

import matplotlib.pyplot as plt

from utils.viz_utils import plot_recon_error_history, plot_recon_grid


# Dummy trainer for error history
class DummyTrainer:
    def __init__(self):
        # history with three snapshots of errors for a class
        self.history = {
            "cls": {
                "layer_errors": [
                    torch.tensor([1.0, 2.0, 3.0]),
                    torch.tensor([2.0, 3.0, 4.0]),
                    torch.tensor([3.0, 4.0, 5.0]),
                ]
            }
        }
        # stub ae for attributes if needed
        self.ae = None


# Dummy AE for grid
class DummyAE:
    def __init__(self):
        pass

    def grid_recon(self, x: torch.Tensor, view_shape=None):
        # echo input flat + input flat reversed
        B = x.size(0)
        flat = x.view(B, -1)
        # create paired as [flat, flat] for each sample
        paired = torch.cat([flat, flat], dim=0)
        if view_shape:
            return paired.view(-1, *view_shape)
        return paired


@pytest.fixture
def dummy_trainer():
    return DummyTrainer()


@pytest.fixture
def toy_input():
    # single sample of 4 features to make 2x2 image
    return torch.tensor([[0.0, 1.0, 2.0, 3.0]])


def test_plot_recon_error_history(dummy_trainer):
    fig = plot_recon_error_history(dummy_trainer, "cls", figsize=(4, 3))
    # one axis
    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]
    # one line plotted
    lines = ax.get_lines()
    assert len(lines) == 1
    ydata = lines[0].get_ydata()
    # means: [2.0, 3.0, 4.0]
    assert list(ydata) == [2.0, 3.0, 4.0]
    # labels and title
    assert ax.get_xlabel() == "Iteration"
    assert ax.get_ylabel() == "Mean Reconstruction Error"
    assert "Class cls" in ax.get_title()


def test_plot_recon_grid_default(toy_input):
    ae = DummyAE()
    fig = plot_recon_grid(ae, toy_input, view_shape=None, ncols=2, figsize=(4, 4))
    assert isinstance(fig, plt.Figure)
    # grid should have 1 sample => paired length 2, ncols=2 => nrows=1
    axes = fig.axes
    # two axes for the two images
    assert len(axes) == 2
    # first axis image array matches
    img = axes[0].images[0].get_array()
    # img shape: computed side = sqrt(4)=2 => 2x2
    assert img.shape == (2, 2)


def test_plot_recon_grid_with_view_shape(toy_input):
    # view as (1,2,2)
    ae = DummyAE()
    fig = plot_recon_grid(ae, toy_input, view_shape=(1, 2, 2), ncols=2)
    assert isinstance(fig, plt.Figure)
    axes = fig.axes
    assert len(axes) == 2
    img = axes[1].images[0].get_array()
    assert img.shape == (2, 2)


if __name__ == "__main__":
    pytest.main()
