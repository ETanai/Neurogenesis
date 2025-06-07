import pytest
import torch

from models.autoencoder import AutoEncoder


@pytest.mark.parametrize(
    "batch_size, input_dim, hidden_sizes, activation",
    [
        (4, 28 * 28, [64, 32], "relu"),
        (2, 16, [5], "tanh"),
    ],
)
def test_autoencoder_forward_shapes(batch_size, input_dim, hidden_sizes, activation):
    # create dummy input: (batch_size, 1, sqrt(input_dim), sqrt(input_dim)) if square
    side = int(input_dim**0.5)
    x = torch.randn(batch_size, 1, side, side)

    ae = AutoEncoder(input_dim=input_dim, hidden_sizes=hidden_sizes, activation=activation)
    out = ae(x)

    # reconstructed should be flattened back to (batch_size, input_dim)
    assert isinstance(out, dict)
    assert "recon" in out and "latent" in out

    recon = out["recon"]
    latent = out["latent"]

    assert recon.shape == (batch_size, input_dim), f"got {recon.shape}"
    # last hidden size defines latent dim
    assert latent.shape == (batch_size, hidden_sizes[-1]), f"got {latent.shape}"


def test_autoencoder_invalid_hidden_sizes():
    with pytest.raises(ValueError):
        AutoEncoder(input_dim=10, hidden_sizes=[], activation="relu")
