import pytest
import torch

from models.ng_autoencoder import NGAutoEncoder

# Reproducibility
torch.manual_seed(0)


@pytest.fixture
def toy_input():
    # batch of 4 samples, 3 features
    return torch.randn(4, 3)


@pytest.fixture
def simple_ae():
    # input_dim=3, hidden sizes [2]
    return NGAutoEncoder(
        input_dim=3, hidden_sizes=[2], activation="identity", activation_last="identity"
    )


def test_forward_shape(simple_ae, toy_input):
    out = simple_ae.forward(toy_input)
    # recon and latent keys
    assert "recon" in out and "latent" in out
    recon, latent = out["recon"], out["latent"]
    # recon shape: (batch, input_dim)
    assert recon.shape == (4, 3)
    # latent shape: (batch, hidden_sizes[0])
    assert latent.shape == (4, 2)


def test_forward_partial_reconstruction(simple_ae, toy_input):
    # partial through layer 0 (only one hidden layer)
    x_hat = simple_ae.forward_partial(toy_input, layer_idx=0)
    # shape matches recon
    assert x_hat.shape == (4, 3)
    # if weights are identity-like, may not exactly equal; check differentiable path
    # ensure gradient flows
    x_hat.sum().backward()
    # params have grad
    grads = [p.grad is not None for p in simple_ae.parameters()]
    assert any(grads)


def test_reconstruction_error_zero():
    # identical tensors -> zero error
    x = torch.randn(5, 10)
    err = NGAutoEncoder.reconstruction_error(x, x)
    assert torch.allclose(err, torch.zeros_like(err))


def test_reconstruction_error_positive(toy_input):
    # shifted input
    x = toy_input
    x_hat = toy_input + 1.0
    err = NGAutoEncoder.reconstruction_error(x_hat, x)
    # error > 0 for all samples
    assert torch.all(err > 0)


if __name__ == "__main__":
    pytest.main()
