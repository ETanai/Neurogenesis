import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

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
    assert "recon" in out and "latent" in out
    recon, latent = out["recon"], out["latent"]
    assert recon.shape == (4, 3)
    assert latent.shape == (4, 2)


def test_forward_partial_reconstruction(simple_ae, toy_input):
    x_hat = simple_ae.forward_partial(toy_input, layer_idx=0)
    assert x_hat.shape == (4, 3)
    # ensure gradient flows
    x_hat.sum().backward()
    grads = [p.grad is not None for p in simple_ae.parameters()]
    assert any(grads)


def test_reconstruction_error_zero():
    x = torch.randn(5, 10)
    err = NGAutoEncoder.reconstruction_error(x, x)
    assert torch.allclose(err, torch.zeros_like(err))


def test_reconstruction_error_positive(toy_input):
    x = toy_input
    x_hat = toy_input + 1.0
    err = NGAutoEncoder.reconstruction_error(x_hat, x)
    assert torch.all(err > 0)


def test_set_requires_grad_freeze_old(simple_ae):
    # freeze old parameters
    simple_ae.set_requires_grad(freeze_old=True)
    for name, param in simple_ae.named_parameters():
        if "weight_old" in name or "bias_old" in name:
            assert not param.requires_grad, f"{name} should be frozen"
        else:
            assert param.requires_grad, f"{name} should be trainable"


def test_set_requires_grad_unfreeze_all(simple_ae):
    # unfreeze everything
    simple_ae.set_requires_grad(freeze_old=False)
    for name, param in simple_ae.named_parameters():
        assert param.requires_grad, f"{name} should be trainable"


def test_add_new_nodes_single_layer(simple_ae):
    enc_layer = simple_ae.encoder[0]
    dec_layer = simple_ae.decoder[0]
    assert simple_ae.hidden_sizes[0] == 2
    assert enc_layer.out_features_old == 2
    assert enc_layer.out_features_new == 0
    orig_dec_in = dec_layer.in_features
    simple_ae.add_new_nodes(level_idx=0, num_new=3)
    assert simple_ae.hidden_sizes[0] == 5
    assert enc_layer.out_features == 5
    assert dec_layer.in_features == orig_dec_in + 3


def test_plasticity_phase_updates_new_only(simple_ae, toy_input):
    simple_ae.add_new_nodes(level_idx=0, num_new=2)
    loader = DataLoader(TensorDataset(toy_input), batch_size=2)
    enc_layer = simple_ae.encoder[0]
    old_w_old = enc_layer.weight_old.clone()
    old_w_new = enc_layer.weight_new.clone()
    simple_ae.plasticity_phase(loader, level_idx=0, epochs=1, lr=1e-2)
    assert torch.allclose(enc_layer.weight_old, old_w_old, atol=1e-6)
    assert not torch.allclose(enc_layer.weight_new, old_w_new)


def test_stability_phase_updates_all(simple_ae, toy_input):
    simple_ae.add_new_nodes(level_idx=0, num_new=2)
    loader = DataLoader(TensorDataset(toy_input), batch_size=2)
    orig = {name: param.clone() for name, param in simple_ae.named_parameters()}
    simple_ae.stability_phase(loader, lr=1e-3, epochs=1)
    for name, param in simple_ae.named_parameters():
        assert not torch.allclose(param, orig[name])


def test_stability_phase_with_ir(simple_ae, toy_input):
    simple_ae.add_new_nodes(level_idx=0, num_new=1)
    loader = DataLoader(TensorDataset(toy_input), batch_size=2)

    class DummyIR:
        def __init__(self, input_dim):
            self.input_dim = input_dim

        def sample_images(self, cls, n):
            return torch.ones(n, self.input_dim)

    ir = DummyIR(input_dim=3)
    orig = {name: param.clone() for name, param in simple_ae.named_parameters()}
    simple_ae.stability_phase(loader, lr=1e-3, epochs=1, ir=ir, class_id=0, replay_size=2)
    for name, param in simple_ae.named_parameters():
        assert not torch.allclose(param, orig[name])


if __name__ == "__main__":
    pytest.main()
