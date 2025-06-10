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


def test_grid_recon_flat(simple_ae, toy_input):
    # no reshaping
    out = simple_ae.grid_recon(toy_input)
    B, F_in = toy_input.shape
    # grid_recon should double the batch (orig + recon)
    assert out.shape == (2 * B, F_in)
    # even rows == original flatten, odd rows == reconstructions
    torch.testing.assert_allclose(out[0], toy_input.view(B, -1)[0])
    torch.testing.assert_allclose(out[1], out[0])  # identity AE: recon == orig


def test_grid_recon_view_shape(simple_ae, toy_input):
    # reshape into (1,3)
    out = simple_ae.grid_recon(toy_input, view_shape=(1, 3))
    B, F_in = toy_input.shape
    assert out.shape == (2 * B, 1, 3)
    # check a sample
    torch.testing.assert_allclose(out[2], toy_input.view(B, -1)[1].view(1, 3))


# --- in tests/training/test_neurogenesis_trainer.py ---


def test_history_tracking(dummy_loader):
    # stub AE so that no growth occurs (errors below threshold)
    class AEStub:
        hidden_sizes = [1, 1]

        def forward_partial(self, x, level):
            return x.view(x.size(0), -1)

        @staticmethod
        def reconstruction_error(x_hat, x):
            # always small error => no growth
            return torch.zeros(x.size(0))

        def add_new_nodes(self, *args, **kwargs):
            pass

        def plasticity_phase(self, *args, **kwargs):
            pass

        def stability_phase(self, *args, **kwargs):
            pass

    class IRStub:
        def fit(self, loader):
            pass

        def sample_images(self, *args, **kwargs):
            return torch.zeros(1, 1)

    from training.neurogenesis_trainer import NeurogenesisTrainer

    ae = AEStub()
    ir = IRStub()
    thresholds = [0.1, 0.1]
    max_nodes = [5, 5]
    trainer = NeurogenesisTrainer(ae, ir, thresholds, max_nodes, max_outliers=0.5)
    trainer.learn_class(class_id="test", loader=dummy_loader)

    # history should have an entry for each layer, once before growth
    hist = trainer.history["test"]["layer_errors"]
    # two layers => two snapshots
    assert len(hist) == 2
    # each Tensor should have length == number of samples (4)
    for errs in hist:
        assert isinstance(errs, torch.Tensor)
        assert errs.numel() == len(dummy_loader.dataset)


if __name__ == "__main__":
    pytest.main()
