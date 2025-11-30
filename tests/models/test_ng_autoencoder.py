import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.ng_autoencoder import EarlyStopper, NGAutoEncoder
from training.neurogenesis_trainer import NeurogenesisTrainer

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


class DummyAE:
    def __init__(self, hidden_sizes):
        self.hidden_sizes = hidden_sizes.copy()
        self.added = []
        self.plastic_calls = []
        self.stability_calls = []
        self.encoder = []

        class _Layer:
            def __init__(self, width):
                self.n_out_features = width

            def add_plastic_nodes(self, num_new):
                self.n_out_features += num_new

            def adjust_input_size(self, num_new):
                self.n_out_features += num_new

        for width in self.hidden_sizes:
            self.encoder.append(_Layer(width))
            self.encoder.append(object())

    def forward_partial(self, x, level):
        # return dummy reconstruction (same shape as x flattened)
        return x.view(x.size(0), -1)

    @staticmethod
    def reconstruction_error(x_hat, x):
        # return constant error of 1 for each sample
        return torch.ones(x.size(0))

    def add_new_nodes(self, level, num_new):
        self.added.append((level, num_new))
        # simulate growth of hidden_sizes
        self.hidden_sizes[level] += num_new
        layer = self.encoder[2 * level]
        if hasattr(layer, "add_plastic_nodes"):
            layer.add_plastic_nodes(num_new)

    def _plastic_to_mature(self):
        return None

    def plasticity_phase(self, loader, level, epochs, lr, **kwargs):
        self.plastic_calls.append((level, epochs, lr))
        return {"epoch_loss": [1.0]}

    def stability_phase(
        self,
        loader,
        level,
        lr,
        epochs,
        old_x=None,
        early_stop_cfg=None,
        forward_fn=None,
    ):
        replay_size = 0 if old_x is None else old_x.size(0)
        self.stability_calls.append((level, epochs, lr, replay_size))
        return {"epoch_loss": [0.5]}


class DummyIR:
    def __init__(self):
        self.fitted = False
        self._classes: list[int] = []

    def fit(self, loader):
        self.fitted = True
        self._classes = [0]

    def available_classes(self):
        return self._classes

    def sample_images(self, class_id, replay_size):
        # return dummy samples of correct feature size
        # feature size doesn't matter; new_loader passes x only
        return torch.zeros(replay_size, 28 * 28)


@pytest.fixture
def dummy_loader():
    # 4 samples, 1 feature
    x = torch.randn(4, 1)
    y = torch.zeros(4, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=2)


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


def test_early_stopper_goal_triggers():
    stopper = EarlyStopper(min_delta=0.5, patience=3, mode="min", goal=0.2)
    assert not stopper.step(0.8)
    assert stopper.step(0.2)
    assert stopper.should_stop


def test_set_requires_grad_freeze_old(simple_ae):
    # freeze old parameters
    simple_ae.set_requires_grad(freeze_old=True)
    for name, param in simple_ae.named_parameters():
        if "weight_mature" in name or "bias_mature" in name:
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
    assert enc_layer.out_features_mature == 2
    assert enc_layer.out_features_plastic == 0
    orig_dec_in = dec_layer.in_features
    simple_ae.add_new_nodes(0, 3)
    assert simple_ae.hidden_sizes[0] == 5
    assert enc_layer.out_features == 5
    assert dec_layer.in_features == orig_dec_in + 3


def test_plasticity_phase_updates_new_only(simple_ae, toy_input):
    simple_ae.add_new_nodes(0, 2)
    loader = DataLoader(TensorDataset(toy_input), batch_size=2)
    enc_layer = simple_ae.encoder[0]
    old_w_old = enc_layer.weight_mature.clone()
    old_w_new = enc_layer.weight_plastic.clone()
    simple_ae.plasticity_phase(loader, level=0, epochs=1, lr=1e-2)
    assert torch.allclose(enc_layer.weight_mature, old_w_old, atol=1e-3)
    assert not torch.allclose(enc_layer.weight_plastic, old_w_new)


def test_plasticity_phase_epoch_logger(simple_ae, toy_input):
    simple_ae.add_new_nodes(0, 1)
    loader = DataLoader(TensorDataset(toy_input), batch_size=2)
    calls = []

    def _logger(epoch_idx, summary):
        calls.append((epoch_idx, summary.get("phase"), summary.get("loss")))

    simple_ae.plasticity_phase(loader, level=0, epochs=2, lr=1e-3, epoch_logger=_logger)
    assert len(calls) == 2
    assert calls[0][0] == 0
    assert all(step == "plasticity" for _, step, _ in calls)
    assert all(isinstance(loss, float) for _, _, loss in calls)


def test_stability_phase_updates_all(simple_ae, toy_input):
    simple_ae.add_new_nodes(0, 2)
    loader = DataLoader(TensorDataset(toy_input), batch_size=2)
    orig = {name: param.clone() for name, param in simple_ae.named_parameters()}
    simple_ae.stability_phase(loader, level=0, lr=1e-3, epochs=1, old_x=None)
    for name, param in simple_ae.named_parameters():
        assert not torch.allclose(param, orig[name])


def test_stability_phase_with_ir(simple_ae, toy_input):
    simple_ae.add_new_nodes(0, 1)
    loader = DataLoader(TensorDataset(toy_input), batch_size=2)
    replay = torch.ones(loader.dataset.tensors[0].size(0), simple_ae.input_dim)
    orig = {name: param.clone() for name, param in simple_ae.named_parameters()}
    simple_ae.stability_phase(loader, level=0, lr=1e-3, epochs=1, old_x=replay)
    for name, param in simple_ae.named_parameters():
        assert not torch.allclose(param, orig[name])


# def test_grid_recon_flat(simple_ae, toy_input):
#     # no reshaping
#     out = simple_ae.grid_recon(toy_input)
#     B, F_in = toy_input.shape
#     # grid_recon should double the batch (orig + recon)
#     assert out.shape == (2 * B, F_in)
#     # even rows == original flatten, odd rows == reconstructions
#     torch.testing.assert_allclose(out[0], toy_input.view(B, -1)[0])
#     torch.testing.assert_allclose(out[1], out[0])  # identity AE: recon == orig


def test_grid_recon_view_shape(simple_ae, toy_input):
    # reshape into (1,3)
    out = simple_ae.grid_recon(toy_input, view_shape=(1, 3))
    B, F_in = toy_input.shape
    assert out.shape == (2 * B, 1, 3)
    # check a sample
    torch.testing.assert_allclose(out[2], toy_input.view(B, -1)[1].view(1, 3))


# --- in tests/training/test_neurogenesis_trainer.py ---


def test_learn_class_calls(monkeypatch, dummy_loader):
    # Setup DummyAE with 2 layers
    ae = DummyAE(hidden_sizes=[1, 1])
    ir = DummyIR()
    # set thresholds low so reconstruction_error=1 > threshold
    thresholds = [0.5, 0.5]
    max_nodes = [2, 2]
    max_outliers = 0.5
    trainer = NeurogenesisTrainer(ae, ir, thresholds, max_nodes, max_outliers, base_lr=0.1)

    # Perform learning
    trainer.learn_class(class_id=7, loader=dummy_loader)

    # IR.fit should be called
    assert ir.fitted

    for level in range(len(max_nodes)):
        level_adds = [num for lvl, num in ae.added if lvl == level]
        assert len(level_adds) >= 1
        assert all(num > 0 for num in level_adds)

    assert len(ae.plastic_calls) >= len(ae.added)
    assert len(ae.stability_calls) >= len(ae.added)
    assert any(lr < 0.01 for _, _, lr in ae.plastic_calls)


def test_get_recon_errors_simple(dummy_loader):
    # Test _get_recon_errors with real data and AE returning zeros
    # Monkeypatch AE
    class AEStub:
        hidden_sizes = [1]

        def forward_partial(self, x, level):
            return torch.zeros(x.size(0), 1)

        @staticmethod
        def reconstruction_error(x_hat, x):
            return torch.tensor([0.2, 0.2])

    ae = AEStub()
    from training.neurogenesis_trainer import NeurogenesisTrainer

    trainer = NeurogenesisTrainer(ae, None, thresholds=[0], max_nodes=[1], max_outliers=0)
    errs = trainer._get_recon_errors(dummy_loader, level=0)
    # loader has 4 samples, batch size 2, returns two batches of size2 with error 0.2 => length4
    assert torch.allclose(errs, torch.tensor([0.2, 0.2, 0.2, 0.2]))


if __name__ == "__main__":
    pytest.main()
