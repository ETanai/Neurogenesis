import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from training.neurogenesis_trainer import NeurogenesisTrainer


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

        self._layer_cls = _Layer
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
        self.promotions = getattr(self, "promotions", 0) + 1

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

    # For each level we should attempt at least one growth with positive additions
    for level in range(len(max_nodes)):
        level_adds = [num for lvl, num in ae.added if lvl == level]
        assert len(level_adds) >= 1
        assert all(num > 0 for num in level_adds)

    # plasticity/stability phases should be invoked for every growth round
    assert len(ae.plastic_calls) >= len(ae.added)
    assert len(ae.stability_calls) >= len(ae.added)

    # Next-layer fine-tuning should use a reduced learning rate at least once
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
