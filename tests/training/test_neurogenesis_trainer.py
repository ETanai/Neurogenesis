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

    def plasticity_phase(self, loader, level, epochs, lr):
        self.plastic_calls.append((level, epochs, lr))

    def stability_phase(self, loader, lr, epochs, ir, class_id, replay_size):
        self.stability_calls.append((lr, epochs, class_id, replay_size))


class DummyIR:
    def __init__(self):
        self.fitted = False

    def fit(self, loader):
        self.fitted = True

    def sample_images(self, class_id, replay_size):
        # return dummy samples of correct feature size
        # feature size doesn't matter; new_loader passes x only
        return torch.zeros(replay_size, 1)


@pytest.fixture
def dummy_loader():
    # 4 samples, 1 feature
    x = torch.randn(4, 1)
    return DataLoader(TensorDataset(x), batch_size=2)


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

    # For each level, should add nodes twice (max_nodes=2)
    expected_adds = [(0, int(max_outliers * 4))] * 2 + [(1, int(max_outliers * 4))] * 2
    assert ae.added == expected_adds

    # plasticity_phase calls: two for each level in loop + one for next layer per level 0
    # Level 0: 2 plasticity in loop, then 1 next-layer plasticity
    # Level 1: 2 plasticity in loop, no next-layer
    assert len(ae.plastic_calls) == 5
    # stability_phase calls: 2 per level in loop
    assert len(ae.stability_calls) == 4


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
