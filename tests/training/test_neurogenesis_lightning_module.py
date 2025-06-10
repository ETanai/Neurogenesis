import pytest
import torch
from pytorch_lightning.loggers import MLFlowLogger
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from models.ng_autoencoder import NGAutoEncoder
from training.neurogenesis_lightning_module import NeurogenesisLightningModule, build_mlflow_logger
from training.neurogenesis_trainer import NeurogenesisTrainer


class DummyAE(NGAutoEncoder):
    def __init__(self):
        super().__init__(
            input_dim=3, hidden_sizes=[2], activation="identity", activation_last="identity"
        )

    def forward(self, x: Tensor):
        # identity mapping
        return {"recon": x.view(x.size(0), -1), "latent": x.view(x.size(0), -1)}


class DummyTrainer(NeurogenesisTrainer):
    def __init__(self):
        # dummy: thresholds etc not used
        super().__init__(ae=None, ir=None, thresholds=[0.1], max_nodes=[1], max_outliers=0.1)
        self.learn_called = False
        self.history = {"cls": {"layer_errors": [torch.tensor([0.5, 0.5])]}}

    def learn_class(self, class_id, loader):
        assert class_id == "cls"
        # consume loader
        list(loader)
        self.learn_called = True


@pytest.fixture
def dummy_batch():
    # returns (x, y)
    x = torch.randn(4, 3)
    y = torch.zeros(4, dtype=torch.long)
    return x, y


@pytest.fixture
def dummy_loader():
    x = torch.randn(2, 3)
    return DataLoader(TensorDataset(x, torch.zeros(2, dtype=torch.long)), batch_size=2)


@pytest.fixture
def module(tmp_path):
    # instantiate with small dims
    mod = NeurogenesisLightningModule(
        input_dim=3,
        hidden_sizes=[2],
        activation="identity",
        activation_last="identity",
        thresholds=[0.1],
        max_nodes=[1],
        max_outliers=0.1,
        base_lr=1e-3,
        plasticity_epochs=1,
        stability_epochs=1,
        next_layer_epochs=1,
    )
    # inject dummy AE and trainer
    mod.ae = DummyAE()
    mod.trainer_ng = DummyTrainer()
    return mod


def test_configure_optimizers(module):
    assert module.configure_optimizers() == []


def test_training_step_logs(module, dummy_batch):
    logs = {}

    # monkeypatch log method
    def fake_log(key, value, **kwargs):
        logs[key] = value

    module.log = fake_log
    loss = module.training_step(dummy_batch, 0)
    # loss matches MSE between identical inputs
    assert torch.isclose(loss, torch.tensor(0.0))
    assert "pretrain_loss" in logs


def test_set_class_loader_and_on_epoch_end(module, dummy_loader):
    # prepare logging capture
    logs = {}
    module.log = lambda key, val, **kw: logs.setdefault(key, val)
    # set class loader
    module.set_class_loader("cls", dummy_loader)
    # call epoch end
    module.on_train_epoch_end()
    # trainer learned
    assert module.trainer_ng.learn_called
    # logs contain layer0_recon_error and layer0_neurons
    assert "layer0_recon_error" in logs
    assert "layer0_neurons" in logs


def test_build_mlflow_logger(tmp_path):
    uri = "http://test"
    logger = build_mlflow_logger(experiment_name="exp", tracking_uri=uri)
    assert isinstance(logger, MLFlowLogger)
    assert logger._experiment_name == "exp"
    assert logger._tracking_uri == uri


if __name__ == "__main__":
    pytest.main()
