import pytest
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.loggers import MLFlowLogger

from training.train_ng_ae import build_mlflow_logger

# import the decorated main; use __wrapped__ to call original


@pytest.fixture
def cfg_minimal(tmp_path):
    # Build minimal config matching config/train.yaml structure
    cfg = OmegaConf.create(
        {
            "trainer": {"max_epochs": 1, "accelerator": "cpu", "log_every_n_steps": 1},
            "datamodule": {"data_dir": str(tmp_path), "batch_size": 2, "num_workers": 0},
            "model": {"hidden_sizes": [2], "activation": "identity", "activation_last": "identity"},
            "neurogenesis": {
                "thresholds": [0.1],
                "max_nodes": [1],
                "max_outliers": 0.1,
                "base_lr": 1e-3,
                "plasticity_epochs": 1,
                "stability_epochs": 1,
                "next_layer_epochs": 1,
            },
            "ir": {"enabled": False, "class_sequence": []},
            "mlflow": {"tracking_uri": "http://test", "experiment_name": "exp"},
        }
    )
    return cfg


class DummyDM:
    def __init__(self, data_dir, batch_size, num_workers):
        self.called = True

    def train_dataloader(self):
        return torch.utils.data.DataLoader(torch.zeros(4, 1), batch_size=2)

    def get_class_dataset(self, class_id):
        # return object with to_dataloader
        class DS:
            def to_dataloader(self, batch_size):
                return torch.utils.data.DataLoader(torch.zeros(2, 1), batch_size=batch_size)

        return DS()


class DummySolver(pl.Trainer):
    def __init__(self, **kwargs):
        super().__init__(**{})
        self.fit_calls = []

    def fit(self, model, *args, **kwargs):
        self.fit_calls.append((args, kwargs))


@pytest.fixture(autouse=True)
def patch_external(monkeypatch):
    # patch DM and Trainer
    monkeypatch.setattr("data.mnist_datamodule.MNISTDataModule", DummyDM)
    monkeypatch.setattr(pl, "Trainer", DummySolver)


@pytest.fixture
def mlflow_cfg():
    from omegaconf import OmegaConf

    return OmegaConf.create({"mlflow": {"experiment_name": "exp", "tracking_uri": "uri"}})


def test_build_mlflow_logger(mlflow_cfg):
    logger = build_mlflow_logger(mlflow_cfg)
    assert isinstance(logger, MLFlowLogger)
    assert logger._experiment_name == "exp"
    assert logger._tracking_uri == "uri"


if __name__ == "__main__":
    pytest.main()
