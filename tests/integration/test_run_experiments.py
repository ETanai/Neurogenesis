from pathlib import Path
from typing import Any
import json

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from scripts.run_experiments import run


def make_toy_cfg():
    return OmegaConf.create(
        {
            "seed": 7,
            "data": {
                "name": "toy",
                "batch_size": 8,
                "num_workers": 0,
                "data_dir": None,
                "num_classes": 3,
                "train_samples_per_class": 12,
                "val_samples_per_class": 6,
                "noise_scale": 0.02,
                "seed": 101,
            },
            "model": {
                "input_dim": 28 * 28,
                "hidden_sizes": [6, 3],
                "activation": "tanh",
                "activation_latent": "identity",
                "activation_last": "sigmoid",
            },
            "experiment": {
                "dataset": "toy",
                "regime": "ndl_ir",
                "base_classes": [0],
                "incremental_classes": [1],
                "threshold": {"percentile": 0.95, "margin": 0.02, "minimum": 0.0},
            },
            "training": {
                "device": "cpu",
                "validate": True,
                "pretrain_epochs": 1,
                "base_lr": 5e-3,
                "weight_decay": 0.0,
                "incremental_epochs": 2,
            },
            "neurogenesis": {
                "max_nodes": [2, 2],
                "max_outlier_fraction": 0.2,
                "plasticity_epochs": 1,
                "stability_epochs": 1,
                "next_layer_epochs": 1,
                "factor_max_new_nodes": 0.6,
                "factor_new_nodes": 0.5,
                "early_stop": {"min_delta": 1e-4, "patience": 2, "mode": "min"},
            },
            "replay": {
                "enabled": True,
                "cov_eps": 1e-4,
                "stats_batch_size": 4,
                "batch_ratio": 0.5,
            },
            "logging": {"mlflow": {"enabled": False}},
        }
    )


@pytest.mark.parametrize(
    "regime, expect_replay",
    [
        ("ndl_ir", True),
        ("ndl", False),
        ("cl_ir", True),
        ("cl", False),
    ],
)
def test_run_pipeline_with_toy_data(regime, expect_replay):
    cfg = make_toy_cfg()
    cfg.experiment["regime"] = regime
    result = run(cfg)
    model = result["model"]
    replay = result["replay"]
    trainer = result["trainer"]

    assert len(model.hidden_sizes) == 2
    if expect_replay:
        assert replay is not None
    else:
        assert replay is None

    assert trainer._class_count == 1
    assert 1 in trainer.history
    if regime.startswith("ndl"):
        assert result["growth_reports"][1]["levels"]
        assert "foreground_mse" in result["eval_records"][-1]


@pytest.mark.parametrize("regime", ["ndl", "cl"])
def test_no_replay_regimes_stay_replay_free_with_dataset_backend(regime):
    cfg = make_toy_cfg()
    cfg.experiment.regime = regime
    cfg.replay.mode = "dataset"
    cfg.replay.enabled = True

    result = run(cfg)

    assert result["replay"] is None
    assert result["trainer"].ir is None


def test_threshold_refresh_uses_only_previously_learned_classes():
    cfg = make_toy_cfg()
    cfg.experiment.incremental_classes = [1, 2]
    cfg.experiment.threshold.refresh_before_class = True

    result = run(cfg)
    history = result["threshold_history"]

    assert [entry["stage"] for entry in history] == [
        "initial",
        "before_class",
        "before_class",
    ]
    assert history[1]["class_id"] == 1
    assert history[1]["calibration_classes"] == [0]
    assert history[2]["class_id"] == 2
    assert history[2]["calibration_classes"] == [0, 1]
    assert result["trainer"].thresholds == history[-1]["thresholds"]


def test_threshold_refresh_can_use_intrinsic_replay_without_old_class_refits():
    cfg = make_toy_cfg()
    cfg.experiment.incremental_classes = [1, 2]
    cfg.experiment.threshold.refresh_before_class = True
    cfg.experiment.threshold.refresh_source = "replay"
    cfg.experiment.threshold.refresh_samples_per_class = 8
    cfg.replay.mode = "intrinsic"
    cfg.replay.reuse_previous_stats = True

    result = run(cfg)
    refreshed = result["threshold_history"][1:]

    assert [entry["source"] for entry in refreshed] == ["replay", "replay"]
    assert refreshed[0]["calibration_classes"] == [0]
    assert refreshed[1]["calibration_classes"] == [0, 1]


def test_cl_control_hidden_sizes_override_experiment_model():
    cfg = make_toy_cfg()
    cfg.experiment.regime = "cl_ir"
    cfg.experiment.control_hidden_sizes = [8, 4]
    cfg.replay.sampling_mode = "paper"
    cfg.replay.per_class_batch_ratio = 0.5

    result = run(cfg)

    assert result["model"].hidden_sizes == [8, 4]
    assert result["trainer"].ae.hidden_sizes == [8, 4]
    assert result["trainer"].replay_mode == "paper"
    assert result["trainer"].replay_per_class_ratio == 0.5


def test_base_checkpoint_round_trip(tmp_path):
    checkpoint = tmp_path / "base.pt"
    first = make_toy_cfg()
    first.experiment.skip_incremental_training = True
    first.training.base_checkpoint_out = str(checkpoint)

    trained = run(first)
    assert checkpoint.is_file()

    loaded_cfg = make_toy_cfg()
    loaded_cfg.experiment.skip_incremental_training = True
    loaded_cfg.training.base_checkpoint = str(checkpoint)
    loaded = run(loaded_cfg)

    assert loaded["training_stats"]["pretrain_parameter_updates"] == 0
    for expected, actual in zip(
        trained["model"].state_dict().values(), loaded["model"].state_dict().values()
    ):
        torch.testing.assert_close(actual, expected)


def test_base_checkpoint_restores_rng_for_identical_incremental_run(tmp_path):
    checkpoint = tmp_path / "base_rng.pt"
    uninterrupted_cfg = make_toy_cfg()
    uninterrupted_cfg.training.base_checkpoint_out = str(checkpoint)
    uninterrupted = run(uninterrupted_cfg)

    resumed_cfg = make_toy_cfg()
    resumed_cfg.training.base_checkpoint = str(checkpoint)
    resumed = run(resumed_cfg)

    assert uninterrupted["model"].hidden_sizes == resumed["model"].hidden_sizes
    for expected, actual in zip(
        uninterrupted["model"].state_dict().values(), resumed["model"].state_dict().values()
    ):
        torch.testing.assert_close(actual, expected)


def test_incremental_checkpoint_resumes_at_completed_class_boundary(tmp_path):
    checkpoint = tmp_path / "incremental.pt"

    first_cfg = make_toy_cfg()
    first_cfg.neurogenesis.thresholds = [0.0, 0.0]
    first_cfg.training.incremental_checkpoint_out = str(checkpoint)
    first = run(first_cfg)
    assert checkpoint.is_file()

    resumed_cfg = make_toy_cfg()
    resumed_cfg.neurogenesis.thresholds = [0.0, 0.0]
    resumed_cfg.experiment.incremental_classes = [1, 2]
    resumed_cfg.training.incremental_checkpoint = str(checkpoint)
    resumed_cfg.training.incremental_checkpoint_out = str(checkpoint)
    resumed = run(resumed_cfg)

    uninterrupted_cfg = make_toy_cfg()
    uninterrupted_cfg.neurogenesis.thresholds = [0.0, 0.0]
    uninterrupted_cfg.experiment.incremental_classes = [1, 2]
    uninterrupted = run(uninterrupted_cfg)
    uninterrupted["model"]._plastic_to_mature()

    assert resumed["trainer"]._class_count == 2
    assert set(resumed["replay"].available_classes()) == {0, 1, 2}
    assert resumed["model"].hidden_sizes[0] >= first["model"].hidden_sizes[0]

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["learned"] == [0, 1, 2]
    assert payload["hidden_sizes"] == resumed["model"].hidden_sizes
    assert payload["eval_records"]
    assert payload["growth_reports"]
    assert payload["training_stats"]["neurogenesis_parameter_updates"] > 0
    assert resumed["model"].hidden_sizes == uninterrupted["model"].hidden_sizes
    for expected, actual in zip(
        uninterrupted["model"].state_dict().values(),
        resumed["model"].state_dict().values(),
    ):
        torch.testing.assert_close(actual, expected)


def test_incremental_checkpoint_rejects_non_prefix_stream(tmp_path):
    checkpoint = tmp_path / "incremental.pt"
    first_cfg = make_toy_cfg()
    first_cfg.training.incremental_checkpoint_out = str(checkpoint)
    run(first_cfg)

    bad_cfg = make_toy_cfg()
    bad_cfg.experiment.incremental_classes = [2]
    bad_cfg.training.incremental_checkpoint = str(checkpoint)
    with pytest.raises(ValueError, match="not a prefix"):
        run(bad_cfg)


def test_toy_run_logs_ir_quality_artifact(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient

    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    cfg = make_toy_cfg()
    cfg.experiment.skip_incremental_training = True
    cfg.replay.mode = "intrinsic"
    cfg.replay.ir_sampling_mode = "mean_only"
    cfg.logging.ir_quality_samples_per_class = 4
    cfg.logging.mlflow = {
        "enabled": True,
        "tracking_uri": tracking_uri,
        "experiment_name": "toy-ir-quality",
        "run_name": "toy-ir-quality",
        "metric_filter": None,
    }

    run(cfg)

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    exp = client.get_experiment_by_name("toy-ir-quality")
    runs = client.search_runs([exp.experiment_id], "tags.mlflow.runName = 'toy-ir-quality'")
    assert runs
    artifacts = client.list_artifacts(runs[0].info.run_id, "diagnostics")
    assert any(artifact.path == "diagnostics/ir_quality_step_1.json" for artifact in artifacts)
    artifact_path = client.download_artifacts(
        runs[0].info.run_id,
        "diagnostics/ir_quality_step_1.json",
        str(tmp_path / "artifacts"),
    )
    payload = json.loads(Path(artifact_path).read_text())
    cls_payload = payload["classes"]["0"]
    assert "roundtrip" in cls_payload
    assert "nearest_neighbor" in cls_payload
    assert "clean_feature_stats" in cls_payload


def _write_sd19_sample(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    arr = (rng.random((28, 28)) * 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def test_run_pipeline_with_sd19(tmp_path):
    root = tmp_path / "sd19"
    for cls in ("0", "1"):
        cls_dir = root / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            _write_sd19_sample(cls_dir / f"img_{i}.png", seed=10 * (int(cls) + 1) + i)

    cfg = make_toy_cfg()
    cfg.data.name = "sd19"
    cfg.data.data_dir = str(root)
    cfg.data.batch_size = 2
    cfg.experiment.dataset = "sd19"
    cfg.experiment.base_classes = [0]
    cfg.experiment.incremental_classes = [1]
    cfg.experiment.regime = "cl_ir"
    cfg.training.incremental_epochs = 1
    cfg.training.pretrain_epochs = 1
    cfg.replay.enabled = True

    result = run(cfg)
    replay = result["replay"]
    trainer = result["trainer"]
    assert trainer._class_count == 1
    assert replay is not None
    summary = replay.describe()
    assert set(summary.keys()) == {0, 1}
    for stats in summary.values():
        assert stats["count"] > 0
        assert stats["cov_condition"] >= 0

    # ensure SD-19 tensors are normalized and channel-first
    from data.sd19_datamodule import SD19DataModule

    dm = SD19DataModule(data_dir=str(root), batch_size=2, num_workers=0)
    dm.setup()
    subset = dm.get_class_dataset(0, use_val_transforms=False)
    x0, y0 = subset[0]
    assert x0.shape[0] == 1  # channel first
    assert float(x0.min()) >= 0.0 and float(x0.max()) <= 1.0
    assert y0 == 0


def test_run_pipeline_with_mnist_smoke(monkeypatch, tmp_path):
    from torchvision import datasets

    def _make_stub_dataset(train: bool):
        n = 12 if train else 6
        data = torch.rand(n, 28, 28)
        labels = torch.tensor([0, 1] * (n // 2))
        return (data.mul(255).to(torch.uint8), labels)

    class _StubMNIST:
        def __init__(self, data_dir: str, train: bool, download: bool, transform: Any):
            data, labels = _make_stub_dataset(train)
            self.data = data
            self.targets = labels

    monkeypatch.setattr(datasets, "MNIST", _StubMNIST)

    cfg = make_toy_cfg()
    cfg.data.name = "mnist"
    cfg.data.batch_size = 4
    cfg.data.data_dir = str(tmp_path / "mnist")
    cfg.experiment.dataset = "mnist"
    cfg.experiment.base_classes = [0]
    cfg.experiment.incremental_classes = [1]
    cfg.experiment.regime = "cl_ir"
    cfg.training.incremental_epochs = 1
    cfg.training.pretrain_epochs = 1
    cfg.replay.enabled = True

    result = run(cfg)
    replay = result["replay"]
    trainer = result["trainer"]
    assert trainer._class_count == 1
    assert replay is not None
    assert set(replay.available_classes()) == {0, 1}


def test_mlflow_logging(monkeypatch, tmp_path):
    import scripts.run_experiments as runner

    class _StubMLflow:
        def __init__(self):
            self.metrics = []
            self.params = []
            self.dicts = []
            self.texts = []
            self.artifacts = []
            self.images = []
            self.figures = []
            self.run_name = None

        def set_tracking_uri(self, uri):
            self.uri = uri

        def set_experiment(self, name):
            self.experiment = name

        def start_run(self, run_name=None):
            self.run_name = run_name
            return self

        def log_metric(self, key, value, step=None):
            self.metrics.append((key, value, step))

        def log_params(self, params):
            self.params.append(params)

        def log_dict(self, payload, artifact_file):
            self.dicts.append((artifact_file, payload))

        def log_text(self, text, artifact_file):
            self.texts.append((artifact_file, text))

        def log_artifact(self, path, artifact_path=None):
            self.artifacts.append((Path(path).name, artifact_path))

        def log_image(self, image_bytes, artifact_file):
            self.images.append((artifact_file, len(image_bytes)))

        def log_figure(self, fig, artifact_file):
            self.figures.append(artifact_file)

        def end_run(self):
            self.ended = True

    stub = _StubMLflow()
    monkeypatch.setattr(runner, "mlflow", stub)

    cfg = make_toy_cfg()
    cfg.logging.mlflow.enabled = True
    cfg.logging.mlflow.tracking_uri = str(tmp_path / "mlruns")
    cfg.logging.mlflow.experiment_name = "neurogenesis-tests"
    cfg.logging.mlflow.run_name = "integration"

    run(cfg)

    metric_keys = {key for key, _, _ in stub.metrics}
    assert any(key.startswith("metrics/val_mean_level") for key in metric_keys)
    assert any(key.startswith("replay/class_0") for key in metric_keys)
    assert any(key.startswith("global_level_0_size") for key in metric_keys)
    artifact_names = {name for name, _ in stub.artifacts}
    assert any(name.endswith("metrics.csv") for name in artifact_names)
    if runner.plot_recon_grid_mlflow is not None:
        assert any(name.startswith("figures/reconstructions_step") for name, _ in stub.images)
    else:
        assert not stub.images
    dict_artifacts = {name for name, _ in stub.dicts}
    assert "config_snapshot.json" in dict_artifacts
