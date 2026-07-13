from pathlib import Path

import pytest
from omegaconf import OmegaConf

from scripts.run_paper_config import _compose_cfg, _validate_paper_run
from scripts.run_experiments import _neurogenesis_early_stop_cfg, _replay_refresh_plan


PAPER_DIR = Path(__file__).resolve().parents[1] / "config" / "paper"


def _resolved_runs(filename: str):
    data = OmegaConf.to_container(OmegaConf.load(PAPER_DIR / filename), resolve=True)
    for run in data["runs"]:
        yield run["name"], _compose_cfg(run["overrides"])


@pytest.mark.parametrize(
    "filename",
    ["figure4.yaml", "mnist_cl_ir.yaml", "mnist_ndl_ir.yaml", "sd19_ndl_ir.yaml"],
)
def test_paper_ir_runs_resolve_to_isolated_intrinsic_replay(filename):
    found_ir = False
    for run_name, cfg in _resolved_runs(filename):
        if str(cfg.experiment.regime).lower() not in {"cl_ir", "ndl_ir"}:
            continue
        found_ir = True
        _validate_paper_run(cfg, run_name=run_name)
        assert cfg.replay.mode == "intrinsic"
        assert cfg.replay.reuse_previous_stats is True
    assert found_ir


def test_dataset_replay_control_is_preserved():
    [(run_name, cfg)] = list(_resolved_runs("sd19_ndl.yaml"))

    _validate_paper_run(cfg, run_name=run_name)

    assert cfg.experiment.regime == "ndl"
    assert cfg.replay.enabled is True
    assert cfg.replay.mode == "dataset"


def test_figure4_controls_depend_on_corresponding_grown_networks():
    data = OmegaConf.to_container(OmegaConf.load(PAPER_DIR / "figure4.yaml"), resolve=True)
    runs = {run["name"]: run for run in data["runs"]}
    order = [run["name"] for run in data["runs"]]

    assert runs["cl"]["control_size_from"] == "ndl"
    assert runs["cl_ir"]["control_size_from"] == "ndl_ir"
    assert order.index("ndl") < order.index("cl")
    assert order.index("ndl_ir") < order.index("cl_ir")


def test_sd19_growth_config_requests_twenty_shuffled_curricula():
    data = OmegaConf.to_container(
        OmegaConf.load(PAPER_DIR / "sd19_growth_20.yaml"), resolve=True
    )

    assert data["repetitions"] == 20
    assert data["runs"][0]["shuffle_incremental_classes"] is True
    assert "replay.mode=dataset" in data["runs"][0]["overrides"]


def test_ir_validation_rejects_dataset_replay_label():
    cfg = _compose_cfg(
        [
            "data=mnist",
            "experiment=mnist_incremental",
            "experiment.regime=ndl_ir",
            "replay.enabled=true",
            "replay.mode=dataset",
        ]
    )

    with pytest.raises(ValueError, match="dataset-replay control"):
        _validate_paper_run(cfg, run_name="mislabeled_ir")


def test_ir_validation_rejects_old_data_refresh():
    cfg = _compose_cfg(
        [
            "data=mnist",
            "experiment=mnist_incremental",
            "experiment.regime=ndl_ir",
            "replay.enabled=true",
            "replay.mode=intrinsic",
            "replay.reuse_previous_stats=false",
        ]
    )

    with pytest.raises(ValueError, match="reuse_previous_stats=true"):
        _validate_paper_run(cfg, run_name="leaky_ir")


def test_intrinsic_refresh_never_reopens_old_class_data():
    classes, reset = _replay_refresh_plan(
        replay_mode="intrinsic",
        reuse_previous_stats=False,
        learned_so_far=[1, 7, 0],
        incoming_class=0,
    )

    assert classes == [0]
    assert reset is False


def test_dataset_refresh_keeps_clean_replay_upper_bound():
    classes, reset = _replay_refresh_plan(
        replay_mode="dataset",
        reuse_previous_stats=False,
        learned_so_far=[1, 7, 0],
        incoming_class=0,
    )

    assert classes == [1, 7, 0]
    assert reset is True


def test_paper_run_accepts_disabled_phase_early_stopping():
    cfg = _compose_cfg(
        [
            "data=mnist",
            "experiment=mnist_incremental",
            "neurogenesis.early_stop=null",
        ]
    )

    assert _neurogenesis_early_stop_cfg(cfg) == {}


@pytest.mark.parametrize(
    "filename",
    [
        "figure4.yaml",
        "mnist_ndl.yaml",
        "mnist_ndl_ir.yaml",
        "sd19_growth_20.yaml",
        "sd19_ndl.yaml",
        "sd19_ndl_ir.yaml",
    ],
)
def test_paper_ndl_runs_fail_on_exhausted_unresolved_outliers(filename):
    found_ndl = False
    for run_name, cfg in _resolved_runs(filename):
        if str(cfg.experiment.regime).lower() not in {"ndl", "ndl_ir"}:
            continue
        found_ndl = True
        _validate_paper_run(cfg, run_name=run_name)
        assert cfg.neurogenesis.unresolved_outlier_action == "error"
    assert found_ndl


def test_paper_validator_rejects_silent_unresolved_outliers():
    cfg = _compose_cfg(
        [
            "data=mnist",
            "experiment=mnist_incremental",
            "experiment.regime=ndl",
            "neurogenesis.unresolved_outlier_action=record",
        ]
    )

    with pytest.raises(ValueError, match="unresolved_outlier_action=error"):
        _validate_paper_run(cfg, run_name="silent_ndl")
