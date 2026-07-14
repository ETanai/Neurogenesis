from scripts.benchmark_training_resources import condition_specs
from scripts.run_early_stop_ablation import _compose_cfg


def test_resource_benchmark_has_all_ndl_and_endpoint_matched_standard_conditions():
    specs = condition_specs()
    assert {name for name in specs if name.startswith("ndl_")} == {
        "ndl_original_data",
        "ndl_intrinsic_replay",
        "ndl_no_replay",
    }
    assert len(specs) == 7
    for endpoint in ("original_data", "intrinsic_replay", "no_replay"):
        ndl = specs[f"ndl_{endpoint}"]
        standard = specs[f"standard_end_to_end_{endpoint}_size"]
        assert standard["expected_widths"] == ndl["expected_widths"]
    assert specs["standard_stacked_original_data_size"]["expected_widths"] == specs[
        "ndl_original_data"
    ]["expected_widths"]


def test_standard_resource_conditions_train_jointly_without_replay():
    specs = condition_specs()
    for name, spec in specs.items():
        if spec["family"] != "standard_autoencoder":
            continue
        cfg = _compose_cfg(spec["overrides"])
        assert list(cfg.experiment.base_classes) == list(range(10))
        assert list(cfg.experiment.incremental_classes) == []
        assert cfg.experiment.regime == "cl"
        assert cfg.replay.enabled is False
        assert list(cfg.experiment.control_hidden_sizes) == spec["expected_widths"]


def test_ndl_resource_conditions_disable_cached_checkpoints():
    specs = condition_specs()
    for name, spec in specs.items():
        if spec["family"] != "neurogenesis":
            continue
        cfg = _compose_cfg(spec["overrides"])
        assert cfg.training.base_checkpoint is None
        assert cfg.training.base_checkpoint_out is None
