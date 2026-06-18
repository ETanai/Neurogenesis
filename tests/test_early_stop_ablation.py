from scripts.run_early_stop_ablation import (
    _stage1_specs,
    _stage2_specs,
    _stage3_specs,
    _stage4_specs,
    _stage5_specs,
    _to_full,
    _with_intrinsic_single0,
    RunSummary,
)


def _summary(name, family, overrides):
    return RunSummary(
        stage="stage",
        name=name,
        description="",
        family=family,
        run_name=name,
        run_id=name,
        overrides=list(overrides),
        final_l3_mse=0.01,
        layer_sizes=[1, 1, 1, 1],
        per_class_l3_mse={},
        diagnostics={},
        score=0.01,
        status="completed",
    )


def test_stage1_global_grid_matches_protocol():
    specs = _stage1_specs()

    assert len(specs) == 5
    joined = [" ".join(spec.overrides) for spec in specs]
    assert any("neurogenesis.early_stop.min_delta=1e-5" in item for item in joined)
    assert any("neurogenesis.early_stop.patience=10" in item for item in joined)
    assert all("experiment.incremental_classes=[0]" in item for item in joined)
    assert all("replay.mode=dataset" in item for item in joined)


def test_stage2_goal_factor_grid_matches_protocol():
    specs = _stage2_specs(("training.base_lr=0.001",))
    joined = {spec.name: " ".join(spec.overrides) for spec in specs}

    assert set(joined) == {
        "current_goal",
        "stricter_stability",
        "stricter_both",
        "very_strict_stability",
        "no_goal_stop",
    }
    assert "threshold_goal_factor_stability=0.3" in joined["very_strict_stability"]
    assert "neurogenesis.early_stop.use_threshold_goal=false" in joined["no_goal_stop"]


def test_stage3_layer_specific_overrides_are_present():
    specs = _stage3_specs(("training.base_lr=0.001",))
    joined = {spec.name: " ".join(spec.overrides) for spec in specs}

    assert "neurogenesis.early_stop_by_level.0.min_delta=1e-4" in joined["layer_specific"]
    assert "neurogenesis.early_stop_by_level.3.patience=10" in joined["layer_specific"]
    assert "neurogenesis.early_stop_by_level.3.min_delta=1e-6" in joined["layer_specific_l3_strong"]


def test_intrinsic_single0_conversion_keeps_single_class_and_adds_ir_knobs():
    overrides = ("replay.mode=dataset", "experiment.incremental_classes=[0]")

    converted = _with_intrinsic_single0(overrides)

    assert "replay.mode=intrinsic" in converted
    assert "experiment.incremental_classes=[0]" in converted
    assert "replay.ir_sampling_mode=gaussian_shrink" in converted
    assert "replay.ir_cov_shrinkage=0.25" in converted


def test_full_conversion_drops_single_class_and_sets_replay_mode():
    overrides = ("replay.mode=dataset", "experiment.incremental_classes=[0]")

    dataset = _to_full(overrides, replay_mode="dataset")
    intrinsic = _to_full(overrides, replay_mode="intrinsic")

    assert "experiment.incremental_classes=[0]" not in dataset
    assert "replay.mode=dataset" in dataset
    assert "experiment.incremental_classes=[0]" not in intrinsic
    assert "replay.mode=intrinsic" in intrinsic
    assert "replay.ir_sampling_mode=gaussian_shrink" in intrinsic


def test_stage4_and_stage5_specs_promote_candidates():
    global_candidate = _summary("global", "global", ["replay.mode=dataset"])
    layer_candidate = _summary("layer", "phase_layer", ["replay.mode=dataset"])

    stage4 = _stage4_specs([global_candidate, layer_candidate])
    stage5 = _stage5_specs([global_candidate, layer_candidate])

    assert len(stage4) == 2
    assert all(spec.stage == "stage4_ir_single0" for spec in stage4)
    assert len(stage5) == 4
    assert any(spec.name.startswith("full_intrinsic") for spec in stage5)
    assert any(spec.name.startswith("full_dataset") for spec in stage5)
