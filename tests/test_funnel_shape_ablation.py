from scripts.run_funnel_shape_ablation import (
    RunSummary,
    _funnel_score,
    _shape_metrics,
    _stage1_specs,
    _stage2_specs,
    _stage3_specs,
    _to_full,
)


def _summary(name, mse, sizes):
    return RunSummary(
        stage="stage",
        name=name,
        description="",
        family="family",
        run_name=name,
        run_id=name,
        overrides=["replay.mode=dataset", "experiment.incremental_classes=[0]"],
        final_l3_mse=mse,
        layer_sizes=list(sizes),
        per_class_l3_mse={},
        diagnostics={},
        score=mse,
        status="completed",
    )


def test_stage1_grid_contains_funnel_pressure_variants():
    specs = _stage1_specs()
    joined = {spec.name: " ".join(spec.overrides) for spec in specs}

    assert set(joined) == {
        "baseline_no_shape_pressure",
        "l3_scale_growth",
        "l3_scale_gate",
        "l3_scale_both",
        "l3_strong_gate",
        "all_layers_scale_both",
    }
    assert "neurogenesis.shape_pressure_mode=scale_growth" in joined["l3_scale_growth"]
    assert "neurogenesis.shape_pressure_mode=scale_gate" in joined["l3_scale_gate"]
    assert "neurogenesis.shape_pressure_mode=scale_both" in joined["l3_scale_both"]
    assert "+neurogenesis.shape_target_ratio_by_level.3=0.5" in joined["l3_scale_gate"]
    assert "+neurogenesis.shape_target_ratio_by_level.1=0.75" in joined["all_layers_scale_both"]
    assert all("experiment.incremental_classes=[0]" in item for item in joined.values())


def test_shape_metrics_detect_strict_funnel_and_l3_violation():
    good = _shape_metrics([225, 135, 83, 40])
    bad = _shape_metrics([208, 112, 79, 180])

    assert good["strict_funnel"] is True
    assert good["funnel_violations"] == 0
    assert good["l3_over_l2"] < 1
    assert bad["strict_funnel"] is False
    assert bad["funnel_violations"] == 1
    assert bad["l3_over_l2"] > 1


def test_funnel_score_penalizes_bad_shape_even_with_slightly_better_mse():
    funnel = _summary("funnel", 0.0100, [225, 135, 83, 40])
    non_funnel = _summary("non_funnel", 0.0090, [208, 112, 79, 180])

    assert _funnel_score(funnel) < _funnel_score(non_funnel)


def test_full_conversion_drops_single_class_and_adds_ir_knobs():
    overrides = ("replay.mode=dataset", "experiment.incremental_classes=[0]")

    dataset = _to_full(overrides, replay_mode="dataset")
    intrinsic = _to_full(overrides, replay_mode="intrinsic")

    assert "experiment.incremental_classes=[0]" not in dataset
    assert "replay.mode=dataset" in dataset
    assert "experiment.incremental_classes=[0]" not in intrinsic
    assert "replay.mode=intrinsic" in intrinsic
    assert "replay.ir_sampling_mode=gaussian_shrink" in intrinsic


def test_stage2_and_stage3_specs_preserve_promoted_overrides():
    candidate = _summary("candidate", 0.01, [10, 8, 6, 4])

    stage2 = _stage2_specs([candidate])
    stage3 = _stage3_specs([candidate])

    assert stage2[0].stage == "stage2_full_dataset"
    assert "experiment.incremental_classes=[0]" not in stage2[0].overrides
    assert "replay.mode=dataset" in stage2[0].overrides
    assert stage3[0].stage == "stage3_full_intrinsic"
    assert "replay.mode=intrinsic" in stage3[0].overrides
