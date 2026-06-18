from scripts.run_early_stop_ablation import RunSummary
from scripts.run_outlier_quota_ablation import (
    _filter_specs,
    _promotion_candidates,
    _quota_score,
    _stage1_specs,
    _stage2_specs,
    _stage3_specs,
    _stage4_specs,
    _stage5_specs,
    _to_full,
)


def _summary(name, family, mse, sizes):
    return RunSummary(
        stage="stage1_single0",
        name=name,
        description="",
        family=family,
        run_name=name,
        run_id=name,
        overrides=["replay.mode=dataset", "experiment.incremental_classes=[0]"],
        final_l3_mse=mse,
        layer_sizes=list(sizes),
        per_class_l3_mse={},
        diagnostics={"cap_hits_by_level": {"3": 0}},
        score=mse,
        status="completed",
    )


def test_stage1_grid_contains_fixed_adaptive_and_hybrid_quota_specs():
    specs = _stage1_specs()
    joined = {spec.name: " ".join(spec.overrides) for spec in specs}

    assert "l3_quota_010" in joined
    assert "l3_quota_030" in joined
    assert "shape_gate_04_p15" in joined
    assert "shape_gate_05_p20" in joined
    assert "hybrid_l3_quota_020_gate_05_p15" in joined
    assert "+neurogenesis.max_outlier_fraction_by_level.3=0.2" in joined[
        "l3_quota_020"
    ]
    assert "neurogenesis.shape_pressure_mode=scale_gate" in joined[
        "shape_gate_04_p20"
    ]
    assert all("experiment.incremental_classes=[0]" in item for item in joined.values())


def test_filter_specs_accepts_stage_prefix():
    specs = _stage1_specs()

    selected = _filter_specs(specs, "stage1")

    assert len(selected) == len(specs)


def test_full_conversion_drops_single0_and_adds_intrinsic_knobs():
    overrides = ("replay.mode=dataset", "experiment.incremental_classes=[0]")

    dataset = _to_full(overrides, replay_mode="dataset")
    intrinsic = _to_full(overrides, replay_mode="intrinsic")

    assert "experiment.incremental_classes=[0]" not in dataset
    assert "replay.mode=dataset" in dataset
    assert "experiment.incremental_classes=[0]" not in intrinsic
    assert "replay.mode=intrinsic" in intrinsic
    assert "replay.ir_sampling_mode=gaussian_shrink" in intrinsic


def test_quota_score_prefers_funnel_shape_over_slight_mse_gain():
    funnel = _summary("funnel", "fixed_quota", 0.0100, [240, 120, 90, 70])
    non_funnel = _summary("non_funnel", "fixed_quota", 0.0095, [240, 120, 90, 120])

    assert _quota_score(funnel) < _quota_score(non_funnel)


def test_promotion_candidates_pick_one_per_family():
    fixed = _summary("fixed", "fixed_quota", 0.010, [240, 120, 90, 70])
    adaptive = _summary("adaptive", "adaptive_quota", 0.011, [240, 120, 90, 80])
    hybrid = _summary("hybrid", "hybrid_quota", 0.012, [240, 120, 90, 85])

    promoted = _promotion_candidates([fixed, adaptive, hybrid], limit=3)

    assert [item.family for item in promoted] == [
        "fixed_quota",
        "adaptive_quota",
        "hybrid_quota",
    ]


def test_later_stage_specs_preserve_promoted_overrides():
    candidate = _summary("candidate", "fixed_quota", 0.01, [240, 120, 90, 70])
    candidate.overrides = [
        "replay.mode=dataset",
        "experiment.incremental_classes=[0]",
        "experiment.threshold.percentile=0.975",
    ]

    stage2 = _stage2_specs([candidate])
    stage3 = _stage3_specs(candidate)
    stage4 = _stage4_specs(candidate)
    stage5 = _stage5_specs(candidate)

    assert stage2[0].stage == "stage2_full_dataset"
    assert "experiment.incremental_classes=[0]" not in stage2[0].overrides
    assert any("experiment.threshold.percentile=0.995" in spec.overrides for spec in stage3)
    assert any("threshold_goal_factor_stability=0.3" in " ".join(spec.overrides) for spec in stage4)
    assert stage5[0].stage == "stage5_intrinsic"
    assert "replay.mode=intrinsic" in stage5[0].overrides
