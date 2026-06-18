from scripts.run_growth_shape_ablation import (
    _stage1_specs,
    _stage2_promotions,
    _stage2_specs,
    _stage3_specs,
    _to_full,
    RunSummary,
)


def _summary(name, family, mse, overrides):
    return RunSummary(
        stage="stage1_single0",
        name=name,
        description="",
        family=family,
        run_name=name,
        run_id=name,
        overrides=list(overrides),
        final_l3_mse=mse,
        layer_sizes=[1, 1, 1, 1],
        per_class_l3_mse={},
        diagnostics={},
        score=mse,
        status="completed",
    )


def test_stage1_grid_matches_growth_shape_protocol():
    specs = _stage1_specs()
    joined = {spec.name: " ".join(spec.overrides) for spec in specs}

    assert set(joined) == {
        "baseline_best_global",
        "paper_shape_caps",
        "l3_cap_20",
        "l3_cap_5",
        "small_global_growth_0002",
        "small_global_growth_0001",
        "absolute_global_1",
        "l3_absolute_1",
        "l3_small_factor",
    }
    assert "neurogenesis.max_nodes=[25,35,8,20]" in joined["paper_shape_caps"]
    assert "neurogenesis.factor_new_nodes=0.002" in joined["small_global_growth_0002"]
    assert "neurogenesis.growth_mode=absolute" in joined["absolute_global_1"]
    assert "+neurogenesis.growth_mode_by_level.3=absolute" in joined["l3_absolute_1"]
    assert "+neurogenesis.factor_new_nodes_by_level.3=0.001" in joined["l3_small_factor"]
    assert all("experiment.incremental_classes=[0]" in item for item in joined.values())
    assert all("replay.mode=dataset" in item for item in joined.values())


def test_stage2_promotes_one_candidate_per_growth_family():
    specs = {spec.name: spec for spec in _stage1_specs()}
    candidates = [
        _summary("paper_shape_caps", "cap", 0.03, specs["paper_shape_caps"].overrides),
        _summary("l3_cap_20", "cap", 0.02, specs["l3_cap_20"].overrides),
        _summary(
            "small_global_growth_0002",
            "small_proportional",
            0.01,
            specs["small_global_growth_0002"].overrides,
        ),
        _summary("l3_absolute_1", "absolute_l3", 0.04, specs["l3_absolute_1"].overrides),
    ]

    promoted = _stage2_promotions(candidates)

    assert [item.name for item in promoted] == [
        "l3_cap_20",
        "small_global_growth_0002",
        "l3_absolute_1",
    ]


def test_full_conversion_drops_single0_and_adds_intrinsic_knobs():
    overrides = ("replay.mode=dataset", "experiment.incremental_classes=[0]")

    dataset = _to_full(overrides, replay_mode="dataset")
    intrinsic = _to_full(overrides, replay_mode="intrinsic")

    assert "experiment.incremental_classes=[0]" not in dataset
    assert "replay.mode=dataset" in dataset
    assert "experiment.incremental_classes=[0]" not in intrinsic
    assert "replay.mode=intrinsic" in intrinsic
    assert "replay.ir_sampling_mode=gaussian_shrink" in intrinsic
    assert "replay.ir_cov_shrinkage=0.25" in intrinsic


def test_stage2_and_stage3_specs_use_promoted_overrides():
    base = _summary(
        "candidate",
        "cap",
        0.01,
        ("replay.mode=dataset", "experiment.incremental_classes=[0]"),
    )

    stage2 = _stage2_specs([base])
    stage3 = _stage3_specs([base])

    assert stage2[0].stage == "stage2_full_dataset"
    assert stage2[0].name.startswith("full_dataset")
    assert "experiment.incremental_classes=[0]" not in stage2[0].overrides
    assert stage3[0].stage == "stage3_full_intrinsic"
    assert "replay.mode=intrinsic" in stage3[0].overrides
