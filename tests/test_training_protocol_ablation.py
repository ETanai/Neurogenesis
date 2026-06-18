from scripts.run_training_protocol_ablation import (
    BASELINE_CLEAN_DATASET,
    _best_completed,
    _decoder_lr_specs,
    _full_specs,
    _ir_specs,
    _lr_length_specs,
    _quick_overrides,
    _replay_composition_specs,
    _stability_lr_specs,
    _threshold_growth_specs,
    _to_full_mnist_overrides,
    RunSummary,
)


def _summary(name, mse, overrides=None):
    return RunSummary(
        stage="stage1",
        name=name,
        description="",
        run_name=name,
        run_id=name,
        overrides=list(overrides or []),
        final_l3_mse=mse,
        layer_sizes=[],
        per_class_l3_mse={},
        status="completed",
    )


def test_stage1_lr_length_grid_matches_plan():
    specs = _lr_length_specs()

    assert len(specs) == 12
    joined = [" ".join(spec.overrides) for spec in specs]
    assert any("training.base_lr=0.0001" in item for item in joined)
    assert any("training.base_lr=0.001" in item for item in joined)
    assert any("neurogenesis.plasticity_epochs=1000" in item for item in joined)
    assert any("neurogenesis.stability_epochs=1000" in item for item in joined)
    assert all("experiment.incremental_classes=[0]" in item for item in joined)
    assert all("replay.mode=dataset" in item for item in joined)
    assert all("neurogenesis.next_layer_optimization=paper_columns" in item for item in joined)


def test_adaptive_stage1_specs_append_to_best_schedule():
    base = _lr_length_specs()[0].overrides

    stability = _stability_lr_specs(base)
    decoder = _decoder_lr_specs((*base, "neurogenesis.stability_lr_ratio=0.1"))

    assert len(stability) == 4
    assert "neurogenesis.stability_lr_ratio=1.0" in stability[-1].overrides
    assert len(decoder) == 3
    assert "neurogenesis.plasticity_decoder_lr_ratio=0.1" in decoder[-1].overrides
    assert "neurogenesis.stability_lr_ratio=0.1" in decoder[-1].overrides


def test_full_promotion_removes_single_digit_overrides():
    overrides = (
        "experiment.incremental_classes=[0]",
        "experiment.incremental_train_limit_per_class=16",
        "training.base_lr=0.001",
    )

    full = _to_full_mnist_overrides(overrides)

    assert "experiment.incremental_classes=[0]" not in full
    assert "experiment.incremental_train_limit_per_class=16" not in full
    assert "training.base_lr=0.001" in full


def test_full_specs_use_promoted_overrides():
    promoted = [_summary("candidate", 0.01, ["training.base_lr=0.001"])]

    specs = _full_specs(promoted)

    assert len(specs) == 1
    assert specs[0].stage == "stage2_full_dataset"
    assert "training.base_lr=0.001" in specs[0].overrides


def test_later_stage_specs_cover_planned_knobs():
    base = ("replay.mode=dataset", "training.base_lr=0.001")

    threshold_growth = " ".join(" ".join(spec.overrides) for spec in _threshold_growth_specs(base))
    replay = {spec.name for spec in _replay_composition_specs(base)}
    ir = " ".join(" ".join(spec.overrides) for spec in _ir_specs(base))

    assert "experiment.threshold.percentile=0.995" in threshold_growth
    assert "neurogenesis.factor_new_nodes=0.02" in threshold_growth
    assert "s4_replay_paper_0_5" in replay
    assert "s4_replay_balanced_4" in replay
    assert "replay.mode=intrinsic" in ir
    assert "replay.ir_sampling_mode=gaussian_shrink" in ir
    assert "replay.ir_noise_scale=0.75" in ir


def test_best_completed_sorts_by_l3_mse_and_ignores_missing():
    best = _best_completed(
        [
            _summary("missing", None),
            _summary("worse", BASELINE_CLEAN_DATASET),
            _summary("better", 0.01),
        ],
        limit=2,
    )

    assert [item.name for item in best] == ["better", "worse"]


def test_quick_overrides_make_runs_tiny():
    overrides = _quick_overrides()

    assert "training.pretrain_epochs=1" in overrides
    assert "experiment.incremental_train_limit_per_class=16" in overrides
    assert "neurogenesis.plasticity_epochs=2" in overrides
