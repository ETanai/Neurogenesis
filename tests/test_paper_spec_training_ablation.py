import pytest

from scripts.run_early_stop_ablation import RunSummary
from scripts.run_paper_spec_training_ablation import (
    PAPER_LOCKED_OVERRIDES,
    baseline_specs,
    base_pretraining_specs,
    control_specs,
    full_promotion_specs,
    ir_replay_specs,
    single0_schedule_specs,
    threshold_growth_specs,
    _guard_paper_locked,
    _to_full,
)


def _summary(name, overrides, sizes=None):
    return RunSummary(
        stage="stage",
        name=name,
        description="",
        family="family",
        run_name=name,
        run_id=name,
        overrides=list(overrides),
        final_l3_mse=0.01,
        layer_sizes=list(sizes or [220, 130, 90, 50]),
        per_class_l3_mse={"1": 0.002, "7": 0.004},
        diagnostics={"cap_hits_by_level": {"3": 0}},
        score=0.01,
        status="completed",
    )


def test_baseline_specs_lock_paper_mechanisms_and_include_three_replay_conditions():
    specs = baseline_specs()
    joined = {spec.name: " ".join(spec.overrides) for spec in specs}

    assert set(joined) == {
        "literal_ndl_ir",
        "literal_ndl_dataset",
        "literal_ndl_no_replay",
    }
    assert "neurogenesis.objective_mode=paper_level_ae" in joined["literal_ndl_ir"]
    assert "neurogenesis.next_layer_optimization=paper_columns" in joined["literal_ndl_ir"]
    assert "replay.ir_sampling_mode=gaussian_full" in joined["literal_ndl_ir"]
    assert "training.pretrain_finetune_epochs=0" in joined["literal_ndl_ir"]
    assert "replay.mode=dataset" in joined["literal_ndl_dataset"]
    assert "replay.enabled=false" in joined["literal_ndl_no_replay"]


def test_base_pretraining_grid_has_expected_size_and_no_finetune():
    specs = base_pretraining_specs()
    joined = [" ".join(spec.overrides) for spec in specs]

    assert len(specs) == 4 * 5 * 4 * 2 * 3
    assert all("experiment.skip_incremental_training=true" in item for item in joined)
    assert all("training.pretrain_finetune_epochs=0" in item for item in joined)
    assert any("training.pretrain_epochs=200" in item for item in joined)
    assert any("training.denoising_std=0.05" in item for item in joined)


def test_single0_and_threshold_specs_are_generated_from_promoted_candidate():
    candidate = _summary("candidate", PAPER_LOCKED_OVERRIDES)

    single0 = single0_schedule_specs([candidate])
    threshold = threshold_growth_specs([_summary("s0", single0[0].overrides)])

    assert single0
    assert "experiment.incremental_classes=[0]" in single0[0].overrides
    assert any("neurogenesis.plasticity_epochs=2000" in spec.overrides for spec in single0)
    assert threshold
    assert any("experiment.threshold.percentile=0.995" in spec.overrides for spec in threshold)
    assert any("+neurogenesis.max_outlier_fraction_by_level.3=0.45" in spec.overrides for spec in threshold)
    assert all("training.pretrain_finetune_epochs=0" in spec.overrides for spec in threshold)


def test_ir_and_full_specs_keep_paper_gaussian_ir():
    candidate = _summary("candidate", PAPER_LOCKED_OVERRIDES)

    ir_specs = ir_replay_specs([candidate])
    full_specs = full_promotion_specs([candidate])

    assert any("neurogenesis.stability_replay_per_class_ratio=2.0" in spec.overrides for spec in ir_specs)
    assert any("replay.cov_eps=1e-3" in spec.overrides for spec in ir_specs)
    assert all("replay.ir_sampling_mode=gaussian_full" in spec.overrides for spec in ir_specs)
    assert any("replay.mode=dataset" in spec.overrides for spec in full_specs)
    assert any("replay.mode=intrinsic" in spec.overrides for spec in full_specs)


def test_full_conversion_restores_full_incremental_sequence():
    converted = _to_full(
        (
            *PAPER_LOCKED_OVERRIDES,
            "experiment.incremental_classes=[0]",
            "experiment.skip_incremental_training=true",
            "replay.mode=dataset",
        ),
        replay_mode="intrinsic",
    )

    assert "experiment.incremental_classes=[0]" not in converted
    assert "experiment.skip_incremental_training=true" not in converted
    assert "experiment.incremental_classes=[0,2,3,4,5,6,8,9]" in converted
    assert "replay.mode=intrinsic" in converted


def test_control_specs_use_candidate_sizes_and_cl_ir_regime():
    candidate = _summary("full_candidate", PAPER_LOCKED_OVERRIDES, sizes=[240, 120, 90, 60])

    specs = control_specs([candidate])
    joined = " ".join(specs[0].overrides)

    assert specs[0].family == "control"
    assert "experiment.regime=cl_ir" in joined
    assert "experiment.control_hidden_sizes=[240,120,90,60]" in joined
    assert "training.pretrain_finetune_epochs=0" in joined


def test_guard_rejects_finetune_and_nonpaper_modes():
    bad_finetune = baseline_specs()[0]
    bad_finetune = type(bad_finetune)(
        bad_finetune.stage,
        bad_finetune.name,
        bad_finetune.description,
        (*bad_finetune.overrides, "training.pretrain_finetune_epochs=10"),
        bad_finetune.family,
    )

    with pytest.raises(ValueError):
        _guard_paper_locked(bad_finetune)

    bad_objective = baseline_specs()[0]
    bad_objective = type(bad_objective)(
        bad_objective.stage,
        bad_objective.name,
        bad_objective.description,
        (*bad_objective.overrides, "neurogenesis.objective_mode=global_partial"),
        bad_objective.family,
    )

    with pytest.raises(ValueError):
        _guard_paper_locked(bad_objective)
