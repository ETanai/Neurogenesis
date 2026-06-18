from scripts.run_global_coupling_ablation import (
    base_specs,
    full_specs,
    ir_specs,
    single0_specs,
)
from scripts.run_early_stop_ablation import RunSummary


def _summary(name, overrides, sizes=None):
    return RunSummary(
        stage="single0",
        name=name,
        description="",
        family="family",
        run_name=name,
        run_id=name,
        overrides=list(overrides),
        final_l3_mse=0.01,
        layer_sizes=list(sizes or [240, 120, 90, 40]),
        per_class_l3_mse={},
        diagnostics={},
        score=0.01,
        status="completed",
    )


def test_base_specs_include_no_finetune_and_base_finetune_checks():
    specs = base_specs()
    joined = {spec.name: " ".join(spec.overrides) for spec in specs}

    assert set(joined) == {
        "paper_local_no_finetune",
        "paper_local_base_finetune5",
        "paper_local_base_finetune10",
    }
    assert "training.pretrain_finetune_epochs=0" in joined["paper_local_no_finetune"]
    assert "training.pretrain_finetune_epochs=10" in joined["paper_local_base_finetune10"]
    assert all("experiment.skip_incremental_training=true" in item for item in joined.values())


def test_single0_specs_include_all_coupling_triggers_and_scopes():
    specs = single0_specs()
    joined = {spec.name: " ".join(spec.overrides) for spec in specs}

    assert "single0_no_coupling" in joined
    assert "single0_after_class_e5_decoder_only" in joined
    assert "single0_after_level_e5_freeze_old_encoder" in joined
    assert "single0_after_growth_round_e1_all" in joined
    assert "neurogenesis.global_coupling.trigger=after_class" in joined[
        "single0_after_class_e5_decoder_only"
    ]
    assert "neurogenesis.global_coupling.scope=decoder_only" in joined[
        "single0_after_class_e5_decoder_only"
    ]
    assert all("neurogenesis.objective_mode=paper_level_ae" in item for item in joined.values())


def test_full_and_ir_promotions_preserve_candidate_coupling_settings():
    candidate = _summary(
        "candidate",
        (
            "replay.mode=dataset",
            "experiment.incremental_classes=[0]",
            "neurogenesis.global_coupling.enabled=true",
            "neurogenesis.global_coupling.trigger=after_level",
            "neurogenesis.global_coupling.epochs=5",
        ),
    )

    full = full_specs([candidate])[0]
    ir = ir_specs([candidate])[0]

    assert "experiment.incremental_classes=[0]" not in full.overrides
    assert "experiment.incremental_classes=[0,2,3,4,5,6,8,9]" in full.overrides
    assert "replay.mode=dataset" in full.overrides
    assert "replay.mode=intrinsic" in ir.overrides
    assert "neurogenesis.global_coupling.trigger=after_level" in ir.overrides
