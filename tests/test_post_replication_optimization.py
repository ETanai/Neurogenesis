from scripts.run_post_replication_optimization import ndl_specs, pc_specs, promotion_table


def test_pc_screen_contains_bp_limit_precision_and_global_candidates():
    specs = {spec.name: spec for spec in pc_specs(45)}
    assert "training.predictive_coding.update_mode=backprop_equivalent" in specs["pc_bp_equivalent"].overrides
    assert "training.base_lr=0.001" in specs["pc_bp_reference"].overrides
    assert any("layer_precisions=" in item for item in specs["pc_reconstruction_precision"].overrides)
    assert "training.predictive_coding.global_loss_weight=0.05" in specs["pc_global_005"].overrides
    assert "training.predictive_coding.consolidation_epochs=1" in specs["pc_consolidate_e1"].overrides
    assert all("model.activation=sigmoid" in spec.overrides for spec in specs.values())
    assert all("experiment.incremental_classes=[0]" in spec.overrides for spec in specs.values())


def test_ndl_screen_uses_identical_checkpoint_and_changes_only_coupling():
    specs = {spec.name: spec for spec in ndl_specs(45)}
    checkpoint_values = {
        next(item for item in spec.overrides if item.startswith("training.base_checkpoint="))
        for spec in specs.values()
    }
    assert len(checkpoint_values) == 1
    assert "neurogenesis.global_coupling.scope=all" in specs["ndl_coupling_e3_lr005"].overrides
    assert "neurogenesis.global_coupling.scope=decoder_only" in specs["ndl_decoder_coupling_e3"].overrides


def _row(family, name, seed, mse, forgetting):
    return {
        "stage": "screen", "family": family, "name": name, "seed": seed,
        "status": "completed", "macro_mse": mse,
        "mean_positive_forgetting": forgetting,
    }


def test_promotion_requires_ten_percent_gain_and_forgetting_constraint():
    rows = []
    for seed in (45, 46, 47):
        rows.extend(
            [
                _row("predictive_coding", "pc_local", seed, 0.10, 0.01),
                _row("predictive_coding", "good", seed, 0.089, 0.011),
                _row("predictive_coding", "too_forgetful", seed, 0.08, 0.013),
                _row("ndl", "ndl_baseline", seed, 0.10, 0.01),
            ]
        )
    gates = {row["name"]: row for row in promotion_table(rows)}
    assert gates["good"]["promoted"] is True
    assert gates["too_forgetful"]["promoted"] is False
