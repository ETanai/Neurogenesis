import torch
from argparse import Namespace
from models.ng_autoencoder import NGAutoEncoder
from scripts.run_experiments import probe_growth_wiring
from scripts.run_diagnostic_protocol import (
    _base_runs,
    _capacity_run,
    _default_tracking_uri,
    _extra_overrides,
    _filter_runs,
    _microscope_runs,
    _single_digit_runs,
    _sweep_runs,
)


def test_diagnostic_protocol_specs_cover_core_runs():
    names = {spec.name for spec in [*_base_runs(), *_single_digit_runs()]}

    assert "base_manual_thresholds" in names
    assert "base_estimated_thresholds" in names
    assert "single_0_manual_thresholds" in names
    assert "single_0_estimated_thresholds" in names


def test_capacity_run_uses_derived_hidden_sizes():
    spec = _capacity_run([10, 8, 6, 4])

    assert spec.name == "capacity_single_0_cl_dataset_replay"
    assert "experiment.regime=cl_ir" in spec.overrides
    assert "experiment.model.hidden_sizes=[10,8,6,4]" in spec.overrides


def test_sweep_runs_include_key_knobs():
    overrides = [" ".join(spec.overrides) for spec in _sweep_runs()]

    assert any("experiment.threshold.percentile=0.995" in item for item in overrides)
    assert any("neurogenesis.factor_new_nodes=0.001" in item for item in overrides)
    assert any("training.base_lr=0.001" in item for item in overrides)
    assert _default_tracking_uri().startswith("sqlite:///")


def test_microscope_runs_include_growth_wiring_and_tiny_overfits():
    specs = _microscope_runs()
    names = {spec.name for spec in specs}

    assert "growth_wiring" in names
    assert "replay_balance_recheck" in names
    assert "tiny_overfit_32_ndl_paper_local" in names
    assert "tiny_overfit_32_ndl_full_reconstruction" in names
    assert "tiny_overfit_128_cl_dataset_replay" in names

    growth = next(spec for spec in specs if spec.name == "growth_wiring")
    assert "neurogenesis.growth_wiring_probe=true" in growth.overrides
    assert "experiment.skip_incremental_training=true" in growth.overrides


def test_filter_runs_matches_exact_and_prefix():
    specs = _microscope_runs()

    assert [spec.name for spec in _filter_runs(specs, "growth_wiring")] == ["growth_wiring"]
    selected = _filter_runs(specs, "tiny_overfit_32")
    assert len(selected) == 4
    assert all(spec.name.startswith("tiny_overfit_32") for spec in selected)


def test_extra_overrides_append_manual_overrides_after_quick():
    overrides = _extra_overrides(
        Namespace(
            quick=True,
            override=[
                "neurogenesis.max_nodes=[4,4,4,2]",
                "training.pretrain_epochs=2",
            ],
        )
    )

    assert "neurogenesis.max_nodes=[2,2,2,2]" in overrides
    assert overrides[-2:] == (
        "neurogenesis.max_nodes=[4,4,4,2]",
        "training.pretrain_epochs=2",
    )


def test_growth_wiring_probe_detects_expected_gradients_and_freeze():
    model = NGAutoEncoder(
        input_dim=4,
        hidden_sizes=[3, 2],
        activation="identity",
        activation_latent="identity",
        activation_last="identity",
    )
    x = torch.randn(5, 4)

    records = probe_growth_wiring(model, x, num_new=1, lr=1.0e-2, device=torch.device("cpu"))

    current_local = next(
        rec
        for rec in records
        if rec["level"] == 0
        and rec["objective"] == "local"
        and rec["group"] == "current_encoder_plastic_rows"
    )
    mirror_local = next(
        rec
        for rec in records
        if rec["level"] == 0
        and rec["objective"] == "local"
        and rec["group"] == "mirror_decoder_new_input_columns"
    )
    next_full = next(
        rec
        for rec in records
        if rec["level"] == 0
        and rec["objective"] == "full"
        and rec["group"] == "next_encoder_new_input_columns"
    )
    old_encoder = next(
        rec
        for rec in records
        if rec["level"] == 0
        and rec["objective"] == "local"
        and rec["group"] == "current_encoder_mature"
    )

    assert current_local["connectivity_grad_l2"] > 0
    assert current_local["plasticity_delta_l2"] > 0
    assert mirror_local["connectivity_grad_l2"] > 0
    assert mirror_local["plasticity_delta_l2"] > 0
    assert next_full["connectivity_grad_l2"] > 0
    assert next_full["plasticity_delta_l2"] == 0
    assert old_encoder["plasticity_delta_l2"] == 0
