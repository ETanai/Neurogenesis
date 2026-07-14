import pytest

from scripts.run_organic_growth_ablation import (
    BASE_OVERRIDES,
    annotate_comparisons,
    confirmation_specs,
    full_specs,
    invariance_specs,
    screen_specs,
    summarize_result,
)


def test_screen_matrix_keeps_dataset_replay_and_disables_shape_pressure():
    specs = screen_specs()
    joined = {spec.name: " ".join(spec.overrides) for spec in specs}

    assert len(specs) == 7
    assert "replay.mode=dataset" in BASE_OVERRIDES
    assert "neurogenesis.shape_pressure_mode=none" in BASE_OVERRIDES
    assert all("experiment.incremental_classes=[0,2,3]" in value for value in joined.values())
    assert "neurogenesis.max_nodes_scope=global" in joined["cap_driven_reference"]
    assert "neurogenesis.max_nodes_stream=[50,70,16,40]" in joined[
        "organic_a_abs1_stream2x"
    ]
    assert "neurogenesis.max_nodes_per_class=[4,5,2,3]" in joined[
        "organic_c_abs1_class_small"
    ]
    assert "neurogenesis.max_nodes_stream=[100,140,32,80]" in joined[
        "organic_e_abs1_hybrid"
    ]


def test_invariance_matrix_doubles_only_relevant_safety_limit():
    specs = {spec.name: spec for spec in invariance_specs()}

    assert specs["organic_a_abs1_stream4x"].paired_to == "organic_a_abs1_stream2x"
    assert "neurogenesis.max_nodes_stream=[100,140,32,80]" in specs[
        "organic_a_abs1_stream4x"
    ].overrides
    assert "neurogenesis.max_nodes_per_class=[4,5,2,3]" in specs[
        "organic_e_abs1_hybrid_stream8x"
    ].overrides
    assert "neurogenesis.max_nodes_stream=[200,280,64,160]" in specs[
        "organic_e_abs1_hybrid_stream8x"
    ].overrides


def test_full_specs_restore_published_curriculum():
    assert all(
        "experiment.incremental_classes=[0,2,3,4,5,6,8,9]" in spec.overrides
        for spec in full_specs()
    )


def test_confirmation_matrix_labels_replay_provenance_and_no_replay_controls():
    specs = {spec.name: spec for spec in confirmation_specs()}

    assert set(specs) == {
        "ndl_dataset_oracle_refresh",
        "ndl_intrinsic_refresh",
        "ndl_no_replay_refresh",
        "cl_dataset_oracle_matched",
        "cl_intrinsic_matched",
        "cl_no_replay_matched",
    }
    assert "experiment.threshold.refresh_source=dataset" in specs[
        "ndl_dataset_oracle_refresh"
    ].overrides
    assert "experiment.threshold.refresh_source=replay" in specs[
        "ndl_intrinsic_refresh"
    ].overrides
    assert "replay.reuse_previous_stats=true" in specs[
        "ndl_intrinsic_refresh"
    ].overrides
    assert "experiment.regime=ndl" in specs["ndl_no_replay_refresh"].overrides
    assert "replay.enabled=false" in specs["ndl_no_replay_refresh"].overrides
    assert "experiment.regime=cl" in specs["cl_no_replay_matched"].overrides
    assert specs["cl_dataset_oracle_matched"].base_checkpoint_group != specs[
        "cl_intrinsic_matched"
    ].base_checkpoint_group


class _Model:
    hidden_sizes = [225, 130, 82, 38]

    @staticmethod
    def parameters():
        return iter(())


def test_summary_reports_organicity_and_performance_metrics():
    result = {
        "model": _Model(),
        "growth_reports": {
            0: {
                "model_update_steps": 10,
                "levels": [
                    {
                        "stop_reason": "outlier_quota_reached",
                        "unresolved_outliers": False,
                        "budget_exhausted": False,
                    }
                ],
            },
            2: {
                "model_update_steps": 4,
                "levels": [
                    {
                        "stop_reason": "stream_cap_exhausted",
                        "unresolved_outliers": True,
                        "budget_exhausted": True,
                    }
                ],
            },
        },
        "training_stats": {
            "neurogenesis_parameter_updates": 14,
            "incremental_parameter_updates": 0,
        },
        "eval_records": [
            {"step": 2, "scope": "class", "class_id": 1, "layer": 3, "mean": 0.02},
            {"step": 3, "scope": "aggregate", "class_id": "", "layer": 3, "mean": 0.03, "foreground_mse": 0.2},
            {"step": 3, "scope": "class", "class_id": 1, "layer": 3, "mean": 0.025, "foreground_mse": 0.1},
        ],
    }

    summary = summarize_result(result)

    assert summary["macro_mse"] == 0.03
    assert summary["foreground_mse"] == 0.2
    assert summary["strict_funnel"] is True
    assert summary["quota_stop_fraction"] == 0.5
    assert summary["updated_class_fraction"] == 1.0
    assert summary["unresolved_exhausted_level_count"] == 1
    assert summary["mean_positive_forgetting"] == pytest.approx(0.005)


def test_summary_uses_classical_incremental_update_counter():
    result = {
        "model": _Model(),
        "growth_reports": {},
        "training_stats": {
            "neurogenesis_parameter_updates": 0,
            "incremental_parameter_updates": 123,
        },
        "eval_records": [],
    }

    summary = summarize_result(result)

    assert summary["model_update_steps"] == 123
    assert summary["incremental_parameter_updates"] == 123


def test_comparison_annotation_applies_cap_invariance_and_organicity_gates():
    common = {
        "status": "completed",
        "seed": 42,
        "quota_stop_fraction": 1.0,
        "updated_class_fraction": 1.0,
        "mean_positive_forgetting": 0.01,
    }
    reference = {
        **common,
        "stage": "screen",
        "name": "cap_driven_reference",
        "family": "reference",
        "macro_mse": 0.10,
        "parameter_count": 100,
        "model_update_steps": 100,
        "added_widths": [10, 10, 5, 5],
    }
    candidate = {
        **common,
        "stage": "screen",
        "name": "organic_a_abs1_stream2x",
        "family": "loose_stream",
        "macro_mse": 0.102,
        "parameter_count": 80,
        "model_update_steps": 80,
        "added_widths": [10, 10, 5, 5],
    }
    doubled = {
        **common,
        "stage": "invariance",
        "name": "organic_a_abs1_stream4x",
        "family": "loose_stream",
        "paired_to": "organic_a_abs1_stream2x",
        "macro_mse": 0.103,
        "parameter_count": 80,
        "model_update_steps": 80,
        "added_widths": [10, 10, 5, 5],
    }
    rows = [reference, candidate, doubled]

    annotate_comparisons(rows)

    assert candidate["cap_invariance_pass"] is True
    assert doubled["cap_invariance_pass"] is True
    assert candidate["preliminary_organicity_gate"] is True
