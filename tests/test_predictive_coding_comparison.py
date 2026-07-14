import json

from scripts.run_predictive_coding_comparison import (
    INCREMENTAL_ORDER,
    PC_CONDITIONS,
    condition_spec,
    write_aggregate,
)


def test_conditions_reuse_exact_fixed_control_checkpoints():
    dataset = condition_spec("pc_uniform_original_data", 42)
    no_replay = condition_spec("pc_usage_no_replay", 42)

    assert dataset["widths"] == [207, 105, 77, 23]
    assert no_replay["widths"] == [207, 105, 78, 23]
    assert dataset["checkpoint"].name == "cl_207_sigmoid_seed_42.pt"
    assert no_replay["checkpoint"].name == "seed_42.pt"
    assert "experiment.regime=cl_ir" in dataset["overrides"]
    assert "replay.enabled=true" in dataset["overrides"]
    assert "experiment.regime=cl" in no_replay["overrides"]
    assert "replay.enabled=false" in no_replay["overrides"]


def test_usage_and_uniform_conditions_differ_only_in_plasticity_setting():
    uniform = condition_spec("pc_uniform_original_data", 43)
    usage = condition_spec("pc_usage_original_data", 43)
    filtered_uniform = [
        value for value in uniform["overrides"]
        if not value.startswith("training.predictive_coding.plasticity_mode=")
    ]
    filtered_usage = [
        value for value in usage["overrides"]
        if not value.startswith("training.predictive_coding.plasticity_mode=")
    ]
    assert filtered_uniform == filtered_usage


def test_aggregate_combines_existing_bp_controls_with_pc_rows(tmp_path):
    per_class = {str(class_id): 0.1 for class_id in [1, 7, *INCREMENTAL_ORDER]}
    rows = [
        {
            "condition": condition,
            "seed": 42,
            "status": "completed",
            "macro_mse": 0.1,
            "foreground_mse": 0.2,
            "mean_positive_forgetting": 0.01,
            "per_class_mse": per_class,
        }
        for condition in PC_CONDITIONS
    ]

    aggregate = write_aggregate(rows, tmp_path, [42])

    assert len(aggregate) == 6
    assert all(item["seed_count"] == 1 for item in aggregate)
    assert (tmp_path / "aggregate.csv").is_file()
    combined = json.loads((tmp_path / "comparison_rows.json").read_text(encoding="utf-8"))
    assert len(combined) == 6
