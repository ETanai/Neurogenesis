from scripts.run_sd19_comparison import SCREEN_CLASSES, evaluate_gate


def _rows(ndl_mse, cl_mse, ndl_forgetting, cl_forgetting):
    rows = []
    for index, seed in enumerate((42, 43, 44)):
        rows.extend(
            [
                {
                    "condition": "ndl_dataset",
                    "seed": seed,
                    "status": "completed",
                    "macro_mse": ndl_mse[index],
                    "mean_positive_forgetting": ndl_forgetting[index],
                },
                {
                    "condition": "cl_dataset",
                    "seed": seed,
                    "status": "completed",
                    "macro_mse": cl_mse[index],
                    "mean_positive_forgetting": cl_forgetting[index],
                },
            ]
        )
    return rows


def test_screen_classes_cover_upper_and_lower_case_boundaries():
    assert SCREEN_CLASSES == [10, 15, 22, 35, 36, 61]


def test_gate_passes_only_for_consistent_reconstruction_and_retention_advantage():
    gate = evaluate_gate(
        _rows(
            ndl_mse=[0.08, 0.09, 0.08],
            cl_mse=[0.10, 0.10, 0.10],
            ndl_forgetting=[0.01, 0.02, 0.01],
            cl_forgetting=[0.02, 0.02, 0.02],
        )
    )
    assert gate["passed"] is True
    assert gate["ndl_seed_win_count"] == 3


def test_gate_rejects_better_forgetting_without_reconstruction_advantage():
    gate = evaluate_gate(
        _rows(
            ndl_mse=[0.11, 0.12, 0.09],
            cl_mse=[0.10, 0.10, 0.10],
            ndl_forgetting=[0.0, 0.0, 0.0],
            cl_forgetting=[0.02, 0.02, 0.02],
        )
    )
    assert gate["passed"] is False
    assert gate["ndl_seed_win_count"] == 1


def test_gate_rejects_incomplete_pair_set():
    rows = _rows(
        ndl_mse=[0.08, 0.08, 0.08],
        cl_mse=[0.10, 0.10, 0.10],
        ndl_forgetting=[0.0, 0.0, 0.0],
        cl_forgetting=[0.01, 0.01, 0.01],
    )[:-1]
    assert evaluate_gate(rows)["passed"] is False
