import json

import pytest

from scripts.summarize_confirmation import _estimate, aggregate


def test_estimate_reports_student_t_interval():
    estimate = _estimate([1.0, 2.0, 3.0, 4.0, 5.0])

    assert estimate["mean"] == 3.0
    assert estimate["std"] == pytest.approx(1.58113883)
    assert estimate["ci95_low"] == pytest.approx(1.03707, abs=1e-4)
    assert estimate["ci95_high"] == pytest.approx(4.96293, abs=1e-4)


def test_aggregate_merges_manifests_and_rejects_duplicate_seeds(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(
        json.dumps(
            [
                {"status": "completed", "seed": 42, "macro_mse": 0.1, "final_widths": [2, 1]},
                {"status": "failed", "seed": 43, "macro_mse": 9.0, "final_widths": [9, 9]},
            ]
        ),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(
            [{"status": "completed", "seed": 43, "macro_mse": 0.2, "final_widths": [3, 1]}]
        ),
        encoding="utf-8",
    )

    result = aggregate([("condition", first, None), ("condition", second, None)])

    assert result[0]["seeds"] == [42, 43]
    assert result[0]["metrics"]["macro_mse"]["mean"] == pytest.approx(0.15)
    assert result[0]["final_widths_observed"] == [[2, 1], [3, 1]]

    with pytest.raises(ValueError, match="Duplicate seed 42"):
        aggregate([("condition", first, None), ("condition", first, None)])


def test_aggregate_can_filter_a_shared_manifest_by_run_name(tmp_path):
    manifest = tmp_path / "shared.json"
    manifest.write_text(
        json.dumps(
            [
                {"status": "completed", "name": "wanted", "seed": 42, "macro_mse": 0.1},
                {"status": "completed", "name": "other", "seed": 42, "macro_mse": 9.0},
            ]
        ),
        encoding="utf-8",
    )

    result = aggregate([("condition", manifest, "wanted")])

    assert result[0]["seed_count"] == 1
    assert result[0]["metrics"]["macro_mse"]["mean"] == 0.1
