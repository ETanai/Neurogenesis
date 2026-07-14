import csv
import json

import scripts.export_replication_figure_data as exporter


def test_export_contains_all_figure_mappings(tmp_path, monkeypatch):
    monkeypatch.setattr(exporter, "OUTPUT", tmp_path)
    exporter.main()

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["figures"]) == {
        "performance_comparison.png",
        "per_class_comparison.png",
        "mechanism_comparison.png",
        "claim_direction_comparison.png",
        "cross_dataset_effects.png",
        "sd19_feasibility_comparison.png",
        "training_resource_comparison.png",
        "predictive_coding_comparison.png",
        "post_replication_optimization.png",
    }
    for names in manifest["figures"].values():
        assert all((tmp_path / name).is_file() for name in names)


def test_export_preserves_seed_level_observations(tmp_path, monkeypatch):
    monkeypatch.setattr(exporter, "OUTPUT", tmp_path)
    exporter.main()

    with (tmp_path / "mnist_seed_metrics.csv").open(encoding="utf-8", newline="") as handle:
        mnist = list(csv.DictReader(handle))
    with (tmp_path / "mnist_per_class_seed.csv").open(encoding="utf-8", newline="") as handle:
        per_class = list(csv.DictReader(handle))
    assert len(mnist) == 6 * 10
    assert len(per_class) == 6 * 10 * 10
    assert {int(row["seed"]) for row in mnist} == set(range(42, 52))


def test_exported_resource_ratios_match_base_values(tmp_path, monkeypatch):
    monkeypatch.setattr(exporter, "OUTPUT", tmp_path)
    exporter.main()

    with (tmp_path / "training_resource_ratios.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    intrinsic_time = next(
        row
        for row in rows
        if row["replay_regime"] == "intrinsic_replay" and row["metric"] == "wall_seconds"
    )
    assert float(intrinsic_time["ndl_over_standard"]) > 6.8
