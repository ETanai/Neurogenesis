import pytest
from omegaconf import OmegaConf

from scripts.run_experiments import (
    _band_distance,
    _compute_paper_fit_metrics,
    _extract_final_val_mean,
)


def test_band_distance_is_zero_inside_band() -> None:
    assert _band_distance(235.0, 220.0, 250.0, normalize="band_width") == 0.0


def test_band_distance_positive_outside_band() -> None:
    d = _band_distance(200.0, 220.0, 250.0, normalize="band_width")
    assert d > 0.0


def test_extract_final_val_mean_uses_latest_step() -> None:
    records = [
        {"step": 2, "layer": 3, "mean": 0.07},
        {"step": 3, "layer": 2, "mean": 0.04},
        {"step": 5, "layer": 3, "mean": 0.05},
    ]
    out = _extract_final_val_mean(records, layer_idx=3)
    assert out == 0.05


def test_compute_paper_fit_metrics_deterministic() -> None:
    cfg = OmegaConf.create(
        {
            "paper_fit": {
                "quality": {
                    "metric_layer": 3,
                    "normalize_min": 0.02,
                    "normalize_max": 0.08,
                    "clamp": True,
                },
                "growth": {
                    "target_bands": [[220, 250], [115, 145], [80, 95], [30, 45]],
                    "normalize": "band_width",
                    "clamp_per_level": 1.0,
                },
                "weights": {"quality": 0.6, "growth": 0.4},
            }
        }
    )
    metrics = _compute_paper_fit_metrics(
        cfg=cfg,
        final_val_mean=0.05,
        final_sizes=[235, 130, 90, 40],
    )
    assert metrics["summary/paper_fit_quality_term"] == pytest.approx(0.5)
    assert metrics["summary/growth_distance_total"] == pytest.approx(0.0)
    assert metrics["summary/paper_fit_score"] == pytest.approx(0.3)
