import pytest
from omegaconf import OmegaConf

from scripts.run_experiments import (
    _band_distance,
    _compute_paper_fit_metrics,
    _derive_auto_outlier_count,
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


def test_compute_paper_fit_metrics_with_growth_shape_penalty() -> None:
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
                "growth_shape": {
                    "enabled": True,
                    "late_early_ratio_target": 0.5,
                    "early_window": 2,
                    "late_window": 2,
                    "max_total_growth": 100.0,
                    "weight": 0.25,
                },
                "weights": {"quality": 0.6, "growth": 0.4},
                "quality_first": {
                    "quality_weight": 0.75,
                    "growth_weight": 0.15,
                    "growth_shape_weight": 0.10,
                    "quality_floor": 0.035,
                    "quality_floor_penalty": 0.2,
                },
            }
        }
    )
    # Late growth dominates and total growth exceeds budget -> positive shape penalty.
    growth_deltas = [[5.0, 0.0, 0.0, 0.0], [5.0, 0.0, 0.0, 0.0], [40.0, 0.0, 0.0, 0.0]]
    metrics = _compute_paper_fit_metrics(
        cfg=cfg,
        final_val_mean=0.05,
        final_sizes=[235, 130, 90, 40],
        growth_deltas=growth_deltas,
    )
    cfg_pass = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg_pass.paper_fit.quality_first.quality_floor = 0.06
    metrics_pass = _compute_paper_fit_metrics(
        cfg=cfg_pass,
        final_val_mean=0.05,
        final_sizes=[235, 130, 90, 40],
        growth_deltas=growth_deltas,
    )
    assert metrics["summary/growth_shape_term"] > 0.0
    assert metrics["summary/growth_late_early_ratio_total"] > 0.5
    assert metrics["summary/growth_late_early_ratio_l1"] > 0.5
    assert metrics["summary/growth_early_total"] > 0.0
    assert metrics["summary/growth_late_total"] > 0.0
    assert 0.0 <= metrics["summary/growth_decline_index"] <= 1.0
    assert metrics["summary/paper_fit_score_quality_first"] > metrics_pass[
        "summary/paper_fit_score_quality_first"
    ]
    assert metrics["summary/quality_floor_pass"] == 0.0


def test_derive_auto_outlier_count() -> None:
    assert _derive_auto_outlier_count(5421, 0.85) == 4608


def test_compute_paper_fit_metrics_logs_deep_gain_per_growth() -> None:
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
                "growth_shape": {"enabled": True, "early_window": 2, "late_window": 2, "weight": 0.1},
                "weights": {"quality": 0.6, "growth": 0.4},
            }
        }
    )
    growth_deltas = [[10.0, 0.0, 0.0, 0.0], [20.0, 0.0, 0.0, 0.0]]
    metrics = _compute_paper_fit_metrics(
        cfg=cfg,
        base_val_mean=0.09,
        final_val_mean=0.06,
        final_sizes=[235, 130, 90, 40],
        growth_deltas=growth_deltas,
    )
    assert metrics["summary/deep_gain_per_growth"] == pytest.approx((0.09 - 0.06) / 30.0)
