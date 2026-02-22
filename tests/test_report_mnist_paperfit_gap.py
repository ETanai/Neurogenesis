from scripts.report_mnist_paperfit_gap import (
    AcceptanceTargets,
    RunSummary,
    _acceptance_flags,
    _layer_ratio_warnings,
    _rank_candidates,
    _recommendation,
)


def _mk(
    *,
    val_l0: float | None,
    val_l3: float | None,
    growth_total: float | None,
    ratio_total: float | None,
) -> RunSummary:
    return RunSummary(
        label="x",
        run_name="r",
        val_l0=val_l0,
        val_l3=val_l3,
        base_step_l3=0.06 if val_l3 is not None else None,
        size_l0=None,
        size_l1=None,
        size_l2=None,
        size_l3=None,
        growth_total=growth_total,
        ratio_total=ratio_total,
        ratio_l3=ratio_total,
        ratio_l4=None,
        paper_fit=None,
        paper_fit_qfirst=0.5,
        quality_floor_pass=0.0,
        deep_gain_per_growth=0.0,
        cap_hits_l3=0.0,
        cap_hits_l4=0.0,
        growth_early_total=growth_total,
        growth_late_total=(growth_total * ratio_total if (growth_total is not None and ratio_total is not None) else None),
        growth_decline_index=(1.0 - min(1.0, ratio_total) if ratio_total is not None else None),
        deep_saturation_flag=False,
        comparable=(growth_total is not None and ratio_total is not None),
    )


def test_acceptance_flags_and_recommendation() -> None:
    t = AcceptanceTargets(val_l3_max=0.033, val_l0_max=0.015, ratio_max=0.65, growth_min=600.0)

    good = _mk(val_l0=0.014, val_l3=0.032, growth_total=800.0, ratio_total=0.5)
    assert _acceptance_flags(good, t) == (True, True, True, True)
    assert _recommendation(good, t) == "promote"

    low_growth = _mk(val_l0=0.014, val_l3=0.032, growth_total=120.0, ratio_total=0.5)
    assert _acceptance_flags(low_growth, t) == (True, True, True, False)
    assert _recommendation(low_growth, t) == "rerun with lower outlier fraction"

    bad_quality = _mk(val_l0=0.03, val_l3=0.05, growth_total=900.0, ratio_total=0.4)
    assert _acceptance_flags(bad_quality, t) == (False, False, True, True)
    assert _recommendation(bad_quality, t) == "reject"

    # Missing growth/ratio metrics -> historical/non-comparable behavior should reject.
    missing_metrics = _mk(val_l0=0.01, val_l3=0.03, growth_total=None, ratio_total=None)
    assert _recommendation(missing_metrics, t) == "reject"


def test_rank_candidates_prefers_quality_floor_then_qfirst() -> None:
    a = _mk(val_l0=0.02, val_l3=0.04, growth_total=900.0, ratio_total=0.5)
    a.run_name = "a"
    a.paper_fit_qfirst = 0.7
    a.quality_floor_pass = 1.0
    a.deep_gain_per_growth = 1e-5

    b = _mk(val_l0=0.018, val_l3=0.039, growth_total=920.0, ratio_total=0.5)
    b.run_name = "b"
    b.paper_fit_qfirst = 0.6
    b.quality_floor_pass = 1.0
    b.deep_gain_per_growth = 2e-5

    c = _mk(val_l0=0.01, val_l3=0.03, growth_total=None, ratio_total=None)
    c.run_name = "c"
    c.paper_fit_qfirst = 0.1
    c.quality_floor_pass = 0.0

    ranked = _rank_candidates([a, b, c])
    assert ranked[0].run_name == "b"
    assert ranked[1].run_name == "a"


def test_deep_layer_ratio_warning_trigger() -> None:
    row = _mk(val_l0=0.02, val_l3=0.04, growth_total=900.0, ratio_total=0.4)
    row.run_name = "warn"
    row.ratio_l3 = 0.95
    row.ratio_l4 = 0.91
    warns = _layer_ratio_warnings([row])
    assert len(warns) == 2


def test_deep_saturation_warning_trigger() -> None:
    row = _mk(val_l0=0.02, val_l3=0.04, growth_total=900.0, ratio_total=0.4)
    row.run_name = "sat"
    row.cap_hits_l3 = 9.0
    row.cap_hits_l4 = 11.0
    warns = _layer_ratio_warnings([row])
    assert len(warns) == 2
    assert "deep saturation at L3" in warns[0]
