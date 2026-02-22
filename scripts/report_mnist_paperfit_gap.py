"""Report MNIST paper-fit gap against historical anchor runs."""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from typing import Sequence


DB_PATH = "/workspace/Neurogenesis/mlflow.db"


@dataclass
class RunSummary:
    label: str
    run_name: str
    val_l0: float | None
    val_l3: float | None
    base_step_l3: float | None
    size_l0: float | None
    size_l1: float | None
    size_l2: float | None
    size_l3: float | None
    growth_total: float | None
    ratio_total: float | None
    ratio_l3: float | None
    ratio_l4: float | None
    paper_fit: float | None
    paper_fit_qfirst: float | None
    quality_floor_pass: float | None
    deep_gain_per_growth: float | None
    cap_hits_l3: float | None
    cap_hits_l4: float | None
    growth_early_total: float | None
    growth_late_total: float | None
    growth_decline_index: float | None
    deep_saturation_flag: bool
    comparable: bool


@dataclass
class AcceptanceTargets:
    val_l3_max: float = 0.033
    val_l0_max: float = 0.015
    ratio_max: float = 0.65
    growth_min: float = 600.0


def _metric(cur: sqlite3.Cursor, run_id: str, key: str) -> float | None:
    row = cur.execute(
        "select value from latest_metrics where run_uuid=? and key=?",
        (run_id, key),
    ).fetchone()
    return None if row is None else float(row[0])


def _base_step_l3(cur: sqlite3.Cursor, run_id: str) -> float | None:
    row = cur.execute(
        """
        select value
        from metrics
        where run_uuid=? and key='metrics/val_mean_level_3'
        order by step asc, timestamp asc
        limit 1
        """,
        (run_id,),
    ).fetchone()
    return None if row is None else float(row[0])


def _is_true(v: float | None) -> bool:
    return v is not None and float(v) >= 0.5


def _by_name(cur: sqlite3.Cursor, run_name: str, label: str) -> RunSummary | None:
    row = cur.execute("select run_uuid,name from runs where name=?", (run_name,)).fetchone()
    if row is None:
        return None
    run_id = row[0]
    val_l3 = _metric(cur, run_id, "metrics/val_mean_level_3")
    base_l3 = _base_step_l3(cur, run_id)
    growth_total = _metric(cur, run_id, "summary/growth_total_size_sum")
    deep_gain_per_growth = None
    if val_l3 is not None and base_l3 is not None and growth_total is not None:
        deep_gain_per_growth = (base_l3 - val_l3) / max(growth_total, 1.0)

    ratio_total = _metric(cur, run_id, "summary/growth_late_early_ratio_total")
    return RunSummary(
        label=label,
        run_name=row[1],
        val_l0=_metric(cur, run_id, "metrics/val_mean_level_0"),
        val_l3=val_l3,
        base_step_l3=base_l3,
        size_l0=_metric(cur, run_id, "global_level_0_size"),
        size_l1=_metric(cur, run_id, "global_level_1_size"),
        size_l2=_metric(cur, run_id, "global_level_2_size"),
        size_l3=_metric(cur, run_id, "global_level_3_size"),
        growth_total=growth_total,
        ratio_total=ratio_total,
        ratio_l3=_metric(cur, run_id, "summary/growth_late_early_ratio_l3"),
        ratio_l4=_metric(cur, run_id, "summary/growth_late_early_ratio_l4"),
        paper_fit=_metric(cur, run_id, "summary/paper_fit_score"),
        paper_fit_qfirst=_metric(cur, run_id, "summary/paper_fit_score_quality_first"),
        quality_floor_pass=_metric(cur, run_id, "summary/quality_floor_pass"),
        deep_gain_per_growth=deep_gain_per_growth,
        cap_hits_l3=_metric(cur, run_id, "summary/cap_hits/layer_2_cumulative"),
        cap_hits_l4=_metric(cur, run_id, "summary/cap_hits/layer_3_cumulative"),
        growth_early_total=_metric(cur, run_id, "summary/growth_early_total"),
        growth_late_total=_metric(cur, run_id, "summary/growth_late_total"),
        growth_decline_index=_metric(cur, run_id, "summary/growth_decline_index"),
        deep_saturation_flag=bool(
            (_metric(cur, run_id, "summary/cap_hits/layer_2_cumulative") or 0.0) >= 8
            or (_metric(cur, run_id, "summary/cap_hits/layer_3_cumulative") or 0.0) >= 8
        ),
        comparable=(growth_total is not None and ratio_total is not None),
    )


def _rank_candidates(candidates: Sequence[RunSummary]) -> list[RunSummary]:
    passed_floor = [c for c in candidates if _is_true(c.quality_floor_pass)]
    if passed_floor:
        return sorted(
            passed_floor,
            key=lambda x: (
                x.paper_fit_qfirst if x.paper_fit_qfirst is not None else float("inf"),
                x.val_l3 if x.val_l3 is not None else float("inf"),
                -(x.deep_gain_per_growth if x.deep_gain_per_growth is not None else float("-inf")),
                -(x.growth_decline_index if x.growth_decline_index is not None else float("-inf")),
            ),
        )
    return sorted(
        candidates,
        key=lambda x: (
            x.val_l3 if x.val_l3 is not None else float("inf"),
            x.val_l0 if x.val_l0 is not None else float("inf"),
        ),
    )


def _best_from_experiment(cur: sqlite3.Cursor, experiment_name: str, label: str) -> RunSummary | None:
    exp = cur.execute(
        "select experiment_id from experiments where name=?",
        (experiment_name,),
    ).fetchone()
    if exp is None:
        return None
    exp_id = int(exp[0])
    run_rows = cur.execute(
        """
        select name
        from runs
        where experiment_id=? and status='FINISHED' and name not like '%summary%'
        """,
        (exp_id,),
    ).fetchall()
    if not run_rows:
        return None
    candidates = [_by_name(cur, r[0], label) for r in run_rows]
    candidates = [c for c in candidates if c is not None]
    if not candidates:
        return None
    ranked = _rank_candidates(candidates)
    return ranked[0]


def _fmt(v: float | None) -> str:
    if v is None:
        return "NA"
    return f"{v:.6f}"


def _acceptance_flags(r: RunSummary, t: AcceptanceTargets) -> tuple[bool, bool, bool, bool]:
    pass_v3 = r.val_l3 is not None and r.val_l3 <= t.val_l3_max
    pass_v0 = r.val_l0 is not None and r.val_l0 <= t.val_l0_max
    pass_ratio = r.ratio_total is not None and r.ratio_total <= t.ratio_max
    pass_growth = r.growth_total is not None and r.growth_total >= t.growth_min
    return pass_v3, pass_v0, pass_ratio, pass_growth


def _recommendation(r: RunSummary, t: AcceptanceTargets) -> str:
    if not r.comparable or r.val_l0 is None or r.val_l3 is None:
        return "reject"
    pass_v3, pass_v0, pass_ratio, pass_growth = _acceptance_flags(r, t)
    if pass_v3 and pass_v0 and pass_ratio and pass_growth:
        return "promote"
    if not pass_growth:
        return "rerun with lower outlier fraction"
    return "reject"


def _layer_ratio_warnings(rows: Sequence[RunSummary]) -> list[str]:
    warns: list[str] = []
    for r in rows:
        if r.ratio_l3 is not None and r.ratio_l3 > 0.9:
            warns.append(
                f"{r.label}:{r.run_name} has high late/early ratio at L3 ({r.ratio_l3:.3f} > 0.9)."
            )
        if r.ratio_l4 is not None and r.ratio_l4 > 0.9:
            warns.append(
                f"{r.label}:{r.run_name} has high late/early ratio at L4 ({r.ratio_l4:.3f} > 0.9)."
            )
        if r.cap_hits_l3 is not None and r.cap_hits_l3 >= 8:
            warns.append(
                f"{r.label}:{r.run_name} shows deep saturation at L3 (cap hits={r.cap_hits_l3:.0f})."
            )
        if r.cap_hits_l4 is not None and r.cap_hits_l4 >= 8:
            warns.append(
                f"{r.label}:{r.run_name} shows deep saturation at L4 (cap hits={r.cap_hits_l4:.0f})."
            )
    return warns


def _promotion_candidates(rows: Sequence[RunSummary], t: AcceptanceTargets) -> list[RunSummary]:
    out: list[RunSummary] = []
    for r in rows:
        pass_v3, pass_v0, pass_ratio, pass_growth = _acceptance_flags(r, t)
        if r.comparable and pass_v3 and pass_v0 and pass_ratio and pass_growth:
            out.append(r)
    return _rank_candidates(out)


def _print_table(rows: Sequence[RunSummary], t: AcceptanceTargets) -> None:
    print(
        "| Label | Run | comparable | val_l0 | val_l3 | base_l3 | sizes [l0,l1,l2,l3] | "
        "growth_total | early_growth | late_growth | decline_index | late/early_total | deep_gain_per_growth | deep_saturation | paper_fit | paper_fit_qfirst | "
        "pass_v3 | pass_v0 | pass_ratio | pass_growth | recommendation |"
    )
    print("|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        sizes = f"[{_fmt(r.size_l0)},{_fmt(r.size_l1)},{_fmt(r.size_l2)},{_fmt(r.size_l3)}]"
        pass_v3, pass_v0, pass_ratio, pass_growth = _acceptance_flags(r, t)
        print(
            f"| {r.label} | {r.run_name} | {r.comparable} | {_fmt(r.val_l0)} | {_fmt(r.val_l3)} | "
            f"{_fmt(r.base_step_l3)} | {sizes} | {_fmt(r.growth_total)} | {_fmt(r.growth_early_total)} | {_fmt(r.growth_late_total)} | {_fmt(r.growth_decline_index)} | {_fmt(r.ratio_total)} | "
            f"{_fmt(r.deep_gain_per_growth)} | {r.deep_saturation_flag} | {_fmt(r.paper_fit)} | {_fmt(r.paper_fit_qfirst)} | "
            f"{pass_v3} | {pass_v0} | {pass_ratio} | {pass_growth} | {_recommendation(r, t)} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Report MNIST paper-fit gap.")
    parser.add_argument(
        "--experiment-name",
        default="mnist-paperfit-fidelity-quality",
        help="Prefix or full experiment name for new sweep (default: mnist-paperfit-fidelity-quality).",
    )
    args = parser.parse_args()
    targets = AcceptanceTargets()

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    exp_row = cur.execute(
        "select name from experiments where name like ? order by experiment_id desc limit 1",
        (f"{args.experiment_name}%",),
    ).fetchone()
    new_best = None
    if exp_row is not None:
        new_best = _best_from_experiment(cur, exp_row[0], "new_best")

    old_best = _by_name(
        cur,
        "baseline_opt_refined_high_growth_outlier_009_ratio_20_20260211_203513",
        "old_best_refined",
    )
    paperfit_bal_best = _by_name(
        cur,
        "paperfit_bal_p3_qrecovery_e75_manualtight_s07_20260214_142512",
        "paperfit_bal_best",
    )

    rows = [r for r in [new_best, old_best, paperfit_bal_best] if r is not None]
    if not rows:
        print("No matching runs found.")
        return

    _print_table(rows, targets)

    if new_best is not None:
        pass_v3, pass_v0, pass_ratio, pass_growth = _acceptance_flags(new_best, targets)
        print("\nAcceptance checks (new_best):")
        print(f"- val_mean_level_3 <= {targets.val_l3_max}: {pass_v3}")
        print(f"- val_mean_level_0 <= {targets.val_l0_max}: {pass_v0}")
        print(f"- growth_late_early_ratio_total <= {targets.ratio_max}: {pass_ratio}")
        print(f"- growth_total_size_sum >= {targets.growth_min}: {pass_growth}")
        print(f"- recommendation: {_recommendation(new_best, targets)}")

    promotions = _promotion_candidates(rows, targets)
    print("\nPromotion candidates:")
    if not promotions:
        print("- none")
    else:
        for item in promotions:
            print(
                f"- {item.label}:{item.run_name} (qfirst={_fmt(item.paper_fit_qfirst)}, "
                f"val_l3={_fmt(item.val_l3)}, deep_gain_per_growth={_fmt(item.deep_gain_per_growth)})"
            )

    warns = _layer_ratio_warnings(rows)
    if warns:
        print("\nTrajectory warnings:")
        for w in warns:
            print(f"- {w}")


if __name__ == "__main__":
    main()
