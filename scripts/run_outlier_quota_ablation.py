"""Run staged outlier-quota ablations for paper-like NDL funnel shape."""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as _dt
import json
import math
import sys
from pathlib import Path
from typing import Iterable, Sequence

from omegaconf import open_dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_early_stop_ablation import (  # noqa: E402
    AblationSpec,
    RunSummary,
    _compose_cfg,
    _default_tracking_uri,
    _extract_mlflow_summary,
    _fmt,
    _query_run_id,
)
from scripts.run_experiments import run  # noqa: E402


REFERENCE_NO_PRESSURE_MSE = 0.009343640878796577
REFERENCE_NO_PRESSURE_SIZES = [240, 116, 90, 146]
REFERENCE_STRONG_GATE_MSE = 0.009741042740643024
REFERENCE_STRONG_GATE_SIZES = [240, 116, 90, 87]
REFERENCE_CAPPED_STRONG_GATE_MSE = 0.010744967497885227
REFERENCE_CAPPED_STRONG_GATE_SIZES = [240, 120, 90, 105]
REFERENCE_PAPER_AE_MSE = 0.008744465455412865
REFERENCE_PAPER_AE_SIZES = [225, 135, 83, 40]

BASE_OVERRIDES = (
    "data=mnist",
    "experiment=mnist_incremental",
    "experiment.regime=ndl_ir",
    "experiment.base_classes=[1,7]",
    "replay.enabled=true",
    "replay.mode=dataset",
    "neurogenesis.thresholds=null",
    "neurogenesis.next_layer_optimization=paper_columns",
    "training.base_lr=0.001",
    "training.pretrain_epochs=50",
    "training.pretrain_finetune_epochs=10",
    "training.pretrain_mode=stacked_denoising",
    "neurogenesis.stability_lr_ratio=1.0",
    "experiment.threshold.percentile=0.975",
    "neurogenesis.stability_replay_mode=paper",
    "neurogenesis.stability_replay_per_class_ratio=1.0",
    "neurogenesis.early_stop.use_threshold_goal=true",
    "neurogenesis.early_stop.threshold_goal_factor_plasticity=0.8",
    "neurogenesis.early_stop.threshold_goal_factor_stability=0.5",
    "neurogenesis.early_stop.min_delta=3e-6",
    "neurogenesis.early_stop.patience=5",
)

SINGLE_0_OVERRIDES = (
    *BASE_OVERRIDES,
    "experiment.incremental_classes=[0]",
)


def _level_key(path: str, level: int, value: float | int) -> str:
    return f"+neurogenesis.{path}.{level}={value}"


def _stage1_specs() -> list[AblationSpec]:
    fixed = [
        ("l3_quota_010", 0.10),
        ("l3_quota_015", 0.15),
        ("l3_quota_020", 0.20),
        ("l3_quota_025", 0.25),
        ("l3_quota_030", 0.30),
    ]
    specs = [
        AblationSpec(
            "stage1_single0",
            name,
            f"Single-0 clean replay with L3 accepted outlier fraction {quota}.",
            (
                *SINGLE_0_OVERRIDES,
                _level_key("max_outlier_fraction_by_level", 3, quota),
            ),
            "fixed_quota",
        )
        for name, quota in fixed
    ]
    specs.extend(
        [
            AblationSpec(
                "stage1_single0",
                "shape_gate_04_p15",
                "Single-0 clean replay with adaptive L3 quota: target 0.4, power 1.5.",
                (
                    *SINGLE_0_OVERRIDES,
                    "neurogenesis.shape_pressure_mode=scale_gate",
                    _level_key("shape_target_ratio_by_level", 3, 0.4),
                    "neurogenesis.shape_gate_power=1.5",
                ),
                "adaptive_quota",
            ),
            AblationSpec(
                "stage1_single0",
                "shape_gate_04_p20",
                "Single-0 clean replay with adaptive L3 quota: target 0.4, power 2.0.",
                (
                    *SINGLE_0_OVERRIDES,
                    "neurogenesis.shape_pressure_mode=scale_gate",
                    _level_key("shape_target_ratio_by_level", 3, 0.4),
                    "neurogenesis.shape_gate_power=2.0",
                ),
                "adaptive_quota",
            ),
            AblationSpec(
                "stage1_single0",
                "shape_gate_05_p20",
                "Single-0 clean replay with adaptive L3 quota: target 0.5, power 2.0.",
                (
                    *SINGLE_0_OVERRIDES,
                    "neurogenesis.shape_pressure_mode=scale_gate",
                    _level_key("shape_target_ratio_by_level", 3, 0.5),
                    "neurogenesis.shape_gate_power=2.0",
                ),
                "adaptive_quota",
            ),
            AblationSpec(
                "stage1_single0",
                "hybrid_l3_quota_020_gate_05_p15",
                "Single-0 clean replay with fixed L3 quota 0.20 plus mild shape gate.",
                (
                    *SINGLE_0_OVERRIDES,
                    _level_key("max_outlier_fraction_by_level", 3, 0.20),
                    "neurogenesis.shape_pressure_mode=scale_gate",
                    _level_key("shape_target_ratio_by_level", 3, 0.5),
                    "neurogenesis.shape_gate_power=1.5",
                ),
                "hybrid_quota",
            ),
        ]
    )
    return specs


def _to_full(overrides: Sequence[str], *, replay_mode: str) -> tuple[str, ...]:
    converted: list[str] = []
    for item in overrides:
        if item.startswith("experiment.incremental_classes="):
            continue
        if item == "replay.mode=dataset":
            converted.append(f"replay.mode={replay_mode}")
        else:
            converted.append(item)
    if replay_mode == "intrinsic":
        converted.extend(
            (
                "replay.ir_sampling_mode=gaussian_shrink",
                "replay.ir_cov_shrinkage=0.25",
            )
        )
    return tuple(converted)


def _stage2_specs(promoted: Sequence[RunSummary]) -> list[AblationSpec]:
    return [
        AblationSpec(
            "stage2_full_dataset",
            f"full_dataset_{idx}_{summary.name}",
            f"Full MNIST dataset replay promotion from {summary.name}.",
            _to_full(summary.overrides, replay_mode="dataset"),
            summary.family,
        )
        for idx, summary in enumerate(promoted, start=1)
    ]


def _stage3_specs(best: RunSummary) -> list[AblationSpec]:
    percentiles = [0.95, 0.975, 0.985, 0.995]
    base = tuple(
        item
        for item in _to_full(best.overrides, replay_mode="dataset")
        if not item.startswith("experiment.threshold.percentile=")
    )
    return [
        AblationSpec(
            "stage3_threshold",
            f"threshold_p{str(percentile).replace('.', '_')}",
            f"Full MNIST threshold percentile interaction at {percentile}.",
            (*base, f"experiment.threshold.percentile={percentile}"),
            best.family,
        )
        for percentile in percentiles
    ]


def _stage4_specs(best: RunSummary) -> list[AblationSpec]:
    base = tuple(_to_full(best.overrides, replay_mode="dataset"))
    stripped = tuple(
        item
        for item in base
        if not item.startswith("neurogenesis.early_stop.use_threshold_goal=")
        and not item.startswith("neurogenesis.early_stop.threshold_goal_factor_plasticity=")
        and not item.startswith("neurogenesis.early_stop.threshold_goal_factor_stability=")
    )
    return [
        AblationSpec(
            "stage4_goal",
            "strict_both",
            "Full MNIST: strict plasticity and stability threshold goals.",
            (
                *stripped,
                "neurogenesis.early_stop.use_threshold_goal=true",
                "neurogenesis.early_stop.threshold_goal_factor_plasticity=0.8",
                "neurogenesis.early_stop.threshold_goal_factor_stability=0.5",
            ),
            best.family,
        ),
        AblationSpec(
            "stage4_goal",
            "strict_stability",
            "Full MNIST: normal plasticity, strict stability threshold goal.",
            (
                *stripped,
                "neurogenesis.early_stop.use_threshold_goal=true",
                "neurogenesis.early_stop.threshold_goal_factor_plasticity=0.9",
                "neurogenesis.early_stop.threshold_goal_factor_stability=0.5",
            ),
            best.family,
        ),
        AblationSpec(
            "stage4_goal",
            "very_strict_stability",
            "Full MNIST: normal plasticity, very strict stability threshold goal.",
            (
                *stripped,
                "neurogenesis.early_stop.use_threshold_goal=true",
                "neurogenesis.early_stop.threshold_goal_factor_plasticity=0.9",
                "neurogenesis.early_stop.threshold_goal_factor_stability=0.3",
            ),
            best.family,
        ),
        AblationSpec(
            "stage4_goal",
            "no_goal_reference",
            "Full MNIST: disable threshold-goal early stop as a reference.",
            (*stripped, "neurogenesis.early_stop.use_threshold_goal=false"),
            best.family,
        ),
    ]


def _stage5_specs(best: RunSummary) -> list[AblationSpec]:
    return [
        AblationSpec(
            "stage5_intrinsic",
            f"intrinsic_{best.name}",
            f"Full MNIST intrinsic replay confirmation from {best.name}.",
            _to_full(best.overrides, replay_mode="intrinsic"),
            best.family,
        )
    ]


def _shape_metrics(sizes: Sequence[int]) -> dict[str, float | int | bool]:
    if len(sizes) < 2:
        return {"funnel_violations": 0, "strict_funnel": False, "l3_over_l2": math.nan}
    violations = sum(1 for left, right in zip(sizes, sizes[1:]) if right >= left)
    l3_over_l2 = float(sizes[3] / max(sizes[2], 1)) if len(sizes) > 3 else math.nan
    return {
        "funnel_violations": violations,
        "strict_funnel": violations == 0,
        "l3_over_l2": l3_over_l2,
    }


def _quota_score(summary: RunSummary) -> float:
    if summary.final_l3_mse is None:
        return float("inf")
    diagnostics = summary.diagnostics or {}
    shape = _shape_metrics(summary.layer_sizes)
    l3_ratio = float(shape["l3_over_l2"])
    l3_ratio_penalty = max(l3_ratio - 0.8, 0.0) if math.isfinite(l3_ratio) else 1.0
    l3_caps = float(diagnostics.get("cap_hits_by_level", {}).get("3") or 0.0)
    l3_outliers = float(
        diagnostics.get("final_outlier_fraction_by_level_mean", {}).get("3") or 0.0
    )
    return (
        float(summary.final_l3_mse)
        + 0.02 * int(shape["funnel_violations"])
        + 0.01 * l3_ratio_penalty
        + 0.001 * l3_caps
        + 0.002 * l3_outliers
        + 0.00001 * sum(summary.layer_sizes)
    )


def _best_completed(
    summaries: Iterable[RunSummary],
    *,
    limit: int,
    family: str | None = None,
    require_funnel: bool = False,
) -> list[RunSummary]:
    completed = []
    for summary in summaries:
        if summary.status != "completed" or summary.final_l3_mse is None:
            continue
        if family is not None and summary.family != family:
            continue
        if require_funnel and not _shape_metrics(summary.layer_sizes)["strict_funnel"]:
            continue
        completed.append(summary)
    return sorted(completed, key=_quota_score)[:limit]


def _promotion_candidates(summaries: Sequence[RunSummary], limit: int = 3) -> list[RunSummary]:
    promoted: list[RunSummary] = []
    for family in ("fixed_quota", "adaptive_quota", "hybrid_quota"):
        best = _best_completed(summaries, limit=1, family=family)
        promoted.extend(best)
    if len(promoted) < limit:
        seen = {summary.name for summary in promoted}
        for summary in _best_completed(summaries, limit=limit):
            if summary.name not in seen:
                promoted.append(summary)
            if len(promoted) >= limit:
                break
    return promoted[:limit]


def _quick_overrides() -> tuple[str, ...]:
    return (
        "training.pretrain_epochs=1",
        "training.pretrain_finetune_epochs=0",
        "experiment.incremental_train_limit_per_class=16",
        "neurogenesis.plasticity_epochs=2",
        "neurogenesis.stability_epochs=2",
        "neurogenesis.next_layer_epochs=2",
        "neurogenesis.max_nodes=[2,2,2,2]",
        "logging.mlflow.metric_filter.class_metrics=summary",
    )


def _with_extra(spec: AblationSpec, extra: Sequence[str]) -> AblationSpec:
    if not extra:
        return spec
    return AblationSpec(
        spec.stage, spec.name, spec.description, (*spec.overrides, *extra), spec.family
    )


def _filter_specs(specs: Sequence[AblationSpec], only: str | None) -> list[AblationSpec]:
    if not only:
        return list(specs)
    selectors = [item.strip() for item in only.split(",") if item.strip()]
    return [
        spec
        for spec in specs
        if any(
            spec.name == selector
            or spec.name.startswith(selector)
            or spec.stage == selector
            or spec.stage.startswith(selector)
            for selector in selectors
        )
    ]


def _run_spec(
    spec: AblationSpec,
    *,
    tracking_uri: str,
    experiment_name: str,
    timestamp: str,
    dry_run: bool,
) -> RunSummary:
    run_name = f"outlierquota_{spec.name}_{timestamp}"
    overrides = [
        *spec.overrides,
        f"logging.mlflow.tracking_uri={tracking_uri}",
        f"logging.mlflow.experiment_name={experiment_name}",
        f"logging.mlflow.run_name={run_name}",
    ]
    print(f"\n=== {spec.stage}: {spec.name} ===")
    print(spec.description)
    print(" ".join(overrides))
    if dry_run:
        return RunSummary(
            stage=spec.stage,
            name=spec.name,
            description=spec.description,
            family=spec.family,
            run_name=run_name,
            run_id=None,
            overrides=list(spec.overrides),
            final_l3_mse=None,
            layer_sizes=[],
            per_class_l3_mse={},
            diagnostics={},
            score=None,
            status="dry_run",
        )

    cfg = _compose_cfg(overrides)
    with open_dict(cfg):
        cfg.paper_experiment = f"outlier_quota_ablation/{spec.stage}/{spec.name}"

    log_path = None
    if getattr(_run_spec, "quiet_output_dir", None) is not None:
        log_dir = getattr(_run_spec, "quiet_output_dir")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{spec.name}.log"

    if log_path is not None:
        with log_path.open("w", encoding="utf-8") as fh:
            with contextlib.redirect_stdout(fh), contextlib.redirect_stderr(fh):
                run(cfg)
        print(f"Captured run log: {log_path}")
    else:
        run(cfg)

    run_id = _query_run_id(tracking_uri, run_name)
    final_l3, sizes, per_class, diagnostics, score = _extract_mlflow_summary(
        tracking_uri, run_id
    )
    return RunSummary(
        stage=spec.stage,
        name=spec.name,
        description=spec.description,
        family=spec.family,
        run_name=run_name,
        run_id=run_id,
        overrides=list(spec.overrides),
        final_l3_mse=final_l3,
        layer_sizes=sizes,
        per_class_l3_mse=per_class,
        diagnostics=diagnostics,
        score=score,
        status="completed",
    )


def _run_specs(
    specs: Sequence[AblationSpec],
    *,
    args: argparse.Namespace,
    timestamp: str,
    output_dir: Path,
    summaries: list[RunSummary],
    max_runs_state: dict[str, int],
) -> list[RunSummary]:
    selected = _filter_specs(specs, args.only)
    completed: list[RunSummary] = []
    extra = (*(_quick_overrides() if args.quick else ()), *tuple(args.override or ()))
    for spec in selected:
        if args.max_runs is not None and max_runs_state["count"] >= args.max_runs:
            break
        summary = _run_spec(
            _with_extra(spec, extra),
            tracking_uri=args.tracking_uri,
            experiment_name=args.experiment_name,
            timestamp=timestamp,
            dry_run=args.dry_run,
        )
        summaries.append(summary)
        completed.append(summary)
        max_runs_state["count"] += 1
        _write_outputs(output_dir, summaries, _payload(args, timestamp))
    return completed


def _write_outputs(output_dir: Path, summaries: Sequence[RunSummary], payload: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = {**payload, "runs": [summary.as_dict() for summary in summaries]}
    (output_dir / "summary.json").write_text(
        json.dumps(data, indent=2) + "\n", encoding="utf-8"
    )

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "stage",
                "family",
                "name",
                "run_id",
                "final_l3_mse",
                "quota_score",
                "layer_sizes",
                "strict_funnel",
                "l3_over_l2",
                "l3_rounds_mean",
                "l3_cap_hits",
                "l3_outlier_fraction",
                "status",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            diagnostics = summary.diagnostics or {}
            shape = _shape_metrics(summary.layer_sizes)
            writer.writerow(
                {
                    "stage": summary.stage,
                    "family": summary.family,
                    "name": summary.name,
                    "run_id": summary.run_id or "",
                    "final_l3_mse": summary.final_l3_mse
                    if summary.final_l3_mse is not None
                    else "",
                    "quota_score": ""
                    if summary.final_l3_mse is None
                    else _quota_score(summary),
                    "layer_sizes": " ".join(map(str, summary.layer_sizes)),
                    "strict_funnel": shape["strict_funnel"],
                    "l3_over_l2": _fmt(shape["l3_over_l2"]),
                    "l3_rounds_mean": diagnostics.get(
                        "growth_rounds_by_level_mean", {}
                    ).get("3", ""),
                    "l3_cap_hits": diagnostics.get("cap_hits_by_level", {}).get("3", ""),
                    "l3_outlier_fraction": diagnostics.get(
                        "final_outlier_fraction_by_level_mean", {}
                    ).get("3", ""),
                    "status": summary.status,
                }
            )

    lines = [
        "# Outlier-Quota Ablation Summary",
        "",
        f"- no shape pressure: `{REFERENCE_NO_PRESSURE_MSE:.5f}` sizes `{REFERENCE_NO_PRESSURE_SIZES}`",
        f"- strong L3 gate: `{REFERENCE_STRONG_GATE_MSE:.5f}` sizes `{REFERENCE_STRONG_GATE_SIZES}`",
        f"- capped phase strong gate: `{REFERENCE_CAPPED_STRONG_GATE_MSE:.5f}` sizes `{REFERENCE_CAPPED_STRONG_GATE_SIZES}`",
        f"- paper-size fixed AE: `{REFERENCE_PAPER_AE_MSE:.5f}` sizes `{REFERENCE_PAPER_AE_SIZES}`",
        "",
        "| Stage | Family | Run | L3 MSE | Quota score | Sizes | Strict funnel | L3/L2 | L3 rounds | L3 caps | L3 outliers | MLflow run |",
        "|---|---|---|---:|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for summary in summaries:
        diagnostics = summary.diagnostics or {}
        shape = _shape_metrics(summary.layer_sizes)
        rounds = diagnostics.get("growth_rounds_by_level_mean", {})
        caps = diagnostics.get("cap_hits_by_level", {})
        outliers = diagnostics.get("final_outlier_fraction_by_level_mean", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    summary.stage,
                    summary.family,
                    summary.name,
                    "" if summary.final_l3_mse is None else f"{summary.final_l3_mse:.5f}",
                    ""
                    if summary.final_l3_mse is None
                    else f"{_quota_score(summary):.5f}",
                    " ".join(map(str, summary.layer_sizes)),
                    str(shape["strict_funnel"]),
                    _fmt(shape["l3_over_l2"]),
                    _fmt(rounds.get("3")),
                    _fmt(caps.get("3")),
                    _fmt(outliers.get("3")),
                    summary.run_id or "",
                ]
            )
            + " |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _payload(args: argparse.Namespace, timestamp: str) -> dict:
    return {
        "timestamp": timestamp,
        "tracking_uri": args.tracking_uri,
        "experiment_name": args.experiment_name,
        "quick": bool(args.quick),
        "dry_run": bool(args.dry_run),
        "score": {
            "formula": "mse + 0.02*funnel_violations + 0.01*max(l3/l2 - 0.8, 0) + 0.001*l3_caps + 0.002*l3_outliers + 0.00001*total_size",
        },
        "references": {
            "no_shape_pressure": {
                "mse": REFERENCE_NO_PRESSURE_MSE,
                "sizes": REFERENCE_NO_PRESSURE_SIZES,
            },
            "strong_l3_gate": {
                "mse": REFERENCE_STRONG_GATE_MSE,
                "sizes": REFERENCE_STRONG_GATE_SIZES,
            },
            "capped_phase_strong_gate": {
                "mse": REFERENCE_CAPPED_STRONG_GATE_MSE,
                "sizes": REFERENCE_CAPPED_STRONG_GATE_SIZES,
            },
            "paper_size_fixed_ae": {
                "mse": REFERENCE_PAPER_AE_MSE,
                "sizes": REFERENCE_PAPER_AE_SIZES,
            },
        },
    }


def _dry_seed(name: str, family: str, overrides: Sequence[str]) -> RunSummary:
    return RunSummary(
        stage="seed",
        name=name,
        description="dry-run seed",
        family=family,
        run_name="",
        run_id=None,
        overrides=list(overrides),
        final_l3_mse=None,
        layer_sizes=[],
        per_class_l3_mse={},
        diagnostics={},
        score=None,
        status="dry_seed",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("stage1", "stage2", "stage3", "stage4", "stage5", "all"),
        default="stage1",
    )
    parser.add_argument("--experiment-name", default="neurogenesis-diagnostics")
    parser.add_argument("--tracking-uri", default=_default_tracking_uri())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-runs", action="store_true")
    parser.add_argument("--only", default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--promote-top", type=int, default=3)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (
        REPO_ROOT / "outputs" / "diagnostics" / "outlier_quota_ablation" / timestamp
    )
    if args.quiet:
        _run_spec.quiet_output_dir = output_dir / "logs"  # type: ignore[attr-defined]

    if args.list_runs:
        stage1 = _stage1_specs()
        seeds = [
            _dry_seed("l3_quota_020", "fixed_quota", stage1[2].overrides),
            _dry_seed("shape_gate_04_p20", "adaptive_quota", stage1[6].overrides),
            _dry_seed("hybrid_l3_quota_020_gate_05_p15", "hybrid_quota", stage1[8].overrides),
        ]
        specs = [
            *stage1,
            *_stage2_specs(seeds),
            *_stage3_specs(seeds[0]),
            *_stage4_specs(seeds[0]),
            *_stage5_specs(seeds[0]),
        ]
        for spec in _filter_specs(specs, args.only):
            print(f"{spec.stage}/{spec.name}: {spec.description}")
        return

    summaries: list[RunSummary] = []
    max_runs_state = {"count": 0}
    stage1_summaries: list[RunSummary] = []
    if args.stage in {"stage1", "all"}:
        stage1_summaries = _run_specs(
            _stage1_specs(),
            args=args,
            timestamp=timestamp,
            output_dir=output_dir,
            summaries=summaries,
            max_runs_state=max_runs_state,
        )

    candidates = _promotion_candidates(stage1_summaries or summaries, args.promote_top)
    if args.stage in {"stage2", "all"}:
        if not candidates and not args.dry_run:
            raise SystemExit("--stage stage2 requires completed stage1 candidates.")
        seeds = candidates or [
            _dry_seed("l3_quota_020", "fixed_quota", _stage1_specs()[2].overrides),
            _dry_seed("shape_gate_04_p20", "adaptive_quota", _stage1_specs()[6].overrides),
            _dry_seed(
                "hybrid_l3_quota_020_gate_05_p15",
                "hybrid_quota",
                _stage1_specs()[8].overrides,
            ),
        ]
        stage2 = _run_specs(
            _stage2_specs(seeds[: args.promote_top]),
            args=args,
            timestamp=timestamp,
            output_dir=output_dir,
            summaries=summaries,
            max_runs_state=max_runs_state,
        )
        candidates = _best_completed(stage2, limit=1, require_funnel=True) or _best_completed(
            stage2, limit=1
        )

    best = candidates[0] if candidates else _dry_seed(
        "l3_quota_020", "fixed_quota", _stage1_specs()[2].overrides
    )
    if args.stage in {"stage3", "all"}:
        _run_specs(
            _stage3_specs(best),
            args=args,
            timestamp=timestamp,
            output_dir=output_dir,
            summaries=summaries,
            max_runs_state=max_runs_state,
        )
    if args.stage in {"stage4", "all"}:
        _run_specs(
            _stage4_specs(best),
            args=args,
            timestamp=timestamp,
            output_dir=output_dir,
            summaries=summaries,
            max_runs_state=max_runs_state,
        )
    if args.stage in {"stage5", "all"}:
        _run_specs(
            _stage5_specs(best),
            args=args,
            timestamp=timestamp,
            output_dir=output_dir,
            summaries=summaries,
            max_runs_state=max_runs_state,
        )

    _write_outputs(output_dir, summaries, _payload(args, timestamp))
    print(f"Wrote summaries to {output_dir}")


if __name__ == "__main__":
    main()
