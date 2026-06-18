"""Run staged funnel-shape pressure ablations for NDL growth diagnostics."""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as _dt
import json
import math
import sys
from pathlib import Path
from typing import Sequence

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


BEST_CLEAN_MSE = 0.009213654324412346
BEST_CLEAN_SIZES = [225, 119, 83, 140]
BEST_FUNNEL_REFERENCE_MSE = 0.008744465455412865
BEST_FUNNEL_REFERENCE_SIZES = [225, 135, 83, 40]
BEST_IR_MSE = 0.02005576342344284
BEST_IR_SIZES = [208, 112, 79, 180]

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
    "neurogenesis.early_stop.min_delta=3e-6",
    "neurogenesis.early_stop.patience=5",
    "neurogenesis.early_stop.use_threshold_goal=false",
)

SINGLE_0_OVERRIDES = (
    *BASE_OVERRIDES,
    "experiment.incremental_classes=[0]",
)


def _target(level: int, value: float) -> str:
    return f"+neurogenesis.shape_target_ratio_by_level.{level}={value}"


def _stage1_specs() -> list[AblationSpec]:
    l3_target = (
        "neurogenesis.shape_target_ratio=1.0",
        _target(3, 0.5),
    )
    all_targets = (
        "neurogenesis.shape_target_ratio=1.0",
        _target(1, 0.75),
        _target(2, 0.70),
        _target(3, 0.50),
    )
    return [
        AblationSpec(
            "stage1_single0",
            "baseline_no_shape_pressure",
            "Single-0 clean replay with no funnel-shape pressure.",
            SINGLE_0_OVERRIDES,
            "baseline",
        ),
        AblationSpec(
            "stage1_single0",
            "l3_scale_growth",
            "Single-0 clean replay: reduce L3 growth once L3/L2 exceeds 0.5.",
            (
                *SINGLE_0_OVERRIDES,
                "neurogenesis.shape_pressure_mode=scale_growth",
                *l3_target,
                "neurogenesis.shape_min_growth_scale=0.25",
            ),
            "l3_soft",
        ),
        AblationSpec(
            "stage1_single0",
            "l3_scale_gate",
            "Single-0 clean replay: require more L3 outliers once L3/L2 exceeds 0.5.",
            (
                *SINGLE_0_OVERRIDES,
                "neurogenesis.shape_pressure_mode=scale_gate",
                *l3_target,
                "neurogenesis.shape_gate_power=1.0",
            ),
            "l3_gate",
        ),
        AblationSpec(
            "stage1_single0",
            "l3_scale_both",
            "Single-0 clean replay: combine L3 growth scaling and stricter gate.",
            (
                *SINGLE_0_OVERRIDES,
                "neurogenesis.shape_pressure_mode=scale_both",
                *l3_target,
                "neurogenesis.shape_min_growth_scale=0.25",
                "neurogenesis.shape_gate_power=1.0",
            ),
            "l3_both",
        ),
        AblationSpec(
            "stage1_single0",
            "l3_strong_gate",
            "Single-0 clean replay: stronger L3 bottleneck gate.",
            (
                *SINGLE_0_OVERRIDES,
                "neurogenesis.shape_pressure_mode=scale_gate",
                "neurogenesis.shape_target_ratio=1.0",
                _target(3, 0.4),
                "neurogenesis.shape_gate_power=2.0",
            ),
            "l3_gate",
        ),
        AblationSpec(
            "stage1_single0",
            "all_layers_scale_both",
            "Single-0 clean replay: keep the full hierarchy funnel-shaped.",
            (
                *SINGLE_0_OVERRIDES,
                "neurogenesis.shape_pressure_mode=scale_both",
                *all_targets,
                "neurogenesis.shape_min_growth_scale=0.25",
                "neurogenesis.shape_gate_power=1.0",
            ),
            "all_layers",
        ),
    ]


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


def _shape_metrics(sizes: Sequence[int]) -> dict[str, float | int | bool]:
    if len(sizes) < 2:
        return {
            "funnel_violations": 0,
            "strict_funnel": False,
            "l3_over_l2": math.nan,
        }
    violations = sum(1 for left, right in zip(sizes, sizes[1:]) if right >= left)
    l3_over_l2 = (
        float(sizes[3] / max(sizes[2], 1)) if len(sizes) > 3 else math.nan
    )
    return {
        "funnel_violations": violations,
        "strict_funnel": violations == 0,
        "l3_over_l2": l3_over_l2,
    }


def _funnel_score(summary: RunSummary) -> float:
    if summary.final_l3_mse is None:
        return float("inf")
    metrics = _shape_metrics(summary.layer_sizes)
    violations = int(metrics["funnel_violations"])
    l3_over_l2 = float(metrics["l3_over_l2"])
    l3_penalty = max(l3_over_l2 - 1.0, 0.0) if math.isfinite(l3_over_l2) else 1.0
    return (
        float(summary.final_l3_mse)
        + 0.02 * violations
        + 0.01 * l3_penalty
        + 0.00001 * sum(summary.layer_sizes)
    )


def _best_funnel(summaries: Sequence[RunSummary], *, limit: int) -> list[RunSummary]:
    completed = [
        summary
        for summary in summaries
        if summary.status == "completed" and summary.final_l3_mse is not None
    ]
    return sorted(completed, key=_funnel_score)[:limit]


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


def _stage3_specs(promoted: Sequence[RunSummary]) -> list[AblationSpec]:
    return [
        AblationSpec(
            "stage3_full_intrinsic",
            f"full_intrinsic_{idx}_{summary.name}",
            f"Full MNIST intrinsic replay confirmation from {summary.name}.",
            _to_full(summary.overrides, replay_mode="intrinsic"),
            summary.family,
        )
        for idx, summary in enumerate(promoted[:2], start=1)
    ]


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
    run_name = f"funnelshape_{spec.name}_{timestamp}"
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
        cfg.paper_experiment = f"funnel_shape_ablation/{spec.stage}/{spec.name}"

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
                "funnel_score",
                "layer_sizes",
                "strict_funnel",
                "violations",
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
                    "funnel_score": ""
                    if summary.final_l3_mse is None
                    else _funnel_score(summary),
                    "layer_sizes": " ".join(map(str, summary.layer_sizes)),
                    "strict_funnel": shape["strict_funnel"],
                    "violations": shape["funnel_violations"],
                    "l3_over_l2": _fmt(shape["l3_over_l2"]),
                    "l3_rounds_mean": diagnostics.get(
                        "growth_rounds_by_level_mean", {}
                    ).get("3", ""),
                    "l3_cap_hits": diagnostics.get("cap_hits_by_level", {}).get(
                        "3", ""
                    ),
                    "l3_outlier_fraction": diagnostics.get(
                        "final_outlier_fraction_by_level_mean", {}
                    ).get("3", ""),
                    "status": summary.status,
                }
            )

    lines = [
        "# Funnel-Shape Ablation Summary",
        "",
        f"- paper-size fixed AE: `{BEST_FUNNEL_REFERENCE_MSE:.5f}` sizes `{BEST_FUNNEL_REFERENCE_SIZES}`",
        f"- best clean NDL so far: `{BEST_CLEAN_MSE:.5f}` sizes `{BEST_CLEAN_SIZES}`",
        f"- best IR NDL so far: `{BEST_IR_MSE:.5f}` sizes `{BEST_IR_SIZES}`",
        "",
        "| Stage | Family | Run | L3 MSE | Funnel score | Sizes | Strict funnel | L3/L2 | L3 rounds | L3 caps | L3 outliers | MLflow run |",
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
                    else f"{_funnel_score(summary):.5f}",
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
            "formula": "mse + 0.02*funnel_violations + 0.01*max(l3/l2 - 1, 0) + 0.00001*total_size",
        },
        "baselines": {
            "paper_size_fixed_ae": {
                "mse": BEST_FUNNEL_REFERENCE_MSE,
                "sizes": BEST_FUNNEL_REFERENCE_SIZES,
            },
            "best_clean_ndl": {"mse": BEST_CLEAN_MSE, "sizes": BEST_CLEAN_SIZES},
            "best_ir_ndl": {"mse": BEST_IR_MSE, "sizes": BEST_IR_SIZES},
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("stage1", "stage2", "stage3", "all"), default="stage1")
    parser.add_argument("--experiment-name", default="neurogenesis-diagnostics")
    parser.add_argument("--tracking-uri", default=_default_tracking_uri())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-runs", action="store_true")
    parser.add_argument("--only", default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--promote-top", type=int, default=2)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


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


def main() -> None:
    args = _parse_args()
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (
        REPO_ROOT / "outputs" / "diagnostics" / "funnel_shape_ablation" / timestamp
    )
    if args.quiet:
        _run_spec.quiet_output_dir = output_dir / "logs"  # type: ignore[attr-defined]

    if args.list_runs:
        dry_promoted = [
            _dry_seed("l3_scale_gate", "l3_gate", _stage1_specs()[2].overrides),
            _dry_seed("l3_scale_both", "l3_both", _stage1_specs()[3].overrides),
        ]
        specs = [*_stage1_specs(), *_stage2_specs(dry_promoted), *_stage3_specs(dry_promoted)]
        for spec in _filter_specs(specs, args.only):
            print(f"{spec.stage}/{spec.name}: {spec.description}")
        return

    summaries: list[RunSummary] = []
    max_runs_state = {"count": 0}

    stage1: list[RunSummary] = []
    if args.stage in {"stage1", "all"}:
        stage1 = _run_specs(
            _stage1_specs(),
            args=args,
            timestamp=timestamp,
            output_dir=output_dir,
            summaries=summaries,
            max_runs_state=max_runs_state,
        )

    promoted_stage1 = _best_funnel(stage1 or summaries, limit=args.promote_top)
    if args.dry_run and not promoted_stage1:
        promoted_stage1 = [
            _dry_seed("l3_scale_gate", "l3_gate", _stage1_specs()[2].overrides),
            _dry_seed("l3_scale_both", "l3_both", _stage1_specs()[3].overrides),
        ]

    stage2: list[RunSummary] = []
    if args.stage in {"stage2", "all"}:
        if not promoted_stage1 and not args.dry_run:
            raise SystemExit("--stage stage2 requires completed stage1 results.")
        stage2 = _run_specs(
            _stage2_specs(promoted_stage1),
            args=args,
            timestamp=timestamp,
            output_dir=output_dir,
            summaries=summaries,
            max_runs_state=max_runs_state,
        )

    promoted_stage2 = _best_funnel(stage2, limit=min(args.promote_top, 2))
    if args.dry_run and not promoted_stage2:
        promoted_stage2 = promoted_stage1[:2]

    if args.stage in {"stage3", "all"}:
        if not promoted_stage2 and not args.dry_run:
            raise SystemExit("--stage stage3 requires completed stage2 results.")
        _run_specs(
            _stage3_specs(promoted_stage2),
            args=args,
            timestamp=timestamp,
            output_dir=output_dir,
            summaries=summaries,
            max_runs_state=max_runs_state,
        )

    _write_outputs(output_dir, summaries, _payload(args, timestamp))
    print(f"\nFunnel-shape ablation summary written to {output_dir}")


if __name__ == "__main__":
    main()
