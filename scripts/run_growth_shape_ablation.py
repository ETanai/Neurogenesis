"""Run staged growth-shape ablations for NDL funnel diagnostics."""

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


PAPER_AE_MSE = 0.008744465455412865
PAPER_AE_SIZES = [225, 135, 83, 40]
BEST_CLEAN_MSE = 0.009343640878796577
BEST_CLEAN_SIZES = [240, 116, 90, 146]
BEST_IR_MSE = 0.02059805952012539
BEST_IR_SIZES = [240, 187, 359, 180]

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


def _stage1_specs() -> list[AblationSpec]:
    return [
        AblationSpec(
            "stage1_single0",
            "baseline_best_global",
            "Single-0 clean replay with current best growth settings.",
            SINGLE_0_OVERRIDES,
            "baseline",
        ),
        AblationSpec(
            "stage1_single0",
            "paper_shape_caps",
            "Single-0 clean replay with paper-target total growth caps.",
            (*SINGLE_0_OVERRIDES, "neurogenesis.max_nodes=[25,35,8,20]"),
            "cap",
        ),
        AblationSpec(
            "stage1_single0",
            "l3_cap_20",
            "Single-0 clean replay with relaxed lower caps and L3 capped at 20.",
            (*SINGLE_0_OVERRIDES, "neurogenesis.max_nodes=[100,100,100,20]"),
            "cap",
        ),
        AblationSpec(
            "stage1_single0",
            "l3_cap_5",
            "Single-0 clean replay with relaxed lower caps and L3 capped at 5.",
            (*SINGLE_0_OVERRIDES, "neurogenesis.max_nodes=[100,100,100,5]"),
            "cap",
        ),
        AblationSpec(
            "stage1_single0",
            "small_global_growth_0002",
            "Single-0 clean replay with smaller proportional growth.",
            (
                *SINGLE_0_OVERRIDES,
                "neurogenesis.factor_new_nodes=0.002",
                "neurogenesis.factor_max_new_nodes=0.05",
            ),
            "small_proportional",
        ),
        AblationSpec(
            "stage1_single0",
            "small_global_growth_0001",
            "Single-0 clean replay with very small proportional growth.",
            (
                *SINGLE_0_OVERRIDES,
                "neurogenesis.factor_new_nodes=0.001",
                "neurogenesis.factor_max_new_nodes=0.05",
            ),
            "small_proportional",
        ),
        AblationSpec(
            "stage1_single0",
            "absolute_global_1",
            "Single-0 clean replay adding one node per unresolved growth round.",
            (
                *SINGLE_0_OVERRIDES,
                "neurogenesis.growth_mode=absolute",
                "neurogenesis.absolute_new_nodes=1",
            ),
            "absolute_l3",
        ),
        AblationSpec(
            "stage1_single0",
            "l3_absolute_1",
            "Single-0 clean replay with only L3 switched to one-node growth.",
            (
                *SINGLE_0_OVERRIDES,
                "neurogenesis.growth_mode=proportional",
                "+neurogenesis.growth_mode_by_level.3=absolute",
                "+neurogenesis.absolute_new_nodes_by_level.3=1",
                "neurogenesis.max_nodes=[100,100,100,20]",
            ),
            "absolute_l3",
        ),
        AblationSpec(
            "stage1_single0",
            "l3_small_factor",
            "Single-0 clean replay with only L3 using smaller proportional growth.",
            (
                *SINGLE_0_OVERRIDES,
                "+neurogenesis.factor_new_nodes_by_level.3=0.001",
                "+neurogenesis.factor_max_new_nodes_by_level.3=0.05",
                "neurogenesis.max_nodes=[100,100,100,20]",
            ),
            "small_proportional",
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


def _best_completed(
    summaries: Iterable[RunSummary], *, family: str | None = None, limit: int = 1
) -> list[RunSummary]:
    scored = [
        summary
        for summary in summaries
        if summary.score is not None
        and math.isfinite(float(summary.score))
        and (family is None or summary.family == family)
    ]
    return sorted(scored, key=lambda item: float(item.score))[:limit]


def _stage2_promotions(stage1: Sequence[RunSummary]) -> list[RunSummary]:
    promoted: list[RunSummary] = []
    seen: set[str] = set()
    for family in ("cap", "small_proportional", "absolute_l3"):
        for summary in _best_completed(stage1, family=family, limit=1):
            promoted.append(summary)
            seen.add(summary.name)
    for summary in _best_completed(stage1, limit=3):
        if summary.name not in seen:
            promoted.append(summary)
            seen.add(summary.name)
        if len(promoted) >= 3:
            break
    return promoted[:3]


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
    run_name = f"growthshape_{spec.name}_{timestamp}"
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
        cfg.paper_experiment = f"growth_shape_ablation/{spec.stage}/{spec.name}"

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
    config_dir = output_dir / "per_run_configs"
    config_dir.mkdir(exist_ok=True)
    data = {**payload, "runs": [summary.as_dict() for summary in summaries]}
    (output_dir / "summary.json").write_text(
        json.dumps(data, indent=2) + "\n", encoding="utf-8"
    )
    for summary in summaries:
        config_path = config_dir / f"{summary.name}.json"
        config_path.write_text(
            json.dumps(
                {
                    "stage": summary.stage,
                    "family": summary.family,
                    "run_name": summary.run_name,
                    "run_id": summary.run_id,
                    "overrides": summary.overrides,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
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
                "score",
                "layer_sizes",
                "l2_rounds_mean",
                "l3_rounds_mean",
                "l3_cap_hits",
                "l3_outlier_fraction",
                "status",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            diagnostics = summary.diagnostics or {}
            writer.writerow(
                {
                    "stage": summary.stage,
                    "family": summary.family,
                    "name": summary.name,
                    "run_id": summary.run_id or "",
                    "final_l3_mse": summary.final_l3_mse
                    if summary.final_l3_mse is not None
                    else "",
                    "score": summary.score if summary.score is not None else "",
                    "layer_sizes": " ".join(map(str, summary.layer_sizes)),
                    "l2_rounds_mean": diagnostics.get(
                        "growth_rounds_by_level_mean", {}
                    ).get("2", ""),
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
        "# Growth-Shape Ablation Summary",
        "",
        f"- paper-size fixed AE: `{PAPER_AE_MSE:.5f}` sizes `{PAPER_AE_SIZES}`",
        f"- best old clean NDL: `{BEST_CLEAN_MSE:.5f}` sizes `{BEST_CLEAN_SIZES}`",
        f"- best old IR NDL: `{BEST_IR_MSE:.5f}` sizes `{BEST_IR_SIZES}`",
        "",
        "| Stage | Family | Run | L3 MSE | Score | Sizes | L2 rounds | L3 rounds | L3 caps | L3 outliers | MLflow run |",
        "|---|---|---|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for summary in summaries:
        diagnostics = summary.diagnostics or {}
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
                    "" if summary.score is None else f"{summary.score:.5f}",
                    " ".join(map(str, summary.layer_sizes)),
                    _fmt(rounds.get("2")),
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
        "baselines": {
            "paper_size_fixed_ae": {"mse": PAPER_AE_MSE, "sizes": PAPER_AE_SIZES},
            "best_old_clean_ndl": {"mse": BEST_CLEAN_MSE, "sizes": BEST_CLEAN_SIZES},
            "best_old_ir_ndl": {"mse": BEST_IR_MSE, "sizes": BEST_IR_SIZES},
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
        REPO_ROOT / "outputs" / "diagnostics" / "growth_shape_ablation" / timestamp
    )
    if args.quiet:
        _run_spec.quiet_output_dir = output_dir / "logs"  # type: ignore[attr-defined]

    summaries: list[RunSummary] = []
    max_runs_state = {"count": 0}

    if args.list_runs:
        specs = _stage1_specs()
        dry_promoted = [
            _dry_seed("paper_shape_caps", "cap", _stage1_specs()[1].overrides),
            _dry_seed("small_global_growth_0002", "small_proportional", _stage1_specs()[4].overrides),
            _dry_seed("l3_absolute_1", "absolute_l3", _stage1_specs()[7].overrides),
        ]
        specs.extend(_stage2_specs(dry_promoted))
        specs.extend(_stage3_specs(dry_promoted[:2]))
        for spec in _filter_specs(specs, args.only):
            print(f"{spec.stage}/{spec.name}: {spec.description}")
        return

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

    promoted_stage1 = _stage2_promotions(stage1 or summaries)
    if args.dry_run and not promoted_stage1:
        promoted_stage1 = [
            _dry_seed("paper_shape_caps", "cap", _stage1_specs()[1].overrides),
            _dry_seed("small_global_growth_0002", "small_proportional", _stage1_specs()[4].overrides),
            _dry_seed("l3_absolute_1", "absolute_l3", _stage1_specs()[7].overrides),
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

    promoted_stage2 = _best_completed(stage2, limit=2)
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
    print(f"\nGrowth-shape ablation summary written to {output_dir}")


if __name__ == "__main__":
    main()
