"""Run staged global-coupling diagnostics for paper-local NDL."""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_early_stop_ablation import (  # noqa: E402
    AblationSpec,
    RunSummary,
    _default_tracking_uri,
    _filter_specs,
    _run_spec,
    _with_extra,
)


PAPER_LOCAL_BASE = (
    "data=mnist",
    "experiment=mnist_incremental",
    "experiment.regime=ndl_ir",
    "experiment.base_classes=[1,7]",
    "experiment.incremental_classes=[0,2,3,4,5,6,8,9]",
    "experiment.model.hidden_sizes=[200,100,75,20]",
    "neurogenesis.thresholds=null",
    "neurogenesis.objective_mode=paper_level_ae",
    "neurogenesis.next_layer_optimization=paper_columns",
    "neurogenesis.plasticity_decoder_lr_ratio=0.01",
    "neurogenesis.stability_replay_mode=paper",
    "replay.enabled=true",
    "replay.mode=dataset",
    "replay.ir_sampling_mode=gaussian_full",
    "replay.ir_cov_shrinkage=0.0",
    "replay.ir_noise_scale=1.0",
    "training.pretrain_mode=stacked_denoising",
    "training.pretrain_finetune_epochs=0",
)


def _replace(overrides: Sequence[str], key: str, value: object) -> tuple[str, ...]:
    prefix = f"{key}="
    return (*[item for item in overrides if not item.startswith(prefix)], f"{key}={value}")


def _single0(overrides: Sequence[str]) -> tuple[str, ...]:
    return _replace(overrides, "experiment.incremental_classes", "[0]")


def _full(overrides: Sequence[str], replay_mode: str) -> tuple[str, ...]:
    out = [
        item
        for item in overrides
        if not item.startswith("experiment.incremental_classes=")
        and not item.startswith("experiment.skip_incremental_training=")
        and not item.startswith("replay.mode=")
    ]
    out.extend(("experiment.incremental_classes=[0,2,3,4,5,6,8,9]", f"replay.mode={replay_mode}"))
    return tuple(out)


def _enable_coupling(
    overrides: Sequence[str], *, trigger: str, epochs: int, scope: str, lr_ratio: float = 0.01
) -> tuple[str, ...]:
    return (
        *overrides,
        "neurogenesis.global_coupling.enabled=true",
        f"neurogenesis.global_coupling.trigger={trigger}",
        f"neurogenesis.global_coupling.epochs={epochs}",
        f"neurogenesis.global_coupling.scope={scope}",
        f"neurogenesis.global_coupling.lr_ratio={lr_ratio}",
    )


def base_specs() -> list[AblationSpec]:
    specs = []
    for epochs in (0, 5, 10):
        name = "paper_local_no_finetune" if epochs == 0 else f"paper_local_base_finetune{epochs}"
        specs.append(
            AblationSpec(
                "base",
                name,
                "Base-only paper-local pretraining/coupling check.",
                (
                    *_replace(PAPER_LOCAL_BASE, "experiment.skip_incremental_training", "true"),
                    f"training.pretrain_finetune_epochs={epochs}",
                ),
                "base",
            )
        )
    return specs


def single0_specs() -> list[AblationSpec]:
    base = _single0(PAPER_LOCAL_BASE)
    specs = [
        AblationSpec(
            "single0",
            "single0_no_coupling",
            "Single-0 paper-local baseline without global coupling.",
            base,
            "none",
        )
    ]
    for epochs in (1, 5, 10):
        for scope in ("all", "decoder_only", "freeze_old_encoder"):
            specs.append(
                AblationSpec(
                    "single0",
                    f"single0_after_class_e{epochs}_{scope}",
                    "Single-0: global coupling after each class.",
                    _enable_coupling(base, trigger="after_class", epochs=epochs, scope=scope),
                    "after_class",
                )
            )
    for epochs in (1, 5):
        for scope in ("all", "freeze_old_encoder"):
            specs.append(
                AblationSpec(
                    "single0",
                    f"single0_after_level_e{epochs}_{scope}",
                    "Single-0: global coupling after each level.",
                    _enable_coupling(base, trigger="after_level", epochs=epochs, scope=scope),
                    "after_level",
                )
            )
    for scope in ("all", "freeze_old_encoder"):
        specs.append(
            AblationSpec(
                "single0",
                f"single0_after_growth_round_e1_{scope}",
                "Single-0: global coupling after each growth round.",
                _enable_coupling(base, trigger="after_growth_round", epochs=1, scope=scope),
                "after_growth_round",
            )
        )
    return specs


def full_specs(candidates: Sequence[RunSummary]) -> list[AblationSpec]:
    seeds = candidates or [_seed(single0_specs()[1])]
    return [
        AblationSpec(
            "full",
            f"full_dataset_{idx}_{summary.name}",
            f"Full MNIST clean replay promotion from {summary.name}.",
            _full(summary.overrides, "dataset"),
            "full_dataset",
        )
        for idx, summary in enumerate(seeds, start=1)
    ]


def ir_specs(candidates: Sequence[RunSummary]) -> list[AblationSpec]:
    seeds = candidates or [_seed(single0_specs()[1])]
    return [
        AblationSpec(
            "ir",
            f"full_ir_{idx}_{summary.name}",
            f"Full MNIST IR promotion from {summary.name}.",
            _full(summary.overrides, "intrinsic"),
            "ir",
        )
        for idx, summary in enumerate(seeds, start=1)
    ]


def control_specs(candidates: Sequence[RunSummary]) -> list[AblationSpec]:
    specs = []
    for idx, summary in enumerate(candidates, start=1):
        sizes = summary.layer_sizes or [200, 100, 75, 20]
        specs.append(
            AblationSpec(
                "ir",
                f"matched_cl_ir_{idx}_{summary.name}",
                f"Matched CL+IR control for {summary.name}.",
                (
                    "data=mnist",
                    "experiment=mnist_incremental",
                    "experiment.regime=cl_ir",
                    "experiment.base_classes=[1,7]",
                    "experiment.incremental_classes=[0,2,3,4,5,6,8,9]",
                    f"experiment.control_hidden_sizes=[{','.join(str(size) for size in sizes)}]",
                    "replay.enabled=true",
                    "replay.mode=intrinsic",
                    "replay.ir_sampling_mode=gaussian_full",
                    "training.pretrain_mode=stacked_denoising",
                    "training.pretrain_finetune_epochs=0",
                ),
                "control",
            )
        )
    return specs


def _seed(spec: AblationSpec) -> RunSummary:
    return RunSummary(
        spec.stage,
        spec.name,
        spec.description,
        spec.family,
        spec.name,
        None,
        list(spec.overrides),
        None,
        [],
        {},
        {},
        None,
        "seed",
    )


def _quick_overrides() -> tuple[str, ...]:
    return (
        "training.pretrain_epochs=1",
        "experiment.incremental_train_limit_per_class=16",
        "neurogenesis.plasticity_epochs=2",
        "neurogenesis.stability_epochs=2",
        "neurogenesis.next_layer_epochs=2",
        "neurogenesis.max_nodes=[2,2,2,2]",
        "neurogenesis.global_coupling.epochs=1",
        "logging.mlflow.metric_filter.class_metrics=summary",
    )


def _score(summary: RunSummary) -> float:
    if summary.final_l3_mse is None or not math.isfinite(float(summary.final_l3_mse)):
        return float("inf")
    sizes = summary.layer_sizes
    size_penalty = 0.0
    if len(sizes) >= 3:
        size_penalty = 0.00001 * sum(sizes)
        if sizes[1] > 220 or sizes[2] > 180:
            size_penalty += 0.05
    retention = sum(summary.per_class_l3_mse.get(cls, 0.0) for cls in ("1", "7"))
    return float(summary.final_l3_mse) + size_penalty + 0.1 * retention


def _promote(summaries: Sequence[RunSummary], stage: str, limit: int) -> list[RunSummary]:
    candidates = [
        item
        for item in summaries
        if item.stage == stage and item.status == "completed" and item.final_l3_mse is not None
    ]
    return sorted(candidates, key=_score)[:limit]


def _attach_score(summary: RunSummary) -> RunSummary:
    summary.score = _score(summary)
    return summary


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
    if args.list_runs:
        for spec in selected:
            print(f"{spec.stage}/{spec.name}: {spec.description}")
        raise SystemExit(0)
    extra = (*(_quick_overrides() if args.quick else ()), *tuple(args.override or ()))
    completed = []
    (output_dir / "per_run_configs").mkdir(parents=True, exist_ok=True)
    for spec in selected:
        if args.max_runs is not None and max_runs_state["count"] >= args.max_runs:
            break
        spec = _with_extra(spec, extra)
        _write_run_config(output_dir / "per_run_configs", spec)
        summary = _run_spec(
            spec,
            tracking_uri=args.tracking_uri,
            experiment_name=args.experiment_name,
            timestamp=timestamp,
            dry_run=args.dry_run,
        )
        summary = _attach_score(summary)
        summaries.append(summary)
        completed.append(summary)
        max_runs_state["count"] += 1
        _write_outputs(output_dir, summaries, args, timestamp)
    return completed


def _write_run_config(path: Path, spec: AblationSpec) -> None:
    payload = {
        "stage": spec.stage,
        "name": spec.name,
        "description": spec.description,
        "family": spec.family,
        "overrides": list(spec.overrides),
    }
    (path / f"{spec.stage}__{spec.name}.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def _write_outputs(
    output_dir: Path, summaries: Sequence[RunSummary], args: argparse.Namespace, timestamp: str
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": timestamp,
        "tracking_uri": args.tracking_uri,
        "experiment_name": args.experiment_name,
        "runs": [summary.as_dict() for summary in summaries],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["stage", "family", "name", "run_id", "l3_mse", "score", "sizes", "status"],
        )
        writer.writeheader()
        for item in summaries:
            writer.writerow(
                {
                    "stage": item.stage,
                    "family": item.family,
                    "name": item.name,
                    "run_id": item.run_id or "",
                    "l3_mse": item.final_l3_mse if item.final_l3_mse is not None else "",
                    "score": item.score if item.score is not None else "",
                    "sizes": " ".join(map(str, item.layer_sizes)),
                    "status": item.status,
                }
            )
    lines = [
        "# Global Coupling Ablation Summary",
        "",
        "| Stage | Family | Run | L3 MSE | Score | Sizes | MLflow run |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for item in summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    item.stage,
                    item.family,
                    item.name,
                    "" if item.final_l3_mse is None else f"{item.final_l3_mse:.5f}",
                    "" if item.score is None else f"{item.score:.5f}",
                    " ".join(map(str, item.layer_sizes)),
                    item.run_id or "",
                ]
            )
            + " |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _list_specs(stage: str) -> list[AblationSpec]:
    if stage == "base":
        return base_specs()
    if stage == "single0":
        return single0_specs()
    if stage == "full":
        return full_specs([])
    if stage == "ir":
        return [*ir_specs([]), *control_specs([_seed(single0_specs()[1])])]
    return [*base_specs(), *single0_specs(), *full_specs([]), *ir_specs([])]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("base", "single0", "full", "ir", "all"), default="single0")
    parser.add_argument("--experiment-name", default="neurogenesis-diagnostics")
    parser.add_argument("--tracking-uri", default=_default_tracking_uri())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--promote-top", type=int, default=3)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-runs", action="store_true")
    parser.add_argument("--only", default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (
        REPO_ROOT / "outputs" / "diagnostics" / "global_coupling_ablation" / timestamp
    )
    if args.list_runs:
        for spec in _filter_specs(_list_specs(args.stage), args.only):
            print(f"{spec.stage}/{spec.name}: {spec.description}")
        return

    summaries: list[RunSummary] = []
    state = {"count": 0}
    _run_spec.run_prefix = "coupling"  # type: ignore[attr-defined]
    if args.quiet:
        _run_spec.quiet_output_dir = output_dir / "logs"  # type: ignore[attr-defined]

    if args.stage in {"base", "all"}:
        _run_specs(base_specs(), args=args, timestamp=timestamp, output_dir=output_dir, summaries=summaries, max_runs_state=state)

    single0_results: list[RunSummary] = []
    if args.stage in {"single0", "all"}:
        single0_results = _run_specs(single0_specs(), args=args, timestamp=timestamp, output_dir=output_dir, summaries=summaries, max_runs_state=state)

    single0_candidates = _promote(single0_results or summaries, "single0", args.promote_top)
    full_results: list[RunSummary] = []
    if args.stage in {"full", "all"}:
        full_results = _run_specs(full_specs(single0_candidates), args=args, timestamp=timestamp, output_dir=output_dir, summaries=summaries, max_runs_state=state)

    if args.stage in {"ir", "all"}:
        full_candidates = _promote(full_results or summaries, "full", 1)
        ir_results = _run_specs(ir_specs(full_candidates or single0_candidates[:1]), args=args, timestamp=timestamp, output_dir=output_dir, summaries=summaries, max_runs_state=state)
        ir_candidates = _promote(ir_results, "ir", 1)
        if ir_candidates:
            _run_specs(control_specs(ir_candidates), args=args, timestamp=timestamp, output_dir=output_dir, summaries=summaries, max_runs_state=state)

    _write_outputs(output_dir, summaries, args, timestamp)
    print(f"\nGlobal coupling ablation summary written to {output_dir}")


if __name__ == "__main__":
    main()
