"""Run staged ablations for optimizing the NDL training protocol."""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as _dt
import json
import math
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import open_dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_experiments import run  # noqa: E402


BASELINE_FIXED_AE = 0.009263875398039818
BASELINE_CLEAN_DATASET = 0.022396644577383995
BASELINE_IR = 0.03315376117825508
BASELINE_NO_REPLAY = 0.03889000415802002


@dataclass(frozen=True)
class AblationSpec:
    stage: str
    name: str
    description: str
    overrides: tuple[str, ...]


@dataclass
class RunSummary:
    stage: str
    name: str
    description: str
    run_name: str
    run_id: str | None
    overrides: list[str]
    final_l3_mse: float | None
    layer_sizes: list[int]
    per_class_l3_mse: dict[str, float]
    status: str

    def as_dict(self) -> dict:
        return {
            "stage": self.stage,
            "name": self.name,
            "description": self.description,
            "run_name": self.run_name,
            "run_id": self.run_id,
            "overrides": self.overrides,
            "final_l3_mse": self.final_l3_mse,
            "layer_sizes": self.layer_sizes,
            "per_class_l3_mse": self.per_class_l3_mse,
            "status": self.status,
        }


COMMON_OVERRIDES = (
    "data=mnist",
    "experiment=mnist_incremental",
    "experiment.regime=ndl_ir",
    "experiment.base_classes=[1,7]",
    "replay.enabled=true",
    "replay.mode=dataset",
    "neurogenesis.thresholds=null",
    "neurogenesis.next_layer_optimization=paper_columns",
)

SINGLE_0_OVERRIDES = (
    *COMMON_OVERRIDES,
    "experiment.incremental_classes=[0]",
)


def _default_tracking_uri() -> str:
    return f"sqlite:///{(REPO_ROOT / 'mlflow.db').resolve().as_posix()}"


def _compose_cfg(overrides: Sequence[str]):
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "config")):
        return compose(config_name="train", overrides=list(overrides))


def _sqlite_path_from_uri(uri: str) -> Path | None:
    if not uri.startswith("sqlite:///"):
        return None
    return Path(uri.removeprefix("sqlite:///"))


def _query_run_id(tracking_uri: str, run_name: str) -> str | None:
    db_path = _sqlite_path_from_uri(tracking_uri)
    if db_path is None or not db_path.exists():
        return None
    con = sqlite3.connect(db_path)
    try:
        row = con.execute(
            "select run_uuid from tags where key='mlflow.runName' and value=? "
            "order by rowid desc limit 1",
            (run_name,),
        ).fetchone()
        return str(row[0]) if row else None
    finally:
        con.close()


def _latest_metric(con: sqlite3.Connection, run_id: str, key: str) -> float | None:
    row = con.execute(
        "select value from metrics where run_uuid=? and key=? "
        "order by step desc, timestamp desc limit 1",
        (run_id, key),
    ).fetchone()
    if row is None:
        return None
    return float(row[0])


def _extract_mlflow_summary(tracking_uri: str, run_id: str | None) -> tuple[float | None, list[int], dict[str, float]]:
    if not run_id:
        return None, [], {}
    db_path = _sqlite_path_from_uri(tracking_uri)
    if db_path is None or not db_path.exists():
        return None, [], {}
    con = sqlite3.connect(db_path)
    try:
        final_l3 = _latest_metric(con, run_id, "metrics/val_mean_level_3")
        if final_l3 is None:
            final_l3 = _latest_metric(con, run_id, "val/mse")

        sizes: list[int] = []
        for level in range(4):
            value = _latest_metric(con, run_id, f"summary/layer_{level}_cumulative_size")
            if value is not None:
                sizes.append(int(round(value)))

        per_class: dict[str, float] = {}
        rows = con.execute(
            "select key, value from metrics where run_uuid=? "
            "and key like 'metrics/val_class_%_mean_level_3' "
            "order by step desc, timestamp desc",
            (run_id,),
        ).fetchall()
        for key, value in rows:
            cls = key.removeprefix("metrics/val_class_").removesuffix("_mean_level_3")
            if cls not in per_class:
                per_class[cls] = float(value)
        return final_l3, sizes, dict(sorted(per_class.items(), key=lambda item: int(item[0])))
    finally:
        con.close()


def _run_spec(
    spec: AblationSpec,
    *,
    tracking_uri: str,
    experiment_name: str,
    timestamp: str,
    dry_run: bool,
) -> RunSummary:
    run_name = f"ablation_{spec.name}_{timestamp}"
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
            run_name=run_name,
            run_id=None,
            overrides=list(spec.overrides),
            final_l3_mse=None,
            layer_sizes=[],
            per_class_l3_mse={},
            status="dry_run",
        )

    cfg = _compose_cfg(overrides)
    with open_dict(cfg):
        cfg.paper_experiment = f"training_protocol_ablation/{spec.stage}/{spec.name}"

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
    final_l3, sizes, per_class = _extract_mlflow_summary(tracking_uri, run_id)
    return RunSummary(
        stage=spec.stage,
        name=spec.name,
        description=spec.description,
        run_name=run_name,
        run_id=run_id,
        overrides=list(spec.overrides),
        final_l3_mse=final_l3,
        layer_sizes=sizes,
        per_class_l3_mse=per_class,
        status="completed",
    )


def _lr_length_specs() -> list[AblationSpec]:
    specs: list[AblationSpec] = []
    for lr in (1.0e-4, 3.0e-4, 1.0e-3):
        for plasticity_epochs in (500, 1000):
            for stability_epochs in (500, 1000):
                name = (
                    f"s1_lr_{lr:.0e}_p{plasticity_epochs}_s{stability_epochs}"
                    .replace("-", "m")
                    .replace("+", "")
                )
                specs.append(
                    AblationSpec(
                        "stage1_lr_length",
                        name,
                        "Single-0 gate: learning-rate and phase-length sweep.",
                        (
                            *SINGLE_0_OVERRIDES,
                            f"training.base_lr={lr}",
                            f"neurogenesis.plasticity_epochs={plasticity_epochs}",
                            f"neurogenesis.stability_epochs={stability_epochs}",
                            "neurogenesis.next_layer_epochs=500",
                        ),
                    )
                )
    return specs


def _stability_lr_specs(base: Sequence[str]) -> list[AblationSpec]:
    return [
        AblationSpec(
            "stage1_stability_lr",
            f"s1_stability_lr_{ratio}".replace(".", "_"),
            "Single-0 gate: stability LR ratio sweep from current best schedule.",
            (*base, f"neurogenesis.stability_lr_ratio={ratio}"),
        )
        for ratio in (0.01, 0.03, 0.1, 1.0)
    ]


def _decoder_lr_specs(base: Sequence[str]) -> list[AblationSpec]:
    return [
        AblationSpec(
            "stage1_decoder_lr",
            f"s1_decoder_lr_{ratio}".replace(".", "_"),
            "Single-0 gate: decoder LR ratio during plasticity.",
            (*base, f"neurogenesis.plasticity_decoder_lr_ratio={ratio}"),
        )
        for ratio in (0.01, 0.03, 0.1)
    ]


def _pretraining_specs(base: Sequence[str]) -> list[AblationSpec]:
    specs: list[AblationSpec] = []
    for pretrain_epochs in (14, 28, 50):
        for finetune_epochs in (0, 5, 10):
            specs.append(
                AblationSpec(
                    "stage1_pretraining",
                    f"s1_pretrain_{pretrain_epochs}_finetune_{finetune_epochs}",
                    "Single-0 gate: base pretraining and end-to-end finetune sweep.",
                    (
                        *base,
                        f"training.pretrain_epochs={pretrain_epochs}",
                        f"training.pretrain_finetune_epochs={finetune_epochs}",
                        "training.pretrain_mode=stacked_denoising",
                    ),
                )
            )
    return specs


def _weight_decay_specs(base: Sequence[str]) -> list[AblationSpec]:
    return [
        AblationSpec(
            "stage1_weight_decay",
            f"s1_weight_decay_{wd:.0e}".replace("-", "m"),
            "Single-0 gate: weight decay sweep from current best schedule.",
            (*base, f"training.weight_decay={wd}"),
        )
        for wd in (0.0, 1.0e-5, 1.0e-4, 1.0e-3)
    ]


def _to_full_mnist_overrides(single_overrides: Sequence[str]) -> tuple[str, ...]:
    drop_prefixes = (
        "experiment.incremental_classes=",
        "experiment.incremental_train_limit_per_class=",
    )
    return tuple(item for item in single_overrides if not item.startswith(drop_prefixes))


def _full_specs(promoted: Sequence[RunSummary]) -> list[AblationSpec]:
    specs: list[AblationSpec] = []
    for idx, summary in enumerate(promoted, start=1):
        specs.append(
            AblationSpec(
                "stage2_full_dataset",
                f"s2_full_{idx}_{summary.name}",
                f"Full MNIST clean dataset replay promotion from {summary.name}.",
                _to_full_mnist_overrides(summary.overrides),
            )
        )
    return specs


def _threshold_growth_specs(base: Sequence[str]) -> list[AblationSpec]:
    specs: list[AblationSpec] = []
    for percentile in (0.975, 0.985, 0.995):
        specs.append(
            AblationSpec(
                "stage3_threshold",
                f"s3_threshold_p{str(percentile).replace('.', '_')}",
                "Full MNIST threshold percentile polish.",
                (*base, f"experiment.threshold.percentile={percentile}", "neurogenesis.thresholds=null"),
            )
        )
    for factor_new in (0.005, 0.01, 0.02):
        for factor_max in (0.1, 0.2):
            specs.append(
                AblationSpec(
                    "stage3_growth",
                    f"s3_growth_new_{str(factor_new).replace('.', '_')}_max_{str(factor_max).replace('.', '_')}",
                    "Full MNIST growth policy polish.",
                    (
                        *base,
                        f"neurogenesis.factor_new_nodes={factor_new}",
                        f"neurogenesis.factor_max_new_nodes={factor_max}",
                    ),
                )
            )
    return specs


def _replay_composition_specs(base: Sequence[str]) -> list[AblationSpec]:
    return [
        AblationSpec(
            "stage4_replay",
            "s4_replay_ratio_1",
            "Replay composition: ratio mode, 1.0.",
            (*base, "neurogenesis.stability_replay_mode=ratio", "neurogenesis.stability_replay_ratio=1.0"),
        ),
        AblationSpec(
            "stage4_replay",
            "s4_replay_ratio_2",
            "Replay composition: ratio mode, 2.0.",
            (*base, "neurogenesis.stability_replay_mode=ratio", "neurogenesis.stability_replay_ratio=2.0"),
        ),
        AblationSpec(
            "stage4_replay",
            "s4_replay_paper_0_5",
            "Replay composition: paper mode, per-class ratio 0.5.",
            (*base, "neurogenesis.stability_replay_mode=paper", "neurogenesis.stability_replay_per_class_ratio=0.5"),
        ),
        AblationSpec(
            "stage4_replay",
            "s4_replay_paper_1",
            "Replay composition: paper mode, per-class ratio 1.0.",
            (*base, "neurogenesis.stability_replay_mode=paper", "neurogenesis.stability_replay_per_class_ratio=1.0"),
        ),
        AblationSpec(
            "stage4_replay",
            "s4_replay_balanced_4",
            "Replay composition: balanced mode, max ratio 4.0.",
            (*base, "neurogenesis.stability_replay_mode=balanced", "neurogenesis.stability_replay_balanced_max_ratio=4.0"),
        ),
    ]


def _ir_specs(base: Sequence[str]) -> list[AblationSpec]:
    intrinsic_base = tuple(
        "replay.mode=intrinsic" if item == "replay.mode=dataset" else item for item in base
    )
    return [
        AblationSpec(
            "stage5_ir",
            "s5_ir_gaussian_full",
            "Intrinsic replay retest with paper-like full covariance Gaussian sampling.",
            (
                *intrinsic_base,
                "replay.ir_sampling_mode=gaussian_full",
                "replay.ir_cov_shrinkage=0.0",
                "replay.ir_noise_scale=1.0",
            ),
        ),
        AblationSpec(
            "stage5_ir",
            "s5_ir_shrink_0_25",
            "Intrinsic replay covariance shrinkage 0.25.",
            (*intrinsic_base, "replay.ir_sampling_mode=gaussian_shrink", "replay.ir_cov_shrinkage=0.25"),
        ),
        AblationSpec(
            "stage5_ir",
            "s5_ir_shrink_0_5",
            "Intrinsic replay covariance shrinkage 0.5.",
            (*intrinsic_base, "replay.ir_sampling_mode=gaussian_shrink", "replay.ir_cov_shrinkage=0.5"),
        ),
        AblationSpec(
            "stage5_ir",
            "s5_ir_noise_0_75",
            "Intrinsic replay full Gaussian with reduced latent noise.",
            (*intrinsic_base, "replay.ir_sampling_mode=gaussian_full", "replay.ir_noise_scale=0.75"),
        ),
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
    return AblationSpec(spec.stage, spec.name, spec.description, (*spec.overrides, *extra))


def _filter_specs(specs: Sequence[AblationSpec], only: str | None) -> list[AblationSpec]:
    if not only:
        return list(specs)
    selectors = [item.strip() for item in only.split(",") if item.strip()]
    return [
        spec
        for spec in specs
        if any(spec.name == selector or spec.name.startswith(selector) or spec.stage == selector for selector in selectors)
    ]


def _best_completed(summaries: Iterable[RunSummary], *, limit: int = 1) -> list[RunSummary]:
    scored = [
        summary
        for summary in summaries
        if summary.final_l3_mse is not None and math.isfinite(float(summary.final_l3_mse))
    ]
    return sorted(scored, key=lambda item: float(item.final_l3_mse))[:limit]


def _write_outputs(output_dir: Path, summaries: Sequence[RunSummary], payload: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = {**payload, "runs": [summary.as_dict() for summary in summaries]}
    (output_dir / "summary.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "stage",
                "name",
                "run_id",
                "final_l3_mse",
                "layer_sizes",
                "status",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "stage": summary.stage,
                    "name": summary.name,
                    "run_id": summary.run_id or "",
                    "final_l3_mse": "" if summary.final_l3_mse is None else summary.final_l3_mse,
                    "layer_sizes": " ".join(map(str, summary.layer_sizes)),
                    "status": summary.status,
                }
            )

    lines = [
        "# Training Protocol Ablation Summary",
        "",
        f"- fixed AE ceiling: `{BASELINE_FIXED_AE:.5f}`",
        f"- best clean dataset baseline: `{BASELINE_CLEAN_DATASET:.5f}`",
        f"- best IR baseline: `{BASELINE_IR:.5f}`",
        "",
        "| Stage | Run | L3 MSE | Layer sizes | MLflow run |",
        "|---|---|---:|---|---|",
    ]
    for summary in summaries:
        mse = "" if summary.final_l3_mse is None else f"{summary.final_l3_mse:.5f}"
        sizes = " ".join(map(str, summary.layer_sizes))
        lines.append(f"| {summary.stage} | {summary.name} | {mse} | {sizes} | {summary.run_id or ''} |")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_promotions(path: Path) -> list[RunSummary]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summaries: list[RunSummary] = []
    for item in payload.get("runs", []):
        summaries.append(
            RunSummary(
                stage=item.get("stage", ""),
                name=item.get("name", ""),
                description=item.get("description", ""),
                run_name=item.get("run_name", ""),
                run_id=item.get("run_id"),
                overrides=list(item.get("overrides", [])),
                final_l3_mse=item.get("final_l3_mse"),
                layer_sizes=list(item.get("layer_sizes", [])),
                per_class_l3_mse=dict(item.get("per_class_l3_mse", {})),
                status=item.get("status", "loaded"),
            )
        )
    return _best_completed(summaries, limit=5)


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
    completed: list[RunSummary] = []
    extra = (*(_quick_overrides() if args.quick else ()), *tuple(args.override or ()))
    for spec in selected:
        if args.max_runs is not None and max_runs_state["count"] >= args.max_runs:
            break
        spec = _with_extra(spec, extra)
        summary = _run_spec(
            spec,
            tracking_uri=args.tracking_uri,
            experiment_name=args.experiment_name,
            timestamp=timestamp,
            dry_run=args.dry_run,
        )
        summaries.append(summary)
        completed.append(summary)
        max_runs_state["count"] += 1
        _write_outputs(
            output_dir,
            summaries,
            {
                "timestamp": timestamp,
                "tracking_uri": args.tracking_uri,
                "experiment_name": args.experiment_name,
                "quick": bool(args.quick),
                "dry_run": bool(args.dry_run),
            },
        )
    return completed


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
    parser.add_argument("--promote-from", type=Path, default=None)
    parser.add_argument("--promote-top", type=int, default=5)
    parser.add_argument("--force-later-stages", action="store_true")
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
        REPO_ROOT / "outputs" / "diagnostics" / "training_protocol_ablation" / timestamp
    )
    summaries: list[RunSummary] = []
    max_runs_state = {"count": 0}
    if args.quiet:
        _run_spec.quiet_output_dir = output_dir / "logs"  # type: ignore[attr-defined]

    promoted: list[RunSummary] = []
    if args.promote_from is not None:
        promoted = _load_promotions(args.promote_from)

    if args.stage in {"stage1", "all"} and not promoted:
        block = _run_specs(
            _lr_length_specs(),
            args=args,
            timestamp=timestamp,
            output_dir=output_dir,
            summaries=summaries,
            max_runs_state=max_runs_state,
        )
        best = _best_completed(block)
        if best:
            block = _run_specs(
                _stability_lr_specs(best[0].overrides),
                args=args,
                timestamp=timestamp,
                output_dir=output_dir,
                summaries=summaries,
                max_runs_state=max_runs_state,
            )
            best = _best_completed([*best, *block])
        if best:
            block = _run_specs(
                _decoder_lr_specs(best[0].overrides),
                args=args,
                timestamp=timestamp,
                output_dir=output_dir,
                summaries=summaries,
                max_runs_state=max_runs_state,
            )
            best = _best_completed([*best, *block])
        if best:
            block = _run_specs(
                _pretraining_specs(best[0].overrides),
                args=args,
                timestamp=timestamp,
                output_dir=output_dir,
                summaries=summaries,
                max_runs_state=max_runs_state,
            )
            best = _best_completed([*best, *block])
        if best:
            block = _run_specs(
                _weight_decay_specs(best[0].overrides),
                args=args,
                timestamp=timestamp,
                output_dir=output_dir,
                summaries=summaries,
                max_runs_state=max_runs_state,
            )
            best = _best_completed([*best, *block])
        promoted = _best_completed(summaries, limit=args.promote_top)

    if args.stage in {"stage2", "all"}:
        if not promoted:
            raise SystemExit("--stage stage2 requires --promote-from or completed stage1 results.")
        _run_specs(
            _full_specs(promoted[: args.promote_top]),
            args=args,
            timestamp=timestamp,
            output_dir=output_dir,
            summaries=summaries,
            max_runs_state=max_runs_state,
        )

    full_best = _best_completed(
        [summary for summary in summaries if summary.stage == "stage2_full_dataset"],
        limit=1,
    )
    if args.stage in {"stage3", "stage4", "stage5"} and promoted and not full_best:
        full_best = promoted[:1]

    improved_clean = bool(
        full_best
        and full_best[0].final_l3_mse is not None
        and float(full_best[0].final_l3_mse) < BASELINE_CLEAN_DATASET
    )

    if args.stage in {"stage3", "all"}:
        if not full_best:
            raise SystemExit("--stage stage3 requires --promote-from or completed stage2 results.")
        if improved_clean or args.force_later_stages or args.quick or args.dry_run:
            _run_specs(
                _threshold_growth_specs(full_best[0].overrides),
                args=args,
                timestamp=timestamp,
                output_dir=output_dir,
                summaries=summaries,
                max_runs_state=max_runs_state,
            )
        else:
            print("Skipping Stage 3: no promoted clean replay run beat the baseline.")

    stage3_or_full_best = _best_completed(
        [
            summary
            for summary in summaries
            if summary.stage in {"stage2_full_dataset", "stage3_threshold", "stage3_growth"}
        ],
        limit=1,
    ) or full_best

    if args.stage in {"stage4", "all"}:
        if not stage3_or_full_best:
            raise SystemExit("--stage stage4 requires a promoted clean replay schedule.")
        if improved_clean or args.force_later_stages or args.quick or args.dry_run:
            _run_specs(
                _replay_composition_specs(stage3_or_full_best[0].overrides),
                args=args,
                timestamp=timestamp,
                output_dir=output_dir,
                summaries=summaries,
                max_runs_state=max_runs_state,
            )
        else:
            print("Skipping Stage 4: no promoted clean replay run beat the baseline.")

    final_clean_best = _best_completed(
        [
            summary
            for summary in summaries
            if summary.stage
            in {"stage2_full_dataset", "stage3_threshold", "stage3_growth", "stage4_replay"}
        ],
        limit=1,
    ) or stage3_or_full_best

    if args.stage in {"stage5", "all"}:
        if not final_clean_best:
            raise SystemExit("--stage stage5 requires a promoted clean replay schedule.")
        if improved_clean or args.force_later_stages or args.quick or args.dry_run:
            _run_specs(
                _ir_specs(final_clean_best[0].overrides),
                args=args,
                timestamp=timestamp,
                output_dir=output_dir,
                summaries=summaries,
                max_runs_state=max_runs_state,
            )
        else:
            print("Skipping Stage 5: clean replay has not improved yet.")

    _write_outputs(
        output_dir,
        summaries,
        {
            "timestamp": timestamp,
            "tracking_uri": args.tracking_uri,
            "experiment_name": args.experiment_name,
            "quick": bool(args.quick),
            "dry_run": bool(args.dry_run),
            "baselines": {
                "fixed_ae": BASELINE_FIXED_AE,
                "clean_dataset": BASELINE_CLEAN_DATASET,
                "ir": BASELINE_IR,
                "no_replay": BASELINE_NO_REPLAY,
            },
        },
    )
    print(f"\nAblation summary written to {output_dir}")


if __name__ == "__main__":
    main()
