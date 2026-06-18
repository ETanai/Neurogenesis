"""Run staged early-stop ablations for NDL growth diagnostics."""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as _dt
import json
import math
import re
import sqlite3
import statistics as _stats
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


BASELINE_CLEAN_MSE = 0.011472370475530624
BASELINE_CLEAN_SIZES = [254, 236, 135, 160]
BASELINE_COMPACT_CLEAN_MSE = 0.01292346604168415
BASELINE_COMPACT_CLEAN_SIZES = [226, 148, 102, 160]
BASELINE_IR_MSE = 0.02020144648849964
BASELINE_IR_SIZES = [318, 258, 322, 180]
PAPER_TARGET_SIZES = [225, 135, 83, 40]
INITIAL_SIZES = [200, 100, 75, 20]
INCREMENTAL_CLASSES = [0, 2, 3, 4, 5, 6, 8, 9]


@dataclass(frozen=True)
class AblationSpec:
    stage: str
    name: str
    description: str
    overrides: tuple[str, ...]
    family: str


@dataclass
class RunSummary:
    stage: str
    name: str
    description: str
    family: str
    run_name: str
    run_id: str | None
    overrides: list[str]
    final_l3_mse: float | None
    layer_sizes: list[int]
    per_class_l3_mse: dict[str, float]
    diagnostics: dict
    score: float | None
    status: str

    def as_dict(self) -> dict:
        return {
            "stage": self.stage,
            "name": self.name,
            "description": self.description,
            "family": self.family,
            "run_name": self.run_name,
            "run_id": self.run_id,
            "overrides": self.overrides,
            "final_l3_mse": self.final_l3_mse,
            "layer_sizes": self.layer_sizes,
            "per_class_l3_mse": self.per_class_l3_mse,
            "diagnostics": self.diagnostics,
            "score": self.score,
            "status": self.status,
        }


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
)

SINGLE_0_DATASET_OVERRIDES = (
    *BASE_OVERRIDES,
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
    return None if row is None else float(row[0])


def _metric_series(con: sqlite3.Connection, run_id: str, key: str) -> list[float]:
    rows = con.execute(
        "select step, value from metrics where run_uuid=? and key=? order by step, timestamp",
        (run_id, key),
    ).fetchall()
    return [float(row[1]) for row in rows]


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else float(_stats.mean(values))


def _median(values: Sequence[float]) -> float | None:
    return None if not values else float(_stats.median(values))


def _extract_mlflow_summary(
    tracking_uri: str, run_id: str | None
) -> tuple[float | None, list[int], dict[str, float], dict, float | None]:
    if not run_id:
        return None, [], {}, {}, None
    db_path = _sqlite_path_from_uri(tracking_uri)
    if db_path is None or not db_path.exists():
        return None, [], {}, {}, None
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
        per_class = dict(sorted(per_class.items(), key=lambda item: int(item[0])))

        diagnostics = _extract_diagnostics(con, run_id)
        score = _score_run(final_l3, sizes, diagnostics)
        return final_l3, sizes, per_class, diagnostics, score
    finally:
        con.close()


def _extract_diagnostics(con: sqlite3.Connection, run_id: str) -> dict:
    present_classes = _present_growth_classes(con, run_id)
    growth_rounds: dict[str, dict[str, int]] = {}
    growth_increments_by_level: dict[str, list[float]] = {str(i): [] for i in range(4)}
    rounds_by_level = {str(i): [] for i in range(4)}
    for class_id in present_classes:
        growth_rounds[str(class_id)] = {}
        for level in range(4):
            seq = _metric_series(con, run_id, f"class_{class_id}/growth_level_{level}")
            compact: list[float] = []
            for value in seq:
                if not compact or value != compact[-1]:
                    compact.append(value)
            increments = [
                right - left for left, right in zip(compact, compact[1:]) if right > left
            ]
            growth_rounds[str(class_id)][str(level)] = len(increments)
            growth_increments_by_level[str(level)].extend(increments)
            rounds_by_level[str(level)].append(len(increments))

    cap_hits_by_level: dict[str, int] = {}
    outlier_fraction_by_level: dict[str, float | None] = {}
    for level in range(4):
        cap_hits = 0
        outlier_fractions: list[float] = []
        for class_id in present_classes:
            hit = _latest_metric(
                con, run_id, f"diagnostics/growth/class_{class_id}/level_{level}_hit_cap"
            )
            if hit:
                cap_hits += int(hit)
            frac = _latest_metric(
                con,
                run_id,
                f"diagnostics/outliers/class_{class_id}/level_{level}_final_fraction",
            )
            if frac is not None:
                outlier_fractions.append(frac)
        cap_hits_by_level[str(level)] = cap_hits
        outlier_fraction_by_level[str(level)] = _mean(outlier_fractions)

    phase_epoch_stats = _phase_epoch_stats(con, run_id)
    return {
        "classes_observed": present_classes,
        "growth_rounds_by_class_level": growth_rounds,
        "growth_rounds_by_level_mean": {
            level: _mean(values) for level, values in rounds_by_level.items()
        },
        "growth_rounds_by_level_median": {
            level: _median(values) for level, values in rounds_by_level.items()
        },
        "growth_increment_by_level_mean": {
            level: _mean(values) for level, values in growth_increments_by_level.items()
        },
        "growth_increment_by_level_median": {
            level: _median(values) for level, values in growth_increments_by_level.items()
        },
        "growth_increment_by_level_total": {
            level: float(sum(values)) for level, values in growth_increments_by_level.items()
        },
        "cap_hits_by_level": cap_hits_by_level,
        "final_outlier_fraction_by_level_mean": outlier_fraction_by_level,
        "phase_epoch_stats": phase_epoch_stats,
    }


def _present_growth_classes(con: sqlite3.Connection, run_id: str) -> list[int]:
    rows = con.execute(
        "select distinct key from metrics where run_uuid=? and key like 'class_%/growth_level_%'",
        (run_id,),
    ).fetchall()
    classes: set[int] = set()
    for (key,) in rows:
        match = re.match(r"class_(\d+)/growth_level_\d+$", key)
        if match:
            classes.add(int(match.group(1)))
    return sorted(classes) if classes else list(INCREMENTAL_CLASSES)


def _phase_epoch_stats(con: sqlite3.Connection, run_id: str) -> dict:
    stats: dict[str, dict[str, list[float]]] = {
        phase: {str(level): [] for level in range(4)}
        for phase in ("plasticity", "stability")
    }
    slopes: dict[str, dict[str, list[float]]] = {
        phase: {str(level): [] for level in range(4)}
        for phase in ("plasticity", "stability")
    }
    rows = con.execute("select distinct key from metrics where run_uuid=?", (run_id,)).fetchall()
    for (key,) in rows:
        if key.startswith("plasticity/"):
            match = re.match(r"plasticity/class_\d+/level_(\d+)/round_\d+_loss$", key)
            phase = "plasticity"
        elif key.startswith("stability/") and key.endswith("_eval_loss"):
            match = re.match(
                r"stability/class_\d+/level_(\d+)/round_\d+_eval_loss$", key
            )
            phase = "stability"
        else:
            continue
        if not match:
            continue
        level = match.group(1)
        values = _metric_series(con, run_id, key)
        if not values:
            continue
        stats[phase][level].append(len(values))
        diffs = [values[i - 1] - values[i] for i in range(1, len(values))]
        slopes[phase][level].append(sum(diffs[-5:]) if diffs else 0.0)

    result: dict[str, dict[str, dict[str, float | None]]] = {}
    for phase in ("plasticity", "stability"):
        result[phase] = {}
        for level in range(4):
            key = str(level)
            result[phase][key] = {
                "epoch_mean": _mean(stats[phase][key]),
                "epoch_median": _median(stats[phase][key]),
                "last5_improvement_mean": _mean(slopes[phase][key]),
                "last5_improvement_median": _median(slopes[phase][key]),
            }
    return result


def _score_run(final_l3: float | None, sizes: Sequence[int], diagnostics: dict) -> float | None:
    if final_l3 is None or not math.isfinite(float(final_l3)):
        return None
    rounds = diagnostics.get("growth_rounds_by_level_mean", {})
    outliers = diagnostics.get("final_outlier_fraction_by_level_mean", {})
    cap_hits = diagnostics.get("cap_hits_by_level", {})
    total_size = sum(sizes) if sizes else 0
    return float(final_l3) + (
        0.001 * float(rounds.get("2") or 0.0)
        + 0.0015 * float(rounds.get("3") or 0.0)
        + 0.01 * float(outliers.get("3") or 0.0)
        + 0.001 * float(cap_hits.get("3") or 0.0)
        + 0.00001 * float(total_size)
    )


def _run_spec(
    spec: AblationSpec,
    *,
    tracking_uri: str,
    experiment_name: str,
    timestamp: str,
    dry_run: bool,
) -> RunSummary:
    run_prefix = getattr(_run_spec, "run_prefix", "earlystop")
    run_name = f"{run_prefix}_{spec.name}_{timestamp}"
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
        cfg.paper_experiment = f"early_stop_ablation/{spec.stage}/{spec.name}"

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


def _stage1_specs() -> list[AblationSpec]:
    grid = [
        ("current", "1e-4", 2),
        ("global_md3e5_p3", "3e-5", 3),
        ("global_md1e5_p5", "1e-5", 5),
        ("global_md3e6_p5", "3e-6", 5),
        ("global_md1e6_p10", "1e-6", 10),
    ]
    return [
        AblationSpec(
            "stage1_global",
            name,
            "Single-0 clean replay: global min_delta/patience gate.",
            (
                *SINGLE_0_DATASET_OVERRIDES,
                f"neurogenesis.early_stop.min_delta={min_delta}",
                f"neurogenesis.early_stop.patience={patience}",
            ),
            "global",
        )
        for name, min_delta, patience in grid
    ]


def _stage2_specs(base: Sequence[str]) -> list[AblationSpec]:
    return [
        AblationSpec(
            "stage2_goal",
            "current_goal",
            "Single-0 clean replay: current phase goal factors.",
            (
                *base,
                "neurogenesis.early_stop.use_threshold_goal=true",
                "neurogenesis.early_stop.threshold_goal_factor_plasticity=0.9",
                "neurogenesis.early_stop.threshold_goal_factor_stability=0.7",
            ),
            "global",
        ),
        AblationSpec(
            "stage2_goal",
            "stricter_stability",
            "Single-0 clean replay: stricter stability goal.",
            (
                *base,
                "neurogenesis.early_stop.use_threshold_goal=true",
                "neurogenesis.early_stop.threshold_goal_factor_plasticity=0.9",
                "neurogenesis.early_stop.threshold_goal_factor_stability=0.5",
            ),
            "global",
        ),
        AblationSpec(
            "stage2_goal",
            "stricter_both",
            "Single-0 clean replay: stricter plasticity and stability goals.",
            (
                *base,
                "neurogenesis.early_stop.use_threshold_goal=true",
                "neurogenesis.early_stop.threshold_goal_factor_plasticity=0.8",
                "neurogenesis.early_stop.threshold_goal_factor_stability=0.5",
            ),
            "global",
        ),
        AblationSpec(
            "stage2_goal",
            "very_strict_stability",
            "Single-0 clean replay: very strict stability goal.",
            (
                *base,
                "neurogenesis.early_stop.use_threshold_goal=true",
                "neurogenesis.early_stop.threshold_goal_factor_plasticity=0.9",
                "neurogenesis.early_stop.threshold_goal_factor_stability=0.3",
            ),
            "global",
        ),
        AblationSpec(
            "stage2_goal",
            "no_goal_stop",
            "Single-0 clean replay: disable threshold-goal early stop.",
            (*base, "neurogenesis.early_stop.use_threshold_goal=false"),
            "global",
        ),
    ]


def _stage3_specs(base: Sequence[str]) -> list[AblationSpec]:
    common = (
        *base,
        "+neurogenesis.early_stop_by_level.0.min_delta=1e-4",
        "+neurogenesis.early_stop_by_level.0.patience=2",
        "+neurogenesis.early_stop_by_level.1.min_delta=3e-5",
        "+neurogenesis.early_stop_by_level.1.patience=3",
        "+neurogenesis.early_stop_by_level.2.min_delta=1e-5",
        "+neurogenesis.early_stop_by_level.2.patience=5",
        "+neurogenesis.early_stop_by_level.3.min_delta=3e-6",
        "+neurogenesis.early_stop_by_level.3.patience=10",
    )
    return [
        AblationSpec(
            "stage3_layer",
            "layer_specific",
            "Single-0 clean replay: layer-specific early-stop policy.",
            common,
            "phase_layer",
        ),
        AblationSpec(
            "stage3_layer",
            "layer_specific_l3_strong",
            "Single-0 clean replay: stronger top-layer patience.",
            (
                *common,
                "neurogenesis.early_stop_by_level.3.min_delta=1e-6",
                "neurogenesis.early_stop_by_level.3.patience=10",
            ),
            "phase_layer",
        ),
    ]


def _with_intrinsic_single0(overrides: Sequence[str]) -> tuple[str, ...]:
    converted = [
        "replay.mode=intrinsic" if item == "replay.mode=dataset" else item
        for item in overrides
    ]
    return (
        *converted,
        "replay.ir_sampling_mode=gaussian_shrink",
        "replay.ir_cov_shrinkage=0.25",
    )


def _to_full(overrides: Sequence[str], *, replay_mode: str) -> tuple[str, ...]:
    drop_prefixes = ("experiment.incremental_classes=",)
    converted = []
    for item in overrides:
        if item.startswith(drop_prefixes):
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


def _stage4_specs(promoted: Sequence[RunSummary]) -> list[AblationSpec]:
    specs: list[AblationSpec] = []
    for idx, summary in enumerate(promoted, start=1):
        specs.append(
            AblationSpec(
                "stage4_ir_single0",
                f"ir_single0_{idx}_{summary.name}",
                f"Single-0 IR confirmation from {summary.name}.",
                _with_intrinsic_single0(summary.overrides),
                summary.family,
            )
        )
    return specs


def _stage5_specs(promoted: Sequence[RunSummary]) -> list[AblationSpec]:
    specs: list[AblationSpec] = []
    seen: set[tuple[str, str]] = set()
    for idx, summary in enumerate(promoted, start=1):
        for replay_mode in ("dataset", "intrinsic"):
            key = (summary.name, replay_mode)
            if key in seen:
                continue
            seen.add(key)
            specs.append(
                AblationSpec(
                    "stage5_full",
                    f"full_{replay_mode}_{idx}_{summary.name}",
                    f"Full MNIST {replay_mode} replay promotion from {summary.name}.",
                    _to_full(summary.overrides, replay_mode=replay_mode),
                    summary.family,
                )
            )
    return specs


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


def _best_completed(
    summaries: Iterable[RunSummary], *, limit: int = 1, family: str | None = None
) -> list[RunSummary]:
    scored = [
        summary
        for summary in summaries
        if summary.score is not None
        and math.isfinite(float(summary.score))
        and (family is None or summary.family == family)
    ]
    return sorted(scored, key=lambda item: float(item.score))[:limit]


def _write_outputs(output_dir: Path, summaries: Sequence[RunSummary], payload: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = {**payload, "runs": [summary.as_dict() for summary in summaries]}
    (output_dir / "summary.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

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
                    "final_l3_mse": summary.final_l3_mse if summary.final_l3_mse is not None else "",
                    "score": summary.score if summary.score is not None else "",
                    "layer_sizes": " ".join(map(str, summary.layer_sizes)),
                    "l2_rounds_mean": diagnostics.get("growth_rounds_by_level_mean", {}).get("2", ""),
                    "l3_rounds_mean": diagnostics.get("growth_rounds_by_level_mean", {}).get("3", ""),
                    "l3_cap_hits": diagnostics.get("cap_hits_by_level", {}).get("3", ""),
                    "l3_outlier_fraction": diagnostics.get("final_outlier_fraction_by_level_mean", {}).get("3", ""),
                    "status": summary.status,
                }
            )

    lines = [
        "# Early-Stop Ablation Summary",
        "",
        f"- best clean baseline: `{BASELINE_CLEAN_MSE:.5f}` sizes `{BASELINE_CLEAN_SIZES}`",
        f"- compact clean baseline: `{BASELINE_COMPACT_CLEAN_MSE:.5f}` sizes `{BASELINE_COMPACT_CLEAN_SIZES}`",
        f"- best IR baseline: `{BASELINE_IR_MSE:.5f}` sizes `{BASELINE_IR_SIZES}`",
        f"- paper target sizes: `{PAPER_TARGET_SIZES}`",
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


def _fmt(value) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.4f}"


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
        _write_outputs(output_dir, summaries, _payload(args, timestamp))
    return completed


def _payload(args: argparse.Namespace, timestamp: str) -> dict:
    return {
        "timestamp": timestamp,
        "tracking_uri": args.tracking_uri,
        "experiment_name": args.experiment_name,
        "quick": bool(args.quick),
        "dry_run": bool(args.dry_run),
        "score": {
            "formula": "mse + 0.001*l2_rounds + 0.0015*l3_rounds + 0.01*l3_outlier + 0.001*l3_cap_hits + 0.00001*total_size",
        },
        "baselines": {
            "best_clean": {"mse": BASELINE_CLEAN_MSE, "sizes": BASELINE_CLEAN_SIZES},
            "compact_clean": {
                "mse": BASELINE_COMPACT_CLEAN_MSE,
                "sizes": BASELINE_COMPACT_CLEAN_SIZES,
            },
            "best_ir": {"mse": BASELINE_IR_MSE, "sizes": BASELINE_IR_SIZES},
            "paper_target_sizes": PAPER_TARGET_SIZES,
        },
    }


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
    parser.add_argument("--promote-top", type=int, default=2)
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
        REPO_ROOT / "outputs" / "diagnostics" / "early_stop_ablation" / timestamp
    )
    summaries: list[RunSummary] = []
    max_runs_state = {"count": 0}
    if args.quiet:
        _run_spec.quiet_output_dir = output_dir / "logs"  # type: ignore[attr-defined]

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

    best_global = _best_completed(stage1 or summaries, family="global")
    if args.stage in {"stage2", "all"}:
        if not best_global and not args.dry_run:
            raise SystemExit("--stage stage2 requires completed stage1 global results.")
        base = best_global[0].overrides if best_global else _stage1_specs()[2].overrides
        stage2 = _run_specs(
            _stage2_specs(base),
            args=args,
            timestamp=timestamp,
            output_dir=output_dir,
            summaries=summaries,
            max_runs_state=max_runs_state,
        )
        best_global = _best_completed([*(best_global or []), *stage2], family="global")

    if args.stage in {"stage3", "all"}:
        if not best_global and not args.dry_run:
            raise SystemExit("--stage stage3 requires completed global results.")
        base = best_global[0].overrides if best_global else _stage1_specs()[2].overrides
        _run_specs(
            _stage3_specs(base),
            args=args,
            timestamp=timestamp,
            output_dir=output_dir,
            summaries=summaries,
            max_runs_state=max_runs_state,
        )

    single0_candidates = _best_completed(
        [
            summary
            for summary in summaries
            if summary.stage in {"stage1_global", "stage2_goal", "stage3_layer"}
        ],
        limit=max(args.promote_top, 2),
    )

    if args.stage in {"stage4", "all"}:
        if not single0_candidates and not args.dry_run:
            raise SystemExit("--stage stage4 requires completed single-0 candidates.")
        promoted = single0_candidates or [
            RunSummary(
                "stage1_global",
                "global_md1e5_p5",
                "",
                "global",
                "",
                None,
                list(_stage1_specs()[2].overrides),
                None,
                [],
                {},
                {},
                None,
                "dry_seed",
            )
        ]
        _run_specs(
            _stage4_specs(promoted[: args.promote_top]),
            args=args,
            timestamp=timestamp,
            output_dir=output_dir,
            summaries=summaries,
            max_runs_state=max_runs_state,
        )

    if args.stage in {"stage5", "all"}:
        global_best = _best_completed(single0_candidates, family="global", limit=1)
        layer_best = _best_completed(single0_candidates, family="phase_layer", limit=1)
        promoted = [*global_best, *layer_best]
        if not promoted and not args.dry_run:
            raise SystemExit("--stage stage5 requires global and layer candidates.")
        if not promoted:
            promoted = [
                RunSummary(
                    "stage1_global",
                    "global_md1e5_p5",
                    "",
                    "global",
                    "",
                    None,
                    list(_stage1_specs()[2].overrides),
                    None,
                    [],
                    {},
                    {},
                    None,
                    "dry_seed",
                )
            ]
        _run_specs(
            _stage5_specs(promoted[: args.promote_top]),
            args=args,
            timestamp=timestamp,
            output_dir=output_dir,
            summaries=summaries,
            max_runs_state=max_runs_state,
        )

    _write_outputs(output_dir, summaries, _payload(args, timestamp))
    print(f"\nEarly-stop ablation summary written to {output_dir}")


if __name__ == "__main__":
    main()
