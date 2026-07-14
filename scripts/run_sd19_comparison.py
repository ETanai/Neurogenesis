"""Run the staged, paired SD19 neurogenesis comparison.

The feasibility screen grows an NDL autoencoder first and then trains a
conventional learner at the exact seed-specific endpoint.  Both conditions use
original old-class images, so failure cannot be attributed to intrinsic replay
quality.  The full stage is intentionally blocked unless the screen gate passes
or ``--force-full`` is supplied.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from omegaconf import open_dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_early_stop_ablation import _compose_cfg  # noqa: E402
from scripts.run_experiments import run  # noqa: E402


INITIAL_WIDTHS = [1000, 500, 250, 50]
BASE_CLASSES = list(range(10))
SCREEN_CLASSES = [10, 15, 22, 35, 36, 61]  # A, F, M, Z, a, z
FULL_CLASSES = list(range(10, 62))

COMMON_OVERRIDES = (
    "data=sd19",
    "experiment=sd19_incremental",
    "experiment.base_classes=[0,1,2,3,4,5,6,7,8,9]",
    "data.offline_resize=false",
    "data.resize_progress_bar=false",
    "data.num_workers=4",
    "data.batch_size=128",
    "model.hidden_sizes=[1000,500,250,50]",
    "model.activation=sigmoid",
    "model.activation_latent=identity",
    "training.base_lr=0.001",
    "training.pretrain_mode=stacked_denoising",
    "training.denoising_dropout=0.1",
    "training.denoising_std=0.0",
    "training.pretrain_finetune_epochs=0",
    "training.incremental_epochs=3",
    "neurogenesis.thresholds=null",
    "neurogenesis.objective_mode=paper_level_ae",
    "neurogenesis.next_layer_optimization=paper_columns",
    "neurogenesis.early_stop=null",
    "neurogenesis.shape_pressure_mode=none",
    "neurogenesis.max_outlier_fraction=0.1",
    "neurogenesis.unresolved_outlier_action=record",
    "neurogenesis.plasticity_epochs=3",
    "neurogenesis.stability_epochs=11",
    "neurogenesis.next_layer_epochs=3",
    "neurogenesis.plasticity_decoder_lr_ratio=0.01",
    "neurogenesis.stability_lr_ratio=0.01",
    "neurogenesis.next_layer_lr_ratio=0.01",
    "neurogenesis.stability_replay_mode=paper",
    "neurogenesis.stability_replay_per_class_ratio=1.0",
    "neurogenesis.stability_schedule=mixed",
    "neurogenesis.max_nodes=[100,100,100,20]",
    "neurogenesis.max_nodes_scope=global",
    "neurogenesis.growth_mode=proportional",
    "neurogenesis.factor_new_nodes=0.01",
    "neurogenesis.max_nodes_per_class=null",
    "neurogenesis.max_nodes_stream=null",
    "replay.enabled=true",
    "replay.mode=dataset",
    "replay.sampling_mode=paper",
    "replay.per_class_batch_ratio=1.0",
    "logging.mlflow.enabled=false",
)


def _class_override(classes: Sequence[int]) -> str:
    return "experiment.incremental_classes=" + str(list(classes)).replace(" ", "")


def _final_records(result: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = result.get("eval_records", [])
    if not records:
        return {}, []
    final_step = max(int(row["step"]) for row in records)
    top_layer = len(result["model"].hidden_sizes) - 1
    final = [
        row
        for row in records
        if int(row.get("step", -1)) == final_step
        and int(row.get("layer", -1)) == top_layer
    ]
    aggregate = next((row for row in final if row.get("scope") == "aggregate"), {})
    per_class = [row for row in final if row.get("scope") == "class"]
    return aggregate, per_class


def summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    model = result["model"]
    aggregate, per_class = _final_records(result)
    top_layer = len(model.hidden_sizes) - 1
    histories: dict[int, list[float]] = {}
    for row in result.get("eval_records", []):
        if row.get("scope") != "class" or int(row.get("layer", -1)) != top_layer:
            continue
        histories.setdefault(int(row["class_id"]), []).append(float(row["mean"]))
    forgetting = [
        max(values[-1] - min(values), 0.0) for values in histories.values() if values
    ]
    reports = list(result.get("growth_reports", {}).values())
    levels = [level for report in reports for level in report.get("levels", [])]
    stats = result.get("training_stats", {})
    widths = [int(value) for value in model.hidden_sizes]
    return {
        "final_widths": widths,
        "added_widths": [value - initial for value, initial in zip(widths, INITIAL_WIDTHS)],
        "macro_mse": aggregate.get("mean"),
        "foreground_mse": aggregate.get("foreground_mse"),
        "mean_positive_forgetting": statistics.fmean(forgetting) if forgetting else 0.0,
        "per_class_mse": {
            str(row["class_id"]): float(row["mean"]) for row in per_class
        },
        "quota_stop_fraction": (
            sum(level.get("stop_reason") == "outlier_quota_reached" for level in levels)
            / len(levels)
            if levels
            else 0.0
        ),
        "unresolved_exhausted_level_count": sum(
            bool(level.get("unresolved_outliers")) and bool(level.get("budget_exhausted"))
            for level in levels
        ),
        "updated_class_fraction": (
            sum(int(report.get("model_update_steps", 0)) > 0 for report in reports)
            / len(reports)
            if reports
            else 0.0
        ),
        "model_update_steps": int(
            stats.get("neurogenesis_parameter_updates", 0)
            or stats.get("incremental_parameter_updates", 0)
        ),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "growth_reports": result.get("growth_reports", {}),
    }


def evaluate_gate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "completed"]
    by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for row in completed:
        by_seed.setdefault(int(row["seed"]), {})[str(row["condition"])] = row
    pairs = [pair for pair in by_seed.values() if {"ndl_dataset", "cl_dataset"} <= set(pair)]
    ratios = [
        float(pair["ndl_dataset"]["macro_mse"]) / float(pair["cl_dataset"]["macro_mse"])
        for pair in pairs
    ]
    forgetting_deltas = [
        float(pair["ndl_dataset"]["mean_positive_forgetting"])
        - float(pair["cl_dataset"]["mean_positive_forgetting"])
        for pair in pairs
    ]
    ndl_wins = sum(value < 1.0 for value in ratios)
    finite = all(math.isfinite(value) for value in [*ratios, *forgetting_deltas])
    passed = bool(
        len(pairs) >= 3
        and finite
        and statistics.fmean(ratios) <= 0.95
        and ndl_wins >= math.ceil(2 * len(pairs) / 3)
        and statistics.fmean(forgetting_deltas) <= 0.0
    )
    return {
        "passed": passed,
        "pair_count": len(pairs),
        "mean_macro_mse_ratio_ndl_over_cl": statistics.fmean(ratios) if ratios else None,
        "ndl_seed_win_count": ndl_wins,
        "mean_forgetting_delta_ndl_minus_cl": (
            statistics.fmean(forgetting_deltas) if forgetting_deltas else None
        ),
        "criteria": {
            "minimum_pairs": 3,
            "mean_macro_mse_ratio_max": 0.95,
            "minimum_ndl_win_fraction": 2 / 3,
            "mean_forgetting_delta_max": 0.0,
        },
    }


def _write(output_dir: Path, rows: list[dict[str, Any]], gate: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    (output_dir / "gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    columns = [
        "stage", "condition", "seed", "status", "macro_mse", "foreground_mse",
        "mean_positive_forgetting", "final_widths", "quota_stop_fraction",
        "updated_class_fraction", "unresolved_exhausted_level_count",
        "model_update_steps", "parameter_count", "runtime_seconds", "error",
    ]
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            item = dict(row)
            item["final_widths"] = " ".join(map(str, row.get("final_widths", [])))
            writer.writerow(item)
    lines = [
        "# SD19 paired comparison", "",
        f"Promotion gate: **{'PASS' if gate.get('passed') else 'FAIL'}**", "",
        "| Stage | Condition | Seed | Macro MSE | Foreground MSE | Forgetting | Widths | Updates | Status |",
        "|---|---|---:|---:|---:|---:|---|---:|---|",
    ]
    for row in rows:
        def fmt(key: str) -> str:
            value = row.get(key)
            return "" if value is None else f"{float(value):.6f}"
        lines.append(
            f"| {row['stage']} | {row['condition']} | {row['seed']} | "
            f"{fmt('macro_mse')} | {fmt('foreground_mse')} | "
            f"{fmt('mean_positive_forgetting')} | {row.get('final_widths', '')} | "
            f"{row.get('model_update_steps', '')} | {row['status']} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metadata(stage: str, classes: Sequence[int], seeds: Sequence[int]) -> dict[str, Any]:
    return {
        "stage": stage,
        "base_classes": BASE_CLASSES,
        "incremental_classes": list(classes),
        "seeds": list(seeds),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "screen_sample_limits": {"train_per_class": 512, "validation_per_class": 128},
    }


def run_stage(
    *, stage: str, seeds: Sequence[int], output_dir: Path, quiet: bool, resume: bool
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    classes = SCREEN_CLASSES if stage == "screen" else FULL_CLASSES
    summary_path = output_dir / "summary.json"
    rows: list[dict[str, Any]] = []
    if resume and summary_path.exists():
        rows = json.loads(summary_path.read_text(encoding="utf-8"))
    completed = {
        (str(row.get("condition")), int(row.get("seed", -1)))
        for row in rows if row.get("status") == "completed"
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(
        json.dumps(_metadata(stage, classes, seeds), indent=2) + "\n", encoding="utf-8"
    )
    sample_overrides = (
        "data.default_per_class_limit_train=512",
        "data.default_per_class_limit_val=128",
        "training.pretrain_epochs=20",
    ) if stage == "screen" else (
        "data.default_per_class_limit_train=null",
        "data.default_per_class_limit_val=null",
        "training.pretrain_epochs=50",
    )
    for seed in seeds:
        endpoint: list[int] | None = None
        existing_ndl = next(
            (row for row in rows if row.get("condition") == "ndl_dataset" and int(row.get("seed", -1)) == seed and row.get("status") == "completed"),
            None,
        )
        if existing_ndl is not None:
            endpoint = [int(value) for value in existing_ndl["final_widths"]]
        for condition in ("ndl_dataset", "cl_dataset"):
            if (condition, int(seed)) in completed:
                continue
            if condition == "cl_dataset" and endpoint is None:
                raise RuntimeError(f"Seed {seed} CL control requires its completed NDL endpoint")
            regime_overrides = (
                ("experiment.regime=ndl_ir",)
                if condition == "ndl_dataset"
                else (
                    "experiment.regime=cl_ir",
                    "experiment.control_hidden_sizes=" + str(endpoint).replace(" ", ""),
                )
            )
            overrides = (
                *COMMON_OVERRIDES,
                _class_override(classes),
                *sample_overrides,
                *regime_overrides,
                f"seed={seed}",
            )
            identity = {
                "stage": stage,
                "condition": condition,
                "seed": int(seed),
                "overrides": list(overrides),
            }
            print(f"\n=== SD19 {stage}/{condition}/seed_{seed} ===")
            cfg = _compose_cfg(overrides)
            with open_dict(cfg):
                cfg.paper_experiment = f"sd19_comparison/{stage}/{condition}/seed_{seed}"
            try:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                if quiet:
                    log_dir = output_dir / "logs"
                    log_dir.mkdir(parents=True, exist_ok=True)
                    log_path = log_dir / f"{condition}_seed_{seed}.log"
                    with log_path.open("w", encoding="utf-8") as handle:
                        with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
                            result = run(cfg)
                    identity["log_path"] = str(log_path)
                else:
                    result = run(cfg)
                summary = summarize_result(result)
                summary["runtime_seconds"] = time.perf_counter() - started
                summary["peak_cuda_bytes"] = (
                    int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
                )
                row = {**identity, "status": "completed", **summary}
                rows = [
                    old for old in rows
                    if not (old.get("condition") == condition and int(old.get("seed", -1)) == seed)
                ]
                rows.append(row)
                if condition == "ndl_dataset":
                    endpoint = list(summary["final_widths"])
            except Exception as exc:
                rows.append({**identity, "status": "failed", "error": repr(exc)})
                gate = evaluate_gate(rows)
                _write(output_dir, rows, gate)
                raise
            gate = evaluate_gate(rows)
            _write(output_dir, rows, gate)
    gate = evaluate_gate(rows)
    _write(output_dir, rows, gate)
    return rows, gate


def _parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required")
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("screen", "full"), default="screen")
    parser.add_argument("--seeds", type=_parse_seeds, default=[42, 43, 44])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--screen-dir", type=Path, default=REPO_ROOT / "outputs" / "sd19" / "feasibility_screen")
    parser.add_argument("--force-full", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "full" and not args.force_full:
        gate_path = args.screen_dir / "gate.json"
        if not gate_path.exists() or not bool(json.loads(gate_path.read_text())["passed"]):
            raise SystemExit(
                "SD19 feasibility gate has not passed; use --force-full only for an explicitly non-promoted run."
            )
    output_dir = args.output_dir or (
        REPO_ROOT / "outputs" / "sd19" / ("feasibility_screen" if args.stage == "screen" else "full_comparison")
    )
    _, gate = run_stage(
        stage=args.stage,
        seeds=args.seeds,
        output_dir=output_dir,
        quiet=args.quiet,
        resume=args.resume,
    )
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
