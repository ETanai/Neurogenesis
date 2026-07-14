"""Compare fixed-size backpropagation and predictive coding on incremental MNIST.

This research extension reuses the exact seed-specific base checkpoints and
endpoint sizes of the confirmed conventional-learning controls.  Thus every
pair starts from identical weights; only incremental optimization differs.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
from omegaconf import open_dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_early_stop_ablation import _compose_cfg  # noqa: E402
from scripts.run_experiments import run  # noqa: E402
from scripts.run_organic_growth_ablation import BASE_OVERRIDES, summarize_result  # noqa: E402

DEFAULT_OUTPUT = ROOT / "outputs" / "predictive_coding" / "fixed_endpoint_comparison_lr1e4"
CONTROL_ROOT = ROOT / "outputs" / "ablations" / "organic_growth"
ORIGINAL_CONTROL = CONTROL_ROOT / "confirmation_controls_seeds42_51_v2" / "summary.json"
NO_REPLAY_CONTROL = (
    CONTROL_ROOT / "confirmation_cl_noreplay_seed_matched_correction" / "summary.json"
)
INCREMENTAL_ORDER = [0, 2, 3, 4, 5, 6, 8, 9]
PC_CONDITIONS = (
    "pc_uniform_original_data",
    "pc_usage_original_data",
    "pc_uniform_no_replay",
    "pc_usage_no_replay",
)


def _parse_csv(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one value is required")
    return values


def _control_rows() -> dict[tuple[str, int], dict[str, Any]]:
    original = json.loads(ORIGINAL_CONTROL.read_text(encoding="utf-8"))
    no_replay = json.loads(NO_REPLAY_CONTROL.read_text(encoding="utf-8"))
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for source in original:
        if source.get("status") == "completed" and source.get("name") == "cl_dataset_oracle_matched":
            rows[("bp_original_data", int(source["seed"]))] = source
    for source in no_replay:
        if source.get("status") == "completed" and source.get("name") == "cl_no_replay_seed_matched":
            rows[("bp_no_replay", int(source["seed"]))] = source
    return rows


def condition_spec(
    condition: str,
    seed: int,
    *,
    pc_weight_lr: float = 1.0e-4,
    inference_steps: int = 5,
    inference_lr: float = 0.05,
) -> dict[str, Any]:
    if condition not in PC_CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}")
    controls = _control_rows()
    replay_regime = "original_data" if condition.endswith("original_data") else "no_replay"
    baseline = controls[(f"bp_{replay_regime}", int(seed))]
    widths = [int(value) for value in baseline["final_widths"]]
    usage = condition.startswith("pc_usage")
    if replay_regime == "original_data":
        checkpoint = (
            CONTROL_ROOT
            / "confirmation_controls_seeds42_51_v2"
            / "base_checkpoints"
            / f"cl_207_sigmoid_seed_{seed}.pt"
        )
        regime_overrides = (
            "experiment.regime=cl_ir",
            "replay.enabled=true",
            "replay.mode=dataset",
        )
    else:
        checkpoint = (
            CONTROL_ROOT
            / "confirmation_cl_noreplay_seed_matched_correction"
            / "base_checkpoints"
            / f"seed_{seed}.pt"
        )
        regime_overrides = (
            "experiment.regime=cl",
            "replay.enabled=false",
            "replay.mode=dataset",
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    overrides = [
        *(value for value in BASE_OVERRIDES if not value.startswith("training.base_lr=")),
        "experiment.incremental_classes=[0,2,3,4,5,6,8,9]",
        "training.incremental_epochs=3",
        "model.activation=sigmoid",
        "model.activation_latent=identity",
        "experiment.control_hidden_sizes=" + str(widths).replace(" ", ""),
        *regime_overrides,
        "training.incremental_optimizer=predictive_coding",
        f"training.base_lr={pc_weight_lr}",
        f"training.predictive_coding.inference_steps={inference_steps}",
        f"training.predictive_coding.inference_lr={inference_lr}",
        "training.predictive_coding.plasticity_mode=" + ("usage" if usage else "uniform"),
        f"training.base_checkpoint={checkpoint.resolve().as_posix()}",
        "training.base_checkpoint_out=null",
        "logging.mlflow.enabled=false",
        f"seed={seed}",
    ]
    return {
        "condition": condition,
        "optimizer": "predictive_coding",
        "plasticity_mode": "usage" if usage else "uniform",
        "replay_regime": replay_regime,
        "seed": int(seed),
        "widths": widths,
        "pc_weight_lr": float(pc_weight_lr),
        "inference_steps": int(inference_steps),
        "inference_lr": float(inference_lr),
        "checkpoint": checkpoint.resolve(),
        "overrides": overrides,
    }


def _weight_displacement(model, checkpoint: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    initial = payload["state_dict"]
    totals: dict[str, float] = {}
    total_sq = 0.0
    with torch.no_grad():
        for name, final in model.state_dict().items():
            if name not in initial or not final.is_floating_point() or final.shape != initial[name].shape:
                continue
            squared = float((final.detach().cpu() - initial[name].detach().cpu()).square().sum().item())
            group = ".".join(name.split(".")[:2])
            totals[group] = totals.get(group, 0.0) + squared
            total_sq += squared
    return {
        "total_l2": math.sqrt(total_sq),
        "by_module_l2": {key: math.sqrt(value) for key, value in sorted(totals.items())},
    }


def _run_one(spec: dict[str, Any], output: Path, quiet: bool) -> dict[str, Any]:
    cfg = _compose_cfg(spec["overrides"])
    with open_dict(cfg):
        cfg.paper_experiment = f"predictive_coding/{spec['condition']}/seed_{spec['seed']}"
    started = time.perf_counter()
    log_path = output / "logs" / f"{spec['condition']}_seed_{spec['seed']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if quiet:
        with log_path.open("w", encoding="utf-8") as handle:
            with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
                result = run(cfg)
    else:
        result = run(cfg)
    summary = summarize_result(result)
    return {
        "condition": spec["condition"],
        "optimizer": spec["optimizer"],
        "plasticity_mode": spec["plasticity_mode"],
        "replay_regime": spec["replay_regime"],
        "seed": spec["seed"],
        "status": "completed",
        "runtime_seconds": time.perf_counter() - started,
        "base_checkpoint": str(spec["checkpoint"]),
        "overrides": spec["overrides"],
        "optimizer_diagnostics": result.get("optimizer_diagnostics", {}),
        "weight_displacement": _weight_displacement(result["model"], spec["checkpoint"]),
        "eval_records": result.get("eval_records", []),
        **summary,
    }


def _portable_control(condition: str, seed: int, source: dict[str, Any]) -> dict[str, Any]:
    return {
        "condition": condition,
        "optimizer": "backprop",
        "plasticity_mode": "uniform",
        "replay_regime": condition.removeprefix("bp_"),
        "seed": seed,
        "status": "completed",
        **{
            key: source.get(key)
            for key in (
                "final_widths", "macro_mse", "foreground_mse", "per_class_mse",
                "mean_positive_forgetting", "model_update_steps", "parameter_count",
                "runtime_seconds",
            )
        },
        "source": source.get("log_path"),
    }


def _mean_ci(values: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return {"mean": mean, "std": 0.0, "ci95_low": mean, "ci95_high": mean}
    # Student-t critical values for the planned n=3 screen and n=10 confirmation.
    critical = {2: 4.303, 9: 2.262}.get(len(values) - 1, 1.96)
    std = statistics.stdev(values)
    half = critical * std / math.sqrt(len(values))
    return {"mean": mean, "std": std, "ci95_low": mean - half, "ci95_high": mean + half}


def write_aggregate(rows: list[dict[str, Any]], output: Path, seeds: list[int]) -> list[dict[str, Any]]:
    controls = _control_rows()
    combined = [
        _portable_control(condition, seed, controls[(condition, seed)])
        for condition in ("bp_original_data", "bp_no_replay")
        for seed in seeds
    ] + [row for row in rows if row.get("status") == "completed" and int(row["seed"]) in seeds]
    order = [
        "bp_original_data", "pc_uniform_original_data", "pc_usage_original_data",
        "bp_no_replay", "pc_uniform_no_replay", "pc_usage_no_replay",
    ]
    aggregate = []
    for condition in order:
        selected = [row for row in combined if row["condition"] == condition]
        if not selected:
            continue
        aggregate.append(
            {
                "condition": condition,
                "seed_count": len(selected),
                "seeds": sorted(int(row["seed"]) for row in selected),
                "metrics": {
                    metric: _mean_ci([float(row[metric]) for row in selected])
                    for metric in ("macro_mse", "foreground_mse", "mean_positive_forgetting")
                },
                "per_class_mse": {
                    str(class_id): _mean_ci(
                        [float(row["per_class_mse"][str(class_id)]) for row in selected]
                    )
                    for class_id in [1, 7, *INCREMENTAL_ORDER]
                },
            }
        )
    (output / "comparison_rows.json").write_text(
        json.dumps(combined, indent=2) + "\n", encoding="utf-8"
    )
    (output / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "aggregate.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["condition", "seed_count", "metric", "mean", "std", "ci95_low", "ci95_high"],
        )
        writer.writeheader()
        for condition in aggregate:
            for metric, values in condition["metrics"].items():
                writer.writerow(
                    {"condition": condition["condition"], "seed_count": condition["seed_count"], "metric": metric, **values}
                )
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=_parse_csv, default=["42", "43", "44"])
    parser.add_argument("--conditions", type=_parse_csv, default=list(PC_CONDITIONS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pc-weight-lr", type=float, default=1.0e-4)
    parser.add_argument("--inference-steps", type=int, default=5)
    parser.add_argument("--inference-lr", type=float, default=0.05)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    seeds = [int(seed) for seed in args.seeds]
    invalid = sorted(set(args.conditions) - set(PC_CONDITIONS))
    if invalid:
        raise ValueError(f"Unknown conditions: {invalid}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    rows = json.loads(summary_path.read_text(encoding="utf-8")) if args.resume and summary_path.exists() else []
    completed = {
        (row["condition"], int(row["seed"]))
        for row in rows
        if row.get("status") == "completed"
    }
    for seed in seeds:
        for condition in args.conditions:
            if (condition, seed) in completed:
                continue
            spec = condition_spec(
                condition,
                seed,
                pc_weight_lr=args.pc_weight_lr,
                inference_steps=args.inference_steps,
                inference_lr=args.inference_lr,
            )
            identity = {key: value for key, value in spec.items() if key not in {"checkpoint"}}
            try:
                row = _run_one(spec, args.output_dir, args.quiet)
            except Exception as error:
                row = {**identity, "status": "failed", "error": repr(error)}
                rows.append(row)
                summary_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
                raise
            rows.append(row)
            summary_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
            write_aggregate(rows, args.output_dir, seeds)
    write_aggregate(rows, args.output_dir, seeds)


if __name__ == "__main__":
    main()
