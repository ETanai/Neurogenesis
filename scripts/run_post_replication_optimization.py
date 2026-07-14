"""Gated optimization screens for NDL and fixed-size predictive coding.

The frozen replication remains untouched.  Screen candidates use seeds 45-47,
one incoming class, 512 training examples, and identical seed-specific base
checkpoints.  A candidate must improve macro MSE by at least 10% without adding
more than 0.002 absolute forgetting before it can enter the full curriculum.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from omegaconf import open_dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_early_stop_ablation import _compose_cfg  # noqa: E402
from scripts.run_experiments import run  # noqa: E402
from scripts.run_organic_growth_ablation import BASE_OVERRIDES, summarize_result  # noqa: E402

RUN_ROOT = ROOT / "outputs" / "ablations" / "organic_growth"
DEFAULT_OUTPUT = ROOT / "outputs" / "optimization" / "post_replication_gated"
DEV_SEEDS = [45, 46, 47]


@dataclass(frozen=True)
class OptimizationSpec:
    family: str
    name: str
    description: str
    overrides: tuple[str, ...]
    baseline: str


def _parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required")
    return seeds


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _control_row(seed: int) -> dict[str, Any]:
    rows = _load_rows(RUN_ROOT / "confirmation_cl_noreplay_seed_matched_correction" / "summary.json")
    matches = [row for row in rows if row.get("status") == "completed" and int(row["seed"]) == seed]
    if len(matches) != 1:
        raise ValueError(f"Expected one no-replay CL control for seed {seed}")
    return matches[0]


def _ndl_source(seed: int) -> dict[str, Any]:
    sources = [
        RUN_ROOT / "full_threshold_refresh_seeds42_46" / "summary.json",
        RUN_ROOT / "confirmation_ndl_seeds47_51" / "summary.json",
    ]
    matches = []
    for path in sources:
        matches.extend(
            row
            for row in _load_rows(path)
            if row.get("status") == "completed"
            and int(row["seed"]) == seed
            and row.get("name") in {"organic_c_abs1_class_small", "ndl_dataset_oracle_refresh"}
        )
    if len(matches) != 1:
        raise ValueError(f"Expected one clean NDL source for seed {seed}, found {len(matches)}")
    return matches[0]


def _checkpoint(family: str, seed: int) -> Path:
    if family == "predictive_coding":
        return (
            RUN_ROOT
            / "confirmation_cl_noreplay_seed_matched_correction"
            / "base_checkpoints"
            / f"seed_{seed}.pt"
        ).resolve()
    candidates = [
        RUN_ROOT / "full_threshold_refresh_seeds42_46" / "base_checkpoints" / f"seed_{seed}.pt",
        RUN_ROOT / "confirmation_ndl_seeds47_51" / "base_checkpoints" / f"ndl_200_sigmoid_seed_{seed}.pt",
    ]
    path = next((candidate for candidate in candidates if candidate.is_file()), candidates[-1])
    return path.resolve()


def _replace(overrides: Sequence[str], key: str, value: object) -> tuple[str, ...]:
    prefix = f"{key}="
    return (*[item for item in overrides if not item.startswith(prefix)], f"{key}={value}")


def pc_specs(seed: int, *, full: bool = False) -> list[OptimizationSpec]:
    control = _control_row(seed)
    widths = str([int(value) for value in control["final_widths"]]).replace(" ", "")
    checkpoint = _checkpoint("predictive_coding", seed)
    classes = "[0,2,3,4,5,6,8,9]" if full else "[0]"
    common = tuple(value for value in BASE_OVERRIDES if not value.startswith("training.base_lr="))
    common = (
        *common,
        "experiment.regime=cl",
        "replay.enabled=false",
        "model.activation=sigmoid",
        "model.activation_latent=identity",
        f"experiment.control_hidden_sizes={widths}",
        f"experiment.incremental_classes={classes}",
        "experiment.incremental_train_limit_per_class=null" if full else "experiment.incremental_train_limit_per_class=512",
        "training.incremental_epochs=3",
        "training.incremental_optimizer=predictive_coding",
        "training.base_lr=0.0001",
        "training.predictive_coding.inference_steps=5",
        "training.predictive_coding.inference_lr=0.05",
        "training.predictive_coding.plasticity_mode=uniform",
        f"training.base_checkpoint={checkpoint.as_posix()}",
        "training.base_checkpoint_out=null",
        "logging.mlflow.enabled=false",
        f"seed={seed}",
    )

    def make(name: str, description: str, *extra: str) -> OptimizationSpec:
        return OptimizationSpec(
            "predictive_coding", name, description, (*common, *extra), "pc_local"
        )

    return [
        make(
            "pc_bp_equivalent",
            "Direct-BP pipeline control at the PC weight rate (1e-4).",
            "training.predictive_coding.update_mode=backprop_equivalent",
        ),
        make(
            "pc_bp_reference",
            "Direct-BP reference at the established incremental rate (1e-3).",
            "training.predictive_coding.update_mode=backprop_equivalent",
            "training.base_lr=0.001",
        ),
        make("pc_local", "Uniform local predictive coding baseline."),
        make(
            "pc_reconstruction_precision",
            "Higher precision on the top encoder and pixel reconstruction errors.",
            "training.predictive_coding.layer_precisions=[1,1,1,2,1,1,1,4]",
        ),
        make(
            "pc_global_005",
            "Local predictive coding plus a weak global reconstruction term.",
            "training.predictive_coding.global_loss_weight=0.05",
        ),
        make(
            "pc_global_020",
            "Local predictive coding plus a moderate global reconstruction term.",
            "training.predictive_coding.global_loss_weight=0.2",
        ),
        make(
            "pc_precision_global",
            "Reconstruction precision plus weak global coupling.",
            "training.predictive_coding.layer_precisions=[1,1,1,2,1,1,1,4]",
            "training.predictive_coding.global_loss_weight=0.05",
        ),
        make(
            "pc_inference10",
            "Deeper activation inference at the selected weight rate.",
            "training.predictive_coding.inference_steps=10",
        ),
        make(
            "pc_consolidate_e1",
            "Local predictive coding followed by one BP consolidation epoch.",
            "training.predictive_coding.consolidation_epochs=1",
            "training.predictive_coding.consolidation_lr_ratio=10.0",
        ),
        make(
            "pc_consolidate_e3",
            "Local predictive coding followed by three BP consolidation epochs.",
            "training.predictive_coding.consolidation_epochs=3",
            "training.predictive_coding.consolidation_lr_ratio=10.0",
        ),
    ]


def ndl_specs(seed: int, *, full: bool = False) -> list[OptimizationSpec]:
    source = _ndl_source(seed)
    checkpoint = _checkpoint("ndl", seed)
    classes = "[0,2,3,4,5,6,8,9]" if full else "[0]"
    stripped = tuple(
        item
        for item in source["overrides"]
        if not item.startswith("seed=")
        and not item.startswith("training.base_checkpoint")
        and not item.startswith("training.incremental_checkpoint")
        and not item.startswith("experiment.incremental_classes=")
        and not item.startswith("experiment.incremental_train_limit_per_class=")
        and not item.startswith("neurogenesis.global_coupling.")
    )
    common = (
        *stripped,
        f"experiment.incremental_classes={classes}",
        "experiment.incremental_train_limit_per_class=null" if full else "experiment.incremental_train_limit_per_class=512",
        f"training.base_checkpoint={checkpoint.as_posix()}",
        "training.base_checkpoint_out=null",
        "logging.mlflow.enabled=false",
        f"seed={seed}",
    )

    def make(name: str, description: str, *extra: str) -> OptimizationSpec:
        return OptimizationSpec("ndl", name, description, (*common, *extra), "ndl_baseline")

    return [
        make("ndl_baseline", "Frozen paper-compatible NDL training baseline."),
        make(
            "ndl_coupling_e1_lr001",
            "One all-weight consolidation epoch after each class.",
            "neurogenesis.global_coupling.enabled=true",
            "neurogenesis.global_coupling.trigger=after_class",
            "neurogenesis.global_coupling.epochs=1",
            "neurogenesis.global_coupling.scope=all",
            "neurogenesis.global_coupling.lr_ratio=0.01",
        ),
        make(
            "ndl_coupling_e3_lr001",
            "Three low-rate all-weight consolidation epochs after each class.",
            "neurogenesis.global_coupling.enabled=true",
            "neurogenesis.global_coupling.trigger=after_class",
            "neurogenesis.global_coupling.epochs=3",
            "neurogenesis.global_coupling.scope=all",
            "neurogenesis.global_coupling.lr_ratio=0.01",
        ),
        make(
            "ndl_coupling_e1_lr005",
            "One moderate-rate all-weight consolidation epoch after each class.",
            "neurogenesis.global_coupling.enabled=true",
            "neurogenesis.global_coupling.trigger=after_class",
            "neurogenesis.global_coupling.epochs=1",
            "neurogenesis.global_coupling.scope=all",
            "neurogenesis.global_coupling.lr_ratio=0.05",
        ),
        make(
            "ndl_coupling_e3_lr005",
            "Three moderate-rate all-weight consolidation epochs after each class.",
            "neurogenesis.global_coupling.enabled=true",
            "neurogenesis.global_coupling.trigger=after_class",
            "neurogenesis.global_coupling.epochs=3",
            "neurogenesis.global_coupling.scope=all",
            "neurogenesis.global_coupling.lr_ratio=0.05",
        ),
        make(
            "ndl_decoder_coupling_e3",
            "Decoder-only consolidation control.",
            "neurogenesis.global_coupling.enabled=true",
            "neurogenesis.global_coupling.trigger=after_class",
            "neurogenesis.global_coupling.epochs=3",
            "neurogenesis.global_coupling.scope=decoder_only",
            "neurogenesis.global_coupling.lr_ratio=0.05",
        ),
        make(
            "ndl_coupling_e5_lr005",
            "Five moderate-rate all-weight consolidation epochs after each class.",
            "neurogenesis.global_coupling.enabled=true",
            "neurogenesis.global_coupling.trigger=after_class",
            "neurogenesis.global_coupling.epochs=5",
            "neurogenesis.global_coupling.scope=all",
            "neurogenesis.global_coupling.lr_ratio=0.05",
        ),
    ]


def promotion_table(rows: list[dict[str, Any]], stage: str = "screen") -> list[dict[str, Any]]:
    completed = [row for row in rows if row.get("status") == "completed" and row.get("stage") == stage]
    table = []
    for family, baseline_name in (("predictive_coding", "pc_local"), ("ndl", "ndl_baseline")):
        family_rows = [row for row in completed if row["family"] == family]
        baseline_rows = [row for row in family_rows if row["name"] == baseline_name]
        if not baseline_rows:
            continue
        baseline_mse = statistics.fmean(float(row["macro_mse"]) for row in baseline_rows)
        baseline_forgetting = statistics.fmean(
            float(row["mean_positive_forgetting"]) for row in baseline_rows
        )
        for name in sorted({row["name"] for row in family_rows}):
            selected = [row for row in family_rows if row["name"] == name]
            mse = statistics.fmean(float(row["macro_mse"]) for row in selected)
            forgetting = statistics.fmean(float(row["mean_positive_forgetting"]) for row in selected)
            passed = (
                name != baseline_name
                and len(selected) == len(baseline_rows)
                and mse <= 0.90 * baseline_mse
                and forgetting <= baseline_forgetting + 0.002
            )
            table.append(
                {
                    "family": family,
                    "name": name,
                    "seed_count": len(selected),
                    "macro_mse": mse,
                    "macro_ratio_to_baseline": mse / baseline_mse,
                    "mean_positive_forgetting": forgetting,
                    "forgetting_delta": forgetting - baseline_forgetting,
                    "promoted": passed,
                }
            )
    return table


def _run_spec(spec: OptimizationSpec, seed: int, stage: str, output: Path, quiet: bool) -> dict[str, Any]:
    cfg = _compose_cfg(spec.overrides)
    with open_dict(cfg):
        cfg.paper_experiment = f"post_replication_optimization/{stage}/{spec.name}/seed_{seed}"
    log_path = output / "logs" / f"{stage}_{spec.name}_seed_{seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if quiet:
        with log_path.open("w", encoding="utf-8") as handle:
            with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
                result = run(cfg)
    else:
        result = run(cfg)
    summary = summarize_result(result)
    return {
        "stage": stage,
        "family": spec.family,
        "name": spec.name,
        "description": spec.description,
        "baseline": spec.baseline,
        "seed": seed,
        "status": "completed",
        "runtime_seconds": time.perf_counter() - started,
        "overrides": list(spec.overrides),
        "optimizer_diagnostics": result.get("optimizer_diagnostics", {}),
        "log_path": str(log_path),
        **summary,
    }


def _write_outputs(rows: list[dict[str, Any]], output: Path) -> list[dict[str, Any]]:
    (output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    gates = promotion_table(rows)
    (output / "promotion_gates.json").write_text(json.dumps(gates, indent=2) + "\n", encoding="utf-8")
    with (output / "promotion_gates.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "family", "name", "seed_count", "macro_mse", "macro_ratio_to_baseline",
            "mean_positive_forgetting", "forgetting_delta", "promoted",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(gates)
    return gates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("screen", "full", "all"), default="screen")
    parser.add_argument("--families", choices=("all", "predictive_coding", "ndl"), default="all")
    parser.add_argument("--seeds", type=_parse_seeds, default=DEV_SEEDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--force-full", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "summary.json"
    rows = _load_rows(path) if args.resume and path.exists() else []

    stages = ["screen", "full"] if args.stage == "all" else [args.stage]
    for stage in stages:
        full = stage == "full"
        gates = promotion_table(rows)
        promoted = {row["name"] for row in gates if row["promoted"]}
        if full and not promoted and not args.force_full:
            print("No screen candidate passed the promotion gate; full stage skipped.")
            continue
        completed = {
            (row["stage"], row["name"], int(row["seed"]))
            for row in rows
            if row.get("status") == "completed"
        }
        for seed in args.seeds:
            specs = []
            if args.families in {"all", "predictive_coding"}:
                specs.extend(pc_specs(seed, full=full))
            if args.families in {"all", "ndl"}:
                specs.extend(ndl_specs(seed, full=full))
            if full and not args.force_full:
                specs = [
                    spec
                    for spec in specs
                    if spec.name in promoted
                    or spec.name == spec.baseline
                    or spec.name == "pc_bp_equivalent"
                    or spec.name == "pc_bp_reference"
                ]
            for spec in specs:
                if (stage, spec.name, seed) in completed:
                    continue
                identity = {
                    "stage": stage, "family": spec.family, "name": spec.name,
                    "description": spec.description, "baseline": spec.baseline, "seed": seed,
                    "overrides": list(spec.overrides),
                }
                try:
                    row = _run_spec(spec, seed, stage, args.output_dir, args.quiet)
                except Exception as error:
                    rows.append({**identity, "status": "failed", "error": repr(error)})
                    _write_outputs(rows, args.output_dir)
                    raise
                rows.append(row)
                _write_outputs(rows, args.output_dir)
    gates = _write_outputs(rows, args.output_dir)
    print(json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()
