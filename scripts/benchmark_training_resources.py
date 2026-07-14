"""Benchmark wall time and peak memory for NDL and endpoint-matched standard AEs.

Each condition runs in a fresh subprocess so CUDA and process-RSS peaks reset.
The resource profile uses seed 42; it complements rather than replaces the
multi-seed accuracy confirmation.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import resource
import subprocess
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

RUN_ROOT = ROOT / "outputs" / "ablations" / "organic_growth"
DEFAULT_OUTPUT = ROOT / "outputs" / "benchmarks" / "training_resources_seed42"
SEED = 42

NDL_SOURCES = {
    "ndl_original_data": RUN_ROOT / "full_threshold_refresh_seeds42_46" / "summary.json",
    "ndl_intrinsic_replay": RUN_ROOT / "full_intrinsic_refresh_seeds42_46" / "summary.json",
    "ndl_no_replay": RUN_ROOT / "confirmation_ndl_noreplay_seeds42_51_serial" / "summary.json",
}


def _source_row(condition: str) -> dict[str, Any]:
    rows = json.loads(NDL_SOURCES[condition].read_text(encoding="utf-8"))
    matches = [row for row in rows if row.get("status") == "completed" and int(row["seed"]) == SEED]
    if len(matches) != 1:
        raise ValueError(f"Expected one seed-{SEED} source row for {condition}, found {len(matches)}")
    return matches[0]


def _strip_checkpoint_overrides(overrides: list[str]) -> list[str]:
    prefixes = (
        "training.base_checkpoint=",
        "training.base_checkpoint_out=",
        "training.incremental_checkpoint=",
        "training.incremental_checkpoint_out=",
    )
    return [value for value in overrides if not value.startswith(prefixes) and not value.startswith("seed=")]


def _standard_overrides(widths: list[int], mode: str) -> list[str]:
    return [
        *BASE_OVERRIDES,
        "experiment.base_classes=[0,1,2,3,4,5,6,7,8,9]",
        "experiment.incremental_classes=[]",
        "experiment.regime=cl",
        "experiment.control_hidden_sizes=" + str(widths).replace(" ", ""),
        "replay.enabled=false",
        "training.pretrain_epochs=50",
        f"training.pretrain_mode={mode}",
        "training.pretrain_finetune_epochs=0",
        "training.log_pretrain_thresholds=false",
        "model.activation=sigmoid",
        "model.activation_latent=identity",
        "logging.mlflow.enabled=false",
    ]


def condition_specs() -> dict[str, dict[str, Any]]:
    ndl_rows = {condition: _source_row(condition) for condition in NDL_SOURCES}
    specs: dict[str, dict[str, Any]] = {}
    for condition, row in ndl_rows.items():
        specs[condition] = {
            "family": "neurogenesis",
            "training_mode": condition.removeprefix("ndl_"),
            "endpoint_source": condition,
            "expected_widths": row["final_widths"],
            "overrides": [
                *_strip_checkpoint_overrides(list(row["overrides"])),
                "training.base_checkpoint=null",
                "training.base_checkpoint_out=null",
                f"seed={SEED}",
            ],
        }
    endpoint_rows = {
        "original_data_size": ndl_rows["ndl_original_data"],
        "intrinsic_replay_size": ndl_rows["ndl_intrinsic_replay"],
        "no_replay_size": ndl_rows["ndl_no_replay"],
    }
    for endpoint_name, row in endpoint_rows.items():
        widths = [int(value) for value in row["final_widths"]]
        condition = f"standard_end_to_end_{endpoint_name}"
        specs[condition] = {
            "family": "standard_autoencoder",
            "training_mode": "end_to_end",
            "endpoint_source": endpoint_name.removesuffix("_size"),
            "expected_widths": widths,
            "overrides": [*_standard_overrides(widths, "end_to_end"), f"seed={SEED}"],
        }
    # One stacked control quantifies the paper recipe's layer-wise schedule
    # overhead. The other two compact endpoints differ by only one neuron, so
    # repeating this 200-layer-epoch diagnostic would add cost without insight.
    compact = [int(value) for value in endpoint_rows["original_data_size"]["final_widths"]]
    specs["standard_stacked_original_data_size"] = {
        "family": "standard_autoencoder",
        "training_mode": "stacked_denoising",
        "endpoint_source": "original_data",
        "expected_widths": compact,
        "overrides": [*_standard_overrides(compact, "stacked_denoising"), f"seed={SEED}"],
    }
    return specs


def _worker(condition: str, result_path: Path, log_path: Path) -> None:
    spec = condition_specs()[condition]
    cfg = _compose_cfg(spec["overrides"])
    with open_dict(cfg):
        cfg.paper_experiment = f"training_resource_benchmark/{condition}/seed_{SEED}"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
            run_result = run(cfg)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - started
    summary = summarize_result(run_result)
    stats = run_result["training_stats"]
    result = {
        "condition": condition,
        "family": spec["family"],
        "training_mode": spec["training_mode"],
        "endpoint_source": spec["endpoint_source"],
        "seed": SEED,
        "status": "completed",
        "wall_seconds": wall_seconds,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None,
        "peak_process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "pretrain_update_steps": int(stats.get("pretrain_parameter_updates", 0)),
        "incremental_update_steps": int(stats.get("neurogenesis_parameter_updates", 0) or stats.get("incremental_parameter_updates", 0)),
        "total_update_steps": int(stats.get("total_parameter_updates", 0)),
        "parameter_count": summary["parameter_count"],
        "final_widths": summary["final_widths"],
        "macro_mse": summary["macro_mse"],
        "foreground_mse": summary["foreground_mse"],
        "mean_positive_forgetting": summary["mean_positive_forgetting"],
        "quota_stop_fraction": summary["quota_stop_fraction"],
        "unresolved_exhausted_level_count": summary["unresolved_exhausted_level_count"],
        "overrides": spec["overrides"],
        "log_path": str(log_path),
    }
    if result["final_widths"] != spec["expected_widths"]:
        raise RuntimeError(
            f"{condition} ended at {result['final_widths']}, expected {spec['expected_widths']}"
        )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def _write_summary(output_dir: Path, specs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for condition in specs:
        path = output_dir / "per_condition" / f"{condition}.json"
        if path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Training resource benchmark (seed 42)", "",
        "| Condition | Family | Widths | Wall s | Peak CUDA allocated MiB | Peak CUDA reserved MiB | Peak RSS MiB | Total updates | Macro MSE |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        mib = 1024**2
        lines.append(
            f"| {row['condition']} | {row['family']} | {row['final_widths']} | "
            f"{row['wall_seconds']:.2f} | {row['peak_cuda_allocated_bytes']/mib:.1f} | "
            f"{row['peak_cuda_reserved_bytes']/mib:.1f} | {row['peak_process_rss_bytes']/mib:.1f} | "
            f"{row['total_update_steps']} | {row['macro_mse']:.6f} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def _orchestrate(output_dir: Path, only: set[str] | None, resume: bool) -> None:
    specs = condition_specs()
    selected = [condition for condition in specs if only is None or condition in only]
    unknown = set() if only is None else only - set(specs)
    if unknown:
        raise ValueError(f"Unknown conditions: {sorted(unknown)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "seed": SEED,
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "conditions": list(specs),
        "requested_conditions": selected,
        "memory_definitions": {
            "peak_cuda_allocated_bytes": "Peak live tensor memory reported by PyTorch",
            "peak_cuda_reserved_bytes": "Peak CUDA caching-allocator reservation reported by PyTorch",
            "peak_process_rss_bytes": "Maximum resident process memory from getrusage",
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    for condition in selected:
        result_path = output_dir / "per_condition" / f"{condition}.json"
        if resume and result_path.exists() and json.loads(result_path.read_text()).get("status") == "completed":
            print(f"=== skipping completed {condition} ===")
            continue
        print(f"=== benchmarking {condition} ===", flush=True)
        command = [
            sys.executable, str(Path(__file__).resolve()), "--worker", condition,
            "--result-path", str(result_path),
            "--log-path", str(output_dir / "logs" / f"{condition}.log"),
        ]
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = str(SEED)
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        _write_summary(output_dir, specs)
    _write_summary(output_dir, specs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--only", default=None, help="Comma-separated condition names")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", default=None)
    parser.add_argument("--result-path", type=Path)
    parser.add_argument("--log-path", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.result_path is None or args.log_path is None:
            parser.error("--worker requires --result-path and --log-path")
        _worker(args.worker, args.result_path, args.log_path)
        return
    only = {item.strip() for item in args.only.split(",") if item.strip()} if args.only else None
    _orchestrate(args.output_dir, only, args.resume)


if __name__ == "__main__":
    main()
