"""Run exact fixed-capacity CL controls for promoted optimized-NDL endpoints."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

from omegaconf import open_dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_early_stop_ablation import _compose_cfg  # noqa: E402
from scripts.run_experiments import run  # noqa: E402
from scripts.run_organic_growth_ablation import BASE_OVERRIDES, summarize_result  # noqa: E402

SOURCE = ROOT / "outputs" / "optimization" / "post_replication_gated" / "summary.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "optimization" / "optimized_ndl_matched_controls"


def endpoint_by_seed() -> dict[int, list[int]]:
    rows = json.loads(SOURCE.read_text(encoding="utf-8"))
    return {
        int(row["seed"]): [int(value) for value in row["final_widths"]]
        for row in rows
        if row.get("status") == "completed"
        and row.get("stage") == "full"
        and row.get("name") == "ndl_coupling_e5_lr005"
    }


def overrides_for(seed: int, widths: list[int]) -> list[str]:
    return [
        *BASE_OVERRIDES,
        "experiment.regime=cl_ir",
        "experiment.incremental_classes=[0,2,3,4,5,6,8,9]",
        "experiment.control_hidden_sizes=" + str(widths).replace(" ", ""),
        "training.incremental_epochs=3",
        "model.activation=sigmoid",
        "model.activation_latent=identity",
        "replay.enabled=true",
        "replay.mode=dataset",
        "training.base_checkpoint=null",
        "training.base_checkpoint_out=null",
        "logging.mlflow.enabled=false",
        f"seed={seed}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="45,46,47")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    endpoints = endpoint_by_seed()
    missing = sorted(set(seeds) - set(endpoints))
    if missing:
        raise ValueError(f"Missing optimized NDL endpoints for seeds {missing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "summary.json"
    rows = json.loads(path.read_text()) if args.resume and path.exists() else []
    completed = {int(row["seed"]) for row in rows if row.get("status") == "completed"}
    for seed in seeds:
        if seed in completed:
            continue
        widths = endpoints[seed]
        overrides = overrides_for(seed, widths)
        cfg = _compose_cfg(overrides)
        with open_dict(cfg):
            cfg.paper_experiment = f"optimized_ndl_matched_control/seed_{seed}"
        log_path = args.output_dir / "logs" / f"seed_{seed}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        try:
            if args.quiet:
                with log_path.open("w", encoding="utf-8") as handle:
                    with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
                        result = run(cfg)
            else:
                result = run(cfg)
            row = {
                "condition": "cl_original_data_optimized_ndl_size",
                "seed": seed,
                "status": "completed",
                "matched_widths": widths,
                "runtime_seconds": time.perf_counter() - started,
                "overrides": overrides,
                "log_path": str(log_path),
                **summarize_result(result),
            }
        except Exception as error:
            row = {
                "condition": "cl_original_data_optimized_ndl_size",
                "seed": seed,
                "status": "failed",
                "matched_widths": widths,
                "error": repr(error),
            }
            rows.append(row)
            path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
            raise
        rows.append(row)
        path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
