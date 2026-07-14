"""Correct the MNIST no-replay control with seed-specific NDL endpoint matching."""

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

NDL_SOURCE = (
    ROOT
    / "outputs"
    / "ablations"
    / "organic_growth"
    / "confirmation_ndl_noreplay_seeds42_51_serial"
    / "summary.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "ablations"
    / "organic_growth"
    / "confirmation_cl_noreplay_seed_matched_correction"
)


def _parse_seeds(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("At least one seed is required")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=_parse_seeds, default=list(range(42, 52)))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    source_rows = json.loads(NDL_SOURCE.read_text(encoding="utf-8"))
    endpoints = {
        int(row["seed"]): [int(value) for value in row["final_widths"]]
        for row in source_rows
        if row.get("status") == "completed" and row.get("name") == "ndl_no_replay_refresh"
    }
    missing = sorted(set(args.seeds) - set(endpoints))
    if missing:
        raise ValueError(f"Missing completed NDL no-replay endpoints for seeds {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    rows = json.loads(summary_path.read_text()) if args.resume and summary_path.exists() else []
    completed = {int(row["seed"]) for row in rows if row.get("status") == "completed"}
    for seed in args.seeds:
        if seed in completed:
            continue
        widths = endpoints[seed]
        checkpoint = (args.output_dir / "base_checkpoints" / f"seed_{seed}.pt").resolve()
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        overrides = (
            *BASE_OVERRIDES,
            "experiment.incremental_classes=[0,2,3,4,5,6,8,9]",
            "training.incremental_epochs=3",
            "model.activation=sigmoid",
            "model.activation_latent=identity",
            "experiment.regime=cl",
            "experiment.control_hidden_sizes=" + str(widths).replace(" ", ""),
            "replay.enabled=false",
            "replay.mode=dataset",
            f"seed={seed}",
            f"training.base_checkpoint_out={checkpoint.as_posix()}",
        )
        identity = {
            "stage": "correction",
            "name": "cl_no_replay_seed_matched",
            "family": "cl_no_replay_matched",
            "seed": seed,
            "paired_to": "ndl_no_replay_refresh",
            "matched_widths": widths,
            "overrides": list(overrides),
        }
        cfg = _compose_cfg(overrides)
        with open_dict(cfg):
            cfg.paper_experiment = f"mnist_no_replay_correction/seed_{seed}"
        started = time.perf_counter()
        try:
            if args.quiet:
                log_path = args.output_dir / "logs" / f"seed_{seed}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("w", encoding="utf-8") as handle:
                    with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
                        result = run(cfg)
                identity["log_path"] = str(log_path)
            else:
                result = run(cfg)
            summary = summarize_result(result)
            summary["runtime_seconds"] = time.perf_counter() - started
            row = {**identity, "status": "completed", **summary}
        except Exception as exc:
            row = {**identity, "status": "failed", "error": repr(exc)}
            rows.append(row)
            summary_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
            raise
        rows.append(row)
        summary_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
