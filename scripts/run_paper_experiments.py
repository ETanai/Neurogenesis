from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from hydra import compose, initialize

from scripts.run_experiments import run as run_single


@dataclass(frozen=True)
class PaperExperiment:
    name: str
    description: str
    overrides: tuple[str, ...]


PAPER_EXPERIMENTS: tuple[PaperExperiment, ...] = (
    PaperExperiment(
        name="mnist_cl",
        description="MNIST baseline: classical learning, no intrinsic replay.",
        overrides=(
            "data=mnist",
            "experiment=mnist_incremental",
            "experiment.regime=cl",
            "replay.enabled=false",
        ),
    ),
    PaperExperiment(
        name="mnist_cl_ir",
        description="MNIST baseline: classical learning with intrinsic replay.",
        overrides=(
            "data=mnist",
            "experiment=mnist_incremental",
            "experiment.regime=cl_ir",
            "replay.enabled=true",
        ),
    ),
    PaperExperiment(
        name="mnist_ndl",
        description="MNIST neurogenesis without intrinsic replay.",
        overrides=(
            "data=mnist",
            "experiment=mnist_incremental",
            "experiment.regime=ndl",
            "replay.enabled=false",
        ),
    ),
    PaperExperiment(
        name="mnist_ndl_ir",
        description="MNIST neurogenesis with intrinsic replay (main paper result).",
        overrides=(
            "data=mnist",
            "experiment=mnist_incremental",
            "experiment.regime=ndl_ir",
            "replay.enabled=true",
        ),
    ),
    PaperExperiment(
        name="sd19_ndl",
        description="SD-19 neurogenesis without intrinsic replay (training data replay).",
        overrides=(
            "data=sd19",
            "experiment=sd19_incremental",
            "experiment.regime=ndl",
            "replay.enabled=false",
        ),
    ),
    PaperExperiment(
        name="sd19_ndl_ir",
        description="SD-19 neurogenesis with intrinsic replay enabled for comparison.",
        overrides=(
            "data=sd19",
            "experiment=sd19_incremental",
            "experiment.regime=ndl_ir",
            "replay.enabled=true",
        ),
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Neurogenesis-Deep-Learning paper experiments sequentially."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/paper_experiments_summary.json"),
        help="Path to the JSON summary that aggregates metrics from every run.",
    )
    return parser.parse_args()


def _ensure_outputs_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _to_list(seq: Iterable[int]) -> list[int]:
    return [int(x) for x in seq]


def run_suite(output_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)
    results: list[dict] = []
    with initialize(version_base=None, config_path="config"):
        for spec in PAPER_EXPERIMENTS:
            cfg = compose(config_name="train", overrides=list(spec.overrides))
            start = time.perf_counter()
            artifacts = run_single(cfg)
            runtime = time.perf_counter() - start

            model = artifacts["model"]
            stats = artifacts.get("training_stats", {})
            total_params = sum(p.numel() for p in model.parameters())
            record = {
                "name": spec.name,
                "description": spec.description,
                "dataset": cfg.data.name,
                "regime": cfg.experiment.regime,
                "base_classes": _to_list(cfg.experiment.base_classes),
                "incremental_classes": _to_list(cfg.experiment.incremental_classes),
                "hidden_sizes": list(model.hidden_sizes),
                "total_parameters": int(total_params),
                "runtime_sec": runtime,
                "pretrain_parameter_updates": int(stats.get("pretrain_parameter_updates", 0)),
                "neurogenesis_parameter_updates": int(
                    stats.get("neurogenesis_parameter_updates", 0)
                ),
                "incremental_parameter_updates": int(
                    stats.get("incremental_parameter_updates", 0)
                ),
                "total_parameter_updates": int(stats.get("total_parameter_updates", 0)),
                "uses_replay": bool("ir" in str(cfg.experiment.regime).lower()),
                "hydra_overrides": list(spec.overrides),
            }
            results.append(record)
    _ensure_outputs_dir(output_path)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[paper_experiments] Summary written to {output_path}")


def main() -> None:
    args = _parse_args()
    run_suite(args.output)


if __name__ == "__main__":
    main()
