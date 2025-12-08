"""Run one or more Neurogenesis experiments defined in a paper config."""

from __future__ import annotations

import argparse
import datetime as _dt
import random
from pathlib import Path
from typing import Iterable, Sequence

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, open_dict

from scripts.run_experiments import run
from scripts.plot_paper_results import plot_from_config, plot_figure4_panels

REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MLFLOW_DIR = (REPO_ROOT / "mlruns").resolve()

try:
    import mlflow
except ImportError:  # pragma: no cover - optional dependency
    mlflow = None


def _generate_seeds(num_reps: int, seed_cfg: dict) -> list[int]:
    base = int(seed_cfg.get("base", 42))
    strategy = str(seed_cfg.get("strategy", "sequential")).lower()
    if num_reps <= 1:
        return [base]
    if strategy == "random":
        rng = random.SystemRandom()
        return [rng.randrange(1, 2**32 - 1) for _ in range(num_reps)]
    return [base + i for i in range(num_reps)]


def _compose_cfg(overrides: Sequence[str]):
    config_dir = Path(__file__).resolve().parents[1] / "config"
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="train", overrides=list(overrides))
    return cfg


def _ensure_runs(runs: Iterable[dict]) -> list[dict]:
    run_list = [dict(run) for run in runs]
    if not run_list:
        raise ValueError("Paper config must define at least one run under 'runs'.")
    return run_list


def _make_summary_paths(config_path: Path, timestamp: str) -> tuple[Path, Path, Path]:
    root = (
        Path(__file__).resolve().parents[1]
        / "outputs"
        / "paper_runs"
        / config_path.stem
        / timestamp
    )
    root.mkdir(parents=True, exist_ok=True)
    figure_path = root / f"{config_path.stem}_summary.png"
    json_path = root / f"{config_path.stem}_summary.json"
    return root, figure_path, json_path


def _log_summary_run(
    *,
    experiment_name: str,
    tracking_uri: str | None,
    config_path: Path,
    repetitions: int,
    runs_per_rep: int,
    seeds: list[int],
    artifact_paths: list[Path],
) -> None:
    if mlflow is None:
        print(
            "[run_paper_config] mlflow is not installed; skipping summary run logging "
            f"for experiment '{experiment_name}'."
        )
        return
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    run_name = f"{config_path.stem}_summary"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "paper_config": config_path.as_posix(),
                "repetitions": repetitions,
                "runs_per_repetition": runs_per_rep,
                "seeds": ",".join(str(s) for s in seeds),
            }
        )
        mlflow.log_metrics(
            {
                "summary/total_child_runs": repetitions * runs_per_rep,
                "summary/repetitions": repetitions,
            }
        )
        for artifact in artifact_paths:
            if artifact.exists():
                mlflow.log_artifact(str(artifact), artifact_path="summary")


def _default_tracking_uri() -> str:
    """Return a file-based MLflow tracking URI under the repo's mlruns directory."""

    _DEFAULT_MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    return _DEFAULT_MLFLOW_DIR.as_uri()


def run_from_config(config_path: Path) -> None:
    data = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(data, dict):
        raise ValueError(f"Paper config {config_path} must be a mapping at the top level.")

    repetitions = max(int(data.get("repetitions", 1)), 1)
    seed_cfg = data.get("seed", {}) or {}
    seeds = _generate_seeds(repetitions, seed_cfg)
    runs = _ensure_runs(data.get("runs", []))
    mlflow_cfg = data.get("mlflow", {}) or {}
    experiment_name_base = mlflow_cfg.get("experiment_name")
    tracking_uri = mlflow_cfg.get("tracking_uri")
    if tracking_uri:
        resolved_tracking_uri = tracking_uri
    else:
        resolved_tracking_uri = _default_tracking_uri()
        print(
            "[run_paper_config] No MLflow tracking URI provided; defaulting to local file store "
            f"at {resolved_tracking_uri}."
        )
    run_prefix = mlflow_cfg.get("run_name_prefix", config_path.stem)
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{experiment_name_base}_{timestamp}" if experiment_name_base else None
    summary_root, figure_path, json_path = _make_summary_paths(config_path, timestamp)
    extra_artifacts: list[Path] = []

    for rep_idx in range(repetitions):
        seed_value = seeds[rep_idx]
        rep_suffix = f"_rep{rep_idx + 1}" if repetitions > 1 else ""
        for run_spec in runs:
            run_name = str(run_spec.get("name", "run"))
            description = run_spec.get("description", run_name)
            overrides: list[str] = list(run_spec.get("overrides", []))
            if not overrides:
                raise ValueError(
                    f"Run '{run_name}' in {config_path} must specify Hydra overrides."
                )
            mlflow_run_name = f"{run_prefix}_{run_name}{rep_suffix}_{timestamp}"
            overrides.extend(
                [
                    f"logging.mlflow.run_name={mlflow_run_name}",
                    f"seed={seed_value}",
                ]
            )
            cfg = _compose_cfg(overrides)
            with open_dict(cfg):
                cfg.paper_experiment = f"{config_path.stem}/{run_name}{rep_suffix}"
                if experiment_name:
                    cfg.logging.mlflow.experiment_name = experiment_name
                cfg.logging.mlflow.tracking_uri = resolved_tracking_uri
            print(
                "\n=== Running {cfg_name}: {desc} | repeat {cur}/{total} | seed={seed} ===".format(
                    cfg_name=config_path.stem,
                    desc=description,
                    cur=rep_idx + 1,
                    total=repetitions,
                    seed=seed_value,
                )
            )
            run(cfg)

    if experiment_name:
        print(f"\n=== Generating summary artifacts for experiment '{experiment_name}' ===")
        plot_from_config(
            config_path,
            figure_path,
            json_path=json_path,
            experiment_name_override=experiment_name,
        )

        if config_path.stem == "figure4":
            panels_path = summary_root / f"{config_path.stem}_panels.png"
            plot_figure4_panels(
                config_path,
                panels_path,
                experiment_name_override=experiment_name,
            )
            extra_artifacts.append(panels_path)

        _log_summary_run(
            experiment_name=experiment_name,
            tracking_uri=resolved_tracking_uri,
            config_path=config_path,
            repetitions=repetitions,
            runs_per_rep=len(runs),
            seeds=seeds,
            artifact_paths=[figure_path, json_path, *extra_artifacts],
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paper experiments defined in config/paper/*.yaml"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the paper config YAML (e.g., config/paper/mnist_cl.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_from_config(args.config)


if __name__ == "__main__":
    main()
