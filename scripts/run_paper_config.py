"""Run one or more Neurogenesis experiments defined in a paper config."""

from __future__ import annotations

import argparse
import datetime as _dt
import random
import time
from pathlib import Path
from typing import Iterable, Sequence

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, open_dict

try:
    from scripts.run_experiments import run
    from scripts.plot_paper_results import plot_from_config, plot_figure4_panels
except ModuleNotFoundError:  # pragma: no cover - script-style invocation fallback
    from run_experiments import run
    from plot_paper_results import plot_from_config, plot_figure4_panels

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


def _pareto_ranks(points: dict[str, tuple[float, float]]) -> dict[str, int]:
    """Assign Pareto front rank (1 = best front) for minimization objectives."""
    if not points:
        return {}

    remaining = dict(points)
    ranks: dict[str, int] = {}
    rank = 1

    def _dominates(a: tuple[float, float], b: tuple[float, float]) -> bool:
        return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])

    while remaining:
        frontier: list[str] = []
        for run_a, point_a in remaining.items():
            dominated = False
            for run_b, point_b in remaining.items():
                if run_a == run_b:
                    continue
                if _dominates(point_b, point_a):
                    dominated = True
                    break
            if not dominated:
                frontier.append(run_a)

        for run_id in frontier:
            ranks[run_id] = rank
            remaining.pop(run_id, None)
        rank += 1

    return ranks


def _log_quality_growth_pareto_rank(*, experiment_name: str, tracking_uri: str | None) -> None:
    if mlflow is None:
        return
    try:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is None:
            return
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            max_results=50000,
        )
        points: dict[str, tuple[float, float]] = {}
        for run_obj in runs:
            rid = run_obj.info.run_id
            m = run_obj.data.metrics
            quality = m.get("summary/paper_fit_quality_term")
            if quality is None:
                quality = m.get("metrics/val_mean_level_3")
            growth = m.get("summary/growth_distance_total")
            if quality is None or growth is None:
                continue
            points[rid] = (float(quality), float(growth))

        ranks = _pareto_ranks(points)
        now_ms = int(time.time() * 1000)
        for run_id, pareto_rank in ranks.items():
            client.log_metric(
                run_id=run_id,
                key="summary/quality_growth_pareto_rank",
                value=float(pareto_rank),
                timestamp=now_ms,
                step=0,
            )
    except Exception as exc:  # pragma: no cover - best-effort logging
        print(f"[run_paper_config] Warning: failed to log Pareto ranks: {exc}")


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
    set_paper_experiment_tag = bool(data.get("set_paper_experiment_tag", True))
    # Paper configs should default to strict paper-fidelity behavior unless explicitly disabled.
    is_paper_config = "config/paper/" in config_path.as_posix()
    enforce_paper_fidelity_default = is_paper_config
    enforce_paper_fidelity = bool(
        data.get("enforce_paper_fidelity", enforce_paper_fidelity_default)
    )
    # For paper runs, keep paper_experiment tagging enabled by default for traceability/fidelity checks.
    if is_paper_config and "set_paper_experiment_tag" not in data:
        set_paper_experiment_tag = True
    if is_paper_config and enforce_paper_fidelity:
        set_paper_experiment_tag = True
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
            has_seed_override = any(str(item).startswith("seed=") for item in overrides)
            overrides.extend(
                [
                    f"logging.mlflow.run_name={mlflow_run_name}",
                ]
            )
            if not has_seed_override:
                overrides.append(f"seed={seed_value}")
            cfg = _compose_cfg(overrides)
            with open_dict(cfg):
                if set_paper_experiment_tag:
                    cfg.paper_experiment = f"{config_path.stem}/{run_name}{rep_suffix}"
                else:
                    cfg.paper_experiment = ""
                cfg.enforce_paper_fidelity = enforce_paper_fidelity
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
        _log_quality_growth_pareto_rank(
            experiment_name=experiment_name,
            tracking_uri=resolved_tracking_uri,
        )
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
