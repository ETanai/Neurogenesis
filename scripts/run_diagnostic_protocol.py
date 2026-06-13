"""Run staged diagnostics for paper-accurate Neurogenesis behavior."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, open_dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_experiments import run


@dataclass(frozen=True)
class DiagnosticRun:
    name: str
    description: str
    overrides: tuple[str, ...]


COMMON_OVERRIDES = (
    "data=mnist",
    "experiment=mnist_incremental",
    "experiment.base_classes=[1,7]",
    "replay.enabled=true",
    "replay.mode=dataset",
)


def _default_tracking_uri() -> str:
    return f"sqlite:///{(REPO_ROOT / 'mlflow.db').resolve().as_posix()}"


def _compose_cfg(overrides: Sequence[str]):
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "config")):
        return compose(config_name="train", overrides=list(overrides))


def _run_one(
    spec: DiagnosticRun,
    *,
    experiment_name: str,
    tracking_uri: str,
    timestamp: str,
) -> dict:
    run_name = f"diagnostic_{spec.name}_{timestamp}"
    cfg = _compose_cfg(
        [
            *COMMON_OVERRIDES,
            *spec.overrides,
            f"logging.mlflow.run_name={run_name}",
        ]
    )
    with open_dict(cfg):
        cfg.paper_experiment = f"diagnostics/{spec.name}"
        cfg.logging.mlflow.experiment_name = experiment_name
        cfg.logging.mlflow.tracking_uri = tracking_uri

    print(f"\n=== Diagnostic {spec.name}: {spec.description} ===")
    result = run(cfg)
    model = result.get("model") if isinstance(result, dict) else None
    hidden_sizes = list(getattr(model, "hidden_sizes", [])) if model is not None else []
    return {
        "name": spec.name,
        "description": spec.description,
        "run_name": run_name,
        "paper_experiment": f"diagnostics/{spec.name}",
        "overrides": list(spec.overrides),
        "final_hidden_sizes": hidden_sizes,
    }


def _base_runs() -> list[DiagnosticRun]:
    return [
        DiagnosticRun(
            "base_manual_thresholds",
            "Run A/B: base-only audit with configured manual thresholds.",
            (
                "experiment.regime=ndl_ir",
                "experiment.incremental_classes=[]",
            ),
        ),
        DiagnosticRun(
            "base_estimated_thresholds",
            "Run A/B: base-only audit with thresholds estimated from base classes.",
            (
                "experiment.regime=ndl_ir",
                "experiment.incremental_classes=[]",
                "neurogenesis.thresholds=null",
            ),
        ),
    ]


def _single_digit_runs() -> list[DiagnosticRun]:
    return [
        DiagnosticRun(
            "single_0_manual_thresholds",
            "Run C: NDL+dataset replay on digit 0 using manual thresholds.",
            (
                "experiment.regime=ndl_ir",
                "experiment.incremental_classes=[0]",
            ),
        ),
        DiagnosticRun(
            "single_0_estimated_thresholds",
            "Run C: NDL+dataset replay on digit 0 using estimated thresholds.",
            (
                "experiment.regime=ndl_ir",
                "experiment.incremental_classes=[0]",
                "neurogenesis.thresholds=null",
            ),
        ),
    ]


def _capacity_run(hidden_sizes: Sequence[int]) -> DiagnosticRun:
    if not hidden_sizes:
        hidden_sizes = [261, 135, 175, 40]
    sizes = ",".join(str(int(v)) for v in hidden_sizes)
    return DiagnosticRun(
        "capacity_single_0_cl_dataset_replay",
        "Run D: CL+dataset replay with capacity derived from the single-digit NDL run.",
        (
            "experiment.regime=cl_ir",
            "experiment.incremental_classes=[0]",
            f"experiment.model.hidden_sizes=[{sizes}]",
        ),
    )


def _full_regime_runs() -> list[DiagnosticRun]:
    return [
        DiagnosticRun(
            "full_cl",
            "Run E: full MNIST conventional learning without replay.",
            ("experiment.regime=cl", "replay.enabled=false"),
        ),
        DiagnosticRun(
            "full_cl_dataset_replay",
            "Run E: full MNIST conventional learning with dataset replay.",
            ("experiment.regime=cl_ir", "replay.enabled=true", "replay.mode=dataset"),
        ),
        DiagnosticRun(
            "full_ndl",
            "Run E: full MNIST neurogenesis without replay.",
            ("experiment.regime=ndl", "replay.enabled=false"),
        ),
        DiagnosticRun(
            "full_ndl_dataset_replay",
            "Run E: full MNIST neurogenesis with dataset replay.",
            ("experiment.regime=ndl_ir", "replay.enabled=true", "replay.mode=dataset"),
        ),
    ]


def _sweep_runs() -> list[DiagnosticRun]:
    base = (
        "experiment.regime=ndl_ir",
        "experiment.incremental_classes=[0]",
        "neurogenesis.thresholds=null",
    )
    runs: list[DiagnosticRun] = []
    for percentile in (0.95, 0.975, 0.985, 0.995):
        runs.append(
            DiagnosticRun(
                f"sweep_threshold_p{str(percentile).replace('.', '_')}",
                f"Sweep: estimated threshold percentile {percentile}.",
                (*base, f"experiment.threshold.percentile={percentile}"),
            )
        )
    for value in (0.005, 0.002, 0.001):
        runs.append(
            DiagnosticRun(
                f"sweep_factor_new_nodes_{str(value).replace('.', '_')}",
                f"Sweep: factor_new_nodes={value}.",
                (*base, f"neurogenesis.factor_new_nodes={value}"),
            )
        )
    for value in (0.02, 0.05, 0.1, 0.2):
        runs.append(
            DiagnosticRun(
                f"sweep_factor_max_new_nodes_{str(value).replace('.', '_')}",
                f"Sweep: factor_max_new_nodes={value}.",
                (*base, f"neurogenesis.factor_max_new_nodes={value}"),
            )
        )
    for value in (1.0e-4, 3.0e-4, 1.0e-3):
        runs.append(
            DiagnosticRun(
                f"sweep_base_lr_{value:.0e}".replace("-", "m"),
                f"Sweep: base learning rate {value}.",
                (*base, f"training.base_lr={value}"),
            )
        )
    for value in (5, 10):
        runs.append(
            DiagnosticRun(
                f"sweep_finetune_epochs_{value}",
                f"Sweep: stacked pretraining plus {value} end-to-end finetune epochs.",
                (*base, f"training.pretrain_finetune_epochs={value}"),
            )
        )
    for value in (1.0, 2.0, 4.0):
        runs.append(
            DiagnosticRun(
                f"sweep_replay_ratio_{int(value)}",
                f"Sweep: stability dataset replay ratio {value}.",
                (*base, f"neurogenesis.stability_replay_ratio={value}"),
            )
        )
    return runs


def _tiny_overfit_common(limit: int) -> tuple[str, ...]:
    return (
        "experiment.incremental_classes=[0]",
        f"experiment.incremental_train_limit_per_class={int(limit)}",
        "neurogenesis.thresholds=null",
        "training.incremental_epochs=80",
        "neurogenesis.plasticity_epochs=80",
        "neurogenesis.stability_epochs=80",
        "neurogenesis.next_layer_epochs=20",
        "neurogenesis.max_nodes=[100,100,100,20]",
    )


def _microscope_runs() -> list[DiagnosticRun]:
    runs: list[DiagnosticRun] = [
        DiagnosticRun(
            "growth_wiring",
            "Microscope: probe gradients and updates after adding nodes at each level.",
            (
                "experiment.regime=ndl_ir",
                "experiment.incremental_classes=[0]",
                "experiment.skip_incremental_training=true",
                "neurogenesis.thresholds=null",
                "neurogenesis.growth_wiring_probe=true",
            ),
        ),
        DiagnosticRun(
            "replay_balance_recheck",
            "Microscope: rerun single-0 estimated-threshold NDL with corrected replay counters.",
            (
                "experiment.regime=ndl_ir",
                "experiment.incremental_classes=[0]",
                "neurogenesis.thresholds=null",
            ),
        ),
    ]
    for limit in (32, 128):
        common = _tiny_overfit_common(limit)
        runs.extend(
            [
                DiagnosticRun(
                    f"tiny_overfit_{limit}_cl_dataset_replay",
                    f"Microscope: fixed-capacity CL+dataset replay overfit on {limit} clean zeros.",
                    (
                        "experiment.regime=cl_ir",
                        "experiment.model.hidden_sizes=[300,200,175,40]",
                        *common,
                    ),
                ),
                DiagnosticRun(
                    f"tiny_overfit_{limit}_ndl_paper_local",
                    f"Microscope: NDL paper-local objective overfit on {limit} clean zeros.",
                    (
                        "experiment.regime=ndl_ir",
                        "neurogenesis.objective_mode=paper_local",
                        *common,
                    ),
                ),
                DiagnosticRun(
                    f"tiny_overfit_{limit}_ndl_full_reconstruction",
                    f"Microscope: NDL full-reconstruction objective overfit on {limit} clean zeros.",
                    (
                        "experiment.regime=ndl_ir",
                        "neurogenesis.objective_mode=full_reconstruction",
                        *common,
                    ),
                ),
                DiagnosticRun(
                    f"tiny_overfit_{limit}_ndl_local_plasticity_full_stability",
                    f"Microscope: NDL local plasticity with full-reconstruction stability on {limit} clean zeros.",
                    (
                        "experiment.regime=ndl_ir",
                        "neurogenesis.objective_mode=local_plasticity_full_stability",
                        *common,
                    ),
                ),
            ]
        )
    return runs


def _filter_runs(specs: Sequence[DiagnosticRun], only: str | None) -> list[DiagnosticRun]:
    if not only:
        return list(specs)
    selectors = [item.strip() for item in str(only).split(",") if item.strip()]
    if not selectors:
        return list(specs)
    selected = [
        spec
        for spec in specs
        if any(spec.name == selector or spec.name.startswith(selector) for selector in selectors)
    ]
    if not selected:
        available = ", ".join(spec.name for spec in specs)
        raise SystemExit(f"No diagnostic runs matched --only={only!r}. Available: {available}")
    return selected


def _quick_overrides() -> tuple[str, ...]:
    return (
        "training.pretrain_epochs=1",
        "training.incremental_epochs=1",
        "experiment.incremental_train_limit_per_class=16",
        "neurogenesis.plasticity_epochs=2",
        "neurogenesis.stability_epochs=2",
        "neurogenesis.next_layer_epochs=2",
        "neurogenesis.max_nodes=[2,2,2,2]",
    )


def _with_extra(specs: Sequence[DiagnosticRun], extra: Sequence[str]) -> list[DiagnosticRun]:
    if not extra:
        return list(specs)
    return [
        DiagnosticRun(spec.name, spec.description, (*spec.overrides, *extra))
        for spec in specs
    ]


def _extra_overrides(args: argparse.Namespace) -> tuple[str, ...]:
    quick = _quick_overrides() if args.quick else ()
    manual = tuple(args.override or ())
    return (*quick, *manual)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("core", "full", "sweeps", "microscope", "all"),
        default="core",
        help="core runs A-D; full runs E; sweeps runs one-knob sweeps; microscope runs mechanism diagnostics.",
    )
    parser.add_argument(
        "--experiment-name",
        default="neurogenesis-diagnostics",
        help="MLflow experiment name for diagnostic runs.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow tracking URI. Defaults to repo-local SQLite mlflow.db.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use tiny epoch/sample budgets to validate plumbing, not paper behavior.",
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="Print selected run names and exit.",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Run only matching diagnostic names. Accepts comma-separated exact names or prefixes.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Append a Hydra override to every selected diagnostic run. May be passed more than once.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    tracking_uri = args.tracking_uri or _default_tracking_uri()
    extra = _extra_overrides(args)

    specs: list[DiagnosticRun] = []
    if args.suite in {"core", "all"}:
        specs.extend(_base_runs())
        specs.extend(_single_digit_runs())
    if args.suite in {"full", "all"}:
        specs.extend(_full_regime_runs())
    if args.suite in {"sweeps", "all"}:
        specs.extend(_sweep_runs())
    if args.suite in {"microscope", "all"}:
        specs.extend(_microscope_runs())

    specs = _filter_runs(_with_extra(specs, extra), args.only)
    if args.list_runs:
        for spec in specs:
            print(f"{spec.name}: {spec.description}")
        return

    output_root = REPO_ROOT / "outputs" / "diagnostics" / timestamp
    output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    hidden_for_capacity: list[int] = []

    for spec in specs:
        summary = _run_one(
            spec,
            experiment_name=args.experiment_name,
            tracking_uri=tracking_uri,
            timestamp=timestamp,
        )
        summaries.append(summary)
        if spec.name == "single_0_estimated_thresholds":
            hidden_for_capacity = list(summary.get("final_hidden_sizes", []))

    if args.suite in {"core", "all"}:
        capacity_spec = _with_extra([_capacity_run(hidden_for_capacity)], extra)[0]
        summaries.append(
            _run_one(
                capacity_spec,
                experiment_name=args.experiment_name,
                tracking_uri=tracking_uri,
                timestamp=timestamp,
            )
        )

    payload = {
        "timestamp": timestamp,
        "suite": args.suite,
        "quick": bool(args.quick),
        "tracking_uri": tracking_uri,
        "experiment_name": args.experiment_name,
        "runs": summaries,
        "score_fields": {
            "visual_status": "review MLflow reconstruction artifacts manually",
            "quantitative_metrics": [
                "metrics/val_class_<digit>_mean_level_<level>",
                "diagnostics/outliers/class_<digit>/level_<level>_final_fraction",
                "summary/layer_<level>_growth_total",
                "diagnostics/param_delta/*",
                "diagnostics/replay/*",
            ],
        },
    }
    summary_path = output_root / "diagnostic_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nDiagnostic summary written to {summary_path}")
    print("Open MLflow with:")
    print(f"  mlflow ui --backend-store-uri {tracking_uri} --host 127.0.0.1 --port 5000")


if __name__ == "__main__":
    main()
