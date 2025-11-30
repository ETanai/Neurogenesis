"""Recreate every experiment and figure referenced in the Neurogenesis paper."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plot_paper_results import ARTIFACT_ROOT, MLFLOW_DB, plot_from_config, plot_figure4_panels
from scripts.run_paper_config import run_from_config


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    description: str
    config_rel_path: str

    @property
    def config_path(self) -> Path:
        return REPO_ROOT / self.config_rel_path


@dataclass(frozen=True)
class FigureSpec:
    name: str
    description: str
    generator: str
    params: dict[str, Any]


EXPERIMENT_SPECS: list[ExperimentSpec] = [
    ExperimentSpec(
        name="figure4_runs",
        description="MNIST incremental ablations covering the Figure 4 comparisons.",
        config_rel_path="config/paper/figure4.yaml",
    ),
    ExperimentSpec(
        name="mnist_cl",
        description="Classical learning baseline on MNIST (no replay).",
        config_rel_path="config/paper/mnist_cl.yaml",
    ),
    ExperimentSpec(
        name="mnist_cl_ir",
        description="Classical learning baseline on MNIST with intrinsic replay.",
        config_rel_path="config/paper/mnist_cl_ir.yaml",
    ),
    ExperimentSpec(
        name="mnist_ndl",
        description="Neurogenesis without intrinsic replay on MNIST.",
        config_rel_path="config/paper/mnist_ndl.yaml",
    ),
    ExperimentSpec(
        name="mnist_ndl_ir",
        description="Neurogenesis with intrinsic replay on MNIST (main result).",
        config_rel_path="config/paper/mnist_ndl_ir.yaml",
    ),
    ExperimentSpec(
        name="sd19_ndl",
        description="SD-19 neurogenesis using dataset replay (no intrinsic replay).",
        config_rel_path="config/paper/sd19_ndl.yaml",
    ),
    ExperimentSpec(
        name="sd19_ndl_ir",
        description="SD-19 neurogenesis with dataset replay plus intrinsic replay.",
        config_rel_path="config/paper/sd19_ndl_ir.yaml",
    ),
]


FIGURE_SPECS: list[FigureSpec] = [
    FigureSpec(
        name="figure4_curves",
        description="Aggregate reconstruction curves for the Figure 4 experiment.",
        generator="plot_from_config",
        params={
            "config_rel_path": "config/paper/figure4.yaml",
            "output_rel_path": "outputs/figure4_summary.png",
            "json_rel_path": "outputs/figure4_summary.json",
            "layer": None,
        },
    ),
    FigureSpec(
        name="figure4_panels",
        description="Full Figure 4 panel layout (A–F).",
        generator="figure4_panels",
        params={
            "config_rel_path": "config/paper/figure4.yaml",
            "output_rel_path": "outputs/figure4_full.png",
        },
    ),
    FigureSpec(
        name="figure5_recon_timeline",
        description="Reconstruction progression for CL+IR vs NDL+IR (Figure 5).",
        generator="reconstruction_timeline",
        params={
            "experiment_name": "neurogenesis-paper-figure4",
            "output_rel_path": "outputs/figure5_recon_timeline.png",
            "groups": [
                {
                    "label": "CL + IR",
                    "paper_experiment_like": "figure4/cl_ir%",
                    "layer_index": 3,
                },
                {
                    "label": "NDL + IR",
                    "paper_experiment_like": "figure4/ndl_ir%",
                    "layer_index": 3,
                },
            ],
            "title": "Figure 5 – Reconstruction fidelity as classes accumulate",
        },
    ),
]


ALIAS_MAP: dict[str, list[str]] = {
    "figure4": ["figure4_runs", "figure4_curves", "figure4_panels"],
    "figure5": ["figure5_recon_timeline"],
}


def _generate_plot_from_config(params: dict[str, Any]) -> None:
    config_rel = params.get("config_rel_path")
    output_rel = params.get("output_rel_path")
    if not config_rel or not output_rel:
        raise ValueError("plot_from_config generator requires config_rel_path and output_rel_path.")
    layer = params.get("layer")
    json_rel = params.get("json_rel_path")
    config_path = REPO_ROOT / config_rel
    output_path = REPO_ROOT / output_rel
    json_path = REPO_ROOT / json_rel if json_rel else None
    plot_from_config(config_path, output_path, layer=layer, json_path=json_path)


def _generate_figure4_panels(params: dict[str, Any]) -> None:
    config_rel = params.get("config_rel_path")
    output_rel = params.get("output_rel_path")
    if not config_rel or not output_rel:
        raise ValueError("figure4_panels generator requires config_rel_path and output_rel_path.")
    config_path = REPO_ROOT / config_rel
    output_path = REPO_ROOT / output_rel
    override = params.get("experiment_name_override")
    plot_figure4_panels(
        config_path,
        output_path,
        experiment_name_override=override,
    )


def _resolve_experiment_id(conn: sqlite3.Connection, experiment_name: str) -> int:
    row = conn.execute("SELECT experiment_id FROM experiments WHERE name=?", (experiment_name,)).fetchone()
    if row is None:
        raise RuntimeError(f"No MLflow experiment named '{experiment_name}' found in {MLFLOW_DB}.")
    return int(row[0])


def _select_best_run_for_pattern(
    conn: sqlite3.Connection,
    experiment_id: int,
    value_like: str,
    metric_key: str,
) -> str:
    rows = conn.execute(
        """
        SELECT r.run_uuid
        FROM runs r
        JOIN params p ON r.run_uuid = p.run_uuid
        WHERE r.experiment_id=? AND r.status='FINISHED' AND p.key='paper_experiment' AND p.value LIKE ?
        """,
        (experiment_id, value_like),
    ).fetchall()
    if not rows:
        raise RuntimeError(
            f"No runs found in experiment_id={experiment_id} matching paper_experiment LIKE '{value_like}'."
        )
    best_run: str | None = None
    best_value: float | None = None
    for (run_id,) in rows:
        metric_row = conn.execute(
            "SELECT value FROM metrics WHERE run_uuid=? AND key=? ORDER BY step DESC LIMIT 1",
            (run_id, metric_key),
        ).fetchone()
        if metric_row is None:
            continue
        value = float(metric_row[0])
        if best_value is None or value < best_value:
            best_value = value
            best_run = run_id
    if best_run is None:
        raise RuntimeError(
            f"No matching metrics '{metric_key}' found for runs matching '{value_like}'."
        )
    return best_run


def _collect_reconstruction_images(
    experiment_id: int,
    run_id: str,
    *,
    artifact_subdir: str,
    prefix: str,
    suffix: str,
) -> list[tuple[int, Path]]:
    run_dir = ARTIFACT_ROOT / str(experiment_id) / run_id / "artifacts" / artifact_subdir
    if not run_dir.exists():
        return []
    items: list[tuple[int, Path]] = []
    for path in sorted(run_dir.iterdir()):
        name = path.name
        if name.endswith(".mpl.png"):
            continue
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue
        step_part = name[len(prefix) : len(name) - len(suffix)]
        if not step_part.isdigit():
            continue
        items.append((int(step_part), path))
    items.sort(key=lambda pair: pair[0])
    return items


def _generate_reconstruction_timeline(params: dict[str, Any]) -> None:
    experiment_name = params.get("experiment_name")
    group_specs = params.get("groups") or []
    output_rel = params.get("output_rel_path")
    if not experiment_name or not output_rel or not group_specs:
        raise ValueError(
            "reconstruction_timeline generator requires experiment_name, output_rel_path, and groups."
        )
    artifact_subdir = params.get("artifact_subdir", "figures")
    prefix = params.get("image_prefix", "reconstructions_step_")
    suffix = params.get("image_suffix", ".png")
    default_layer = params.get("default_layer", None)
    global_step_limit = params.get("max_steps")

    with sqlite3.connect(MLFLOW_DB) as conn:
        experiment_id = _resolve_experiment_id(conn, str(experiment_name))
        rows: list[dict[str, Any]] = []
        for group in group_specs:
            pattern = group.get("paper_experiment_like")
            if not pattern:
                raise ValueError("Each group must define 'paper_experiment_like'.")
            label = group.get("label", pattern)
            layer_idx = group.get("layer_index", default_layer)
            if layer_idx is None:
                raise ValueError(
                    f"Group '{label}' must provide layer_index when default_layer is not set."
                )
            metric_key = f"metrics/val_mean_level_{int(layer_idx)}"
            run_id = _select_best_run_for_pattern(conn, experiment_id, pattern, metric_key)
            images = _collect_reconstruction_images(
                experiment_id,
                run_id,
                artifact_subdir=artifact_subdir,
                prefix=prefix,
                suffix=suffix,
            )
            if not images:
                raise RuntimeError(
                    f"No reconstruction artifacts found for run {run_id} (group '{label}')."
                )
            step_limit = group.get("max_steps", global_step_limit)
            if step_limit is not None:
                images = images[: int(step_limit)]
            rows.append({"label": label, "run_id": run_id, "images": images})

    if not rows:
        raise RuntimeError("No groups resolved for reconstruction timeline.")

    max_cols = max(len(row["images"]) for row in rows)
    if max_cols == 0:
        raise RuntimeError("Reconstruction timeline has zero columns.")

    cell_w = float(params.get("cell_width", 2.4))
    cell_h = float(params.get("cell_height", 2.4))
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(rows), max_cols, figsize=(max_cols * cell_w, len(rows) * cell_h))
    if len(rows) == 1:
        axes = [axes]
    if max_cols == 1:
        axes = [[ax] for ax in axes]

    for row_idx, row in enumerate(rows):
        data = row["images"]
        for col_idx in range(max_cols):
            ax = axes[row_idx][col_idx]
            if col_idx >= len(data):
                ax.axis("off")
                continue
            step, path = data[col_idx]
            img = mpimg.imread(str(path))
            ax.imshow(img)
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(f"Step {step}", fontsize=9)
        axes[row_idx][0].set_ylabel(row["label"], rotation=0, labelpad=25, va="center")

    title = params.get("title")
    if title:
        fig.suptitle(str(title), fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig.tight_layout()

    output_path = REPO_ROOT / output_rel
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=int(params.get("dpi", 150)))
    plt.close(fig)


FIGURE_GENERATORS: dict[str, Callable[[dict[str, Any]], None]] = {
    "plot_from_config": _generate_plot_from_config,
    "reconstruction_timeline": _generate_reconstruction_timeline,
    "figure4_panels": _generate_figure4_panels,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run every Neurogenesis paper experiment and regenerate the published figures."
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="TARGET",
        help=(
            "Subset of targets to run (e.g., figure4_runs figure4_plot). "
            "Use --list-targets to inspect names."
        ),
    )
    parser.add_argument(
        "--skip-experiments",
        action="store_true",
        help="Skip model training runs and only regenerate figures.",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip figure generation and only run experiments.",
    )
    parser.add_argument(
        "--list-targets",
        action="store_true",
        help="Print all experiment/figure targets and exit.",
    )
    return parser.parse_args()


def _print_available_targets() -> None:
    print("Experiments:")
    for spec in EXPERIMENT_SPECS:
        print(f"  - {spec.name:13} {spec.description} [{spec.config_rel_path}]")
    print("\nFigures:")
    for spec in FIGURE_SPECS:
        out_path = spec.params.get("output_rel_path")
        suffix = f" -> {out_path}" if out_path else ""
        print(f"  - {spec.name:13} {spec.description}{suffix}")
    if ALIAS_MAP:
        print("\nAliases:")
        for alias, targets in ALIAS_MAP.items():
            print(f"  - {alias:13} {', '.join(targets)}")


def _resolve_targets(only: list[str] | None) -> tuple[list[ExperimentSpec], list[FigureSpec]]:
    if not only:
        return list(EXPERIMENT_SPECS), list(FIGURE_SPECS)

    tokens = [token.lower() for token in only]
    if "all" in tokens:
        return list(EXPERIMENT_SPECS), list(FIGURE_SPECS)

    exp_lookup = {spec.name.lower(): spec for spec in EXPERIMENT_SPECS}
    fig_lookup = {spec.name.lower(): spec for spec in FIGURE_SPECS}

    selected_exp: list[ExperimentSpec] = []
    selected_fig: list[FigureSpec] = []
    exp_added: set[str] = set()
    fig_added: set[str] = set()

    if "experiments" in tokens:
        selected_exp = list(EXPERIMENT_SPECS)
        exp_added = {spec.name.lower() for spec in selected_exp}
    if "figures" in tokens:
        selected_fig = list(FIGURE_SPECS)
        fig_added = {spec.name.lower() for spec in selected_fig}

    specials = {"experiments", "figures"}

    for token in tokens:
        if token in specials:
            continue
        expanded = ALIAS_MAP.get(token, [token])
        matched = False
        for alias in expanded:
            alias_key = alias.lower()
            if alias_key in exp_lookup and alias_key not in exp_added:
                selected_exp.append(exp_lookup[alias_key])
                exp_added.add(alias_key)
                matched = True
            elif alias_key in fig_lookup and alias_key not in fig_added:
                selected_fig.append(fig_lookup[alias_key])
                fig_added.add(alias_key)
                matched = True
        if not matched:
            raise ValueError(f"Unknown target '{token}'. Use --list-targets to inspect valid names.")

    return selected_exp, selected_fig


def _run_experiments(experiments: Iterable[ExperimentSpec]) -> None:
    experiments = list(experiments)
    total = len(experiments)
    for idx, spec in enumerate(experiments, start=1):
        print(
            f"\n=== [{idx}/{total}] Running experiment '{spec.name}': {spec.description} ==="
        )
        run_from_config(spec.config_path)


def _run_figures(figures: Iterable[FigureSpec]) -> None:
    figures = list(figures)
    total = len(figures)
    for idx, spec in enumerate(figures, start=1):
        print(f"\n=== [{idx}/{total}] Generating figure '{spec.name}': {spec.description} ===")
        generator = FIGURE_GENERATORS.get(spec.generator)
        if generator is None:
            raise RuntimeError(f"Unknown figure generator '{spec.generator}' for {spec.name}.")
        params = dict(spec.params)
        generator(params)


def main() -> None:
    args = _parse_args()
    if args.list_targets:
        _print_available_targets()
        return

    try:
        experiments, figures = _resolve_targets(args.only)
    except ValueError as exc:
        print(f"[recreate_paper] {exc}")
        sys.exit(1)

    if args.skip_experiments:
        experiments = []
    if args.skip_figures:
        figures = []

    if not experiments and not figures:
        print("[recreate_paper] Nothing to do. Use --list-targets to inspect available options.")
        return

    if experiments:
        _run_experiments(experiments)
    if figures:
        _run_figures(figures)

    print("\n[recreate_paper] Complete.")


if __name__ == "__main__":
    main()
