"""Aggregate and plot metrics across runs defined in a paper config."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple
from urllib.parse import urlparse

import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra


REPO_ROOT = Path(__file__).resolve().parents[1]
MLFLOW_DB = REPO_ROOT / "mlflow.db"
ARTIFACT_ROOT = REPO_ROOT / "mlartifacts"
FILE_STORE_ROOT = REPO_ROOT / "mlruns"


@dataclass(frozen=True)
class RunInfo:
    run_id: str
    params: Dict[str, str]
    run_root: Path


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if value and value[0] in {'"', "'"} and value[-1] == value[0]:
        return value[1:-1]
    return value


def _read_simple_meta(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    data: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            data[key.strip()] = _strip_quotes(value.strip())
    return data


def _read_params_dir(params_dir: Path) -> Dict[str, str]:
    params: Dict[str, str] = {}
    if not params_dir.exists():
        return params
    for entry in params_dir.iterdir():
        if entry.is_file():
            params[entry.name] = entry.read_text(encoding="utf-8").strip()
    return params


def _artifact_root_from_uri(uri: str | None) -> Path | None:
    if not uri:
        return None
    parsed = urlparse(uri)
    if parsed.scheme and parsed.scheme != "file":
        return None
    path = Path(parsed.path)
    # Most URIs end with /artifacts; strip that to obtain the run root.
    if path.name == "artifacts":
        return path.parent
    return path


def _gather_runs_from_sqlite(experiment_name: str) -> Tuple[bool, list[RunInfo]]:
    if not MLFLOW_DB.exists():
        return False, []
    with sqlite3.connect(MLFLOW_DB) as conn:
        exp_row = conn.execute(
            "SELECT experiment_id FROM experiments WHERE name=?",
            (experiment_name,),
        ).fetchone()
        if exp_row is None:
            return False, []
        experiment_id = str(exp_row[0])
        run_rows = conn.execute(
            "SELECT run_uuid, artifact_uri FROM runs WHERE experiment_id=? AND status='FINISHED'",
            (experiment_id,),
        ).fetchall()
        runs: list[RunInfo] = []
        for run_id, artifact_uri in run_rows:
            run_params = _gather_run_params(conn, run_id)
            run_root = _artifact_root_from_uri(artifact_uri)
            if run_root is None:
                run_root = ARTIFACT_ROOT / experiment_id / run_id
            runs.append(RunInfo(run_id=run_id, params=run_params, run_root=run_root))
    return True, runs


def _gather_runs_from_file_store(experiment_name: str) -> Tuple[bool, list[RunInfo]]:
    if not FILE_STORE_ROOT.exists():
        return False, []
    target_dir: Path | None = None
    for entry in FILE_STORE_ROOT.iterdir():
        if not entry.is_dir() or entry.name.startswith('.'):
            continue
        meta = _read_simple_meta(entry / "meta.yaml")
        if meta.get("name") == experiment_name:
            target_dir = entry
            break
    if target_dir is None:
        return False, []
    runs: list[RunInfo] = []
    for run_dir in sorted(target_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith('.'):
            continue
        meta = _read_simple_meta(run_dir / "meta.yaml")
        if not meta:
            continue
        status = meta.get("status", "")
        if str(status) not in {"3", "FINISHED"}:
            continue
        params = _read_params_dir(run_dir / "params")
        run_id = meta.get("run_id", run_dir.name)
        artifact_uri = meta.get("artifact_uri")
        run_root = _artifact_root_from_uri(artifact_uri) or run_dir
        runs.append(RunInfo(run_id=run_id, params=params, run_root=run_root))
    return True, runs


def _collect_runs(experiment_name: str) -> list[RunInfo]:
    found, runs = _gather_runs_from_sqlite(experiment_name)
    if found:
        if not runs:
            raise RuntimeError(
                f"No completed runs found for experiment '{experiment_name}' in {MLFLOW_DB}."
            )
        return runs
    found, runs = _gather_runs_from_file_store(experiment_name)
    if found:
        if not runs:
            raise RuntimeError(
                f"No completed runs found for experiment '{experiment_name}' in {FILE_STORE_ROOT}."
            )
        return runs
    raise RuntimeError(
        f"No MLflow experiment named '{experiment_name}' found in {MLFLOW_DB} or {FILE_STORE_ROOT}."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot aggregated reconstruction curves (mean±std) for a paper config."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to config/paper/<name>.yaml that was used for the runs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/figure4_summary.png"),
        help="Where to save the generated plot (PNG).",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Which encoder depth to plot (default: highest layer found per run).",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path to store aggregated statistics as JSON.",
    )
    return parser.parse_args()


def _load_paper_config(path: Path) -> dict:
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(cfg, dict):  # pragma: no cover - defensive guard
        raise ValueError(f"Paper config must map keys to values (got {type(cfg)!r}).")
    return cfg


def _normalize_group_name(paper_experiment: str, fallback: str) -> str:
    if not paper_experiment:
        return fallback
    base = paper_experiment.split("/", 1)[-1]
    return re.sub(r"_rep\d+$", "", base)


def _gather_run_params(conn: sqlite3.Connection, run_id: str) -> Dict[str, str]:
    rows = conn.execute("SELECT key, value FROM params WHERE run_uuid=?", (run_id,)).fetchall()
    return {k: v for k, v in rows}


def _load_metrics_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    return rows


def _pick_metrics_file(run_dir: Path) -> Path | None:
    metrics_dir = run_dir / "artifacts" / "metrics"
    if not metrics_dir.exists():
        return None
    candidates = sorted(metrics_dir.glob("*_metrics.csv"))
    return candidates[0] if candidates else None


def _load_run_series(run_root: Path, target_layer: int | None) -> dict[int, float]:
    csv_path = _pick_metrics_file(run_root)
    if not csv_path:
        return {}
    rows = _load_metrics_csv(csv_path)
    if not rows:
        return {}
    layers = {int(float(row["layer"])) for row in rows if row.get("layer") is not None}
    layer_to_use = max(layers) if target_layer is None else target_layer
    series: dict[int, float] = {}
    for row in rows:
        if int(float(row["layer"])) != layer_to_use:
            continue
        step = int(float(row["step"]))
        series[step] = float(row["mean"])
    return series


def _load_run_series_all_layers(run_root: Path) -> dict[int, dict[int, float]]:
    csv_path = _pick_metrics_file(run_root)
    if not csv_path:
        return {}
    rows = _load_metrics_csv(csv_path)
    if not rows:
        return {}
    series: dict[int, dict[int, float]] = defaultdict(dict)
    for row in rows:
        layer_idx = int(float(row["layer"]))
        step = int(float(row["step"]))
        series[layer_idx][step] = float(row["mean"])
    return series


def _aggregate_series(series_list: Iterable[dict[int, float]]):
    steps = sorted({step for series in series_list for step in series.keys()})
    stats = []
    for step in steps:
        values = [series[step] for series in series_list if step in series]
        if not values:
            continue
        mean = statistics.fmean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        stats.append((step, mean, std, len(values)))
    return stats


def main() -> None:
    args = _parse_args()
    plot_from_config(args.config, args.output, layer=args.layer, json_path=args.json)


def plot_from_config(
    config_path: Path,
    output_path: Path,
    *,
    layer: int | None = None,
    json_path: Path | None = None,
    experiment_name_override: str | None = None,
) -> None:
    cfg = _load_paper_config(config_path)
    mlflow_cfg = cfg.get("mlflow", {}) or {}
    experiment_name = experiment_name_override or mlflow_cfg.get("experiment_name")
    if not experiment_name:
        raise ValueError(
            f"Paper config must set mlflow.experiment_name for analysis (missing in {config_path})."
        )

    output_path = Path(output_path)
    json_path = Path(json_path) if json_path else None

    run_infos = _collect_runs(experiment_name)

    label_lookup = {
        entry.get("name"): entry.get("description", entry.get("name"))
        for entry in cfg.get("runs", [])
    }
    group_series: dict[str, list[dict[int, float]]] = defaultdict(list)

    for run in run_infos:
        params = run.params
        paper_tag = params.get("paper_experiment", "")
        run_name = params.get("mlflow.runName", run.run_id)
        group = _normalize_group_name(paper_tag, fallback=run_name)
        series = _load_run_series(run.run_root, target_layer=layer)
        if not series:
            print(f"[WARN] Missing metrics for run {run.run_id}; skipping.")
            continue
        group_series[group].append(series)

    if not group_series:
        raise RuntimeError("No metric series found to plot.")

    ordered_groups = [
        entry.get("name") for entry in cfg.get("runs", []) if entry.get("name") in group_series
    ]
    remaining = [g for g in group_series.keys() if g not in ordered_groups]
    ordered_groups.extend(remaining)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    summary_payload = {}

    for group in ordered_groups:
        series_list = group_series[group]
        stats = _aggregate_series(series_list)
        if not stats:
            continue
        steps, means, stds, counts = zip(*stats)
        label = label_lookup.get(group, group)
        plt.plot(steps, means, label=label)
        lower = [m - s for m, s in zip(means, stds)]
        upper = [m + s for m, s in zip(means, stds)]
        plt.fill_between(steps, lower, upper, alpha=0.2)
        summary_payload[group] = {
            "steps": list(steps),
            "mean": list(means),
            "std": list(stds),
            "count": list(counts),
        }

    plt.xlabel("Evaluation step (classes learned)")
    plt.ylabel("Validation reconstruction error (top layer)")
    plt.title(f"Aggregated reconstruction curves: {experiment_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"[plot_paper_results] Figure saved to {output_path}")

    if json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        print(f"[plot_paper_results] Summary data written to {json_path}")


def _load_digit_sequence(config_data: dict) -> tuple[list[int], list[int]]:
    runs = config_data.get("runs", [])
    if not runs:
        return [], []
    overrides = runs[0].get("overrides", [])
    if not overrides:
        return [], []
    config_dir = Path(__file__).resolve().parents[1] / "config"
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="train", overrides=list(overrides))
    base = list(getattr(cfg.experiment, "base_classes", []))
    incr = list(getattr(cfg.experiment, "incremental_classes", []))
    return [int(x) for x in base], [int(x) for x in incr]


def _pad_for_base(stats: list[tuple[int, float, float, int]], base_len: int, total_len: int):
    if not stats:
        return []
    diff = max(base_len - 1, 0)
    padded = list(stats)
    prefix = [stats[0]] * diff
    aligned = prefix + padded
    if len(aligned) > total_len:
        aligned = aligned[-total_len:]
    if len(aligned) < total_len:
        aligned.extend([aligned[-1]] * (total_len - len(aligned)))
    return aligned


def _digit_labels(base: list[int], incr: list[int]) -> list[str]:
    labels = [str(x) for x in base + incr]
    return labels


_NEURON_METRIC_PATTERN = re.compile(r"global_level_(\d+)_size")


def _load_neuron_series_from_files(run_root: Path) -> dict[int, dict[int, float]]:
    metrics_dir = run_root / "metrics"
    if not metrics_dir.exists():
        return {}
    data: dict[int, dict[int, float]] = defaultdict(dict)
    for metric_file in metrics_dir.iterdir():
        if not metric_file.is_file():
            continue
        match = _NEURON_METRIC_PATTERN.match(metric_file.name)
        if not match:
            continue
        level = int(match.group(1))
        with metric_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                try:
                    value = float(parts[1])
                    step = int(float(parts[2]))
                except ValueError:
                    continue
                data[level][step] = value
    return data


def plot_figure4_panels(
    config_path: Path,
    output_path: Path,
    *,
    experiment_name_override: str | None = None,
) -> None:
    cfg = _load_paper_config(config_path)
    mlflow_cfg = cfg.get("mlflow", {}) or {}
    experiment_name = experiment_name_override or mlflow_cfg.get("experiment_name")
    if not experiment_name:
        raise ValueError("Figure 4 panels require mlflow.experiment_name.")

    base_classes, incremental_classes = _load_digit_sequence(cfg)
    digit_labels = _digit_labels(base_classes, incremental_classes)
    if not digit_labels:
        digit_labels = [str(i + 1) for i in range(10)]
    total_digits = len(digit_labels)
    base_len = len(base_classes) if base_classes else 1

    group_layer_series: dict[str, dict[int, list[dict[int, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    neuron_series_runs: list[dict[int, dict[int, float]]] = []

    run_infos = _collect_runs(experiment_name)
    for run in run_infos:
        params = run.params
        group = _normalize_group_name(
            params.get("paper_experiment", ""), params.get("mlflow.runName", run.run_id)
        )
        layer_series = _load_run_series_all_layers(run.run_root)
        if not layer_series:
            continue
        for layer_idx, series in layer_series.items():
            group_layer_series[group][layer_idx].append(series)
        if group == "ndl_ir":
            neuron_series_runs.append(_load_neuron_series_from_files(run.run_root))

    if not group_layer_series:
        raise RuntimeError("No per-layer series found for Figure 4 generation.")

    layer_count = max(
        max(layers.keys(), default=-1) for layers in group_layer_series.values()
    ) + 1
    layer_labels = [f"Level {idx + 1}" for idx in range(layer_count)]
    layer_colors = ["#1f77b4", "#ff7f0e", "#9467bd", "#bcbd22", "#8c564b"]

    regimes = [
        ("cl", "CL Networks"),
        ("ndl", "NDL Networks"),
        ("cl_ir", "CL + IR Networks"),
        ("ndl_ir", "NDL + IR Networks"),
    ]

    aggregated: dict[str, dict[int, list[tuple[int, float, float, int]]]] = {}
    for regime, _ in regimes:
        layered = {}
        for layer_idx, series_list in group_layer_series.get(regime, {}).items():
            layered[layer_idx] = _aggregate_series(series_list)
        aggregated[regime] = layered

    baseline = None
    top_layer_idx = layer_count - 1
    if aggregated.get("cl", {}).get(top_layer_idx):
        baseline = _pad_for_base(
            aggregated["cl"][top_layer_idx], base_len, total_digits
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes = axes.reshape(3, 2)

    x = list(range(total_digits))

    def _plot_panel(ax, regime_key: str, title: str):
        ax.set_title(title)
        ax.set_xlabel("Digit")
        ax.set_ylabel("Reconstruction Error")
        data = aggregated.get(regime_key, {})
        for layer_idx in range(layer_count):
            stats = data.get(layer_idx)
            if not stats:
                continue
            aligned = _pad_for_base(stats, base_len, total_digits)
            means = [entry[1] for entry in aligned]
            stds = [entry[2] for entry in aligned]
            color = layer_colors[layer_idx % len(layer_colors)]
            ax.plot(
                x,
                means,
                label=layer_labels[layer_idx],
                color=color,
                marker="o",
            )
            ax.fill_between(
                x,
                [m - s for m, s in zip(means, stds)],
                [m + s for m, s in zip(means, stds)],
                color=color,
                alpha=0.15,
            )
        if baseline:
            baseline_vals = [entry[1] for entry in baseline]
            ax.plot(
                x,
                baseline_vals,
                linestyle="--",
                color="black",
                label="Baseline",
            )
        ax.set_xticks(x, digit_labels, rotation=45)
        ax.grid(True, alpha=0.2)

    _plot_panel(axes[0][0], "cl", "CL Networks")
    _plot_panel(axes[0][1], "ndl", "NDL Networks")
    _plot_panel(axes[1][0], "cl_ir", "CL + IR Networks")
    _plot_panel(axes[1][1], "ndl_ir", "NDL + IR Networks")

    ax_combined = axes[2][0]
    ax_combined.set_title("Full Network Reconstruction (Level 4)")
    ax_combined.set_xlabel("Digit")
    ax_combined.set_ylabel("Reconstruction Error")
    for regime_key, title in regimes:
        stats = aggregated.get(regime_key, {}).get(top_layer_idx)
        if not stats:
            continue
        aligned = _pad_for_base(stats, base_len, total_digits)
        means = [entry[1] for entry in aligned]
        stds = [entry[2] for entry in aligned]
        ax_combined.plot(
            x,
            means,
            label=title,
            marker="o",
        )
        ax_combined.fill_between(
            x,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            alpha=0.15,
        )
    ax_combined.set_xticks(x, digit_labels, rotation=45)
    ax_combined.grid(True, alpha=0.2)
    ax_combined.legend()

    ax_neuron = axes[2][1]
    ax_neuron.set_title("Neuron Growth (NDL + IR)")
    ax_neuron.set_xlabel("Digit")
    ax_neuron.set_ylabel("Number of Neurons")
    if neuron_series_runs:
        level_stats: dict[int, list[dict[int, float]]] = defaultdict(list)
        for series in neuron_series_runs:
            for level_idx, entries in series.items():
                level_stats[level_idx].append(entries)
        for level_idx, series_list in level_stats.items():
            stats = _aggregate_series(series_list)
            aligned = _pad_for_base(stats, base_len, total_digits)
            values = [entry[1] for entry in aligned]
            stds = [entry[2] for entry in aligned]
            color = layer_colors[level_idx % len(layer_colors)]
            ax_neuron.plot(x, values, marker="o", label=layer_labels[level_idx], color=color)
            ax_neuron.fill_between(
                x,
                [v - s for v, s in zip(values, stds)],
                [v + s for v, s in zip(values, stds)],
                alpha=0.15,
                color=color,
            )
    ax_neuron.set_xticks(x, digit_labels, rotation=45)
    ax_neuron.grid(True, alpha=0.2)
    ax_neuron.legend()

    for ax in axes.flat:
        if not ax.has_data():
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
