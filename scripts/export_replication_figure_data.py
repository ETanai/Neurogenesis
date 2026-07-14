"""Export portable source data for every figure in the replication report.

The canonical experiment summaries remain under ``outputs/``.  This script
collects their relevant observations into a single JSON bundle and normalized
CSV tables so the published diagrams can be recreated without Python or the
repository's plotting implementation.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "figures" / "replication" / "data"
RUN_ROOT = ROOT / "outputs" / "ablations" / "organic_growth"
AGGREGATE = RUN_ROOT / "confirmation_10seed_aggregate" / "summary.json"
SD19_SUMMARY = ROOT / "outputs" / "sd19" / "feasibility_screen" / "summary.json"
RESOURCE_SUMMARY = (
    ROOT / "outputs" / "benchmarks" / "training_resources_seed42" / "summary.json"
)

ORDER = [
    "cl_dataset_oracle",
    "cl_intrinsic",
    "cl_no_replay",
    "ndl_dataset_oracle",
    "ndl_intrinsic",
    "ndl_no_replay",
]
LABELS = {
    "cl_dataset_oracle": "CL + original data",
    "cl_intrinsic": "CL + intrinsic replay",
    "cl_no_replay": "CL, no replay",
    "ndl_dataset_oracle": "NDL + original data",
    "ndl_intrinsic": "NDL + intrinsic replay",
    "ndl_no_replay": "NDL, no replay",
}
MNIST_CLASSES = [1, 7, 0, 2, 3, 4, 5, 6, 8, 9]
SD19_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 22, 35, 36, 61]
SCALAR_METRICS = [
    "macro_mse",
    "foreground_mse",
    "mean_positive_forgetting",
    "model_update_steps",
    "parameter_count",
    "quota_stop_fraction",
    "updated_class_fraction",
    "unresolved_exhausted_level_count",
    "runtime_seconds",
]


def _load(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {path}")
    return payload


def _completed(path: Path, name: str | None = None) -> list[dict[str, Any]]:
    rows = [row for row in _load(path) if row.get("status") == "completed"]
    return rows if name is None else [row for row in rows if row.get("name") == name]


def _mnist_raw() -> dict[str, list[dict[str, Any]]]:
    controls = RUN_ROOT / "confirmation_controls_seeds42_51_v2" / "summary.json"
    return {
        "cl_dataset_oracle": _completed(controls, "cl_dataset_oracle_matched"),
        "cl_intrinsic": _completed(controls, "cl_intrinsic_matched"),
        "cl_no_replay": _completed(
            RUN_ROOT / "confirmation_cl_noreplay_seed_matched_correction" / "summary.json",
            "cl_no_replay_seed_matched",
        ),
        "ndl_dataset_oracle": [
            *_completed(RUN_ROOT / "full_threshold_refresh_seeds42_46" / "summary.json"),
            *_completed(
                RUN_ROOT / "confirmation_ndl_seeds47_51" / "summary.json",
                "ndl_dataset_oracle_refresh",
            ),
        ],
        "ndl_intrinsic": [
            *_completed(RUN_ROOT / "full_intrinsic_refresh_seeds42_46" / "summary.json"),
            *_completed(RUN_ROOT / "full_intrinsic_refresh_seeds43_46_resume" / "summary.json"),
            *_completed(
                RUN_ROOT / "confirmation_ndl_seeds47_51" / "summary.json",
                "ndl_intrinsic_refresh",
            ),
            *_completed(RUN_ROOT / "confirmation_ndl_ir_seeds48_51_serial" / "summary.json"),
        ],
        "ndl_no_replay": _completed(
            RUN_ROOT / "confirmation_ndl_noreplay_seeds42_51_serial" / "summary.json"
        ),
    }


def _mean_ci(values: list[float]) -> tuple[float, float, float]:
    mean = statistics.fmean(values)
    half_width = 2.262 * statistics.stdev(values) / math.sqrt(len(values))
    return mean, mean - half_width, mean + half_width


def _write_csv(name: str, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    with (OUTPUT / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _json_cell(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    aggregate_rows = _load(AGGREGATE)
    aggregate = {row["condition"]: row for row in aggregate_rows}
    raw = _mnist_raw()
    for condition in ORDER:
        seeds = sorted(int(row["seed"]) for row in raw[condition])
        if seeds != list(range(42, 52)):
            raise ValueError(f"{condition} has unexpected seeds: {seeds}")

    condition_metrics: list[dict[str, Any]] = []
    for condition in ORDER:
        for metric, values in aggregate[condition]["metrics"].items():
            condition_metrics.append(
                {"condition": condition, "label": LABELS[condition], "metric": metric, **values}
            )
    _write_csv(
        "mnist_condition_metrics.csv",
        condition_metrics,
        ["condition", "label", "metric", "mean", "std", "ci95_low", "ci95_high"],
    )

    seed_metrics: list[dict[str, Any]] = []
    per_class_seed: list[dict[str, Any]] = []
    for condition in ORDER:
        for source in sorted(raw[condition], key=lambda row: int(row["seed"])):
            row = {
                "condition": condition,
                "label": LABELS[condition],
                "seed": int(source["seed"]),
                "final_widths": _json_cell(source.get("final_widths")),
                **{metric: source.get(metric) for metric in SCALAR_METRICS},
            }
            seed_metrics.append(row)
            for class_id in MNIST_CLASSES:
                per_class_seed.append(
                    {
                        "condition": condition,
                        "label": LABELS[condition],
                        "seed": int(source["seed"]),
                        "class_id": class_id,
                        "learning_order": MNIST_CLASSES.index(class_id),
                        "mse": source["per_class_mse"][str(class_id)],
                    }
                )
    _write_csv(
        "mnist_seed_metrics.csv",
        seed_metrics,
        ["condition", "label", "seed", "final_widths", *SCALAR_METRICS],
    )
    _write_csv(
        "mnist_per_class_seed.csv",
        per_class_seed,
        ["condition", "label", "seed", "class_id", "learning_order", "mse"],
    )

    per_class_summary: list[dict[str, Any]] = []
    for condition in ORDER:
        for learning_order, class_id in enumerate(MNIST_CLASSES):
            values = [float(row["per_class_mse"][str(class_id)]) for row in raw[condition]]
            mean, low, high = _mean_ci(values)
            per_class_summary.append(
                {
                    "condition": condition,
                    "label": LABELS[condition],
                    "class_id": class_id,
                    "learning_order": learning_order,
                    "mean_mse": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "seed_count": len(values),
                }
            )
    _write_csv(
        "mnist_per_class_summary.csv",
        per_class_summary,
        [
            "condition", "label", "class_id", "learning_order", "mean_mse",
            "ci95_low", "ci95_high", "seed_count",
        ],
    )

    architecture = {
        "initial": [200, 100, 75, 20],
        "paper_final_approx": [225, 135, 84, 40],
        "ndl_dataset_oracle": [
            statistics.fmean(row["final_widths"][i] for row in raw["ndl_dataset_oracle"])
            for i in range(4)
        ],
        "ndl_intrinsic": [232, 140, 91, 44],
        "ndl_no_replay": [
            statistics.fmean(row["final_widths"][i] for row in raw["ndl_no_replay"])
            for i in range(4)
        ],
    }
    architecture_rows = [
        {"series": series, "layer": layer + 1, "neurons": widths[layer]}
        for series, widths in architecture.items()
        for layer in range(4)
    ]
    _write_csv("mnist_architecture.csv", architecture_rows, ["series", "layer", "neurons"])

    claim_rows: list[dict[str, Any]] = []
    for suffix in ("dataset_oracle", "intrinsic", "no_replay"):
        for metric in ("macro_mse", "mean_positive_forgetting"):
            ndl = aggregate[f"ndl_{suffix}"]["metrics"][metric]["mean"]
            cl = aggregate[f"cl_{suffix}"]["metrics"][metric]["mean"]
            claim_rows.append(
                {
                    "replay_regime": suffix,
                    "metric": metric,
                    "ndl_mean": ndl,
                    "cl_mean": cl,
                    "ndl_over_cl": ndl / cl,
                }
            )
    _write_csv(
        "mnist_claim_ratios.csv",
        claim_rows,
        ["replay_regime", "metric", "ndl_mean", "cl_mean", "ndl_over_cl"],
    )

    sd19 = _load(SD19_SUMMARY)
    sd19_indexed = {(row["condition"], int(row["seed"])): row for row in sd19}
    expected_sd19 = {
        (condition, seed)
        for condition in ("cl_dataset", "ndl_dataset")
        for seed in (42, 43, 44)
    }
    if set(sd19_indexed) != expected_sd19:
        raise ValueError("SD19 export requires the complete paired three-seed screen")

    sd19_seed = []
    sd19_class_seed = []
    for source in sorted(sd19, key=lambda row: (row["condition"], int(row["seed"]))):
        sd19_seed.append(
            {
                "condition": source["condition"],
                "seed": int(source["seed"]),
                "final_widths": _json_cell(source["final_widths"]),
                **{metric: source.get(metric) for metric in SCALAR_METRICS},
                "peak_cuda_bytes": source.get("peak_cuda_bytes"),
            }
        )
        for class_id in SD19_CLASSES:
            sd19_class_seed.append(
                {
                    "condition": source["condition"],
                    "seed": int(source["seed"]),
                    "class_id": class_id,
                    "mse": source["per_class_mse"][str(class_id)],
                }
            )
    _write_csv(
        "sd19_seed_metrics.csv",
        sd19_seed,
        ["condition", "seed", "final_widths", *SCALAR_METRICS, "peak_cuda_bytes"],
    )
    _write_csv(
        "sd19_per_class_seed.csv",
        sd19_class_seed,
        ["condition", "seed", "class_id", "mse"],
    )

    sd19_summary_rows = []
    for condition in ("cl_dataset", "ndl_dataset"):
        rows = [row for row in sd19 if row["condition"] == condition]
        for metric in ("macro_mse", "foreground_mse", "mean_positive_forgetting"):
            values = [float(row[metric]) for row in rows]
            sd19_summary_rows.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "mean": statistics.fmean(values),
                    "seed_count": len(values),
                }
            )
        for class_id in SD19_CLASSES:
            values = [float(row["per_class_mse"][str(class_id)]) for row in rows]
            sd19_summary_rows.append(
                {
                    "condition": condition,
                    "metric": f"class_{class_id}_mse",
                    "mean": statistics.fmean(values),
                    "seed_count": len(values),
                }
            )
    _write_csv(
        "sd19_plotted_summary.csv",
        sd19_summary_rows,
        ["condition", "metric", "mean", "seed_count"],
    )

    cross_rows = []
    for suffix in ("dataset_oracle", "intrinsic", "no_replay"):
        for metric in ("macro_mse", "mean_positive_forgetting", "model_update_steps"):
            cl = aggregate[f"cl_{suffix}"]["metrics"][metric]["mean"]
            ndl = aggregate[f"ndl_{suffix}"]["metrics"][metric]["mean"]
            cross_rows.append(
                {
                    "dataset": "MNIST",
                    "replay_regime": suffix,
                    "metric": metric,
                    "cl_mean": cl,
                    "ndl_mean": ndl,
                    "ndl_over_cl": ndl / cl,
                }
            )
    for metric in ("macro_mse", "mean_positive_forgetting", "model_update_steps"):
        cl = statistics.fmean(float(sd19_indexed[("cl_dataset", seed)][metric]) for seed in (42, 43, 44))
        ndl = statistics.fmean(float(sd19_indexed[("ndl_dataset", seed)][metric]) for seed in (42, 43, 44))
        cross_rows.append(
            {
                "dataset": "SD19 screen",
                "replay_regime": "dataset_oracle",
                "metric": metric,
                "cl_mean": cl,
                "ndl_mean": ndl,
                "ndl_over_cl": ndl / cl,
            }
        )
    _write_csv(
        "cross_dataset_ratios.csv",
        cross_rows,
        ["dataset", "replay_regime", "metric", "cl_mean", "ndl_mean", "ndl_over_cl"],
    )

    resources = _load(RESOURCE_SUMMARY)
    resource_fields = sorted({key for row in resources for key in row if key not in {"overrides"}})
    resource_csv = [
        {**row, "final_widths": _json_cell(row["final_widths"])} for row in resources
    ]
    _write_csv("training_resource_conditions.csv", resource_csv, resource_fields)
    resource_index = {row["condition"]: row for row in resources}
    resource_pairs = [
        ("original_data", "ndl_original_data", "standard_end_to_end_original_data_size"),
        ("intrinsic_replay", "ndl_intrinsic_replay", "standard_end_to_end_intrinsic_replay_size"),
        ("no_replay", "ndl_no_replay", "standard_end_to_end_no_replay_size"),
    ]
    resource_ratios = []
    for regime, ndl_key, standard_key in resource_pairs:
        for metric in (
            "wall_seconds", "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes",
            "peak_process_rss_bytes", "total_update_steps", "macro_mse",
        ):
            resource_ratios.append(
                {
                    "replay_regime": regime,
                    "metric": metric,
                    "ndl_value": resource_index[ndl_key][metric],
                    "standard_value": resource_index[standard_key][metric],
                    "ndl_over_standard": resource_index[ndl_key][metric] / resource_index[standard_key][metric],
                }
            )
    _write_csv(
        "training_resource_ratios.csv",
        resource_ratios,
        ["replay_regime", "metric", "ndl_value", "standard_value", "ndl_over_standard"],
    )

    bundle = {
        "schema_version": 1,
        "description": "Portable base observations and summaries for all replication-report figures.",
        "mnist": {"aggregate": aggregate_rows, "seed_observations": raw},
        "sd19": {"seed_observations": sd19},
        "training_resources": {"seed_observations": resources},
    }
    (OUTPUT / "replication_figure_base_data.json").write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    files = {
        "performance_comparison.png": ["mnist_condition_metrics.csv", "mnist_seed_metrics.csv"],
        "per_class_comparison.png": ["mnist_per_class_summary.csv", "mnist_per_class_seed.csv"],
        "mechanism_comparison.png": [
            "mnist_architecture.csv", "mnist_condition_metrics.csv", "mnist_seed_metrics.csv"
        ],
        "claim_direction_comparison.png": ["mnist_claim_ratios.csv"],
        "cross_dataset_effects.png": ["cross_dataset_ratios.csv"],
        "sd19_feasibility_comparison.png": [
            "sd19_plotted_summary.csv", "sd19_seed_metrics.csv", "sd19_per_class_seed.csv"
        ],
        "training_resource_comparison.png": [
            "training_resource_conditions.csv", "training_resource_ratios.csv"
        ],
    }
    manifest = {
        "schema_version": 1,
        "generated_by": "scripts/export_replication_figure_data.py",
        "complete_json_bundle": "replication_figure_base_data.json",
        "figures": files,
        "notes": [
            "CSV files use long/tidy form where practical.",
            "Widths stored in CSV cells are JSON arrays.",
            "Ratios below one favor NDL for error and forgetting metrics.",
            "MNIST confidence intervals use two-sided Student-t intervals over seeds 42-51.",
        ],
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Exported {len(list(OUTPUT.iterdir()))} portable data files to {OUTPUT}")


if __name__ == "__main__":
    main()
