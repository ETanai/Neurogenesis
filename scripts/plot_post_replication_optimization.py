"""Plot and export the gated post-replication optimization results."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NDL_SOURCE = ROOT / "outputs" / "optimization" / "post_replication_gated"
PC_SOURCE = ROOT / "outputs" / "optimization" / "post_replication_pc_corrected"
CL_SOURCE = ROOT / "outputs" / "optimization" / "optimized_ndl_matched_controls" / "summary.json"
FIGURE = ROOT / "docs" / "figures" / "replication" / "post_replication_optimization.png"
DATA = ROOT / "docs" / "figures" / "replication" / "data"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(rows, key):
    return statistics.fmean(float(row[key]) for row in rows)


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ndl_rows = _load(NDL_SOURCE / "summary.json")
    pc_rows = _load(PC_SOURCE / "summary.json")
    cl_rows = _load(CL_SOURCE)
    ndl_gates = [row for row in _load(NDL_SOURCE / "promotion_gates.json") if row["family"] == "ndl"]
    pc_gates = [row for row in _load(PC_SOURCE / "promotion_gates.json") if row["family"] == "predictive_coding"]
    screen_rows = [
        {"family": "predictive_coding", **row} for row in pc_gates
    ] + [{"family": "ndl", **row} for row in ndl_gates]
    DATA.mkdir(parents=True, exist_ok=True)
    _write_csv(
        DATA / "post_replication_optimization_screen.csv",
        screen_rows,
        [
            "family", "name", "seed_count", "macro_mse", "macro_ratio_to_baseline",
            "mean_positive_forgetting", "forgetting_delta", "promoted",
        ],
    )

    full_ndl = [
        row for row in ndl_rows
        if row.get("status") == "completed"
        and row.get("stage") == "full"
        and row.get("name") in {"ndl_baseline", "ndl_coupling_e5_lr005"}
    ]
    full_rows = [
        {
            "condition": row["name"],
            "seed": row["seed"],
            **{key: row[key] for key in (
                "macro_mse", "foreground_mse", "mean_positive_forgetting",
                "model_update_steps", "runtime_seconds", "parameter_count", "final_widths",
            )},
        }
        for row in full_ndl
    ] + [
        {
            "condition": "cl_matched_optimized_endpoint",
            "seed": row["seed"],
            **{key: row[key] for key in (
                "macro_mse", "foreground_mse", "mean_positive_forgetting",
                "model_update_steps", "runtime_seconds", "parameter_count", "final_widths",
            )},
        }
        for row in cl_rows
    ]
    for row in full_rows:
        row["final_widths"] = json.dumps(row["final_widths"], separators=(",", ":"))
    _write_csv(
        DATA / "post_replication_optimization_full_seed.csv",
        full_rows,
        [
            "condition", "seed", "macro_mse", "foreground_mse", "mean_positive_forgetting",
            "model_update_steps", "runtime_seconds", "parameter_count", "final_widths",
        ],
    )

    condition_rows = {
        "ndl_baseline": [row for row in full_ndl if row["name"] == "ndl_baseline"],
        "ndl_optimized": [row for row in full_ndl if row["name"] == "ndl_coupling_e5_lr005"],
        "cl_matched": cl_rows,
    }
    per_class = []
    for condition, rows in condition_rows.items():
        for class_id in [1, 7, 0, 2, 3, 4, 5, 6, 8, 9]:
            per_class.append(
                {
                    "condition": condition,
                    "class_id": class_id,
                    "mean_mse": statistics.fmean(float(row["per_class_mse"][str(class_id)]) for row in rows),
                }
            )
    _write_csv(DATA / "post_replication_optimization_per_class.csv", per_class, ["condition", "class_id", "mean_mse"])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    pc_plot = [row for row in pc_gates if row["name"] not in {"pc_local", "pc_bp_equivalent"}]
    axes[0, 0].barh(
        np.arange(len(pc_plot)),
        [row["macro_ratio_to_baseline"] for row in pc_plot],
        color=["#59A14F" if row["promoted"] else "#E15759" for row in pc_plot],
    )
    axes[0, 0].axvline(0.9, color="#222", linestyle="--", label="10% MSE gate")
    axes[0, 0].set_yticks(np.arange(len(pc_plot)), [row["name"].removeprefix("pc_") for row in pc_plot])
    axes[0, 0].set_title("Corrected predictive-coding screen")
    axes[0, 0].set_xlabel("Macro MSE / local-PC baseline")
    axes[0, 0].legend(frameon=False, fontsize=8)

    ndl_plot = [row for row in ndl_gates if row["name"] != "ndl_baseline"]
    axes[0, 1].barh(
        np.arange(len(ndl_plot)),
        [row["macro_ratio_to_baseline"] for row in ndl_plot],
        color=["#59A14F" if row["promoted"] else "#E15759" for row in ndl_plot],
    )
    axes[0, 1].axvline(0.9, color="#222", linestyle="--", label="10% MSE gate")
    axes[0, 1].set_yticks(np.arange(len(ndl_plot)), [row["name"].removeprefix("ndl_") for row in ndl_plot])
    axes[0, 1].set_title("NDL consolidation screen")
    axes[0, 1].set_xlabel("Macro MSE / NDL baseline")
    axes[0, 1].legend(frameon=False, fontsize=8)

    labels = ["NDL baseline", "Optimized NDL", "Matched CL"]
    keys = ["ndl_baseline", "ndl_optimized", "cl_matched"]
    x = np.arange(3)
    width = 0.35
    axes[1, 0].bar(x - width / 2, [_mean(condition_rows[key], "macro_mse") for key in keys], width, label="Macro MSE")
    axes[1, 0].bar(x + width / 2, [_mean(condition_rows[key], "foreground_mse") for key in keys], width, label="Foreground MSE")
    axes[1, 0].set_xticks(x, labels)
    axes[1, 0].set_title("Promoted full-curriculum accuracy")
    axes[1, 0].set_ylabel("MSE; lower is better")
    axes[1, 0].legend(frameon=False)

    baseline_updates = _mean(condition_rows["ndl_baseline"], "model_update_steps")
    baseline_runtime = _mean(condition_rows["ndl_baseline"], "runtime_seconds")
    axes[1, 1].bar(
        x - width / 2,
        [_mean(condition_rows[key], "model_update_steps") / baseline_updates for key in keys],
        width,
        label="Updates / NDL baseline",
    )
    axes[1, 1].bar(
        x + width / 2,
        [_mean(condition_rows[key], "runtime_seconds") / baseline_runtime for key in keys],
        width,
        label="Runtime / NDL baseline",
    )
    axes[1, 1].axhline(1, color="#222", linestyle="--", linewidth=1)
    axes[1, 1].set_xticks(x, labels)
    axes[1, 1].set_title("Optimization cost")
    axes[1, 1].set_ylabel("Relative cost")
    axes[1, 1].legend(frameon=False, fontsize=8)

    fig.suptitle(
        "Gated post-replication optimization (development seeds 45–47)\n"
        "Screen gates are exploratory; full controls use exact endpoint matching",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
