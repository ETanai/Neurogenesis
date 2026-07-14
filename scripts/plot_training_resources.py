"""Plot the controlled seed-42 training-time and peak-memory benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "outputs" / "benchmarks" / "training_resources_seed42" / "summary.json"
OUTPUT = ROOT / "docs" / "figures" / "replication" / "training_resource_comparison.png"

NDL = ["ndl_original_data", "ndl_intrinsic_replay", "ndl_no_replay"]
STANDARD = [
    "standard_end_to_end_original_data_size",
    "standard_end_to_end_intrinsic_replay_size",
    "standard_end_to_end_no_replay_size",
]
STACKED = "standard_stacked_original_data_size"
ORDER = [STANDARD[0], NDL[0], STANDARD[1], NDL[1], STANDARD[2], NDL[2], STACKED]
LABELS = {
    STANDARD[0]: "Standard AE\nclean endpoint",
    NDL[0]: "NDL +\noriginal data",
    STANDARD[1]: "Standard AE\nIR endpoint",
    NDL[1]: "NDL +\nintrinsic replay",
    STANDARD[2]: "Standard AE\nno-replay endpoint",
    NDL[2]: "NDL,\nno replay",
    STACKED: "Standard stacked\nclean endpoint",
}


def main() -> None:
    rows = json.loads(SUMMARY.read_text(encoding="utf-8"))
    indexed = {row["condition"]: row for row in rows}
    if set(indexed) != set(ORDER):
        raise ValueError(f"Resource plot requires {set(ORDER)}, found {set(indexed)}")
    colors = ["#4E79A7" if indexed[key]["family"] == "standard_autoencoder" else "#E15759" for key in ORDER]
    x = np.arange(len(ORDER))
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    wall = [indexed[key]["wall_seconds"] for key in ORDER]
    axes[0, 0].bar(x, wall, color=colors)
    axes[0, 0].set_title("Total wall-clock training time")
    axes[0, 0].set_ylabel("Seconds")
    axes[0, 0].set_xticks(x, [LABELS[key] for key in ORDER], rotation=25, ha="right")

    ratio_metrics = [
        ("wall_seconds", "Wall time"),
        ("peak_cuda_allocated_bytes", "Peak CUDA allocated"),
        ("peak_cuda_reserved_bytes", "Peak CUDA reserved"),
        ("peak_process_rss_bytes", "Peak process RSS"),
    ]
    positions = np.arange(3)
    width = 0.19
    for index, (metric, label) in enumerate(ratio_metrics):
        ratios = [indexed[ndl][metric] / indexed[standard][metric] for ndl, standard in zip(NDL, STANDARD)]
        axes[0, 1].bar(positions + (index - 1.5) * width, ratios, width, label=label)
    axes[0, 1].axhline(1, color="#222", linestyle="--", linewidth=1)
    axes[0, 1].set_xticks(positions, ["Original data", "Intrinsic replay", "No replay"])
    axes[0, 1].set_title("NDL / endpoint-matched standard AE")
    axes[0, 1].set_ylabel("Resource ratio; above 1 costs more")
    axes[0, 1].legend(frameon=False, fontsize=8)

    updates = [indexed[key]["total_update_steps"] for key in ORDER]
    axes[1, 0].bar(x, updates, color=colors)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Optimizer updates")
    axes[1, 0].set_ylabel("Total parameter-update steps (log scale)")
    axes[1, 0].set_xticks(x, [LABELS[key] for key in ORDER], rotation=25, ha="right")

    for key in ORDER:
        row = indexed[key]
        marker = "D" if key == STACKED else "o"
        axes[1, 1].scatter(
            row["wall_seconds"], row["macro_mse"], s=100, marker=marker,
            color="#4E79A7" if row["family"] == "standard_autoencoder" else "#E15759",
        )
        axes[1, 1].annotate(LABELS[key].replace("\n", " "), (row["wall_seconds"], row["macro_mse"]),
                            xytext=(5, 5), textcoords="offset points", fontsize=8)
    axes[1, 1].set_title("Accuracy versus training time")
    axes[1, 1].set_xlabel("Wall seconds")
    axes[1, 1].set_ylabel("Final macro MSE; lower is better")

    fig.suptitle(
        "Controlled MNIST training-resource benchmark (seed 42, one GPU)\n"
        "Standard AEs train jointly on all digits; NDL starts with 1/7 then grows incrementally",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
