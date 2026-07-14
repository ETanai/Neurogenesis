"""Generate the SD19 feasibility figure from the frozen paired manifest."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "outputs" / "sd19" / "feasibility_screen" / "summary.json"
OUTPUT = ROOT / "docs" / "figures" / "replication" / "sd19_feasibility_comparison.png"
SEEDS = [42, 43, 44]
CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 22, 35, 36, 61]


def main() -> None:
    rows = json.loads(SUMMARY.read_text(encoding="utf-8"))
    indexed = {(row["condition"], int(row["seed"])): row for row in rows}
    expected = {(condition, seed) for condition in ("ndl_dataset", "cl_dataset") for seed in SEEDS}
    if set(indexed) != expected or any(row.get("status") != "completed" for row in rows):
        raise ValueError("SD19 figure requires the complete three-seed paired screen")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = {"cl_dataset": "#4E79A7", "ndl_dataset": "#E15759"}
    labels = {"cl_dataset": "Capacity-matched CL", "ndl_dataset": "NDL"}

    metrics = [
        ("macro_mse", "Macro reconstruction MSE"),
        ("foreground_mse", "Foreground reconstruction MSE"),
        ("mean_positive_forgetting", "Mean positive forgetting"),
    ]
    for axis, (metric, title) in zip(axes.flat[:3], metrics):
        for seed in SEEDS:
            cl = float(indexed[("cl_dataset", seed)][metric])
            ndl = float(indexed[("ndl_dataset", seed)][metric])
            axis.plot([0, 1], [cl, ndl], color="#777", alpha=0.6, marker="o")
        means = [
            statistics.fmean(float(indexed[(condition, seed)][metric]) for seed in SEEDS)
            for condition in ("cl_dataset", "ndl_dataset")
        ]
        axis.scatter([0, 1], means, s=150, marker="D", zorder=4,
                     color=[colors["cl_dataset"], colors["ndl_dataset"]], label="Mean")
        axis.set_xticks([0, 1], [labels["cl_dataset"], labels["ndl_dataset"]])
        axis.set_title(title)
        axis.set_ylabel("Lower is better")
        ratio = means[1] / means[0]
        axis.text(0.5, max(means) * 1.08, f"NDL / CL = {ratio:.2f}×", ha="center")
        axis.set_ylim(0, max(means) * 1.25)

    axis = axes.flat[3]
    x = np.arange(len(CLASSES))
    for condition, linestyle in (("cl_dataset", "--"), ("ndl_dataset", "-")):
        means = [
            statistics.fmean(
                float(indexed[(condition, seed)]["per_class_mse"][str(class_id)])
                for seed in SEEDS
            )
            for class_id in CLASSES
        ]
        axis.plot(x, means, marker="o", linestyle=linestyle, color=colors[condition], label=labels[condition])
    class_labels = [str(value) if value < 10 else {10: "A", 15: "F", 22: "M", 35: "Z", 36: "a", 61: "z"}[value] for value in CLASSES]
    axis.axvline(9.5, color="#333", linestyle=":", linewidth=1)
    axis.text(4.5, axis.get_ylim()[1] * 0.94, "base digits", ha="center")
    axis.text(12.5, axis.get_ylim()[1] * 0.94, "incoming letters", ha="center")
    axis.set_xticks(x, class_labels)
    axis.set_title("Final class-level reconstruction")
    axis.set_xlabel("SD19 class")
    axis.set_ylabel("MSE; lower is better")
    axis.legend(frameon=False)

    fig.suptitle(
        "SD19 clean-replay feasibility screen (three paired seeds)\n"
        "Exact endpoint matching: [1100, 600, 350, 70]",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
