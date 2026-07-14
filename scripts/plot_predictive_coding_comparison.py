"""Plot the fixed-endpoint predictive-coding research extension."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "predictive_coding" / "fixed_endpoint_comparison_lr1e4"
OUTPUT = ROOT / "docs" / "figures" / "replication" / "predictive_coding_comparison.png"
ORDER = ["backprop", "pc_uniform", "pc_usage"]
LABELS = {"backprop": "Backprop", "pc_uniform": "Predictive coding", "pc_usage": "PC + usage plasticity"}
COLORS = {"backprop": "#4E79A7", "pc_uniform": "#E15759", "pc_usage": "#59A14F"}


def _condition(method: str, replay: str) -> str:
    prefix = "bp" if method == "backprop" else method
    return f"{prefix}_{replay}"


def main() -> None:
    aggregate_rows = json.loads((SOURCE / "aggregate.json").read_text(encoding="utf-8"))
    aggregate = {row["condition"]: row for row in aggregate_rows}
    pc_rows = json.loads((SOURCE / "summary.json").read_text(encoding="utf-8"))
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    width = 0.25
    positions = np.arange(2)
    replay_regimes = ["original_data", "no_replay"]

    for axis, metric, title, log_scale in (
        (axes[0, 0], "macro_mse", "Final macro reconstruction MSE", False),
        (axes[0, 1], "mean_positive_forgetting", "Mean positive forgetting", True),
    ):
        for index, method in enumerate(ORDER):
            means, lower, upper = [], [], []
            for replay in replay_regimes:
                values = aggregate[_condition(method, replay)]["metrics"][metric]
                means.append(values["mean"])
                lower.append(max(0.0, values["mean"] - values["ci95_low"]))
                upper.append(max(0.0, values["ci95_high"] - values["mean"]))
            x = positions + (index - 1) * width
            axis.bar(x, means, width, color=COLORS[method], label=LABELS[method])
            axis.errorbar(x, means, yerr=np.array([lower, upper]), fmt="none", ecolor="#222", capsize=3)
        axis.set_xticks(positions, ["Original-data replay", "No replay"])
        axis.set_title(title)
        axis.set_ylabel("Lower is better")
        if log_scale:
            axis.set_yscale("log")
        axis.legend(frameon=False, fontsize=8)

    class_order = [1, 7, 0, 2, 3, 4, 5, 6, 8, 9]
    class_positions = np.arange(len(class_order))
    for method in ORDER:
        condition = _condition(method, "no_replay")
        means = [aggregate[condition]["per_class_mse"][str(cls)]["mean"] for cls in class_order]
        axes[1, 0].plot(class_positions, means, marker="o", color=COLORS[method], label=LABELS[method])
    axes[1, 0].set_title("No-replay final per-class reconstruction")
    axes[1, 0].set_xlabel("Digit in learning order")
    axes[1, 0].set_ylabel("MSE; lower is better")
    axes[1, 0].set_xticks(class_positions, [str(value) for value in class_order])
    axes[1, 0].legend(frameon=False, fontsize=8)

    pc_conditions = [
        "pc_uniform_original_data", "pc_usage_original_data",
        "pc_uniform_no_replay", "pc_usage_no_replay",
    ]
    ratios = []
    for condition in pc_conditions:
        selected = [row for row in pc_rows if row["condition"] == condition]
        ratios.append(
            statistics.fmean(
                sum(row["optimizer_diagnostics"]["energy_after"])
                / sum(row["optimizer_diagnostics"]["energy_before"])
                for row in selected
            )
        )
    axes[1, 1].bar(np.arange(4), ratios, color=["#E15759", "#59A14F", "#E15759", "#59A14F"])
    axes[1, 1].set_xticks(
        np.arange(4),
        ["PC\noriginal", "PC+usage\noriginal", "PC\nno replay", "PC+usage\nno replay"],
    )
    axes[1, 1].axhline(1.0, color="#222", linestyle="--", linewidth=1)
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].set_title("Activation-inference convergence")
    axes[1, 1].set_ylabel("Settled / initial prediction energy")

    fig.suptitle(
        "Fixed-endpoint predictive coding on incremental MNIST (seeds 42–44)\n"
        "Identical pretrained starting weights; five inference steps; PC weight LR 1e-4",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
