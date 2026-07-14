"""Generate figures for the final Neurogenesis replication report."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "figures" / "replication"
AGGREGATE = (
    ROOT
    / "outputs"
    / "ablations"
    / "organic_growth"
    / "confirmation_10seed_aggregate"
    / "summary.json"
)
RUN_ROOT = ROOT / "outputs" / "ablations" / "organic_growth"
SD19_SUMMARY = ROOT / "outputs" / "sd19" / "feasibility_screen" / "summary.json"

LABELS = {
    "cl_dataset_oracle": "CL + original data",
    "cl_intrinsic": "CL + intrinsic replay",
    "cl_no_replay": "CL, no replay",
    "ndl_dataset_oracle": "NDL + original data",
    "ndl_intrinsic": "NDL + intrinsic replay",
    "ndl_no_replay": "NDL, no replay",
}
COLORS = {
    "cl_dataset_oracle": "#4C78A8",
    "cl_intrinsic": "#72A0C1",
    "cl_no_replay": "#9ECAE1",
    "ndl_dataset_oracle": "#E45756",
    "ndl_intrinsic": "#B33C3C",
    "ndl_no_replay": "#F28E7F",
}
ORDER = [
    "cl_dataset_oracle",
    "cl_intrinsic",
    "cl_no_replay",
    "ndl_dataset_oracle",
    "ndl_intrinsic",
    "ndl_no_replay",
]
T_9 = 2.262


def _load(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {path}")
    return payload


def _completed(path: Path, name: str | None = None) -> list[dict[str, Any]]:
    rows = [row for row in _load(path) if row.get("status") == "completed"]
    if name is not None:
        rows = [row for row in rows if row.get("name") == name]
    return rows


def _raw_conditions() -> dict[str, list[dict[str, Any]]]:
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


def _mean_ci(values: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    return mean, T_9 * statistics.stdev(values) / math.sqrt(len(values))


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def performance_figure(aggregate: dict[str, dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    metrics = [
        ("macro_mse", "Macro reconstruction MSE", False),
        ("foreground_mse", "Foreground-pixel MSE", False),
        ("mean_positive_forgetting", "Mean positive forgetting", True),
    ]
    x = np.arange(len(ORDER))
    for axis, (metric, title, log_scale) in zip(axes, metrics):
        means = [aggregate[key]["metrics"][metric]["mean"] for key in ORDER]
        low = [aggregate[key]["metrics"][metric]["ci95_low"] for key in ORDER]
        high = [aggregate[key]["metrics"][metric]["ci95_high"] for key in ORDER]
        errors = np.array([np.array(means) - np.array(low), np.array(high) - np.array(means)])
        axis.bar(x, means, color=[COLORS[key] for key in ORDER], width=0.72)
        axis.errorbar(x, means, yerr=errors, fmt="none", ecolor="#222", capsize=3, lw=1)
        axis.set_title(title)
        axis.set_xticks(x, [LABELS[key] for key in ORDER], rotation=38, ha="right")
        axis.grid(axis="y", alpha=0.25)
        if log_scale:
            axis.set_yscale("log")
    fig.suptitle("Ten-seed MNIST confirmation: lower is better", fontsize=14, y=1.03)
    fig.tight_layout()
    fig.savefig(OUTPUT / "performance_comparison.png")
    plt.close(fig)


def per_class_figure(raw: dict[str, list[dict[str, Any]]]) -> None:
    digits = [1, 7, 0, 2, 3, 4, 5, 6, 8, 9]
    positions = np.arange(len(digits))
    pairs = [
        ("dataset_oracle", "Original-data replay (clean oracle)"),
        ("intrinsic", "Intrinsic replay"),
        ("no_replay", "No replay"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for axis, (suffix, title) in zip(axes, pairs):
        for prefix, linestyle in (("cl", "--"), ("ndl", "-")):
            key = f"{prefix}_{suffix}"
            means, cis = [], []
            for digit in digits:
                values = [float(row["per_class_mse"][str(digit)]) for row in raw[key]]
                mean, ci = _mean_ci(values)
                means.append(mean)
                cis.append(ci)
            axis.plot(
                positions,
                means,
                marker="o",
                linestyle=linestyle,
                color=COLORS[key],
                label="NDL" if prefix == "ndl" else "CL",
            )
            axis.fill_between(
                positions,
                np.array(means) - np.array(cis),
                np.array(means) + np.array(cis),
                color=COLORS[key],
                alpha=0.14,
            )
        axis.set_title(title)
        axis.set_xlabel("Digit (learning order)")
        axis.set_xticks(positions, [str(digit) for digit in digits])
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    axes[0].set_ylabel("Final per-class reconstruction MSE")
    fig.suptitle("Final class-level reconstruction after learning all ten digits", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT / "per_class_comparison.png")
    plt.close(fig)


def mechanism_figure(
    aggregate: dict[str, dict[str, Any]], raw: dict[str, list[dict[str, Any]]]
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    levels = np.arange(1, 5)
    architecture_series = {
        "Initial": [200, 100, 75, 20],
        "Paper final (approx.)": [225, 135, 84, 40],
        "NDL + original data": [
            statistics.fmean(row["final_widths"][i] for row in raw["ndl_dataset_oracle"])
            for i in range(4)
        ],
        "NDL + intrinsic replay": [232, 140, 91, 44],
        "NDL, no replay": [
            statistics.fmean(row["final_widths"][i] for row in raw["ndl_no_replay"])
            for i in range(4)
        ],
    }
    styles = ["--", ":", "-", "-", "-"]
    colors = ["#777", "#111", COLORS["ndl_dataset_oracle"], COLORS["ndl_intrinsic"], COLORS["ndl_no_replay"]]
    for (label, widths), linestyle, color in zip(architecture_series.items(), styles, colors):
        axes[0].plot(levels, widths, marker="o", label=label, linestyle=linestyle, color=color)
    axes[0].set_title("Encoder shape")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Neurons")
    axes[0].set_xticks(levels)
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    update_means = [aggregate[key]["metrics"]["model_update_steps"]["mean"] for key in ORDER]
    axes[1].bar(np.arange(6), update_means, color=[COLORS[key] for key in ORDER])
    axes[1].set_yscale("log")
    axes[1].set_title("Incremental optimizer updates")
    axes[1].set_xticks(np.arange(6), [LABELS[key] for key in ORDER], rotation=38, ha="right")
    axes[1].set_ylabel("Updates (log scale)")
    axes[1].grid(axis="y", alpha=0.25)

    ndl_keys = ["ndl_dataset_oracle", "ndl_intrinsic", "ndl_no_replay"]
    quota = [aggregate[key]["metrics"]["quota_stop_fraction"]["mean"] for key in ndl_keys]
    updated = [aggregate[key]["metrics"]["updated_class_fraction"]["mean"] for key in ndl_keys]
    exhausted = [
        statistics.fmean(float(row["unresolved_exhausted_level_count"]) / 32.0 for row in raw[key])
        for key in ndl_keys
    ]
    x = np.arange(3)
    width = 0.25
    axes[2].bar(x - width, quota, width, label="Quota stops", color="#59A14F")
    axes[2].bar(x, updated, width, label="Classes updated", color="#4E79A7")
    axes[2].bar(x + width, exhausted, width, label="Exhausted loops", color="#E15759")
    axes[2].set_title("Growth behavior")
    axes[2].set_xticks(x, ["Original\ndata", "Intrinsic", "No replay"])
    axes[2].set_ylim(0, 1.08)
    axes[2].set_ylabel("Fraction")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend(frameon=False, fontsize=8)
    fig.suptitle("Capacity, compute, and stopping mechanism", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT / "mechanism_comparison.png")
    plt.close(fig)


def claim_direction_figure(aggregate: dict[str, dict[str, Any]]) -> None:
    replay = ["Original data", "Intrinsic replay", "No replay"]
    suffixes = ["dataset_oracle", "intrinsic", "no_replay"]
    metrics = [
        ("macro_mse", "Reconstruction ratio: NDL / matched CL"),
        ("mean_positive_forgetting", "Forgetting ratio: NDL / matched CL"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for axis, (metric, title) in zip(axes, metrics):
        ratios = [
            aggregate[f"ndl_{suffix}"]["metrics"][metric]["mean"]
            / aggregate[f"cl_{suffix}"]["metrics"][metric]["mean"]
            for suffix in suffixes
        ]
        colors = ["#59A14F" if value < 1 else "#E15759" for value in ratios]
        axis.bar(replay, ratios, color=colors)
        axis.axhline(1.0, color="#222", linestyle="--", linewidth=1)
        axis.axhspan(0, 1, color="#59A14F", alpha=0.08, label="Paper-favored direction")
        for index, value in enumerate(ratios):
            axis.text(index, value * 1.03, f"{value:.2f}×", ha="center", va="bottom")
        axis.set_title(title)
        axis.set_ylabel("Ratio; lower than 1 favors NDL")
        axis.set_ylim(0, max(ratios) * 1.22)
        axis.tick_params(axis="x", rotation=18)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle(
        "Paper claim direction versus replication\n"
        "The paper reports NDL+IR slightly better than CL+IR",
        fontsize=14,
        y=1.08,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT / "claim_direction_comparison.png")
    plt.close(fig)


def cross_dataset_figure(aggregate: dict[str, dict[str, Any]]) -> None:
    sd19_rows = _completed(SD19_SUMMARY)
    sd19 = {
        condition: [row for row in sd19_rows if row["condition"] == condition]
        for condition in ("cl_dataset", "ndl_dataset")
    }
    if any(sorted(int(row["seed"]) for row in rows) != [42, 43, 44] for rows in sd19.values()):
        raise ValueError("Cross-dataset figure requires all three paired SD19 seeds")

    labels = ["MNIST\noriginal data", "MNIST\nintrinsic replay", "MNIST\nno replay", "SD19 screen\noriginal data"]
    suffixes = ["dataset_oracle", "intrinsic", "no_replay"]
    metrics = [
        ("macro_mse", "Reconstruction error", False),
        ("mean_positive_forgetting", "Positive forgetting", True),
        ("model_update_steps", "Incremental optimizer updates", True),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for axis, (metric, title, log_scale) in zip(axes, metrics):
        ratios = [
            aggregate[f"ndl_{suffix}"]["metrics"][metric]["mean"]
            / aggregate[f"cl_{suffix}"]["metrics"][metric]["mean"]
            for suffix in suffixes
        ]
        sd_cl = statistics.fmean(float(row[metric]) for row in sd19["cl_dataset"])
        sd_ndl = statistics.fmean(float(row[metric]) for row in sd19["ndl_dataset"])
        ratios.append(sd_ndl / sd_cl)
        colors = ["#59A14F" if value < 1 else "#E15759" for value in ratios]
        axis.bar(np.arange(4), ratios, color=colors)
        axis.axhline(1.0, color="#222", linestyle="--", linewidth=1)
        for index, value in enumerate(ratios):
            axis.text(index, value * (1.10 if log_scale else 1.03), f"{value:.2f}×", ha="center")
        if log_scale:
            axis.set_yscale("log")
            axis.set_ylim(min(0.2, min(ratios) * 0.7), max(ratios) * 1.7)
        else:
            axis.set_ylim(0, max(ratios) * 1.22)
        axis.set_xticks(np.arange(4), labels, rotation=20, ha="right")
        axis.set_title(title)
        axis.set_ylabel("NDL / matched CL; below 1 favors NDL")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Cross-dataset effect direction", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT / "cross_dataset_effects.png")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _style()
    aggregate = {row["condition"]: row for row in _load(AGGREGATE)}
    raw = _raw_conditions()
    for key in ORDER:
        seeds = sorted(int(row["seed"]) for row in raw[key])
        if seeds != list(range(42, 52)):
            raise ValueError(f"{key} has unexpected seeds: {seeds}")
    performance_figure(aggregate)
    per_class_figure(raw)
    mechanism_figure(aggregate, raw)
    claim_direction_figure(aggregate)
    cross_dataset_figure(aggregate)


if __name__ == "__main__":
    main()
