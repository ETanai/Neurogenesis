"""Run organic-growth ablations and explicitly labelled confirmation controls.

Architecture resemblance is reported afterward and is never part of candidate
selection or training.  Replay provenance is encoded in confirmation run names
and overrides; it must never be inferred from an output directory name.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from omegaconf import open_dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_early_stop_ablation import _compose_cfg  # noqa: E402
from scripts.run_experiments import run  # noqa: E402


INITIAL_SIZES = [200, 100, 75, 20]
PAPER_APPROXIMATE_SIZES = [225, 135, 84, 40]
PUBLISHED_INCREMENTAL_ORDER = [0, 2, 3, 4, 5, 6, 8, 9]

BASE_OVERRIDES = (
    "data=mnist",
    "experiment=mnist_incremental",
    "experiment.regime=ndl_ir",
    "experiment.base_classes=[1,7]",
    "replay.enabled=true",
    "replay.mode=dataset",
    "replay.sampling_mode=paper",
    "replay.per_class_batch_ratio=1.0",
    "neurogenesis.thresholds=null",
    "neurogenesis.objective_mode=paper_level_ae",
    "neurogenesis.next_layer_optimization=paper_columns",
    "neurogenesis.early_stop=null",
    "neurogenesis.shape_pressure_mode=none",
    "neurogenesis.max_outlier_fraction=0.1",
    "neurogenesis.unresolved_outlier_action=record",
    "neurogenesis.outlier_criterion_diagnostics.enabled=true",
    "neurogenesis.outlier_criterion_diagnostics.levels=[]",
    "neurogenesis.outlier_criterion_diagnostics.percentiles=[0.5,0.9,0.95,0.985,0.995]",
    "training.base_lr=0.001",
    "training.pretrain_epochs=50",
    "training.pretrain_mode=stacked_denoising",
    "training.denoising_dropout=0.1",
    "training.denoising_std=0.0",
    "training.pretrain_finetune_epochs=0",
    "neurogenesis.plasticity_epochs=100",
    "neurogenesis.stability_epochs=500",
    "neurogenesis.next_layer_epochs=100",
    "neurogenesis.plasticity_decoder_lr_ratio=0.01",
    "neurogenesis.stability_lr_ratio=0.01",
    "neurogenesis.next_layer_lr_ratio=0.01",
    "neurogenesis.stability_replay_mode=paper",
    "neurogenesis.stability_replay_per_class_ratio=1.0",
    "neurogenesis.stability_schedule=mixed",
    "experiment.threshold.percentile=0.985",
    "logging.mlflow.enabled=false",
)


@dataclass(frozen=True)
class OrganicGrowthSpec:
    stage: str
    name: str
    family: str
    description: str
    overrides: tuple[str, ...]
    paired_to: str | None = None
    base_checkpoint_group: str | None = None


def _policy_overrides(
    *,
    mode: str,
    absolute_nodes: int = 1,
    factor: float = 0.01,
    class_limit: Sequence[int] | None = None,
    stream_limit: Sequence[int] | None = None,
) -> tuple[str, ...]:
    return (
        f"neurogenesis.growth_mode={mode}",
        f"neurogenesis.absolute_new_nodes={int(absolute_nodes)}",
        f"neurogenesis.factor_new_nodes={float(factor)}",
        "neurogenesis.max_nodes_per_class="
        + ("null" if class_limit is None else str(list(class_limit)).replace(" ", "")),
        "neurogenesis.max_nodes_stream="
        + ("null" if stream_limit is None else str(list(stream_limit)).replace(" ", "")),
    )


def screen_specs() -> list[OrganicGrowthSpec]:
    screen = "experiment.incremental_classes=[0,2,3]"
    return [
        OrganicGrowthSpec(
            "screen",
            "cap_driven_reference",
            "reference",
            "Current proportional growth with the published-shape cumulative allowance.",
            (
                screen,
                "neurogenesis.max_nodes=[25,35,8,20]",
                "neurogenesis.max_nodes_scope=global",
                *_policy_overrides(mode="proportional", factor=0.01),
            ),
        ),
        OrganicGrowthSpec(
            "screen",
            "organic_a_abs1_stream2x",
            "loose_stream",
            "One node per round under a 2x stream safety ceiling.",
            (screen, *_policy_overrides(mode="absolute", absolute_nodes=1, stream_limit=[50, 70, 16, 40])),
        ),
        OrganicGrowthSpec(
            "screen",
            "organic_b_abs2_stream2x",
            "loose_stream",
            "Two nodes per round under a 2x stream safety ceiling.",
            (screen, *_policy_overrides(mode="absolute", absolute_nodes=2, stream_limit=[50, 70, 16, 40])),
        ),
        OrganicGrowthSpec(
            "screen",
            "organic_c_abs1_class_small",
            "class_throttle",
            "One node per round with a small class-local throttle.",
            (screen, *_policy_overrides(mode="absolute", absolute_nodes=1, class_limit=[4, 5, 2, 3])),
        ),
        OrganicGrowthSpec(
            "screen",
            "organic_d_abs2_class_small",
            "class_throttle",
            "Two nodes per round with a small class-local throttle.",
            (screen, *_policy_overrides(mode="absolute", absolute_nodes=2, class_limit=[4, 5, 2, 3])),
        ),
        OrganicGrowthSpec(
            "screen",
            "organic_e_abs1_hybrid",
            "hybrid",
            "One node per round with independent class throttle and loose stream ceiling.",
            (
                screen,
                *_policy_overrides(
                    mode="absolute",
                    absolute_nodes=1,
                    class_limit=[4, 5, 2, 3],
                    stream_limit=[100, 140, 32, 80],
                ),
            ),
        ),
        OrganicGrowthSpec(
            "screen",
            "organic_f_prop0005_stream2x",
            "fine_proportional",
            "Small proportional steps under a 2x stream safety ceiling.",
            (screen, *_policy_overrides(mode="proportional", factor=0.0005, stream_limit=[50, 70, 16, 40])),
        ),
    ]


def invariance_specs() -> list[OrganicGrowthSpec]:
    screen = "experiment.incremental_classes=[0,2,3]"
    return [
        OrganicGrowthSpec(
            "invariance", "organic_a_abs1_stream4x", "loose_stream", "Cap-invariance pair for organic A.",
            (screen, *_policy_overrides(mode="absolute", absolute_nodes=1, stream_limit=[100, 140, 32, 80])),
            paired_to="organic_a_abs1_stream2x",
        ),
        OrganicGrowthSpec(
            "invariance", "organic_b_abs2_stream4x", "loose_stream", "Cap-invariance pair for organic B.",
            (screen, *_policy_overrides(mode="absolute", absolute_nodes=2, stream_limit=[100, 140, 32, 80])),
            paired_to="organic_b_abs2_stream2x",
        ),
        OrganicGrowthSpec(
            "invariance", "organic_e_abs1_hybrid_stream8x", "hybrid", "Cap-invariance pair for organic E.",
            (
                screen,
                *_policy_overrides(
                    mode="absolute", absolute_nodes=1, class_limit=[4, 5, 2, 3],
                    stream_limit=[200, 280, 64, 160],
                ),
            ),
            paired_to="organic_e_abs1_hybrid",
        ),
        OrganicGrowthSpec(
            "invariance", "organic_c_abs1_class_double", "class_throttle", "Throttle-sensitivity pair for organic C.",
            (screen, *_policy_overrides(mode="absolute", absolute_nodes=1, class_limit=[8, 10, 4, 6])),
            paired_to="organic_c_abs1_class_small",
        ),
    ]


def full_specs() -> list[OrganicGrowthSpec]:
    full_order = "experiment.incremental_classes=" + str(PUBLISHED_INCREMENTAL_ORDER).replace(" ", "")
    result: list[OrganicGrowthSpec] = []
    for spec in screen_specs():
        overrides = tuple(
            full_order if item.startswith("experiment.incremental_classes=") else item
            for item in spec.overrides
        )
        result.append(
            OrganicGrowthSpec(
                "full", spec.name, spec.family,
                f"Full published curriculum: {spec.description}", overrides, spec.paired_to
            )
        )
    return result


def confirmation_specs() -> list[OrganicGrowthSpec]:
    """Frozen full-curriculum matrix for replay and classical controls.

    ``dataset_oracle`` deliberately retains original old-class images and is
    the clean upper bound requested for debugging.  ``intrinsic`` isolates old
    data after base training.  ``no_replay`` has no replay object at all.
    Classical controls are parameter-matched to the corresponding final NDL
    architecture observed in the five-seed pilot.
    """
    full_order = "experiment.incremental_classes=" + str(
        PUBLISHED_INCREMENTAL_ORDER
    ).replace(" ", "")
    normalized_growth = (
        full_order,
        "training.incremental_epochs=3",
        "neurogenesis.plasticity_epochs=3",
        "neurogenesis.stability_epochs=11",
        "neurogenesis.next_layer_epochs=3",
        "model.activation=sigmoid",
        "model.activation_latent=identity",
    )
    organic_policy = _policy_overrides(
        mode="absolute", absolute_nodes=1, class_limit=[4, 5, 2, 3]
    )
    refresh = (
        "experiment.threshold.refresh_before_class=true",
        "experiment.threshold.refresh_samples_per_class=1024",
    )
    return [
        OrganicGrowthSpec(
            "confirmation",
            "ndl_dataset_oracle_refresh",
            "ndl_dataset_oracle",
            "NDL with original-data replay and clean learned-class threshold refresh.",
            (
                *normalized_growth,
                *organic_policy,
                *refresh,
                "experiment.regime=ndl_ir",
                "replay.enabled=true",
                "replay.mode=dataset",
                "experiment.threshold.refresh_source=dataset",
            ),
            base_checkpoint_group="ndl_200_sigmoid",
        ),
        OrganicGrowthSpec(
            "confirmation",
            "ndl_intrinsic_refresh",
            "ndl_intrinsic",
            "NDL with intrinsic replay only after base training and replay-sourced refresh.",
            (
                *normalized_growth,
                *organic_policy,
                *refresh,
                "experiment.regime=ndl_ir",
                "replay.enabled=true",
                "replay.mode=intrinsic",
                "replay.reuse_previous_stats=true",
                "experiment.threshold.refresh_source=replay",
            ),
            base_checkpoint_group="ndl_200_sigmoid",
        ),
        OrganicGrowthSpec(
            "confirmation",
            "ndl_no_replay_refresh",
            "ndl_no_replay",
            "NDL without replay; clean learned-class data is used only to refresh thresholds.",
            (
                *normalized_growth,
                *organic_policy,
                *refresh,
                "experiment.regime=ndl",
                "replay.enabled=false",
                "replay.mode=dataset",
                "experiment.threshold.refresh_source=dataset",
            ),
            base_checkpoint_group="ndl_200_sigmoid",
        ),
        OrganicGrowthSpec(
            "confirmation",
            "cl_dataset_oracle_matched",
            "cl_dataset_oracle",
            "Classical learning with original-data replay, matched to clean NDL widths.",
            (
                *normalized_growth,
                "experiment.regime=cl_ir",
                "experiment.control_hidden_sizes=[207,105,77,23]",
                "replay.enabled=true",
                "replay.mode=dataset",
            ),
            paired_to="ndl_dataset_oracle_refresh",
            base_checkpoint_group="cl_207_sigmoid",
        ),
        OrganicGrowthSpec(
            "confirmation",
            "cl_intrinsic_matched",
            "cl_intrinsic",
            "Classical learning with intrinsic replay, matched to intrinsic NDL widths.",
            (
                *normalized_growth,
                "experiment.regime=cl_ir",
                "experiment.control_hidden_sizes=[232,140,91,44]",
                "replay.enabled=true",
                "replay.mode=intrinsic",
                "replay.reuse_previous_stats=true",
            ),
            paired_to="ndl_intrinsic_refresh",
            base_checkpoint_group="cl_232_sigmoid",
        ),
        OrganicGrowthSpec(
            "confirmation",
            "cl_no_replay_matched",
            "cl_no_replay",
            "Classical learning without replay, matched to intrinsic NDL widths.",
            (
                *normalized_growth,
                "experiment.regime=cl",
                "experiment.control_hidden_sizes=[232,140,91,44]",
                "replay.enabled=false",
                "replay.mode=dataset",
            ),
            paired_to="ndl_no_replay_refresh",
            base_checkpoint_group="cl_232_sigmoid",
        ),
    ]


def specs_for_stage(stage: str) -> list[OrganicGrowthSpec]:
    specs: list[OrganicGrowthSpec] = []
    if stage in {"screen", "all"}:
        specs.extend(screen_specs())
    if stage in {"invariance", "all"}:
        specs.extend(invariance_specs())
    if stage in {"full", "all"}:
        specs.extend(full_specs())
    if stage == "confirmation":
        specs.extend(confirmation_specs())
    return specs


def _final_top_records(eval_records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if not eval_records:
        return []
    final_step = max(int(record["step"]) for record in eval_records)
    top_layer = max(int(record["layer"]) for record in eval_records)
    return [
        record for record in eval_records
        if int(record["step"]) == final_step and int(record["layer"]) == top_layer
    ]


def summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    model = result["model"]
    training_stats = result["training_stats"]
    reports = list(result.get("growth_reports", {}).values())
    level_reports = [level for report in reports for level in report.get("levels", [])]
    final_records = _final_top_records(result.get("eval_records", []))
    aggregate = next((row for row in final_records if row.get("scope") == "aggregate"), {})
    class_records = [row for row in final_records if row.get("scope") == "class"]

    class_history: dict[int, list[float]] = {}
    top_layer = len(model.hidden_sizes) - 1
    for row in result.get("eval_records", []):
        if row.get("scope") != "class" or int(row.get("layer", -1)) != top_layer:
            continue
        class_history.setdefault(int(row["class_id"]), []).append(float(row["mean"]))
    forgetting = [max(values[-1] - min(values), 0.0) for values in class_history.values() if values]

    quota_stops = sum(level.get("stop_reason") == "outlier_quota_reached" for level in level_reports)
    updated_classes = sum(int(report.get("model_update_steps", 0)) > 0 for report in reports)
    later_reports = reports[2:]
    later_updated_classes = sum(
        int(report.get("model_update_steps", 0)) > 0 for report in later_reports
    )
    top_level_reports = [
        report.get("levels", [])[-1] for report in reports if report.get("levels")
    ]
    acquisition_gains = [
        float(level["initial_mean_mse"]) - float(level["final_mean_mse"])
        for level in top_level_reports
        if level.get("initial_mean_mse") is not None
        and level.get("final_mean_mse") is not None
    ]
    accepted_nodes = sum(int(level.get("nodes_accepted", 0)) for level in level_reports)
    widths = [int(size) for size in model.hidden_sizes]
    added = [width - initial for width, initial in zip(widths, INITIAL_SIZES)]
    return {
        "final_widths": widths,
        "added_widths": added,
        "strict_funnel": all(left > right for left, right in zip(widths, widths[1:])),
        "paper_endpoint_relative_l1": sum(
            abs(width - target) for width, target in zip(widths, PAPER_APPROXIMATE_SIZES)
        ) / sum(PAPER_APPROXIMATE_SIZES),
        "macro_mse": aggregate.get("mean"),
        "foreground_mse": aggregate.get("foreground_mse"),
        "per_class_mse": {
            str(row["class_id"]): float(row["mean"]) for row in class_records
        },
        "mean_positive_forgetting": (
            sum(forgetting) / len(forgetting) if forgetting else 0.0
        ),
        "quota_stop_fraction": quota_stops / len(level_reports) if level_reports else 0.0,
        "updated_class_fraction": updated_classes / len(reports) if reports else 0.0,
        "later_updated_class_fraction": (
            later_updated_classes / len(later_reports) if later_reports else 0.0
        ),
        "updated_classes": [
            report.get("class_id")
            for report in reports
            if int(report.get("model_update_steps", 0)) > 0
        ],
        "mean_incoming_acquisition_gain": (
            sum(acquisition_gains) / len(acquisition_gains) if acquisition_gains else 0.0
        ),
        "acquisition_gain_per_accepted_node": (
            sum(acquisition_gains) / accepted_nodes if accepted_nodes else 0.0
        ),
        "unresolved_exhausted_level_count": sum(
            bool(level.get("unresolved_outliers")) and bool(level.get("budget_exhausted"))
            for level in level_reports
        ),
        "model_update_steps": int(
            training_stats.get("neurogenesis_parameter_updates", 0)
            or training_stats.get("incremental_parameter_updates", 0)
        ),
        "neurogenesis_parameter_updates": int(
            training_stats.get("neurogenesis_parameter_updates", 0)
        ),
        "incremental_parameter_updates": int(
            training_stats.get("incremental_parameter_updates", 0)
        ),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "growth_reports": result.get("growth_reports", {}),
    }


def annotate_comparisons(rows: list[dict[str, Any]]) -> None:
    """Attach paired baseline and cap-invariance checks to completed rows."""
    by_stage_name_seed = {
        (str(row.get("stage")), str(row.get("name")), int(row.get("seed", -1))): row
        for row in rows
        if row.get("status") == "completed"
    }
    for row in rows:
        if row.get("status") != "completed":
            continue
        seed = int(row["seed"])
        comparison_stage = "screen" if row.get("stage") == "invariance" else str(row["stage"])
        baseline = by_stage_name_seed.get(
            (comparison_stage, "cap_driven_reference", seed)
        )
        if baseline is not None and row is not baseline:
            baseline_mse = float(baseline.get("macro_mse") or 0.0)
            row["mse_relative_to_reference"] = (
                None
                if baseline_mse == 0.0
                else float(row["macro_mse"]) / baseline_mse - 1.0
            )
            baseline_params = int(baseline.get("parameter_count") or 0)
            row["parameter_reduction_vs_reference"] = (
                0.0
                if baseline_params == 0
                else 1.0 - int(row["parameter_count"]) / baseline_params
            )
            baseline_updates = int(baseline.get("model_update_steps") or 0)
            row["update_reduction_vs_reference"] = (
                0.0
                if baseline_updates == 0
                else 1.0 - int(row["model_update_steps"]) / baseline_updates
            )

        paired_to = row.get("paired_to")
        if paired_to:
            lower = by_stage_name_seed.get(("screen", str(paired_to), seed))
            if lower is not None:
                lower_added = lower.get("added_widths", [])
                upper_added = row.get("added_widths", [])
                width_changes = [
                    abs(float(upper) - float(lower_value)) / max(abs(float(lower_value)), 1.0)
                    for lower_value, upper in zip(lower_added, upper_added)
                ]
                lower_mse = float(lower.get("macro_mse") or 0.0)
                mse_change = (
                    float("inf")
                    if lower_mse == 0.0
                    else abs(float(row["macro_mse"]) / lower_mse - 1.0)
                )
                passed = bool(
                    width_changes
                    and all(change <= 0.10 for change in width_changes)
                    and mse_change <= 0.05
                )
                row["cap_invariance_width_changes"] = width_changes
                row["cap_invariance_mse_change"] = mse_change
                row["cap_invariance_pass"] = passed
                lower["cap_invariance_width_changes"] = width_changes
                lower["cap_invariance_mse_change"] = mse_change
                lower["cap_invariance_pass"] = passed

    for row in rows:
        if row.get("status") != "completed" or row.get("family") == "reference":
            continue
        performance_ok = row.get("mse_relative_to_reference") is not None and float(
            row["mse_relative_to_reference"]
        ) <= 0.05
        efficiency_ok = max(
            float(row.get("parameter_reduction_vs_reference") or 0.0),
            float(row.get("update_reduction_vs_reference") or 0.0),
        ) >= 0.15
        row["preliminary_organicity_gate"] = bool(
            float(row.get("quota_stop_fraction") or 0.0) >= 0.80
            and float(row.get("updated_class_fraction") or 0.0) >= 0.50
            and row.get("cap_invariance_pass") is True
            and performance_ok
            and (
                float(row.get("mean_positive_forgetting") or 0.0)
                <= float(
                    next(
                        (
                            candidate.get("mean_positive_forgetting") or 0.0
                            for candidate in rows
                            if candidate.get("name") == "cap_driven_reference"
                            and candidate.get("seed") == row.get("seed")
                            and candidate.get("stage")
                            == (
                                "screen"
                                if row.get("stage") == "invariance"
                                else row.get("stage")
                            )
                        ),
                        float("inf"),
                    )
                )
                or efficiency_ok
            )
        )


def _quick_overrides() -> tuple[str, ...]:
    return (
        "training.pretrain_epochs=1",
        "experiment.incremental_train_limit_per_class=16",
        "neurogenesis.plasticity_epochs=1",
        "neurogenesis.stability_epochs=1",
        "neurogenesis.next_layer_epochs=1",
        "data.num_workers=0",
    )


def _selected(specs: Iterable[OrganicGrowthSpec], only: str | None) -> list[OrganicGrowthSpec]:
    if not only:
        return list(specs)
    selectors = {value.strip() for value in only.split(",") if value.strip()}
    return [
        spec for spec in specs
        if spec.name in selectors or spec.family in selectors or spec.stage in selectors
    ]


def _write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    annotate_comparisons(rows)

    def json_safe(value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        return value

    (output_dir / "summary.json").write_text(
        json.dumps(json_safe(rows), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    columns = [
        "stage", "name", "family", "seed", "paired_to", "status", "macro_mse",
        "foreground_mse", "mean_positive_forgetting", "quota_stop_fraction",
        "updated_class_fraction", "unresolved_exhausted_level_count", "final_widths",
        "later_updated_class_fraction", "strict_funnel", "paper_endpoint_relative_l1",
        "model_update_steps", "parameter_count", "runtime_seconds",
        "mse_relative_to_reference", "cap_invariance_pass", "preliminary_organicity_gate",
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["final_widths"] = " ".join(map(str, row.get("final_widths", [])))
            writer.writerow(csv_row)
    lines = [
        "# Organic Growth Ablation", "",
        "Replay provenance is explicit in each confirmation run name and override manifest.", "",
        "| Stage | Run | Seed | MSE | Foreground MSE | Widths | Quota stops | Updated classes | Unresolved exhausted |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        def fmt(value: Any) -> str:
            return "" if value is None else f"{float(value):.5f}"
        lines.append(
            f"| {row['stage']} | {row['name']} | {row['seed']} | "
            f"{fmt(row.get('macro_mse'))} | {fmt(row.get('foreground_mse'))} | "
            f"{' '.join(map(str, row.get('final_widths', [])))} | "
            f"{fmt(row.get('quota_stop_fraction'))} | "
            f"{fmt(row.get('updated_class_fraction'))} | "
            f"{row.get('unresolved_exhausted_level_count', '')} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required")
    return seeds


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("screen", "invariance", "full", "confirmation", "all"),
        default="screen",
    )
    parser.add_argument("--seeds", type=_parse_seeds, default=[42, 43, 44])
    parser.add_argument("--only", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-base-checkpoint-cache", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep completed rows in an existing summary and skip those identities.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (
        REPO_ROOT / "outputs" / "ablations" / "organic_growth" / timestamp
    )
    specs = _selected(specs_for_stage(args.stage), args.only)
    rows: list[dict[str, Any]] = []
    summary_path = output_dir / "summary.json"
    if args.resume and summary_path.is_file():
        loaded = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise ValueError(f"Expected a list in {summary_path}")
        rows = loaded
    completed = {
        (str(row.get("stage")), str(row.get("name")), int(row.get("seed", -1)))
        for row in rows
        if row.get("status") == "completed"
    }
    extra = (*(_quick_overrides() if args.quick else ()), *tuple(args.override))
    checkpoint_dir = output_dir / "base_checkpoints"
    if not args.no_base_checkpoint_cache:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        for seed in args.seeds:
            run_key = (spec.stage, spec.name, int(seed))
            if run_key in completed:
                print(f"=== skipping completed {spec.stage}/{spec.name}/seed_{seed} ===")
                continue
            rows = [
                row
                for row in rows
                if (
                    str(row.get("stage")),
                    str(row.get("name")),
                    int(row.get("seed", -1)),
                )
                != run_key
            ]
            checkpoint_overrides: tuple[str, ...] = ()
            if not args.no_base_checkpoint_cache:
                group = spec.base_checkpoint_group or "shared"
                checkpoint_path = (checkpoint_dir / f"{group}_seed_{seed}.pt").resolve()
                if checkpoint_path.exists():
                    checkpoint_overrides = (
                        f"training.base_checkpoint={checkpoint_path.as_posix()}",
                    )
                else:
                    checkpoint_overrides = (
                        f"training.base_checkpoint_out={checkpoint_path.as_posix()}",
                    )
            overrides = (
                *BASE_OVERRIDES,
                *spec.overrides,
                f"seed={seed}",
                *checkpoint_overrides,
                *extra,
            )
            print(f"\n=== {spec.stage}/{spec.name}/seed_{seed} ===")
            print(spec.description)
            print(" ".join(overrides))
            identity = {
                "stage": spec.stage,
                "name": spec.name,
                "family": spec.family,
                "seed": seed,
                "paired_to": spec.paired_to,
                "description": spec.description,
                "overrides": list(overrides),
            }
            if args.dry_run:
                rows.append({**identity, "status": "dry_run"})
                _write_summary(output_dir, rows)
                continue
            cfg = _compose_cfg(overrides)
            with open_dict(cfg):
                cfg.paper_experiment = f"organic_growth/{spec.stage}/{spec.name}/seed_{seed}"
            try:
                started = time.perf_counter()
                if args.quiet:
                    log_dir = output_dir / "logs"
                    log_dir.mkdir(parents=True, exist_ok=True)
                    log_path = log_dir / f"{spec.stage}_{spec.name}_seed_{seed}.log"
                    with log_path.open("w", encoding="utf-8") as handle:
                        with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(
                            handle
                        ):
                            run_result = run(cfg)
                    identity["log_path"] = str(log_path)
                else:
                    run_result = run(cfg)
                summary = summarize_result(run_result)
                summary["runtime_seconds"] = time.perf_counter() - started
                rows.append({**identity, "status": "completed", **summary})
            except Exception as exc:
                rows.append({**identity, "status": "failed", "error": repr(exc)})
                _write_summary(output_dir, rows)
                raise
            _write_summary(output_dir, rows)


if __name__ == "__main__":
    main()
