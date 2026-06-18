"""Run paper-specified NDL ablations over under-specified training knobs."""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_early_stop_ablation import (  # noqa: E402
    AblationSpec,
    RunSummary,
    _best_completed,
    _default_tracking_uri,
    _filter_specs,
    _run_spec,
    _with_extra,
)


INITIAL_SIZES = [200, 100, 75, 20]
INCREMENTAL_CLASSES = [0, 2, 3, 4, 5, 6, 8, 9]
PRACTICAL_BEST_CLEAN = 0.00930908
PRACTICAL_BEST_IR = 0.02020145

PAPER_LOCKED_OVERRIDES = (
    "data=mnist",
    "experiment=mnist_incremental",
    "experiment.regime=ndl_ir",
    "experiment.base_classes=[1,7]",
    "experiment.incremental_classes=[0,2,3,4,5,6,8,9]",
    "experiment.model.hidden_sizes=[200,100,75,20]",
    "neurogenesis.thresholds=null",
    "neurogenesis.objective_mode=paper_level_ae",
    "neurogenesis.next_layer_optimization=paper_columns",
    "neurogenesis.plasticity_decoder_lr_ratio=0.01",
    "neurogenesis.stability_replay_mode=paper",
    "replay.enabled=true",
    "replay.mode=intrinsic",
    "replay.ir_sampling_mode=gaussian_full",
    "replay.ir_cov_shrinkage=0.0",
    "replay.ir_noise_scale=1.0",
    "training.pretrain_mode=stacked_denoising",
    "training.pretrain_finetune_epochs=0",
)

LOCK_PREFIXES = {
    "neurogenesis.objective_mode": "paper_level_ae",
    "neurogenesis.next_layer_optimization": "paper_columns",
    "neurogenesis.plasticity_decoder_lr_ratio": "0.01",
    "neurogenesis.stability_replay_mode": "paper",
    "replay.ir_sampling_mode": "gaussian_full",
    "replay.ir_cov_shrinkage": "0.0",
    "replay.ir_noise_scale": "1.0",
    "training.pretrain_mode": "stacked_denoising",
    "training.pretrain_finetune_epochs": "0",
}


def _replace_or_add(overrides: Sequence[str], key: str, value: object) -> tuple[str, ...]:
    prefix = f"{key}="
    rendered = f"{key}={value}"
    out = [item for item in overrides if not item.startswith(prefix)]
    out.append(rendered)
    return tuple(out)


def _with_dataset(overrides: Sequence[str]) -> tuple[str, ...]:
    return _replace_or_add(overrides, "replay.mode", "dataset")


def _with_no_replay(overrides: Sequence[str]) -> tuple[str, ...]:
    out = _replace_or_add(overrides, "experiment.regime", "ndl")
    out = _replace_or_add(out, "replay.enabled", "false")
    out = _replace_or_add(out, "replay.mode", "intrinsic")
    return out


def _with_single0(overrides: Sequence[str]) -> tuple[str, ...]:
    return _replace_or_add(overrides, "experiment.incremental_classes", "[0]")


def _with_base_only(overrides: Sequence[str]) -> tuple[str, ...]:
    out = _replace_or_add(overrides, "experiment.skip_incremental_training", "true")
    return out


def _to_full(overrides: Sequence[str], *, replay_mode: str) -> tuple[str, ...]:
    out = [
        item
        for item in overrides
        if not item.startswith("experiment.incremental_classes=")
        and not item.startswith("experiment.skip_incremental_training=")
        and not item.startswith("replay.mode=")
    ]
    out.append("experiment.incremental_classes=[0,2,3,4,5,6,8,9]")
    out.append(f"replay.mode={replay_mode}")
    return tuple(out)


def _control_overrides(candidate: RunSummary) -> tuple[str, ...]:
    sizes = candidate.layer_sizes or INITIAL_SIZES
    out = [
        item
        for item in PAPER_LOCKED_OVERRIDES
        if not item.startswith("experiment.regime=")
        and not item.startswith("experiment.model.hidden_sizes=")
        and not item.startswith("neurogenesis.")
    ]
    out.extend(
        (
            "experiment.regime=cl_ir",
            f"experiment.control_hidden_sizes=[{','.join(str(size) for size in sizes)}]",
            "replay.enabled=true",
            "replay.mode=intrinsic",
            "replay.ir_sampling_mode=gaussian_full",
            "replay.ir_cov_shrinkage=0.0",
            "replay.ir_noise_scale=1.0",
            "training.pretrain_finetune_epochs=0",
        )
    )
    return tuple(dict.fromkeys(out))


def _guard_paper_locked(spec: AblationSpec) -> None:
    merged: dict[str, str] = {}
    for item in spec.overrides:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.lstrip("+")
        merged[key] = value
    for key, expected in LOCK_PREFIXES.items():
        actual = merged.get(key)
        if actual != expected:
            raise ValueError(
                f"{spec.name} violates paper lock {key}={expected}; got {actual!r}"
            )
    forbidden = (
        "neurogenesis.objective_mode=global_partial",
        "neurogenesis.objective_mode=full_reconstruction",
        "neurogenesis.objective_mode=local_plasticity_full_stability",
        "replay.ir_sampling_mode=gaussian_shrink",
        "replay.ir_sampling_mode=gaussian_diag",
        "replay.ir_sampling_mode=mean_only",
        "replay.ir_sampling_mode=mean_plus_noise",
    )
    joined = "\n".join(spec.overrides)
    for token in forbidden:
        if token in joined:
            raise ValueError(f"{spec.name} uses non-paper override {token}")


def _validate_specs(specs: Iterable[AblationSpec]) -> None:
    for spec in specs:
        if spec.family != "control":
            _guard_paper_locked(spec)


def baseline_specs() -> list[AblationSpec]:
    return [
        AblationSpec(
            "baseline",
            "literal_ndl_ir",
            "Literal paper-local NDL+IR baseline, no finetune.",
            PAPER_LOCKED_OVERRIDES,
            "baseline_ir",
        ),
        AblationSpec(
            "baseline",
            "literal_ndl_dataset",
            "Literal paper-local NDL with clean dataset replay upper bound.",
            _with_dataset(PAPER_LOCKED_OVERRIDES),
            "baseline_dataset",
        ),
        AblationSpec(
            "baseline",
            "literal_ndl_no_replay",
            "Literal paper-local NDL without replay.",
            _with_no_replay(PAPER_LOCKED_OVERRIDES),
            "baseline_no_replay",
        ),
    ]


def base_pretraining_specs() -> list[AblationSpec]:
    specs: list[AblationSpec] = []
    for lr in ("1e-4", "3e-4", "1e-3", "3e-3"):
        for epochs in (14, 28, 50, 100, 200):
            for dropout in (0.0, 0.05, 0.1, 0.2):
                for std in (0.0, 0.05):
                    for wd in ("0.0", "1e-5", "1e-4"):
                        name = (
                            f"base_lr{lr}_e{epochs}_drop{str(dropout).replace('.', 'p')}"
                            f"_std{str(std).replace('.', 'p')}_wd{wd.replace('-', 'm')}"
                        )
                        specs.append(
                            AblationSpec(
                                "base",
                                name,
                                "Base-only paper SHL-AE pretraining quality sweep.",
                                (
                                    *_with_base_only(PAPER_LOCKED_OVERRIDES),
                                    f"training.base_lr={lr}",
                                    f"training.pretrain_epochs={epochs}",
                                    f"training.denoising_dropout={dropout}",
                                    f"training.denoising_std={std}",
                                    f"training.weight_decay={wd}",
                                ),
                                "pretrain",
                            )
                        )
    return specs


def _seed_pretraining_specs() -> list[AblationSpec]:
    seeds = [
        ("seed_default", "1e-4", 14, 0.1, 0.0, "0.0"),
        ("seed_current_best_lr", "1e-3", 50, 0.1, 0.0, "0.0"),
        ("seed_long_local", "1e-3", 100, 0.1, 0.0, "0.0"),
    ]
    return [
        AblationSpec(
            "base_seed",
            name,
            "Fallback pretraining seed when no completed base sweep is available.",
            (
                *_with_base_only(PAPER_LOCKED_OVERRIDES),
                f"training.base_lr={lr}",
                f"training.pretrain_epochs={epochs}",
                f"training.denoising_dropout={dropout}",
                f"training.denoising_std={std}",
                f"training.weight_decay={wd}",
            ),
            "pretrain",
        )
        for name, lr, epochs, dropout, std, wd in seeds
    ]


def single0_schedule_specs(pretrain_candidates: Sequence[RunSummary]) -> list[AblationSpec]:
    specs: list[AblationSpec] = []
    seeds = pretrain_candidates or [
        _summary_seed(spec) for spec in _seed_pretraining_specs()[:2]
    ]
    for seed_idx, candidate in enumerate(seeds, start=1):
        base = _with_single0(tuple(candidate.overrides))
        base = tuple(item for item in base if item != "experiment.skip_incremental_training=true")
        for p_epochs in (100, 500, 1000, 2000):
            for s_epochs in (100, 500, 1000, 2000):
                for n_epochs in (100, 500, 1000):
                    for s_lr in ("0.01", "0.03", "0.1"):
                        for n_lr in ("0.01", "0.03", "0.1"):
                            name = (
                                f"s0_{seed_idx}_p{p_epochs}_s{s_epochs}_n{n_epochs}"
                                f"_slr{s_lr.replace('.', 'p')}_nlr{n_lr.replace('.', 'p')}"
                            )
                            specs.append(
                                AblationSpec(
                                    "single0_schedule",
                                    name,
                                    "Single-0 gate: phase length and LR-ratio sweep.",
                                    (
                                        *base,
                                        f"neurogenesis.plasticity_epochs={p_epochs}",
                                        f"neurogenesis.stability_epochs={s_epochs}",
                                        f"neurogenesis.next_layer_epochs={n_epochs}",
                                        f"neurogenesis.stability_lr_ratio={s_lr}",
                                        f"neurogenesis.next_layer_lr_ratio={n_lr}",
                                    ),
                                    "single0_schedule",
                                )
                            )
    return specs


def threshold_growth_specs(schedule_candidates: Sequence[RunSummary]) -> list[AblationSpec]:
    specs: list[AblationSpec] = []
    seeds = schedule_candidates or [
        _summary_seed(single0_schedule_specs([])[0]),
        _summary_seed(single0_schedule_specs([])[1]),
    ]
    for seed_idx, candidate in enumerate(seeds, start=1):
        for percentile in (0.95, 0.975, 0.985, 0.995):
            for max_frac in (0.05, 0.1, 0.2):
                for l3_frac in (0.2, 0.3, 0.45):
                    for factor_new in (0.001, 0.002, 0.005, 0.01):
                        for factor_max in (0.05, 0.1, 0.2):
                            name = (
                                f"tg_{seed_idx}_pct{str(percentile).replace('.', 'p')}"
                                f"_q{str(max_frac).replace('.', 'p')}"
                                f"_l3{str(l3_frac).replace('.', 'p')}"
                                f"_fn{str(factor_new).replace('.', 'p')}"
                                f"_fm{str(factor_max).replace('.', 'p')}"
                            )
                            specs.append(
                                AblationSpec(
                                    "single0_threshold_growth",
                                    name,
                                    "Single-0 gate: threshold, quota, and growth ambiguity sweep.",
                                    (
                                        *candidate.overrides,
                                        f"experiment.threshold.percentile={percentile}",
                                        f"neurogenesis.max_outlier_fraction={max_frac}",
                                        f"+neurogenesis.max_outlier_fraction_by_level.3={l3_frac}",
                                        f"neurogenesis.factor_new_nodes={factor_new}",
                                        f"neurogenesis.factor_max_new_nodes={factor_max}",
                                    ),
                                    "threshold_growth",
                                )
                            )
    return specs


def ir_replay_specs(clean_candidates: Sequence[RunSummary]) -> list[AblationSpec]:
    specs: list[AblationSpec] = []
    seeds = clean_candidates or [_summary_seed(threshold_growth_specs([])[0])]
    for seed_idx, candidate in enumerate(seeds, start=1):
        base = _to_full(candidate.overrides, replay_mode="intrinsic")
        for ratio in (0.25, 0.5, 1.0, 2.0):
            for cov_eps in ("1e-5", "1e-4", "1e-3"):
                name = f"ir_{seed_idx}_r{str(ratio).replace('.', 'p')}_eps{cov_eps.replace('-', 'm')}"
                specs.append(
                    AblationSpec(
                        "ir",
                        name,
                        "Full MNIST NDL+IR replay amount/covariance sweep.",
                        (
                            *base,
                            f"neurogenesis.stability_replay_per_class_ratio={ratio}",
                            f"replay.cov_eps={cov_eps}",
                        ),
                        "ir",
                    )
                )
    return specs


def full_promotion_specs(candidates: Sequence[RunSummary]) -> list[AblationSpec]:
    specs: list[AblationSpec] = []
    seeds = candidates or [
        _summary_seed(single0_schedule_specs([])[0]),
        _summary_seed(threshold_growth_specs([])[0]),
        _summary_seed(ir_replay_specs([])[0]),
    ]
    for idx, candidate in enumerate(seeds, start=1):
        for mode in ("intrinsic", "dataset"):
            specs.append(
                AblationSpec(
                    "full",
                    f"full_{mode}_{idx}_{candidate.name}",
                    f"Full MNIST promotion from {candidate.name} using {mode} replay.",
                    _to_full(candidate.overrides, replay_mode=mode),
                    f"full_{mode}",
                )
            )
    return specs


def control_specs(full_candidates: Sequence[RunSummary]) -> list[AblationSpec]:
    specs: list[AblationSpec] = []
    for idx, candidate in enumerate(full_candidates, start=1):
        specs.append(
            AblationSpec(
                "controls",
                f"matched_cl_ir_{idx}_{candidate.name}",
                f"Matched CL+IR control using final sizes from {candidate.name}.",
                _control_overrides(candidate),
                "control",
            )
        )
    return specs


def _summary_seed(spec: AblationSpec) -> RunSummary:
    return RunSummary(
        stage=spec.stage,
        name=spec.name,
        description=spec.description,
        family=spec.family,
        run_name=spec.name,
        run_id=None,
        overrides=list(spec.overrides),
        final_l3_mse=None,
        layer_sizes=[],
        per_class_l3_mse={},
        diagnostics={},
        score=None,
        status="seed",
    )


def _quick_overrides() -> tuple[str, ...]:
    return (
        "training.pretrain_epochs=1",
        "training.pretrain_finetune_epochs=0",
        "experiment.incremental_train_limit_per_class=16",
        "neurogenesis.plasticity_epochs=2",
        "neurogenesis.stability_epochs=2",
        "neurogenesis.next_layer_epochs=2",
        "neurogenesis.max_nodes=[2,2,2,2]",
        "logging.mlflow.metric_filter.class_metrics=summary",
    )


def _paper_score(summary: RunSummary) -> float:
    if summary.final_l3_mse is None or not math.isfinite(float(summary.final_l3_mse)):
        return float("inf")
    sizes = summary.layer_sizes
    l3_l2 = (sizes[3] / sizes[2]) if len(sizes) >= 4 and sizes[2] else 10.0
    retention = 0.0
    for cls in ("1", "7"):
        if cls in summary.per_class_l3_mse:
            retention += summary.per_class_l3_mse[cls]
    caps = summary.diagnostics.get("cap_hits_by_level", {}) if summary.diagnostics else {}
    return (
        float(summary.final_l3_mse)
        + 0.003 * max(0.0, l3_l2 - 1.0)
        + 0.2 * retention
        + 0.001 * float(caps.get("3") or 0.0)
    )


def _promotion_candidates(
    summaries: Sequence[RunSummary],
    *,
    stages: set[str],
    limit: int,
) -> list[RunSummary]:
    candidates = [
        item
        for item in summaries
        if item.stage in stages
        and item.status == "completed"
        and item.final_l3_mse is not None
    ]
    return sorted(candidates, key=_paper_score)[:limit]


def _run_specs(
    specs: Sequence[AblationSpec],
    *,
    args: argparse.Namespace,
    timestamp: str,
    output_dir: Path,
    summaries: list[RunSummary],
    max_runs_state: dict[str, int],
) -> list[RunSummary]:
    _validate_specs(specs)
    selected = _filter_specs(specs, args.only)
    if args.list_runs:
        for spec in selected:
            print(f"{spec.stage}/{spec.name}: {spec.description}")
        raise SystemExit(0)

    completed: list[RunSummary] = []
    extra = (*(_quick_overrides() if args.quick else ()), *tuple(args.override or ()))
    per_run_dir = output_dir / "per_run_configs"
    per_run_dir.mkdir(parents=True, exist_ok=True)

    for spec in selected:
        if args.max_runs is not None and max_runs_state["count"] >= args.max_runs:
            break
        spec = _with_extra(spec, extra)
        _guard_paper_locked(spec) if spec.family != "control" else None
        _write_run_config(per_run_dir, spec)
        summary = _run_spec(
            spec,
            tracking_uri=args.tracking_uri,
            experiment_name=args.experiment_name,
            timestamp=timestamp,
            dry_run=args.dry_run,
        )
        summary = _attach_paper_diagnostics(summary)
        summaries.append(summary)
        completed.append(summary)
        max_runs_state["count"] += 1
        _write_outputs(output_dir, summaries, _payload(args, timestamp))
    return completed


def _write_run_config(per_run_dir: Path, spec: AblationSpec) -> None:
    payload = {
        "stage": spec.stage,
        "name": spec.name,
        "description": spec.description,
        "family": spec.family,
        "overrides": list(spec.overrides),
    }
    path = per_run_dir / f"{spec.stage}__{spec.name}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _attach_paper_diagnostics(summary: RunSummary) -> RunSummary:
    diagnostics = dict(summary.diagnostics or {})
    sizes = summary.layer_sizes
    if len(sizes) >= 4:
        diagnostics["l3_l2_ratio"] = sizes[3] / sizes[2] if sizes[2] else None
        diagnostics["strict_funnel"] = bool(sizes[0] > sizes[1] > sizes[2] > sizes[3])
    retention = {
        cls: summary.per_class_l3_mse.get(cls)
        for cls in ("1", "7")
        if cls in summary.per_class_l3_mse
    }
    diagnostics["base_retention_l3_mse"] = retention
    return replace(summary, diagnostics=diagnostics, score=_paper_score(summary))


def _write_outputs(output_dir: Path, summaries: Sequence[RunSummary], payload: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = {**payload, "runs": [summary.as_dict() for summary in summaries]}
    (output_dir / "summary.json").write_text(
        json.dumps(data, indent=2) + "\n", encoding="utf-8"
    )

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "stage",
                "family",
                "name",
                "run_id",
                "final_l3_mse",
                "score",
                "layer_sizes",
                "l3_l2_ratio",
                "strict_funnel",
                "l3_cap_hits",
                "class_1_l3_mse",
                "class_7_l3_mse",
                "status",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            diagnostics = summary.diagnostics or {}
            writer.writerow(
                {
                    "stage": summary.stage,
                    "family": summary.family,
                    "name": summary.name,
                    "run_id": summary.run_id or "",
                    "final_l3_mse": summary.final_l3_mse
                    if summary.final_l3_mse is not None
                    else "",
                    "score": summary.score if summary.score is not None else "",
                    "layer_sizes": " ".join(map(str, summary.layer_sizes)),
                    "l3_l2_ratio": diagnostics.get("l3_l2_ratio", ""),
                    "strict_funnel": diagnostics.get("strict_funnel", ""),
                    "l3_cap_hits": diagnostics.get("cap_hits_by_level", {}).get("3", ""),
                    "class_1_l3_mse": summary.per_class_l3_mse.get("1", ""),
                    "class_7_l3_mse": summary.per_class_l3_mse.get("7", ""),
                    "status": summary.status,
                }
            )

    lines = [
        "# Paper-Specified Training Ablation Summary",
        "",
        "- paper locks: SHL-AE level objective, paper columns, LR/100 decoder plasticity, full-covariance Gaussian IR, no finetune",
        f"- practical best clean reference: `{PRACTICAL_BEST_CLEAN:.5f}`",
        f"- practical best IR reference: `{PRACTICAL_BEST_IR:.5f}`",
        "",
        "| Stage | Family | Run | L3 MSE | Score | Sizes | L3/L2 | Funnel | L3 caps | 1 MSE | 7 MSE | MLflow run |",
        "|---|---|---|---:|---:|---|---:|---|---:|---:|---:|---|",
    ]
    for summary in summaries:
        diagnostics = summary.diagnostics or {}
        caps = diagnostics.get("cap_hits_by_level", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    summary.stage,
                    summary.family,
                    summary.name,
                    _fmt(summary.final_l3_mse),
                    _fmt(summary.score),
                    " ".join(map(str, summary.layer_sizes)),
                    _fmt(diagnostics.get("l3_l2_ratio")),
                    str(diagnostics.get("strict_funnel", "")),
                    _fmt(caps.get("3")),
                    _fmt(summary.per_class_l3_mse.get("1")),
                    _fmt(summary.per_class_l3_mse.get("7")),
                    summary.run_id or "",
                ]
            )
            + " |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.5f}"


def _payload(args: argparse.Namespace, timestamp: str) -> dict:
    return {
        "timestamp": timestamp,
        "tracking_uri": args.tracking_uri,
        "experiment_name": args.experiment_name,
        "quick": bool(args.quick),
        "dry_run": bool(args.dry_run),
        "paper_locked_overrides": list(PAPER_LOCKED_OVERRIDES),
        "excluded": {
            "global_end_to_end_finetuning": True,
            "non_paper_ir_sampling_modes": True,
            "non_paper_objective_modes": True,
        },
        "promotion_score": (
            "mse + 0.003*max(0,l3_l2-1) + 0.2*(class1+class7 mse) + "
            "0.001*l3_cap_hits"
        ),
        "references": {
            "practical_best_clean_mse": PRACTICAL_BEST_CLEAN,
            "practical_best_ir_mse": PRACTICAL_BEST_IR,
            "initial_sizes": INITIAL_SIZES,
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("baseline", "base", "single0", "full", "ir", "controls", "all"),
        default="baseline",
    )
    parser.add_argument("--experiment-name", default="neurogenesis-diagnostics")
    parser.add_argument("--tracking-uri", default=_default_tracking_uri())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--promote-top", type=int, default=2)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-runs", action="store_true")
    parser.add_argument("--only", default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def _list_specs_for_stage(stage: str) -> list[AblationSpec]:
    if stage == "baseline":
        return baseline_specs()
    if stage == "base":
        return base_pretraining_specs()
    if stage == "single0":
        schedule = single0_schedule_specs([])
        return [*schedule, *threshold_growth_specs([_summary_seed(schedule[0])])]
    if stage == "ir":
        return ir_replay_specs([])
    if stage == "full":
        return full_promotion_specs([])
    if stage == "controls":
        return control_specs([
            RunSummary(
                "full",
                "example_full_candidate",
                "",
                "full_intrinsic",
                "",
                None,
                list(PAPER_LOCKED_OVERRIDES),
                0.01,
                [220, 130, 90, 50],
                {},
                {},
                0.01,
                "seed",
            )
        ])
    return [
        *baseline_specs(),
        *base_pretraining_specs(),
        *_list_specs_for_stage("single0"),
        *ir_replay_specs([]),
        *full_promotion_specs([]),
        *_list_specs_for_stage("controls"),
    ]


def main() -> None:
    args = _parse_args()
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (
        REPO_ROOT / "outputs" / "diagnostics" / "paper_spec_training_ablation" / timestamp
    )
    if args.list_runs:
        selected = _filter_specs(_list_specs_for_stage(args.stage), args.only)
        for spec in selected:
            print(f"{spec.stage}/{spec.name}: {spec.description}")
        return

    summaries: list[RunSummary] = []
    max_runs_state = {"count": 0}
    _run_spec.run_prefix = "paperspec"  # type: ignore[attr-defined]
    if args.quiet:
        _run_spec.quiet_output_dir = output_dir / "logs"  # type: ignore[attr-defined]

    if args.stage in {"baseline", "all"}:
        _run_specs(
            baseline_specs(),
            args=args,
            timestamp=timestamp,
            output_dir=output_dir,
            summaries=summaries,
            max_runs_state=max_runs_state,
        )

    base_results: list[RunSummary] = []
    if args.stage in {"base", "all"}:
        base_results = _run_specs(
            base_pretraining_specs(),
            args=args,
            timestamp=timestamp,
            output_dir=output_dir,
            summaries=summaries,
            max_runs_state=max_runs_state,
        )

    pretrain_candidates = _promotion_candidates(
        base_results or summaries, stages={"base"}, limit=max(args.promote_top, 2)
    )

    single0_results: list[RunSummary] = []
    if args.stage in {"single0", "all"}:
        schedule_results = _run_specs(
            single0_schedule_specs(pretrain_candidates),
            args=args,
            timestamp=timestamp,
            output_dir=output_dir,
            summaries=summaries,
            max_runs_state=max_runs_state,
        )
        schedule_candidates = _promotion_candidates(
            schedule_results or summaries,
            stages={"single0_schedule"},
            limit=max(args.promote_top, 2),
        )
        threshold_results = _run_specs(
            threshold_growth_specs(schedule_candidates),
            args=args,
            timestamp=timestamp,
            output_dir=output_dir,
            summaries=summaries,
            max_runs_state=max_runs_state,
        )
        single0_results = [*schedule_results, *threshold_results]

    clean_candidates = _promotion_candidates(
        single0_results or summaries,
        stages={"single0_schedule", "single0_threshold_growth"},
        limit=max(args.promote_top, 2),
    )

    ir_results: list[RunSummary] = []
    if args.stage in {"ir", "all"}:
        ir_results = _run_specs(
            ir_replay_specs(clean_candidates),
            args=args,
            timestamp=timestamp,
            output_dir=output_dir,
            summaries=summaries,
            max_runs_state=max_runs_state,
        )

    full_results: list[RunSummary] = []
    if args.stage in {"full", "all"}:
        full_candidates = [
            *_promotion_candidates(
                single0_results or summaries,
                stages={"single0_schedule"},
                limit=args.promote_top,
            ),
            *_promotion_candidates(
                single0_results or summaries,
                stages={"single0_threshold_growth"},
                limit=args.promote_top,
            ),
            *_promotion_candidates(ir_results or summaries, stages={"ir"}, limit=args.promote_top),
        ]
        full_results = _run_specs(
            full_promotion_specs(full_candidates),
            args=args,
            timestamp=timestamp,
            output_dir=output_dir,
            summaries=summaries,
            max_runs_state=max_runs_state,
        )

    if args.stage in {"controls", "all"}:
        control_candidates = _promotion_candidates(
            full_results or summaries, stages={"full"}, limit=args.promote_top
        )
        if not control_candidates and not args.dry_run:
            raise SystemExit("--stage controls requires completed full candidates.")
        _run_specs(
            control_specs(control_candidates),
            args=args,
            timestamp=timestamp,
            output_dir=output_dir,
            summaries=summaries,
            max_runs_state=max_runs_state,
        )

    _write_outputs(output_dir, summaries, _payload(args, timestamp))
    print(f"\nPaper-specified training ablation summary written to {output_dir}")


if __name__ == "__main__":
    main()
