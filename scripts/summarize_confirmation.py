"""Aggregate completed confirmation manifests with seed-level uncertainty.

Example::

    python scripts/summarize_confirmation.py \
      --input ndl_dataset=outputs/run_a/summary.json \
      --input ndl_dataset=outputs/run_b/summary.json \
      --output-dir outputs/confirmation_summary

Only rows explicitly marked ``completed`` are included. Duplicate seeds within
a condition are rejected so a resumed or superseded run cannot be counted
twice accidentally.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


METRICS = (
    "macro_mse",
    "foreground_mse",
    "mean_positive_forgetting",
    "quota_stop_fraction",
    "updated_class_fraction",
    "model_update_steps",
    "parameter_count",
    "runtime_seconds",
)

# Two-sided 95% Student-t critical values for df 1..30; normal approximation
# is sufficient above 30 for this reporting utility.
T_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def _parse_input(value: str) -> tuple[str, Path, str | None]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Inputs must use CONDITION=SUMMARY_JSON")
    condition, path_and_name = value.split("=", 1)
    path, separator, run_name = path_and_name.rpartition("#")
    if not separator:
        path = path_and_name
        run_name = ""
    if not condition.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Inputs must use CONDITION=SUMMARY_JSON")
    return condition.strip(), Path(path).expanduser().resolve(), run_name.strip() or None


def _estimate(values: list[float]) -> dict[str, float | int | None]:
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "ci95_low": None, "ci95_high": None}
    mean = statistics.fmean(values)
    if n == 1:
        return {"n": 1, "mean": mean, "std": None, "ci95_low": None, "ci95_high": None}
    std = statistics.stdev(values)
    critical = T_975.get(n - 1, 1.96)
    half_width = critical * std / math.sqrt(n)
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def aggregate(inputs: list[tuple[str, Path, str | None]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    sources: dict[str, list[str]] = defaultdict(list)
    for condition, path, run_name in inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected a row list in {path}")
        sources[condition].append(
            str(path) if run_name is None else f"{path}#{run_name}"
        )
        for row in payload:
            if row.get("status") != "completed":
                continue
            if run_name is not None and row.get("name") != run_name:
                continue
            seed = int(row["seed"])
            if seed in grouped[condition]:
                raise ValueError(f"Duplicate seed {seed} for condition {condition}")
            grouped[condition][seed] = row

    output: list[dict[str, Any]] = []
    for condition in sorted(grouped):
        seed_rows = grouped[condition]
        rows = [seed_rows[seed] for seed in sorted(seed_rows)]
        widths = sorted({tuple(row.get("final_widths", [])) for row in rows})
        record: dict[str, Any] = {
            "condition": condition,
            "seeds": sorted(seed_rows),
            "seed_count": len(rows),
            "final_widths_observed": [list(value) for value in widths],
            "sources": sources[condition],
            "metrics": {},
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in rows if row.get(metric) is not None]
            record["metrics"][metric] = _estimate(values)
        output.append(record)
    return output


def _write(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(rows, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    columns = ["condition", "seed_count", "seeds", "final_widths_observed"]
    for metric in METRICS:
        columns.extend(
            [f"{metric}_mean", f"{metric}_std", f"{metric}_ci95_low", f"{metric}_ci95_high"]
        )
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            flat: dict[str, Any] = {
                "condition": row["condition"],
                "seed_count": row["seed_count"],
                "seeds": " ".join(map(str, row["seeds"])),
                "final_widths_observed": json.dumps(row["final_widths_observed"]),
            }
            for metric in METRICS:
                estimate = row["metrics"][metric]
                for key in ("mean", "std", "ci95_low", "ci95_high"):
                    flat[f"{metric}_{key}"] = estimate[key]
            writer.writerow(flat)

    lines = [
        "# Confirmation aggregate",
        "",
        "Intervals are two-sided 95% Student-t intervals across independent seeds.",
        "",
        "| Condition | n | Macro MSE (95% CI) | Foreground MSE (95% CI) | Quota stops | Updated classes | Widths |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        metrics = row["metrics"]

        def ci(metric: str) -> str:
            value = metrics[metric]
            if value["mean"] is None:
                return ""
            if value["ci95_low"] is None:
                return f'{value["mean"]:.5f}'
            return f'{value["mean"]:.5f} [{value["ci95_low"]:.5f}, {value["ci95_high"]:.5f}]'

        widths = "; ".join("/".join(map(str, value)) for value in row["final_widths_observed"])
        lines.append(
            f'| {row["condition"]} | {row["seed_count"]} | {ci("macro_mse")} | '
            f'{ci("foreground_mse")} | {metrics["quota_stop_fraction"]["mean"]:.3f} | '
            f'{metrics["updated_class_fraction"]["mean"]:.3f} | {widths} |'
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=_parse_input, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    _write(args.output_dir, aggregate(args.input))


if __name__ == "__main__":
    main()
