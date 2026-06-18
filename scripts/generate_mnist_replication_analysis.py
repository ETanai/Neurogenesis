from __future__ import annotations

import csv
import json
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MLFLOW_DB = REPO_ROOT / "mlflow.db"
MLRUNS = REPO_ROOT / "mlruns" / "3"
OUT = REPO_ROOT / "outputs" / "diagnostics" / "paper_replication_analysis"


@dataclass(frozen=True)
class RunRef:
    label: str
    run_id: str
    mse: str
    sizes: str
    track: str
    notes: str


RUNS = [
    RunRef(
        "Best practical clean NDL+dataset",
        "961b0f5ded9e4314958890bc89d7cb8a",
        "0.00931",
        "[254,154,102,90]",
        "best-effort / clean upper bound",
        "Near fixed-AE ceiling, strict funnel, no L3 cap hits.",
    ),
    RunRef(
        "Best practical NDL+IR",
        "6a776baaf3c547b0bd3ad036e523ba56",
        "0.02020",
        "[318,258,322,180]",
        "best-effort / main IR",
        "Best current IR reference; useful but far below clean replay.",
    ),
    RunRef(
        "Strict paper-level AE comparison",
        "9d1c112defd3450c87306e2c312ff38d",
        "0.01681",
        "[254,280,290,140]",
        "strict mechanism diagnostic",
        "Literal SHL-AE objective under practical schedule; poor growth shape.",
    ),
    RunRef(
        "Paper-size SHL-only ceiling",
        "a11a8d42df054211b78dc08cb93d338b",
        "0.04241",
        "[225,135,83,40]",
        "strict training diagnostic",
        "Architecture with local SHL-only training performs poorly.",
    ),
    RunRef(
        "Paper-size SHL + finetune ceiling",
        "403ce320a04b4a1ab98a093b15f77856",
        "0.00561",
        "[225,135,83,40]",
        "best-effort / architecture ceiling",
        "Shows paper-sized architecture has ample capacity with global coupling.",
    ),
    RunRef(
        "Literal paper-local NDL+IR attempt",
        "159590999b384aa1b71fd0fa18b1ccc6",
        "0.07284 after digit 2",
        "[375,300,275,60] after digit 2",
        "strict paper-text",
        "Stopped during digit 3 after severe cap-driven growth.",
    ),
    RunRef(
        "Strict paper-level AE single-0 gate",
        "67e66083a70e4e32a7ec81bb05138ad9",
        "0.01239",
        "[210,100,79,40]",
        "strict paper-text / single-class gate",
        "Gentler growth plus higher L0/L1 quota prevents early runaway; 0/1/7 reconstructions are recognizable.",
    ),
    RunRef(
        "Strict paper-level AE single-0 + after-class coupling",
        "b09be982a91548a38c4c1cb9fa411a6d",
        "0.00716",
        "[210,100,79,40]",
        "paper-inspired coupling diagnostic",
        "Five full-AE coupling epochs after class completion improve quality while preserving compact growth.",
    ),
    RunRef(
        "Strict paper-level AE single-0 + after-level coupling",
        "ea7ea83581a443b79c2b37fe87ddef15",
        "0.00700",
        "[210,105,175,40]",
        "paper-inspired coupling diagnostic",
        "Quality improves, but coupling between levels changes later growth pressure and expands L2.",
    ),
    RunRef(
        "Strict paper-level AE two-class gate + after-class coupling",
        "0ffa8a6915c241b1ae5bbe09a4ae142f",
        "0.00797",
        "[210,133,179,60]",
        "paper-inspired coupling diagnostic",
        "Quality remains strong for incremental [0,2], but L2/L3 growth shape breaks.",
    ),
    RunRef(
        "Strict paper-level AE two-class gate + L2/L3 quota",
        "32ebd428a91f4229a14b19c98c8cb52c",
        "0.00806",
        "[210,133,79,60]",
        "paper-inspired quota diagnostic",
        "Level-specific quotas remove the class-2 L2 explosion with minimal quality loss; raising L3 quota further had no effect, so L3 cap/growth remains the issue.",
    ),
    RunRef(
        "Strict paper-level AE two-class gate + L2 quota + L3 hard cap",
        "587f083a080242149c3db455637b11c3",
        "0.01043",
        "[210,141,79,40]",
        "paper-inspired shape diagnostic",
        "Hard L3 cap restores a compact top layer but regresses quality and leaves high L3 outlier fractions.",
    ),
    RunRef(
        "Strict paper-level AE two-class gate + L2 quota + mild L3 growth",
        "4bed42cdad5d4631b7264caf1dfc783f",
        "0.00806",
        "[210,133,79,60]",
        "paper-inspired shape diagnostic",
        "Milder L3-specific growth matches the quota baseline final MSE/size but costs more L3 rounds.",
    ),
    RunRef(
        "Global-partial objective under matched two-class quota/coupling policy",
        "abc534fc10824b0db3b8b97ee4fabfc4",
        "0.00981",
        "[210,111,79,57]",
        "objective diagnostic",
        "Improves L3 outlier dynamics relative to paper_level_ae but worsens aggregate MSE.",
    ),
    RunRef(
        "Controlled two-class NDL+IR with Gaussian shrink",
        "fbf73a924a94485b9da14e7a5cf07d61",
        "0.01648",
        "[216,104,79,60]",
        "IR diagnostic",
        "Shrink-IR keeps growth mostly controlled but remains much worse than clean replay and degrades old classes.",
    ),
    RunRef(
        "Controlled two-class NDL+IR with recon-error filter",
        "696b583c18d242fb89a7e5935dc0201a",
        "0.01956",
        "[215,100,79,48]",
        "IR diagnostic",
        "Strict reconstruction-error matching made L3 smaller but worsened quality; filter acceptance logged 0.0 for every class and fell back after resampling.",
    ),
    RunRef(
        "Controlled two-class NDL+IR with latent-percentile filter",
        "7975c4f0a29b45898a740d50df06b6b7",
        "0.02193",
        "[216,100,83,51]",
        "IR diagnostic",
        "Latent p95 filtering accepted samples but worsened quality versus no-filter shrink IR, suggesting filtering is not the primary fix.",
    ),
    RunRef(
        "Controlled two-class NDL+IR with replay ratio 0.5",
        "bc581beb28b844fbb01ac2ca347230b3",
        "0.01891",
        "[215,100,79,48]",
        "IR diagnostic",
        "Lower paper-style replay amount shrinks L3 but worsens MSE versus ratio 1.0; old-class drift remains.",
    ),
    RunRef(
        "Controlled two-class NDL+IR reusing old replay stats",
        "7e0a542ad920461ab2bcfe203f98bbf0",
        "0.01928",
        "[216,100,79,48]",
        "IR diagnostic",
        "Reusing old latent replay stats shrinks L3 but worsens class 0/2/7 quality versus fresh-stat shrink IR.",
    ),
    RunRef(
        "Controlled two-class NDL+IR with covariance shrink 0.50",
        "2fce0521b21e4fa898f969b765815202",
        "0.02030",
        "[216,105,79,48]",
        "IR diagnostic",
        "Stronger covariance shrinkage makes IR more compact but worsens quality versus shrink 0.25.",
    ),
    RunRef(
        "Controlled two-class NDL+IR with noise scale 0.75",
        "87f8fc23da78490a933605d2fb296464",
        "0.02033",
        "[216,105,79,48]",
        "IR diagnostic",
        "Lower sample noise also worsens quality; simple variance reduction is not the fix.",
    ),
    RunRef(
        "Controlled two-class NDL+IR with post-class coupling 10",
        "869ed045263c492d8b48cf3e74ed9d23",
        "0.02027",
        "[216,105,79,45]",
        "IR diagnostic",
        "Longer post-class full-AE coupling shrinks L3 but worsens old-class retention.",
    ),
    RunRef(
        "Controlled two-class NDL+IR with decoder-only coupling",
        "c7d160ca3de645229bc7f836177351d8",
        "0.02066",
        "[216,104,79,48]",
        "IR diagnostic",
        "Freezing the encoder during coupling also worsens quality, so encoder movement is not the sole culprit.",
    ),
    RunRef(
        "Controlled two-class NDL+IR with replay loss weight 2.0",
        "d79194c75086469291bca561be42511a",
        "0.02368",
        "[216,105,79,48]",
        "IR diagnostic",
        "Valid replay-loss upweighting worsens all old/new class MSE versus shrink baseline.",
    ),
    RunRef(
        "Strict paper-level AE two-class gate + L3 shape pressure",
        "c8a6cb45ddca410a94f2a845db9886cc",
        "0.00943",
        "[210,125,79,47]",
        "paper-inspired shape diagnostic",
        "Shape-adaptive pressure gives a middle ground: smaller L3 than quota baseline, better MSE than hard cap.",
    ),
    RunRef(
        "L3 pixel/local outlier criterion diagnostic",
        "e679052a66974c62b4a98135552a07f9",
        "0.02029",
        "[400,111,83,23]",
        "criterion diagnostic",
        "Pixel and local L3 outliers are moderately aligned after training; strict base-derived pixel thresholds remain the likely growth pressure. Not a candidate due to L0 cap growth.",
    ),
    RunRef(
        "L3 quality-growth gate, protected lower cap, factor 1.0",
        "90e419503afd423a829186917d55f118",
        "0.02085",
        "[220,105,79,21]",
        "criterion diagnostic",
        "Two-gate L3 growth enforces compactness but stops too early and hurts class-2 quality.",
    ),
    RunRef(
        "L3 quality-growth gate, protected lower cap, factor 0.9",
        "16cb13d8ab2b44c0a14d8007861be555",
        "0.01919",
        "[220,105,79,23]",
        "criterion diagnostic",
        "Softer two-gate target adds one L3 round for class 2 but remains far worse than quota or shape-pressure candidates.",
    ),
    RunRef(
        "L3 adaptive threshold, percentile 0.55, min iteration 12",
        "bb7c22ca4c1841739a82cadbfab7cb11",
        "0.00806",
        "[210,129,79,60]",
        "criterion diagnostic",
        "Fair strong-baseline run; adaptive threshold activates too late and reproduces the quota baseline.",
    ),
    RunRef(
        "L3 adaptive threshold, percentile 0.55, min iteration 6",
        "dbc8719d6bce45f2b1e0838901470f01",
        "0.00999",
        "[210,185,79,44]",
        "criterion diagnostic",
        "Post-stability quantile threshold compacts L3, but quality regresses and L1 grows.",
    ),
    RunRef(
        "L3 adaptive threshold + L1/L3 shape pressure",
        "3ce8a8d4f052499aa2849c197224a191",
        "0.00975",
        "[210,109,79,44]",
        "criterion diagnostic",
        "Combined threshold and shape guard fixes adaptive-only L1 growth, but still trails shape-only quality.",
    ),
]


FOUR_REGIME = [
    {
        "condition": "CL+dataset",
        "run_id": "954bb87b34be4970a50b3a5c59ee18f9",
        "mse": "0.03214",
        "sizes": "fixed control",
        "notes": "Clean replay conventional-learning control.",
    },
    {
        "condition": "NDL no replay",
        "run_id": "5bba0cc8275845fa9426d12ae58836dc",
        "mse": "0.03889",
        "sizes": "not frozen here",
        "notes": "No-replay reference from prior diagnostics.",
    },
    {
        "condition": "CL+IR matched",
        "run_id": "6278f4bc126c4cf681b1ac6ce976f889",
        "mse": "0.04908",
        "sizes": "matched [240,120,108,29]",
        "notes": "Matched early NDL+IR capacity; worse than NDL+IR.",
    },
    {
        "condition": "CL+IR latest-size matched",
        "run_id": "df2a60a5986d485eaa8f59f2a0b5ce73",
        "mse": "0.01159",
        "sizes": "matched [318,258,322,180]",
        "notes": "Matched current best NDL+IR size and practical pretraining/replay settings; beats current best NDL+IR.",
    },
    {
        "condition": "NDL+IR early best",
        "run_id": "7bec2fae68d04a8db6bc463212af43ef",
        "mse": "0.03315",
        "sizes": "[240,120,108,29]",
        "notes": "Early paper-column IR reference; beats matched CL+IR.",
    },
    {
        "condition": "NDL+IR practical best",
        "run_id": "6a776baaf3c547b0bd3ad036e523ba56",
        "mse": "0.02020",
        "sizes": "[318,258,322,180]",
        "notes": "Best IR after protocol tuning; not paper-shaped.",
    },
    {
        "condition": "NDL+dataset practical best",
        "run_id": "961b0f5ded9e4314958890bc89d7cb8a",
        "mse": "0.00931",
        "sizes": "[254,154,102,90]",
        "notes": "Clean replay upper bound; near architecture ceiling.",
    },
]


CLAIMS = [
    {
        "claim": "Figure 3: base 1/7 AE reconstructs 1/7 and biases unseen digits.",
        "paper_evidence": "Figure 3 and text around lines 166-173.",
        "status": "partial",
        "run": "Base no-finetune 0.02650; base finetune10 0.01467.",
        "result": "Base behavior exists, but strict no-finetune stack is weak.",
        "discrepancy": "Paper does not specify whether stacked denoising included global coupling.",
    },
    {
        "claim": "Figure 4A-C: CL, NDL, and CL+IR controls are comparable.",
        "paper_evidence": "Figure 4 caption and lines 178-184.",
        "status": "partial",
        "run": "CL+dataset 954bb..., NDL no replay 5bba..., CL+IR df2a...",
        "result": "Latest-size matched CL+IR now exists and reaches 0.01159.",
        "discrepancy": "The latest matched CL+IR beats current best NDL+IR, unlike the paper's Figure 4 direction.",
    },
    {
        "claim": "Figure 4D/E: NDL+IR slightly outperforms CL+IR and retains old/new digits.",
        "paper_evidence": "Lines 178-184 and Figure 4D/E.",
        "status": "fail under latest practical match",
        "run": "NDL+IR 6a776... vs matched CL+IR df2a...",
        "result": "Latest-size CL+IR 0.01159 beats best practical NDL+IR 0.02020.",
        "discrepancy": "Current NDL+IR growth/training dynamics do not outperform a sufficiently large fixed CL+IR control.",
    },
    {
        "claim": "Figure 4F: growth is funnel-like and near paper shape.",
        "paper_evidence": "Figure 4F and line 180.",
        "status": "partial/fail",
        "run": "Best clean 961b..., best IR 6a776...",
        "result": "Best clean is funnel [254,154,102,90]; best IR [318,258,322,180] is not.",
        "discrepancy": "Growth/threshold/stability details are under-specified; L2/L3 overgrow.",
    },
    {
        "claim": "Figure 5: NDL+IR retains old digits better across incremental classes.",
        "paper_evidence": "Figure 5 and line 206.",
        "status": "partial/fail pending manual scoring",
        "run": "Best IR 6a776... vs latest matched CL+IR df2a...",
        "result": "Side-by-side visual artifact exists; quantitative latest CL+IR is better.",
        "discrepancy": "Manual old-class retention scoring still needs to be added.",
    },
    {
        "claim": "IR uses top-latent class mean and Cholesky/full covariance Gaussian sampling.",
        "paper_evidence": "Figure 2 and lines 99-107.",
        "status": "pass",
        "run": "Config/artifacts in 6a776...",
        "result": "Repo logs IR quality grids and uses gaussian_full as paper mode.",
        "discrepancy": "Sampling details are matched; quality depends on AE coupling.",
    },
    {
        "claim": "Stability trains current class plus replay from previous classes.",
        "paper_evidence": "Lines 95-99 and 175.",
        "status": "pass",
        "run": "Practical and paper-replay configs.",
        "result": "Implemented as paper replay mode and logged replay composition.",
        "discrepancy": "Replay quantity is not specified by paper and affects results.",
    },
]


DIFFERENCES = [
    ("Class order", "Base [1,7], incremental [0,2,3,4,5,6,8,9].", "Implemented.", "matching"),
    ("Initial architecture", "[200,100,75,20].", "Implemented; practical runs may grow differently.", "matching/variable"),
    ("IR sampling", "Top hidden mean + Cholesky/covariance Gaussian.", "Implemented as gaussian_full.", "matching"),
    ("Plasticity freezing", "Old encoder frozen; decoder LR/100.", "Implemented and tested after LR bug fix.", "matching"),
    ("Level objective", "Train current level as SHL-AE on representation space.", "Implemented as paper_level_ae, but historical best uses global/partial objective.", "fragile"),
    ("Next-layer optimization", "Train connections into next level after growth.", "Implemented as paper_columns option.", "matching/fragile"),
    ("Pretraining finetune", "Not explicit in MNIST method.", "Global finetune is crucial for good reconstruction.", "non-paper but important"),
    ("Thresholds", "User-specified thresholds from old data stats, no exact percentile.", "Estimated percentiles and manual thresholds tested.", "under-specified"),
    ("MaxOutliers", "Allowed outliers mentioned, no value.", "Quota/gates strongly affect growth shape.", "under-specified"),
    ("Growth amount", "New nodes added up to max; exact count rule absent.", "Proportional growth can overgrow; alternatives tested.", "under-specified"),
    ("Training length/early stop", "Briefly trained; no epochs/stop rule.", "Strict L3 stability improves quality and shape.", "under-specified"),
    ("Replay amount", "Samples from previous classes; amount not optimized.", "Paper/ratio/balanced modes differ materially.", "under-specified"),
]


def metric(conn: sqlite3.Connection, run_id: str, key: str) -> float | None:
    row = conn.execute(
        "select value from metrics where run_uuid=? and key=? order by step desc, timestamp desc limit 1",
        (run_id, key),
    ).fetchone()
    return None if row is None else float(row[0])


def run_name(conn: sqlite3.Connection, run_id: str) -> str:
    row = conn.execute("select name from runs where run_uuid=?", (run_id,)).fetchone()
    return "" if row is None else str(row[0])


def copy_if_exists(src: Path, dst: Path) -> str:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return str(dst.relative_to(REPO_ROOT))
    return f"missing: {src.relative_to(REPO_ROOT)}"


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    images = OUT / "artifacts"

    artifact_map = {
        "fig3_base_reconstruction_grid.png": MLRUNS / "2cdd75260f3e4ea5b001b6df7b4de925" / "artifacts" / "figures" / "reconstructions_step_2.png",
        "fig5_incremental_reconstruction_grid.png": MLRUNS / "6a776baaf3c547b0bd3ad036e523ba56" / "artifacts" / "figures" / "reconstruction_timeline.png",
        "ir_samples_grid.png": MLRUNS / "6a776baaf3c547b0bd3ad036e523ba56" / "artifacts" / "figures" / "ir_timeline.png",
        "best_clean_reconstruction_timeline.png": MLRUNS / "961b0f5ded9e4314958890bc89d7cb8a" / "artifacts" / "figures" / "reconstruction_timeline.png",
        "best_ir_recon_ir_combined_timeline.png": MLRUNS / "6a776baaf3c547b0bd3ad036e523ba56" / "artifacts" / "figures" / "recon_ir_combined_timeline.png",
        "latest_matched_clir_reconstruction_timeline.png": MLRUNS / "df2a60a5986d485eaa8f59f2a0b5ce73" / "artifacts" / "figures" / "reconstruction_timeline.png",
        "strictpaper_single0_reconstructions.png": MLRUNS / "67e66083a70e4e32a7ec81bb05138ad9" / "artifacts" / "figures" / "reconstructions_step_3.png",
        "strictpaper_single0_after_class_coupling_reconstructions.png": MLRUNS / "b09be982a91548a38c4c1cb9fa411a6d" / "artifacts" / "figures" / "reconstructions_step_3.png",
        "strictpaper_single0_after_level_coupling_reconstructions.png": MLRUNS / "ea7ea83581a443b79c2b37fe87ddef15" / "artifacts" / "figures" / "reconstructions_step_3.png",
        "strictpaper_multiclass_0_2_after_class_coupling_reconstructions.png": MLRUNS / "0ffa8a6915c241b1ae5bbe09a4ae142f" / "artifacts" / "figures" / "reconstructions_step_4.png",
        "strictpaper_multiclass_0_2_l2l3quota_reconstructions.png": MLRUNS / "32ebd428a91f4229a14b19c98c8cb52c" / "artifacts" / "figures" / "reconstructions_step_4.png",
        "strictpaper_multiclass_0_2_l3cap_reconstructions.png": MLRUNS / "587f083a080242149c3db455637b11c3" / "artifacts" / "figures" / "reconstructions_step_4.png",
        "strictpaper_multiclass_0_2_l3mild_reconstructions.png": MLRUNS / "4bed42cdad5d4631b7264caf1dfc783f" / "artifacts" / "figures" / "reconstructions_step_4.png",
        "objective_compare_global_partial_0_2_reconstructions.png": MLRUNS / "abc534fc10824b0db3b8b97ee4fabfc4" / "artifacts" / "figures" / "reconstructions_step_4.png",
        "ir_retest_0_2_shrink025_recon_ir_timeline.png": MLRUNS / "fbf73a924a94485b9da14e7a5cf07d61" / "artifacts" / "figures" / "recon_ir_combined_timeline.png",
        "strictpaper_multiclass_0_2_shape_l3_reconstructions.png": MLRUNS / "c8a6cb45ddca410a94f2a845db9886cc" / "artifacts" / "figures" / "reconstructions_step_4.png",
    }
    copied = {name: copy_if_exists(src, images / name) for name, src in artifact_map.items()}

    with sqlite3.connect(MLFLOW_DB) as conn:
        reference_rows = []
        for ref in RUNS:
            actual_mse = metric(conn, ref.run_id, "metrics/val_mean_level_3")
            reference_rows.append(
                {
                    "category": ref.label,
                    "run_id": ref.run_id,
                    "run_name": run_name(conn, ref.run_id),
                    "reported_l3_mse": ref.mse,
                    "mlflow_l3_mse": "" if actual_mse is None else f"{actual_mse:.5f}",
                    "sizes": ref.sizes,
                    "track": ref.track,
                    "notes": ref.notes,
                }
            )

    write_csv(
        OUT / "reference_runs.csv",
        reference_rows,
        ["category", "run_id", "run_name", "reported_l3_mse", "mlflow_l3_mse", "sizes", "track", "notes"],
    )
    write_csv(
        OUT / "fig4_metric_comparison.csv",
        FOUR_REGIME,
        ["condition", "run_id", "mse", "sizes", "notes"],
    )

    growth_rows = [
        {
            "condition": "Paper Figure 4 qualitative target",
            "run_id": "paper",
            "sizes": "approx [225,135,83,40]",
            "mse": "not reported numerically",
            "funnel": "yes",
            "notes": "Estimated from paper figure, not exact table data.",
        },
        {
            "condition": "Best practical clean NDL+dataset",
            "run_id": "961b0f5ded9e4314958890bc89d7cb8a",
            "sizes": "[254,154,102,90]",
            "mse": "0.00931",
            "funnel": "yes",
            "notes": "Near ceiling, still larger top layer than paper.",
        },
        {
            "condition": "Best practical NDL+IR",
            "run_id": "6a776baaf3c547b0bd3ad036e523ba56",
            "sizes": "[318,258,322,180]",
            "mse": "0.02020",
            "funnel": "no",
            "notes": "IR replay still triggers large L2/L3 growth.",
        },
        {
            "condition": "Strict paper-level AE",
            "run_id": "9d1c112defd3450c87306e2c312ff38d",
            "sizes": "[254,280,290,140]",
            "mse": "0.01681",
            "funnel": "no",
            "notes": "Literal level objective did not reproduce paper growth.",
        },
        {
            "condition": "Literal paper-local stopped run",
            "run_id": "159590999b384aa1b71fd0fa18b1ccc6",
            "sizes": "[375,300,275,60] after digit 2",
            "mse": "0.07284 after digit 2",
            "funnel": "no",
            "notes": "Stopped after cap-driven failure.",
        },
        {
            "condition": "Strict paper-level AE single-0 gate",
            "run_id": "67e66083a70e4e32a7ec81bb05138ad9",
            "sizes": "[210,100,79,40]",
            "mse": "0.01239 single-0",
            "funnel": "yes",
            "notes": "Gentler growth and L0/L1 quota fixed early strict-paper runaway on the single-0 gate.",
        },
        {
            "condition": "Strict paper-level AE single-0 + after-class coupling",
            "run_id": "b09be982a91548a38c4c1cb9fa411a6d",
            "sizes": "[210,100,79,40]",
            "mse": "0.00716 single-0",
            "funnel": "yes",
            "notes": "Post-class full-AE coupling improves quality without changing growth.",
        },
        {
            "condition": "Strict paper-level AE single-0 + after-level coupling",
            "run_id": "ea7ea83581a443b79c2b37fe87ddef15",
            "sizes": "[210,105,175,40]",
            "mse": "0.00700 single-0",
            "funnel": "no",
            "notes": "Slightly lower MSE, but expands L2 heavily.",
        },
        {
            "condition": "Strict paper-level AE two-class gate + after-class coupling",
            "run_id": "0ffa8a6915c241b1ae5bbe09a4ae142f",
            "sizes": "[210,133,179,60]",
            "mse": "0.00797 two-class",
            "funnel": "no",
            "notes": "Strong reconstructions for 0/1/2/7, but class 2 reintroduces L2/L3 growth pressure.",
        },
        {
            "condition": "Strict paper-level AE two-class gate + L2/L3 quota",
            "run_id": "32ebd428a91f4229a14b19c98c8cb52c",
            "sizes": "[210,133,79,60]",
            "mse": "0.00806 two-class",
            "funnel": "no",
            "notes": "L2 quota removes the class-2 L2 blow-up with minimal quality loss; L3 still reaches 60, and L3 quota 0.60 gave the same result.",
        },
    ]
    write_csv(OUT / "fig4_growth_comparison.csv", growth_rows, ["condition", "run_id", "sizes", "mse", "funnel", "notes"])

    claim_rows = [
        [c["claim"], c["paper_evidence"], c["status"], c["run"], c["result"], c["discrepancy"]]
        for c in CLAIMS
    ]
    claim_md = "# MNIST Paper Claim Matrix\n\n" + md_table(
        ["Claim", "Paper Evidence", "Status", "Best Matching Run", "Result", "Likely Discrepancy"],
        claim_rows,
    ) + "\n"
    (OUT / "paper_claim_matrix.md").write_text(claim_md, encoding="utf-8")

    diff_md = "# Implementation Difference Table\n\n" + md_table(
        ["Topic", "Paper Text", "Repo Behavior", "Bucket"],
        [[*row] for row in DIFFERENCES],
    ) + "\n"
    (OUT / "implementation_difference_table.md").write_text(diff_md, encoding="utf-8")

    metric_md = "# Figure 4 Metric Comparison\n\n" + md_table(
        ["Condition", "Run ID", "L3 MSE", "Sizes", "Notes"],
        [[r["condition"], r["run_id"], r["mse"], r["sizes"], r["notes"]] for r in FOUR_REGIME],
    ) + "\n"
    (OUT / "fig4_metric_comparison.md").write_text(metric_md, encoding="utf-8")

    growth_md = "# Figure 4 Growth Comparison\n\n" + md_table(
        ["Condition", "Run ID", "Sizes", "MSE", "Funnel", "Notes"],
        [[r["condition"], r["run_id"], r["sizes"], r["mse"], r["funnel"], r["notes"]] for r in growth_rows],
    ) + "\n"
    (OUT / "fig4_growth_comparison.md").write_text(growth_md, encoding="utf-8")

    report = f"""# MNIST Paper Replication Report

Date: 2026-06-16

## 1. Executive Summary

This is a MNIST-first replication dossier for the Neurogenesis Deep Learning paper, focused on Figures 3-5. It uses two standards:

- **Strict paper-text replication:** only mechanisms explicitly stated or strongly implied by the paper.
- **Best-effort reproduction:** permits under-specified training details such as global finetuning, stricter stopping rules, and quota tuning when needed to reproduce the qualitative behavior.

The current result is **partial replication with a major negative control update**. The best practical clean dataset-replay run nearly reaches the fixed-AE ceiling and has a funnel shape (`0.00931`, `[254,154,102,90]`). The best practical IR run improves over earlier IR and no-replay controls (`0.02020`) but does not match clean replay, the paper's compact growth shape, or the latest-size matched `CL+IR` control (`0.01159`). A direct transfer of the best clean protocol to full-Gaussian IR also failed early, reaching L3 MSE `0.04066` after digit `0` and cap-like L1/L2 growth during digit `2`. Strict paper-local SHL-AE/no-finetune behavior fails in full MNIST attempts, but a controlled single-0 gate now works with gentler growth and higher L0/L1 quota (`0.01239`, `[210,100,79,40]`). Adding five full-AE coupling epochs after class completion improves that strict single-0 gate to `0.00716` without changing its compact growth. The same candidate stays visually strong on a two-class `[0,2]` gate. Level-specific L2/L3 quotas then remove the class-2 L2 blow-up with minimal quality loss (`0.00806`, `[210,133,79,60]`). A hard L3 cap restores a compact top layer (`[210,141,79,40]`) but regresses L3 MSE to `0.01043`. The latest L3 pixel/local criterion audit shows that local SHL-AE and pixel-space outliers are moderately aligned after training, so the remaining shape issue is now more likely the strict base-derived pixel threshold/quota policy than a completely mismatched local objective. A follow-up two-gate L3 quality rule successfully prevents L3 blow-up, but the tested base-threshold targets (`0.02085` at `[220,105,79,21]` and `0.01919` at `[220,105,79,23]`) sacrifice too much quality. Adaptive post-stability L3 thresholding can stop L3 near the paper target (`0.00999`, `[210,185,79,44]`), but it shifts growth into L1; adding L1/L3 shape pressure fixes that leakage (`0.00975`, `[210,109,79,44]`) but still trails the simpler shape-only run (`0.00943`, `[210,125,79,47]`). On IR, generated-sample filtering, simple replay reduction, old-stat reuse, stronger variance contraction, longer same-form coupling, decoder-only coupling, and blunt replay-loss upweighting do not fix the replay gap: reconstruction-error filtering worsens to `0.01956`, latent p95 filtering worsens to `0.02193`, replay ratio `0.5` gives `0.01891`, `reuse_previous_stats=true` gives `0.01928`, covariance shrink `0.50` gives `0.02030`, noise scale `0.75` gives `0.02033`, coupling epochs `10` gives `0.02027`, decoder-only coupling gives `0.02066`, valid replay loss weight `2.0` gives `0.02368`, the fully main-wired replay-loss rerun `35fcce20e2c14d22912dd9e4fc16f79a` was stopped during class `2` after already degrading after class `0` to L3 MSE `0.03076` with class `7` at `0.04530`, split current-then-replay stability `8a639d5f069a4742a3cf80ebad4f76a6` was stopped before finishing class `0` with old-class L3 MSE `0.02180` and level-0 round 4 still at `5923/5923` outliers, true batch-level interleave `d1375f6bd1b74680bc9919d435271151` failed after digit `0` with L3 MSE `0.02098`, class `7` `0.03117`, sizes `[240,121,90,40]`, and L3 outlier fraction `0.86223`, and class-7-weighted paper replay `693055b142674e8d8baa0c07f1ba82e0` also failed after digit `0` with L3 MSE `0.02110`, class `7` `0.03240`, sizes `[240,107,90,40]`, and L3 outlier fraction `0.93213`.

Leading hypothesis:

> The paper likely relied on an omitted or implicit global coupling/finalization detail, or on threshold/growth/stopping values not reported in the text. Strict local SHL-AE training alone does not reproduce the reported quality in this implementation.

## 2. Paper Claims And Pass/Fail Matrix

See [paper_claim_matrix.md](paper_claim_matrix.md).

{md_table(["Claim", "Status", "Best Matching Evidence"], [[c["claim"], c["status"], c["run"]] for c in CLAIMS])}

## 3. Implementation Fidelity Audit

See [implementation_difference_table.md](implementation_difference_table.md).

{md_table(["Bucket", "Topics"], [
    ["Paper-specified and matching", "class order, initial architecture, IR Gaussian sampling, old encoder freeze, decoder LR/100"],
    ["Paper-specified but fragile", "SHL-AE objective, next-layer training, global RE thresholds"],
    ["Under-specified and important", "LR, epochs, early stopping, thresholds, MaxOutliers, growth amount, replay amount, denoising corruption"],
    ["Non-paper but necessary for quality", "global finetune/coupling, strict L3 stability, quota tuning, practical growth shaping"],
])}

## 4. Quantitative Comparison

See [fig4_metric_comparison.csv](fig4_metric_comparison.csv) and [fig4_metric_comparison.md](fig4_metric_comparison.md).

{md_table(["Condition", "Run ID", "L3 MSE", "Sizes"], [[r["condition"], r["run_id"], r["mse"], r["sizes"]] for r in FOUR_REGIME])}

Main quantitative interpretation:

- Early NDL+IR beat the early matched CL+IR reference (`0.03315` vs `0.04908`), matching the paper's direction.
- The latest-size matched CL+IR control reaches `0.01159`, beating best practical NDL+IR (`0.02020`).
- Clean dataset replay reaches `0.00931`, showing the AE/NDL stack can get near-ceiling quality when replay is ideal and training is tuned.

## 5. Qualitative Reconstruction Comparison

See [visual_scoring_rubric.md](visual_scoring_rubric.md) for manual pass/partial/fail scoring.

Artifacts collected in `artifacts/`:

- Figure 3 base reconstruction grid: `{copied["fig3_base_reconstruction_grid.png"]}`
- Figure 5-style IR reconstruction timeline: `{copied["fig5_incremental_reconstruction_grid.png"]}`
- Best clean reconstruction timeline: `{copied["best_clean_reconstruction_timeline.png"]}`
- Best IR reconstruction + IR combined timeline: `{copied["best_ir_recon_ir_combined_timeline.png"]}`
- Latest-size matched CL+IR reconstruction timeline: `{copied["latest_matched_clir_reconstruction_timeline.png"]}`
- Side-by-side latest CL+IR vs best NDL+IR: `outputs/diagnostics/paper_replication_analysis/artifacts/fig5_matched_clir_vs_ndlir.png`
- Strict paper-level AE single-0 gate reconstructions: `{copied["strictpaper_single0_reconstructions.png"]}`
- Strict paper-level AE single-0 after-class coupling reconstructions: `{copied["strictpaper_single0_after_class_coupling_reconstructions.png"]}`
- Strict paper-level AE `[0,2]` after-class coupling reconstructions: `{copied["strictpaper_multiclass_0_2_after_class_coupling_reconstructions.png"]}`
- Strict paper-level AE `[0,2]` L2/L3 quota reconstructions: `{copied["strictpaper_multiclass_0_2_l2l3quota_reconstructions.png"]}`
- Strict paper-level AE `[0,2]` L3 hard-cap reconstructions: `{copied["strictpaper_multiclass_0_2_l3cap_reconstructions.png"]}`
- Strict paper-level AE `[0,2]` mild L3-growth reconstructions: `{copied["strictpaper_multiclass_0_2_l3mild_reconstructions.png"]}`
- Matched `[0,2]` `global_partial` objective reconstructions: `{copied["objective_compare_global_partial_0_2_reconstructions.png"]}`
- Controlled `[0,2]` shrink-IR reconstruction/IR timeline: `{copied["ir_retest_0_2_shrink025_recon_ir_timeline.png"]}`
- Strict paper-level AE `[0,2]` L3 shape-pressure reconstructions: `{copied["strictpaper_multiclass_0_2_shape_l3_reconstructions.png"]}`

Qualitative status:

- Base reconstructions exist and show the intended biasing behavior, but strict no-finetune base quality is weaker than practical finetuned quality.
- Best practical clean replay reconstructions are close to the architecture ceiling.
- Best practical IR reconstructions are useful but still visibly below clean replay.
- A latest-size matched CL+IR visual timeline now exists; the manual visual rubric rates current NDL+IR as partial/fail relative to the paper claim because matched CL+IR is quantitatively stronger and NDL+IR growth remains non-paper-like.

## 6. Growth-Shape Comparison

See [fig4_growth_comparison.csv](fig4_growth_comparison.csv) and [fig4_growth_comparison.md](fig4_growth_comparison.md).

{md_table(["Condition", "Sizes", "MSE", "Funnel"], [[r["condition"], r["sizes"], r["mse"], r["funnel"]] for r in growth_rows])}

Growth interpretation:

- The best clean run is funnel-shaped but has a larger top layer than the paper target.
- The best IR run is not funnel-shaped; L2/L3 growth remains excessive.
- Strict paper-level AE training does not fix shape; it worsens middle-layer growth.
- The literal strict paper-local run diverges sharply from the paper after only a few classes.
- A gentler strict paper-level AE single-0 gate prevents early L0/L1 runaway and keeps a funnel shape, so strict-paper full MNIST should be retried only after small quota/coupling checks.
- Post-class global coupling improves the strict single-0 MSE from `0.01239` to `0.00716` without changing growth, while after-level coupling reaches `0.00700` but grows L2 to `175`.
- The two-class `[0,2]` gate keeps strong visual/MSE quality but grows to `[210,133,179,60]`, narrowing the next failure to L2/L3 quota and growth pressure.
- Raising L2/L3 accepted-outlier quotas keeps two-class quality (`0.00806`) and removes L2 growth (`179 -> 79`), but L3 still reaches `60`; raising L3 quota further to `.60` gives the same result.
- Hard-capping L3 growth to `+10` restores a compact top layer (`40`) but regresses aggregate L3 MSE to `0.01043` and leaves class-2 L3 outlier fraction near `0.92`.
- A half-speed L3 growth attempt (`factor_new_nodes_by_level.3=0.001`, `factor_max_new_nodes_by_level.3=0.025`) was too conservative: class `0` needed 20 L3 rounds to reach size `40` and still had L3 outlier fraction `0.61185`.
- A milder L3 growth attempt (`factor_new_nodes_by_level.3=0.0015`, `factor_max_new_nodes_by_level.3=0.05`) completed but converged to the same final MSE and sizes as the quota baseline (`0.00806`, `[210,133,79,60]`) with more L3 rounds.
- A shape-adaptive L3 gate (`scale_both`, target ratio `0.4`) gives a useful middle ground: `[210,125,79,47]`, L3 MSE `0.00943`, no L3 cap hit.
- A matched `global_partial` objective run improves L3 outlier behavior (`[210,111,79,57]`, class-0 L3 outlier `0.44791`) but worsens aggregate L3 MSE to `0.00981`, so objective choice matters but does not solve the paper mismatch alone.
- The [L3 outlier criterion audit](l3_outlier_criterion_audit.md) shows why this remains sticky: paper-level training optimizes a local representation-space SHL-AE objective, while growth still uses a strict pixel-space `forward_partial` tail threshold. The pixel/local overlap run shows those two errors are moderately aligned after one training round, so the problem is not simply that the wrong samples are selected. The follow-up two-gate `quality_growth_gate` runs keep L3 tiny (`21-23`) but regress MSE to about `0.019-0.021`, so the base-derived mean-error target is too strict.
- Adaptive post-stability L3 thresholding was tested: `min_iteration=12` reproduced the quota baseline (`0.00806`, `[210,129,79,60]`), while `min_iteration=6` compacted L3 to `44` but regressed MSE to `0.00999` and grew L1 to `185`. Combining adaptive L3 thresholding with L1/L3 shape pressure fixed L1 (`[210,109,79,44]`) but still trailed shape-only MSE, so fixed-quantile thresholding is not the missing paper rule by itself.

## 7. IR Replay Quality Analysis

IR artifacts:

- IR sample timeline: `{copied["ir_samples_grid.png"]}`
- IR quality JSON files are available under `mlruns/3/6a776baaf3c547b0bd3ad036e523ba56/artifacts/diagnostics/`.

IR interpretation:

- The IR mechanism itself is implemented in the paper-like form: class-conditional top-latent Gaussian sampling using full covariance.
- Earlier ablations showed IR samples were broadly comparable to the model's reconstructions, not completely broken.
- The clean funnel protocol transferred to `gaussian_full` IR as run `721a8406b6a9448fa95e26e62e272818`, but was stopped during digit `2`: after digit `0`, L3 MSE was `0.04066`; by digit `2`, L1 reached `200` and L2 reached `275`.
- A controlled `[0,2]` shrink-IR retest (`fbf73a924a94485b9da14e7a5cf07d61`) finishes with L3 MSE `0.01648` and sizes `[216,104,79,60]`, versus clean replay `0.00806` and `[210,133,79,60]`.
- Its per-class L3 MSE shows old-class drift: class `0` `0.02668`, class `1` `0.00944`, class `2` `0.01159`, class `7` `0.01942`.
- A stricter `recon_error_match` p95 filter (`696b583c18d242fb89a7e5935dc0201a`) worsens L3 MSE to `0.01956` while shrinking L3 to `48`; filter acceptance logged `0.0` for all classes, so it mostly resampled then fell back.
- A gentler `latent_percentile` p95 filter (`7975c4f0a29b45898a740d50df06b6b7`) also worsens L3 MSE to `0.02193` with sizes `[216,100,83,51]`; acceptance was nonzero but uneven across classes.
- Lowering paper-style replay amount to `0.5` (`bc581beb28b844fbb01ac2ca347230b3`) improves over filtering but still worsens versus ratio `1.0`: L3 MSE `0.01891`, sizes `[215,100,79,48]`.
- Reusing old replay statistics and only refitting the new class (`7e0a542ad920461ab2bcfe203f98bbf0`) also worsens versus fresh-stat shrink IR: L3 MSE `0.01928`, sizes `[216,100,79,48]`, with class `0` `0.03247`, class `2` `0.01641`, and class `7` `0.02017`.
- Stronger covariance shrinkage (`2fce0521b21e4fa898f969b765815202`) worsens to L3 MSE `0.02030`, sizes `[216,105,79,48]`; lower sample noise (`87f8fc23da78490a933605d2fb296464`) similarly worsens to `0.02033`, sizes `[216,105,79,48]`.
- Doubling post-class full-AE coupling to 10 epochs (`869ed045263c492d8b48cf3e74ed9d23`) shrinks L3 to `45` but worsens L3 MSE to `0.02027`, mainly through worse old-class retention.
- Decoder-only post-class coupling (`c7d160ca3de645229bc7f836177351d8`) worsens further to L3 MSE `0.02066`, sizes `[216,104,79,48]`, so encoder movement during coupling is not the sole culprit.
- Completed replay-loss upweighting (`d79194c75086469291bca561be42511a`, `stability_replay_loss_weight=2.0`) worsens to L3 MSE `0.02368`, sizes `[216,105,79,48]`, with class `0` rising to `0.03978` and class `7` to `0.02748`. After wiring the same knob into the main stability phase too, partial rerun `35fcce20e2c14d22912dd9e4fc16f79a` was stopped during class `2` because it had already degraded after class `0`: L3 MSE `0.03076`, sizes `[274,120,75,29]`, class `7` MSE `0.04530`. A split `current_then_replay` stability schedule (`8a639d5f069a4742a3cf80ebad4f76a6`) was stopped before finishing class `0`: old-class L3 MSE `0.02180`, class `7` MSE `0.03171`, and L0 round 4 still had `5923/5923` outliers. A naive epoch-level `interleave_epochs` run (`100673ff04f9443786c1b5bb7c5a1630`) was also stopped as computationally impractical: after about four minutes it had only reached class `0` level `1`, with `class_0_level_0_avg_loss_iter=0.00598` and `class_0_level_1_avg_loss_iter=0.02379`, and no post-class validation. True batch-level interleave inside one optimizer loop (`d1375f6bd1b74680bc9919d435271151`) was fast enough to evaluate, but failed after digit `0`: L3 MSE `0.02098`, sizes `[240,121,90,40]`, class `7` MSE `0.03117`, and L3 outlier fraction `0.86223`. A class-conditioned paper replay correction that preserved total replay budget but overweighted class `7` by `3x` (`693055b142674e8d8baa0c07f1ba82e0`) also failed after digit `0`: L3 MSE `0.02110`, sizes `[240,107,90,40]`, class `7` MSE `0.03240`, and L3 outlier fraction `0.93213`. Earlier weight runs `f9e6fa8a0aa94d758dd77c007ed80466` and `5f1726f2c424474aa986f8ab2f89b274` are superseded because they were launched before the runner pass-through was verified.
- The IR gap is therefore likely downstream of replay sample distribution or optimization dynamics interacting with NG growth, not a missing sampler primitive or a problem solved by generated-sample filtering, simply less replay, old-stat reuse, lower-variance sampling, more of the same coupling phase, freezing the encoder during coupling, blunt replay-loss upweighting, rearranging current/replay schedules globally, or simple class-weighted replay composition.

## 8. Strict Paper-Text Versus Best-Effort Reproduction

Strict paper-text status:

- Implements the paper's local SHL-AE interpretation, old encoder freeze, decoder LR/100, paper-column next-layer training, and full Gaussian IR.
- Does **not** reproduce the paper's qualitative behavior under full MNIST tested schedules.
- The literal NDL+IR attempt reached `[375,300,275,60]` and `0.07284` after digit `2`, then was stopped during digit `3`.
- A strict single-0 gate with controlled growth now reaches `0.01239` and recognizable `0/1/7`, which shows the local SHL-AE path can work in the smallest incremental case.
- A paper-inspired post-class full-reconstruction coupling phase improves the same gate to `0.00716` while preserving `[210,100,79,40]`, strengthening the omitted-coupling hypothesis.
- The post-class coupling candidate remains high quality on incremental `[0,2]`, but no longer preserves paper-like shape; this argues for solving quotas/growth before full MNIST.
- The L2/L3 quota candidate shows L2 growth was not needed for quality, making top-layer cap/growth the next constrained experiment.
- A matched `global_partial` objective comparison confirms that L3 outlier dynamics are objective-sensitive, but the practical objective is not a simple replacement because its aggregate MSE is worse than `paper_level_ae` under the same policy.
- The L3 criterion audit narrows the remaining clean-replay shape mismatch to the growth trigger: local SHL-AE training can produce good average reconstructions, and local/pixel outlier rankings are moderately aligned, but the base-derived pixel-space L3 tail threshold still demands more nodes.

Best-effort status:

- With global finetune/coupling, stronger LR, strict L3 stability, and quota tuning, clean replay gets very strong: `0.00931`.
- Best practical IR improves to `0.02020`, useful but still below clean replay and not paper-shaped.
- The latest matched CL+IR control reaches `0.01159`, so the current practical branch no longer supports the claim that NDL+IR beats a matched fixed-size CL+IR control under the strongest tested settings.

## 9. Remaining TODOs For Full MNIST Replication

1. Treat item 8 as narrowed: shape-only remains the best compact clean `[0,2]` candidate; fixed-quantile adaptive thresholds are mechanically useful but not better. Further threshold work should be class-conditioned, not another fixed percentile.
2. Continue IR-specific ablations only as a deeper replay-distribution audit. The obvious local fixes are now negative: shrinkage alone leaves L3 MSE `0.01648` on `[0,2]`, p95 reconstruction-error matching worsens it to `0.01956`, p95 latent filtering worsens it to `0.02193`, replay ratio `0.5` gives `0.01891`, true batch interleave gives `0.02098` after digit `0`, and class-7-weighted replay gives `0.02110` after digit `0`.
3. Run multi-seed confirmation for latest matched CL+IR, best NDL+IR, best clean replay, and the strict-paper single-0 candidate.
4. Do not claim SD19 replication from this dossier; SD19 remains out of scope.

## Artifact Index

- [paper_claim_matrix.md](paper_claim_matrix.md)
- [implementation_difference_table.md](implementation_difference_table.md)
- [reference_runs.csv](reference_runs.csv)
- [visual_scoring_rubric.md](visual_scoring_rubric.md)
- [l3_outlier_criterion_audit.md](l3_outlier_criterion_audit.md)
- [fig4_metric_comparison.csv](fig4_metric_comparison.csv)
- [fig4_metric_comparison.md](fig4_metric_comparison.md)
- [fig4_growth_comparison.csv](fig4_growth_comparison.csv)
- [fig4_growth_comparison.md](fig4_growth_comparison.md)
- `artifacts/fig3_base_reconstruction_grid.png`
- `artifacts/fig5_incremental_reconstruction_grid.png`
- `artifacts/ir_samples_grid.png`
- `artifacts/strictpaper_single0_reconstructions.png`
- `artifacts/strictpaper_single0_after_class_coupling_reconstructions.png`
- `artifacts/strictpaper_multiclass_0_2_after_class_coupling_reconstructions.png`
- `artifacts/strictpaper_multiclass_0_2_l2l3quota_reconstructions.png`
- `artifacts/strictpaper_multiclass_0_2_l3cap_reconstructions.png`
- `artifacts/strictpaper_multiclass_0_2_l3mild_reconstructions.png`
- `artifacts/objective_compare_global_partial_0_2_reconstructions.png`
- `artifacts/ir_retest_0_2_shrink025_recon_ir_timeline.png`
- `artifacts/strictpaper_multiclass_0_2_shape_l3_reconstructions.png`
"""
    (OUT / "mnist_replication_report.md").write_text(report, encoding="utf-8")

    print(f"Wrote MNIST replication dossier to {OUT}")


if __name__ == "__main__":
    main()
