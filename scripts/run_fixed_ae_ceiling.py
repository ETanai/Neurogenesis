"""Train a fixed-size non-NG autoencoder on all MNIST classes.

This is a diagnostic ceiling test: it removes neurogenesis and continual
learning, keeps a representative final NDL architecture, and measures how well
the plain MLP autoencoder can reconstruct MNIST when trained normally.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from models.autoencoder import AutoEncoder  # noqa: E402
from utils.intrinsic_replay import IntrinsicReplay  # noqa: E402

try:
    import mlflow
except ImportError:  # pragma: no cover - optional runtime dependency
    mlflow = None


def _parse_sizes(value: str) -> list[int]:
    sizes = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("hidden sizes must be positive integers")
    return sizes


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _mnist_loaders(data_dir: Path, batch_size: int, num_workers: int) -> tuple[DataLoader, DataLoader]:
    transform = transforms.ToTensor()
    train = datasets.MNIST(str(data_dir), train=True, download=True, transform=transform)
    val = datasets.MNIST(str(data_dir), train=False, download=True, transform=transform)
    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


@torch.no_grad()
def _evaluate(model: AutoEncoder, loader: DataLoader, device: torch.device) -> dict[str, object]:
    model.eval()
    totals = {"sum": 0.0, "count": 0}
    per_class = {digit: {"sum": 0.0, "count": 0} for digit in range(10)}
    loss_fn = nn.MSELoss(reduction="none")

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        flat = x.view(x.size(0), -1)
        recon = model(x)["recon"]
        sample_loss = loss_fn(recon, flat).mean(dim=1)
        totals["sum"] += float(sample_loss.sum().item())
        totals["count"] += int(sample_loss.numel())
        for digit in range(10):
            mask = y == digit
            if mask.any():
                vals = sample_loss[mask]
                per_class[digit]["sum"] += float(vals.sum().item())
                per_class[digit]["count"] += int(vals.numel())

    class_mse = {
        str(digit): per_class[digit]["sum"] / max(per_class[digit]["count"], 1)
        for digit in range(10)
    }
    return {
        "mse": totals["sum"] / max(totals["count"], 1),
        "per_class_mse": class_mse,
    }


@torch.no_grad()
def _fixed_digit_batch(dataset: datasets.MNIST, samples_per_class: int) -> tuple[torch.Tensor, torch.Tensor]:
    targets = torch.as_tensor(dataset.targets)
    xs: list[torch.Tensor] = []
    ys: list[int] = []
    for digit in range(10):
        idxs = (targets == digit).nonzero(as_tuple=True)[0][:samples_per_class]
        for idx in idxs.tolist():
            x, y = dataset[idx]
            xs.append(x)
            ys.append(int(y))
    return torch.stack(xs), torch.as_tensor(ys)


@torch.no_grad()
def _save_reconstruction_grid(
    model: AutoEncoder,
    dataset: datasets.MNIST,
    device: torch.device,
    out_path: Path,
    *,
    samples_per_class: int = 4,
) -> None:
    model.eval()
    x, y = _fixed_digit_batch(dataset, samples_per_class)
    recon = model(x.to(device))["recon"].cpu().view_as(x)

    rows = 10
    cols = samples_per_class * 2
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 0.8, rows * 0.8))
    for row, digit in enumerate(range(10)):
        idxs = (y == digit).nonzero(as_tuple=True)[0]
        for col, idx in enumerate(idxs.tolist()):
            axes[row, col * 2].imshow(x[idx, 0], cmap="gray", vmin=0.0, vmax=1.0)
            axes[row, col * 2 + 1].imshow(recon[idx, 0], cmap="gray", vmin=0.0, vmax=1.0)
            axes[row, col * 2].axis("off")
            axes[row, col * 2 + 1].axis("off")
        axes[row, 0].set_ylabel(str(digit), rotation=0, labelpad=10, va="center")
    fig.suptitle("MNIST fixed AE ceiling: original/reconstruction pairs")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


@torch.no_grad()
def _save_ir_comparison_grid(
    model: AutoEncoder,
    replay: IntrinsicReplay,
    dataset: datasets.MNIST,
    device: torch.device,
    out_path: Path,
    *,
    samples_per_class: int = 6,
) -> dict[str, object]:
    model.eval()
    x, y = _fixed_digit_batch(dataset, samples_per_class)
    recon = model(x.to(device))["recon"].cpu().view_as(x)

    ir_rows: list[torch.Tensor] = []
    ir_stats: dict[str, dict[str, float]] = {}
    for digit in range(10):
        samples = replay.sample_image_tensors(digit, samples_per_class, view_shape=(1, 28, 28))
        samples = samples.detach().cpu().clamp(0.0, 1.0)
        ir_rows.append(samples)
        ir_stats[str(digit)] = {
            "pixel_mean": float(samples.mean().item()),
            "pixel_std": float(samples.std(unbiased=False).item()),
            "pixel_min": float(samples.min().item()),
            "pixel_max": float(samples.max().item()),
            "sparsity_lt_0.05": float((samples < 0.05).float().mean().item()),
            "saturation_gt_0.95": float((samples > 0.95).float().mean().item()),
        }

    rows = 30
    cols = samples_per_class
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 0.9, rows * 0.55))
    for digit in range(10):
        clean_idxs = (y == digit).nonzero(as_tuple=True)[0].tolist()
        row_base = digit * 3
        for col, idx in enumerate(clean_idxs):
            axes[row_base, col].imshow(x[idx, 0], cmap="gray", vmin=0.0, vmax=1.0)
            axes[row_base + 1, col].imshow(recon[idx, 0], cmap="gray", vmin=0.0, vmax=1.0)
            axes[row_base + 2, col].imshow(ir_rows[digit][col, 0], cmap="gray", vmin=0.0, vmax=1.0)
            for row in range(row_base, row_base + 3):
                axes[row, col].axis("off")
        axes[row_base, 0].set_ylabel(f"{digit} clean", rotation=0, labelpad=28, va="center")
        axes[row_base + 1, 0].set_ylabel("recon", rotation=0, labelpad=28, va="center")
        axes[row_base + 2, 0].set_ylabel("IR", rotation=0, labelpad=28, va="center")

    fig.suptitle("Fixed all-class AE: clean / reconstruction / intrinsic replay")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return {"per_class_ir_pixel_stats": ir_stats}


def train(args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _device(args.device)
    train_loader, val_loader = _mnist_loaders(args.data_dir, args.batch_size, args.num_workers)

    model = AutoEncoder(
        input_dim=28 * 28,
        hidden_sizes=args.hidden_sizes,
        activation=args.activation,
        activation_last=args.activation_last,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {}
    run_id = None

    if mlflow is not None and args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(args.mlflow_experiment)
        mlflow.start_run(run_name=args.mlflow_run_name)
        run_id = mlflow.active_run().info.run_id
        mlflow.log_params(
            {
                "hidden_sizes": ",".join(map(str, args.hidden_sizes)),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "activation": args.activation,
                "activation_last": args.activation_last,
                "seed": args.seed,
                "diagnostic": "fixed_ae_all_classes_ceiling",
            }
        )

    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            total = 0.0
            count = 0
            for x, _ in train_loader:
                x = x.to(device)
                flat = x.view(x.size(0), -1)
                optimizer.zero_grad(set_to_none=True)
                recon = model(x)["recon"]
                loss = loss_fn(recon, flat)
                loss.backward()
                optimizer.step()
                total += float(loss.item()) * int(x.size(0))
                count += int(x.size(0))

            train_mse = total / max(count, 1)
            val_metrics = _evaluate(model, val_loader, device)
            metrics = {
                "train/mse": train_mse,
                "val/mse": float(val_metrics["mse"]),
            }
            for digit, value in val_metrics["per_class_mse"].items():
                metrics[f"val/class_{digit}_mse"] = float(value)

            if mlflow is not None and args.mlflow_tracking_uri:
                mlflow.log_metrics(metrics, step=epoch)
            print(
                f"epoch {epoch:03d}/{args.epochs} "
                f"train_mse={train_mse:.6f} val_mse={float(val_metrics['mse']):.6f}",
                flush=True,
            )

        grid_path = out_dir / "fixed_ae_ceiling_reconstructions.png"
        _save_reconstruction_grid(model, val_loader.dataset, device, grid_path)

        replay = IntrinsicReplay(
            model.encoder,
            model.decoder,
            eps=args.ir_cov_eps,
            device=device,
            sampling_mode=args.ir_sampling_mode,
            cov_shrinkage=args.ir_cov_shrinkage,
            noise_scale=args.ir_noise_scale,
        )
        replay.fit(train_loader, class_filter=range(10))
        ir_grid_path = out_dir / "fixed_ae_ceiling_ir_comparison.png"
        ir_summary = _save_ir_comparison_grid(
            model,
            replay,
            val_loader.dataset,
            device,
            ir_grid_path,
            samples_per_class=args.ir_samples_per_class,
        )
        model_path = out_dir / "fixed_ae_ceiling_model.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "hidden_sizes": args.hidden_sizes,
                "activation": args.activation,
                "activation_last": args.activation_last,
                "input_dim": 28 * 28,
            },
            model_path,
        )
        final_metrics = _evaluate(model, val_loader, device)
        summary = {
            "hidden_sizes": args.hidden_sizes,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "activation": args.activation,
            "activation_last": args.activation_last,
            "seed": args.seed,
            "device": str(device),
            "val_mse": final_metrics["mse"],
            "per_class_mse": final_metrics["per_class_mse"],
            "reconstruction_grid": str(grid_path),
            "ir_comparison_grid": str(ir_grid_path),
            "model_path": str(model_path),
            "ir_sampling_mode": args.ir_sampling_mode,
            "ir_cov_eps": args.ir_cov_eps,
            "ir_cov_shrinkage": args.ir_cov_shrinkage,
            "ir_noise_scale": args.ir_noise_scale,
            "intrinsic_replay": replay.describe(),
            **ir_summary,
            "mlflow_run_id": run_id,
        }
        summary_path = out_dir / "fixed_ae_ceiling_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")

        if mlflow is not None and args.mlflow_tracking_uri:
            mlflow.log_dict(summary, "diagnostics/fixed_ae_ceiling_summary.json")
            mlflow.log_artifact(str(grid_path), artifact_path="figures")
            mlflow.log_artifact(str(ir_grid_path), artifact_path="figures")
            mlflow.log_artifact(str(model_path), artifact_path="models")
            mlflow.log_artifact(str(summary_path), artifact_path="diagnostics")
    finally:
        if mlflow is not None and args.mlflow_tracking_uri:
            mlflow.end_run()

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-sizes", type=_parse_sizes, default=[240, 120, 106, 29])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--activation-last", default="sigmoid")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "MNIST")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "diagnostics" / "ae_ceiling",
    )
    parser.add_argument("--mlflow-tracking-uri", default=f"sqlite:///{ROOT / 'mlflow.db'}")
    parser.add_argument("--mlflow-experiment", default="neurogenesis-diagnostics")
    parser.add_argument("--mlflow-run-name", default="fixed_ae_all_classes_ceiling")
    parser.add_argument("--ir-sampling-mode", default="gaussian_full")
    parser.add_argument("--ir-cov-eps", type=float, default=1e-5)
    parser.add_argument("--ir-cov-shrinkage", type=float, default=0.0)
    parser.add_argument("--ir-noise-scale", type=float, default=1.0)
    parser.add_argument("--ir-samples-per-class", type=int, default=6)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = train(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
