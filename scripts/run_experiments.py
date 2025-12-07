"""Command-line entry point for running Neurogenesis experiments headlessly."""

from __future__ import annotations

import csv
import inspect
import io
import math
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import hydra
import numpy as np
import torch
import tqdm
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.mnist_datamodule import MNISTDataModule
from models.ng_autoencoder import NGAutoEncoder
from training.base_pretrainer import AutoencoderPretrainer, PretrainingConfig
from training.incremental_trainer import IncrementalTrainer
from training.neurogenesis_trainer import NeurogenesisTrainer
from utils.intrinsic_replay import IntrinsicReplay
from utils.thresholds import ThresholdEstimationConfig, ThresholdEstimator

try:  # optional dependency
    from utils.viz_utils import plot_partial_recon_grid_mlflow, plot_recon_grid_mlflow
except ModuleNotFoundError:  # pragma: no cover - viz utils optional in CI
    plot_partial_recon_grid_mlflow = None
    plot_recon_grid_mlflow = None

try:
    import mlflow
except ImportError:  # pragma: no cover - optional dependency
    mlflow = None


class _MLflowLogger:
    def __init__(
        self,
        *,
        tracking_uri: str,
        experiment_name: str,
        run_name: str,
        metric_filter: dict | None = None,
    ) -> None:
        if mlflow is None:
            raise RuntimeError("mlflow is not installed; disable MLflow logging or install mlflow.")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._run = mlflow.start_run(run_name=run_name)
        filt = metric_filter or {}
        include_prefixes = filt.get("include_prefixes") or []
        exclude_prefixes = filt.get("exclude_prefixes") or []
        self._include_prefixes: tuple[str, ...] = tuple(include_prefixes)
        self._exclude_prefixes: tuple[str, ...] = tuple(exclude_prefixes)

    def _should_log(self, key: str) -> bool:
        if self._include_prefixes:
            return any(key.startswith(prefix) for prefix in self._include_prefixes)
        if self._exclude_prefixes and any(key.startswith(prefix) for prefix in self._exclude_prefixes):
            return False
        return True

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        filtered = {
            key: float(value)
            for key, value in metrics.items()
            if self._should_log(str(key))
        }
        if filtered:
            mlflow.log_metrics(filtered, step=step)

    def log_params(self, params: dict) -> None:
        mlflow.log_params(params)

    def log_dict(self, payload: dict, artifact_file: str) -> None:
        mlflow.log_dict(payload, artifact_file)

    def log_text(self, text: str, artifact_file: str) -> None:
        mlflow.log_text(text, artifact_file)

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_image(self, image_bytes: bytes, artifact_file: str) -> None:
        # mlflow.log_image expects numpy/PIL/mlflow.Image in some versions.
        # If we are given raw bytes (e.g., PNG), fall back to log_artifact.
        parent = Path(artifact_file).parent.as_posix()
        fname = Path(artifact_file).name

        if isinstance(image_bytes, (bytes, bytearray)):
            with tempfile.TemporaryDirectory() as td:
                out_path = Path(td) / fname
                with open(out_path, "wb") as fh:
                    fh.write(image_bytes)
                mlflow.log_artifact(
                    str(out_path), artifact_path=(parent if parent != "." else None)
                )
            return

        try:
            mlflow.log_image(image_bytes, artifact_file)
        except Exception:
            with tempfile.TemporaryDirectory() as td:
                out_path = Path(td) / fname
                # best-effort: if this is a PIL image-like object with .save
                if hasattr(image_bytes, "save"):
                    image_bytes.save(out_path)
                    mlflow.log_artifact(
                        str(out_path), artifact_path=(parent if parent != "." else None)
                    )
                else:
                    raise

    def log_figure(self, fig, artifact_file: str) -> None:
        mlflow.log_figure(fig, artifact_file)

    def finish(self) -> None:
        mlflow.end_run()


class _NullLogger:
    def log_metrics(self, metrics: dict, step: int | None = None) -> None:  # noqa: D401
        return None

    def log_params(self, params: dict) -> None:  # noqa: D401
        return None

    def log_dict(self, payload: dict, artifact_file: str) -> None:  # noqa: D401
        return None

    def log_text(self, text: str, artifact_file: str) -> None:  # noqa: D401
        return None

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:  # noqa: D401
        return None

    def log_image(self, image_bytes: bytes, artifact_file: str) -> None:  # noqa: D401
        return None

    def log_figure(self, fig, artifact_file: str) -> None:  # noqa: D401
        return None

    def finish(self) -> None:  # noqa: D401
        return None


class _TrainingDataReplay:
    """
    Simple replay buffer that stores flattened real samples per class and
    returns them on demand instead of generating intrinsic samples.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        max_per_class: int | None = None,
        storage_device: torch.device | str = "cpu",
    ) -> None:
        self.input_dim = int(input_dim)
        self.max_per_class = None if max_per_class is None else int(max_per_class)
        self.device = torch.device(storage_device)
        self.stats: Dict[int, Dict[str, torch.Tensor]] = {}
        self._class_weights: Dict[int, float] = {}

    def reset(self) -> None:
        self.stats.clear()
        self._class_weights = {}

    def _normalize_samples(self, samples: torch.Tensor) -> torch.Tensor:
        flat = samples.view(samples.size(0), -1)
        if flat.size(1) != self.input_dim:
            raise ValueError(
                f"Replay sample dimensionality mismatch: expected {self.input_dim}, got {flat.size(1)}"
            )
        return flat.to(self.device)

    @torch.no_grad()
    def fit(
        self,
        dataloader: DataLoader,
        *,
        class_filter: Iterable[int] | None = None,
    ) -> None:
        filtered = set(int(c) for c in class_filter) if class_filter is not None else None
        bucket: Dict[int, list[torch.Tensor]] = {}
        quota = {cls: self.max_per_class for cls in (filtered or [])}

        for images, labels in dataloader:
            if filtered is not None:
                mask = [int(lbl) in filtered for lbl in labels.tolist()]
                if not any(mask):
                    continue
            flat = self._normalize_samples(images)
            for sample, cls in zip(flat, labels.tolist()):
                cls = int(cls)
                if filtered is not None and cls not in filtered:
                    continue
                if self.max_per_class is not None:
                    remaining = quota.get(cls, self.max_per_class)
                    if remaining is not None and remaining <= 0:
                        continue
                    quota[cls] = remaining - 1 if remaining is not None else None
                bucket.setdefault(cls, []).append(sample.detach().cpu())

        for cls, samples in bucket.items():
            if not samples:
                continue
            stacked = torch.stack(samples, dim=0)
            if self.max_per_class is not None and stacked.size(0) > self.max_per_class:
                perm = torch.randperm(stacked.size(0))[: self.max_per_class]
                stacked = stacked[perm]
            self.stats[cls] = {
                "samples": stacked.to(self.device),
                "count": int(stacked.size(0)),
            }

        if bucket:
            self._refresh_default_weights()

    def available_classes(self) -> list[int]:
        return sorted(self.stats.keys())

    def _refresh_default_weights(self) -> None:
        if not self.stats:
            self._class_weights = {}
            return
        uniform = 1.0 / len(self.stats)
        self._class_weights = {cls: uniform for cls in self.stats}

    def set_class_weights(self, weights: Dict[int, float]) -> None:
        if not weights:
            self._class_weights = {}
            return
        total = float(sum(weights.values()))
        if total <= 0:
            raise ValueError("Class weights must sum to a positive value")
        self._class_weights = {int(k): float(v) for k, v in weights.items()}

    def get_class_weights(self) -> Dict[int, float]:
        return {int(k): float(v) for k, v in self._class_weights.items()}

    def describe(self) -> Dict[int, Dict[str, float]]:
        return {
            int(cls): {"count": float(stats.get("count", 0))}
            for cls, stats in self.stats.items()
        }

    def _sample_from_class(self, cls: int, n: int) -> torch.Tensor:
        if cls not in self.stats:
            raise KeyError(f"No dataset replay samples stored for class {cls}")
        samples = self.stats[cls]["samples"]
        if samples.numel() == 0:
            raise RuntimeError(f"Replay buffer for class {cls} is empty")
        idx = torch.randint(0, samples.size(0), (n,), device=samples.device)
        return samples.index_select(0, idx).to(self.device)

    @torch.no_grad()
    def sample_images(
        self,
        cls: int | None,
        n: int,
        *,
        class_weights: Dict[int, float] | None = None,
    ) -> torch.Tensor:
        if not self.stats:
            raise RuntimeError("Dataset replay buffer is empty; call fit() first.")
        if cls is None:
            weights = class_weights or self._class_weights
            if not weights:
                raise RuntimeError("Class weights undefined for dataset replay sampling.")
            classes = list(weights.keys())
            probs = torch.tensor(
                [weights[c] for c in classes],
                dtype=torch.float32,
                device=self.device,
            )
            probs = probs / probs.sum()
            draws = torch.multinomial(probs, num_samples=n, replacement=True)
            parts: list[torch.Tensor] = []
            for cls_idx, count in zip(*torch.unique(draws, return_counts=True)):
                parts.append(
                    self._sample_from_class(int(classes[int(cls_idx)]), int(count))
                )
            return torch.cat(parts, dim=0)
        return self._sample_from_class(int(cls), int(n))

def _apply_experiment_model_overrides(cfg: DictConfig) -> None:
    exp_model = cfg.experiment.get("model") if hasattr(cfg.experiment, "get") else None
    if exp_model:
        cfg.model = OmegaConf.merge(cfg.model, exp_model)


def _infer_image_side(model_cfg: DictConfig) -> int | None:
    input_dim = getattr(model_cfg, "input_dim", None)
    if input_dim is None:
        return None
    side = int(math.isqrt(int(input_dim)))
    if side * side != int(input_dim):
        raise ValueError(
            f"Model input_dim={input_dim} is not a perfect square; cannot infer image side."
        )
    return side


def _instantiate_datamodule(cfg: DictConfig, model_cfg: DictConfig) -> MNISTDataModule:
    name = cfg.name
    if name == "mnist":
        dm = MNISTDataModule(
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            data_dir=cfg.data_dir,
        )
    elif name == "sd19":
        from data.sd19_datamodule import SD19DataModule  # lazy import

        image_size = cfg.get("image_size", None)
        if image_size is None:
            image_size = _infer_image_side(model_cfg)

        dm = SD19DataModule(
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            data_dir=cfg.data_dir,
            val_split=cfg.get("val_split", 0.2),
            download=cfg.get("download", True),
            download_url=cfg.get("download_url", None),
            download_root=cfg.get("download_root", None),
            image_size=image_size,
            offline_resize=cfg.get("offline_resize", True),
            resize_progress_bar=cfg.get("resize_progress_bar", True),
            resize_progress_min_files=cfg.get("resize_progress_min_files", 500),
            default_per_class_limit_train=cfg.get("default_per_class_limit_train", None),
            default_per_class_limit_val=cfg.get("default_per_class_limit_val", None),
        )
    elif name == "toy":
        from data.toy_datamodule import ToyDataModule  # lazy import

        dm = ToyDataModule(
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            data_dir=cfg.get("data_dir", None),
            num_classes=cfg.get("num_classes", 4),
            train_samples_per_class=cfg.get("train_samples_per_class", 32),
            val_samples_per_class=cfg.get("val_samples_per_class", 16),
            noise_scale=cfg.get("noise_scale", 0.05),
            seed=cfg.get("seed", 0),
        )
    else:
        raise ValueError(f"Unknown dataset '{name}'")

    dm.setup()
    return dm


def _dataloader(dataset, *, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    # Tune loader for throughput without changing behavior.
    # - pin_memory only helps on CUDA; avoid extra overhead on CPU.
    # - keep workers alive across epochs for speed.
    # - small prefetch improves pipeline when workers > 0.
    use_cuda = torch.cuda.is_available()
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": use_cuda,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        **kwargs,
    )


def _train_loader_for_classes(dm, classes: Sequence[int], *, shuffle: bool) -> DataLoader:
    """
    Reuse a datamodule's persistent train loader by updating its sampler indices.
    Falls back to building a fresh loader if sampler mutation is unsupported.
    """
    # Prefer class/combined loaders on the data module (they reuse workers).
    loader = None
    if len(classes) == 1 and hasattr(dm, "get_class_dataloader"):
        try:
            loader = dm.get_class_dataloader(int(classes[0]))
        except Exception:
            loader = None
    elif hasattr(dm, "get_combined_dataloader"):
        try:
            loader = dm.get_combined_dataloader(classes)
        except Exception:
            loader = None

    # If sampler is mutable, reshuffle indices when requested.
    if loader is not None and hasattr(loader, "sampler") and hasattr(loader.sampler, "set"):
        idxs = list(getattr(loader.sampler, "idxs", []))
        if shuffle and len(idxs) > 1:
            perm = torch.randperm(len(idxs)).tolist()
            idxs = [idxs[i] for i in perm]
        try:
            loader.sampler.set(idxs)
            return loader
        except Exception:
            loader = None

    # Fallback: build a one-off loader
    subset = _get_combined_dataset(dm, classes, split="train")
    return _dataloader(
        subset,
        batch_size=getattr(dm, "hparams", {}).get("batch_size", 32)
        if hasattr(dm, "hparams")
        else 32,
        num_workers=0,
        shuffle=shuffle,
    )


def _prep_device(cfg: DictConfig) -> torch.device:
    requested = cfg.training.device
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _log_config_snapshot(cfg: DictConfig, logger) -> None:
    if isinstance(logger, _NullLogger):
        return
    payload = OmegaConf.to_container(cfg, resolve=True)
    try:
        logger.log_dict(payload, "config_snapshot.json")
    except Exception:
        logger.log_text(OmegaConf.to_yaml(cfg), "config_snapshot.yaml")


def _log_eval_records(records: list[dict], logger) -> None:
    if not records or isinstance(logger, _NullLogger):
        return
    with tempfile.NamedTemporaryFile(
        "w", suffix="_metrics.csv", delete=False, newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
                "layer",
                "classes",
                "mean",
                "max",
                "std",
            ],
        )
        writer.writeheader()
        for row in records:
            writer.writerow(row)
        csv_path = Path(handle.name)
    logger.log_artifact(csv_path, artifact_path="metrics")
    csv_path.unlink(missing_ok=True)


def _maybe_make_visual_batch(
    dm, classes: Sequence[int], device: torch.device, *, split: str = "val"
):
    samples_per_class = max(1, min(4, 16 // max(1, len(classes))))
    if hasattr(dm, "make_class_balanced_batch"):
        try:
            batch = dm.make_class_balanced_batch(
                classes,
                samples_per_class=samples_per_class,
                split=split,
                device=device,
                shuffle=False,
                return_labels=True,
            )
            if isinstance(batch, tuple):
                if len(batch) == 3:
                    return batch
                if len(batch) == 2:
                    imgs, labels = batch
                    return imgs, labels, []
        except Exception:
            pass

    tensors: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    splits: list[int] = []
    total = 0

    for cls in classes:
        try:
            subset = _get_class_dataset(dm, cls, split=split)
        except Exception:
            subset = None
        if subset is None:
            continue
        loader = _dataloader(
            subset,
            batch_size=min(samples_per_class, len(subset) if hasattr(subset, "__len__") else samples_per_class),
            num_workers=0,
            shuffle=False,
        )
        try:
            batch = next(iter(loader))
        except StopIteration:
            continue
        images = batch[0][:samples_per_class]
        tensors.append(images)
        labels.append(torch.full((images.size(0),), int(cls), dtype=torch.long))
        total += images.size(0)
        splits.append(total)

    if not tensors:
        return None

    stacked = torch.cat(tensors, dim=0).to(device)
    label_tensor = torch.cat(labels, dim=0).to(device)
    return stacked, label_tensor, splits


def _infer_view_shape(dm) -> tuple[int, ...] | None:
    candidates = []
    for attr in (
        "full_train",
        "full_val",
        "full",
        "train_dataset",
        "val_dataset",
        "dataset",
        "data",
    ):
        ds = getattr(dm, attr, None)
        if ds is not None:
            candidates.append(ds)

    for ds in candidates:
        sample = None
        try:
            if hasattr(ds, "__len__") and len(ds) == 0:
                continue
        except Exception:
            pass
        try:
            sample = ds[0]
        except Exception:
            continue

        if isinstance(sample, (list, tuple)) and sample:
            sample = sample[0]
        if sample is None:
            continue
        if torch.is_tensor(sample):
            return tuple(sample.shape)
        try:
            arr = np.array(sample)
        except Exception:
            continue
        if arr.size == 0:
            continue
        return tuple(arr.shape)
    return None


def _prepare_plot_array(tensor: torch.Tensor, view_shape: tuple[int, ...] | None) -> np.ndarray:
    arr = tensor.detach().cpu().float().numpy()
    if view_shape is not None and arr.shape != view_shape:
        try:
            arr = arr.reshape(view_shape)
        except Exception:
            pass
    if arr.ndim == 3:
        if arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
            if arr.shape[-1] == 1:
                arr = arr[..., 0]
        elif arr.shape[-1] in (1, 3):
            if arr.shape[-1] == 1:
                arr = arr[..., 0]
        else:
            arr = arr.mean(axis=0)
    elif arr.ndim > 3:
        arr = arr.reshape(arr.shape[0], -1)
        arr = arr.mean(axis=0)
    arr = np.clip(arr, 0.0, 1.0)
    return arr


def _log_matplotlib_artifacts(fig, logger, artifact_basename: str) -> None:
    if isinstance(logger, _NullLogger):
        return
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    logger.log_image(buf.getvalue(), f"{artifact_basename}.png")
    logger.log_figure(fig, f"{artifact_basename}.mpl.png")


def _log_intrinsic_replay_examples(
    replay: IntrinsicReplay | None,
    *,
    classes: Sequence[int],
    view_shape: tuple[int, ...] | None,
    logger,
    step: int,
    samples_per_class: int,
    timeline_records: list[dict] | None = None,
    stage_label: str | None = None,
) -> None:
    if not isinstance(replay, IntrinsicReplay):
        return
    if view_shape is None or isinstance(logger, _NullLogger):
        return
    classes = sorted({int(c) for c in classes})
    if not classes or samples_per_class <= 0:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    collected: list[tuple[int, torch.Tensor]] = []
    timeline_entry: dict | None = None
    if timeline_records is not None:
        timeline_entry = {
            "step": step,
            "label": stage_label or f"Step {step}",
            "class_samples": {},
        }

    for cls in classes:
        try:
            imgs = replay.sample_image_tensors(cls, samples_per_class, view_shape=view_shape)
        except Exception:
            continue
        imgs = imgs.detach().cpu()
        collected.append((int(cls), imgs))
        if timeline_entry is not None:
            timeline_entry["class_samples"][int(cls)] = imgs.clone()

    if not collected:
        return

    fig, axes = plt.subplots(
        len(collected),
        samples_per_class,
        figsize=(max(6.0, samples_per_class * 1.3), max(3.0, len(collected) * 1.3)),
        squeeze=False,
    )
    cmap = "gray" if len(view_shape) >= 1 and view_shape[0] == 1 else None
    for row_idx, (cls, imgs) in enumerate(collected):
        for col_idx in range(min(samples_per_class, imgs.size(0))):
            ax = axes[row_idx][col_idx]
            arr = _prepare_plot_array(imgs[col_idx], view_shape)
            if arr.ndim == 2:
                ax.imshow(arr, cmap=cmap, vmin=0, vmax=1)
            else:
                ax.imshow(arr, vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(f"Class {cls}", rotation=0, labelpad=30, ha="right", va="center", fontsize=9)
        for col_idx in range(imgs.size(0), samples_per_class):
            axes[row_idx][col_idx].axis("off")

    fig.suptitle(f"Intrinsic replay samples (step {step})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _log_matplotlib_artifacts(fig, logger, f"figures/ir_examples_step_{step}")
    try:
        plt.close(fig)
    except Exception:
        pass

    if timeline_entry is not None and timeline_entry["class_samples"]:
        timeline_records.append(timeline_entry)


def _log_ir_timeline(
    records: list[dict],
    *,
    view_shape: tuple[int, ...] | None,
    logger,
) -> None:
    if not records or view_shape is None or isinstance(logger, _NullLogger):
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    class_ids = sorted({cls for rec in records for cls in rec.get("class_samples", {}).keys()})
    if not class_ids:
        return
    n_rows = len(records)
    n_cols = len(class_ids)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(6.0, n_cols * 1.4), max(4.0, n_rows * 1.2)),
        squeeze=False,
    )
    cmap = "gray" if len(view_shape) >= 1 and view_shape[0] == 1 else None
    for row_idx, rec in enumerate(records):
        label = rec.get("label") or f"Step {rec.get('step', row_idx)}"
        class_samples = rec.get("class_samples", {})
        for col_idx, cls in enumerate(class_ids):
            ax = axes[row_idx][col_idx]
            samples = class_samples.get(cls)
            if samples is None or samples.numel() == 0:
                ax.axis("off")
                continue
            arr = _prepare_plot_array(samples[0], view_shape)
            if arr.ndim == 2:
                ax.imshow(arr, cmap=cmap, vmin=0, vmax=1)
            else:
                ax.imshow(arr, vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(f"Class {cls}", fontsize=9)
        axes[row_idx][0].set_ylabel(label, rotation=0, labelpad=25, ha="right", va="center", fontsize=9)

    fig.suptitle("Intrinsic replay timeline", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _log_matplotlib_artifacts(fig, logger, "figures/ir_timeline")
    try:
        plt.close(fig)
    except Exception:
        pass

def _log_reconstruction_artifacts(
    model,
    dm,
    classes: Sequence[int],
    learned_count: int,
    device: torch.device,
    logger,
    step: int,
) -> None:
    if isinstance(logger, _NullLogger) or plot_recon_grid_mlflow is None:
        return
    sample = _maybe_make_visual_batch(dm, classes, device)
    if sample is None:
        return
    images, _, splits = sample
    view_shape = images.shape[1:]
    total_cols = images.shape[0]
    if total_cols == 0:
        return

    def _boundary_from_splits() -> int:
        if learned_count <= 0:
            return 0
        capped_classes = min(learned_count, len(classes))
        if splits:
            idx = min(capped_classes, len(splits))
            return min(total_cols, splits[idx - 1])
        frac = capped_classes / max(1, len(classes))
        return min(total_cols, max(0, int(round(total_cols * frac))))

    boundary = _boundary_from_splits()
    group_titles: Optional[list[str]] = None
    group_splits: Optional[list[int]] = None
    if 0 < boundary < total_cols:
        group_titles = ["Learned", "Unlearned"]
        group_splits = [boundary, total_cols]

    ncols = total_cols
    recon_figsize = (max(6.0, ncols * 0.6), max(4.0, 6.0))

    cpu_images = images.detach().cpu()

    fig, png_bytes = plot_recon_grid_mlflow(
        model,
        cpu_images,
        view_shape=view_shape,
        ncols=ncols,
        figsize=recon_figsize,
        return_mlflow_artifact=True,
        artifact_name="reconstructions.png",
    )
    logger.log_image(png_bytes, f"figures/reconstructions_step_{step}.png")
    logger.log_figure(fig, f"figures/reconstructions_step_{step}.mpl.png")
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:
        pass

    if plot_partial_recon_grid_mlflow is not None:
        try:
            levels = list(range(len(model.hidden_sizes)))
            partial_figsize = (
                max(10.0, ncols * 0.5),
                max(6.0, (len(levels) + 1) * 1.5),
            )
            col_titles = group_titles
            col_splits = group_splits
            fig_partial, png_partial = plot_partial_recon_grid_mlflow(
                model,
                cpu_images,
                view_shape=view_shape,
                levels=levels,
                ncols=ncols,
                figsize=partial_figsize,
                col_group_titles=col_titles,
                col_group_splits=col_splits,
                return_mlflow_artifact=True,
            )
            logger.log_image(png_partial, f"figures/partial_reconstructions_step_{step}.png")
            logger.log_figure(
                fig_partial, f"figures/partial_reconstructions_step_{step}.mpl.png"
            )
            try:
                import matplotlib.pyplot as plt

                plt.close(fig_partial)
            except Exception:
                pass
        except Exception:
            pass


def _log_replay_stats(replay: IntrinsicReplay | None, logger, step: int) -> None:
    if replay is None or isinstance(logger, _NullLogger):
        return
    summary = replay.describe()
    for cls, stats in summary.items():
        payload = {}
        if "count" in stats:
            payload[f"replay/class_{cls}_count"] = stats["count"]
        if "latent_var_mean" in stats:
            payload[f"replay/class_{cls}_latent_var_mean"] = stats["latent_var_mean"]
        if "cov_condition" in stats:
            payload[f"replay/class_{cls}_cov_condition"] = stats["cov_condition"]
        if not payload:
            continue
        logger.log_metrics(payload, step=step)
    weights = replay.get_class_weights()
    if weights:
        logger.log_dict(weights, "replay/class_weights.json")


def _log_training_stats(logger, stats: Dict[str, int]) -> None:
    if isinstance(logger, _NullLogger):
        return
    payload = {f"stats/{key}": float(value) for key, value in stats.items()}
    logger.log_metrics(payload, step=0)


def _log_experiment_identity(logger, cfg: DictConfig) -> None:
    """Record dataset/regime metadata and class curricula as metrics."""
    if isinstance(logger, _NullLogger):
        return
    metrics: dict[str, float] = {}
    dataset = str(cfg.experiment.get("dataset", cfg.data.get("name", "unknown")))
    regime = str(cfg.experiment.get("regime", "unknown"))
    metrics[f"config/dataset/{dataset}"] = 1.0
    metrics[f"config/regime/{regime}"] = 1.0

    base_classes = cfg.experiment.get("base_classes") or []
    for idx, cls in enumerate(base_classes):
        metrics[f"config/pretrain_class/{idx}"] = float(cls)

    incr_classes = cfg.experiment.get("incremental_classes") or []
    for idx, cls in enumerate(incr_classes):
        metrics[f"config/neurogenesis_class/{idx}"] = float(cls)

    if metrics:
        logger.log_metrics(metrics, step=0)


def _build_model(cfg: DictConfig, device: torch.device) -> NGAutoEncoder:
    model = NGAutoEncoder(
        input_dim=cfg.model.input_dim,
        hidden_sizes=list(cfg.model.hidden_sizes),
        activation=cfg.model.activation,
        activation_params=OmegaConf.to_container(cfg.model.get("activation_params", {}), resolve=True)
        if hasattr(cfg.model, "activation_params")
        else None,
        activation_latent=cfg.model.activation_latent,
        activation_latent_params=OmegaConf.to_container(
            cfg.model.get("activation_latent_params", {}), resolve=True
        )
        if hasattr(cfg.model, "activation_latent_params")
        else None,
        activation_last=cfg.model.activation_last,
        activation_last_params=OmegaConf.to_container(
            cfg.model.get("activation_last_params", {}), resolve=True
        )
        if hasattr(cfg.model, "activation_last_params")
        else None,
    )
    return model.to(device)


def _collect_thresholds(
    model: NGAutoEncoder, loader: DataLoader, cfg: DictConfig, logger=None
) -> List[float]:
    thresh_cfg = ThresholdEstimationConfig(
        percentile=cfg.experiment.threshold.percentile,
        margin=cfg.experiment.threshold.margin,
        minimum=cfg.experiment.threshold.minimum,
    )
    estimator = ThresholdEstimator(model, config=thresh_cfg)
    t0 = time.perf_counter()
    loader = tqdm(
        loader,"Threshold Estimation", leave=True)
    vals = estimator.estimate(loader)
    if logger is not None:
        logger.log_metrics({"timing/threshold_estimation_sec": time.perf_counter() - t0})
    return vals


def _dataset_kwargs(func, split: str) -> dict:
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return {}
    params = sig.parameters
    kwargs: dict = {}
    if "split" in params:
        kwargs["split"] = split
    elif "use_val_transforms" in params:
        kwargs["use_val_transforms"] = split != "train"
    return kwargs


def _get_combined_dataset(dm, classes: Sequence[int], split: str = "train"):
    fn = getattr(dm, "get_combined_dataset")
    kwargs = _dataset_kwargs(fn, split)
    return fn(classes, **kwargs)


def _get_class_dataset(dm, class_id: int, split: str = "train"):
    fn = getattr(dm, "get_class_dataset")
    kwargs = _dataset_kwargs(fn, split)
    return fn(class_id, **kwargs)


def _bootstrap_replay(
    replay,
    dm: MNISTDataModule,
    classes: Sequence[int],
    *,
    batch_size: int,
    num_workers: int,
    reset_stats: bool = False,
) -> None:
    if reset_stats:
        reset_fn = getattr(replay, "reset", None)
        if callable(reset_fn):
            reset_fn()
        else:
            if hasattr(replay, "stats"):
                replay.stats.clear()
            if hasattr(replay, "set_class_weights"):
                replay.set_class_weights({})

    t_total = time.perf_counter()
    for cls in classes:
        t_cls = time.perf_counter()
        loader = _train_loader_for_classes(dm, [cls], shuffle=False)
        loader = tqdm(
            loader,
            desc=f"Bootstrapping replay for class {cls}",
            leave=True,
        )
        replay.fit(loader, class_filter=(cls,))
        try:
            # best-effort logging if replay has a logger via outer scope
            pass
        except Exception:
            pass
    # aggregate timing is logged by caller where logger is available


def _pretrain(
    model: NGAutoEncoder, dm: MNISTDataModule, cfg: DictConfig, device: torch.device, logger
) -> AutoencoderPretrainer:
    base_classes = cfg.experiment.base_classes
    train_subset = _get_combined_dataset(dm, base_classes, split="train")
    train_loader = _train_loader_for_classes(dm, base_classes, shuffle=True)

    val_loader = None
    if cfg.training.validate:
        val_subset = _get_combined_dataset(dm, base_classes, split="val")
        val_loader = _dataloader(
            val_subset,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            shuffle=False,
        )

    pre_cfg = PretrainingConfig(
        epochs=cfg.training.pretrain_epochs,
        lr=cfg.training.base_lr,
        weight_decay=cfg.training.weight_decay,
        device=str(device),
    )
    trainer = AutoencoderPretrainer(model, pre_cfg)
    t0 = time.perf_counter()

    train_loader = tqdm(
        train_loader,
        desc="Pretraining",
        leave=False,
    )

    if val_loader is not None:
        val_loader = tqdm(
            val_loader,
            desc="Validation",
            leave=False,
        )

    trainer.fit(train_loader, val_loader=val_loader, log_fn=logger.log_metrics)
    logger.log_metrics({"timing/pretrain_total_sec": time.perf_counter() - t0})
    return trainer


def _evaluate(
    trainer: NeurogenesisTrainer | IncrementalTrainer,
    dm: MNISTDataModule,
    classes: Sequence[int],
    cfg: DictConfig,
    step: int,
    logger,
) -> list[dict]:
    subset = _get_combined_dataset(dm, classes, split="val")
    loader = _dataloader(
        subset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=False,
    )
    t0 = time.perf_counter()
    mean_losses, max_losses, std_losses = trainer.test_all_levels(loader)
    logger.log_metrics({"timing/eval_sec": time.perf_counter() - t0}, step=step)
    classes_repr = ",".join(str(int(c)) for c in classes)
    records: list[dict] = []
    for idx, (mean_v, max_v, std_v) in enumerate(zip(mean_losses, max_losses, std_losses)):
        logger.log_metrics({f"metrics/val_mean_level_{idx}": mean_v}, step=step)
        logger.log_metrics({f"metrics/val_max_level_{idx}": max_v}, step=step)
        logger.log_metrics({f"metrics/val_std_level_{idx}": std_v}, step=step)
        records.append(
            {
                "step": step,
                "layer": idx,
                "classes": classes_repr,
                "mean": mean_v,
                "max": max_v,
                "std": std_v,
            }
        )
    return records


def _resolve_regime(cfg: DictConfig) -> tuple[bool, bool]:
    regime = cfg.experiment.get("regime")
    if regime is None:
        return True, bool(cfg.replay.get("enabled", True))

    mapping = {
        "ndl_ir": (True, True),
        "ndl": (True, False),
        "cl_ir": (False, True),
        "cl": (False, False),
    }
    key = str(regime).lower()
    if key not in mapping:
        raise ValueError(
            f"Unknown experiment regime '{regime}'. Expected one of {sorted(mapping.keys())}."
        )
    return mapping[key]


def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    t_run0 = time.perf_counter()
    _seed_everything(cfg.seed)
    device = _prep_device(cfg)

    _apply_experiment_model_overrides(cfg)

    training_stats = {
        "pretrain_parameter_updates": 0,
        "neurogenesis_parameter_updates": 0,
        "incremental_parameter_updates": 0,
        "total_parameter_updates": 0,
    }

    dm = _instantiate_datamodule(cfg.data, cfg.model)
    view_shape = _infer_view_shape(dm)
    log_cfg = getattr(cfg, "logging", {})
    default_samples = 6
    if hasattr(log_cfg, "get"):
        ir_samples_per_class = int(log_cfg.get("ir_samples_per_class", default_samples) or default_samples)
    else:
        ir_samples_per_class = default_samples
    ir_samples_per_class = max(1, int(ir_samples_per_class))
    ir_timeline_records: list[dict] = []
    mlflow_cfg = cfg.logging.mlflow
    if mlflow_cfg.enabled and mlflow is None:
        print("[WARN] mlflow not available; disabling MLflow logging for this run.")
    if mlflow_cfg.enabled and mlflow is not None:
        logger = _MLflowLogger(
            tracking_uri=mlflow_cfg.tracking_uri,
            experiment_name=mlflow_cfg.experiment_name,
            run_name=mlflow_cfg.run_name,
            metric_filter=mlflow_cfg.get("metric_filter"),
        )
        logger.log_params(
            {
                "dataset": cfg.experiment.dataset,
                "regime": cfg.experiment.regime,
                "base_classes": list(cfg.experiment.base_classes),
                "incremental_classes": list(cfg.experiment.incremental_classes),
                "hidden_sizes": list(cfg.model.hidden_sizes),
                "paper_experiment": getattr(cfg, "paper_experiment", ""),
            }
        )
        _log_config_snapshot(cfg, logger)
    else:
        logger = _NullLogger()

    _log_experiment_identity(logger, cfg)

    model = _build_model(cfg, device)

    t_pre0 = time.perf_counter()
    pretrainer = _pretrain(model, dm, cfg, device, logger)
    logger.log_metrics({"timing/pretrain_sec": time.perf_counter() - t_pre0})
    training_stats["pretrain_parameter_updates"] = getattr(pretrainer, "update_steps", 0)
    model.reset_update_counter()

    replay_mode = str(cfg.replay.get("mode", "intrinsic")).lower()
    use_neurogenesis, regime_replay = _resolve_regime(cfg)
    replay_enabled = bool(cfg.replay.get("enabled", True))
    use_replay = (regime_replay and replay_enabled) or (
        replay_mode == "dataset" and replay_enabled
    )

    thresholds: List[float] = []
    if use_neurogenesis:
        threshold_subset = _get_combined_dataset(dm, cfg.experiment.base_classes, split="train")
        threshold_loader = _dataloader(
            threshold_subset,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            shuffle=False,
        )
        thresholds = _collect_thresholds(model, threshold_loader, cfg, logger=logger)
        if len(thresholds) != len(cfg.model.hidden_sizes):
            raise RuntimeError("Threshold count does not match hidden layers")
        # log resolved thresholds per level
        for i, thr in enumerate(thresholds):
            logger.log_metrics({f"threshold/level_{i}": float(thr)}, step=0)

    replay = None
    if use_replay:
        if replay_mode == "dataset":
            replay = _TrainingDataReplay(
                input_dim=cfg.model.input_dim,
                max_per_class=cfg.replay.get("dataset_max_samples_per_class", None),
                storage_device=cfg.replay.get("dataset_storage_device", "cpu"),
            )
        elif replay_mode == "intrinsic":
            replay = IntrinsicReplay(
                model.encoder, model.decoder, eps=cfg.replay.cov_eps, device=device
            )
        else:
            raise ValueError(
                f"Unknown replay.mode '{cfg.replay.get('mode')}'. Expected 'intrinsic' or 'dataset'."
            )
        t_boot0 = time.perf_counter()
        _bootstrap_replay(
            replay,
            dm,
            cfg.experiment.base_classes,
            batch_size=min(cfg.replay.stats_batch_size, cfg.data.batch_size),
            num_workers=cfg.data.num_workers,
            reset_stats=True,
        )
        logger.log_metrics(
            {"timing/replay_bootstrap_sec": time.perf_counter() - t_boot0},
            step=len(cfg.experiment.base_classes),
        )
        _log_replay_stats(replay, logger, step=len(cfg.experiment.base_classes))

    if use_neurogenesis:
        trainer = NeurogenesisTrainer(
            ae=model,
            ir=replay,
            thresholds=thresholds,
            max_nodes=list(cfg.neurogenesis.max_nodes),
            max_outliers=cfg.neurogenesis.max_outlier_fraction,
            base_lr=cfg.training.base_lr,
            plasticity_epochs=cfg.neurogenesis.plasticity_epochs,
            stability_epochs=cfg.neurogenesis.stability_epochs,
            next_layer_epochs=cfg.neurogenesis.next_layer_epochs,
            factor_max_new_nodes=cfg.neurogenesis.factor_max_new_nodes,
            factor_new_nodes=cfg.neurogenesis.factor_new_nodes,
            logger=logger,
            early_stop_cfg=cfg.neurogenesis.early_stop,
        )
    else:
        trainer = IncrementalTrainer(
            ae=model,
            ir=replay,
            base_lr=cfg.training.base_lr,
            epochs=cfg.training.incremental_epochs,
            weight_decay=cfg.training.weight_decay,
            replay_ratio=cfg.replay.get("batch_ratio", 1.0),
            device=device,
            logger=logger,
        )

    learned: list[int] = list(cfg.experiment.base_classes)
    all_classes: list[int] = list(cfg.experiment.base_classes) + list(
        cfg.experiment.incremental_classes
    )
    trainer.log_global_sizes()
    eval_records: list[dict] = []
    eval_records.extend(_evaluate(trainer, dm, learned, cfg, step=len(learned), logger=logger))
    _log_reconstruction_artifacts(
        model, dm, all_classes, len(learned), device, logger, step=len(learned)
    )
    _log_intrinsic_replay_examples(
        replay,
        classes=learned,
        view_shape=view_shape,
        logger=logger,
        step=len(learned),
        samples_per_class=ir_samples_per_class,
        timeline_records=ir_timeline_records,
        stage_label=f"Step {len(learned)} (base)",
    )

    for stage, class_id in enumerate(cfg.experiment.incremental_classes, start=1):
        subset = _get_class_dataset(dm, class_id, split="train")
        t_learn0 = time.perf_counter()
        loader = _train_loader_for_classes(dm, [class_id], shuffle=True)
        trainer.learn_class(class_id, loader)
        logger.log_metrics(
            {"timing/learn_class_sec": time.perf_counter() - t_learn0}, step=len(learned) + 1
        )
        # recompute replay stats for all learned classes with the updated encoder
        learned_so_far = learned + [class_id]
        trainer.log_global_sizes()
        if replay is not None:
            t_boot_inc0 = time.perf_counter()
            _bootstrap_replay(
                replay,
                dm,
                learned_so_far,
                batch_size=min(cfg.replay.stats_batch_size, cfg.data.batch_size),
                num_workers=cfg.data.num_workers,
                reset_stats=True,
            )
            logger.log_metrics(
                {"timing/replay_update_sec": time.perf_counter() - t_boot_inc0},
                step=len(learned_so_far),
            )
        learned.append(class_id)
        _log_replay_stats(replay, logger, step=len(learned))
        eval_records.extend(_evaluate(trainer, dm, learned, cfg, step=len(learned), logger=logger))
        t_fig0 = time.perf_counter()
        _log_reconstruction_artifacts(
            model, dm, all_classes, len(learned), device, logger, step=len(learned)
        )
        logger.log_metrics(
            {"timing/recon_artifacts_sec": time.perf_counter() - t_fig0}, step=len(learned)
        )
        _log_intrinsic_replay_examples(
            replay,
            classes=learned,
            view_shape=view_shape,
            logger=logger,
            step=len(learned),
            samples_per_class=ir_samples_per_class,
            timeline_records=ir_timeline_records,
            stage_label=f"Step {len(learned)} (added {class_id})",
        )

    if use_neurogenesis:
        training_stats["neurogenesis_parameter_updates"] = getattr(model, "update_steps", 0)
    else:
        training_stats["incremental_parameter_updates"] = getattr(trainer, "update_steps", 0)
    training_stats["total_parameter_updates"] = (
        training_stats["pretrain_parameter_updates"]
        + training_stats["neurogenesis_parameter_updates"]
        + training_stats["incremental_parameter_updates"]
    )
    _log_training_stats(logger, training_stats)

    _log_eval_records(eval_records, logger)
    _log_ir_timeline(ir_timeline_records, view_shape=view_shape, logger=logger)
    logger.log_metrics({"timing/total_run_sec": time.perf_counter() - t_run0})
    logger.finish()
    return {"model": model, "trainer": trainer, "replay": replay, "training_stats": training_stats}


@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
