"""Command-line entry point for running Neurogenesis experiments headlessly."""

from __future__ import annotations

import csv
from copy import deepcopy
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
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm

from data.mnist_datamodule import MNISTDataModule
from models.ng_autoencoder import NGAutoEncoder
from training.base_pretrainer import AutoencoderPretrainer, PretrainingConfig
from training.incremental_trainer import IncrementalTrainer
from training.neurogenesis_trainer import NeurogenesisTrainer
from utils.dataset_replay import DatasetReplay
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
        self._class_metrics: str = str(filt.get("class_metrics", "full")).lower()

    def _allow_class_metric(self, key: str) -> bool:
        if self._class_metrics == "off":
            return False
        if self._class_metrics == "full":
            return True
        if self._class_metrics != "summary":
            return True
        # Summary-only class metrics: keep growth + size + avg loss + outlier fraction.
        if "/growth_level_" in key:
            return True
        if "_level_" in key and (key.endswith("_size") or key.endswith("_cumulative_size")):
            return True
        if "_avg_loss_iter" in key:
            return True
        if key.endswith("_outlier_fraction") or key.endswith("_outlier_fraction_round"):
            return True
        return False

    def _should_log(self, key: str) -> bool:
        if key.startswith("class_") and not self._allow_class_metric(key):
            return False
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
            if hasattr(mlflow, "log_metrics"):
                mlflow.log_metrics(filtered, step=step)
            else:
                for key, value in filtered.items():
                    mlflow.log_metric(key, value, step=step)

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

        if isinstance(image_bytes, (bytes, bytearray)) and hasattr(mlflow, "log_image"):
            try:
                mlflow.log_image(image_bytes, artifact_file)
                return
            except Exception:
                pass

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



def _apply_experiment_model_overrides(cfg: DictConfig) -> None:
    exp_model = cfg.experiment.get("model") if hasattr(cfg.experiment, "get") else None
    if exp_model:
        cfg.model = OmegaConf.merge(cfg.model, exp_model)


def _apply_control_model_overrides(cfg: DictConfig) -> None:
    """Optionally size CL controls to match a completed enlarged NDL network."""
    regime = str(cfg.experiment.get("regime", "")).lower()
    if regime not in {"cl", "cl_ir"}:
        return
    hidden_sizes = cfg.experiment.get("control_hidden_sizes", None)
    if hidden_sizes is None:
        return
    sizes = [int(size) for size in list(hidden_sizes)]
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError(
            "experiment.control_hidden_sizes must be a non-empty list of positive integers."
        )
    cfg.model.hidden_sizes = sizes


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
            invert_colors=cfg.get("invert_colors", False),
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


def _limit_dataset(dataset, limit: int | None):
    if limit is None:
        return dataset
    try:
        n_total = len(dataset)
    except Exception:
        return dataset
    n_take = max(0, min(int(limit), int(n_total)))
    return Subset(dataset, list(range(n_take)))


def _train_loader_for_classes(
    dm,
    classes: Sequence[int],
    *,
    shuffle: bool,
    limit_per_class: int | None = None,
) -> DataLoader:
    """
    Reuse a datamodule's persistent train loader by updating its sampler indices.
    Falls back to building a fresh loader if sampler mutation is unsupported.
    """
    if limit_per_class is not None:
        limited = [
            _limit_dataset(_get_class_dataset(dm, int(cls), split="train"), int(limit_per_class))
            for cls in classes
        ]
        subset = limited[0] if len(limited) == 1 else ConcatDataset(limited)
        batch_size = (
            getattr(dm, "hparams", {}).get("batch_size", 32)
            if hasattr(dm, "hparams")
            else 32
        )
        return _dataloader(subset, batch_size=batch_size, num_workers=0, shuffle=shuffle)

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


def _neurogenesis_early_stop_cfg(cfg: DictConfig) -> dict:
    """Bundle base and diagnostic early-stop overrides for the trainer."""
    early_stop = OmegaConf.to_container(cfg.neurogenesis.early_stop, resolve=True)
    early_stop = dict(early_stop or {})
    for key in (
        "early_stop_by_phase",
        "early_stop_by_level",
        "early_stop_by_phase_and_level",
    ):
        value = cfg.neurogenesis.get(key, None)
        if value:
            early_stop[key] = OmegaConf.to_container(value, resolve=True)
    return early_stop


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
    default_fields = [
        "step",
        "scope",
        "class_id",
        "layer",
        "classes",
        "mean",
        "max",
        "std",
    ]
    extra_fields = sorted(
        {key for row in records for key in row.keys() if key not in default_fields}
    )
    with tempfile.NamedTemporaryFile(
        "w", suffix="_metrics.csv", delete=False, newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[*default_fields, *extra_fields],
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


def _source_index_for_subset(dataset, local_idx: int):
    """Best-effort source index for deterministic diagnostic sample manifests."""

    if isinstance(dataset, Subset):
        mapped = dataset.indices[int(local_idx)]
        return _source_index_for_subset(dataset.dataset, int(mapped))
    return int(local_idx)


def _make_fixed_visual_batch(
    dm,
    classes: Sequence[int],
    device: torch.device,
    *,
    split: str = "val",
    samples_per_class: int | None = None,
):
    """Select deterministic visual samples and return a sample-id manifest."""

    classes = [int(cls) for cls in classes]
    if not classes:
        return None
    if samples_per_class is None:
        samples_per_class = max(1, min(4, 16 // max(1, len(classes))))

    tensors: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    splits: list[int] = []
    manifest: list[dict] = []
    total = 0

    for cls in classes:
        try:
            subset = _get_class_dataset(dm, cls, split=split)
        except Exception:
            continue
        try:
            subset_len = len(subset)
        except Exception:
            subset_len = 0
        if subset_len <= 0:
            continue
        take = min(int(samples_per_class), int(subset_len))
        images: list[torch.Tensor] = []
        for local_idx in range(take):
            sample = subset[local_idx]
            image = sample[0] if isinstance(sample, (tuple, list)) else sample
            if not torch.is_tensor(image):
                image = torch.as_tensor(np.array(image))
            images.append(image.float())
            manifest.append(
                {
                    "class_id": int(cls),
                    "split": split,
                    "subset_index": int(local_idx),
                    "source_index": _source_index_for_subset(subset, local_idx),
                }
            )
        if not images:
            continue
        class_tensor = torch.stack(images, dim=0)
        tensors.append(class_tensor)
        labels.append(torch.full((class_tensor.size(0),), int(cls), dtype=torch.long))
        total += class_tensor.size(0)
        splits.append(total)

    if not tensors:
        return None
    stacked = torch.cat(tensors, dim=0).to(device)
    label_tensor = torch.cat(labels, dim=0).to(device)
    return stacked, label_tensor, splits, manifest


def _unpack_visual_batch(sample):
    if sample is None:
        return None
    if len(sample) == 4:
        return sample
    if len(sample) == 3:
        images, labels, splits = sample
        return images, labels, splits, []
    if len(sample) == 2:
        images, labels = sample
        return images, labels, [], []
    raise ValueError("Unexpected visual batch shape")


def _infer_view_shape(dm) -> tuple[int, ...] | None:
    image_shape = getattr(dm, "image_shape", None)
    if image_shape is not None:
        try:
            return tuple(image_shape)
        except Exception:
            pass
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


def _tensor_stats(flat: torch.Tensor) -> dict:
    if flat.numel() == 0:
        return {}
    flat = flat.detach().float().cpu()
    return {
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "sparsity_le_0_05": float((flat <= 0.05).float().mean().item()),
        "saturation_ge_0_95": float((flat >= 0.95).float().mean().item()),
    }


def _recon_error_summary(model: NGAutoEncoder, flat: torch.Tensor, *, levels: int) -> dict:
    summary: dict[str, dict[str, float]] = {}
    if flat.numel() == 0:
        return summary
    for level in range(levels):
        recon = model.forward_partial(flat, level)
        err = model.reconstruction_error(recon, flat).detach().float().cpu()
        summary[f"level_{level}"] = {
            "mean": float(err.mean().item()),
            "std": float(err.std(unbiased=False).item()),
            "min": float(err.min().item()),
            "max": float(err.max().item()),
        }
    return summary


def _latent_diagnostics(replay: IntrinsicReplay, cls: int, latents: torch.Tensor) -> dict:
    if cls not in replay.stats or latents.numel() == 0:
        return {}
    mu = replay.stats[cls]["mean"].detach()
    L = replay.stats[cls]["L"].detach()
    centered = latents.detach() - mu[None, :]
    euclid = torch.linalg.norm(centered, dim=1).detach().float().cpu()
    diag = torch.clamp(torch.diag(L @ L.T), min=1.0e-12)
    z_diag = centered / torch.sqrt(diag)[None, :]
    diag_maha = torch.linalg.norm(z_diag, dim=1).detach().float().cpu()
    try:
        solved = torch.linalg.solve_triangular(
            L,
            centered.T,
            upper=False,
        ).T
        maha = torch.linalg.norm(solved, dim=1).detach().float().cpu()
    except RuntimeError:
        maha = diag_maha
    cov = (L @ L.T).detach().cpu()
    cond = torch.linalg.cond(cov).item() if cov.numel() > 0 else float("nan")
    return {
        "mean_distance_mean": float(euclid.mean().item()),
        "mean_distance_std": float(euclid.std(unbiased=False).item()),
        "mahalanobis_mean": float(maha.mean().item()),
        "mahalanobis_std": float(maha.std(unbiased=False).item()),
        "diag_mahalanobis_mean": float(diag_maha.mean().item()),
        "diag_mahalanobis_std": float(diag_maha.std(unbiased=False).item()),
        "cov_condition": float(cond),
    }


def _nearest_distance_summary(query: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    if query.numel() == 0 or reference.numel() == 0:
        return {}
    q = query.detach().float()
    r = reference.detach().float()
    distances = torch.cdist(q, r)
    nearest = distances.min(dim=1).values.detach().cpu()
    return {
        "mean": float(nearest.mean().item()),
        "std": float(nearest.std(unbiased=False).item()),
        "min": float(nearest.min().item()),
        "max": float(nearest.max().item()),
    }


def _encode_all_levels(model: NGAutoEncoder, flat: torch.Tensor) -> dict[str, torch.Tensor]:
    encoded: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for level in range(len(model.hidden_sizes)):
            _recon, lat = model.forward_partial(flat, level, ret_lat=True)
            encoded[f"level_{level}"] = lat.detach()
    return encoded


def _feature_stats_by_level(model: NGAutoEncoder, flat: torch.Tensor) -> dict:
    stats: dict[str, dict[str, float]] = {}
    for level, lat in _encode_all_levels(model, flat).items():
        stats[level] = _tensor_stats(lat)
    return stats


def _filter_acceptance_diagnostics(
    replay: IntrinsicReplay,
    cls: int,
    *,
    n_candidates: int,
) -> dict[str, float]:
    if not hasattr(replay, "_sample_class_latent_raw") or not hasattr(replay, "_filter_latents"):
        return {}
    if str(getattr(replay, "filter_mode", "none")) == "none":
        return {"acceptance_rate": 1.0, "accepted": float(n_candidates), "candidates": float(n_candidates)}
    try:
        candidates = replay._sample_class_latent_raw(cls, n_candidates)
        mask = replay._filter_latents(cls, candidates)
    except Exception:
        return {}
    accepted = int(mask.sum().item())
    return {
        "acceptance_rate": float(accepted / max(int(n_candidates), 1)),
        "accepted": float(accepted),
        "candidates": float(n_candidates),
    }


def _log_ir_quality_diagnostics(
    replay: IntrinsicReplay | None,
    model: NGAutoEncoder,
    dm,
    classes: Sequence[int],
    *,
    view_shape: tuple[int, ...] | None,
    cfg: DictConfig,
    device: torch.device,
    logger,
    step: int,
) -> None:
    if not isinstance(replay, IntrinsicReplay) or isinstance(logger, _NullLogger):
        return
    if view_shape is None:
        return
    samples_per_class = int(
        getattr(cfg.logging, "ir_quality_samples_per_class", 64)
        if hasattr(cfg, "logging")
        else 64
    )
    nn_samples_per_class = int(
        getattr(cfg.logging, "ir_nn_samples_per_class", 256)
        if hasattr(cfg, "logging")
        else 256
    )
    samples_per_class = max(1, samples_per_class)
    nn_samples_per_class = max(samples_per_class, nn_samples_per_class)
    plot_per_class = max(1, min(6, samples_per_class))
    classes = sorted({int(c) for c in classes if int(c) in replay.available_classes()})
    if not classes:
        return

    payload: dict[str, Any] = {
        "step": int(step),
        "sampling_mode": replay.sampling_mode,
        "cov_shrinkage": float(replay.cov_shrinkage),
        "noise_scale": float(replay.noise_scale),
        "samples_per_class": int(samples_per_class),
        "classes": {},
    }
    figure_rows: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    input_dim = int(cfg.model.input_dim)

    for cls in classes:
        try:
            subset = _get_class_dataset(dm, cls, split="val")
            loader = _dataloader(
                subset,
                batch_size=nn_samples_per_class,
                num_workers=0,
                shuffle=False,
            )
            batch = next(iter(loader))
        except Exception:
            continue
        clean_nn = batch[0][:nn_samples_per_class].to(device)
        clean = clean_nn[:samples_per_class]
        clean_flat = clean.view(clean.size(0), input_dim)
        clean_nn_flat = clean_nn.view(clean_nn.size(0), input_dim)
        try:
            latents = replay.sample_latent(cls, samples_per_class)
            generated_flat = replay.decoder(latents)
        except Exception:
            continue
        generated_flat = generated_flat.detach()
        with torch.no_grad():
            clean_recon = model(clean_flat)
            if isinstance(clean_recon, dict):
                clean_recon = clean_recon.get("recon")
            clean_recon = clean_recon.detach()
            generated_reencoded = model.encoder(generated_flat)
            clean_top = model.encoder(clean_nn_flat)
            roundtrip = torch.linalg.norm(generated_reencoded - latents, dim=1).detach().cpu()
            top_nn = _nearest_distance_summary(generated_reencoded, clean_top)
            pixel_nn = _nearest_distance_summary(generated_flat, clean_nn_flat)

        payload["classes"][str(cls)] = {
            "clean_pixel_stats": _tensor_stats(clean_flat),
            "clean_recon_pixel_stats": _tensor_stats(clean_recon),
            "generated_pixel_stats": _tensor_stats(generated_flat),
            "clean_feature_stats": _feature_stats_by_level(model, clean_flat),
            "generated_feature_stats": _feature_stats_by_level(model, generated_flat),
            "clean_recon_error": _recon_error_summary(
                model, clean_flat, levels=len(model.hidden_sizes)
            ),
            "generated_recon_error": _recon_error_summary(
                model, generated_flat, levels=len(model.hidden_sizes)
            ),
            "latent": _latent_diagnostics(replay, cls, latents),
            "filter_acceptance": _filter_acceptance_diagnostics(
                replay,
                cls,
                n_candidates=max(samples_per_class, 256),
            ),
            "roundtrip": {
                "mean": float(roundtrip.mean().item()),
                "std": float(roundtrip.std(unbiased=False).item()),
                "min": float(roundtrip.min().item()),
                "max": float(roundtrip.max().item()),
            },
            "nearest_neighbor": {
                "pixel": pixel_nn,
                "top_latent": top_nn,
            },
        }
        figure_rows.append(
            (
                cls,
                clean[:plot_per_class].detach().cpu(),
                clean_recon[:plot_per_class].view(-1, *view_shape).detach().cpu(),
                generated_flat[:plot_per_class].view(-1, *view_shape).detach().cpu(),
            )
        )

    if not payload["classes"]:
        return
    logger.log_dict(payload, f"diagnostics/ir_quality_step_{step}.json")
    metric_payload = {}
    for cls, cls_payload in payload["classes"].items():
        gen_level = cls_payload["generated_recon_error"].get(
            f"level_{len(model.hidden_sizes) - 1}", {}
        )
        clean_level = cls_payload["clean_recon_error"].get(
            f"level_{len(model.hidden_sizes) - 1}", {}
        )
        if "mean" in gen_level:
            metric_payload[f"diagnostics/ir_quality/class_{cls}/generated_top_mse"] = gen_level["mean"]
        if "mean" in clean_level:
            metric_payload[f"diagnostics/ir_quality/class_{cls}/clean_top_mse"] = clean_level["mean"]
        latent = cls_payload.get("latent", {})
        if "mahalanobis_mean" in latent:
            metric_payload[f"diagnostics/ir_quality/class_{cls}/mahalanobis_mean"] = latent["mahalanobis_mean"]
        filt = cls_payload.get("filter_acceptance", {})
        if "acceptance_rate" in filt:
            metric_payload[f"diagnostics/ir_quality/class_{cls}/filter_acceptance_rate"] = filt["acceptance_rate"]
        roundtrip = cls_payload.get("roundtrip", {})
        if "mean" in roundtrip:
            metric_payload[f"diagnostics/ir_quality/class_{cls}/roundtrip_mean"] = roundtrip["mean"]
        nn = cls_payload.get("nearest_neighbor", {})
        if "mean" in nn.get("pixel", {}):
            metric_payload[f"diagnostics/ir_quality/class_{cls}/pixel_nn_mean"] = nn["pixel"]["mean"]
        if "mean" in nn.get("top_latent", {}):
            metric_payload[f"diagnostics/ir_quality/class_{cls}/top_latent_nn_mean"] = nn["top_latent"]["mean"]
    if metric_payload:
        logger.log_metrics(metric_payload, step=step)

    if not figure_rows:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    n_rows = len(figure_rows) * 3
    fig, axes = plt.subplots(
        n_rows,
        plot_per_class,
        figsize=(max(6.0, plot_per_class * 1.2), max(4.0, n_rows * 0.8)),
        squeeze=False,
    )
    cmap = "gray" if len(view_shape) >= 1 and view_shape[0] == 1 else None
    for row_group, (cls, clean, clean_recon, generated) in enumerate(figure_rows):
        for row_offset, (label, imgs) in enumerate(
            (("Clean", clean), ("Recon", clean_recon), ("IR", generated))
        ):
            row_idx = row_group * 3 + row_offset
            for col_idx in range(plot_per_class):
                ax = axes[row_idx][col_idx]
                if col_idx < imgs.size(0):
                    arr = _prepare_plot_array(imgs[col_idx], view_shape)
                    if arr.ndim == 2:
                        ax.imshow(arr, cmap=cmap, vmin=0, vmax=1)
                    else:
                        ax.imshow(arr, vmin=0, vmax=1)
                else:
                    ax.axis("off")
                ax.set_xticks([])
                ax.set_yticks([])
                if col_idx == 0:
                    ax.set_ylabel(
                        f"{cls} {label}",
                        rotation=0,
                        labelpad=28,
                        ha="right",
                        va="center",
                        fontsize=8,
                    )
    fig.suptitle(f"IR quality step {step}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _log_matplotlib_artifacts(fig, logger, f"figures/ir_quality_step_{step}")
    try:
        plt.close(fig)
    except Exception:
        pass


def _render_ir_timeline(records: list[dict], view_shape: tuple[int, ...] | None):
    if not records or view_shape is None:
        return None
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    class_ids = sorted({cls for rec in records for cls in rec.get("class_samples", {}).keys()})
    if not class_ids:
        return None
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
    return fig


def _render_recon_timeline(records: list[dict]):
    if not records:
        return None
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    counts = [rec["original"].size(0) for rec in records if rec.get("original") is not None]
    if not counts:
        return None
    n_cols = max(counts)
    view_shape = tuple(records[0]["original"].shape[1:])
    fig, axes = plt.subplots(
        len(records) * 2,
        n_cols,
        figsize=(max(6.0, n_cols * 1.6), max(4.0, len(records) * 1.5)),
        squeeze=False,
    )
    cmap = "gray" if len(view_shape) >= 1 and view_shape[0] == 1 else None
    for idx, rec in enumerate(records):
        originals = rec.get("original")
        recons = rec.get("recon")
        if originals is None or recons is None:
            continue
        orig_cols = originals.size(0)
        recon_cols = recons.size(0)
        learned_cols = min(rec.get("learned_boundary", orig_cols) or orig_cols, orig_cols)
        row_orig = 2 * idx
        row_recon = row_orig + 1
        for col in range(n_cols):
            if col >= orig_cols or col >= recon_cols:
                axes[row_orig][col].axis("off")
                axes[row_recon][col].axis("off")
                continue
            arr_orig = _prepare_plot_array(originals[col], view_shape)
            axes[row_orig][col].imshow(arr_orig, cmap=cmap, vmin=0, vmax=1)
            axes[row_orig][col].set_xticks([])
            axes[row_orig][col].set_yticks([])
            arr_recon = _prepare_plot_array(recons[col], view_shape)
            axes[row_recon][col].imshow(arr_recon, cmap=cmap, vmin=0, vmax=1)
            axes[row_recon][col].set_xticks([])
            axes[row_recon][col].set_yticks([])
            if col >= learned_cols:
                for ax_sel in (axes[row_orig][col], axes[row_recon][col]):
                    for spine in ax_sel.spines.values():
                        spine.set_edgecolor("orange")
                        spine.set_linewidth(2.0)
        axes[row_orig][0].set_ylabel(rec.get("label", f"Step {idx}"), rotation=0, labelpad=30, ha="right", va="center", fontsize=9)
        axes[row_recon][0].set_ylabel("Recon", rotation=0, labelpad=30, ha="right", va="center", fontsize=9)

    fig.suptitle("Reconstruction timeline", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def _fig_to_image(fig):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    try:
        img = plt.imread(buf, format="png")
    except Exception:
        return None
    return img


def _log_combined_timeline(
    recon_records: list[dict],
    ir_records: list[dict],
    *,
    view_shape: tuple[int, ...] | None,
    logger,
) -> None:
    if isinstance(logger, _NullLogger):
        return
    fig_recon = _render_recon_timeline(recon_records)
    fig_ir = _render_ir_timeline(ir_records, view_shape)
    if fig_recon is None and fig_ir is None:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    images = []
    labels = []
    if fig_recon is not None:
        img = _fig_to_image(fig_recon)
        if img is not None:
            images.append(img)
            labels.append("Reconstruction timeline")
        try:
            plt.close(fig_recon)
        except Exception:
            pass
    if fig_ir is not None:
        img = _fig_to_image(fig_ir)
        if img is not None:
            images.append(img)
            labels.append("Intrinsic replay timeline")
        try:
            plt.close(fig_ir)
        except Exception:
            pass
    if not images:
        return
    heights = [img.shape[0] for img in images]
    width = max(img.shape[1] for img in images)
    fig, axes = plt.subplots(len(images), 1, figsize=(width / 80, sum(heights) / 80))
    if len(images) == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images, labels):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    _log_matplotlib_artifacts(fig, logger, "figures/recon_ir_combined_timeline")
    try:
        plt.close(fig)
    except Exception:
        pass


def _log_ir_timeline(records: list[dict], *, view_shape: tuple[int, ...] | None, logger) -> None:
    if isinstance(logger, _NullLogger):
        return
    fig = _render_ir_timeline(records, view_shape)
    if fig is None:
        return
    _log_matplotlib_artifacts(fig, logger, "figures/ir_timeline")
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:
        pass


def _log_recon_timeline(records: list[dict], logger) -> None:
    if isinstance(logger, _NullLogger):
        return
    fig = _render_recon_timeline(records)
    if fig is None:
        return
    _log_matplotlib_artifacts(fig, logger, "figures/reconstruction_timeline")
    try:
        import matplotlib.pyplot as plt

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
    *,
    timeline_records: list[dict] | None = None,
    stage_label: str | None = None,
    timeline_max_samples: int | None = None,
) -> None:
    if isinstance(logger, _NullLogger) or plot_recon_grid_mlflow is None:
        return
    sample = _make_fixed_visual_batch(dm, classes, device)
    if sample is None:
        sample = _maybe_make_visual_batch(dm, classes, device)
    if sample is None:
        return
    images, _, splits, sample_manifest = _unpack_visual_batch(sample)
    view_shape = images.shape[1:]
    total_cols = images.shape[0]
    if total_cols == 0:
        return
    if sample_manifest:
        logger.log_dict(
            {
                "step": int(step),
                "classes": [int(cls) for cls in classes],
                "samples": sample_manifest,
            },
            f"diagnostics/fixed_visual_samples_step_{step}.json",
        )

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

    if timeline_records is not None:
        outputs = None
        try:
            was_training = bool(getattr(model, "training", False))
        except Exception:
            was_training = None
        with torch.no_grad():
            try:
                if hasattr(model, "eval"):
                    model.eval()
                outputs = model(images)
            finally:
                if was_training is not None and hasattr(model, "train"):
                    model.train(was_training)
        recon = None
        if isinstance(outputs, dict):
            recon = outputs.get("recon")
        else:
            recon = outputs
        if recon is not None:
            recon = recon.view(recon.size(0), *view_shape).detach().cpu()
            max_cols = min(cpu_images.size(0), recon.size(0))
            if timeline_max_samples is not None and timeline_max_samples > 0:
                max_cols = min(max_cols, timeline_max_samples)
            if max_cols > 0:
                timeline_records.append(
                    {
                        "label": stage_label or f"Step {step}",
                        "original": cpu_images[:max_cols].clone(),
                        "recon": recon[:max_cols].clone(),
                        "learned_boundary": min(boundary, max_cols),
                    }
                )

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


def _outlier_stats_for_loader(
    model: NGAutoEncoder,
    loader: DataLoader,
    thresholds: Sequence[float],
    device: torch.device,
) -> list[dict]:
    stats: list[dict] = []
    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        with torch.no_grad():
            for level, threshold in enumerate(thresholds):
                errors: list[torch.Tensor] = []
                for batch in loader:
                    x = batch[0].to(device, non_blocking=True)
                    recon = model.forward_partial(x, level)
                    err = model.reconstruction_error(recon, x)
                    errors.append(err.detach().cpu())
                if errors:
                    all_errs = torch.cat(errors)
                    n_total = int(all_errs.numel())
                    n_out = int((all_errs > float(threshold)).sum().item())
                    mean_v = float(all_errs.mean().item())
                    max_v = float(all_errs.max().item())
                else:
                    n_total = 0
                    n_out = 0
                    mean_v = 0.0
                    max_v = 0.0
                stats.append(
                    {
                        "level": int(level),
                        "threshold": float(threshold),
                        "n_total": n_total,
                        "n_outliers": n_out,
                        "outlier_fraction": n_out / max(n_total, 1),
                        "mean_error": mean_v,
                        "max_error": max_v,
                    }
                )
    finally:
        model.train(was_training)
    return stats


def _log_threshold_audit(
    model: NGAutoEncoder,
    dm,
    cfg: DictConfig,
    thresholds: Sequence[float],
    device: torch.device,
    logger,
) -> None:
    if isinstance(logger, _NullLogger) or not thresholds:
        return
    base_classes = [int(cls) for cls in cfg.experiment.base_classes]
    incremental_classes = [int(cls) for cls in cfg.experiment.incremental_classes]
    classes = base_classes + [cls for cls in incremental_classes if cls not in base_classes]
    records: list[dict] = []
    metrics: dict[str, float] = {}

    for split in ("train", "val"):
        for cls in classes:
            try:
                dataset = _get_class_dataset(dm, cls, split=split)
            except Exception:
                continue
            loader = _dataloader(
                dataset,
                batch_size=cfg.data.batch_size,
                num_workers=cfg.data.num_workers,
                shuffle=False,
            )
            for stat in _outlier_stats_for_loader(model, loader, thresholds, device):
                level = int(stat["level"])
                role = "base" if cls in base_classes else "future"
                record = {
                    "split": split,
                    "class_id": int(cls),
                    "role": role,
                    **stat,
                }
                records.append(record)
                prefix = f"diagnostics/threshold_audit/{split}/class_{cls}/level_{level}"
                metrics[f"{prefix}_outlier_fraction"] = float(stat["outlier_fraction"])
                metrics[f"{prefix}_n_outliers"] = float(stat["n_outliers"])
                metrics[f"{prefix}_mean_error"] = float(stat["mean_error"])

    if metrics:
        logger.log_metrics(metrics, step=0)
    if records:
        logger.log_dict(
            {
                "threshold_source": "manual"
                if cfg.neurogenesis.get("thresholds", None) is not None
                else "estimated",
                "percentile": float(cfg.experiment.threshold.percentile),
                "records": records,
            },
            "diagnostics/threshold_audit.json",
        )


def _group_slice_l2(items: list[tuple[torch.nn.Parameter | None, object | None]]) -> float:
    total = 0.0
    for param, slc in items:
        if param is None:
            continue
        tensor = param if slc is None else param[slc]
        total += float(torch.sum(tensor.detach().float().cpu() ** 2).item())
    return math.sqrt(total)


def _group_grad_l2(items: list[tuple[torch.nn.Parameter | None, object | None]]) -> float:
    total = 0.0
    for param, slc in items:
        if param is None or param.grad is None:
            continue
        grad = param.grad if slc is None else param.grad[slc]
        total += float(torch.sum(grad.detach().float().cpu() ** 2).item())
    return math.sqrt(total)


def _snapshot_group(items: list[tuple[torch.nn.Parameter | None, object | None]]) -> list[torch.Tensor | None]:
    snapshots: list[torch.Tensor | None] = []
    for param, slc in items:
        if param is None:
            snapshots.append(None)
            continue
        tensor = param if slc is None else param[slc]
        snapshots.append(tensor.detach().clone())
    return snapshots


def _group_delta_l2(
    items: list[tuple[torch.nn.Parameter | None, object | None]],
    snapshots: list[torch.Tensor | None],
) -> float:
    total = 0.0
    for (param, slc), before in zip(items, snapshots):
        if param is None or before is None:
            continue
        tensor = param if slc is None else param[slc]
        diff = tensor.detach().cpu().float() - before.detach().cpu().float()
        total += float(torch.sum(diff * diff).item())
    return math.sqrt(total)


def _growth_probe_groups(model: NGAutoEncoder, level: int, num_new: int) -> dict[str, list[tuple[torch.nn.Parameter | None, object | None]]]:
    enc = model._encoder_layer(level)
    dec_mirror = model._decoder_layer(level)
    groups: dict[str, list[tuple[torch.nn.Parameter | None, object | None]]] = {
        "current_encoder_plastic_rows": [(enc.weight_plastic, None), (enc.bias_plastic, None)],
        "current_encoder_mature": [(enc.weight_mature, None), (enc.bias_mature, None)],
        "mirror_decoder_new_input_columns": [
            (dec_mirror.weight_mature, (slice(None), slice(-num_new, None)))
        ],
    }
    if dec_mirror.weight_plastic is not None:
        groups["mirror_decoder_new_input_columns"].append(
            (dec_mirror.weight_plastic, (slice(None), slice(-num_new, None)))
        )
    if level + 1 < len(model.hidden_sizes):
        enc_next = model._encoder_layer(level + 1)
        dec_next = model._decoder_layer(level + 1)
        groups["next_encoder_new_input_columns"] = [
            (enc_next.weight_mature, (slice(None), slice(-num_new, None)))
        ]
        if enc_next.weight_plastic is not None:
            groups["next_encoder_new_input_columns"].append(
                (enc_next.weight_plastic, (slice(None), slice(-num_new, None)))
            )
        groups["next_decoder_plastic_rows"] = [
            (dec_next.weight_plastic, None),
            (dec_next.bias_plastic, None),
        ]
    return groups


def _growth_probe_reconstruction(model: NGAutoEncoder, x: torch.Tensor, level: int, objective: str):
    if objective == "local":
        return model.forward_partial(x, level)
    if objective == "next":
        next_level = min(level + 1, len(model.hidden_sizes) - 1)
        return model.forward_partial(x, next_level)
    if objective == "full":
        out = model(x)
        return out["recon"] if isinstance(out, dict) else out
    raise ValueError(f"Unknown growth probe objective '{objective}'")


def probe_growth_wiring(
    model: NGAutoEncoder,
    sample_images: torch.Tensor,
    *,
    num_new: int = 2,
    lr: float = 1.0e-4,
    device: torch.device | None = None,
) -> list[dict]:
    """Probe computational gradients and plasticity-step updates after growth."""

    if device is None:
        device = next(model.parameters()).device
    x = sample_images.to(device).view(sample_images.size(0), -1)
    records: list[dict] = []
    objectives = ("local", "next", "full")

    for level in range(len(model.hidden_sizes)):
        for objective in objectives:
            if objective == "next" and level + 1 >= len(model.hidden_sizes):
                continue

            conn_model = deepcopy(model).to(device)
            conn_model.add_new_nodes(level, int(num_new))
            conn_model.zero_grad(set_to_none=True)
            for param in conn_model.parameters():
                param.requires_grad_(True)
            conn_groups = _growth_probe_groups(conn_model, level, int(num_new))
            conn_recon = _growth_probe_reconstruction(conn_model, x, level, objective)
            conn_loss = conn_model.reconstruction_error(conn_recon, x).mean()
            conn_loss.backward()
            conn_grad = {name: _group_grad_l2(items) for name, items in conn_groups.items()}

            step_model = deepcopy(model).to(device)
            step_model.add_new_nodes(level, int(num_new))
            step_groups = _growth_probe_groups(step_model, level, int(num_new))
            snapshots = {name: _snapshot_group(items) for name, items in step_groups.items()}
            opt = step_model._optim_plasticity(level, lr)
            step_model.zero_grad(set_to_none=True)
            step_recon = _growth_probe_reconstruction(step_model, x, level, objective)
            step_loss = step_model.reconstruction_error(step_recon, x).mean()
            step_loss.backward()
            step_grad = {name: _group_grad_l2(items) for name, items in step_groups.items()}
            opt.step()
            step_delta = {
                name: _group_delta_l2(items, snapshots[name])
                for name, items in step_groups.items()
            }

            expected_conn = {"current_encoder_plastic_rows", "mirror_decoder_new_input_columns"}
            if objective in {"next", "full"} and level + 1 < len(model.hidden_sizes):
                expected_conn.update({"next_encoder_new_input_columns", "next_decoder_plastic_rows"})
            expected_step = {"current_encoder_plastic_rows", "mirror_decoder_new_input_columns"}

            for group in sorted(step_groups):
                records.append(
                    {
                        "level": int(level),
                        "objective": objective,
                        "group": group,
                        "connectivity_grad_l2": float(conn_grad.get(group, 0.0)),
                        "plasticity_grad_l2": float(step_grad.get(group, 0.0)),
                        "plasticity_delta_l2": float(step_delta.get(group, 0.0)),
                        "expected_connectivity_nonzero": bool(group in expected_conn),
                        "expected_plasticity_update": bool(group in expected_step),
                    }
                )
    return records


def _log_growth_wiring_probe(
    model: NGAutoEncoder,
    dm,
    cfg: DictConfig,
    device: torch.device,
    logger,
) -> None:
    if isinstance(logger, _NullLogger):
        return
    if not bool(cfg.neurogenesis.get("growth_wiring_probe", False)):
        return
    classes = list(cfg.experiment.base_classes) + list(cfg.experiment.incremental_classes)
    sample = _make_fixed_visual_batch(dm, classes, device, samples_per_class=2)
    if sample is None:
        sample = _maybe_make_visual_batch(dm, classes, device)
    if sample is None:
        return
    images, _, _, _ = _unpack_visual_batch(sample)
    records = probe_growth_wiring(
        model,
        images,
        num_new=int(cfg.neurogenesis.get("growth_wiring_nodes", 2)),
        lr=float(cfg.training.base_lr),
        device=device,
    )
    payload = {"records": records}
    logger.log_dict(payload, "diagnostics/growth_wiring.json")
    metrics: dict[str, float] = {}
    for rec in records:
        prefix = (
            f"diagnostics/growth_wiring/level_{rec['level']}/"
            f"{rec['objective']}/{rec['group']}"
        )
        metrics[f"{prefix}_connectivity_grad_l2"] = float(rec["connectivity_grad_l2"])
        metrics[f"{prefix}_plasticity_grad_l2"] = float(rec["plasticity_grad_l2"])
        metrics[f"{prefix}_plasticity_delta_l2"] = float(rec["plasticity_delta_l2"])
    if metrics:
        logger.log_metrics(metrics, step=0)


def _log_tiny_overfit_summary(
    trainer: NeurogenesisTrainer | IncrementalTrainer,
    dm,
    class_id: int,
    cfg: DictConfig,
    logger,
    *,
    step: int,
    limit: int,
) -> dict:
    dataset = _limit_dataset(_get_class_dataset(dm, int(class_id), split="train"), int(limit))
    loader = _dataloader(
        dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=False,
    )
    mean_losses, max_losses, std_losses = trainer.test_all_levels(loader)
    record = {
        "class_id": int(class_id),
        "train_limit": int(limit),
        "mean_by_level": [float(v) for v in mean_losses],
        "max_by_level": [float(v) for v in max_losses],
        "std_by_level": [float(v) for v in std_losses],
    }
    if not isinstance(logger, _NullLogger):
        metrics = {
            f"diagnostics/tiny_overfit/class_{class_id}/train_mean_level_{idx}": float(value)
            for idx, value in enumerate(mean_losses)
        }
        logger.log_metrics(metrics, step=step)
        logger.log_dict(record, "diagnostics/tiny_overfit_summary.json")
    return record


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
        mode=cfg.training.get("pretrain_mode", "end_to_end"),
        denoising_dropout=cfg.training.get("denoising_dropout", 0.0),
        denoising_std=cfg.training.get("denoising_std", 0.0),
        finetune_epochs=cfg.training.get("pretrain_finetune_epochs", 0),
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

    threshold_subset = _get_combined_dataset(dm, base_classes, split="train")
    threshold_loader = _dataloader(
        threshold_subset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=False,
    )

    def _log_pretrain_thresholds(epoch_idx: int) -> None:
        if isinstance(logger, _NullLogger):
            return
        thresh_cfg = ThresholdEstimationConfig(
            percentile=cfg.experiment.threshold.percentile,
            margin=cfg.experiment.threshold.margin,
            minimum=cfg.experiment.threshold.minimum,
        )
        estimator = ThresholdEstimator(model, config=thresh_cfg)
        values = estimator.estimate(threshold_loader)
        metrics = {f"pretrain/threshold_level_{i}": float(v) for i, v in enumerate(values)}
        logger.log_metrics(metrics, step=epoch_idx)

    epoch_hook = _log_pretrain_thresholds if bool(cfg.training.get("log_pretrain_thresholds", True)) else None
    trainer.fit(
        train_loader,
        val_loader=val_loader,
        log_fn=logger.log_metrics,
        epoch_hook=epoch_hook,
    )
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
                "scope": "aggregate",
                "class_id": "",
                "layer": idx,
                "classes": classes_repr,
                "mean": mean_v,
                "max": max_v,
                "std": std_v,
            }
        )

    for cls in classes:
        cls_int = int(cls)
        try:
            cls_subset = _get_class_dataset(dm, cls_int, split="val")
        except Exception:
            continue
        cls_loader = _dataloader(
            cls_subset,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            shuffle=False,
        )
        cls_mean, cls_max, cls_std = trainer.test_all_levels(cls_loader)
        for idx, (mean_v, max_v, std_v) in enumerate(zip(cls_mean, cls_max, cls_std)):
            logger.log_metrics(
                {f"metrics/val_class_{cls_int}_mean_level_{idx}": mean_v}, step=step
            )
            logger.log_metrics(
                {f"metrics/val_class_{cls_int}_max_level_{idx}": max_v}, step=step
            )
            logger.log_metrics(
                {f"metrics/val_class_{cls_int}_std_level_{idx}": std_v}, step=step
            )
            records.append(
                {
                    "step": step,
                    "scope": "class",
                    "class_id": cls_int,
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
    _apply_control_model_overrides(cfg)

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
    recon_timeline_records: list[dict] = []
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
        manual_thresholds = None
        try:
            manual_thresholds = cfg.neurogenesis.get("thresholds", None)
        except Exception:
            manual_thresholds = None

        if manual_thresholds is not None:
            thresholds = [float(v) for v in list(manual_thresholds)]
        else:
            threshold_subset = _get_combined_dataset(dm, cfg.experiment.base_classes, split="train")
            threshold_loader = _dataloader(
                threshold_subset,
                batch_size=cfg.data.batch_size,
                num_workers=cfg.data.num_workers,
                shuffle=False,
            )
            thresholds = _collect_thresholds(model, threshold_loader, cfg, logger=logger)

        if len(thresholds) != len(cfg.model.hidden_sizes):
            raise RuntimeError(
                "Threshold count does not match hidden layers: "
                f"got {len(thresholds)} thresholds for {len(cfg.model.hidden_sizes)} hidden layers."
            )
        thresh_factor = 1.0
        try:
            if bool(cfg.neurogenesis.early_stop.get("use_threshold_goal", False)):
                thresh_factor = float(cfg.neurogenesis.early_stop.get("threshold_goal_factor", 1.0))
        except AttributeError:
            pass
        for i, thr in enumerate(thresholds):
            logger.log_metrics({f"threshold/level_{i}": float(thr)}, step=0)
            logger.log_metrics({f"threshold/effective_level_{i}": float(thr * thresh_factor)}, step=0)
        _log_threshold_audit(model, dm, cfg, thresholds, device, logger)
        _log_growth_wiring_probe(model, dm, cfg, device, logger)

    replay = None
    if use_replay:
        if replay_mode == "dataset":
            replay = DatasetReplay(
                input_dim=cfg.model.input_dim,
                max_per_class=cfg.replay.get("dataset_max_samples_per_class", None),
                storage_device=cfg.replay.get("dataset_storage_device", "cpu"),
            )
        elif replay_mode == "intrinsic":
            replay = IntrinsicReplay(
                model.encoder,
                model.decoder,
                eps=cfg.replay.cov_eps,
                device=device,
                sampling_mode=cfg.replay.get("ir_sampling_mode", "gaussian_full"),
                cov_shrinkage=cfg.replay.get("ir_cov_shrinkage", 0.0),
                noise_scale=cfg.replay.get("ir_noise_scale", 1.0),
                filter_mode=cfg.replay.get("ir_filter_mode", "none"),
                filter_percentile=cfg.replay.get("ir_filter_percentile", 0.95),
                filter_max_resample=cfg.replay.get("ir_filter_max_resample", 10),
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
        _log_ir_quality_diagnostics(
            replay,
            model,
            dm,
            cfg.experiment.base_classes,
            view_shape=view_shape,
            cfg=cfg,
            device=device,
            logger=logger,
            step=len(cfg.experiment.base_classes),
        )

    if use_neurogenesis:
        trainer = NeurogenesisTrainer(
            ae=model,
            ir=replay,
            thresholds=thresholds,
            max_nodes=list(cfg.neurogenesis.max_nodes),
            max_outliers=cfg.neurogenesis.get(
                "max_outliers", None
            )
            if cfg.neurogenesis.get("max_outliers", None) is not None
            else cfg.neurogenesis.max_outlier_fraction,
            max_outliers_by_level=cfg.neurogenesis.get("max_outliers_by_level", {}),
            max_outlier_fraction_by_level=cfg.neurogenesis.get(
                "max_outlier_fraction_by_level", {}
            ),
            base_lr=cfg.training.base_lr,
            plasticity_epochs=cfg.neurogenesis.plasticity_epochs,
            stability_epochs=cfg.neurogenesis.stability_epochs,
            next_layer_epochs=cfg.neurogenesis.next_layer_epochs,
            factor_max_new_nodes=cfg.neurogenesis.factor_max_new_nodes,
            factor_new_nodes=cfg.neurogenesis.factor_new_nodes,
            logger=logger,
            early_stop_cfg=_neurogenesis_early_stop_cfg(cfg),
            objective_mode=cfg.neurogenesis.get("objective_mode", "paper_local"),
            plasticity_decoder_lr_ratio=cfg.neurogenesis.get(
                "plasticity_decoder_lr_ratio", 0.01
            ),
            stability_lr_ratio=cfg.neurogenesis.get("stability_lr_ratio", 0.01),
            next_layer_lr_ratio=cfg.neurogenesis.get("next_layer_lr_ratio", 0.01),
            next_layer_optimization=cfg.neurogenesis.get(
                "next_layer_optimization", "broad"
            ),
            replay_old_limit=cfg.neurogenesis.get("replay_old_limit", None),
            stability_replay_mode=cfg.neurogenesis.get("stability_replay_mode", "ratio"),
            stability_replay_ratio=cfg.neurogenesis.get("stability_replay_ratio", 1.0),
            stability_replay_ratio_base=cfg.neurogenesis.get("stability_replay_ratio_base", 1.0),
            stability_replay_ratio_max=cfg.neurogenesis.get("stability_replay_ratio_max", 4.0),
            stability_replay_balanced_max_ratio=cfg.neurogenesis.get(
                "stability_replay_balanced_max_ratio", 4.0
            ),
            stability_replay_per_class_ratio=cfg.neurogenesis.get(
                "stability_replay_per_class_ratio", 1.0
            ),
            stability_replay_class_weights=cfg.neurogenesis.get(
                "stability_replay_class_weights", {}
            ),
            stability_replay_loss_weight=cfg.neurogenesis.get(
                "stability_replay_loss_weight", 1.0
            ),
            stability_schedule=cfg.neurogenesis.get("stability_schedule", "mixed"),
            stability_current_epochs_ratio=cfg.neurogenesis.get(
                "stability_current_epochs_ratio", 1.0
            ),
            stability_replay_epochs_ratio=cfg.neurogenesis.get(
                "stability_replay_epochs_ratio", 1.0
            ),
            growth_mode=cfg.neurogenesis.get("growth_mode", "proportional"),
            growth_mode_by_level=cfg.neurogenesis.get("growth_mode_by_level", {}),
            absolute_new_nodes=cfg.neurogenesis.get("absolute_new_nodes", 1),
            absolute_new_nodes_by_level=cfg.neurogenesis.get(
                "absolute_new_nodes_by_level", {}
            ),
            factor_new_nodes_by_level=cfg.neurogenesis.get(
                "factor_new_nodes_by_level", {}
            ),
            factor_max_new_nodes_by_level=cfg.neurogenesis.get(
                "factor_max_new_nodes_by_level", {}
            ),
            shape_pressure_mode=cfg.neurogenesis.get("shape_pressure_mode", "none"),
            shape_target_ratio=cfg.neurogenesis.get("shape_target_ratio", 1.0),
            shape_target_ratio_by_level=cfg.neurogenesis.get(
                "shape_target_ratio_by_level", {}
            ),
            shape_min_growth_scale=cfg.neurogenesis.get("shape_min_growth_scale", 0.25),
            shape_growth_scale_power=cfg.neurogenesis.get(
                "shape_growth_scale_power", 1.0
            ),
            shape_gate_power=cfg.neurogenesis.get("shape_gate_power", 1.0),
            shape_max_gate_multiplier=cfg.neurogenesis.get(
                "shape_max_gate_multiplier", 10.0
            ),
            global_coupling_cfg=cfg.neurogenesis.get("global_coupling", {}),
            outlier_criterion_diagnostics=cfg.neurogenesis.get(
                "outlier_criterion_diagnostics", {}
            ),
            quality_growth_gate=cfg.neurogenesis.get(
                "quality_growth_gate", {}
            ),
            adaptive_outlier_threshold=cfg.neurogenesis.get(
                "adaptive_outlier_threshold", {}
            ),
        )
    else:
        trainer = IncrementalTrainer(
            ae=model,
            ir=replay,
            base_lr=cfg.training.base_lr,
            epochs=cfg.training.incremental_epochs,
            weight_decay=cfg.training.weight_decay,
            replay_ratio=cfg.replay.get("batch_ratio", 1.0),
            replay_mode=cfg.replay.get("sampling_mode", "ratio"),
            replay_per_class_ratio=cfg.replay.get("per_class_batch_ratio", 1.0),
            device=device,
            logger=logger,
        )

    learned: list[int] = list(cfg.experiment.base_classes)
    configured_incremental_classes = list(cfg.experiment.incremental_classes)
    incremental_classes = (
        []
        if bool(cfg.experiment.get("skip_incremental_training", False))
        else configured_incremental_classes
    )
    all_classes: list[int] = list(cfg.experiment.base_classes) + configured_incremental_classes
    trainer.log_global_sizes()
    eval_records: list[dict] = []
    eval_records.extend(_evaluate(trainer, dm, learned, cfg, step=len(learned), logger=logger))
    base_stage_label = f"Step {len(learned)} (base)"
    _log_reconstruction_artifacts(
        model,
        dm,
        all_classes,
        len(learned),
        device,
        logger,
        step=len(learned),
        timeline_records=recon_timeline_records,
        stage_label=base_stage_label,
    )
    _log_intrinsic_replay_examples(
        replay,
        classes=learned,
        view_shape=view_shape,
        logger=logger,
        step=len(learned),
        samples_per_class=ir_samples_per_class,
        timeline_records=ir_timeline_records,
        stage_label=base_stage_label,
    )

    train_limit_per_class = cfg.experiment.get("incremental_train_limit_per_class", None)
    for stage, class_id in enumerate(incremental_classes, start=1):
        t_learn0 = time.perf_counter()
        loader = _train_loader_for_classes(
            dm,
            [class_id],
            shuffle=True,
            limit_per_class=train_limit_per_class,
        )
        if isinstance(trainer, NeurogenesisTrainer):
            eval_sample = _make_fixed_visual_batch(dm, learned, device)
            if eval_sample is None:
                eval_sample = _maybe_make_visual_batch(dm, learned, device)
            if eval_sample is not None:
                eval_images, _, _, _ = _unpack_visual_batch(eval_sample)
                trainer.set_recon_eval_batch(eval_images)
        trainer.learn_class(class_id, loader)
        logger.log_metrics(
            {"timing/learn_class_sec": time.perf_counter() - t_learn0}, step=len(learned) + 1
        )
        if train_limit_per_class is not None:
            _log_tiny_overfit_summary(
                trainer,
                dm,
                int(class_id),
                cfg,
                logger,
                step=len(learned) + 1,
                limit=int(train_limit_per_class),
            )
        # Recompute replay stats. By default we refresh every learned class with
        # the updated encoder; reuse_previous_stats keeps old class replay fixed
        # and only stores the newly learned class distribution.
        learned_so_far = learned + [class_id]
        trainer.log_global_sizes()
        if replay is not None:
            t_boot_inc0 = time.perf_counter()
            reuse_previous_stats = bool(cfg.replay.get("reuse_previous_stats", False))
            replay_update_classes = [class_id] if reuse_previous_stats else learned_so_far
            _bootstrap_replay(
                replay,
                dm,
                replay_update_classes,
                batch_size=min(cfg.replay.stats_batch_size, cfg.data.batch_size),
                num_workers=cfg.data.num_workers,
                reset_stats=not reuse_previous_stats,
            )
            logger.log_metrics(
                {"timing/replay_update_sec": time.perf_counter() - t_boot_inc0},
                step=len(learned_so_far),
            )
            logger.log_metrics(
                {
                    "replay/reuse_previous_stats": float(reuse_previous_stats),
                    "replay/refit_class_count": float(len(replay_update_classes)),
                },
                step=len(learned_so_far),
            )
        learned.append(class_id)
        _log_replay_stats(replay, logger, step=len(learned))
        _log_ir_quality_diagnostics(
            replay,
            model,
            dm,
            learned,
            view_shape=view_shape,
            cfg=cfg,
            device=device,
            logger=logger,
            step=len(learned),
        )
        eval_records.extend(_evaluate(trainer, dm, learned, cfg, step=len(learned), logger=logger))
        t_fig0 = time.perf_counter()
        _log_reconstruction_artifacts(
            model,
            dm,
            all_classes,
            len(learned),
            device,
            logger,
            step=len(learned),
            timeline_records=recon_timeline_records,
            stage_label=f"Step {len(learned)} (added {class_id})",
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
    _log_recon_timeline(recon_timeline_records, logger)
    _log_combined_timeline(
        recon_timeline_records,
        ir_timeline_records,
        view_shape=view_shape,
        logger=logger,
    )
    logger.log_metrics({"timing/total_run_sec": time.perf_counter() - t_run0})
    logger.finish()
    return {"model": model, "trainer": trainer, "replay": replay, "training_stats": training_stats}


@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
