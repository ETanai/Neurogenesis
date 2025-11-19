"""Command-line entry point for running Neurogenesis experiments headlessly."""

from __future__ import annotations

import csv
import inspect
import tempfile
import time
from pathlib import Path
from typing import List, Sequence

import hydra
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
    def __init__(self, *, tracking_uri: str, experiment_name: str, run_name: str) -> None:
        if mlflow is None:
            raise RuntimeError("mlflow is not installed; disable MLflow logging or install mlflow.")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._run = mlflow.start_run(run_name=run_name)

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        for key, value in metrics.items():
            mlflow.log_metric(key, float(value), step=step)

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
        try:
            mlflow.log_image(image_bytes, artifact_file)
            return
        except TypeError:
            pass

        parent = Path(artifact_file).parent.as_posix()
        fname = Path(artifact_file).name
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / fname
            with open(out_path, "wb") as fh:
                fh.write(image_bytes)
            mlflow.log_artifact(str(out_path), artifact_path=(parent if parent != "." else None))

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


def _instantiate_datamodule(cfg: DictConfig) -> MNISTDataModule:
    name = cfg.name
    if name == "mnist":
        dm = MNISTDataModule(
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            data_dir=cfg.data_dir,
        )
    elif name == "sd19":
        from data.sd19_datamodule import SD19DataModule  # lazy import

        dm = SD19DataModule(
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            data_dir=cfg.data_dir,
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
    if hasattr(dm, "make_class_balanced_batch"):
        try:
            batch = dm.make_class_balanced_batch(
                classes,
                samples_per_class=min(4, max(1, 16 // max(1, len(classes)))),
                split=split,
                device=device,
                shuffle=False,
                return_labels=True,
            )
            if isinstance(batch, tuple) and len(batch) >= 2:
                return batch
        except Exception:
            pass

    subset = _get_combined_dataset(dm, classes, split=split)
    loader = _dataloader(
        subset,
        batch_size=min(16, len(subset) if hasattr(subset, "__len__") else 16),
        num_workers=0,
        shuffle=False,
    )
    try:
        batch = next(iter(loader))
    except StopIteration:
        return None
    images, labels = batch
    return images.to(device), labels.to(device), []


def _log_reconstruction_artifacts(
    model, dm, classes: Sequence[int], device: torch.device, logger, step: int
) -> None:
    if isinstance(logger, _NullLogger) or plot_recon_grid_mlflow is None:
        return
    sample = _maybe_make_visual_batch(dm, classes, device)
    if sample is None:
        return
    images, labels, splits = sample
    view_shape = images.shape[1:]

    fig, png_bytes = plot_recon_grid_mlflow(
        model,
        images.detach().cpu(),
        view_shape=view_shape,
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
            fig_partial, png_partial = plot_partial_recon_grid_mlflow(
                model,
                images.detach().cpu(),
                view_shape=view_shape,
                levels=levels,
                col_group_titles=[str(int(l)) for l in labels.detach().cpu().tolist()]
                if len(labels.shape)
                else None,
                col_group_splits=splits if splits else None,
                return_mlflow_artifact=True,
            )
            logger.log_image(png_partial, f"figures/partial_reconstructions_step_{step}.png")
            logger.log_figure(fig_partial, f"figures/partial_reconstructions_step_{step}.mpl.png")
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
        payload = {
            f"replay/class_{cls}_count": stats["count"],
            f"replay/class_{cls}_latent_var_mean": stats["latent_var_mean"],
            f"replay/class_{cls}_cov_condition": stats["cov_condition"],
        }
        logger.log_metrics(payload, step=step)
    weights = replay.get_class_weights()
    if weights:
        logger.log_dict(weights, "replay/class_weights.json")


def _build_model(cfg: DictConfig, device: torch.device) -> NGAutoEncoder:
    model = NGAutoEncoder(
        input_dim=cfg.model.input_dim,
        hidden_sizes=list(cfg.model.hidden_sizes),
        activation=cfg.model.activation,
        activation_latent=cfg.model.activation_latent,
        activation_last=cfg.model.activation_last,
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
    replay: IntrinsicReplay,
    dm: MNISTDataModule,
    classes: Sequence[int],
    *,
    batch_size: int,
    num_workers: int,
) -> None:
    t_total = time.perf_counter()
    for cls in classes:
        t_cls = time.perf_counter()
        subset = _get_class_dataset(dm, cls, split="train")
        loader = _dataloader(subset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        loader = tqdm(
            loader,
            desc=f"Bootstrapping replay for class {cls}",
            leave=True,
        )
        replay.fit(loader)
        try:
            # best-effort logging if replay has a logger via outer scope
            pass
        except Exception:
            pass
    # aggregate timing is logged by caller where logger is available


def _pretrain(
    model: NGAutoEncoder, dm: MNISTDataModule, cfg: DictConfig, device: torch.device, logger
) -> None:
    base_classes = cfg.experiment.base_classes
    train_subset = _get_combined_dataset(dm, base_classes, split="train")
    train_loader = _dataloader(
        train_subset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=True,
    )

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
        leave=True,
    )

    if val_loader is not None:
        val_loader = tqdm(
            val_loader,
            desc="Validation",
            leave=True,
        )

    trainer.fit(train_loader, val_loader=val_loader, log_fn=logger.log_metrics)
    logger.log_metrics({"timing/pretrain_total_sec": time.perf_counter() - t0})


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

    dm = _instantiate_datamodule(cfg.data)
    mlflow_cfg = cfg.logging.mlflow
    if mlflow_cfg.enabled and mlflow is None:
        print("[WARN] mlflow not available; disabling MLflow logging for this run.")
    if mlflow_cfg.enabled and mlflow is not None:
        logger = _MLflowLogger(
            tracking_uri=mlflow_cfg.tracking_uri,
            experiment_name=mlflow_cfg.experiment_name,
            run_name=mlflow_cfg.run_name,
        )
        logger.log_params(
            {
                "dataset": cfg.experiment.dataset,
                "base_classes": list(cfg.experiment.base_classes),
                "incremental_classes": list(cfg.experiment.incremental_classes),
                "hidden_sizes": list(cfg.model.hidden_sizes),
            }
        )
        _log_config_snapshot(cfg, logger)
    else:
        logger = _NullLogger()

    model = _build_model(cfg, device)

    t_pre0 = time.perf_counter()
    _pretrain(model, dm, cfg, device, logger)
    logger.log_metrics({"timing/pretrain_sec": time.perf_counter() - t_pre0})

    use_neurogenesis, use_replay = _resolve_regime(cfg)

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

    replay = None
    if use_replay:
        replay = IntrinsicReplay(
            model.encoder, model.decoder, eps=cfg.replay.cov_eps, device=device
        )
        t_boot0 = time.perf_counter()
        _bootstrap_replay(
            replay,
            dm,
            cfg.experiment.base_classes,
            batch_size=min(cfg.replay.stats_batch_size, cfg.data.batch_size),
            num_workers=cfg.data.num_workers,
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
    trainer.log_global_sizes()
    eval_records: list[dict] = []
    eval_records.extend(_evaluate(trainer, dm, learned, cfg, step=len(learned), logger=logger))
    _log_reconstruction_artifacts(model, dm, learned, device, logger, step=len(learned))

    for stage, class_id in enumerate(cfg.experiment.incremental_classes, start=1):
        subset = _get_class_dataset(dm, class_id, split="train")
        t_learn0 = time.perf_counter()
        loader = _dataloader(
            subset,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            shuffle=True,
        )
        trainer.learn_class(class_id, loader)
        logger.log_metrics(
            {"timing/learn_class_sec": time.perf_counter() - t_learn0}, step=len(learned) + 1
        )
        learned.append(class_id)
        trainer.log_global_sizes()
        if replay is not None:
            t_boot_inc0 = time.perf_counter()
            _bootstrap_replay(
                replay,
                dm,
                [class_id],
                batch_size=min(cfg.replay.stats_batch_size, cfg.data.batch_size),
                num_workers=cfg.data.num_workers,
            )
            logger.log_metrics(
                {"timing/replay_update_sec": time.perf_counter() - t_boot_inc0}, step=len(learned)
            )
        _log_replay_stats(replay, logger, step=len(learned))
        eval_records.extend(_evaluate(trainer, dm, learned, cfg, step=len(learned), logger=logger))
        t_fig0 = time.perf_counter()
        _log_reconstruction_artifacts(model, dm, learned, device, logger, step=len(learned))
        logger.log_metrics(
            {"timing/recon_artifacts_sec": time.perf_counter() - t_fig0}, step=len(learned)
        )

    _log_eval_records(eval_records, logger)
    logger.log_metrics({"timing/total_run_sec": time.perf_counter() - t_run0})
    logger.finish()
    return {"model": model, "trainer": trainer, "replay": replay}


@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
