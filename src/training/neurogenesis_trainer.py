import math
from typing import Any, Callable, List, Optional

import torch
from pytorch_lightning.loggers import MLFlowLogger
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from models.ng_autoencoder import NGAutoEncoder
from utils.intrinsic_replay import IntrinsicReplay


class NeurogenesisTrainer:
    """
    Orchestrates sequential class learning with neurogenesis.
    Stores per-class, per-layer reconstruction-error history for analysis.
    """

    def __init__(
        self,
        ae: NGAutoEncoder,
        ir: IntrinsicReplay,
        thresholds: List[float],
        max_nodes: List[int],
        max_outliers: float,
        base_lr: float = 1e-3,
        plasticity_epochs: int = 5,
        stability_epochs: int = 2,
        next_layer_epochs: int = 1,
        factor_max_new_nodes: float = 0.1,
        factor_new_nodes: float = 0.1,
        logger: MLFlowLogger = None,
        mean_layer_losses=None,
        early_stop_cfg: dict = None,
        replay_old_limit: int | None = None,
        stability_replay_mode: str = "ratio",
        stability_replay_ratio: float = 1.0,
        stability_replay_ratio_base: float = 1.0,
        stability_replay_ratio_max: float = 4.0,
        stability_replay_balanced_max_ratio: float = 4.0,
    ):
        self.ae = ae
        self.ir = ir
        self.thresholds = thresholds
        self.max_nodes = max_nodes
        self.max_outliers = max_outliers
        self.base_lr = base_lr
        self.logger = logger
        self.factor_max_new_nodes = factor_max_new_nodes
        self.factor_new_nodes = factor_new_nodes
        self.mean_layer_losses = mean_layer_losses
        self.early_stop_cfg = early_stop_cfg
        self.replay_old_limit = None if replay_old_limit is None else int(replay_old_limit)
        self.stability_replay_mode = str(stability_replay_mode or "ratio").lower()
        self.stability_replay_ratio = float(stability_replay_ratio)
        self.stability_replay_ratio_base = float(stability_replay_ratio_base)
        self.stability_replay_ratio_max = float(stability_replay_ratio_max)
        self.stability_replay_balanced_max_ratio = float(stability_replay_balanced_max_ratio)
        self._phase_loss_history: dict[tuple[Any, int, int, str], list[float]] = {}
        self._recon_eval_batch: torch.Tensor | None = None

        # counter for how many classes we've learned so far
        self._class_count = 0

        # epoch settings per phase
        self.plasticity_epochs = plasticity_epochs
        self.stability_epochs = stability_epochs
        self.next_layer_epochs = next_layer_epochs

        # History: class_id -> {'layer_errors': List[List[Tensor]]}
        self.history: dict[Any, dict[str, List[List[Tensor]]]] = {}

    def _log_outlier_metrics(
        self,
        *,
        class_id: Any,
        level: int,
        iteration: int,
        n_outliers: int,
        total_seen: int,
    ) -> None:
        if not self.logger:
            return
        fraction = n_outliers / max(total_seen, 1)
        metric_prefix = f"class_{class_id}"
        metrics = {
            f"{metric_prefix}/level_{level}_n_outliers_round": n_outliers,
            f"{metric_prefix}/level_{level}_outlier_fraction_round": fraction,
        }
        self.logger.log_metrics(metrics, step=iteration)

    def _build_replay_sampler(
        self, device: torch.device, *, n_old_classes: int
    ) -> tuple[Callable[[int], Optional[torch.Tensor]] | None, bool]:
        if self.ir is None or not self.ir.available_classes():
            return None, False
        remaining = self.replay_old_limit
        mode = self.stability_replay_mode
        replay_only = mode in {"only", "only_balanced"}
        if mode == "ratio_schedule":
            ratio = min(
                self.stability_replay_ratio_base * max(n_old_classes, 1),
                self.stability_replay_ratio_max,
            )
        elif mode == "balanced":
            ratio = min(max(n_old_classes, 1), self.stability_replay_balanced_max_ratio)
        else:
            ratio = self.stability_replay_ratio

        ratio = max(float(ratio), 0.0)
        replay_classes = [int(cls) for cls in self.ir.available_classes()]
        class_cursor = 0

        def _sample(batch_size: int) -> Optional[torch.Tensor]:
            sync_fn = getattr(self.ir, "sync_encoder_latent_dim", None)
            if callable(sync_fn):
                try:
                    sync_fn()
                except Exception:
                    pass
            nonlocal remaining, class_cursor
            take = int(math.ceil(batch_size * ratio))
            if take <= 0:
                return None
            if remaining is not None:
                if remaining <= 0:
                    return None
                take = min(take, remaining)
                remaining -= take
            if mode == "only_balanced" and replay_classes:
                n_classes = len(replay_classes)
                counts = [0] * n_classes
                if take >= n_classes:
                    base = take // n_classes
                    rem = take % n_classes
                    for i in range(n_classes):
                        counts[i] = base
                    for i in range(rem):
                        idx = (class_cursor + i) % n_classes
                        counts[idx] += 1
                else:
                    for i in range(take):
                        idx = (class_cursor + i) % n_classes
                        counts[idx] += 1

                chunks: list[torch.Tensor] = []
                for idx, count in enumerate(counts):
                    if count <= 0:
                        continue
                    cls = replay_classes[idx]
                    chunk = self.ir.sample_images(cls, count)
                    chunks.append(chunk)

                if not chunks:
                    return None
                replay_flat = torch.cat(chunks, dim=0)
                if replay_flat.size(0) > 1:
                    order = torch.randperm(replay_flat.size(0), device=replay_flat.device)
                    replay_flat = replay_flat.index_select(0, order)
                class_cursor = (class_cursor + take) % max(n_classes, 1)
            else:
                replay_flat = self.ir.sample_images(None, take)
            return replay_flat.to(device, non_blocking=True)

        return _sample, replay_only

    def _build_phase_early_stop_cfg(
        self, level: Optional[int] = None, *, phase: Optional[str] = None
    ) -> Optional[dict]:
        """Return a per-phase early-stop config with optional threshold goal."""
        if not self.early_stop_cfg:
            return None
        cfg = dict(self.early_stop_cfg)
        use_goal = cfg.pop("use_threshold_goal", False)
        factor = cfg.pop("threshold_goal_factor", 1.0)
        factor_plasticity = cfg.pop("threshold_goal_factor_plasticity", None)
        factor_stability = cfg.pop("threshold_goal_factor_stability", None)
        if phase == "plasticity" and factor_plasticity is not None:
            factor = float(factor_plasticity)
        elif phase == "stability" and factor_stability is not None:
            factor = float(factor_stability)
        if use_goal and self.thresholds:
            idx = len(self.thresholds) - 1 if level is None else max(level, 0)
            idx = min(idx, len(self.thresholds) - 1)
            cfg["goal"] = self.thresholds[idx] * factor
        return cfg

    def _model_device(self) -> torch.device:
        params = getattr(self.ae, "parameters", None)
        if params is None:
            return torch.device("cpu")
        first = next(params(), None)
        if first is None:
            return torch.device("cpu")
        return first.device

    def _get_recon_errors(self, loader: DataLoader, level: int) -> Tensor:
        """
        Compute reconstruction errors at specified encoder level for all samples in loader.
        Returns a CPU tensor to avoid building gigantic autograd graphs on the GPU.
        """
        device = self._model_device()
        errors: list[Tensor] = []
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(device, non_blocking=True)
                x_hat = self.ae.forward_partial(x, level)
                err = self.ae.reconstruction_error(x_hat, x).detach().cpu()
                errors.append(err)
        return torch.cat(errors) if errors else torch.empty(0)

    def _get_outliers(self, loader: DataLoader, level: int):
        device = self._model_device()

        # 1) Collect all per-sample errors into a single flat Tensor
        errors = []
        for batch in loader:
            x, _ = batch
            x = x.to(device, non_blocking=True)
            err = self.ae.reconstruction_error(
                self.ae.forward_partial(x, level), x
            )  # shape: [batch_size]
            errors.append(err.detach().cpu())
        errors = torch.cat(errors)  # [N]
        # print(f"[DEBUG] Errors[:10] = {errors[:10].tolist()}  mean={errors.mean().item():.4f}")

        # 2) Compare to threshold
        thr = self.thresholds[level]
        mask = errors > thr
        # print(f"[DEBUG] threshold = {thr:.4f}, errors>thr mask[:10] = {mask[:10].tolist()}")
        n_outliers = int(mask.sum().item())
        # print(f"[DEBUG] n_outliers = {n_outliers} / {len(errors)}")

        # 3) Find the *real* dataset indices you iterated over
        if hasattr(loader, "sampler") and hasattr(loader.sampler, "idxs"):
            all_indices = loader.sampler.idxs
        else:
            # if loader.dataset is a Subset, its .indices attr points into the full dataset
            all_indices = getattr(loader.dataset, "indices", list(range(len(loader.dataset))))
        # print(f"[DEBUG] first 10 all_indices = {all_indices[:10]}")

        # 4) Map mask→real indices
        outlier_real_idxs = [all_indices[i] for i, m in enumerate(mask) if m]
        # print(f"[DEBUG] first 10 outlier_real_idxs = {outlier_real_idxs[:10]}")

        # 5) Build subset & return loader
        # If the original loader.dataset is already a Subset, its `.indices`
        # refer to the underlying base dataset. We must create the new Subset
        # from that base dataset, not from the Subset itself, otherwise the
        # indices would be interpreted relative to the Subset and can go
        # out-of-range. For non-Subset datasets, use the dataset directly.
        base_ds = loader.dataset.dataset if isinstance(loader.dataset, Subset) else loader.dataset
        subset = Subset(base_ds, outlier_real_idxs)
        outlier_loader = DataLoader(
            subset,
            batch_size=loader.batch_size,
            shuffle=False,
            num_workers=loader.num_workers,
            pin_memory=loader.pin_memory,
        )
        total = int(errors.numel())
        return n_outliers, outlier_loader, total

    def _limit_loader(self, loader: DataLoader, n_samples: int) -> DataLoader:
        """
        Returns a DataLoader that yields only the first n_samples from the original loader.
        """
        total = min(n_samples, len(loader.dataset))
        limited_idxs = list(range(total))
        limited_subset = Subset(loader.dataset, limited_idxs)
        return DataLoader(
            limited_subset,
            batch_size=loader.batch_size,
            shuffle=False,
            num_workers=loader.num_workers,
            pin_memory=loader.pin_memory,
        )

    def log_global_sizes(self):
        if self.logger:
            global_metrics = {
                f"global_level_{i}_size": sz for i, sz in enumerate(self.ae.hidden_sizes)
            }
            self.logger.log_metrics(global_metrics, step=self._class_count)

    def _build_epoch_logger(self, class_id: Any, level: int, round_idx: int):
        if not self.logger:
            return None

        metric_prefix = f"class_{class_id}"
        artifact_prefix = f"plasticity/class_{class_id}/level_{level}/round_{round_idx}"
        log_dict_fn = getattr(self.logger, "log_dict", None)
        log_metrics_fn = getattr(self.logger, "log_metrics", None)

        def _callback(epoch_idx: int, summary: dict):
            phase = summary.get("phase", "train")
            if callable(log_dict_fn):
                payload = {
                    "class_id": class_id,
                    "level": level,
                    "round": round_idx,
                    "phase": phase,
                    "epoch": epoch_idx + 1,
                    **summary,
                }
                log_dict_fn(payload, f"{artifact_prefix}/epoch_{epoch_idx + 1}.json")
            if "loss" in summary:
                key = (class_id, level, round_idx, str(phase))
                self._phase_loss_history.setdefault(key, []).append(float(summary["loss"]))
            if callable(log_metrics_fn) and "loss" in summary:
                loss = float(summary["loss"])
                metrics = {
                    f"{phase}/class_{class_id}/level_{level}/round_{round_idx}_loss": loss
                }
                log_metrics_fn(metrics, step=epoch_idx + 1)
            if callable(log_metrics_fn) and "eval_loss" in summary:
                eval_loss = float(summary["eval_loss"])
                metrics = {
                    f"{phase}/class_{class_id}/level_{level}/round_{round_idx}_eval_loss": eval_loss
                }
                log_metrics_fn(metrics, step=epoch_idx + 1)

        return _callback

    def _log_phase_loss_plot(self, class_id: Any, level: int, round_idx: int) -> None:
        if not self.logger:
            return
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return
        plast_key = (class_id, level, round_idx, "plasticity")
        stab_key = (class_id, level, round_idx, "stability")
        plast = self._phase_loss_history.get(plast_key, [])
        stab = self._phase_loss_history.get(stab_key, [])
        if not plast and not stab:
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        if plast:
            ax.plot(range(1, len(plast) + 1), plast, label="plasticity")
        if stab:
            ax.plot(range(1, len(stab) + 1), stab, label="stability")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Class {class_id} Level {level} Round {round_idx}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        artifact = f"figures/phase_losses/class_{class_id}/level_{level}/round_{round_idx}.png"
        try:
            self.logger.log_figure(fig, artifact)
        finally:
            plt.close(fig)

    def set_recon_eval_batch(self, batch: torch.Tensor | None) -> None:
        self._recon_eval_batch = batch

    def learn_class(self, class_id: Any, loader: DataLoader) -> None:
        num_layers = len(self.ae.hidden_sizes)
        sizes_before = list(self.ae.hidden_sizes)
        self.history[class_id] = {"layer_errors": [[] for _ in range(num_layers)]}
        outlier_history: dict[int, list[dict]] = {lvl: [] for lvl in range(num_layers)}

        # ---- log "class learned" ----
        self._class_count += 1
        if self.logger:
            self.logger.log_metrics({"classes_learned": self._class_count}, step=self._class_count)

        device = self._model_device()
        n_old_classes = len(self.ir.available_classes()) if self.ir is not None else 0
        replay_sampler, replay_only = self._build_replay_sampler(
            device, n_old_classes=n_old_classes
        )

        pbar_levels = tqdm(range(num_layers), desc=f"[Class {class_id}] Layers", unit="lvl")
        for level in pbar_levels:
            added = 0
            self.ae._plastic_to_mature()
            n_plastic_neurons = 0
            n_outliers, outliers_loader, total_seen = self._get_outliers(loader, level)
            fraction = n_outliers / max(total_seen, 1)
            iteration_idx = 0
            self._log_outlier_metrics(
                class_id=class_id,
                level=level,
                iteration=iteration_idx,
                n_outliers=n_outliers,
                total_seen=total_seen,
            )
            outlier_history[level].append(
                {
                    "iteration": iteration_idx,
                    "n_outliers": n_outliers,
                    "total_seen": total_seen,
                    "fraction": fraction,
                }
            )

            if self.logger:
                self.logger.log_metrics(
                    {f"class_{class_id}/growth_level_{level}": self.ae.hidden_sizes[level]},
                    step=added,
                )
            max_rounds = self.max_nodes[level]
            pbar_growth = tqdm(
                range(max_rounds), desc=f"  Level {level} Growth", unit="rnd", leave=False
            )

            step_plasticety = 0
            step_stability = 0

            for _ in pbar_growth:
                if not (fraction > self.max_outliers and added < max_rounds):
                    break

                num_new = int(math.ceil(self.factor_new_nodes * n_outliers))
                nodes_existing = self.ae.encoder[2 * level].n_out_features
                max_new = int(math.ceil(self.factor_max_new_nodes * nodes_existing))
                num_new = int(min([num_new, max_new]))

                self.ae.add_new_nodes(level, num_new)
                n_plastic_neurons += num_new
                if self.logger:
                    self.logger.log_metrics(
                        {f"class_{class_id}_level_{level}_n_plastic_neurons": n_plastic_neurons},
                        step=added,
                    )
                    current_sizes = {
                        f"class_{class_id}_level_{lvl}_cumulative_size": sz
                        for lvl, sz in enumerate(self.ae.hidden_sizes)
                    }
                    self.logger.log_metrics(current_sizes, step=added)
                if self.logger:
                    self.logger.log_metrics(
                        {f"class_{class_id}/growth_level_{level}": self.ae.hidden_sizes[level]},
                        step=added,
                    )

                last_loss = 1
                round_idx = added + 1
                epoch_logger = self._build_epoch_logger(class_id, level, round_idx)
                phase_es_cfg = self._build_phase_early_stop_cfg(level, phase="plasticity")

                hist = self.ae.plasticity_phase(
                    loader=outliers_loader,
                    level=level,
                    epochs=self.plasticity_epochs,
                    lr=self.base_lr,
                    early_stop_cfg=phase_es_cfg,
                    forward_fn=lambda x: self.ae.forward_partial(x, level),
                    epoch_logger=epoch_logger,
                )

                mean_loss = hist["epoch_loss"][-1]
                delta_loss = last_loss - mean_loss if last_loss is not None else float("inf")

                if self.logger:
                    self.logger.log_metrics(
                        {
                            f"class_{class_id}_level_{level}_loss_plasticity": mean_loss,
                            f"class_{class_id}_level_{level}_delta_loss_plasticity": delta_loss,
                        },
                        step_plasticety,
                    )
                    self.logger.log_metrics(
                        {
                            f"class_{class_id}_level_{level}_loss_plasticity_iter": mean_loss,
                            f"class_{class_id}_level_{level}_delta_loss_plasticity_iter": delta_loss,
                        },
                        step=self._class_count,
                    )
                step_plasticety += 1
                last_loss = mean_loss

                last_loss = 1
                phase_es_cfg = self._build_phase_early_stop_cfg(level, phase="stability")
                hist = self.ae.stability_phase(
                    loader=outliers_loader,
                    # Keep stability updates aligned with the currently growing level.
                    level=level,
                    lr=self.base_lr,
                    epochs=self.stability_epochs,
                    old_x=replay_sampler,
                    replay_only=replay_only,
                    eval_batch=self._recon_eval_batch,
                    early_stop_on_eval=self._recon_eval_batch is not None,
                    early_stop_cfg=phase_es_cfg,
                    epoch_logger=epoch_logger,
                )

                self._log_phase_loss_plot(class_id, level, round_idx)

                mean_loss = hist["epoch_loss"][-1]
                delta_loss = last_loss - mean_loss if last_loss is not None else float("inf")

                if self.logger:
                    self.logger.log_metrics(
                        {
                            f"class_{class_id}_level_{level}_loss_stability": mean_loss,
                            f"class_{class_id}_level_{level}_delta_loss_stability": delta_loss,
                        },
                        step_stability,
                    )
                    self.logger.log_metrics(
                        {
                            f"class_{class_id}_level_{level}_loss_stability_iter": mean_loss,
                            f"class_{class_id}_level_{level}_delta_loss_stability_iter": delta_loss,
                        },
                        step=self._class_count,
                    )
                step_stability += 1
                last_loss = mean_loss

                errs = self._get_recon_errors(loader, level)
                self.history[class_id]["layer_errors"][level].append(errs.clone())
                if self.logger:
                    self.logger.log_metrics(
                        {f"class_{class_id}_level_{level}_avg_loss": errs.mean().item()},
                        step=added,
                    )
                    self.logger.log_metrics(
                        {f"class_{class_id}_level_{level}_avg_loss_iter": errs.mean().item()},
                        step=self._class_count,
                    )

                added += 1
                n_outliers, outliers_loader, total_seen = self._get_outliers(loader, level)
                iteration_idx += 1
                self._log_outlier_metrics(
                    class_id=class_id,
                    level=level,
                    iteration=iteration_idx,
                    n_outliers=n_outliers,
                    total_seen=total_seen,
                )
                fraction = n_outliers / max(total_seen, 1)
                outlier_history[level].append(
                    {
                        "iteration": iteration_idx,
                        "n_outliers": n_outliers,
                        "total_seen": total_seen,
                        "fraction": fraction,
                    }
                )
                if self.logger:
                    self.logger.log_metrics(
                        {
                            f"class_{class_id}_level_{level}_n_outliers": n_outliers,
                            f"class_{class_id}_level_{level}_outlier_fraction": fraction,
                        },
                        step=added,
                    )

            pbar_growth.close()

            # Next-layer plasticity & stability
            if level + 1 < num_layers and step_plasticety > 0:
                phase_es_cfg = self._build_phase_early_stop_cfg(level + 1, phase="plasticity")
                self.ae.plasticity_phase(
                    loader,
                    level + 1,
                    epochs=self.next_layer_epochs,
                    lr=self.base_lr / 100,
                    early_stop_cfg=phase_es_cfg,
                )
                phase_es_cfg = self._build_phase_early_stop_cfg(level + 1, phase="stability")
                self.ae.stability_phase(
                    loader,
                    level + 1,
                    lr=self.base_lr / 100,
                    epochs=self.stability_epochs,
                    old_x=replay_sampler,
                    replay_only=replay_only,
                    eval_batch=self._recon_eval_batch,
                    early_stop_on_eval=self._recon_eval_batch is not None,
                    early_stop_cfg=phase_es_cfg,
                )

            if self.logger:
                size = self.ae.hidden_sizes[level]
                self.logger.log_metrics(
                    {f"class_{class_id}_level_{level}_size": size}, step=self._class_count
                )

            pbar_levels.update(1)
        pbar_levels.close()
        # Fit IR stats for the incoming class

        if self.logger and outlier_history:
            lines = [f"Outlier progression for class {class_id}"]
            for level_idx in range(num_layers):
                entries = outlier_history.get(level_idx) or []
                if not entries:
                    continue
                lines.append(f"Level {level_idx}:")
                for entry in entries:
                    iteration = entry["iteration"]
                    n_out = entry["n_outliers"]
                    total = entry["total_seen"]
                    frac = entry["fraction"]
                    lines.append(
                        f"  iter {iteration:02d}: {n_out} / {total} ({frac:.4f})"
                    )
            if len(lines) == 1:
                lines.append("No outlier measurements recorded.")
            text = "\n".join(lines)
            self.logger.log_text(text, f"neurogenesis/class_{class_id}_outliers.txt")

        loader = tqdm(
            loader,
            desc=f"[Class {class_id}] Fitting IR",
            unit="batch",
            leave=False,
        )
        if self.ir is not None:
            self.ir.fit(loader)
        self.log_global_sizes()

        # Refresh replay statistics with the now-updated encoder
        if self.ir is not None:
            self.ir.fit(loader)

        if self.logger and sizes_before:
            summary_metrics = {}
            for idx, (before, after) in enumerate(zip(sizes_before, self.ae.hidden_sizes)):
                summary_metrics[f"summary/layer_{idx}_growth_total"] = after - before
                summary_metrics[f"summary/layer_{idx}_cumulative_size"] = after
            self.logger.log_metrics(summary_metrics, step=self._class_count)

    def test_all_levels(self, loader: DataLoader) -> List[float]:
        """
        Evaluate the autoencoder at each encoder depth and return the mean reconstruction loss per level.
        """
        mean_losses: List[float] = []
        max_losses: List[float] = []
        std_losses: List[float] = []

        # Ensure model is in evaluation mode
        self.ae.eval()
        with torch.no_grad():
            for level in range(len(self.ae.hidden_sizes)):
                # Compute reconstruction errors at this level
                errors = self._get_recon_errors(loader, level)
                # Mean loss for the level
                mean_loss = errors.mean().item()
                max_loss = errors.max().item()
                std_loss = errors.std().item()
                mean_losses.append(mean_loss)
                max_losses.append(max_loss)
                std_losses.append(std_loss)

                # Log metric if logger is available
                if self.logger:
                    self.logger.log_metrics({"test_mean_loss": mean_loss}, level)
                    self.logger.log_metrics({"test_max_loss": max_loss}, level)
                    self.logger.log_metrics({"test_std_loss": std_loss}, level)

        # Restore training mode
        self.ae.train()
        return mean_losses, max_losses, std_losses
