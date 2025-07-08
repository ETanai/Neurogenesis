import math
from typing import Any, List

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
        logger: MLFlowLogger = None,
    ):
        self.ae = ae
        self.ir = ir
        self.thresholds = thresholds
        self.max_nodes = max_nodes
        self.max_outliers = max_outliers
        self.base_lr = base_lr
        self.logger = logger

        # counter for how many classes we've learned so far
        self._class_count = 0

        # epoch settings per phase
        self.plasticity_epochs = plasticity_epochs
        self.stability_epochs = stability_epochs
        self.next_layer_epochs = next_layer_epochs

        # History: class_id -> {'layer_errors': List[List[Tensor]]}
        self.history: dict[Any, dict[str, List[List[Tensor]]]] = {}

    def _get_recon_errors(self, loader: DataLoader, level: int) -> Tensor:
        """
        Compute reconstruction errors at specified encoder level for all samples in loader.
        """
        errors = []
        for batch in loader:
            x = batch[0]
            x_hat = self.ae.forward_partial(x, level)
            errors.append(self.ae.reconstruction_error(x_hat, x))
        return torch.cat(errors)

    def _get_outliers(self, loader: DataLoader, level: int):
        """
        Identify outlier samples whose reconstruction error exceeds threshold at given level.
        Returns:
          - n_outliers: int
          - outlier_loader: DataLoader with only outlier samples
        """
        errors = []
        indices = []
        all_data = []
        for batch in loader:
            x, y = batch[0], batch[1] if len(batch) > 1 else None
            err = self.ae.reconstruction_error(self.ae.forward_partial(x, level), x)
            batch_size = x.size(0)
            for i in range(batch_size):
                errors.append(err[i].item())
                all_data.append((x[i], y[i] if y is not None else None))
                indices.append(len(indices))
        errors = torch.tensor(errors)
        mask = errors > self.thresholds[level]
        n_outliers = int(mask.sum().item())

        # gather outlier samples
        outlier_indices = [i for i, m in enumerate(mask) if m]
        # assuming uniform dataset mapping, use Subset
        subset = Subset(loader.dataset, outlier_indices)
        outlier_loader = DataLoader(
            subset,
            batch_size=loader.batch_size,
            shuffle=False,
            num_workers=loader.num_workers,
            pin_memory=loader.pin_memory,
        )
        return n_outliers, outlier_loader

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

    def learn_class(self, class_id: Any, loader: DataLoader) -> None:
        num_layers = len(self.ae.hidden_sizes)
        self.history[class_id] = {"layer_errors": [[] for _ in range(num_layers)]}

        # ---- log "class learned" ----
        self._class_count += 1
        if self.logger:
            self.logger.log_metrics({"classes_learned": self._class_count}, step=self._class_count)

        # Fit IR stats
        self.ir.fit(loader)
        device = next(self.ae.parameters()).device
        old_x = None
        if self.ir is not None and class_id is not None and len(loader) > 0:
            old_x = self.ir.sample_images(class_id, len(loader)).to(device)

        pbar_levels = tqdm(range(num_layers), desc=f"[Class {class_id}] Layers", unit="lvl")
        for level in pbar_levels:
            n_outliers, outliers_loader = self._get_outliers(loader, level)
            if self.logger:
                self.logger.log_metrics(
                    {f"class_{class_id}_level_{level}_n_outliers": n_outliers},
                    step=self._class_count,
                )

            added = 0
            max_rounds = self.max_nodes[level]
            pbar_growth = tqdm(
                range(max_rounds), desc=f"  Level {level} Growth", unit="rnd", leave=False
            )
            for i_growth in pbar_growth:
                if not (n_outliers > self.max_outliers * n_outliers and added < max_rounds):
                    break

                num_new = int(math.ceil(0.1 * n_outliers))
                self.ae.add_new_nodes(level, num_new)
                # Plasticity phase (epochs)

                for i in tqdm(
                    range(self.plasticity_epochs), desc="   Plasticity", unit="ep", leave=False
                ):
                    losses_plasticety = []
                    losses_stability = []
                    for batch in outliers_loader:
                        x = batch[0].to(device, non_blocking=True)
                        opt = self.ae._optim_plasticity(level, self.base_lr)
                        opt.zero_grad(set_to_none=True)
                        x_hat = self.ae.forward_partial(x, level + 1)
                        if isinstance(x_hat, dict):
                            x_hat = x_hat["recon"]
                        loss = self.ae.reconstruction_error(x_hat, x).mean()
                        losses_plasticety.append(loss.item())
                        loss.backward()
                        opt.step()

                    if self.logger:
                        self.logger.log_metrics(
                            {
                                f"class_{class_id}_level_{level}_loss_plasticety": sum(
                                    losses_plasticety
                                )
                                / len(losses_plasticety)
                            },
                            step=i_growth * i * added + i,
                        )

                # Stability phase (epochs)
                for _ in tqdm(
                    range(self.stability_epochs), desc="   Stability", unit="ep", leave=False
                ):
                    for batch in outliers_loader:
                        x = batch[0].to(device, non_blocking=True)
                        if old_x is not None:
                            k = x.size(0)
                            idx = torch.randint(0, old_x.size(0), (k,), device=device)
                            x = torch.cat([x, old_x[idx].view([k, 1, 28, 28])], dim=0)
                        opt = self.ae._optim_stability(
                            int(len(self.ae.encoder) / 2) - 1, self.base_lr
                        )
                        opt.zero_grad(set_to_none=True)
                        x_hat = self.ae(x)
                        if isinstance(x_hat, dict):
                            x_hat = x_hat["recon"]
                        loss = self.ae.reconstruction_error(x_hat, x).mean()
                        losses_stability.append(loss)
                        loss.backward()
                        opt.step()

                    if self.logger:
                        self.logger.log_metrics(
                            {
                                f"class_{class_id}_level_{level}_loss_plasticety": sum(
                                    losses_stability
                                )
                                / len(losses_stability)
                            },
                            step=i_growth * i * added + i,
                        )

                # recompute errors & log
                errs = self._get_recon_errors(loader, level)
                self.history[class_id]["layer_errors"][level].append(errs.clone())
                if self.logger:
                    self.logger.log_metrics(
                        {
                            f"class_{class_id}_level_{level}_avg_loss_step_{added}": errs.mean().item()
                        },
                        step=self._class_count,
                    )

                added += 1
                n_outliers, _ = self._get_outliers(loader, level)
                if self.logger:
                    self.logger.log_metrics(
                        {f"class_{class_id}_level_{level}_n_outliers": n_outliers},
                        step=self._class_count,
                    )

                pbar_growth.update(1)

            pbar_growth.close()

            # Next-layer plasticity & stability
            if level + 1 < num_layers:
                self.ae.plasticity_phase(
                    loader, level + 1, epochs=self.next_layer_epochs, lr=self.base_lr / 100
                )
                self.ae.stability_phase(
                    loader, lr=self.base_lr / 100, epochs=self.stability_epochs, old_x=old_x
                )

            if self.logger:
                size = self.ae.hidden_sizes[level]
                self.logger.log_metrics(
                    {f"class_{class_id}_level_{level}_size": size}, step=self._class_count
                )

            pbar_levels.update(1)
        pbar_levels.close()

        if self.logger:
            global_metrics = {
                f"global_level_{i}_size": sz for i, sz in enumerate(self.ae.hidden_sizes)
            }
            self.logger.log_metrics(global_metrics, step=self._class_count)

    def test_all_levels(self, loader: DataLoader) -> List[float]:
        """
        Evaluate the autoencoder at each encoder depth and return the mean reconstruction loss per level.
        """
        mean_losses: List[float] = []

        # Ensure model is in evaluation mode
        self.ae.eval()
        with torch.no_grad():
            for level in range(len(self.ae.hidden_sizes)):
                # Compute reconstruction errors at this level
                errors = self._get_recon_errors(loader, level)
                # Mean loss for the level
                mean_loss = errors.mean().item()
                mean_losses.append(mean_loss)

                # Log metric if logger is available
                if self.logger:
                    self.logger.log_metrics({f"test_level_{level}_mean_loss": mean_loss})

        # Restore training mode
        self.ae.train()
        return mean_losses
