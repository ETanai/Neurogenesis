"""Utility to pretrain the neurogenesis autoencoder on base classes."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.ng_autoencoder import NGAutoEncoder


@dataclass
class PretrainingConfig:
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "auto"
    mode: str = "joint"
    stacked_level_epochs: Optional[int] = None
    denoising_enabled: bool = False
    denoising_noise_type: str = "gaussian"
    denoising_noise_std: float = 0.2
    denoising_mask_prob: float = 0.2


class AutoencoderPretrainer:
    """Simple SGD loop that trains the autoencoder on clean reconstructions."""

    def __init__(self, model: NGAutoEncoder, config: PretrainingConfig) -> None:
        self.model = model
        self.config = config
        device = config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.update_steps = 0

    def _corrupt_inputs(self, x: torch.Tensor) -> torch.Tensor:
        if not bool(self.config.denoising_enabled):
            return x
        noise_type = str(self.config.denoising_noise_type).lower()
        if noise_type == "gaussian":
            std = max(float(self.config.denoising_noise_std), 0.0)
            if std == 0.0:
                return x
            return torch.clamp(x + torch.randn_like(x) * std, 0.0, 1.0)
        if noise_type in {"mask", "dropout"}:
            p = min(max(float(self.config.denoising_mask_prob), 0.0), 1.0)
            if p == 0.0:
                return x
            keep = (torch.rand_like(x) > p).to(x.dtype)
            return x * keep
        raise ValueError(f"Unknown denoising noise type '{self.config.denoising_noise_type}'.")

    def fit(
        self,
        train_loader: DataLoader,
        *,
        val_loader: Optional[DataLoader] = None,
        log_fn: Optional[callable] = None,
        epoch_hook: Optional[callable] = None,
    ) -> Dict[str, list[float]]:
        if self.config.mode == "stacked":
            return self._fit_stacked(
                train_loader,
                val_loader=val_loader,
                log_fn=log_fn,
                epoch_hook=epoch_hook,
            )

        if self.config.mode != "joint":
            raise ValueError(f"Unknown pretraining mode '{self.config.mode}'.")

        history: Dict[str, list[float]] = {"train_loss": []}
        if val_loader is not None:
            history["val_loss"] = []

        for epoch in range(self.config.epochs):
            t0 = time.perf_counter()
            train_loss = self._run_epoch(train_loader, training=True)
            history["train_loss"].append(train_loss)

            if log_fn is not None:
                log_fn({"pretrain/train_loss": train_loss}, epoch)
                log_fn({"timing/pretrain_epoch_sec": time.perf_counter() - t0}, epoch)

            if val_loader is not None:
                t1 = time.perf_counter()
                val_loss = self._run_epoch(val_loader, training=False)
                history["val_loss"].append(val_loss)
                if log_fn is not None:
                    log_fn({"pretrain/val_loss": val_loss}, epoch)
                    log_fn({"timing/val_epoch_sec": time.perf_counter() - t1}, epoch)

            if epoch_hook is not None:
                try:
                    epoch_hook(epoch)
                except Exception:
                    pass

        return history

    def _run_epoch(self, loader: DataLoader, *, training: bool) -> float:
        if training:
            self.model.train()
        else:
            self.model.eval()

        # Accumulate losses on device to avoid per-batch CPU sync (.item()).
        loss_sum = torch.zeros((), device=self.device)
        n_batches = 0
        with torch.set_grad_enabled(training):
            for batch in loader:
                x = batch[0].to(self.device, non_blocking=True)
                x_in = self._corrupt_inputs(x) if training else x
                out = self.model(x_in)
                recon = out["recon"] if isinstance(out, dict) else out
                loss = F.mse_loss(recon, x.view(x.size(0), -1))

                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizer.step()
                    self.update_steps += 1

                # accumulate without synchronizing to CPU each step
                loss_sum = loss_sum + loss.detach()
                n_batches += 1

        if n_batches == 0:
            return 0.0
        return float((loss_sum / n_batches).item())

    def _fit_stacked(
        self,
        train_loader: DataLoader,
        *,
        val_loader: Optional[DataLoader] = None,
        log_fn: Optional[callable] = None,
        epoch_hook: Optional[callable] = None,
    ) -> Dict[str, list[float]]:
        n_levels = len(self.model.hidden_sizes)
        epochs_per_level = (
            int(self.config.stacked_level_epochs)
            if self.config.stacked_level_epochs is not None
            else int(self.config.epochs)
        )
        if epochs_per_level <= 0:
            raise ValueError("stacked_level_epochs must be >= 1")

        history: Dict[str, list[float]] = {"train_loss": []}
        if val_loader is not None:
            history["val_loss"] = []

        global_epoch = 0
        for level in range(n_levels):
            self._set_stacked_trainable(level)
            params = [p for p in self.model.parameters() if p.requires_grad]
            if not params:
                raise RuntimeError(f"No trainable parameters found for stacked level {level}")
            optimizer = torch.optim.Adam(
                params,
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            )

            for epoch in range(epochs_per_level):
                t0 = time.perf_counter()
                train_loss = self._run_epoch_partial(
                    train_loader,
                    optimizer,
                    level=level,
                    training=True,
                )
                history["train_loss"].append(train_loss)
                if log_fn is not None:
                    log_fn(
                        {
                            f"pretrain/stacked/level_{level}/train_loss": train_loss,
                            "pretrain/train_loss": train_loss,
                        },
                        global_epoch,
                    )
                    log_fn(
                        {"timing/pretrain_epoch_sec": time.perf_counter() - t0},
                        global_epoch,
                    )

                if val_loader is not None:
                    t1 = time.perf_counter()
                    val_loss = self._run_epoch_partial(
                        val_loader,
                        optimizer,
                        level=level,
                        training=False,
                    )
                    history["val_loss"].append(val_loss)
                    if log_fn is not None:
                        log_fn(
                            {
                                f"pretrain/stacked/level_{level}/val_loss": val_loss,
                                "pretrain/val_loss": val_loss,
                            },
                            global_epoch,
                        )
                        log_fn({"timing/val_epoch_sec": time.perf_counter() - t1}, global_epoch)

                if epoch_hook is not None:
                    try:
                        epoch_hook(global_epoch)
                    except Exception:
                        pass
                global_epoch += 1

        # Restore standard behavior for downstream training.
        for p in self.model.parameters():
            p.requires_grad_(True)

        return history

    def _set_stacked_trainable(self, level: int) -> None:
        for p in self.model.parameters():
            p.requires_grad_(False)

        enc = self.model._encoder_layer(level)
        dec = self.model._decoder_layer(level)
        for p in enc.parameters():
            p.requires_grad_(True)
        for p in dec.parameters():
            p.requires_grad_(True)

    def _run_epoch_partial(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        *,
        level: int,
        training: bool,
    ) -> float:
        if training:
            self.model.train()
        else:
            self.model.eval()

        loss_sum = torch.zeros((), device=self.device)
        n_batches = 0
        with torch.set_grad_enabled(training):
            for batch in loader:
                x = batch[0].to(self.device, non_blocking=True)
                x_in = self._corrupt_inputs(x) if training else x
                recon = self.model.forward_partial(x_in, layer_idx=level)
                loss = F.mse_loss(recon, x.view(x.size(0), -1))

                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    self.update_steps += 1

                loss_sum = loss_sum + loss.detach()
                n_batches += 1

        if n_batches == 0:
            return 0.0
        return float((loss_sum / n_batches).item())
