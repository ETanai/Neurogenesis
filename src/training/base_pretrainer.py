"""Utility to pretrain the neurogenesis autoencoder on base classes."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from models.ng_autoencoder import NGAutoEncoder


@dataclass
class PretrainingConfig:
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "auto"
    mode: str = "end_to_end"
    denoising_dropout: float = 0.0
    denoising_std: float = 0.0
    finetune_epochs: int = 0


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

    def fit(
        self,
        train_loader: DataLoader,
        *,
        val_loader: Optional[DataLoader] = None,
        log_fn: Optional[callable] = None,
        epoch_hook: Optional[callable] = None,
    ) -> Dict[str, list[float]]:
        if self.config.mode in {"stacked", "stacked_denoising"}:
            return self._fit_stacked(
                train_loader,
                val_loader=val_loader,
                log_fn=log_fn,
                epoch_hook=epoch_hook,
            )
        if self.config.mode != "end_to_end":
            raise ValueError(
                f"Unknown pretraining mode '{self.config.mode}'. "
                "Expected 'end_to_end', 'stacked', or 'stacked_denoising'."
            )
        return self._fit_end_to_end(
            train_loader,
            val_loader=val_loader,
            log_fn=log_fn,
            epoch_hook=epoch_hook,
        )

    def _fit_end_to_end(
        self,
        train_loader: DataLoader,
        *,
        val_loader: Optional[DataLoader] = None,
        log_fn: Optional[callable] = None,
        epoch_hook: Optional[callable] = None,
    ) -> Dict[str, list[float]]:
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

    def _fit_stacked(
        self,
        train_loader: DataLoader,
        *,
        val_loader: Optional[DataLoader] = None,
        log_fn: Optional[callable] = None,
        epoch_hook: Optional[callable] = None,
    ) -> Dict[str, list[float]]:
        history: Dict[str, list[float]] = {"train_loss": []}
        denoise = self.config.mode == "stacked_denoising"
        global_epoch = 0

        for level in range(len(self.model.hidden_sizes)):
            optimizer = self._make_level_optimizer(level)
            for epoch in range(self.config.epochs):
                t0 = time.perf_counter()
                loss = self._run_stacked_level_epoch(
                    train_loader,
                    level=level,
                    optimizer=optimizer,
                    training=True,
                    denoise=denoise,
                )
                history["train_loss"].append(loss)
                if log_fn is not None:
                    log_fn({f"pretrain/level_{level}_train_loss": loss}, global_epoch)
                    log_fn({"timing/pretrain_epoch_sec": time.perf_counter() - t0}, global_epoch)

                if val_loader is not None:
                    val_loss = self._run_stacked_level_epoch(
                        val_loader,
                        level=level,
                        optimizer=optimizer,
                        training=False,
                        denoise=False,
                    )
                    history.setdefault("val_loss", []).append(val_loss)
                    if log_fn is not None:
                        log_fn({f"pretrain/level_{level}_val_loss": val_loss}, global_epoch)

                if epoch_hook is not None:
                    try:
                        epoch_hook(global_epoch)
                    except Exception:
                        pass
                global_epoch += 1

        self._restore_trainable_parameters()

        if self.config.finetune_epochs > 0:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            )
            for epoch in range(self.config.finetune_epochs):
                t0 = time.perf_counter()
                loss = self._run_epoch(train_loader, training=True)
                history["train_loss"].append(loss)
                if log_fn is not None:
                    log_fn({"pretrain/finetune_train_loss": loss}, global_epoch)
                    log_fn({"timing/pretrain_epoch_sec": time.perf_counter() - t0}, global_epoch)
                if val_loader is not None:
                    val_loss = self._run_epoch(val_loader, training=False)
                    history.setdefault("val_loss", []).append(val_loss)
                    if log_fn is not None:
                        log_fn({"pretrain/finetune_val_loss": val_loss}, global_epoch)
                if epoch_hook is not None:
                    try:
                        epoch_hook(global_epoch)
                    except Exception:
                        pass
                global_epoch += 1

        self._restore_trainable_parameters()
        return history

    def _make_level_optimizer(self, level: int) -> torch.optim.Optimizer:
        for param in self.model.parameters():
            param.requires_grad_(False)
        params = []
        for module in (self.model._encoder_layer(level), self.model._decoder_layer(level)):
            for param in module.parameters():
                param.requires_grad_(True)
                params.append(param)
        return torch.optim.Adam(params, lr=self.config.lr, weight_decay=self.config.weight_decay)

    def _restore_trainable_parameters(self) -> None:
        for param in self.model.parameters():
            param.requires_grad_(True)

    def _encode_level_input(self, x: Tensor, level: int) -> Tensor:
        out = x.view(x.size(0), -1)
        with torch.no_grad():
            for module in self.model.encoder[: 2 * level]:
                out = module(out)
        return out.detach()

    def _reconstruct_level(self, z: Tensor, level: int) -> Tensor:
        out = z
        for module in self.model.encoder[2 * level : 2 * level + 2]:
            out = module(out)
        start = 2 * (len(self.model.hidden_sizes) - 1 - level)
        for module in self.model.decoder[start : start + 2]:
            out = module(out)
        return out

    def _corrupt(self, x: Tensor) -> Tensor:
        out = x
        if self.config.denoising_dropout > 0:
            keep = torch.rand_like(out) >= float(self.config.denoising_dropout)
            out = out * keep.to(out.dtype)
        if self.config.denoising_std > 0:
            out = out + torch.randn_like(out) * float(self.config.denoising_std)
        return out

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
                out = self.model(x)
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

    def _run_stacked_level_epoch(
        self,
        loader: DataLoader,
        *,
        level: int,
        optimizer: torch.optim.Optimizer,
        training: bool,
        denoise: bool,
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
                target = self._encode_level_input(x, level)
                inputs = self._corrupt(target) if denoise and training else target
                recon = self._reconstruct_level(inputs, level)
                loss = F.mse_loss(recon, target)

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
