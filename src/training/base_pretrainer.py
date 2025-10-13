"""Utility to pretrain the neurogenesis autoencoder on base classes."""

from __future__ import annotations

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

    def fit(
        self,
        train_loader: DataLoader,
        *,
        val_loader: Optional[DataLoader] = None,
        log_fn: Optional[callable] = None,
    ) -> Dict[str, list[float]]:
        history: Dict[str, list[float]] = {"train_loss": []}
        if val_loader is not None:
            history["val_loss"] = []

        for epoch in range(self.config.epochs):
            train_loss = self._run_epoch(train_loader, training=True)
            history["train_loss"].append(train_loss)

            if log_fn is not None:
                log_fn({"pretrain/train_loss": train_loss}, epoch)

            if val_loader is not None:
                val_loss = self._run_epoch(val_loader, training=False)
                history["val_loss"].append(val_loss)
                if log_fn is not None:
                    log_fn({"pretrain/val_loss": val_loss}, epoch)

        return history

    def _run_epoch(self, loader: DataLoader, *, training: bool) -> float:
        if training:
            self.model.train()
        else:
            self.model.eval()

        losses: list[float] = []
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

                losses.append(float(loss.detach().cpu().item()))

        return float(torch.tensor(losses).mean().item()) if losses else 0.0

