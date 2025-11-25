"""Incremental training loop without neurogenesis expansion."""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from models.ng_autoencoder import NGAutoEncoder
from utils.intrinsic_replay import IntrinsicReplay


class IncrementalTrainer:
    """Train the autoencoder sequentially without adding new neurons."""

    def __init__(
        self,
        *,
        ae: NGAutoEncoder,
        ir: Optional[IntrinsicReplay],
        base_lr: float,
        epochs: int,
        weight_decay: float = 0.0,
        replay_ratio: float = 1.0,
        device: Optional[torch.device] = None,
        logger: Optional[object] = None,
    ) -> None:
        self.ae = ae
        self.ir = ir
        self.base_lr = float(base_lr)
        self.epochs = int(max(1, epochs))
        self.weight_decay = float(weight_decay)
        self.replay_ratio = max(0.0, float(replay_ratio))
        self.logger = logger

        if device is None:
            try:
                device = next(ae.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        self.device = device

        self.history: Dict[int, Dict[str, List[float]]] = {}
        self._class_count = 0
        self.update_steps = 0

    # ------------------------------------------------------------------
    def _model_device(self) -> torch.device:
        return self.device

    def _ensure_optimizer(self):
        return torch.optim.Adam(
            self.ae.parameters(), lr=self.base_lr, weight_decay=self.weight_decay
        )

    def _step_loss(self, batch: Tensor) -> Tensor:
        out = self.ae(batch)
        recon = out["recon"] if isinstance(out, dict) else out
        return F.mse_loss(recon, batch.view(batch.size(0), -1))

    def _augment_with_replay(self, inputs: Tensor) -> Tensor:
        if self.ir is None or not self.ir.available_classes():
            return inputs
        if self.replay_ratio <= 0:
            return inputs

        n_new = inputs.size(0)
        n_replay = int(math.ceil(self.replay_ratio * n_new))
        if n_replay <= 0:
            return inputs

        replay_flat = self.ir.sample_images(None, n_replay)
        replay_tensor = replay_flat.view(n_replay, *inputs.shape[1:]).to(inputs.device)
        return torch.cat([inputs, replay_tensor], dim=0)

    # ------------------------------------------------------------------
    def learn_class(self, class_id: int, loader: DataLoader) -> None:
        self._class_count += 1
        self.ae.to(self.device)
        self.ae.train()
        self.ae.set_requires_grad(freeze_old=False)

        optimizer = self._ensure_optimizer()

        if self.ir is not None:
            self.ir.fit(loader)

        history: Dict[str, List[float]] = {"train_loss": []}

        for epoch in range(self.epochs):
            losses: List[float] = []
            for batch in loader:
                images, _ = batch
                images = images.to(self.device, non_blocking=True)
                mixed = self._augment_with_replay(images)

                loss = self._step_loss(mixed)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                self.update_steps += 1

                losses.append(float(loss.detach().cpu().item()))

            mean_loss = float(torch.tensor(losses).mean().item()) if losses else 0.0
            history["train_loss"].append(mean_loss)
            if self.logger is not None:
                self.logger.log_metrics(
                    {"incremental/train_loss": mean_loss},
                    step=self._class_count * self.epochs + epoch,
                )

        self.history[class_id] = history
        if self.ir is not None:
            self.ir.fit(loader)

    # ------------------------------------------------------------------
    def _get_recon_errors(self, loader: DataLoader, level: int) -> Tensor:
        errors: List[Tensor] = []
        for batch in loader:
            images = batch[0].to(self.device, non_blocking=True)
            recon = self.ae.forward_partial(images, level)
            errors.append(self.ae.reconstruction_error(recon, images))
        return torch.cat(errors) if errors else torch.empty(0, device=self.device)

    def test_all_levels(self, loader: DataLoader):
        mean_losses: List[float] = []
        max_losses: List[float] = []
        std_losses: List[float] = []

        self.ae.eval()
        with torch.no_grad():
            for level in range(len(self.ae.hidden_sizes)):
                errors = self._get_recon_errors(loader, level)
                if errors.numel() == 0:
                    mean_losses.append(0.0)
                    max_losses.append(0.0)
                    std_losses.append(0.0)
                    continue
                mean_losses.append(errors.mean().item())
                max_losses.append(errors.max().item())
                std_losses.append(errors.std().item())
        self.ae.train()
        return mean_losses, max_losses, std_losses

    def log_global_sizes(self):
        if self.logger is None:
            return
        payload = {f"global_level_{idx}_size": size for idx, size in enumerate(self.ae.hidden_sizes)}
        self.logger.log_metrics(payload, step=self._class_count)


__all__ = ["IncrementalTrainer"]
