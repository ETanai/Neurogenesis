"""Utilities for estimating reconstruction-error thresholds per encoder layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch
from torch.utils.data import DataLoader

from models.ng_autoencoder import NGAutoEncoder


@dataclass
class ThresholdEstimationConfig:
    """Configuration for :class:`ThresholdEstimator`."""

    percentile: float = 0.95
    margin: float = 0.0
    minimum: float = 1e-6


class ThresholdEstimator:
    """Compute per-layer reconstruction-error thresholds for neurogenesis."""

    def __init__(
        self,
        model: NGAutoEncoder,
        *,
        config: ThresholdEstimationConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.config = config or ThresholdEstimationConfig()
        if not 0.0 < self.config.percentile <= 1.0:
            raise ValueError("percentile must be within (0, 1]")

        self.device = device or next(model.parameters()).device

    def estimate(self, loader: DataLoader) -> List[float]:
        """Return thresholds (one per encoder level) based on provided data."""

        thresholds: list[float] = []
        self.model.eval()
        quantile = torch.tensor(self.config.percentile, device=self.device)

        with torch.no_grad():
            for level in range(len(self.model.hidden_sizes)):
                errs: list[torch.Tensor] = []
                for batch in loader:
                    x = batch[0].to(self.device, non_blocking=True)
                    recon = self.model.forward_partial(x, level)
                    err = self.model.reconstruction_error(recon, x)
                    errs.append(err.detach().cpu())

                if errs:
                    all_errs = torch.cat(errs)
                    thr = torch.quantile(all_errs, quantile.item())
                    value = float(thr.item() + self.config.margin)
                    thresholds.append(max(value, self.config.minimum))
                else:
                    thresholds.append(float("inf"))

        self.model.train()
        return thresholds


def compute_thresholds(
    model: NGAutoEncoder,
    loaders: Iterable[DataLoader],
    *,
    config: ThresholdEstimationConfig | None = None,
) -> List[float]:
    """Convenience helper to estimate thresholds from multiple loaders."""

    estimator = ThresholdEstimator(model, config=config)
    agg: list[list[torch.Tensor]] = [[] for _ in model.hidden_sizes]

    for loader in loaders:
        for idx, value in enumerate(estimator.estimate(loader)):
            agg[idx].append(torch.tensor([value]))

    thresholds: list[float] = []
    for tensors in agg:
        if tensors:
            thresholds.append(float(torch.cat(tensors).mean().item()))
        else:
            thresholds.append(float("inf"))
    return thresholds

