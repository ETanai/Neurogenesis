"""Fixed-capacity incremental training with predictive-coding inference.

This is a research extension, not part of the paper replication.  The model's
feed-forward encoder/decoder graph supplies one local predictor per affine
layer.  Hidden activities are relaxed against adjacent prediction errors, then
each predictor is updated from its own settled local error with all activity
targets detached.  Consequently, the weight update has no end-to-end gradient
path through other layers.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from models.ng_autoencoder import NGAutoEncoder
from training.incremental_trainer import IncrementalTrainer


class PredictiveCodingTrainer(IncrementalTrainer):
    """Train a fixed autoencoder by activation inference and local errors."""

    def __init__(
        self,
        *,
        ae: NGAutoEncoder,
        inference_steps: int = 5,
        inference_lr: float = 0.1,
        plasticity_mode: str = "uniform",
        usage_decay: float = 0.99,
        usage_exponent: float = 0.5,
        plasticity_min: float = 0.25,
        plasticity_max: float = 4.0,
        layer_precisions: list[float] | None = None,
        global_loss_weight: float = 0.0,
        update_mode: str = "local",
        consolidation_epochs: int = 0,
        consolidation_lr_ratio: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(ae=ae, **kwargs)
        if inference_steps < 1:
            raise ValueError("inference_steps must be positive")
        if inference_lr <= 0:
            raise ValueError("inference_lr must be positive")
        plasticity_mode = str(plasticity_mode).lower()
        if plasticity_mode not in {"uniform", "usage"}:
            raise ValueError("plasticity_mode must be 'uniform' or 'usage'")
        if not 0 <= usage_decay < 1:
            raise ValueError("usage_decay must be in [0, 1)")
        if not 0 < plasticity_min <= plasticity_max:
            raise ValueError("plasticity bounds must satisfy 0 < min <= max")
        update_mode = str(update_mode).lower()
        if update_mode not in {"local", "backprop_equivalent"}:
            raise ValueError("update_mode must be 'local' or 'backprop_equivalent'")
        if global_loss_weight < 0:
            raise ValueError("global_loss_weight must be non-negative")
        if consolidation_epochs < 0 or consolidation_lr_ratio <= 0:
            raise ValueError("consolidation settings require epochs >= 0 and lr_ratio > 0")

        self.inference_steps = int(inference_steps)
        self.inference_lr = float(inference_lr)
        self.plasticity_mode = plasticity_mode
        self.usage_decay = float(usage_decay)
        self.usage_exponent = float(usage_exponent)
        self.plasticity_min = float(plasticity_min)
        self.plasticity_max = float(plasticity_max)
        pair_count = len(self._predictor_pairs())
        self.layer_precisions = (
            [1.0] * pair_count
            if layer_precisions is None
            else [float(value) for value in layer_precisions]
        )
        if len(self.layer_precisions) != pair_count or any(value <= 0 for value in self.layer_precisions):
            raise ValueError(
                f"layer_precisions must contain {pair_count} positive values"
            )
        self.global_loss_weight = float(global_loss_weight)
        self.update_mode = update_mode
        self.consolidation_epochs = int(consolidation_epochs)
        self.consolidation_lr_ratio = float(consolidation_lr_ratio)
        self._usage: list[Tensor | None] = [None] * len(self._predictor_pairs())
        self.diagnostics: dict[str, object] = {
            "inference_steps": self.inference_steps,
            "inference_lr": self.inference_lr,
            "plasticity_mode": self.plasticity_mode,
            "layer_precisions": self.layer_precisions,
            "global_loss_weight": self.global_loss_weight,
            "update_mode": self.update_mode,
            "consolidation_epochs": self.consolidation_epochs,
            "consolidation_lr_ratio": self.consolidation_lr_ratio,
            "energy_before": [],
            "energy_after": [],
            "local_loss": [],
            "class_summaries": {},
        }

    def _predictor_pairs(self) -> list[tuple[nn.Module, nn.Module]]:
        modules = [*list(self.ae.encoder), *list(self.ae.decoder)]
        if len(modules) % 2:
            raise ValueError("Predictive coding requires affine/activation pairs")
        pairs = [(modules[index], modules[index + 1]) for index in range(0, len(modules), 2)]
        if any(not hasattr(linear, "weight_mature") for linear, _ in pairs):
            raise TypeError("Predictive coding currently requires NGLinear predictors")
        return pairs

    @staticmethod
    def _predict(pair: tuple[nn.Module, nn.Module], state: Tensor) -> Tensor:
        linear, activation = pair
        return activation(linear(state))

    def feedforward_states(self, batch: Tensor) -> list[Tensor]:
        state = batch.view(batch.size(0), -1)
        states = [state]
        for pair in self._predictor_pairs():
            state = self._predict(pair, state)
            states.append(state)
        return states

    def prediction_energy(self, states: list[Tensor], *, reduction: str = "sum") -> Tensor:
        """Return adjacent-layer prediction energy.

        Inference uses a sum so a state's step size does not shrink with batch
        or layer width. Weight learning separately uses mean local losses to
        retain learning-rate comparability with the MSE backprop baseline.
        """
        errors = [
            precision * (states[index + 1] - self._predict(pair, states[index])).square()
            for index, (pair, precision) in enumerate(
                zip(self._predictor_pairs(), self.layer_precisions)
            )
        ]
        if reduction == "sum":
            return 0.5 * sum(error.sum() for error in errors)
        if reduction == "mean":
            return 0.5 * sum(error.mean() for error in errors)
        raise ValueError("reduction must be 'sum' or 'mean'")

    def infer_states(self, batch: Tensor) -> tuple[list[Tensor], float, float]:
        """Relax hidden states while clamping input and reconstruction target."""
        target = batch.view(batch.size(0), -1).detach()
        with torch.no_grad():
            initial = self.feedforward_states(batch)
        variables = [state.detach().requires_grad_(True) for state in initial[1:-1]]
        states = [target, *variables, target]
        before = float(self.prediction_energy(states).detach().item() / batch.size(0))

        for _ in range(self.inference_steps):
            energy = self.prediction_energy(states)
            gradients = torch.autograd.grad(energy, variables, create_graph=False)
            variables = [
                (state - self.inference_lr * gradient).detach().requires_grad_(True)
                for state, gradient in zip(variables, gradients)
            ]
            states = [target, *variables, target]

        after = float(self.prediction_energy(states).detach().item() / batch.size(0))
        return [state.detach() for state in states], before, after

    def _update_usage(self, feedforward_states: list[Tensor]) -> None:
        if self.plasticity_mode != "usage":
            return
        for index, state in enumerate(feedforward_states[1:]):
            # The final state consists of input pixels, not adaptable hidden
            # neurons. Keep its optimizer step uniform.
            if index == len(self._usage) - 1:
                self._usage[index] = None
                continue
            observed = state.detach().abs().mean(dim=0)
            previous = self._usage[index]
            if previous is None:
                self._usage[index] = observed
            else:
                self._usage[index] = self.usage_decay * previous + (1 - self.usage_decay) * observed

    def initialize_usage(self, loader: DataLoader) -> None:
        """Seed utilization statistics from base classes without changing weights."""
        if self.plasticity_mode != "usage":
            return
        self.ae.eval()
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(self.device, non_blocking=True)
                self._update_usage(self.feedforward_states(images))
        self.ae.train()

    def plasticity_factors(self) -> list[Tensor]:
        factors = []
        for usage in self._usage:
            if usage is None or self.plasticity_mode == "uniform":
                factors.append(torch.ones(1, device=self.device))
                continue
            factor = (usage + 1.0e-6).pow(-self.usage_exponent)
            factor = factor / factor.mean().clamp_min(1.0e-12)
            factors.append(factor.clamp(self.plasticity_min, self.plasticity_max))
        return factors

    @staticmethod
    def _snapshots(pairs: Iterable[tuple[nn.Module, nn.Module]]) -> list[tuple[Tensor, Tensor]]:
        return [
            (linear.weight_mature.detach().clone(), linear.bias_mature.detach().clone())
            for linear, _ in pairs
        ]

    def _apply_usage_scaled_steps(
        self,
        pairs: list[tuple[nn.Module, nn.Module]],
        snapshots: list[tuple[Tensor, Tensor]],
    ) -> None:
        if self.plasticity_mode != "usage":
            return
        factors = self.plasticity_factors()
        with torch.no_grad():
            for (linear, _), (old_weight, old_bias), factor in zip(pairs, snapshots, factors):
                if factor.numel() == 1:
                    continue
                linear.weight_mature.copy_(
                    old_weight + (linear.weight_mature - old_weight) * factor[:, None]
                )
                linear.bias_mature.copy_(old_bias + (linear.bias_mature - old_bias) * factor)

    def train_batch(self, batch: Tensor, optimizer: torch.optim.Optimizer) -> dict[str, float]:
        pairs = self._predictor_pairs()
        with torch.no_grad():
            feedforward = self.feedforward_states(batch)
        self._update_usage(feedforward)
        if self.update_mode == "backprop_equivalent":
            target = batch.view(batch.size(0), -1)
            recon = self.ae(batch)["recon"]
            loss = F.mse_loss(recon, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            self.update_steps += 1
            value = float(loss.detach().item())
            return {"loss": value, "energy_before": value, "energy_after": value}
        states, before, after = self.infer_states(batch)

        local_losses = [
            precision * F.mse_loss(self._predict(pair, states[index]), states[index + 1])
            for index, (pair, precision) in enumerate(zip(pairs, self.layer_precisions))
        ]
        loss = sum(local_losses)
        if self.global_loss_weight > 0:
            target = batch.view(batch.size(0), -1)
            loss = loss + self.global_loss_weight * F.mse_loss(self.ae(batch)["recon"], target)
        snapshots = self._snapshots(pairs) if self.plasticity_mode == "usage" else []
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        self._apply_usage_scaled_steps(pairs, snapshots)
        self.update_steps += 1

        return {
            "loss": float(loss.detach().item()),
            "energy_before": before,
            "energy_after": after,
        }

    def learn_class(self, class_id: int, loader: DataLoader) -> None:
        self._class_count += 1
        self.ae.to(self.device)
        self.ae.train()
        self.ae.set_requires_grad(freeze_old=False)
        optimizer = self._ensure_optimizer()

        if self.ir is not None:
            self.ir.fit(loader)

        history: dict[str, list[float]] = {
            "train_loss": [],
            "energy_before": [],
            "energy_after": [],
        }
        for epoch in range(self.epochs):
            batch_results: list[dict[str, float]] = []
            for images, _ in loader:
                images = images.to(self.device, non_blocking=True)
                mixed = self._augment_with_replay(images)
                batch_results.append(self.train_batch(mixed, optimizer))
            for key in history:
                value = (
                    sum(result["loss" if key == "train_loss" else key] for result in batch_results)
                    / len(batch_results)
                    if batch_results
                    else 0.0
                )
                history[key].append(value)
            if self.logger is not None:
                self.logger.log_metrics(
                    {
                        "incremental/train_loss": history["train_loss"][-1],
                        "predictive_coding/energy_before": history["energy_before"][-1],
                        "predictive_coding/energy_after": history["energy_after"][-1],
                    },
                    step=self._class_count * self.epochs + epoch,
                )

        consolidation_losses: list[float] = []
        if self.consolidation_epochs > 0 and self.update_mode == "local":
            consolidation_optimizer = torch.optim.Adam(
                self.ae.parameters(),
                lr=self.base_lr * self.consolidation_lr_ratio,
                weight_decay=self.weight_decay,
            )
            for _ in range(self.consolidation_epochs):
                epoch_losses = []
                for images, _ in loader:
                    images = images.to(self.device, non_blocking=True)
                    mixed = self._augment_with_replay(images)
                    target = mixed.view(mixed.size(0), -1)
                    loss = F.mse_loss(self.ae(mixed)["recon"], target)
                    consolidation_optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    consolidation_optimizer.step()
                    self.update_steps += 1
                    epoch_losses.append(float(loss.detach().item()))
                consolidation_losses.append(
                    sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
                )

        self.history[class_id] = history
        self.diagnostics["energy_before"].extend(history["energy_before"])
        self.diagnostics["energy_after"].extend(history["energy_after"])
        self.diagnostics["local_loss"].extend(history["train_loss"])
        factors = self.plasticity_factors()
        self.diagnostics["class_summaries"][str(class_id)] = {
            "mean_energy_before": sum(history["energy_before"]) / len(history["energy_before"]),
            "mean_energy_after": sum(history["energy_after"]) / len(history["energy_after"]),
            "mean_local_loss": sum(history["train_loss"]) / len(history["train_loss"]),
            "plasticity_factor_min": min(float(value.min().item()) for value in factors),
            "plasticity_factor_max": max(float(value.max().item()) for value in factors),
            "consolidation_loss": consolidation_losses,
        }
        if self.ir is not None:
            self.ir.fit(loader)


__all__ = ["PredictiveCodingTrainer"]
