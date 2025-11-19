# src/utils/intrinsic_replay.py

from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader


class IntrinsicReplay:
    """
    Stores per-class Gaussian stats in latent space of an encoder,
    and can sample + decode new examples via the paired decoder.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        eps: float = 1e-6,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            encoder: maps input images -> latent vectors
            decoder: maps latent vectors -> flattened reconstructions
            eps: small diagonal noise to make covariance PD
            device: where to accumulate / sample (defaults to encoder device)
        """
        # Establish device first, then move submodules consistently.
        self.eps = eps
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.encoder = encoder.eval().to(self.device)
        self.decoder = decoder.eval().to(self.device)

        # Will hold per-class: {"class_id": {"mean": (d,), "L": (d,d)}}
        self.stats: Dict[int, Dict[str, torch.Tensor]] = {}
        self._class_weights: Dict[int, float] = {}

    @torch.no_grad()
    def fit(self, dataloader: DataLoader, *, class_filter: Optional[Iterable[int]] = None) -> None:
        """
        Pass every batch through the encoder and accumulate per-class latents
        to compute mean and covariance.
        Expects dataloader to yield (imgs, labels).
        """
        accum: Dict[int, list[torch.Tensor]] = {}

        for imgs, labels in dataloader:
            imgs = imgs.to(self.device, non_blocking=True)
            inputs = imgs.view(imgs.shape[0], self.encoder[0].in_features)
            latents = self.encoder(inputs)  # (B, D) on self.device
            for z, y in zip(latents, labels.tolist()):
                if class_filter is not None and y not in class_filter:
                    continue
                # Keep tensors on the target device for consistent math
                accum.setdefault(int(y), []).append(z)

        for cls, z_list in accum.items():
            Z = torch.stack(z_list, dim=0)  # (N_cls, D) on self.device
            mu = Z.mean(dim=0)  # (D,)
            # unbiased covariance: (Z - mu).T @ (Z - mu) / (N-1)
            Zc = Z - mu
            if Zc.size(0) > 1:
                cov = (Zc.T @ Zc) / (Zc.size(0) - 1)
            else:
                cov = torch.zeros((Zc.size(1), Zc.size(1)), device=Z.device)
            # make PD
            cov = cov + self.eps * torch.eye(cov.size(0), device=Z.device)
            # cholesky: cov = L @ L.T
            L = torch.linalg.cholesky(cov)
            latent_var_mean = Z.var(dim=0, unbiased=False).mean().item()

            self.stats[cls] = {
                "mean": mu.to(self.device),
                "L": L.to(self.device),
                "count": int(Z.size(0)),
                "latent_var_mean": float(latent_var_mean),
            }

        if accum:
            self._refresh_default_weights()

    @torch.no_grad()
    def sample_latent(
        self,
        cls: Optional[int],
        n: int,
        *,
        class_weights: Optional[Dict[int, float]] = None,
    ) -> torch.Tensor:
        """
        Draw `n` latent vectors. If ``cls`` is ``None`` we mix according to weights.
        """
        if not self.stats:
            raise RuntimeError("No intrinsic replay statistics have been computed yet.")

        if cls is None:
            weights = class_weights or self._class_weights
            if not weights:
                raise RuntimeError("Class weights are undefined for intrinsic replay sampling.")
            classes = list(weights.keys())
            probs = torch.tensor([weights[c] for c in classes], dtype=torch.float, device=self.device)
            probs = probs / probs.sum()
            draw = torch.multinomial(probs, num_samples=n, replacement=True)
            latents: list[torch.Tensor] = []
            for cls_idx, count in zip(*torch.unique(draw, return_counts=True)):
                latents.append(self.sample_latent(int(classes[int(cls_idx)]), int(count)))
            return torch.cat(latents, dim=0)

        if cls not in self.stats:
            raise KeyError(f"No stats for class {cls}; did you .fit()?")

        mu = self.stats[cls]["mean"]  # (D,)
        L = self.stats[cls]["L"]  # (D,D)
        D = mu.size(0)

        eps = torch.randn(n, D, device=self.device)
        return mu[None, :] + eps @ L.T

    @torch.no_grad()
    def sample_images(
        self,
        cls: Optional[int],
        n: int,
        *,
        class_weights: Optional[Dict[int, float]] = None,
    ) -> torch.Tensor:
        """
        Returns `n` reconstructed images (flattened) via decoder.
        Shape: (n, *) matching decoder output, e.g. (n, 28*28).
        """
        zs = self.sample_latent(cls, n, class_weights=class_weights)  # (n, D)
        recons = self.decoder(zs)  # (n, input_dim)
        return recons

    @torch.no_grad()
    def sample_image_tensors(
        self, cls: int, n: int, view_shape: Optional[tuple[int, ...]] = None
    ) -> torch.Tensor:
        """
        Like `sample_images` but reshaped to real image tensors.
        e.g. view_shape=(1,28,28) -> returns (n,1,28,28)
        """
        recons = self.sample_images(cls, n)
        if view_shape is not None:
            return recons.view(n, *view_shape)
        return recons

    def available_classes(self) -> list[int]:
        return sorted(self.stats.keys())

    def set_class_weights(self, weights: Dict[int, float]) -> None:
        if not weights:
            self._class_weights = {}
            return
        total = float(sum(weights.values()))
        if total <= 0:
            raise ValueError("Class weights must sum to a positive value")
        self._class_weights = {int(k): float(v) for k, v in weights.items()}

    def _refresh_default_weights(self) -> None:
        if not self.stats:
            self._class_weights = {}
            return
        uniform = 1.0 / len(self.stats)
        self._class_weights = {cls: uniform for cls in self.stats}

    def describe(self) -> Dict[int, Dict[str, float]]:
        summary: Dict[int, Dict[str, float]] = {}
        for cls, stats in self.stats.items():
            L = stats["L"].detach().cpu()
            cov = L @ L.T
            cond = torch.linalg.cond(cov).item() if cov.numel() > 0 else float("nan")
            summary[int(cls)] = {
                "count": float(stats.get("count", 0)),
                "latent_var_mean": float(stats.get("latent_var_mean", 0.0)),
                "cov_condition": float(cond),
            }
        return summary

    def get_class_weights(self) -> Dict[int, float]:
        return {int(k): float(v) for k, v in self._class_weights.items()}
