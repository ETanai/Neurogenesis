# src/utils/intrinsic_replay.py

from typing import Dict, Optional

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
        self.encoder = encoder.eval().to(device)
        self.decoder = decoder.eval().to(device)
        self.eps = eps
        self.device = device or next(encoder.parameters()).device

        # Will hold per-class: {"class_id": {"mean": (d,), "L": (d,d)}}
        self.stats: Dict[int, Dict[str, torch.Tensor]] = {}

    @torch.no_grad()
    def fit(self, dataloader: DataLoader):
        """
        Pass every batch through the encoder and accumulate per-class latents
        to compute mean and covariance.
        Expects dataloader to yield (imgs, labels).
        """
        accum: Dict[int, list[torch.Tensor]] = {}

        for imgs, labels in dataloader:
            imgs = imgs.to(self.device)
            inputs = imgs.view(imgs.shape[0], self.encoder[0].in_features)
            latents = self.encoder(inputs)  # (B, D)
            for z, y in zip(latents, labels.tolist()):
                accum.setdefault(y, []).append(z.cpu())

        for cls, z_list in accum.items():
            Z = torch.stack(z_list, dim=0)  # (N_cls, D)
            mu = Z.mean(dim=0)  # (D,)
            # unbiased covariance: (Z - mu).T @ (Z - mu) / (N-1)
            Zc = Z - mu
            cov = (Zc.T @ Zc) / (Zc.size(0) - 1)
            # make PD
            cov += self.eps * torch.eye(cov.size(0))
            # cholesky: cov = L @ L.T
            L = torch.linalg.cholesky(cov)

            self.stats[cls] = {
                "mean": mu.to(self.device),
                "L": L.to(self.device),
            }

    @torch.no_grad()
    def sample_latent(self, cls: int, n: int) -> torch.Tensor:
        """
        Draw `n` latent vectors for class `cls`: z = mu + L @ eps
        """
        if cls not in self.stats:
            raise KeyError(f"No stats for class {cls}; did you .fit() ?")

        mu = self.stats[cls]["mean"]  # (D,)
        L = self.stats[cls]["L"]  # (D,D)
        D = mu.size(0)

        # standard normal eps: (n, D)
        eps = torch.randn(n, D, device=self.device)
        # sample: (n, D) = mu + eps @ L.T
        return mu[None, :] + eps @ L.T

    @torch.no_grad()
    def sample_images(self, cls: int, n: int) -> torch.Tensor:
        """
        Returns `n` reconstructed images (flattened) via decoder.
        Shape: (n, *) matching decoder output, e.g. (n, 28*28).
        """
        zs = self.sample_latent(cls, n)  # (n, D)
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
