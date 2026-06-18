# src/utils/intrinsic_replay.py

from __future__ import annotations

import math
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
        eps: float = 1e-5,
        device: Optional[torch.device] = None,
        sampling_mode: str = "gaussian_full",
        cov_shrinkage: float = 0.0,
        noise_scale: float = 1.0,
        filter_mode: str = "none",
        filter_percentile: float = 0.95,
        filter_max_resample: int = 10,
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
        self.sampling_mode = str(sampling_mode or "gaussian_full").lower()
        valid_modes = {
            "gaussian_full",
            "gaussian_diag",
            "gaussian_shrink",
            "mean_plus_noise",
            "mean_only",
        }
        if self.sampling_mode not in valid_modes:
            raise ValueError(
                f"Unknown intrinsic replay sampling mode '{sampling_mode}'. "
                f"Expected one of {sorted(valid_modes)}."
            )
        self.cov_shrinkage = min(max(float(cov_shrinkage), 0.0), 1.0)
        self.noise_scale = max(float(noise_scale), 0.0)
        self.filter_mode = str(filter_mode or "none").lower()
        valid_filters = {
            "none",
            "latent_percentile",
            "roundtrip_percentile",
            "recon_error_match",
        }
        if self.filter_mode not in valid_filters:
            raise ValueError(
                f"Unknown intrinsic replay filter mode '{filter_mode}'. "
                f"Expected one of {sorted(valid_filters)}."
            )
        self.filter_percentile = min(max(float(filter_percentile), 0.0), 1.0)
        self.filter_max_resample = max(0, int(filter_max_resample))
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.encoder = encoder.eval().to(self.device)
        self.decoder = decoder.eval().to(self.device)

        # Will hold per-class: {"class_id": {"mean": (d,), "L": (d,d)}}
        self.stats: Dict[int, Dict[str, torch.Tensor]] = {}
        self._class_weights: Dict[int, float] = {}
        self.latent_dim = self._infer_encoder_latent_dim()

    def _infer_encoder_latent_dim(self) -> int:
        modules = list(self.encoder)
        for module in reversed(modules):
            out_features = getattr(module, "out_features", None)
            if out_features is not None:
                return int(out_features)
        raise RuntimeError("Could not infer encoder latent dimensionality.")

    def sync_encoder_latent_dim(self) -> None:
        target_dim = self._infer_encoder_latent_dim()
        if target_dim > self.latent_dim:
            self._expand_stats(target_dim)
        else:
            self.latent_dim = target_dim

    def _expand_stats(self, target_dim: int) -> None:
        current = self.latent_dim
        if target_dim <= current:
            self.latent_dim = target_dim
            return
        delta = target_dim - current
        if not self.stats:
            self.latent_dim = target_dim
            return
        jitter = math.sqrt(max(self.eps, 1e-12))
        for cls, stats in self.stats.items():
            mean = stats["mean"]
            pad = torch.zeros(delta, device=mean.device, dtype=mean.dtype)
            stats["mean"] = torch.cat([mean, pad], dim=0)

            L = stats["L"]
            new_L = L.new_zeros(target_dim, target_dim)
            new_L[:current, :current] = L
            new_L[current:, current:] = torch.eye(delta, device=L.device, dtype=L.dtype) * jitter
            stats["L"] = new_L
        self.latent_dim = target_dim

    def _safe_cholesky(self, cov: torch.Tensor) -> torch.Tensor:
        """
        Robust Cholesky that retries with increasing jitter along the diagonal to
        handle near-singular or non-PD covariances (common with few samples).
        """
        # enforce symmetry to reduce numerical noise
        cov = 0.5 * (cov + cov.T)
        eye = torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)
        jitter = self.eps
        for _ in range(7):
            try:
                return torch.linalg.cholesky(cov + jitter * eye)
            except RuntimeError:
                jitter *= 10.0
        # eigenvalue fallback: clamp to make PSD
        evals, evecs = torch.linalg.eigh(cov)
        min_pos = max(float(self.eps), float(evals.max().item()) * 1e-12)
        evals_clamped = torch.clamp(evals, min=min_pos)
        cov_pd = (evecs @ torch.diag(evals_clamped) @ evecs.T)
        cov_pd = 0.5 * (cov_pd + cov_pd.T) + self.eps * eye
        return torch.linalg.cholesky(cov_pd)

    @torch.no_grad()
    def fit(self, dataloader: DataLoader, *, class_filter: Optional[Iterable[int]] = None) -> None:
        """
        Pass every batch through the encoder and accumulate per-class latents
        to compute mean and covariance.
        Expects dataloader to yield (imgs, labels).
        """
        self.sync_encoder_latent_dim()
        accum: Dict[int, list[torch.Tensor]] = {}
        input_accum: Dict[int, list[torch.Tensor]] = {}

        for imgs, labels in dataloader:
            imgs = imgs.to(self.device, non_blocking=True)
            inputs = imgs.view(imgs.shape[0], self.encoder[0].in_features)
            latents = self.encoder(inputs)  # (B, D) on self.device
            for sample, z, y in zip(inputs, latents, labels.tolist()):
                if class_filter is not None and y not in class_filter:
                    continue
                # Keep tensors on the target device for consistent math
                accum.setdefault(int(y), []).append(z)
                input_accum.setdefault(int(y), []).append(sample)

        for cls, z_list in accum.items():
            Z = torch.stack(z_list, dim=0)  # (N_cls, D) on self.device
            X = torch.stack(input_accum.get(cls, []), dim=0)
            mu = Z.mean(dim=0)  # (D,)
            # unbiased covariance: (Z - mu).T @ (Z - mu) / (N-1)
            Zc = Z - mu
            if Zc.size(0) > 1:
                cov = (Zc.T @ Zc) / (Zc.size(0) - 1)
            else:
                cov = torch.zeros((Zc.size(1), Zc.size(1)), device=Z.device)
            # make PD (robust to rank deficiency)
            L = self._safe_cholesky(cov)
            latent_var_mean = Z.var(dim=0, unbiased=False).mean().item()
            filter_stats = self._fit_filter_stats(X, Z, mu, L)

            self.stats[cls] = {
                "mean": mu.to(self.device),
                "L": L.to(self.device),
                "count": int(Z.size(0)),
                "latent_var_mean": float(latent_var_mean),
                "filter": filter_stats,
            }

        if accum:
            self._refresh_default_weights()

    @torch.no_grad()
    def _class_covariance(self, cls: int) -> torch.Tensor:
        if cls not in self.stats:
            raise KeyError(f"No stats for class {cls}; did you .fit()?")
        L = self.stats[cls]["L"]
        cov = L @ L.T
        return 0.5 * (cov + cov.T)

    @torch.no_grad()
    def _mahalanobis(self, cls: int, latents: torch.Tensor) -> torch.Tensor:
        mu = self.stats[cls]["mean"]
        L = self.stats[cls]["L"]
        centered = latents - mu[None, :]
        try:
            solved = torch.linalg.solve_triangular(L, centered.T, upper=False).T
            return torch.linalg.norm(solved, dim=1)
        except RuntimeError:
            cov = self._class_covariance(cls)
            diag = torch.clamp(torch.diag(cov), min=1.0e-12)
            return torch.linalg.norm(centered / torch.sqrt(diag)[None, :], dim=1)

    @torch.no_grad()
    def _roundtrip_distance(self, latents: torch.Tensor) -> torch.Tensor:
        decoded = self.decoder(latents)
        encoded = self.encoder(decoded)
        return torch.linalg.norm(encoded - latents, dim=1)

    @torch.no_grad()
    def _self_reconstruction_error(self, flat: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(flat)
        decoded = self.decoder(encoded)
        return torch.mean((decoded - flat) ** 2, dim=1)

    @torch.no_grad()
    def _fit_filter_stats(
        self,
        inputs: torch.Tensor,
        latents: torch.Tensor,
        mu: torch.Tensor,
        L: torch.Tensor,
    ) -> Dict[str, float]:
        if inputs.numel() == 0 or latents.numel() == 0:
            return {}
        centered = latents - mu[None, :]
        try:
            solved = torch.linalg.solve_triangular(L, centered.T, upper=False).T
            maha = torch.linalg.norm(solved, dim=1)
        except RuntimeError:
            cov = L @ L.T
            diag = torch.clamp(torch.diag(cov), min=1.0e-12)
            maha = torch.linalg.norm(centered / torch.sqrt(diag)[None, :], dim=1)
        roundtrip = self._roundtrip_distance(latents)
        recon = self._self_reconstruction_error(inputs)

        def _quantile(values: torch.Tensor, q: float) -> float:
            if values.numel() == 0:
                return float("nan")
            return float(torch.quantile(values.detach().float(), q).item())

        q_hi = self.filter_percentile
        q_lo = 1.0 - self.filter_percentile
        return {
            "latent_mahalanobis_p": _quantile(maha, q_hi),
            "roundtrip_p": _quantile(roundtrip, q_hi),
            "recon_error_low_p": _quantile(recon, q_lo),
            "recon_error_high_p": _quantile(recon, q_hi),
            "clean_recon_error_mean": float(recon.mean().item()),
            "clean_recon_error_std": float(recon.std(unbiased=False).item()),
            "roundtrip_mean": float(roundtrip.mean().item()),
            "roundtrip_std": float(roundtrip.std(unbiased=False).item()),
            "latent_mahalanobis_mean": float(maha.mean().item()),
            "latent_mahalanobis_std": float(maha.std(unbiased=False).item()),
        }

    @torch.no_grad()
    def _sample_class_latent_raw(self, cls: int, n: int) -> torch.Tensor:
        if cls not in self.stats:
            raise KeyError(f"No stats for class {cls}; did you .fit()?")

        mu = self.stats[cls]["mean"]
        D = mu.size(0)
        if self.sampling_mode == "mean_only" or n <= 0:
            return mu[None, :].expand(int(n), D).clone()

        eps = torch.randn(int(n), D, device=self.device)
        cov = self._class_covariance(cls)
        diag = torch.clamp(torch.diag(cov), min=0.0)

        if self.sampling_mode == "gaussian_full":
            L = self.stats[cls]["L"]
            return mu[None, :] + self.noise_scale * (eps @ L.T)

        if self.sampling_mode == "gaussian_diag":
            std = torch.sqrt(diag)
            return mu[None, :] + self.noise_scale * eps * std[None, :]

        if self.sampling_mode == "gaussian_shrink":
            if self.cov_shrinkage <= 0.0:
                L = self.stats[cls]["L"]
                return mu[None, :] + self.noise_scale * (eps @ L.T)
            if self.cov_shrinkage >= 1.0:
                std = torch.sqrt(diag)
                return mu[None, :] + self.noise_scale * eps * std[None, :]
            diag_cov = torch.diag(diag)
            shrink = self.cov_shrinkage
            cov_eff = (1.0 - shrink) * cov + shrink * diag_cov
            L_eff = self._safe_cholesky(cov_eff)
            return mu[None, :] + self.noise_scale * (eps @ L_eff.T)

        if self.sampling_mode == "mean_plus_noise":
            scale = torch.sqrt(torch.clamp(diag.mean(), min=0.0))
            return mu[None, :] + self.noise_scale * eps * scale

        raise RuntimeError(f"Unhandled intrinsic replay sampling mode '{self.sampling_mode}'.")

    @torch.no_grad()
    def _filter_latents(self, cls: int, latents: torch.Tensor) -> torch.Tensor:
        if self.filter_mode == "none" or latents.numel() == 0:
            return torch.ones(latents.size(0), dtype=torch.bool, device=latents.device)
        stats = self.stats.get(cls, {}).get("filter", {})
        if not stats:
            return torch.ones(latents.size(0), dtype=torch.bool, device=latents.device)

        if self.filter_mode == "latent_percentile":
            threshold = stats.get("latent_mahalanobis_p", float("nan"))
            if not math.isfinite(float(threshold)):
                return torch.ones(latents.size(0), dtype=torch.bool, device=latents.device)
            return self._mahalanobis(cls, latents) <= float(threshold)

        if self.filter_mode == "roundtrip_percentile":
            threshold = stats.get("roundtrip_p", float("nan"))
            if not math.isfinite(float(threshold)):
                return torch.ones(latents.size(0), dtype=torch.bool, device=latents.device)
            return self._roundtrip_distance(latents) <= float(threshold)

        if self.filter_mode == "recon_error_match":
            low = stats.get("recon_error_low_p", float("nan"))
            high = stats.get("recon_error_high_p", float("nan"))
            if not math.isfinite(float(low)) or not math.isfinite(float(high)):
                return torch.ones(latents.size(0), dtype=torch.bool, device=latents.device)
            generated = self.decoder(latents)
            err = self._self_reconstruction_error(generated)
            return (err >= float(low)) & (err <= float(high))

        return torch.ones(latents.size(0), dtype=torch.bool, device=latents.device)

    @torch.no_grad()
    def _sample_class_latent(self, cls: int, n: int) -> torch.Tensor:
        if self.filter_mode == "none" or n <= 0:
            return self._sample_class_latent_raw(cls, n)

        accepted: list[torch.Tensor] = []
        remaining = int(n)
        attempts = 0
        while remaining > 0 and attempts <= self.filter_max_resample:
            draw_count = max(remaining * 2, remaining)
            candidates = self._sample_class_latent_raw(cls, draw_count)
            mask = self._filter_latents(cls, candidates)
            if mask.any():
                chosen = candidates[mask][:remaining]
                accepted.append(chosen)
                remaining -= int(chosen.size(0))
            attempts += 1

        if remaining > 0:
            accepted.append(self._sample_class_latent_raw(cls, remaining))
        return torch.cat(accepted, dim=0)[: int(n)]

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

        return self._sample_class_latent(int(cls), int(n))

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
    def sample_images_with_labels(
        self,
        cls: Optional[int],
        n: int,
        *,
        class_weights: Optional[Dict[int, float]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cls is not None:
            recons = self.sample_images(int(cls), int(n), class_weights=class_weights)
            labels = torch.full((int(n),), int(cls), dtype=torch.long, device=recons.device)
            return recons, labels

        if not self.stats:
            raise RuntimeError("No intrinsic replay statistics have been computed yet.")
        weights = class_weights or self._class_weights
        if not weights:
            raise RuntimeError("Class weights are undefined for intrinsic replay sampling.")
        classes = list(weights.keys())
        probs = torch.tensor([weights[c] for c in classes], dtype=torch.float, device=self.device)
        probs = probs / probs.sum()
        draw = torch.multinomial(probs, num_samples=int(n), replacement=True)
        images: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        for cls_idx, count in zip(*torch.unique(draw, return_counts=True)):
            cls_value = int(classes[int(cls_idx)])
            count_int = int(count)
            recons = self.sample_images(cls_value, count_int)
            images.append(recons)
            labels.append(
                torch.full((count_int,), cls_value, dtype=torch.long, device=recons.device)
            )
        return torch.cat(images, dim=0), torch.cat(labels, dim=0)

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
                "sampling_mode": self.sampling_mode,
                "cov_shrinkage": float(self.cov_shrinkage),
                "noise_scale": float(self.noise_scale),
                "filter_mode": self.filter_mode,
                "filter_percentile": float(self.filter_percentile),
                "filter_max_resample": float(self.filter_max_resample),
            }
            filt = stats.get("filter", {})
            for key, value in filt.items():
                summary[int(cls)][f"filter_{key}"] = float(value)
        return summary

    def get_class_weights(self) -> Dict[int, float]:
        return {int(k): float(v) for k, v in self._class_weights.items()}
