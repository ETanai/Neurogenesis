from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
from torch.utils.data import DataLoader


class DatasetReplay:
    """Stores real samples per class to replay without decoder generation."""

    def __init__(
        self,
        *,
        input_dim: int,
        max_per_class: int | None = None,
        storage_device: torch.device | str = "cpu",
    ) -> None:
        self.input_dim = int(input_dim)
        self.max_per_class = None if max_per_class is None else int(max_per_class)
        self.device = torch.device(storage_device)
        self.stats: Dict[int, Dict[str, torch.Tensor]] = {}
        self._class_weights: Dict[int, float] = {}

    def reset(self) -> None:
        self.stats.clear()
        self._class_weights = {}

    def _normalize_samples(self, samples: torch.Tensor) -> torch.Tensor:
        flat = samples.view(samples.size(0), -1)
        if flat.size(1) != self.input_dim:
            raise ValueError(
                f"Replay sample dimensionality mismatch: expected {self.input_dim}, got {flat.size(1)}"
            )
        return flat.to(self.device)

    @torch.no_grad()
    def fit(
        self,
        dataloader: DataLoader,
        *,
        class_filter: Iterable[int] | None = None,
    ) -> None:
        filtered = set(int(c) for c in class_filter) if class_filter is not None else None
        bucket: Dict[int, list[torch.Tensor]] = {}
        quota = {cls: self.max_per_class for cls in (filtered or [])}

        for images, labels in dataloader:
            if filtered is not None:
                mask = [int(lbl) in filtered for lbl in labels.tolist()]
                if not any(mask):
                    continue
            flat = self._normalize_samples(images)
            for sample, cls in zip(flat, labels.tolist()):
                cls = int(cls)
                if filtered is not None and cls not in filtered:
                    continue
                if self.max_per_class is not None:
                    remaining = quota.get(cls, self.max_per_class)
                    if remaining is not None and remaining <= 0:
                        continue
                    quota[cls] = remaining - 1 if remaining is not None else None
                bucket.setdefault(cls, []).append(sample.detach().cpu())

        for cls, samples in bucket.items():
            if not samples:
                continue
            stacked = torch.stack(samples, dim=0)
            if self.max_per_class is not None and stacked.size(0) > self.max_per_class:
                perm = torch.randperm(stacked.size(0))[: self.max_per_class]
                stacked = stacked[perm]
            self.stats[cls] = {
                "samples": stacked.to(self.device),
                "count": int(stacked.size(0)),
            }

        if bucket:
            self._refresh_default_weights()

    def available_classes(self) -> list[int]:
        return sorted(self.stats.keys())

    def _refresh_default_weights(self) -> None:
        if not self.stats:
            self._class_weights = {}
            return
        uniform = 1.0 / len(self.stats)
        self._class_weights = {cls: uniform for cls in self.stats}

    def set_class_weights(self, weights: Dict[int, float]) -> None:
        if not weights:
            self._class_weights = {}
            return
        total = float(sum(weights.values()))
        if total <= 0:
            raise ValueError("Class weights must sum to a positive value")
        self._class_weights = {int(k): float(v) for k, v in weights.items()}

    def get_class_weights(self) -> Dict[int, float]:
        return {int(k): float(v) for k, v in self._class_weights.items()}

    def describe(self) -> Dict[int, Dict[str, float]]:
        return {
            int(cls): {"count": float(stats.get("count", 0))}
            for cls, stats in self.stats.items()
        }

    def _sample_from_class(self, cls: int, n: int) -> torch.Tensor:
        if cls not in self.stats:
            raise KeyError(f"No dataset replay samples stored for class {cls}")
        samples = self.stats[cls]["samples"]
        if samples.numel() == 0:
            raise RuntimeError(f"Replay buffer for class {cls} is empty")
        idx = torch.randint(0, samples.size(0), (n,), device=samples.device)
        return samples.index_select(0, idx).to(self.device)

    @torch.no_grad()
    def sample_images(
        self,
        cls: Optional[int],
        n: int,
        *,
        class_weights: Dict[int, float] | None = None,
    ) -> torch.Tensor:
        if not self.stats:
            raise RuntimeError("Dataset replay buffer is empty; call fit() first.")
        if cls is None:
            weights = class_weights or self._class_weights
            if not weights:
                raise RuntimeError("Class weights undefined for dataset replay sampling.")
            classes = list(weights.keys())
            probs = torch.tensor(
                [weights[c] for c in classes],
                dtype=torch.float32,
                device=self.device,
            )
            probs = probs / probs.sum()
            draws = torch.multinomial(probs, num_samples=n, replacement=True)
            parts: list[torch.Tensor] = []
            for cls_idx, count in zip(*torch.unique(draws, return_counts=True)):
                parts.append(
                    self._sample_from_class(int(classes[int(cls_idx)]), int(count))
                )
            return torch.cat(parts, dim=0)
        return self._sample_from_class(int(cls), int(n))

    @torch.no_grad()
    def sample_image_tensors(
        self,
        cls: Optional[int],
        n: int,
        *,
        view_shape: tuple[int, ...] | None = None,
        class_weights: Dict[int, float] | None = None,
    ) -> torch.Tensor:
        flat = self.sample_images(cls, n, class_weights=class_weights)
        if view_shape is not None:
            return flat.view(n, *view_shape)
        return flat

