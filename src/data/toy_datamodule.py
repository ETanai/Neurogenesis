"""Toy data module that mimics MNIST-style shapes for fast tests."""

from __future__ import annotations

import math
from typing import Any, Sequence

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, Subset


class _ToyDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor):
        self.images = images.float()
        self.labels = labels.long()

    def __len__(self) -> int:  # noqa: D401
        return int(self.images.size(0))

    def __getitem__(self, idx: int):  # noqa: D401
        return self.images[idx], self.labels[idx]


class ToyDataModule(pl.LightningDataModule):
    """Generate low-resolution digit-like blobs for neurogenesis tests."""

    def __init__(
        self,
        batch_size: int = 16,
        num_workers: int = 0,
        data_dir: str | None = None,
        *,
        num_classes: int = 4,
        train_samples_per_class: int = 32,
        val_samples_per_class: int = 16,
        noise_scale: float = 0.05,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self._train: _ToyDataset | None = None
        self._val: _ToyDataset | None = None

    @property
    def image_shape(self) -> tuple[int, int, int]:
        return (1, 28, 28)

    def setup(self, stage: Any = None) -> None:  # noqa: D401
        g = torch.Generator().manual_seed(int(self.hparams.seed))
        self._train = self._make_dataset(
            split="train",
            samples_per_class=int(self.hparams.train_samples_per_class),
            generator=g,
        )
        self._val = self._make_dataset(
            split="val",
            samples_per_class=int(self.hparams.val_samples_per_class),
            generator=g,
        )

    def _make_dataset(
        self,
        *,
        split: str,
        samples_per_class: int,
        generator: torch.Generator,
    ) -> _ToyDataset:
        imgs: list[torch.Tensor] = []
        labels: list[int] = []
        for cls in range(int(self.hparams.num_classes)):
            base = self._base_pattern(cls)
            noise = torch.randn(
                samples_per_class,
                *self.image_shape,
                generator=generator,
            ) * float(self.hparams.noise_scale)
            data = (base + noise).clamp(0.0, 1.0)
            imgs.append(data)
            labels.extend([cls] * samples_per_class)
        images = torch.cat(imgs, dim=0)
        labels_t = torch.tensor(labels, dtype=torch.long)
        perm = torch.randperm(images.size(0), generator=generator)
        return _ToyDataset(images[perm], labels_t[perm])

    def _base_pattern(self, cls: int) -> torch.Tensor:
        _channels, h, w = self.image_shape
        rows = math.ceil(math.sqrt(int(self.hparams.num_classes)))
        cols = math.ceil(int(self.hparams.num_classes) / rows)
        r = cls // cols
        c_idx = cls % cols
        grid_y = torch.linspace(0, 1, steps=h)
        grid_x = torch.linspace(0, 1, steps=w)
        yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")
        center_y = (r + 0.5) / rows
        center_x = (c_idx + 0.5) / cols
        sigma = 0.12 + 0.02 * (cls % 3)
        blob = torch.exp(-((yy - center_y) ** 2 + (xx - center_x) ** 2) / (2 * sigma**2))
        blob = blob / blob.max()
        return blob.view(1, h, w)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train,
            batch_size=int(self.hparams.batch_size),
            shuffle=True,
            num_workers=int(self.hparams.num_workers),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val,
            batch_size=int(self.hparams.batch_size),
            shuffle=False,
            num_workers=int(self.hparams.num_workers),
        )

    def _get_subset(self, class_ids: Sequence[int], *, split: str = "train") -> Subset:
        dataset = self._train if split == "train" else self._val
        if dataset is None:
            raise RuntimeError("ToyDataModule.setup() must be called before accessing data.")
        mask = torch.isin(dataset.labels, torch.tensor(list(class_ids)))
        idxs = mask.nonzero(as_tuple=True)[0].tolist()
        return Subset(dataset, idxs)

    def get_class_dataset(self, class_id: int, *, split: str = "train") -> Subset:
        return self._get_subset([int(class_id)], split=split)

    def get_combined_dataset(self, class_ids: Sequence[int], *, split: str = "train") -> Subset:
        return self._get_subset(class_ids, split=split)
