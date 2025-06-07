from typing import Any

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 128, num_workers: int = 4, data_dir: str = "./data"):
        super().__init__()
        self.save_hyperparameters()
        self.transform = transforms.ToTensor()

    def setup(self, stage: str | None = None) -> None:  # noqa: D401
        self.train_ds = datasets.MNIST(
            self.hparams.data_dir, train=True, download=True, transform=self.transform
        )
        self.val_ds = datasets.MNIST(
            self.hparams.data_dir, train=False, download=True, transform=self.transform
        )

    def train_dataloader(self) -> DataLoader[Any]:
        kwargs = {
            "dataset": self.train_ds,
            "batch_size": self.hparams.batch_size,
            "shuffle": True,
            "num_workers": self.hparams.num_workers,
        }
        if self.hparams.num_workers > 0:
            kwargs.update(persistent_workers=True, pin_memory=True)
        return DataLoader(**kwargs)

    def val_dataloader(self) -> DataLoader[Any]:
        kwargs = {
            "dataset": self.val_ds,
            "batch_size": self.hparams.batch_size,
            "shuffle": False,
            "num_workers": self.hparams.num_workers,
        }
        if self.hparams.num_workers > 0:
            kwargs.update(persistent_workers=True, pin_memory=True)
        return DataLoader(**kwargs)
