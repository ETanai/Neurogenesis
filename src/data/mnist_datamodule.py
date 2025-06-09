from typing import Any, Sequence

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class MNISTDataModule(pl.LightningDataModule):
    """
    A LightningDataModule for MNIST that can optionally restrict to a subset of classes.

    Args:
        batch_size: batch size
        num_workers: number of DataLoader workers
        data_dir: where to download/store the MNIST data
        classes: if provided, only these digit classes will be kept (e.g. [0,2,5])
    """

    def __init__(
        self,
        batch_size: int = 128,
        num_workers: int = 4,
        data_dir: str = "./data",
        classes: Sequence[int] | None = None,
    ):
        super().__init__()
        # this will let you refer to self.hparams.classes and have it in logger/hyperparams
        self.save_hyperparameters("batch_size", "num_workers", "data_dir", "classes")

        self.transform = transforms.ToTensor()

    def setup(self, stage: str | None = None) -> None:
        # load the full MNIST train/val
        full_train = datasets.MNIST(
            self.hparams.data_dir,
            train=True,
            download=True,
            transform=self.transform,
        )
        full_val = datasets.MNIST(
            self.hparams.data_dir,
            train=False,
            download=True,
            transform=self.transform,
        )

        # if classes was passed, filter each dataset down to just those labels
        if self.hparams.classes is not None:
            cls_tensor = torch.tensor(self.hparams.classes)
            # for train
            train_targets = full_train.targets
            train_mask = torch.isin(train_targets, cls_tensor)
            train_idxs = train_mask.nonzero(as_tuple=True)[0].tolist()
            self.train_ds = Subset(full_train, train_idxs)

            # for val
            val_targets = full_val.targets
            val_mask = torch.isin(val_targets, cls_tensor)
            val_idxs = val_mask.nonzero(as_tuple=True)[0].tolist()
            self.val_ds = Subset(full_val, val_idxs)
        else:
            # no filtering
            self.train_ds = full_train
            self.val_ds = full_val

    def train_dataloader(self) -> DataLoader[Any]:
        kwargs = dict(
            dataset=self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )
        if self.hparams.num_workers > 0:
            kwargs.update(persistent_workers=True, pin_memory=True)
        return DataLoader(**kwargs)

    def val_dataloader(self) -> DataLoader[Any]:
        kwargs = dict(
            dataset=self.val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )
        if self.hparams.num_workers > 0:
            kwargs.update(persistent_workers=True, pin_memory=True)
        return DataLoader(**kwargs)
