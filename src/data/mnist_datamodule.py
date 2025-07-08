from typing import Any, Sequence

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data._utils.collate import default_collate
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
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__()
        # this will let you refer to self.hparams.classes and have it in logger/hyperparams
        self.save_hyperparameters("batch_size", "num_workers", "data_dir", "classes")

        self.transform = transforms.ToTensor()
        self.device = device

    def _move_collate(self, batch):
        x, y = default_collate(batch)
        return x.to(self.device), y.to(self.device)

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

        self.full_train = full_train
        self.full_val = full_val

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
            collate_fn=self._move_collate,
        )
        if self.hparams.num_workers > 0:
            kwargs.update(persistent_workers=True)
        return DataLoader(**kwargs)

    def val_dataloader(self) -> DataLoader[Any]:
        kwargs = dict(
            dataset=self.val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=self._move_collate,
        )
        if self.hparams.num_workers > 0:
            kwargs.update(persistent_workers=True)
        return DataLoader(**kwargs)

    def get_class_dataset(self, class_id: int) -> Subset:
        """Return a Subset for a single class for training."""
        t = self.full_train.targets
        idxs = (t == class_id).nonzero(as_tuple=True)[0].tolist()
        return Subset(self.full_train, idxs)

    def get_class_dataloader(self, class_id: int) -> DataLoader:
        """Return a DataLoader for a single class."""
        ds = self.get_class_dataset(class_id)
        return DataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers > 0,
            collate_fn=self._move_collate,
        )

    def get_combined_dataset(self, class_ids: Sequence[int]) -> Subset:
        """Return a Subset for multiple classes for training."""
        cls_tensor = torch.tensor(list(class_ids))
        t = self.full_train.targets
        mask = torch.isin(t, cls_tensor)
        idxs = mask.nonzero(as_tuple=True)[0].tolist()
        return Subset(self.full_train, idxs)

    def get_combined_dataloader(self, class_ids: Sequence[int]) -> DataLoader:
        """Return a DataLoader for multiple classes."""
        ds = self.get_combined_dataset(class_ids)
        return DataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers > 0,
        )
