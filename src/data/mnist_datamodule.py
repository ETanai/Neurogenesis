from typing import Any, Sequence

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
from torchvision import datasets, transforms


class _MemMNIST(Dataset):
    """MNIST fully in RAM, raw tensors in [0,1]. Uses external sampler for subsetting."""

    def __init__(self, train: bool, data_dir: str):
        t = transforms.ToTensor()
        ds = datasets.MNIST(data_dir, train=train, download=True, transform=t)
        self.data = ds.data.float().div(255).unsqueeze(1)  # [N,1,28,28]
        self.targets = ds.targets

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i], self.targets[i]


class _ListSampler(Sampler[int]):
    """Sampler with mutable index list."""

    def __init__(self, idxs: Sequence[int]):
        self.idxs = list(idxs)

    def set(self, idxs: Sequence[int] | torch.Tensor) -> None:
        self.idxs = list(idxs)

    def __iter__(self):
        return iter(self.idxs)

    def __len__(self) -> int:
        return len(self.idxs)


class MNISTDataModule(pl.LightningDataModule):
    """
    LightningDataModule for MNIST.
    Fully in-RAM, single persistent loader, dynamic subset switching.

    Maintains original signatures:
      - train_dataloader()
      - val_dataloader()
      - get_class_dataloader(class_id)
      - get_combined_dataloader(class_ids)
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
        # preserve signature params
        self.save_hyperparameters()

    def setup(self, stage: Any = None) -> None:
        # load full datasets into RAM
        self.full_train = _MemMNIST(train=True, data_dir=self.hparams.data_dir)
        self.full_val = _MemMNIST(train=False, data_dir=self.hparams.data_dir)

        # dynamic samplers over full sets
        train_idxs = list(range(len(self.full_train)))
        val_idxs = list(range(len(self.full_val)))

        # optional global class filter
        if self.hparams.classes is not None:
            cls = torch.as_tensor(self.hparams.classes)
            mask = torch.isin(self.full_train.targets, cls)
            train_idxs = mask.nonzero(as_tuple=True)[0].tolist()
            mask = torch.isin(self.full_val.targets, cls)
            val_idxs = mask.nonzero(as_tuple=True)[0].tolist()

        self.train_sampler = _ListSampler(train_idxs)
        self.val_sampler = _ListSampler(val_idxs)

        # persistent single loaders
        self._train_loader = DataLoader(
            self.full_train,
            batch_size=self.hparams.batch_size,
            sampler=self.train_sampler,
            num_workers=0,
            pin_memory=True,
        )
        self._val_loader = DataLoader(
            self.full_val,
            batch_size=self.hparams.batch_size,
            sampler=self.val_sampler,
            num_workers=0,
            pin_memory=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self._train_loader

    def val_dataloader(self) -> DataLoader:
        return self._val_loader

    def get_class_dataloader(self, class_id: int) -> DataLoader:
        """Switch to only that digit class."""
        idxs = (self.full_train.targets == class_id).nonzero(as_tuple=True)[0]
        self.train_sampler.set(idxs)
        return self._train_loader

    def get_combined_dataloader(self, class_ids: Sequence[int]) -> DataLoader:
        """Switch to multiple classes."""
        cls = torch.as_tensor(list(class_ids))
        mask = torch.isin(self.full_train.targets, cls)
        idxs = mask.nonzero(as_tuple=True)[0]
        self.train_sampler.set(idxs)
        return self._train_loader

    def get_class_dataset(self, class_id: int) -> Subset:
        t = self.full_train.targets
        idxs = (t == class_id).nonzero(as_tuple=True)[0].tolist()
        return Subset(self.full_train, idxs)

    def get_combined_dataset(self, class_ids: Sequence[int]) -> Subset:
        cls = torch.as_tensor(list(class_ids))
        t = self.full_train.targets
        mask = torch.isin(t, cls)
        idxs = mask.nonzero(as_tuple=True)[0].tolist()
        return Subset(self.full_train, idxs)
