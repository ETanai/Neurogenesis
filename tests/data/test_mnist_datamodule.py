import pytest
import torch
from torch.utils.data import Dataset

from data.mnist_datamodule import MNISTDataModule


class DummyDataset(Dataset):
    """Simple in-memory dataset of random 28×28 images and dummy labels."""

    def __init__(self, size=10):
        self.data = torch.randn(size, 1, 28, 28)
        self.targets = list(range(size))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


@pytest.fixture(autouse=True)
def patch_mnist(monkeypatch):
    """Replace torchvision.datasets.MNIST with DummyDataset."""
    import torchvision.datasets as vdatasets

    monkeypatch.setattr(vdatasets, "MNIST", lambda *args, **kwargs: DummyDataset())
    yield


def test_setup_creates_datasets():
    dm = MNISTDataModule(batch_size=4, num_workers=0, data_dir=".")
    # Before setup, attributes shouldn't exist
    assert not hasattr(dm, "train_ds")
    assert not hasattr(dm, "val_ds")

    dm.setup()
    # After setup, dataset attributes must exist
    assert hasattr(dm, "train_ds")
    assert hasattr(dm, "val_ds")
    # Should be instances of our DummyDataset
    assert isinstance(dm.train_ds, DummyDataset)
    assert isinstance(dm.val_ds, DummyDataset)


@pytest.mark.parametrize("loader_fn", ["train_dataloader", "val_dataloader"])
def test_dataloader_batches(loader_fn):
    dm = MNISTDataModule(batch_size=4, num_workers=0, data_dir=".")
    dm.setup()

    loader = getattr(dm, loader_fn)()
    batch = next(iter(loader))
    imgs, labels = batch

    # Check batch shapes
    assert imgs.shape == (4, 1, 28, 28)
    assert labels.shape == (4,)
