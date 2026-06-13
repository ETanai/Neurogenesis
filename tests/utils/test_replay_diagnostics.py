import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils.dataset_replay import DatasetReplay
from utils.intrinsic_replay import IntrinsicReplay


def test_dataset_replay_samples_with_labels():
    x = torch.arange(24, dtype=torch.float32).view(6, 1, 2, 2) / 24.0
    y = torch.tensor([0, 0, 1, 1, 1, 2])
    replay = DatasetReplay(input_dim=4)
    replay.fit(DataLoader(TensorDataset(x, y), batch_size=2))

    samples, labels = replay.sample_images_with_labels(None, 12)

    assert samples.shape == (12, 4)
    assert labels.shape == (12,)
    assert set(labels.tolist()).issubset({0, 1, 2})


def test_intrinsic_replay_samples_with_labels():
    torch.manual_seed(0)
    x = torch.rand(8, 4)
    y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    encoder = nn.Sequential(nn.Linear(4, 2))
    decoder = nn.Sequential(nn.Linear(2, 4), nn.Sigmoid())
    replay = IntrinsicReplay(encoder, decoder, device=torch.device("cpu"))
    replay.fit(DataLoader(TensorDataset(x, y), batch_size=4))

    samples, labels = replay.sample_images_with_labels(None, 10)

    assert samples.shape == (10, 4)
    assert labels.shape == (10,)
    assert set(labels.tolist()).issubset({0, 1})
