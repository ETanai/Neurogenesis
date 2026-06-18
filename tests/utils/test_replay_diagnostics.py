import torch
import pytest
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


@pytest.mark.parametrize(
    "mode,kwargs",
    [
        ("gaussian_full", {}),
        ("gaussian_diag", {}),
        ("gaussian_shrink", {"cov_shrinkage": 0.5}),
        ("mean_plus_noise", {"noise_scale": 0.25}),
        ("mean_only", {}),
    ],
)
def test_intrinsic_replay_sampling_modes_return_shapes_and_labels(mode, kwargs):
    torch.manual_seed(3)
    x = torch.rand(10, 4)
    y = torch.tensor([0] * 5 + [1] * 5)
    encoder = nn.Sequential(nn.Linear(4, 3))
    decoder = nn.Sequential(nn.Linear(3, 4), nn.Sigmoid())
    replay = IntrinsicReplay(
        encoder,
        decoder,
        device=torch.device("cpu"),
        sampling_mode=mode,
        **kwargs,
    )
    replay.fit(DataLoader(TensorDataset(x, y), batch_size=5))

    samples, labels = replay.sample_images_with_labels(None, 8)

    assert samples.shape == (8, 4)
    assert labels.shape == (8,)
    assert set(labels.tolist()).issubset({0, 1})


@pytest.mark.parametrize(
    "mode,kwargs",
    [
        ("gaussian_diag", {}),
        ("gaussian_shrink", {"cov_shrinkage": 0.5}),
        ("mean_plus_noise", {"noise_scale": 0.5}),
        ("mean_only", {}),
    ],
)
def test_intrinsic_replay_sampling_modes_are_seed_deterministic(mode, kwargs):
    torch.manual_seed(5)
    x = torch.rand(12, 4)
    y = torch.tensor([0] * 6 + [1] * 6)
    encoder = nn.Sequential(nn.Linear(4, 3))
    decoder = nn.Sequential(nn.Linear(3, 4), nn.Sigmoid())
    replay = IntrinsicReplay(
        encoder,
        decoder,
        device=torch.device("cpu"),
        sampling_mode=mode,
        **kwargs,
    )
    replay.fit(DataLoader(TensorDataset(x, y), batch_size=6))

    torch.manual_seed(11)
    first = replay.sample_latent(0, 7)
    torch.manual_seed(11)
    second = replay.sample_latent(0, 7)

    assert torch.allclose(first, second)


def test_intrinsic_replay_shrinkage_zero_matches_full_covariance():
    torch.manual_seed(13)
    x = torch.rand(12, 4)
    y = torch.tensor([0] * 6 + [1] * 6)
    encoder = nn.Sequential(nn.Linear(4, 3))
    decoder = nn.Sequential(nn.Linear(3, 4), nn.Sigmoid())
    full = IntrinsicReplay(
        encoder,
        decoder,
        device=torch.device("cpu"),
        sampling_mode="gaussian_full",
    )
    shrink = IntrinsicReplay(
        encoder,
        decoder,
        device=torch.device("cpu"),
        sampling_mode="gaussian_shrink",
        cov_shrinkage=0.0,
    )
    loader = DataLoader(TensorDataset(x, y), batch_size=6)
    full.fit(loader)
    shrink.fit(loader)

    torch.manual_seed(17)
    full_latent = full.sample_latent(0, 9)
    torch.manual_seed(17)
    shrink_latent = shrink.sample_latent(0, 9)

    assert torch.allclose(full_latent, shrink_latent)


def test_intrinsic_replay_shrinkage_one_matches_diagonal_covariance():
    torch.manual_seed(19)
    x = torch.rand(12, 4)
    y = torch.tensor([0] * 6 + [1] * 6)
    encoder = nn.Sequential(nn.Linear(4, 3))
    decoder = nn.Sequential(nn.Linear(3, 4), nn.Sigmoid())
    diag = IntrinsicReplay(
        encoder,
        decoder,
        device=torch.device("cpu"),
        sampling_mode="gaussian_diag",
    )
    shrink = IntrinsicReplay(
        encoder,
        decoder,
        device=torch.device("cpu"),
        sampling_mode="gaussian_shrink",
        cov_shrinkage=1.0,
    )
    loader = DataLoader(TensorDataset(x, y), batch_size=6)
    diag.fit(loader)
    shrink.fit(loader)

    torch.manual_seed(23)
    diag_latent = diag.sample_latent(0, 9)
    torch.manual_seed(23)
    shrink_latent = shrink.sample_latent(0, 9)

    assert torch.allclose(diag_latent, shrink_latent)


def test_intrinsic_replay_filter_none_matches_unfiltered_sampling():
    torch.manual_seed(29)
    x = torch.rand(12, 4)
    y = torch.tensor([0] * 6 + [1] * 6)
    encoder = nn.Sequential(nn.Linear(4, 3))
    decoder = nn.Sequential(nn.Linear(3, 4), nn.Sigmoid())
    base = IntrinsicReplay(
        encoder,
        decoder,
        device=torch.device("cpu"),
        filter_mode="none",
    )
    explicit = IntrinsicReplay(
        encoder,
        decoder,
        device=torch.device("cpu"),
        filter_mode="none",
        filter_percentile=0.5,
        filter_max_resample=0,
    )
    loader = DataLoader(TensorDataset(x, y), batch_size=6)
    base.fit(loader)
    explicit.fit(loader)

    torch.manual_seed(31)
    base_latent = base.sample_latent(0, 8)
    torch.manual_seed(31)
    explicit_latent = explicit.sample_latent(0, 8)

    assert torch.allclose(base_latent, explicit_latent)


@pytest.mark.parametrize(
    "filter_mode",
    ["latent_percentile", "roundtrip_percentile", "recon_error_match"],
)
def test_intrinsic_replay_filters_preserve_shapes(filter_mode):
    torch.manual_seed(37)
    x = torch.rand(16, 4)
    y = torch.tensor([0] * 8 + [1] * 8)
    encoder = nn.Sequential(nn.Linear(4, 3))
    decoder = nn.Sequential(nn.Linear(3, 4), nn.Sigmoid())
    replay = IntrinsicReplay(
        encoder,
        decoder,
        device=torch.device("cpu"),
        filter_mode=filter_mode,
        filter_percentile=0.9,
        filter_max_resample=2,
    )
    replay.fit(DataLoader(TensorDataset(x, y), batch_size=8))

    samples, labels = replay.sample_images_with_labels(None, 10)

    assert samples.shape == (10, 4)
    assert labels.shape == (10,)
    assert set(labels.tolist()).issubset({0, 1})
    assert "filter" in replay.stats[0]


def test_intrinsic_replay_filter_falls_back_when_acceptance_is_low():
    torch.manual_seed(41)
    x = torch.rand(8, 4)
    y = torch.tensor([0] * 4 + [1] * 4)
    encoder = nn.Sequential(nn.Linear(4, 2))
    decoder = nn.Sequential(nn.Linear(2, 4), nn.Sigmoid())
    replay = IntrinsicReplay(
        encoder,
        decoder,
        device=torch.device("cpu"),
        filter_mode="latent_percentile",
        filter_percentile=0.0,
        filter_max_resample=0,
    )
    replay.fit(DataLoader(TensorDataset(x, y), batch_size=4))

    latents = replay.sample_latent(0, 6)

    assert latents.shape == (6, 2)
