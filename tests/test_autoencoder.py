import torch

from models.autoencoder import AutoEncoder


def test_shapes() -> None:
    ae = AutoEncoder(28 * 28, [64, 32])
    dummy = torch.randn(4, 1, 28, 28)
    out = ae(dummy)
    assert out["recon"].shape == (4, 28 * 28)
    assert out["latent"].shape[-1] == 32
    assert out["latent"].shape[-1] == 32
