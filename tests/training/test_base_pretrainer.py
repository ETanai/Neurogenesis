import torch
from torch.utils.data import DataLoader, TensorDataset

from models.ng_autoencoder import NGAutoEncoder
from training.base_pretrainer import AutoencoderPretrainer, PretrainingConfig


def _make_loader(n_samples: int = 8, input_dim: int = 4, batch_size: int = 4) -> DataLoader:
    x = torch.randn(n_samples, input_dim)
    y = torch.zeros(n_samples, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def test_pretrainer_joint_mode_update_steps():
    model = NGAutoEncoder(input_dim=4, hidden_sizes=[3, 2], activation_last="identity")
    loader = _make_loader()
    cfg = PretrainingConfig(epochs=2, lr=1e-3, device="cpu", mode="joint")
    trainer = AutoencoderPretrainer(model, cfg)

    trainer.fit(loader)

    # 8 samples / batch 4 => 2 batches per epoch, 2 epochs.
    assert trainer.update_steps == 4


def test_pretrainer_stacked_mode_runs_per_level_and_logs():
    model = NGAutoEncoder(input_dim=4, hidden_sizes=[3, 2], activation_last="identity")
    loader = _make_loader()
    cfg = PretrainingConfig(
        epochs=2,
        lr=1e-3,
        device="cpu",
        mode="stacked",
        stacked_level_epochs=1,
    )
    trainer = AutoencoderPretrainer(model, cfg)
    logs: list[tuple[dict, int]] = []

    trainer.fit(loader, log_fn=lambda metrics, step: logs.append((metrics, step)))

    # 2 levels * 1 epoch each * 2 batches.
    assert trainer.update_steps == 4
    keys = {k for metrics, _ in logs for k in metrics.keys()}
    assert "pretrain/stacked/level_0/train_loss" in keys
    assert "pretrain/stacked/level_1/train_loss" in keys


def test_pretrainer_denoising_corrupts_inputs_only():
    model = NGAutoEncoder(input_dim=4, hidden_sizes=[3, 2], activation_last="identity")
    loader = _make_loader()
    cfg = PretrainingConfig(
        epochs=1,
        lr=1e-3,
        device="cpu",
        mode="joint",
        denoising_enabled=True,
        denoising_noise_type="mask",
        denoising_mask_prob=1.0,
    )
    trainer = AutoencoderPretrainer(model, cfg)
    x = next(iter(loader))[0]
    x_corrupt = trainer._corrupt_inputs(x)
    # Mask noise with p=1.0 should fully zero the input while target remains original in training loop.
    assert torch.allclose(x_corrupt, torch.zeros_like(x))
    assert not torch.allclose(x, torch.zeros_like(x))
