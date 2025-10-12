import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from data.mnist_datamodule import MNISTDataModule
from training.neurogenesis_lightning_module import NeurogenesisLightningModule


def build_simple_loader(batch_size: int, num_workers: int, data_dir: str):
    """Return train and val DataLoaders using torchvision MNIST directly."""
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    val_ds = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader


def main(use_simple: bool, batch_size: int, num_workers: int, data_dir: str):
    """Train the NeurogenesisLightningModule on MNIST, with optional simple loader."""
    pl.seed_everything(42, workers=True)
    globals()["test_zero_xes"] = 0
    use_simple = False
    # Data
    if use_simple:
        train_loader, val_loader = build_simple_loader(batch_size, num_workers, data_dir)
        dm = None
    else:
        dm = MNISTDataModule(
            batch_size=batch_size,
            num_workers=num_workers,
            data_dir=data_dir,
            classes=None,
        )
        dm.prepare_data()
        dm.setup()
        train_loader, val_loader = None, None

    # Model
    model = NeurogenesisLightningModule(
        input_dim=28 * 28,
        hidden_sizes=[1000, 500, 250, 30],
        activation="leaky_relu",
        activation_latent="identity",
        activation_last="sigmoid",
        thresholds=[0.05, 0.05, 0.05, 0.05],
        max_nodes=[200, 100, 60, 30],
        max_outliers=0.01,
        base_lr=1e-3,
        pretrain_epochs=15,
        plasticity_epochs=5,
        stability_epochs=2,
        next_layer_epochs=1,
    )

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        monitor="pretrain_loss",
        save_top_k=1,
        mode="min",
        filename="mnist-{epoch:02d}-{pretrain_loss:.4f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=model.hparams.pretrain_epochs,
        accelerator="auto",
        callbacks=[checkpoint_cb, lr_monitor],
        log_every_n_steps=50,
    )

    # Fit
    if use_simple:
        # Pass simple DataLoaders directly as positional args
        trainer.fit(model, train_loader, val_loader)
    else:
        # Use the LightningDataModule
        trainer.fit(model, dm)

    print("test_zero_xes", globals()["test_zero_xes"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AE on MNIST with optional loader")
    parser.add_argument(
        "--use_simple",
        action="store_true",
        help="Use simple MNIST DataLoader instead of DataModule",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_dir", type=str, default="data")
    args = parser.parse_args()
    main(args.use_simple, args.batch_size, args.num_workers, args.data_dir)
