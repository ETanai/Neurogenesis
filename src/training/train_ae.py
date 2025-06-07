import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn, optim

from models.autoencoder import AutoEncoder
from src.data.mnist_datamodule import MNISTDataModule


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    dm: MNISTDataModule = instantiate(cfg.datamodule)
    model = AutoEncoder(
        input_dim=28 * 28,
        hidden_sizes=cfg.model.hidden_sizes,
        activation=cfg.model.activation,
    )
    optimizer = lambda params: optim.Adam(params, lr=cfg.model.lr)  # noqa: E731

    class LitWrapper(pl.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.ae = model
            self.loss_fn = nn.MSELoss()

        def forward(self, x):
            return self.ae(x)

        def training_step(self, batch, batch_idx):  # noqa: D401
            imgs, _ = batch
            out = self(imgs)
            loss = self.loss_fn(out["recon"], imgs.view(imgs.size(0), -1))
            self.log("train_loss", loss)
            return loss

        def validation_step(self, batch, batch_idx):  # noqa: D401
            imgs, _ = batch
            out = self(imgs)
            loss = self.loss_fn(out["recon"], imgs.view(imgs.size(0), -1))
            self.log("val_loss", loss)

        def configure_optimizers(self):
            return optimizer(self.parameters())

    tb_logger = TensorBoardLogger(save_dir="experiments/logs", name="mnist_ae")

    trainer = pl.Trainer(**cfg.trainer, logger=tb_logger)
    trainer.fit(LitWrapper(), dm["datamodule"])


if __name__ == "__main__":
    main()
