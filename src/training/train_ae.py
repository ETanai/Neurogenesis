import logging

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger
from torch import cat, nn, optim

from data.mnist_datamodule import MNISTDataModule
from models.autoencoder import AutoEncoder

from .intrinsic_replay_runner import run_intrinsic_replay

LOGGER = logging.getLogger(__name__)


class LitWrapper(pl.LightningModule):
    def __init__(
        self,
        ae: AutoEncoder | None = None,
        lr: float = 1e-3,
        input_dim: int = 28 * 28,
        hidden_sizes: list[int] | None = None,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.ae = ae or AutoEncoder(
            input_dim=input_dim,
            hidden_sizes=hidden_sizes or [64, 32],
            activation=activation,
        )
        self.lr = lr
        self.loss_fn = nn.MSELoss()
        self._last_val_imgs = None
        self._last_val_recons = None
        self._logger_override = None
        self._current_epoch_override = None

    @property
    def logger(self):
        return self._logger_override or super().logger

    @logger.setter
    def logger(self, value) -> None:
        self._logger_override = value

    @property
    def current_epoch(self):
        if self._current_epoch_override is not None:
            return self._current_epoch_override
        return super().current_epoch

    @current_epoch.setter
    def current_epoch(self, value) -> None:
        self._current_epoch_override = value

    def forward(self, x):
        return self.ae(x)

    def training_step(self, batch, batch_idx):
        imgs, _ = batch
        out = self(imgs)
        loss = self.loss_fn(out["recon"], imgs.view(imgs.size(0), -1))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, _ = batch
        out = self(imgs)
        self._last_val_imgs = imgs
        self._last_val_recons = out["recon"].view_as(imgs)
        loss = self.loss_fn(out["recon"], imgs.view(imgs.size(0), -1))
        if self._trainer is not None:
            self.log("val_loss", loss)
        LOGGER.info("val_loss: %s", loss.item())

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def validation_epoch_end(self, outputs=None) -> None:
        if self._last_val_imgs is None and hasattr(self, "_last_val_batch"):
            imgs, _ = self._last_val_batch
            out = self(imgs)
            self._last_val_imgs = imgs
            self._last_val_recons = out["recon"].view_as(imgs)

        if self._last_val_imgs is None or self._last_val_recons is None:
            return

        grid = cat(
            [
                cat([img, recon], dim=1)
                for img, recon in zip(self._last_val_imgs, self._last_val_recons)
            ],
            dim=2,
        )
        logger_target = getattr(self.logger, "experiment", self.logger)
        logger_target.add_image("val/example_inputs_vs_recon", grid, self.current_epoch)
        self._last_val_imgs = None
        self._last_val_recons = None

    def on_validation_epoch_end(self) -> None:
        self.validation_epoch_end()


def _run(cfg: DictConfig) -> None:
    # 1) setup
    dm: MNISTDataModule = instantiate(cfg.datamodule)["datamodule"]
    model = AutoEncoder(
        input_dim=28 * 28,
        hidden_sizes=cfg.model.hidden_sizes,
        activation=cfg.model.activation,
    )

    # 3) loggers & trainer
    tb_logger = TensorBoardLogger(save_dir="experiments/logs", name="mnist_ae")
    mlf_logger = MLFlowLogger(
        experiment_name="mnist_autoencoder",
        tracking_uri=cfg.mlflow.tracking_uri,  # e.g. "http://localhost:5000"
    )
    trainer = pl.Trainer(**cfg.trainer, logger=[tb_logger, mlf_logger])

    # 4) fit
    trainer.fit(LitWrapper(ae=model, lr=cfg.model.lr), dm)

    # 5) optional intrinsic‐replay
    if cfg.ir.enabled:
        run_intrinsic_replay(
            encoder=model.encoder,
            decoder=model.decoder,
            dataloader=dm.train_dataloader(),
            mlf_logger=mlf_logger,
            n_samples_per_class=cfg.ir.n_samples_per_class,
            device=trainer.strategy.root_device,
        )


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def _hydra_main(cfg: DictConfig) -> None:
    _run(cfg)


def main(cfg: DictConfig | None = None) -> None:
    if cfg is None:
        return _hydra_main()
    return main.__wrapped__(cfg)


main.__wrapped__ = _run


if __name__ == "__main__":
    main()
