import tempfile

import hydra
import pytorch_lightning as pl
import torchvision
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger
from torch import cat, nn, optim

from data.mnist_datamodule import MNISTDataModule
from models.autoencoder import AutoEncoder


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

        def validation_step(self, batch, batch_idx):
            imgs, _ = batch
            out = self(imgs)
            # store for end-of-epoch logging
            self._last_val_imgs = imgs
            self._last_val_recons = out["recon"].view_as(imgs)
            loss = self.loss_fn(out["recon"], imgs.view(imgs.size(0), -1))
            self.log("val_loss", loss)

        def configure_optimizers(self):
            return optimizer(self.parameters())

        def on_validation_epoch_end(self) -> None:
            """Replaces old validation_epoch_end."""
            # build a grid of original vs. reconstructed
            grid = torchvision.utils.make_grid(
                cat([self._last_val_imgs, self._last_val_recons], dim=0),
                nrow=self._last_val_imgs.size(0),
            )
            # log it; TensorBoardLogger supports .add_image
            self.logger.experiment.add_image(
                "val/example_inputs_vs_recon", grid, self.current_epoch
            )
            # --- MLflow logging ---
            # Convert to a CPU PIL image
            np_img = grid.permute(1, 2, 0).cpu().numpy()  # H x W x C, in [0,1]
            # If it's in [0,1], scale to [0,255] uint8
            # img_uint8 = (np_img * 255).astype("uint8")
            # Log as a run-scoped image artifact under the key "val_reconstructions"
            # 1) Convert to uint8 PIL
            np_img = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            pil_img = Image.fromarray(np_img)

            # 2) Write out to a temp file
            with tempfile.TemporaryDirectory() as td:
                path = f"{td}/epoch_{self.current_epoch}.png"
                pil_img.save(path)

                # 3) Log that file under the 'val_reconstructions' artifact path
                mlflow_client = self.loggers[1].experiment  # <- this is an MlflowClient
                run_id = self.loggers[1].run_id  # <- the run that Lightning just started

                # Log the file into that exact run
                mlflow_client.log_artifact(
                    run_id=run_id,
                    artifact_path="val_reconstructions",  # folder under Artifacts
                    local_path=path,
                )

            # (optionally) clear stored batch
            self._last_val_imgs = None
            self._last_val_recons = None

    tb_logger = TensorBoardLogger(save_dir="experiments/logs", name="mnist_ae")
    mlf_logger = MLFlowLogger(
        experiment_name="mnist_autoencoder",
        tracking_uri="http://localhost:5000",  # local folder, or your remote MLflow server URI
    )

    trainer = pl.Trainer(**cfg.trainer, logger=[tb_logger, mlf_logger])
    trainer.fit(LitWrapper(), dm["datamodule"])


if __name__ == "__main__":
    main()
