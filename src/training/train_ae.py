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
from utils.intrinsic_replay import IntrinsicReplay  # << your helper from before


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    # 1) setup
    dm: MNISTDataModule = instantiate(cfg.datamodule)["datamodule"]
    model = AutoEncoder(
        input_dim=28 * 28,
        hidden_sizes=cfg.model.hidden_sizes,
        activation=cfg.model.activation,
    )
    optimizer = lambda params: optim.Adam(params, lr=cfg.model.lr)

    # 2) LightningModule
    class LitWrapper(pl.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.ae = model
            self.loss_fn = nn.MSELoss()

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
            self.log("val_loss", loss)

        def configure_optimizers(self):
            return optimizer(self.parameters())

        def on_validation_epoch_end(self) -> None:
            grid = torchvision.utils.make_grid(
                cat([self._last_val_imgs, self._last_val_recons], dim=0),
                nrow=self._last_val_imgs.size(0),
            )
            # TensorBoard
            self.logger.experiment.add_image(
                "val/example_inputs_vs_recon", grid, self.current_epoch
            )
            # clear
            self._last_val_imgs = None
            self._last_val_recons = None

    # 3) loggers & trainer
    tb_logger = TensorBoardLogger(save_dir="experiments/logs", name="mnist_ae")
    mlf_logger = MLFlowLogger(
        experiment_name="mnist_autoencoder",
        tracking_uri=cfg.mlflow.tracking_uri,  # e.g. "http://localhost:5000"
    )
    trainer = pl.Trainer(**cfg.trainer, logger=[tb_logger, mlf_logger])

    # 4) fit
    trainer.fit(LitWrapper(), dm)

    # 5) optional intrinsic‐replay
    if cfg.ir.enabled:
        # a) build IR on train set
        ir = IntrinsicReplay(
            encoder=model.encoder, decoder=model.decoder, device=trainer.strategy.root_device
        )
        ir.fit(dm.train_dataloader())

        # b) for each class, sample & log
        mlflow_client = mlf_logger.experiment
        run_id = mlf_logger.run_id

        for cls, stats in ir.stats.items():
            # sample a small grid of n x n
            n = cfg.ir.n_samples_per_class
            imgs = ir.sample_image_tensors(cls, n, view_shape=(1, 28, 28))
            grid = torchvision.utils.make_grid(imgs, nrow=int(n**0.5))

            # convert to uint8 PIL
            np_img = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            pil_img = Image.fromarray(np_img)

            # write out & log as artifact
            with tempfile.TemporaryDirectory() as td:
                path = f"{td}/ir_class_{cls}.png"
                pil_img.save(path)
                mlflow_client.log_artifact(
                    run_id=run_id,
                    artifact_path=f"ir_replay/class_{cls}",
                    local_path=path,
                )

        print(f"✅ intrinsic‐replay artifacts logged under run {run_id}")


if __name__ == "__main__":
    main()
