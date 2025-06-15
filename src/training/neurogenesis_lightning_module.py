import pytorch_lightning as pl
import torch.nn.functional as F
from torch import Tensor

from models.ng_autoencoder import NGAutoEncoder
from training.neurogenesis_trainer import NeurogenesisTrainer
from utils.intrinsic_replay import IntrinsicReplay
from utils.viz_utils import plot_recon_error_history, plot_recon_grid


class NeurogenesisLightningModule(pl.LightningModule):
    """
    LightningModule wrapping NGAutoEncoder and NeurogenesisTrainer.
    Handles pre-training, sequential neurogenesis, and logging to MLflow.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: list[int],
        activation: str,
        activation_last: str,
        thresholds: list[float],
        max_nodes: list[int],
        max_outliers: float,
        base_lr: float,
        plasticity_epochs: int,
        stability_epochs: int,
        next_layer_epochs: int,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Core components
        self.ae = NGAutoEncoder(
            input_dim=self.hparams.input_dim,
            hidden_sizes=self.hparams.hidden_sizes,
            activation=self.hparams.activation,
            activation_last=self.hparams.activation_last,
        )
        self.ir = IntrinsicReplay(self.ae.encoder, self.ae.decoder)
        self.trainer_ng = NeurogenesisTrainer(
            ae=self.ae,
            ir=self.ir,
            thresholds=self.hparams.thresholds,
            max_nodes=self.hparams.max_nodes,
            max_outliers=self.hparams.max_outliers,
            base_lr=self.hparams.base_lr,
            plasticity_epochs=self.hparams.plasticity_epochs,
            stability_epochs=self.hparams.stability_epochs,
            next_layer_epochs=self.hparams.next_layer_epochs,
        )
        self.class_loader = None  # set via set_class_loader

    def configure_optimizers(self):
        # Not used; internal optimizers handled by NeurogenesisTrainer
        return []

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, _ = batch
        # Pretrain AE: log MSE loss
        out = self.ae(x)
        loss = F.mse_loss(out["recon"], x.view(x.size(0), -1))
        self.log("pretrain_loss", loss, on_step=True, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        # After pretrain epochs, trigger neurogenesis if a class loader is set
        if self.class_loader is not None:
            class_id, loader = self.class_loader
            # Run the neurogenesis learning loop
            self.trainer_ng.learn_class(class_id, loader)
            # Log RE history and neuron counts
            for lvl, errs in enumerate(self.trainer_ng.history[class_id]["layer_errors"]):
                self.log(f"layer{lvl}_recon_error", errs.mean().item(), on_epoch=True)
            for lvl, count in enumerate(self.ae.hidden_sizes):
                self.log(f"layer{lvl}_neurons", count, on_epoch=True)

                # Now log artifact figures via MLflow, if logger available
            if hasattr(self, "logger") and self.logger is not None:
                run_id = self.logger.run_id
                # 1) Reconstruction-error history
                fig1 = plot_recon_error_history(self.trainer_ng, class_id)
                self.logger.experiment.log_figure(
                    run_id, fig1, f"{class_id}_recon_error_history.png"
                )
                # 2) Reconstruction grid (use first batch of loader)
                x_batch, _ = next(iter(loader))
                # derive view_shape assuming square images
                side = int(self.hparams.input_dim**0.5)
                fig2 = plot_recon_grid(self.ae, x_batch, view_shape=(1, side, side))
                self.logger.experiment.log_figure(run_id, fig2, f"{class_id}_recon_grid.png")
