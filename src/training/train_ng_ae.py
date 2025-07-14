import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import MLFlowLogger

from data.mnist_datamodule import MNISTDataModule
from training.neurogenesis_lightning_module import NeurogenesisLightningModule


def build_mlflow_logger(cfg: DictConfig) -> MLFlowLogger:
    """
    Create MLFlow logger using Hydra config.
    """
    return MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name
        if "experiment_name" in cfg.mlflow
        else "neurogenesis",
        tracking_uri=cfg.mlflow.tracking_uri,
    )


@hydra.main(config_path="config", config_name="train")
def main(cfg: DictConfig):
    # print config for debugging
    print(OmegaConf.to_yaml(cfg))

    # Instantiate data module
    dm = MNISTDataModule(
        data_dir=cfg.datamodule.data_dir,
        batch_size=cfg.datamodule.batch_size,
        num_workers=cfg.datamodule.num_workers,
    )

    # Build LightningModule
    model = NeurogenesisLightningModule(
        input_dim=28 * 28,
        hidden_sizes=cfg.model.hidden_sizes,
        activation=cfg.model.activation,
        activation_last=cfg.model.activation_last,
        thresholds=cfg.neurogenesis.thresholds,
        max_nodes=cfg.neurogenesis.max_nodes,
        max_outliers=cfg.neurogenesis.max_outliers,
        base_lr=cfg.neurogenesis.base_lr,
        plasticity_epochs=cfg.neurogenesis.plasticity_epochs,
        stability_epochs=cfg.neurogenesis.stability_epochs,
        next_layer_epochs=cfg.neurogenesis.next_layer_epochs,
    )

    # Logger
    mlflow_logger = build_mlflow_logger(cfg)

    # Trainer
    trainer = pl.Trainer(
        logger=mlflow_logger,
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices if "devices" in cfg.trainer else None,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
    )

    # Pretrain autoencoder on initial classes
    if cfg.ir.enabled:
        # initial pretrain loader (e.g., classes specified in cfg)
        pretrain_loader = dm.train_dataloader()
        trainer.fit(model, train_dataloaders=pretrain_loader)

    # Sequential neurogenesis per class
    for class_id in cfg.ir.class_sequence:
        # prepare loader for this class only
        class_dataset = dm.get_class_dataset(class_id)
        class_loader = class_dataset.to_dataloader(batch_size=cfg.datamodule.batch_size)
        # set loader in LightningModule
        model.set_class_loader(class_id, class_loader)
        # trigger one epoch to run on_train_epoch_end
        trainer.fit(model)


if __name__ == "__main__":
    main()
