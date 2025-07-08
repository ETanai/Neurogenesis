import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.loggers import MLFlowLogger

# project modules
from data.mnist_datamodule import MNISTDataModule
from training.neurogenesis_lightning_module import NeurogenesisLightningModule
from training.neurogenesis_trainer import NeurogenesisTrainer
from utils.viz_utils import plot_recon_error_history, plot_recon_grid

if __name__ == "__main__":
    cfg = OmegaConf.load("C:/Users/Admin/Documents/GitHub/Neurogenesis/config/notebook_config.yaml")
    print(OmegaConf.to_yaml(cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = OmegaConf.load("C:/Users/Admin/Documents/GitHub/Neurogenesis/config/notebook_config.yaml")
    print(OmegaConf.to_yaml(cfg))

    dm = MNISTDataModule(
        data_dir=cfg.datamodule.data_dir,
        batch_size=cfg.datamodule.batch_size,
        num_workers=cfg.datamodule.num_workers,
        classes=cfg.datamodule.cls_pretraining,  # or list of classes, e.g. [1,7]
    )
    # Download and set up datasets
    dm.prepare_data()
    dm.setup()
    # preview batch shapes
    batch = next(iter(dm.train_dataloader()))
    print([x.shape for x in batch])

    cfg.trainer

    model = NeurogenesisLightningModule(
        input_dim=28 * 28,
        hidden_sizes=cfg.model.hidden_sizes,
        activation=cfg.model.activation,
        activation_latent=cfg.model.activation_latent,
        activation_last=cfg.model.activation_last,
        thresholds=cfg.neurogenesis.thresholds,
        max_nodes=cfg.neurogenesis.max_nodes,
        max_outliers=cfg.neurogenesis.max_outliers,
        base_lr=cfg.neurogenesis.base_lr,
        plasticity_epochs=cfg.neurogenesis.plasticity_epochs,
        stability_epochs=cfg.neurogenesis.stability_epochs,
        next_layer_epochs=cfg.neurogenesis.next_layer_epochs,
    )
    model.to(device)
    # Instantiate MLflow Logger directly
    logger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name, tracking_uri=cfg.mlflow.tracking_uri
    )
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=model.hparams.pretrain_epochs,
        accelerator="gpu" if device.type == "cuda" else "cpu",
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        devices=1,
    )

    trainer.fit(model, dm)

    # torch.save(model.ae.state_dict(), "Pretrained_200-100-60-20_1-2.pth")

    # # assuming you saved only the ae.state_dict()
    # state = torch.load("Pretrained_200-100-60-20_1-2.pth", map_location=device)
    # model.ae.load_state_dict(state)
    # model.to(device)
    # model.ae.to(device)

    # Extract the pretrained AE & IR
    ae = model.ae
    ir = model.ir

    # Build the NeurogenesisTrainer with the same hyperparams
    trainer_ng = NeurogenesisTrainer(
        ae=ae,
        ir=ir,
        thresholds=None,
        max_nodes=cfg.neurogenesis.max_nodes,
        max_outliers=cfg.neurogenesis.max_outliers,
        base_lr=cfg.neurogenesis.base_lr,
        plasticity_epochs=cfg.neurogenesis.plasticity_epochs,
        stability_epochs=cfg.neurogenesis.stability_epochs,
        next_layer_epochs=cfg.neurogenesis.next_layer_epochs,
        logger=logger,
    )

    threshoulds = (
        trainer_ng.test_all_levels(dm.get_combined_dataloader(cfg.datamodule.cls_pretraining)) * 1.5
    )
    trainer_ng.thresholds = threshoulds

    # Grow network one class at a time
    for cls in cfg.ir.class_sequence:
        loader = dm.get_class_dataloader(cls)
        trainer_ng.learn_class(class_id=cls, loader=loader)

        # Report and visualize growth
        print(f"After class {cls}, hidden sizes = {ae.hidden_sizes}")
        fig = plot_recon_error_history(trainer_ng, cls)
        fig.show()

        # Optional: show reconstructions for a batch
        batch = next(iter(loader))[0]
        grid = plot_recon_grid(ae, batch, view_shape=(1, 28, 28))
        grid.show()
