import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.loggers import MLFlowLogger

# project modules (replace import path if you placed the file elsewhere)
from data.sd19_datamodule import SD19DataModule
from training.neurogenesis_lightning_module import NeurogenesisLightningModule
from training.neurogenesis_trainer import NeurogenesisTrainer


def build_sd19_class_sequences():
    """
    Returns:
        pretrain_classes: list[int] -> digits 0-9
        incremental_sequence: list[int] -> 26 uppercase then 26 lowercase
    Mapping assumption:
        0-9  : digits
        10-35: 'A'..'Z'
        36-61: 'a'..'z'
    Adjust if your label mapping differs.
    """
    digits = list(range(0, 10))
    uppercase = list(range(10, 36))
    lowercase = list(range(36, 62))
    return digits, uppercase + lowercase


if __name__ == "__main__":
    # Load (or merge) config – you may keep a separate YAML for SD19
    cfg = OmegaConf.load(
        "C:/Users/Admin/Documents/GitHub/Neurogenesis/config/notebook_config_sd19.yaml"
    )
    print(OmegaConf.to_yaml(cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Derive class sequences according to paper protocol
    pretrain_digits, incremental_sequence = build_sd19_class_sequences()

    # Allow override from config if desired
    if getattr(cfg.pretraining, "classes_pretraining", None):
        pretrain_digits = cfg.pretraining.classes_pretraining
    if getattr(cfg.ir, "class_sequence", None):
        incremental_sequence = cfg.ir.class_sequence

    # SD19 DataModule: expects separate train / val (or use one split twice if unavailable)
    dm = SD19DataModule(
        data_dir=cfg.datamodule.sd19_train_dir,
        batch_size=cfg.datamodule.batch_size,
        num_workers=cfg.datamodule.num_workers,
        classes=pretrain_digits,  # global filter for pretraining digits
        image_size=cfg.datamodule.image_size,
    )
    dm.prepare_data()  # no-op if already present
    dm.setup()

    # Preview batch
    batch = next(iter(dm.train_dataloader()))
    print("Sample batch shapes:", [x.shape for x in batch])

    # Model architecture per SD19 paper (50-dim latent instead of 30 for MNIST illustrative AE)
    # Supply through config or hardcode here; ensure cfg.model.hidden_sizes matches 1000-500-250-50-...
    model = NeurogenesisLightningModule(
        input_dim=28 * 28,
        hidden_sizes=cfg.model.hidden_sizes,  # e.g. [1000, 500, 250, 50, 250, 500, 1000]
        activation=cfg.model.activation,
        activation_latent=cfg.model.activation_latent,
        activation_last=cfg.model.activation_last,
        thresholds=cfg.neurogenesis.thresholds,  # may start as None
        max_nodes=cfg.neurogenesis.max_nodes,
        max_outliers=cfg.neurogenesis.max_outliers,
        base_lr=cfg.neurogenesis.base_lr,
        plasticity_epochs=cfg.neurogenesis.plasticity_epochs_max,
        stability_epochs=cfg.neurogenesis.stability_epochs,
        next_layer_epochs=cfg.neurogenesis.next_layer_epochs,
    ).to(device)

    logger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name, tracking_uri=cfg.mlflow.tracking_uri
    )

    if cfg.trainer.load_model_path is None:
        trainer = pl.Trainer(
            logger=logger,
            max_epochs=model.hparams.pretrain_epochs,
            accelerator="gpu" if device.type == "cuda" else "cpu",
            log_every_n_steps=cfg.trainer.log_every_n_steps,
            devices=1,
        )

        # Pretrain on digits (0–9) only
        trainer.fit(model, dm)

    else:
        sd = sd = torch.load(cfg.trainer.load_model_path, map_location="cpu")
        model.load_state_dict(sd)
        model.to(device)

    # Extract AE and IR for incremental neurogenesis phase
    ae = model.ae.to(device)
    ir = model.ir

    # Build NeurogenesisTrainer (thresholds initially None; we will estimate)
    trainer_ng = NeurogenesisTrainer(
        ae=ae,
        ir=ir,
        thresholds=None,
        max_nodes=cfg.neurogenesis.max_nodes,
        max_outliers=cfg.neurogenesis.max_outliers,
        base_lr=cfg.neurogenesis.base_lr,
        plasticity_epochs=cfg.neurogenesis.plasticity_epochs_max,
        stability_epochs=cfg.neurogenesis.stability_epochs,
        next_layer_epochs=cfg.neurogenesis.next_layer_epochs,
        factor_new_nodes=cfg.neurogenesis.factor_new_nodes,
        factor_max_new_nodes=cfg.neurogenesis.factor_new_nodes,
        logger=logger,
    )

    if (
        cfg.trainer.mean_layer_losses is not None
        and cfg.trainer.max_layer_losses is not None
        and cfg.trainer.std_layer_losses is not None
    ):
        mean_layer_losses = cfg.trainer.mean_layer_losses
        max_layer_losses = cfg.trainer.max_layer_losses
        std_layer_losses = cfg.trainer.std_layer_losses

    else:
        # Compute reconstruction stats across pretraining set (digits)
        (mean_layer_losses, max_layer_losses, std_layer_losses) = trainer_ng.test_all_levels(
            dm.get_combined_dataloader(pretrain_digits)
        )

    # Paper uses reconstruction error triggers; a practical heuristic: mean + k*std (k=3)
    thresholds = torch.tensor(mean_layer_losses) + 3.0 * torch.tensor(std_layer_losses)
    trainer_ng.thresholds = thresholds
    trainer_ng.mean_layer_losses = mean_layer_losses

    for i, t in enumerate(thresholds):
        trainer_ng.logger.log_metrics({"threshold_layer_{}".format(i): t.item()}, step=i)

    trainer_ng.log_global_sizes()
    # === Incremental phase: add uppercase then lowercase letters one class at a time ===
    for cls in incremental_sequence:
        # Switch DataModule sampler to the single class (no intrinsic replay yet)
        # vis_set = dm.get_class_dataset(list(range(cls + 1)))
        # fig, png = plot_recon_grid_mlflow(model, vis_set, return_mlflow_artifact=True)
        # temp_path = f"/temp/recon_grid_{png}.png"
        # with open(temp_path, "wb") as f:
        #     f.write(png)
        # logger.experiment.log_artifact(
        #     logger.run_id, local_path=temp_path, artifact_pathg="recon_gids"
        # )

        loader = dm.get_class_dataloader(cls)
        trainer_ng.learn_class(class_id=cls, loader=loader)

        print(f"After class {cls}, hidden sizes now: {ae.hidden_sizes}")

        # (Optional) Save checkpoint after each class
        if getattr(cfg.trainer, "checkpoint_each_class", False):
            ckpt_path = f"ae_after_class_{cls}.pth"
            torch.save(ae.state_dict(), ckpt_path)
            print(f"Saved {ckpt_path}")
