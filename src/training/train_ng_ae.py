import warnings

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import MLFlowLogger

from scripts.run_experiments import run as run_experiments


def build_mlflow_logger(cfg: DictConfig) -> MLFlowLogger:
    """
    Create MLFlow logger using Hydra config.
    """
    mlflow_cfg = cfg.mlflow if "mlflow" in cfg else cfg.logging.mlflow
    return MLFlowLogger(
        experiment_name=mlflow_cfg.experiment_name
        if "experiment_name" in mlflow_cfg
        else "neurogenesis",
        tracking_uri=mlflow_cfg.tracking_uri,
    )


@hydra.main(config_path="config", config_name="train", version_base=None)
def main(cfg: DictConfig):
    warnings.warn(
        "'training/train_ng_ae.py' is deprecated; use 'scripts/run_experiments.py' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    print(OmegaConf.to_yaml(cfg))
    run_experiments(cfg)


if __name__ == "__main__":
    main()
