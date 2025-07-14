from __future__ import annotations

import tempfile

import torch
import torchvision
from PIL import Image
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader

from utils.intrinsic_replay import IntrinsicReplay


def run_intrinsic_replay(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    dataloader: DataLoader,
    mlf_logger: MLFlowLogger,
    *,
    n_samples_per_class: int = 16,
    device: torch.device | None = None,
) -> list:
    """Builds an :class:`IntrinsicReplay` from the given dataloader and logs
    sampled images to MLflow.

    Args:
        encoder: Trained encoder network.
        decoder: Decoder paired with the encoder.
        dataloader: Dataloader yielding ``(img, label)`` batches.
        mlf_logger: Logger to write replay images to.
        n_samples_per_class: How many images to sample per class.
        device: Device to perform computations on.
    """
    ir = IntrinsicReplay(encoder=encoder, decoder=decoder, device=device)
    ir.fit(dataloader)

    mlflow_client = mlf_logger.experiment
    run_id = mlf_logger.run_id

    imgs_ret = []

    for cls in ir.stats:
        imgs = ir.sample_image_tensors(cls, n_samples_per_class, view_shape=(1, 28, 28))
        imgs_ret.append(imgs)
        grid = torchvision.utils.make_grid(imgs, nrow=int(n_samples_per_class**0.5))

        np_img = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        pil_img = Image.fromarray(np_img)

        with tempfile.TemporaryDirectory() as td:
            path = f"{td}/ir_class_{cls}.png"
            pil_img.save(path)
            mlflow_client.log_artifact(
                run_id=run_id,
                artifact_path=f"ir_replay/class_{cls}",
                local_path=path,
            )

    print(f"✅ intrinsic-replay artifacts logged under run {run_id}")
    return imgs_ret
