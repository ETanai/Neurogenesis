from typing import Any, List

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from models.ng_autoencoder import NGAutoEncoder
from utils.intrinsic_replay import IntrinsicReplay


class NeurogenesisTrainer:
    """
    Orchestrates sequential class learning with neurogenesis.
    Stores reconstruction-error history for analysis.
    """

    def __init__(
        self,
        ae: NGAutoEncoder,
        ir: IntrinsicReplay,
        thresholds: List[float],
        max_nodes: List[int],
        max_outliers: float,
        base_lr: float = 1e-3,
    ):
        self.ae = ae
        self.ir = ir
        self.thresholds = thresholds
        self.max_nodes = max_nodes
        self.max_outliers = max_outliers
        self.base_lr = base_lr
        # History: class_id -> {'layer_errors': [Tensor snapshots]}
        self.history: dict[Any, dict[str, List[Tensor]]] = {}

    def _get_recon_errors(self, loader: DataLoader, level: int) -> Tensor:
        errors = []
        for batch in loader:
            x = batch[0]
            x_hat = self.ae.forward_partial(x, level)
            errors.append(self.ae.reconstruction_error(x_hat, x))
        return torch.cat(errors)

    def learn_class(self, class_id: Any, loader: DataLoader) -> None:
        """
        Learn a new class with neurogenesis, recording RE history.
        """
        # fit IR stats
        self.ir.fit(loader)
        # init history
        self.history[class_id] = {"layer_errors": []}

        num_layers = len(self.ae.hidden_sizes)
        for level in range(num_layers):
            # compute initial errors
            errs = self._get_recon_errors(loader, level)
            self.history[class_id]["layer_errors"].append(errs.clone())
            outliers = errs > self.thresholds[level]
            added = 0

            # growth loop
            while outliers.sum() > self.max_outliers * len(errs) and added < self.max_nodes[level]:
                num_new = int(self.max_outliers * len(errs))
                self.ae.add_new_nodes(level, num_new)
                self.ae.plasticity_phase(loader, level, epochs=5, lr=self.base_lr)
                self.ae.stability_phase(
                    loader,
                    lr=self.base_lr / 100,
                    epochs=2,
                    ir=self.ir,
                    class_id=class_id,
                    replay_size=num_new,
                )
                # recompute and record
                errs = self._get_recon_errors(loader, level)
                self.history[class_id]["layer_errors"].append(errs.clone())
                outliers = errs > self.thresholds[level]
                added += 1

            # one plasticity on next layer
            if level + 1 < num_layers:
                self.ae.plasticity_phase(loader, level + 1, epochs=1, lr=self.base_lr / 100)
