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
        plasticity_epochs: int = 5,
        stability_epochs: int = 2,
        next_layer_epochs: int = 1,
    ):
        self.ae = ae
        self.ir = ir
        self.thresholds = thresholds
        self.max_nodes = max_nodes
        self.max_outliers = max_outliers
        self.base_lr = base_lr
        # epoch settings per phase
        self.plasticity_epochs = plasticity_epochs
        self.stability_epochs = stability_epochs
        self.next_layer_epochs = next_layer_epochs
        # History: class_id -> {'layer_errors': [Tensor snapshots]}
        self.history: dict[Any, dict[str, List[Tensor]]] = {}
        self.ae = ae
        self.ir = ir
        self.thresholds = thresholds
        self.max_nodes = max_nodes
        self.max_outliers = max_outliers
        self.base_lr = base_lr

    def _get_recon_errors(self, loader: DataLoader, level: int) -> Tensor:
        """
        Compute reconstruction errors at specified encoder level for all samples in loader.
        """
        errors = []
        for batch in loader:
            x = batch[0]
            x_hat = self.ae.forward_partial(x, level)
            errors.append(self.ae.reconstruction_error(x_hat, x))
        return torch.cat(errors)

    def learn_class(self, class_id: Any, loader: DataLoader) -> None:
        """
        Learn a new class with neurogenesis:
          - Fit intrinsic replay stats
          - For each layer:
            - Compute outliers > threshold
            - While too many outliers and under max_nodes:
              - Add neurons
              - Plasticity phase (new nodes)
              - Stability phase (with IR replay)
            - One plasticity epoch on next layer
        """
        # fit IR stats on new data
        self.ir.fit(loader)

        num_layers = len(self.ae.hidden_sizes)
        for level in range(num_layers):
            # compute errors and find outliers
            errs = self._get_recon_errors(loader, level)
            outliers = errs > self.thresholds[level]
            added = 0

            # iterative growth loop
            while outliers.sum() > self.max_outliers * len(errs) and added < self.max_nodes[level]:
                num_new = int(self.max_outliers * len(errs))
                self.ae.add_new_nodes(level, num_new)

                # plasticity phase on new nodes
                self.ae.plasticity_phase(
                    loader, level, epochs=self.plasticity_epochs, lr=self.base_lr
                )

                # stability phase with intrinsic replay
                self.ae.stability_phase(
                    loader,
                    lr=self.base_lr / 100,
                    epochs=self.stability_epochs,
                    ir=self.ir,
                    class_id=class_id,
                    replay_size=num_new,
                )

                # recompute and record
                errs = self._get_recon_errors(loader, level)
                self.history[class_id]["layer_errors"].append(errs.clone())
                outliers = errs > self.thresholds[level]
                added += 1

            # plasticity on next layer once
            if level + 1 < num_layers:
                self.ae.plasticity_phase(
                    loader, level + 1, epochs=self.next_layer_epochs, lr=self.base_lr / 100
                )
