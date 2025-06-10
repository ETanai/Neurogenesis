from functools import partial
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from models.ng_linear import NGLinear  # <-- your custom layer

_ACTS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "leaky_relu": partial(nn.LeakyReLU, negative_slope=0.1),
    "sigmoid": nn.Sigmoid,
    "identity": lambda: nn.Identity(),
}


class NGAutoEncoder(nn.Module):
    """Auto‐encoder built with NGLinear (no new nodes by default)."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: List[int],
        activation: str = "relu",
        activation_last: str = "sigmoid",
    ):
        super().__init__()
        if not hidden_sizes:
            raise ValueError("Provide at least one hidden layer")

        self.input_dim = input_dim
        self.hidden_sizes = list(hidden_sizes)
        act_cls = _ACTS[activation]
        act_last_cls = _ACTS[activation_last]

        # encoder modules
        enc_layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in self.hidden_sizes:
            enc_layers.append(NGLinear(prev_dim, out_features_old=h, out_features_new=0))
            enc_layers.append(act_cls())
            prev_dim = h
        self.encoder = nn.Sequential(*enc_layers)

        # decoder modules (mirror)
        dec_layers: list[nn.Module] = []
        prev_dim = self.hidden_sizes[-1]
        for h in reversed(self.hidden_sizes[:-1]):
            dec_layers.append(NGLinear(prev_dim, out_features_old=h, out_features_new=0))
            dec_layers.append(act_cls())
            prev_dim = h
        dec_layers.append(NGLinear(prev_dim, out_features_old=input_dim, out_features_new=0))
        dec_layers.append(act_last_cls())
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        x_flat = x.view(x.size(0), -1)
        z = self.encoder(x_flat)
        recon = self.decoder(z)
        return {"recon": recon, "latent": z}

    def forward_partial(self, x: Tensor, layer_idx: int) -> Tensor:
        x_flat = x.view(x.size(0), -1)
        out = x_flat
        # encode up to layer_idx
        enc_end = 2 * layer_idx + 1
        for idx, module in enumerate(self.encoder):
            out = module(out)
            if idx == enc_end:
                break
        # decode from mirror layer
        n_layers = len(self.hidden_sizes)
        dec_start = 2 * (n_layers - 1 - layer_idx)
        for module in list(self.decoder)[dec_start:]:
            out = module(out)
        return out

    @staticmethod
    def reconstruction_error(x_hat: Tensor, x: Tensor) -> Tensor:
        x_flat = x.view(x.size(0), -1)
        err = F.mse_loss(x_hat, x_flat, reduction="none")
        return err.mean(dim=1)

    def set_requires_grad(self, freeze_old: bool = True) -> None:
        for name, param in self.named_parameters():
            if "weight_old" in name or "bias_old" in name:
                param.requires_grad = not freeze_old
            else:
                param.requires_grad = True

    def add_new_nodes(self, level_idx: int, num_new: int) -> None:
        # (existing implementation)
        enc_pos = 2 * level_idx
        enc_layer: NGLinear = self.encoder[enc_pos]  # type: ignore
        enc_layer.add_new_nodes(num_new)
        self.hidden_sizes[level_idx] += num_new
        if level_idx + 1 < len(self.hidden_sizes):
            next_enc: NGLinear = self.encoder[2 * (level_idx + 1)]  # type: ignore
            next_enc.adjust_input_size(num_new)
        # decoder mirror
        n_layers = len(self.hidden_sizes)
        dec_pos = 2 * (n_layers - 1 - level_idx)
        dec_layer: NGLinear = self.decoder[dec_pos]  # type: ignore
        dec_layer.adjust_input_size(num_new)
        if dec_pos < len(self.decoder) - 2:
            dec_layer.add_new_nodes(num_new)

    def plasticity_phase(
        self,
        data_loader: DataLoader,
        level_idx: int,
        epochs: int,
        lr: float,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Train new neurons at given layer on data_loader.
        Old neurons frozen; only new parameters updated.
        """
        device = device or next(self.parameters()).device
        self.to(device)
        self.set_requires_grad(freeze_old=True)
        # collect trainable params
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=lr)
        for _ in range(epochs):
            for x in data_loader:
                x = x.to(device)
                x_hat = self.forward_partial(x, level_idx)
                loss = F.mse_loss(x_hat, x.view(x.size(0), -1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def stability_phase(
        self,
        data_loader: DataLoader,
        lr: float,
        epochs: int,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Retrain all weights on combined data for stability.
        """
        device = device or next(self.parameters()).device
        self.to(device)
        self.set_requires_grad(freeze_old=False)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(epochs):
            for x in data_loader:
                x = x.to(device)
                recon = self.forward(x)["recon"]
                loss = F.mse_loss(recon, x.view(x.size(0), -1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
