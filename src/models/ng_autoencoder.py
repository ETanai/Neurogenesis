from functools import partial
from typing import List

import torch.nn.functional as F
from torch import Tensor, nn

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
        # go through all but the last hidden size, reversed
        for h in reversed(self.hidden_sizes[:-1]):
            dec_layers.append(NGLinear(prev_dim, out_features_old=h, out_features_new=0))
            dec_layers.append(act_cls())
            prev_dim = h
        # final reconstruction layer back to input_dim
        dec_layers.append(NGLinear(prev_dim, out_features_old=input_dim, out_features_new=0))
        dec_layers.append(act_last_cls())
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        # flatten
        x_flat = x.view(x.size(0), -1)
        z = self.encoder(x_flat)
        recon = self.decoder(z)
        return {"recon": recon, "latent": z}

    def forward_partial(self, x: Tensor, layer_idx: int) -> Tensor:
        """
        Encode input up to layer_idx, then decode back to input space.
        layer_idx: 0-based index into hidden_sizes
        """
        # flatten
        out = x.view(x.size(0), -1)
        # encode up to and including activation at layer_idx
        enc_end = 2 * layer_idx + 1
        for idx, module in enumerate(self.encoder):
            out = module(out)
            if idx == enc_end:
                break
        # decode from corresponding point
        n_layers = len(self.hidden_sizes)
        # compute decoder start index
        dec_start = 2 * (n_layers - 1 - layer_idx)
        for module in list(self.decoder)[dec_start:]:
            out = module(out)
        return out

    @staticmethod
    def reconstruction_error(x_hat: Tensor, x: Tensor) -> Tensor:
        """
        Per-sample MSE reconstruction error between x_hat and original x.
        Returns tensor of shape (batch,).
        """
        x_flat = x.view(x.size(0), -1)
        # per-element squared error, then mean per sample
        err = F.mse_loss(x_hat, x_flat, reduction="none")
        return err.mean(dim=1)
