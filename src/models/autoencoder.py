from typing import Callable, List

import torch
import torch.nn.functional as F
from torch import Tensor, nn

_ACTS: dict[str, Callable[[Tensor], Tensor]] = {
    "relu": F.relu,
    "tanh": torch.tanh,
    "leaky_relu": F.leaky_relu,
    "identity": lambda x: x,
}


class AutoEncoder(nn.Module):
    """Fully-connected auto-encoder with arbitrary depth & width."""

    def __init__(
        self, input_dim: int, hidden_sizes: List[int], activation: str = "relu"
    ):
        super().__init__()
        if len(hidden_sizes) < 1:
            raise ValueError("Provide at least one hidden layer")

        act = _ACTS[activation]
        # encoder
        enc_layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_sizes:
            enc_layers += (
                [nn.Linear(prev, h), nn.ReLU()]
                if activation == "relu"
                else [
                    nn.Linear(prev, h),
                    nn.Tanh(),
                ]
            )
            prev = h
        self.encoder = nn.Sequential(*enc_layers)

        # decoder (mirror)
        dec_layers: list[nn.Module] = []
        for h in reversed(hidden_sizes[:-1]):
            dec_layers += (
                [nn.Linear(prev, h), nn.ReLU()]
                if activation == "relu"
                else [
                    nn.Linear(prev, h),
                    nn.Tanh(),
                ]
            )
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

        self.act = act
        self.input_dim = input_dim

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        x = x.view(x.size(0), -1)  # flatten
        z = self.encoder(x)
        recon = self.decoder(z)
        return {"recon": recon, "latent": z}
