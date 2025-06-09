from functools import partial
from typing import List

from torch import Tensor, nn

_ACTS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "leaky_relu": partial(nn.LeakyReLU, negative_slope=0.1),
    "sigmoid": nn.Sigmoid,
    "identity": lambda x: x,
}


class AutoEncoder(nn.Module):
    """Fully-connected auto-encoder with arbitrary depth & width."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: List[int],
        activation: str = "relu",
        activation_last: str = "sigmoid",
    ):
        super().__init__()
        if len(hidden_sizes) < 1:
            raise ValueError("Provide at least one hidden layer")

        act = _ACTS[activation]
        act_last = _ACTS[activation_last]
        # encoder
        enc_layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_sizes:
            enc_layers.append(nn.Linear(prev, h))
            enc_layers.append(act())
            prev = h
        self.encoder = nn.Sequential(*enc_layers)

        # decoder (mirror)
        dec_layers: list[nn.Module] = []
        for i, h in enumerate(reversed(hidden_sizes[:-1])):
            dec_layers.append(nn.Linear(prev, h))
            dec_layers.append(act())
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        dec_layers.append(act_last())

        self.decoder = nn.Sequential(*dec_layers)

        self.act = act
        self.input_dim = input_dim

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        x = x.view(x.size(0), -1)  # flatten
        z = self.encoder(x)
        recon = self.decoder(z)
        return {"recon": recon, "latent": z}
