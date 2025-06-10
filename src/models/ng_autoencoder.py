from functools import partial
from typing import List

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

        act_cls = _ACTS[activation]
        act_last_cls = _ACTS[activation_last]

        # encoder
        enc_layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_sizes:
            # out_features_new=0 by default, so NGLinear == nn.Linear
            enc_layers.append(NGLinear(prev_dim, out_features_old=h, out_features_new=0))
            enc_layers.append(act_cls())
            prev_dim = h
        self.encoder = nn.Sequential(*enc_layers)

        # decoder (mirror)
        dec_layers: list[nn.Module] = []
        # go through all but the last hidden size, reversed
        for h in reversed(hidden_sizes[:-1]):
            dec_layers.append(NGLinear(prev_dim, out_features_old=h, out_features_new=0))
            dec_layers.append(act_cls())
            prev_dim = h
        # final reconstruction layer back to input_dim
        dec_layers.append(NGLinear(prev_dim, out_features_old=input_dim, out_features_new=0))
        dec_layers.append(act_last_cls())

        self.decoder = nn.Sequential(*dec_layers)

        self.input_dim = input_dim

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        # flatten
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        recon = self.decoder(z)
        return {"recon": recon, "latent": z}
