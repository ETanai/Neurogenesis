from copy import deepcopy
from functools import partial
from typing import Callable, Dict, List, Optional, Union
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW

from models.ng_linear import NGLinear  # <-- your custom layer

class SharpSigmoid(nn.Module):
    """Sigmoid with adjustable slope parameter k."""

    def __init__(self, k: float = 5.0) -> None:
        super().__init__()
        self.k = k

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(self.k * x)


_ACTS: dict[str, Callable[..., nn.Module]] = {
    "relu": lambda **_: nn.ReLU(),
    "tanh": lambda **_: nn.Tanh(),
    "leaky_relu": lambda **cfg: nn.LeakyReLU(negative_slope=cfg.get("negative_slope", 0.1)),
    "sigmoid": lambda **_: nn.Sigmoid(),
    "identity": lambda **_: nn.Identity(),
    "sharp_sigmoid": lambda **cfg: SharpSigmoid(k=cfg.get("k", 5.0)),
}

EpochSummary = Dict[str, Union[int, float, str]]
ReplaySampler = Callable[[int], Optional[Tensor]]


class EarlyStopper:
    def __init__(
        self,
        min_delta: float = 0.0,
        patience: int = 1,
        mode: str = "min",
        goal: Optional[float] = None,
    ):
        assert mode in {"min", "max"}
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.best = float("inf") if mode == "min" else -float("inf")
        self.bad_epochs = 0
        self.should_stop = False
        self.goal = goal

    def step(self, current: float):
        if self.goal is not None:
            goal_reached = (
                current <= self.goal if self.mode == "min" else current >= self.goal
            )
            if goal_reached:
                self.should_stop = True
                return True
        improved = (
            (current < self.best - self.min_delta)
            if self.mode == "min"
            else (current > self.best + self.min_delta)
        )
        if improved:
            self.best = current
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self.should_stop = True
        return self.should_stop


class NGAutoEncoder(nn.Module):
    """Auto‐encoder built with NGLinear (neurogenesis-enabled)."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: List[int],
        activation: str = "relu",
        activation_params: Optional[dict] = None,
        activation_latent: str = "identity",
        activation_latent_params: Optional[dict] = None,
        activation_last: str = "sigmoid",
        activation_last_params: Optional[dict] = None,
    ):
        super().__init__()
        if not hidden_sizes:
            raise ValueError("Provide at least one hidden layer")

        self.input_dim = input_dim
        self.hidden_sizes = list(hidden_sizes)
        def _normalize(params: Optional[dict]) -> dict:
            if not params:
                return {}
            return {str(k): v for k, v in params.items()}

        act_params = _normalize(activation_params)
        act_lat_params = _normalize(activation_latent_params)
        act_last_params = _normalize(activation_last_params)
        if activation not in _ACTS:
            raise KeyError(f"Unknown activation '{activation}'. Available: {list(_ACTS)}")
        if activation_latent not in _ACTS:
            raise KeyError(f"Unknown activation '{activation_latent}'. Available: {list(_ACTS)}")
        if activation_last not in _ACTS:
            raise KeyError(f"Unknown activation '{activation_last}'. Available: {list(_ACTS)}")
        act_factory = _ACTS[activation]
        act_lat_factory = _ACTS[activation_latent]
        act_last_factory = _ACTS[activation_last]

        enc_layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in self.hidden_sizes[:-1]:
            enc_layers.append(NGLinear(prev_dim, out_features_mature=h, out_features_plastic=0))
            enc_layers.append(act_factory(**act_params))
            prev_dim = h
        enc_layers.append(
            NGLinear(prev_dim, out_features_mature=self.hidden_sizes[-1], out_features_plastic=0)
        )
        enc_layers.append(act_lat_factory(**act_lat_params))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers: list[nn.Module] = []
        prev_dim = self.hidden_sizes[-1]
        for h in reversed(self.hidden_sizes[:-1]):
            dec_layers.append(NGLinear(prev_dim, out_features_mature=h, out_features_plastic=0))
            dec_layers.append(act_factory(**act_params))
            prev_dim = h
        dec_layers.append(NGLinear(prev_dim, out_features_mature=input_dim, out_features_plastic=0))
        dec_layers.append(act_last_factory(**act_last_params))
        self.decoder = nn.Sequential(*dec_layers)
        self.update_steps = 0

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        x_flat = x.view(x.size(0), -1)
        z = self.encoder(x_flat)
        recon = self.decoder(z)
        return {"recon": recon, "latent": z}

    def parameters_plastic(self):
        for m in self.modules():
            if hasattr(m, "parameters_plastic"):
                yield from m.parameters_plastic()

    def parameters_mature(self):
        for m in self.modules():
            if hasattr(m, "parameters_mature"):
                yield from m.parameters_mature()

    def parameters_plastic_enc(self):
        for m in self.encoder.modules():
            if hasattr(m, "parameters_plastic"):
                yield from m.parameters_plastic()

    def parameters_mature_enc(self):
        for m in self.encoder.modules():
            if hasattr(m, "parameters_mature"):
                yield from m.parameters_mature()

    def parameters_plastic_dec(self):
        for m in self.decoder.modules():
            if hasattr(m, "parameters_plastic"):
                yield from m.parameters_plastic()

    def parameters_mature_dec(self):
        for m in self.decoder.modules():
            if hasattr(m, "parameters_mature"):
                yield from m.parameters_mature()

    def reset_update_counter(self) -> None:
        self.update_steps = 0

    def forward_partial(self, x: Tensor, layer_idx: int, ret_lat: bool = False) -> Tensor:
        x_flat = x.view(x.size(0), -1)
        # encode up to and including the activation after layer_idx
        cut_enc = 2 * layer_idx + 2  # each layer has [NGLinear, act]
        enc_modules = list(self.encoder[:cut_enc])
        out = x_flat
        for module in enc_modules:
            out = module(out)

        if ret_lat:
            lat = deepcopy(out)

        # decode from the corresponding mirror layer
        n = len(self.hidden_sizes)
        # skip the first 2*(n-1-layer_idx) modules in decoder
        cut_dec_start = 2 * (n - 1 - layer_idx)
        dec_modules = list(self.decoder[cut_dec_start:])
        for module in dec_modules:
            out = module(out)

        if ret_lat:
            return out, lat
        return out

    @staticmethod
    def reconstruction_error(x_hat: Tensor, x: Tensor) -> Tensor:
        x_flat = x.view(x.size(0), -1)
        err = F.mse_loss(x_hat, x_flat, reduction="none")
        return err.mean(dim=1)

    def set_requires_grad(self, freeze_old: bool = True) -> None:
        for name, param in self.named_parameters():
            if "weight_mature" in name or "bias_mature" in name:
                param.requires_grad = not freeze_old
            else:
                param.requires_grad = True

    # -------------------- helpers --------------------
    def _encoder_layer(self, level: int) -> NGLinear:
        # encoder is [lin, act, lin, act, …]  so 2*level
        return self.encoder[2 * level]

    def _decoder_layer(self, level: int) -> NGLinear:
        # mirror position inside decoder
        n = len(self.hidden_sizes)
        return self.decoder[2 * (n - 1 - level)]

    def _new_param_mask(self, module: nn.Module):
        """Return True for every Parameter that lives in a new-block list."""
        return [
            p
            for n, p in module.named_parameters()
            if "weight_new_blocks" in n or "bias_new_blocks" in n
        ]

    # -------------------------------------------------

    def add_new_nodes(self, level: int, num_new: int):
        enc = self._encoder_layer(level)
        enc.add_plastic_nodes(num_new)

        dec_mirror = self._decoder_layer(level)
        dec_mirror.adjust_input_size(num_new)

        if level + 1 < len(self.hidden_sizes):
            enc_next = self._encoder_layer(level + 1)
            enc_next.adjust_input_size(num_new)

        if level + 1 < len(self.hidden_sizes):
            dec_next = self._decoder_layer(level + 1)
            dec_next.add_plastic_nodes(num_new)

        self.hidden_sizes[level] += num_new

    def grid_recon(self, x: Tensor, view_shape: Optional[tuple[int, ...]] = None) -> Tensor:
        """
        Return a batch‐wise concatenation of input vs. reconstruction,
        ready to reshape back into images.

        Args:
            x:      (B, *) input batch (e.g. flattened images).
            view_shape: optional shape to view into (e.g. (1,28,28)).
        Returns:
            Tensor of shape (2*B, *) where even indices are originals,
            odds are their reconstructions.
        """
        # forward
        out = self.forward(x)
        recon = out["recon"]

        # stack originals and recon
        paired = torch.stack([x.view(x.size(0), -1), recon], dim=1)
        paired = paired.flatten(0, 1)  # now (2*B, features)

        if view_shape is not None:
            return paired.view(-1, *view_shape)
        return paired

    def assert_valid_structure(self):
        """
        Verifies that each Linear in encoder+decoder matches
        the sequence defined by [input_dim] + hidden_sizes + [input_dim].
        Raises AssertionError with details on any mismatch.
        """
        layersizes = []
        for layer in list(self.parameters()) + list(self.parameters()):
            if len(layer.shape) == 2:
                layersizes.append(layer.shape)
        for i in range((len(layersizes) - 1)):
            assert layersizes[i][0] == layersizes[i + 1][1]
        return True

    def _optim_plasticity(self, level: int, lr: float) -> AdamW:
        """
        Plasticity phase:
        – train only the new (plastic) nodes at full learning rate
        – freeze all mature (old) parameters
        """
        small_lr = lr / 100.0
        return self._optim_lr_config(
            level=level,
            lr_p=lr,  # new encoder nodes
            lr_m=0.0,  # freeze old encoder
            lr_p_d=small_lr,  # new decoder nodes
            lr_m_d=small_lr,  # old decoder nodes
        )

    def _optim_stability(self, level: int, lr: float) -> AdamW:
        """
        Stability phase:
        – fine‐tune all weights (plastic+mature) at lr/100
        """
        small_lr = lr / 100.0
        return self._optim_lr_config(
            level=level, lr_p=small_lr, lr_m=small_lr, lr_p_d=small_lr, lr_m_d=small_lr
        )

    def _optim_lr_config(
        self,
        level: int,
        lr_p: float,
        lr_m: Optional[float] = None,
        lr_p_d: Optional[float] = None,
        lr_m_d: Optional[float] = None,
    ) -> AdamW:
        """
        Configure AdamW optimizer at given depth `level`:
        - lr_p:     new encoder nodes
        - lr_m:     old encoder nodes
        - lr_p_d:   new decoder nodes
        - lr_m_d:   old decoder nodes
        """

        # default schedules
        lr_m = lr_m or (lr_p / 100.0)
        lr_p_d = lr_p_d or lr_p
        lr_m_d = lr_m_d or (lr_p / 100.0)

        # # pick the layer modules
        # enc = self._encoder_layer(level)
        # dec = self._decoder_layer(level)

        # collect disjoint param lists
        new_enc = [p for p in self.parameters_plastic_enc()]
        old_enc = [p for p in self.parameters_mature_enc()]

        new_dec = [p for p in self.parameters_plastic_dec()]
        old_dec = [p for p in self.parameters_mature_dec()]

        # freeze everything first
        for p in self.parameters():
            p.requires_grad_(False)

        # un-freeze exactly those we'll train
        for p in new_enc:
            p.requires_grad_(True)
        for p in old_enc:
            p.requires_grad_(lr_m != 0)
        for p in new_dec:
            p.requires_grad_(True)
        for p in old_dec:
            p.requires_grad_(lr_m_d != 0)

        # build optimizer groups, skip zero-lr groups
        param_groups = []
        if lr_p != 0:
            param_groups.append({"params": new_enc, "lr": lr_p})
        if lr_m != 0:
            param_groups.append({"params": old_enc, "lr": lr_m})
        if lr_p_d != 0:
            param_groups.append({"params": new_dec, "lr": lr_p_d})
        if lr_m_d != 0:
            param_groups.append({"params": old_dec, "lr": lr_m_d})

        return AdamW(param_groups)

    def plasticity_phase(
        self,
        loader,
        level: int,
        epochs: int,
        lr: float,
        early_stop_cfg: Optional[dict] = None,
        forward_fn: Optional[Callable[[Tensor], Tensor]] = None,
        epoch_logger: Optional[Callable[[int, EpochSummary], None]] = None,
    ):
        opt = self._optim_plasticity(level, lr)
        es = EarlyStopper(**early_stop_cfg) if early_stop_cfg else None
        return self._run_epoch_loop(
            loader,
            opt,
            epochs,
            early_stopper=es,
            forward_fn=forward_fn,
            epoch_logger=epoch_logger,
            loop_label="plasticity",
        )

    def stability_phase(
        self,
        loader,
        level: int,
        lr: float,
        epochs: int,
        old_x: Optional[Union[torch.Tensor, ReplaySampler]],
        early_stop_cfg: Optional[dict] = None,
        forward_fn: Optional[Callable[[Tensor], Tensor]] = None,
        epoch_logger: Optional[Callable[[int, EpochSummary], None]] = None,
    ):
        opt = self._optim_stability(level, lr)
        es = EarlyStopper(**early_stop_cfg) if early_stop_cfg else None
        return self._run_epoch_loop(
            loader,
            opt,
            epochs,
            replay=old_x,
            early_stopper=es,
            forward_fn=forward_fn,
            epoch_logger=epoch_logger,
            loop_label="stability",
        )

    def _run_epoch_loop(
        self,
        loader: torch.utils.data.DataLoader,
        optim: torch.optim.Optimizer,
        epochs: int,
        replay: Optional[Union[torch.Tensor, ReplaySampler]] = None,
        *,
        early_stopper: Optional[EarlyStopper] = None,
        forward_fn: Optional[Callable[[Tensor], Tensor]] = None,
        epoch_logger: Optional[Callable[[int, EpochSummary], None]] = None,
        loop_label: str = "train",
    ) -> Dict[str, list[float]]:
        """
        Generic train loop:
        • one forward/backward per batch
        • optional replay tensor concatenated to every mini-batch
        """
        from time import perf_counter

        from tqdm import tqdm

        device = next(self.parameters()).device
        self.train()

        history = {"epoch_loss": []}
        for epoch in range(epochs):
            batch_losses = []

            # accumulators for timings (seconds)
            t_data = 0.0
            t_concat = 0.0
            t_forward = 0.0
            t_loss = 0.0
            t_backward = 0.0
            t_step = 0.0
            n_batches = 0

            # batch-wise progress bar
            pbar = tqdm(loader, desc=f"Epoch_train {epoch + 1}/{epochs}", leave=False, unit="batch")
            for batch in pbar:
                # -- data transfer timing --
                t0 = perf_counter()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                x = batch[0].to(device, non_blocking=True)
                x = x.view(x.size(0), -1)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = perf_counter()
                t_data += t1 - t0

                # -- concat (replay) timing --
                t0 = perf_counter()
                if replay is not None:
                    k = int(x.size(0))
                    replay_batch: Optional[Tensor] = None
                    if callable(replay):
                        replay_batch = replay(k)
                    else:
                        pool = cast(torch.Tensor, replay)
                        if pool.size(0) > 0 and k > 0:
                            idx = torch.randint(0, pool.size(0), (k,), device=pool.device)
                            replay_batch = pool.index_select(0, idx)
                    if replay_batch is not None and replay_batch.numel() > 0:
                        if replay_batch.device != device:
                            replay_batch = replay_batch.to(device, non_blocking=True)
                        replay_batch = replay_batch.view(replay_batch.size(0), -1)
                        x = torch.cat([x, replay_batch], dim=0)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = perf_counter()
                t_concat += t1 - t0

                # -- forward timing --
                t0 = perf_counter()
                optim.zero_grad(set_to_none=True)
                if forward_fn is None:
                    out = self.forward(x)
                    x_hat = out["recon"] if isinstance(out, dict) else out
                else:
                    x_hat = forward_fn(x)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = perf_counter()
                t_forward += t1 - t0

                # -- loss timing --
                t0 = perf_counter()
                loss = self.reconstruction_error(x_hat, x).mean()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = perf_counter()
                t_loss += t1 - t0

                # -- backward timing --
                t0 = perf_counter()
                loss.backward()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = perf_counter()
                t_backward += t1 - t0

                # -- step timing --
                t0 = perf_counter()
                optim.step()
                self.update_steps += 1
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = perf_counter()
                t_step += t1 - t0

                batch_losses.append(loss.item())
                n_batches += 1

                # update tqdm with running stats
                avg_loss = float(torch.tensor(batch_losses).mean()) if batch_losses else 0.0
                pbar.set_postfix(
                    {
                        "loss": f"{avg_loss:.4f}",
                        "data(ms)": f"{(t_data / max(n_batches, 1)) * 1000:.1f}",
                        "fwd(ms)": f"{(t_forward / max(n_batches, 1)) * 1000:.1f}",
                    }
                )

            pbar.close()

            epoch_loss = float(torch.tensor(batch_losses).mean()) if batch_losses else 0.0
            history["epoch_loss"].append(epoch_loss)

            denom = max(n_batches, 1)
            summary: EpochSummary = {
                "epoch": epoch + 1,
                "loss": epoch_loss,
                # "avg_data_ms": (t_data / denom) * 1000.0,
                # "avg_concat_ms": (t_concat / denom) * 1000.0,
                # "avg_forward_ms": (t_forward / denom) * 1000.0,
                # "avg_loss_ms": (t_loss / denom) * 1000.0,
                # "avg_backward_ms": (t_backward / denom) * 1000.0,
                # "avg_step_ms": (t_step / denom) * 1000.0,
                "phase": loop_label,
            }

            # epoch timing summary
            # if n_batches > 0:
            #     tqdm.write(
            #         f"Epoch {epoch + 1} summary — loss: {summary['loss']:.4f} | "
            #         f"data {summary['avg_data_ms']:.1f}ms | concat {summary['avg_concat_ms']:.1f}ms | "
            #         f"fwd {summary['avg_forward_ms']:.1f}ms | back {summary['avg_backward_ms']:.1f}ms | step {summary['avg_step_ms']:.1f}ms"
            #     )
            if epoch_logger is not None:
                try:
                    epoch_logger(epoch, dict(summary))
                except Exception:
                    # keep training loop robust against logging issues
                    pass

            if early_stopper and early_stopper.step(epoch_loss):
                break
        return history

    def _plastic_to_mature(self):
        for module in self.encoder:
            if isinstance(module, NGLinear):
                module.promote_plastic_to_mature()
        for module in self.decoder:
            if isinstance(module, NGLinear):
                module.promote_plastic_to_mature()
