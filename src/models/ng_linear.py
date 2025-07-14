# src/models/ng_linear.py
import math
from typing import Iterable, Iterator, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class NGLinear(nn.Module):
    """
    Neurogenesis-ready fully-connected layer with *two* parameter groups:
        • mature (frozen)   : weight_mature, bias_mature
        • plastic (trainable): weight_plastic, bias_plastic

    `promote_plastic_to_mature()` moves the plastic block into the mature
    tensors and starts a fresh empty plastic block.

    External code can query the split via:
        layer.parameters_plastic()   -> Iterable[nn.Parameter]
        layer.parameters_mature()    -> Iterable[nn.Parameter]

    The ordinary .parameters() iterator continues to yield the union of both
    groups, so legacy code still works.
    """

    # ---------------- constructor ----------------
    def __init__(
        self,
        in_features: int,
        out_features_mature: int,
        out_features_plastic: int = 0,
        negative_slope: float = 0.01,
    ):
        super().__init__()
        self.in_features = in_features
        self.negative_slope = negative_slope
        self.n_out_features = out_features_mature + out_features_plastic
        self.out_features_mature = out_features_mature
        self.out_features_plastic = out_features_plastic
        # mature (initially frozen) tensors
        self.weight_mature = nn.Parameter(
            torch.empty(out_features_mature, in_features), requires_grad=True
        )
        self.bias_mature = nn.Parameter(torch.empty(out_features_mature), requires_grad=True)

        # single plastic block (may start empty)
        if out_features_plastic > 0:
            w_p, b_p = self._make_block(out_features_plastic, trainable=True)
        else:
            w_p = b_p = None
        self.weight_plastic: Optional[nn.Parameter] = w_p
        self.bias_plastic: Optional[nn.Parameter] = b_p

        self._reset_parameters()

    # -------------- public properties --------------
    @property
    def out_features(self) -> int:
        n_m = self.weight_mature.size(0)
        n_p = 0 if self.weight_plastic is None else self.weight_plastic.size(0)
        return n_m + n_p

    # -------------- forward --------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight_mature, self.bias_mature)
        if self.weight_plastic is not None:
            out_p = F.linear(x, self.weight_plastic, self.bias_plastic)
            out = torch.cat([out, out_p], dim=-1)
        return out

    # -------------- neurogenesis ops --------------
    def add_plastic_nodes(self, num_new: int) -> None:
        if num_new <= 0:
            raise ValueError("num_new must be positive")

        device = self.weight_mature.device
        in_f = self.in_features

        # 1) Create a new Parameter that pads the old plastic block
        if self.weight_plastic is None:
            old_w = torch.empty((0, in_f), device=device)
            old_b = torch.empty((0,), device=device)
        else:
            old_w = self.weight_plastic.data
            old_b = self.bias_plastic.data

        # new block
        new_w_block = torch.empty((num_new, in_f), device=device)
        new_b_block = torch.empty((num_new,), device=device)
        nn.init.kaiming_uniform_(new_w_block, a=math.sqrt(5))
        nn.init.zeros_(new_b_block)

        # concatenate & replace
        w_cat = torch.cat([old_w, new_w_block], dim=0)
        b_cat = torch.cat([old_b, new_b_block], dim=0)
        self.weight_plastic = nn.Parameter(w_cat, requires_grad=True)
        self.bias_plastic = nn.Parameter(b_cat, requires_grad=True)

        # update counters
        self.out_features_plastic += num_new
        self.n_out_features += num_new

    def promote_plastic_to_mature(self) -> None:
        """
        Consolidate plastic block into the mature (frozen) tensors
        and clear the plastic block (set to None).
        """
        if self.weight_plastic is None:
            return  # nothing to do

        with torch.no_grad():
            self.weight_mature = nn.Parameter(
                torch.cat([self.weight_mature, self.weight_plastic.detach()], dim=0),
                requires_grad=True,
            )
            self.bias_mature = nn.Parameter(
                torch.cat([self.bias_mature, self.bias_plastic.detach()], dim=0),
                requires_grad=True,
            )

        # drop the old plastic references
        self.weight_plastic = None
        self.bias_plastic = None
        self.out_features_mature += self.out_features_plastic
        self.out_features_plastic = 0

    # -------------- helpers for optimiser --------------
    def parameters_plastic(self) -> Iterable[nn.Parameter]:
        if self.weight_plastic is not None:
            yield self.weight_plastic
            yield self.bias_plastic

    def parameters_mature(self) -> Iterable[nn.Parameter]:
        yield self.weight_mature
        yield self.bias_mature

    # -------------- internal ----------------
    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight_mature, a=math.sqrt(5))
        nn.init.zeros_(self.bias_mature)
        if self.weight_plastic is not None:
            nn.init.kaiming_uniform_(self.weight_plastic, a=math.sqrt(5))
            nn.init.zeros_(self.bias_plastic)

    def _make_block(self, num_units: int, *, trainable: bool) -> Tuple[nn.Parameter, nn.Parameter]:
        device = next(self.parameters()).device
        w = nn.Parameter(
            torch.empty(num_units, self.in_features, device=device), requires_grad=trainable
        )
        b = nn.Parameter(torch.empty(num_units, device=device), requires_grad=trainable)
        return w, b

    # -------------- override .parameters --------------
    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:  # type: ignore[override]
        # Yield mature then plastic to preserve older ordering, but any order is fine.
        for p in self.parameters_mature():
            yield p
        for p in self.parameters_plastic():
            yield p

    def adjust_input_size(self, num_new_inputs: int) -> None:
        if num_new_inputs <= 0:
            return

        device = self.weight_mature.device
        old_in = self.in_features
        new_in = old_in + num_new_inputs
        self.in_features = new_in

        # helper to rebuild a block entirely
        def rebuild_block(old_param: nn.Parameter) -> nn.Parameter:
            old_data = old_param.data
            out_f, _ = old_data.shape
            extra = torch.empty((out_f, num_new_inputs), device=device)
            nn.init.kaiming_uniform_(extra, a=math.sqrt(5))
            new_data = torch.cat([old_data, extra], dim=1)
            return nn.Parameter(new_data, requires_grad=old_param.requires_grad)

        # 1) mature weights
        self.weight_mature = rebuild_block(self.weight_mature)
        # 2) plastic weights (if any)
        if self.weight_plastic is not None:
            self.weight_plastic = rebuild_block(self.weight_plastic)
