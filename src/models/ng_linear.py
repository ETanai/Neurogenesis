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
    def add_plastic_nodes(self, num_new: int) -> Tuple[nn.Parameter, nn.Parameter]:
        """
        Append `num_new` trainable neurons to the plastic block and return them.
        """
        if num_new <= 0:
            raise ValueError("num_new must be positive")

        if self.weight_plastic is None:
            # first plastic block
            self.weight_plastic, self.bias_plastic = self._make_block(num_new, trainable=True)
        else:
            w_extra, b_extra = self._make_block(num_new, trainable=True)
            with torch.no_grad():
                self.weight_plastic = nn.Parameter(
                    torch.cat([self.weight_plastic, w_extra], dim=0), requires_grad=True
                )
                self.bias_plastic = nn.Parameter(
                    torch.cat([self.bias_plastic, b_extra], dim=0), requires_grad=True
                )
        return self.weight_plastic[-num_new:], self.bias_plastic[-num_new:]

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
        """
        Expand input feature count — grows every existing weight matrix.
        """
        if num_new_inputs <= 0:
            return

        self.in_features += num_new_inputs

        def _grow_weight(W: nn.Parameter) -> None:
            extra = torch.empty(W.size(0), num_new_inputs, dtype=W.dtype, device=W.device)
            nn.init.kaiming_uniform_(extra, a=math.sqrt(5))
            with torch.no_grad():
                W.data = torch.cat([W.data, extra], dim=1)

        # grow mature (always present)
        _grow_weight(self.weight_mature)

        # grow plastic block if it exists
        if self.weight_plastic is not None:
            _grow_weight(self.weight_plastic)
