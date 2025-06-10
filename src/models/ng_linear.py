# src/models/ng_linear.py

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NGLinear(nn.Module):
    """
    A “growing” fully-connected layer for Neurogenesis:
      - keeps separate parameters for old vs. new neurons
      - supports adding neurons (add_new_nodes)
      - supports expanding input dimensionality (adjust_input_size)
      - supports merging new neurons into old (promote_new_to_old)
    """

    def __init__(
        self,
        in_features: int,
        out_features_old: int,
        out_features_new: int = 0,
        negative_slope: float = 0.01,
    ):
        """
        Args:
            in_features:  Number of input features.
            out_features_old:  Number of “frozen” (old) output neurons.
            out_features_new:  Number of “plastic” (new) output neurons to create.
            negative_slope:  If you ever use leaky_relu, pass its slope here.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features_old = out_features_old
        self.out_features_new = out_features_new
        self.negative_slope = negative_slope

        # Initialize parameters
        self.weight_old = nn.Parameter(torch.Tensor(out_features_old, in_features))
        self.bias_old = nn.Parameter(torch.Tensor(out_features_old))
        if out_features_new > 0:
            self.weight_new = nn.Parameter(torch.Tensor(out_features_new, in_features))
            self.bias_new = nn.Parameter(torch.Tensor(out_features_new))
        else:
            self.weight_new = None  # type: Optional[nn.Parameter]
            self.bias_new = None  # type: Optional[nn.Parameter]

        self._reset_parameters()

    @property
    def out_features(self) -> int:
        """Total number of output neurons (old + new)."""
        return self.out_features_old + self.out_features_new

    def _reset_parameters(self) -> None:
        """Init all weights and biases with Kaiming uniform + zero bias."""
        nn.init.kaiming_uniform_(self.weight_old, a=math.sqrt(5))
        nn.init.zeros_(self.bias_old)
        if self.weight_new is not None:
            nn.init.kaiming_uniform_(self.weight_new, a=math.sqrt(5))
            nn.init.zeros_(self.bias_new)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute linear output, concatenating old + new neurons if present.
        """
        out_old = F.linear(x, self.weight_old, self.bias_old)
        if self.weight_new is not None:
            out_new = F.linear(x, self.weight_new, self.bias_new)
            return torch.cat([out_old, out_new], dim=-1)
        return out_old

    def add_new_nodes(self, num_new: int) -> None:
        """
        Grow the layer by num_new neurons:
          - initialize their weights & biases
          - register them as `weight_new`/`bias_new` (or append if already exists)
        """
        w = nn.Parameter(torch.Tensor(num_new, self.in_features))
        b = nn.Parameter(torch.Tensor(num_new))
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        nn.init.zeros_(b)

        if self.weight_new is None:
            self.weight_new, self.bias_new = w, b
        else:
            # append to existing new parameters
            self.weight_new = nn.Parameter(torch.cat([self.weight_new, w], dim=0))
            self.bias_new = nn.Parameter(torch.cat([self.bias_new, b], dim=0))

        self.out_features_new += num_new

    def adjust_input_size(self, num_new_inputs: int) -> None:
        """
        Expand the number of input features by num_new_inputs:
          - grow both old & new weight tensors in their second dimension
        """
        # grow old weights
        new_w_old = torch.Tensor(self.out_features_old, num_new_inputs)
        nn.init.kaiming_uniform_(new_w_old, a=math.sqrt(5))
        self.weight_old = nn.Parameter(torch.cat([self.weight_old, new_w_old], dim=1))
        self.in_features += num_new_inputs

        # grow new weights if they exist
        if self.weight_new is not None:
            new_w_new = torch.Tensor(self.out_features_new, num_new_inputs)
            nn.init.kaiming_uniform_(new_w_new, a=math.sqrt(5))
            self.weight_new = nn.Parameter(torch.cat([self.weight_new, new_w_new], dim=1))

    def promote_new_to_old(self) -> None:
        """
        Merge new neurons into the “old” group (freeze them):
          - concatenate new weights/biases onto old
          - reset new parameters to None
          - update counts
        """
        if self.weight_new is not None:
            # merge weights & biases
            self.weight_old = nn.Parameter(torch.cat([self.weight_old, self.weight_new], dim=0))
            self.bias_old = nn.Parameter(torch.cat([self.bias_old, self.bias_new], dim=0))
            # clear new ones
            self.weight_new, self.bias_new = None, None
            # update counts
            self.out_features_old += self.out_features_new
            self.out_features_new = 0
