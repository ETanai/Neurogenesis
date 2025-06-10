import pytest
import torch
from torch import optim

from src.models.ng_linear import NGLinear


def test_initialization_and_forward():
    batch_size = 4
    in_features = 10
    out_old = 5

    layer = NGLinear(in_features=in_features, out_features_old=out_old, out_features_new=0)
    assert layer.in_features == in_features
    assert layer.out_features_old == out_old
    assert layer.out_features_new == 0
    assert layer.out_features == out_old

    # Forward pass with only old neurons
    x = torch.randn(batch_size, in_features)
    out = layer(x)
    assert out.shape == (batch_size, out_old)


def test_add_new_nodes_and_forward():
    batch_size = 3
    in_features = 8
    out_old = 4
    out_new = 2

    layer = NGLinear(in_features=in_features, out_features_old=out_old, out_features_new=0)
    layer.add_new_nodes(out_new)

    assert layer.out_features_new == out_new
    assert layer.out_features == out_old + out_new

    # Check parameter shapes
    assert layer.weight_new.shape == (out_new, in_features)
    assert layer.bias_new.shape == (out_new,)

    # Forward pass now returns old+new
    x = torch.randn(batch_size, in_features)
    out = layer(x)
    assert out.shape == (batch_size, out_old + out_new)


def test_adjust_input_size():
    batch_size = 2
    in_features = 6
    add_inputs = 3
    out_old = 5
    out_new = 4

    layer = NGLinear(in_features=in_features, out_features_old=out_old, out_features_new=out_new)

    layer.adjust_input_size(add_inputs)

    # in_features updated
    assert layer.in_features == in_features + add_inputs
    # weight_old should have grown in second dim
    assert layer.weight_old.shape == (out_old, in_features + add_inputs)
    # weight_new should have grown similarly
    assert layer.weight_new.shape == (out_new, in_features + add_inputs)

    # Forward still works
    x = torch.randn(batch_size, in_features + add_inputs)
    out = layer(x)
    assert out.shape == (batch_size, out_old + out_new)


def test_promote_new_to_old():
    batch_size = 3
    in_features = 7
    out_old = 3
    out_new = 5

    layer = NGLinear(in_features=in_features, out_features_old=out_old, out_features_new=0)
    layer.add_new_nodes(out_new)
    layer.promote_new_to_old()

    assert layer.out_features_old == out_old + out_new
    assert layer.out_features_new == 0
    # new parameters cleared
    assert layer.weight_new is None and layer.bias_new is None
    # weight_old shape reflects merged neurons
    assert layer.weight_old.shape == (out_old + out_new, in_features)

    # Forward now only uses weight_old
    x = torch.randn(batch_size, in_features)
    out = layer(x)
    assert out.shape == (batch_size, out_old + out_new)


@pytest.mark.parametrize("add_nodes_1, add_nodes_2", [(2, 3), (5, 1)])
def test_multiple_add_and_promote_cycles(add_nodes_1, add_nodes_2):
    in_features = 4
    out_old = 2
    layer = NGLinear(in_features=in_features, out_features_old=out_old, out_features_new=0)

    # first growth cycle
    layer.add_new_nodes(add_nodes_1)
    assert layer.out_features_new == add_nodes_1
    layer.promote_new_to_old()
    assert layer.out_features_old == out_old + add_nodes_1

    # second growth cycle
    layer.add_new_nodes(add_nodes_2)
    assert layer.out_features_new == add_nodes_2
    layer.promote_new_to_old()
    assert layer.out_features_old == out_old + add_nodes_1 + add_nodes_2


def test_optimizer_param_groups():
    base_lr = 1e-3
    layer = NGLinear(in_features=10, out_features_old=4, out_features_new=2)

    # Build the optimizer with two groups:
    opt = optim.Adam(
        [
            {"params": [layer.weight_old, layer.bias_old], "lr": base_lr * 0.01},
            {"params": [layer.weight_new, layer.bias_new], "lr": base_lr},
        ]
    )

    # Now we really should get exactly 2 groups:
    assert len(opt.param_groups) == 2

    old_group, new_group = opt.param_groups

    # Check the old-params group:
    assert old_group["lr"] == pytest.approx(base_lr * 0.01)
    assert set(old_group["params"]) == {layer.weight_old, layer.bias_old}

    # And the new-params group:
    assert new_group["lr"] == pytest.approx(base_lr)
    assert set(new_group["params"]) == {layer.weight_new, layer.bias_new}
