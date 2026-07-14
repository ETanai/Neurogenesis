import copy

import torch
import torch.nn.functional as F

from models.ng_autoencoder import NGAutoEncoder
from training.predictive_coding_trainer import PredictiveCodingTrainer


def _trainer(*, plasticity_mode="uniform", inference_steps=5, inference_lr=0.05):
    torch.manual_seed(7)
    model = NGAutoEncoder(
        input_dim=8,
        hidden_sizes=[6, 3],
        activation="sigmoid",
        activation_latent="identity",
        activation_last="sigmoid",
    )
    return PredictiveCodingTrainer(
        ae=model,
        ir=None,
        base_lr=1.0e-3,
        epochs=1,
        device=torch.device("cpu"),
        plasticity_mode=plasticity_mode,
        inference_steps=inference_steps,
        inference_lr=inference_lr,
    )


def test_inference_clamps_endpoints_and_reduces_prediction_energy():
    trainer = _trainer(inference_steps=8, inference_lr=0.05)
    batch = torch.rand(5, 1, 2, 4)

    states, before, after = trainer.infer_states(batch)

    expected = batch.view(5, 8)
    assert torch.equal(states[0], expected)
    assert torch.equal(states[-1], expected)
    assert after < before


def test_local_predictive_coding_step_updates_every_predictor():
    trainer = _trainer()
    batch = torch.rand(5, 8)
    optimizer = trainer._ensure_optimizer()
    before = [pair[0].weight_mature.detach().clone() for pair in trainer._predictor_pairs()]

    result = trainer.train_batch(batch, optimizer)

    after = [pair[0].weight_mature.detach() for pair in trainer._predictor_pairs()]
    assert trainer.update_steps == 1
    assert result["energy_after"] < result["energy_before"]
    assert all(not torch.equal(old, new) for old, new in zip(before, after))


def test_usage_plasticity_prioritizes_underused_hidden_units_only():
    trainer = _trainer(plasticity_mode="usage")
    trainer._usage = [
        torch.tensor([0.01, 0.1, 0.2, 0.3, 0.4, 0.5]),
        torch.tensor([0.1, 0.2, 0.3]),
        torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        None,
    ]

    factors = trainer.plasticity_factors()

    assert factors[0][0] > factors[0][-1]
    assert torch.isclose(factors[0].mean(), torch.tensor(1.0), atol=0.15)
    assert torch.equal(factors[-1], torch.ones(1))


def test_predictive_coding_configuration_is_validated():
    try:
        _trainer(inference_steps=0)
    except ValueError as error:
        assert "inference_steps" in str(error)
    else:
        raise AssertionError("Expected inference_steps validation")


def test_layer_precisions_weight_local_energy():
    trainer = _trainer(inference_steps=1)
    batch = torch.rand(3, 8)
    with torch.no_grad():
        states = trainer.feedforward_states(batch)
    states[-1] = batch
    uniform = trainer.prediction_energy(states).item()
    trainer.layer_precisions[-1] = 4.0
    weighted = trainer.prediction_energy(states).item()
    assert weighted > uniform


def test_backprop_equivalent_mode_matches_direct_reconstruction_update():
    trainer = _trainer(inference_steps=1)
    reference = copy.deepcopy(trainer.ae)
    trainer.update_mode = "backprop_equivalent"
    batch = torch.rand(4, 8)

    pc_optimizer = torch.optim.Adam(trainer.ae.parameters(), lr=trainer.base_lr)
    ref_optimizer = torch.optim.Adam(reference.parameters(), lr=trainer.base_lr)
    trainer.train_batch(batch, pc_optimizer)
    ref_loss = F.mse_loss(reference(batch)["recon"], batch)
    ref_optimizer.zero_grad(set_to_none=True)
    ref_loss.backward()
    ref_optimizer.step()

    for actual, expected in zip(trainer.ae.parameters(), reference.parameters()):
        assert torch.equal(actual, expected)


def test_hybrid_consolidation_adds_backprop_update_steps():
    trainer = _trainer(inference_steps=1)
    trainer.consolidation_epochs = 1
    trainer.consolidation_lr_ratio = 2.0
    images = torch.rand(4, 8)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(images, torch.zeros(4, dtype=torch.long)),
        batch_size=2,
    )
    trainer.learn_class(0, loader)
    assert trainer.update_steps == 4
    assert len(trainer.diagnostics["class_summaries"]["0"]["consolidation_loss"]) == 1
