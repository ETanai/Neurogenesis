import torch

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
