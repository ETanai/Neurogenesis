from scripts.run_optimized_ndl_matched_controls import endpoint_by_seed, overrides_for


def test_promoted_endpoints_are_available_for_all_development_seeds():
    endpoints = endpoint_by_seed()
    assert set(endpoints) >= {45, 46, 47}
    assert all(len(endpoints[seed]) == 4 for seed in (45, 46, 47))


def test_matched_control_uses_exact_width_and_clean_replay():
    overrides = overrides_for(45, [208, 106, 80, 32])
    assert "experiment.control_hidden_sizes=[208,106,80,32]" in overrides
    assert "experiment.regime=cl_ir" in overrides
    assert "replay.mode=dataset" in overrides
    assert "model.activation=sigmoid" in overrides
