import pytest
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset

from models.ng_autoencoder import NGAutoEncoder
from training.neurogenesis_trainer import NeurogenesisTrainer


class DummyAE:
    def __init__(self, hidden_sizes):
        self.hidden_sizes = hidden_sizes.copy()
        self.added = []
        self.plastic_calls = []
        self.stability_calls = []
        self.encoder = []

        class _Layer:
            def __init__(self, width):
                self.n_out_features = width

            def add_plastic_nodes(self, num_new):
                self.n_out_features += num_new

            def adjust_input_size(self, num_new):
                self.n_out_features += num_new

        self._layer_cls = _Layer
        for width in self.hidden_sizes:
            self.encoder.append(_Layer(width))
            self.encoder.append(object())

    def forward_partial(self, x, level):
        # return dummy reconstruction (same shape as x flattened)
        return x.view(x.size(0), -1)

    def forward_level_ae(self, x, level, ret_target=False):
        out = x.view(x.size(0), -1)
        return (out, out) if ret_target else out

    @staticmethod
    def reconstruction_error(x_hat, x):
        # return constant error of 1 for each sample
        return torch.ones(x.size(0))

    def add_new_nodes(self, level, num_new):
        self.added.append((level, num_new))
        # simulate growth of hidden_sizes
        self.hidden_sizes[level] += num_new
        layer = self.encoder[2 * level]
        if hasattr(layer, "add_plastic_nodes"):
            layer.add_plastic_nodes(num_new)

    def _plastic_to_mature(self):
        self.promotions = getattr(self, "promotions", 0) + 1

    def plasticity_phase(self, loader, level, epochs, lr, **kwargs):
        self.plastic_calls.append((level, epochs, lr))
        return {"epoch_loss": [1.0]}

    def stability_phase(
        self,
        loader,
        level,
        lr,
        epochs,
        old_x=None,
        replay_only=False,
        early_stop_cfg=None,
        forward_fn=None,
        epoch_logger=None,
        replay_interleave_batches=False,
    ):
        if callable(old_x):
            replay_descr = "callable"
        elif isinstance(old_x, torch.Tensor):
            replay_descr = f"tensor:{old_x.size(0)}"
        else:
            replay_descr = "none"
        self.stability_calls.append((level, epochs, lr, replay_descr, bool(replay_only)))
        self.stability_interleave_flags = getattr(self, "stability_interleave_flags", [])
        self.stability_interleave_flags.append(bool(replay_interleave_batches))
        return {"epoch_loss": [0.5]}


class DummyIR:
    def __init__(self):
        self.fitted = False
        self.fit_calls = 0
        self._classes: list[int] = []

    def fit(self, loader):
        self.fitted = True
        self.fit_calls += 1
        self._classes = [0]

    def available_classes(self):
        return self._classes

    def sample_images(self, class_id, replay_size):
        # return dummy samples of correct feature size
        # feature size doesn't matter; new_loader passes x only
        return torch.zeros(replay_size, 28 * 28)


class PaperReplayIR:
    def __init__(self, classes):
        self._classes = list(classes)
        self.calls = []

    def available_classes(self):
        return self._classes

    def sample_images_with_labels(self, class_id, replay_size):
        self.calls.append((int(class_id), int(replay_size)))
        labels = torch.full((int(replay_size),), int(class_id), dtype=torch.long)
        return torch.zeros(int(replay_size), 28 * 28), labels


@pytest.fixture
def dummy_loader():
    # 4 samples, 1 feature
    x = torch.randn(4, 1)
    y = torch.zeros(4, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=2)


def test_learn_class_calls(monkeypatch, dummy_loader):
    # Setup DummyAE with 2 layers
    ae = DummyAE(hidden_sizes=[1, 1])
    ir = DummyIR()
    # set thresholds low so reconstruction_error=1 > threshold
    thresholds = [0.5, 0.5]
    max_nodes = [2, 2]
    max_outliers = 0.5
    trainer = NeurogenesisTrainer(ae, ir, thresholds, max_nodes, max_outliers, base_lr=0.1)

    # Perform learning
    trainer.learn_class(class_id=7, loader=dummy_loader)

    # IR.fit should be called
    assert ir.fitted
    assert ir.fit_calls == 1

    # For each level we should attempt at least one growth with positive additions
    for level in range(len(max_nodes)):
        level_adds = [num for lvl, num in ae.added if lvl == level]
        assert len(level_adds) >= 1
        assert all(num > 0 for num in level_adds)

    # plasticity/stability phases should be invoked for every growth round
    assert len(ae.plastic_calls) >= len(ae.added)
    assert len(ae.stability_calls) >= len(ae.added)

    # Next-layer fine-tuning should use a reduced learning rate at least once
    assert any(lr < 0.01 for _, _, lr in ae.plastic_calls)


def test_get_recon_errors_simple(dummy_loader):
    # Test _get_recon_errors with real data and AE returning zeros
    # Monkeypatch AE
    class AEStub:
        hidden_sizes = [1]

        def forward_partial(self, x, level):
            return torch.zeros(x.size(0), 1)

        @staticmethod
        def reconstruction_error(x_hat, x):
            return torch.tensor([0.2, 0.2])

    ae = AEStub()
    from training.neurogenesis_trainer import NeurogenesisTrainer

    trainer = NeurogenesisTrainer(ae, None, thresholds=[0], max_nodes=[1], max_outliers=0)
    errs = trainer._get_recon_errors(dummy_loader, level=0)
    # loader has 4 samples, batch size 2, returns two batches of size2 with error 0.2 => length4
    assert torch.allclose(errs, torch.tensor([0.2, 0.2, 0.2, 0.2]))


def test_outlier_criterion_diagnostics_logs_pixel_local_overlap():
    class Logger:
        def __init__(self):
            self.metrics = {}

        def log_metrics(self, metrics, step=None):
            self.metrics.update(metrics)

    class AE:
        hidden_sizes = [1]

        def forward_partial(self, x, level):
            pixel = torch.tensor([0.1, 0.9, 0.3, 0.8], device=x.device)
            return pixel[x.long().view(-1)].view(-1, 1)

        def forward_level_ae(self, x, level, ret_target=False):
            local = torch.tensor([0.9, 0.1, 0.8, 0.2], device=x.device)
            out = local[x.long().view(-1)].view(-1, 1)
            target = torch.zeros_like(out)
            return (out, target) if ret_target else out

        @staticmethod
        def reconstruction_error(x_hat, x):
            return x_hat.view(x_hat.size(0), -1).mean(dim=1)

    logger = Logger()
    x = torch.arange(4, dtype=torch.float32).view(-1, 1)
    y = torch.zeros(4, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    trainer = NeurogenesisTrainer(
        AE(),
        None,
        thresholds=[0.5],
        max_nodes=[1],
        max_outliers=0.1,
        logger=logger,
        outlier_criterion_diagnostics={"enabled": True, "levels": [0]},
    )

    n_outliers, _, total = trainer._get_outliers(
        loader, level=0, class_id=7, iteration=3
    )

    assert n_outliers == 2
    assert total == 4
    prefix = "diagnostics/outlier_criterion/class_7/level_0/iteration_3"
    assert logger.metrics[f"{prefix}/pixel_outlier_fraction"] == pytest.approx(0.5)
    assert logger.metrics[f"{prefix}/local_topk_fraction"] == pytest.approx(0.5)
    assert logger.metrics[f"{prefix}/overlap_fraction_of_pixel"] == pytest.approx(0.0)
    assert logger.metrics[f"{prefix}/jaccard"] == pytest.approx(0.0)


def test_threshold_goal_config_injection():
    ae = DummyAE(hidden_sizes=[2, 2])
    trainer = NeurogenesisTrainer(
        ae,
        ir=None,
        thresholds=[0.25, 0.5],
        max_nodes=[1, 1],
        max_outliers=0.1,
        early_stop_cfg={
            "min_delta": 1e-4,
            "patience": 2,
            "mode": "min",
            "use_threshold_goal": True,
            "threshold_goal_factor": 1.2,
        },
    )
    cfg = trainer._build_phase_early_stop_cfg(0)
    assert cfg["goal"] == pytest.approx(0.25 * 1.2)
    assert "use_threshold_goal" not in cfg
    assert "threshold_goal_factor" not in cfg


def test_early_stop_phase_and_level_override_precedence():
    ae = DummyAE(hidden_sizes=[2, 2, 2, 2])
    trainer = NeurogenesisTrainer(
        ae,
        ir=None,
        thresholds=[0.25, 0.5, 0.75, 1.0],
        max_nodes=[1, 1, 1, 1],
        max_outliers=0.1,
        early_stop_cfg={
            "min_delta": 1e-4,
            "patience": 2,
            "mode": "min",
            "use_threshold_goal": True,
            "threshold_goal_factor": 1.0,
            "early_stop_by_phase": {
                "stability": {"min_delta": 3e-5, "patience": 3},
            },
            "early_stop_by_level": {
                "3": {"min_delta": 1e-5, "patience": 5},
            },
            "early_stop_by_phase_and_level": {
                "stability": {"3": {"min_delta": 1e-6, "patience": 10}},
            },
        },
    )

    cfg = trainer._build_phase_early_stop_cfg(3, phase="stability")

    assert cfg["min_delta"] == pytest.approx(1e-6)
    assert cfg["patience"] == 10
    assert cfg["goal"] == pytest.approx(1.0)
    assert "early_stop_by_phase" not in cfg
    assert "early_stop_by_level" not in cfg
    assert "early_stop_by_phase_and_level" not in cfg


def test_early_stop_level_override_applies_after_phase_override():
    ae = DummyAE(hidden_sizes=[2, 2, 2, 2])
    trainer = NeurogenesisTrainer(
        ae,
        ir=None,
        thresholds=[0.25, 0.5, 0.75, 1.0],
        max_nodes=[1, 1, 1, 1],
        max_outliers=0.1,
        early_stop_cfg={
            "min_delta": 1e-4,
            "patience": 2,
            "mode": "min",
            "early_stop_by_phase": {
                "plasticity": {"min_delta": 3e-5, "patience": 3},
            },
            "early_stop_by_level": {
                "2": {"patience": 5},
            },
        },
    )

    cfg = trainer._build_phase_early_stop_cfg(2, phase="plasticity")

    assert cfg["min_delta"] == pytest.approx(3e-5)
    assert cfg["patience"] == 5


def test_phase_new_sample_count_prefers_sampler_length():
    x = torch.randn(10, 1)
    y = torch.zeros(10, dtype=torch.long)
    sampler = SubsetRandomSampler([1, 3, 5])
    loader = DataLoader(TensorDataset(x, y), batch_size=2, sampler=sampler)

    assert NeurogenesisTrainer._phase_new_sample_count(loader, epochs_run=4) == 12


def test_default_growth_request_matches_proportional_policy():
    ae = DummyAE(hidden_sizes=[20])
    trainer = NeurogenesisTrainer(
        ae,
        ir=None,
        thresholds=[0.5],
        max_nodes=[10],
        max_outliers=1,
        factor_new_nodes=0.1,
        factor_max_new_nodes=0.1,
    )

    request = trainer._growth_request(
        level=0, n_outliers=23, nodes_existing=20, n_plastic_neurons=0
    )

    assert request["growth_mode"] == "proportional"
    assert request["requested_new_nodes"] == 3
    assert request["per_round_cap"] == 2
    assert request["actual_new_nodes"] == 2


def test_growth_request_uses_per_level_factor_overrides():
    ae = DummyAE(hidden_sizes=[20, 20])
    trainer = NeurogenesisTrainer(
        ae,
        ir=None,
        thresholds=[0.5, 0.5],
        max_nodes=[10, 10],
        max_outliers=1,
        factor_new_nodes=0.1,
        factor_max_new_nodes=0.1,
        factor_new_nodes_by_level={"1": 0.25},
        factor_max_new_nodes_by_level={"1": 0.5},
    )

    request = trainer._growth_request(
        level=1, n_outliers=20, nodes_existing=20, n_plastic_neurons=0
    )

    assert request["requested_new_nodes"] == 5
    assert request["per_round_cap"] == 10
    assert request["actual_new_nodes"] == 5
    assert request["factor_new_nodes_used"] == pytest.approx(0.25)
    assert request["factor_max_new_nodes_used"] == pytest.approx(0.5)


def test_absolute_growth_request_respects_remaining_max_nodes():
    ae = DummyAE(hidden_sizes=[20])
    trainer = NeurogenesisTrainer(
        ae,
        ir=None,
        thresholds=[0.5],
        max_nodes=[5],
        max_outliers=1,
        factor_max_new_nodes=1.0,
        growth_mode="absolute",
        absolute_new_nodes=3,
    )

    request = trainer._growth_request(
        level=0, n_outliers=200, nodes_existing=20, n_plastic_neurons=4
    )

    assert request["growth_mode"] == "absolute"
    assert request["requested_new_nodes"] == 3
    assert request["actual_new_nodes"] == 1
    assert request["remaining_new_nodes_before_growth"] == 1


def test_global_growth_budget_uses_growth_since_initial_architecture():
    ae = DummyAE(hidden_sizes=[24])
    trainer = NeurogenesisTrainer(
        ae,
        ir=None,
        thresholds=[0.5],
        max_nodes=[5],
        max_nodes_scope="global",
        initial_hidden_sizes=[20],
        max_outliers=1,
        factor_max_new_nodes=1.0,
        growth_mode="absolute",
        absolute_new_nodes=3,
    )

    request = trainer._growth_request(
        level=0, n_outliers=200, nodes_existing=24, n_plastic_neurons=0
    )

    assert request["remaining_new_nodes_before_growth"] == 1
    assert request["actual_new_nodes"] == 1


def test_global_growth_budget_is_exhausted_after_resume_size_reaches_cap():
    ae = DummyAE(hidden_sizes=[25])
    trainer = NeurogenesisTrainer(
        ae,
        ir=None,
        thresholds=[0.5],
        max_nodes=[5],
        max_nodes_scope="global",
        initial_hidden_sizes=[20],
        max_outliers=1,
    )

    assert trainer._growth_budget_remaining(0) == 0


def test_growth_mode_by_level_overrides_global_mode():
    ae = DummyAE(hidden_sizes=[20, 20, 20, 20])
    trainer = NeurogenesisTrainer(
        ae,
        ir=None,
        thresholds=[0.5, 0.5, 0.5, 0.5],
        max_nodes=[10, 10, 10, 10],
        max_outliers=1,
        factor_new_nodes=0.1,
        factor_max_new_nodes=1.0,
        growth_mode="proportional",
        growth_mode_by_level={"3": "absolute"},
        absolute_new_nodes_by_level={"3": 1},
    )

    lower = trainer._growth_request(
        level=2, n_outliers=20, nodes_existing=20, n_plastic_neurons=0
    )
    top = trainer._growth_request(
        level=3, n_outliers=20, nodes_existing=20, n_plastic_neurons=0
    )

    assert lower["growth_mode"] == "proportional"
    assert lower["actual_new_nodes"] == 2
    assert top["growth_mode"] == "absolute"
    assert top["actual_new_nodes"] == 1


def test_shape_pressure_scales_growth_when_layer_exceeds_target_ratio():
    ae = DummyAE(hidden_sizes=[100, 90])
    trainer = NeurogenesisTrainer(
        ae,
        ir=None,
        thresholds=[0.5, 0.5],
        max_nodes=[100, 100],
        max_outliers=1,
        factor_new_nodes=0.1,
        factor_max_new_nodes=1.0,
        shape_pressure_mode="scale_growth",
        shape_target_ratio_by_level={"1": 0.5},
        shape_min_growth_scale=0.25,
        shape_growth_scale_power=1.0,
    )

    request = trainer._growth_request(
        level=1, n_outliers=100, nodes_existing=90, n_plastic_neurons=0
    )

    assert request["shape_size_ratio"] == pytest.approx(0.9)
    assert request["shape_over_target"] == pytest.approx(1.8)
    assert request["growth_scale"] == pytest.approx(1 / 1.8)
    assert request["requested_new_nodes_before_shape"] == 10
    assert request["requested_new_nodes"] == 6
    assert request["actual_new_nodes"] == 6


def test_shape_pressure_gate_raises_required_outlier_count():
    ae = DummyAE(hidden_sizes=[100, 90])
    trainer = NeurogenesisTrainer(
        ae,
        ir=None,
        thresholds=[0.5, 0.5],
        max_nodes=[100, 100],
        max_outliers=0.1,
        shape_pressure_mode="scale_gate",
        shape_target_ratio_by_level={"1": 0.5},
        shape_gate_power=1.0,
        shape_max_gate_multiplier=10.0,
    )

    shape = trainer._shape_pressure(1)
    allowed = trainer._effective_max_outliers_allowed(
        level=1, total_seen=100, shape_info=shape
    )

    assert shape["shape_size_ratio"] == pytest.approx(0.9)
    assert shape["gate_multiplier"] == pytest.approx(1.8)
    assert allowed == 18


def test_max_outlier_fraction_by_level_overrides_global_fraction():
    ae = DummyAE(hidden_sizes=[100, 90])
    trainer = NeurogenesisTrainer(
        ae,
        ir=None,
        thresholds=[0.5, 0.5],
        max_nodes=[100, 100],
        max_outliers=0.1,
        max_outlier_fraction_by_level={"1": 0.25},
    )

    assert trainer._max_outliers_allowed(100, level=0) == 10
    assert trainer._max_outliers_allowed(100, level=1) == 25


def test_max_outliers_by_level_overrides_fraction_settings():
    ae = DummyAE(hidden_sizes=[100, 90])
    trainer = NeurogenesisTrainer(
        ae,
        ir=None,
        thresholds=[0.5, 0.5],
        max_nodes=[100, 100],
        max_outliers=0.1,
        max_outlier_fraction_by_level={"1": 0.25},
        max_outliers_by_level={"1": 7},
    )

    assert trainer._max_outliers_allowed(100, level=1) == 7


def test_shape_gate_multiplies_selected_per_level_quota():
    ae = DummyAE(hidden_sizes=[100, 90])
    trainer = NeurogenesisTrainer(
        ae,
        ir=None,
        thresholds=[0.5, 0.5],
        max_nodes=[100, 100],
        max_outliers=0.1,
        max_outlier_fraction_by_level={"1": 0.25},
        shape_pressure_mode="scale_gate",
        shape_target_ratio_by_level={"1": 0.5},
        shape_gate_power=1.0,
    )

    shape = trainer._shape_pressure(1)
    assert trainer._effective_max_outliers_allowed(
        level=1, total_seen=100, shape_info=shape
    ) == 45


def test_shape_pressure_has_no_effect_when_funnel_ratio_is_satisfied():
    ae = DummyAE(hidden_sizes=[100, 40])
    trainer = NeurogenesisTrainer(
        ae,
        ir=None,
        thresholds=[0.5, 0.5],
        max_nodes=[100, 100],
        max_outliers=0.1,
        factor_new_nodes=0.1,
        factor_max_new_nodes=1.0,
        shape_pressure_mode="scale_both",
        shape_target_ratio_by_level={"1": 0.5},
    )

    shape = trainer._shape_pressure(1)
    request = trainer._growth_request(
        level=1,
        n_outliers=100,
        nodes_existing=40,
        n_plastic_neurons=0,
        shape_info=shape,
    )

    assert shape["shape_size_ratio"] == pytest.approx(0.4)
    assert shape["growth_scale"] == pytest.approx(1.0)
    assert shape["gate_multiplier"] == pytest.approx(1.0)
    assert request["requested_new_nodes"] == 10


def test_quality_growth_gate_disabled_by_default_passes():
    ae = DummyAE(hidden_sizes=[100, 40])
    trainer = NeurogenesisTrainer(
        ae,
        ir=None,
        thresholds=[0.5, 0.25],
        max_nodes=[100, 100],
        max_outliers=0.1,
    )
    trainer._latest_outlier_stats[("0", 1, 0)] = {"pixel_mean": 0.01}

    info = trainer._quality_growth_gate(class_id=0, level=1, iteration=0)

    assert info["enabled"] is False
    assert info["quality_passes"] is True
    assert info["required_mean_error"] == pytest.approx(0.25)


def test_quality_growth_gate_blocks_when_mean_error_is_below_threshold_factor():
    ae = DummyAE(hidden_sizes=[100, 40])
    trainer = NeurogenesisTrainer(
        ae,
        ir=None,
        thresholds=[0.5, 0.25],
        max_nodes=[100, 100],
        max_outliers=0.1,
        quality_growth_gate={"enabled": True, "levels": [1], "threshold_factor": 1.0},
    )
    trainer._latest_outlier_stats[("2", 1, 3)] = {"pixel_mean": 0.2}

    info = trainer._quality_growth_gate(class_id=2, level=1, iteration=3)

    assert info["enabled"] is True
    assert info["mean_error"] == pytest.approx(0.2)
    assert info["required_mean_error"] == pytest.approx(0.25)
    assert info["quality_passes"] is False


def test_quality_growth_gate_uses_per_level_factor_override():
    ae = DummyAE(hidden_sizes=[100, 40])
    trainer = NeurogenesisTrainer(
        ae,
        ir=None,
        thresholds=[0.5, 0.25],
        max_nodes=[100, 100],
        max_outliers=0.1,
        quality_growth_gate={
            "enabled": True,
            "levels": [1],
            "threshold_factor": 1.0,
            "threshold_factor_by_level": {"1": 0.5},
        },
    )
    trainer._latest_outlier_stats[("2", 1, 3)] = {"pixel_mean": 0.2}

    info = trainer._quality_growth_gate(class_id=2, level=1, iteration=3)

    assert info["required_mean_error"] == pytest.approx(0.125)
    assert info["quality_passes"] is True


def test_adaptive_outlier_threshold_disabled_by_default_uses_base_threshold():
    ae = DummyAE(hidden_sizes=[2])
    trainer = NeurogenesisTrainer(
        ae,
        ir=None,
        thresholds=[0.5],
        max_nodes=[1],
        max_outliers=0.1,
    )
    errors = torch.tensor([0.1, 0.2, 0.9])

    info = trainer._effective_outlier_threshold(errors=errors, level=0, iteration=5)

    assert info["enabled"] is False
    assert info["active"] is False
    assert info["threshold"] == pytest.approx(0.5)


def test_adaptive_outlier_threshold_activates_after_min_iteration():
    ae = DummyAE(hidden_sizes=[2, 2])
    trainer = NeurogenesisTrainer(
        ae,
        ir=None,
        thresholds=[0.25, 0.5],
        max_nodes=[1, 1],
        max_outliers=0.1,
        adaptive_outlier_threshold={
            "enabled": True,
            "levels": [1],
            "percentile": 0.5,
            "min_iteration": 1,
        },
    )
    errors = torch.tensor([0.1, 0.2, 0.9])

    before = trainer._effective_outlier_threshold(errors=errors, level=1, iteration=0)
    after = trainer._effective_outlier_threshold(errors=errors, level=1, iteration=1)

    assert before["active"] is False
    assert before["threshold"] == pytest.approx(0.5)
    assert after["active"] is True
    assert after["adaptive_threshold"] == pytest.approx(0.2)
    assert after["threshold"] == pytest.approx(0.5)


def test_get_outliers_uses_adaptive_threshold_when_enabled():
    class Logger:
        def __init__(self):
            self.metrics = {}

        def log_metrics(self, metrics, step=None):
            self.metrics.update(metrics)

    class AE:
        hidden_sizes = [1]

        def forward_partial(self, x, level):
            values = torch.tensor([0.1, 0.2, 0.3, 0.9], device=x.device)
            return values[x.long().view(-1)].view(-1, 1)

        @staticmethod
        def reconstruction_error(x_hat, x):
            return x_hat.view(x_hat.size(0), -1).mean(dim=1)

    logger = Logger()
    x = torch.arange(4, dtype=torch.float32).view(-1, 1)
    y = torch.zeros(4, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    trainer = NeurogenesisTrainer(
        AE(),
        None,
        thresholds=[0.15],
        max_nodes=[1],
        max_outliers=0.1,
        logger=logger,
        adaptive_outlier_threshold={
            "enabled": True,
            "levels": [0],
            "percentile": 0.5,
            "min_iteration": 1,
            "mode": "quantile",
        },
    )

    n_initial, _, _ = trainer._get_outliers(loader, level=0, class_id=2, iteration=0)
    n_adapted, _, _ = trainer._get_outliers(loader, level=0, class_id=2, iteration=1)

    assert n_initial == 3
    assert n_adapted == 2
    prefix = "diagnostics/adaptive_threshold/class_2/level_0/iteration_1"
    assert logger.metrics[f"{prefix}/active"] == pytest.approx(1.0)
    assert logger.metrics[f"{prefix}/threshold"] == pytest.approx(0.25)


if __name__ == "__main__":
    pytest.main()


def test_objective_mode_forward_selection():
    ae = DummyAE(hidden_sizes=[2, 2])

    paper = NeurogenesisTrainer(ae, None, thresholds=[0.5, 0.5], max_nodes=[1, 1], max_outliers=1)
    assert paper.objective_mode == "paper_level_ae"
    assert paper._phase_forward_fn(0, "plasticity") is not None
    assert paper._phase_forward_fn(0, "stability") is not None
    out, target = paper._phase_forward_fn(0, "plasticity")(torch.randn(2, 3))
    assert out.shape == target.shape

    historical = NeurogenesisTrainer(
        ae,
        None,
        thresholds=[0.5, 0.5],
        max_nodes=[1, 1],
        max_outliers=1,
        objective_mode="global_partial",
    )
    assert historical._phase_forward_fn(0, "plasticity") is not None
    assert isinstance(historical._phase_forward_fn(0, "plasticity")(torch.randn(2, 3)), torch.Tensor)

    full = NeurogenesisTrainer(
        ae,
        None,
        thresholds=[0.5, 0.5],
        max_nodes=[1, 1],
        max_outliers=1,
        objective_mode="full_reconstruction",
    )
    assert full._phase_forward_fn(0, "plasticity") is None
    assert full._phase_forward_fn(0, "stability") is None

    mixed = NeurogenesisTrainer(
        ae,
        None,
        thresholds=[0.5, 0.5],
        max_nodes=[1, 1],
        max_outliers=1,
        objective_mode="local_plasticity_full_stability",
    )
    assert mixed._phase_forward_fn(0, "plasticity") is not None
    assert mixed._phase_forward_fn(0, "stability") is None


def test_invalid_objective_mode_rejected():
    with pytest.raises(ValueError):
        NeurogenesisTrainer(
            DummyAE(hidden_sizes=[2]),
            None,
            thresholds=[0.5],
            max_nodes=[1],
            max_outliers=1,
            objective_mode="bogus",
        )


def test_lr_ratio_knobs_are_attached_to_autoencoder():
    ae = DummyAE(hidden_sizes=[2])
    trainer = NeurogenesisTrainer(
        ae,
        None,
        thresholds=[0.5],
        max_nodes=[1],
        max_outliers=1,
        plasticity_decoder_lr_ratio=0.25,
        stability_lr_ratio=0.5,
        next_layer_lr_ratio=0.75,
        next_layer_optimization="paper_columns",
    )

    assert trainer.plasticity_decoder_lr_ratio == 0.25
    assert trainer.stability_lr_ratio == 0.5
    assert trainer.next_layer_lr_ratio == 0.75
    assert trainer.next_layer_optimization == "paper_columns"
    assert ae.plasticity_decoder_lr_ratio == 0.25
    assert ae.stability_lr_ratio == 0.5
    assert ae.next_layer_optimization == "paper_columns"


def test_invalid_next_layer_optimization_rejected():
    with pytest.raises(ValueError):
        NeurogenesisTrainer(
            DummyAE(hidden_sizes=[2]),
            None,
            thresholds=[0.5],
            max_nodes=[1],
            max_outliers=1,
            next_layer_optimization="too_broad",
        )


def test_paper_replay_mode_samples_each_old_class():
    ae = DummyAE(hidden_sizes=[2])
    ir = PaperReplayIR(classes=[1, 7])
    trainer = NeurogenesisTrainer(
        ae,
        ir,
        thresholds=[0.5],
        max_nodes=[1],
        max_outliers=1,
        stability_replay_mode="paper",
        stability_replay_per_class_ratio=0.5,
    )

    sampler, replay_only = trainer._build_replay_sampler(torch.device("cpu"), n_old_classes=2)
    replay = sampler(4)

    assert replay_only is False
    assert replay.shape == (4, 28 * 28)
    assert ir.calls == [(1, 2), (7, 2)]
    assert trainer._snapshot_replay_counters() == {
        "samples": 4,
        "by_class": {1: 2, 7: 2},
    }


def test_paper_replay_mode_respects_total_old_limit():
    ae = DummyAE(hidden_sizes=[2])
    ir = PaperReplayIR(classes=[1, 7])
    trainer = NeurogenesisTrainer(
        ae,
        ir,
        thresholds=[0.5],
        max_nodes=[1],
        max_outliers=1,
        replay_old_limit=3,
        stability_replay_mode="paper",
        stability_replay_per_class_ratio=1.0,
    )

    sampler, _ = trainer._build_replay_sampler(torch.device("cpu"), n_old_classes=2)
    replay = sampler(4)
    replay_after_limit = sampler(4)

    assert replay.shape == (3, 28 * 28)
    assert replay_after_limit is None
    assert ir.calls == [(1, 2), (7, 1)]


def test_paper_replay_class_weights_redistribute_fixed_quota():
    ae = DummyAE(hidden_sizes=[2])
    ir = PaperReplayIR(classes=[1, 7])
    trainer = NeurogenesisTrainer(
        ae,
        ir,
        thresholds=[0.5],
        max_nodes=[1],
        max_outliers=1,
        stability_replay_mode="paper",
        stability_replay_per_class_ratio=1.0,
        stability_replay_class_weights={"1": 1.0, "7": 3.0},
    )

    sampler, _ = trainer._build_replay_sampler(torch.device("cpu"), n_old_classes=2)
    replay = sampler(4)

    assert replay.shape == (8, 28 * 28)
    assert ir.calls == [(1, 2), (7, 6)]
    assert trainer._snapshot_replay_counters() == {
        "samples": 8,
        "by_class": {1: 2, 7: 6},
    }


def test_paper_replay_class_weights_respect_total_old_limit():
    ae = DummyAE(hidden_sizes=[2])
    ir = PaperReplayIR(classes=[1, 7])
    trainer = NeurogenesisTrainer(
        ae,
        ir,
        thresholds=[0.5],
        max_nodes=[1],
        max_outliers=1,
        replay_old_limit=5,
        stability_replay_mode="paper",
        stability_replay_per_class_ratio=1.0,
        stability_replay_class_weights={7: 3.0},
    )

    sampler, _ = trainer._build_replay_sampler(torch.device("cpu"), n_old_classes=2)
    replay = sampler(4)

    assert replay.shape == (5, 28 * 28)
    assert ir.calls == [(1, 1), (7, 4)]


def test_global_coupling_disabled_by_default():
    ae = NGAutoEncoder(
        input_dim=4, hidden_sizes=[3, 2], activation="identity", activation_last="identity"
    )
    trainer = NeurogenesisTrainer(
        ae,
        None,
        thresholds=[0.5, 0.5],
        max_nodes=[1, 1],
        max_outliers=1,
    )

    assert trainer.global_coupling_cfg["enabled"] is False
    assert trainer._global_coupling_enabled("after_class") is False


def test_global_coupling_optimizer_scopes_freeze_expected_parameters():
    ae = NGAutoEncoder(
        input_dim=4, hidden_sizes=[3, 2], activation="identity", activation_last="identity"
    )
    ae.add_new_nodes(0, 1)
    trainer = NeurogenesisTrainer(
        ae,
        None,
        thresholds=[0.5, 0.5],
        max_nodes=[1, 1],
        max_outliers=1,
        global_coupling_cfg={
            "enabled": True,
            "trigger": "after_class",
            "scope": "freeze_old_encoder",
        },
    )

    trainer._make_global_coupling_optimizer(1e-3)

    enc0 = ae._encoder_layer(0)
    assert all(not param.requires_grad for param in enc0.parameters_mature())
    assert all(param.requires_grad for param in enc0.parameters_plastic())
    assert any(param.requires_grad for module in ae.decoder for param in module.parameters())

    trainer.global_coupling_cfg["scope"] = "decoder_only"
    trainer._make_global_coupling_optimizer(1e-3)

    assert all(not param.requires_grad for module in ae.encoder for param in module.parameters())
    assert all(param.requires_grad for module in ae.decoder for param in module.parameters())


def test_global_coupling_uses_full_reconstruction_and_replay_sampler():
    ae = NGAutoEncoder(
        input_dim=4, hidden_sizes=[3], activation="identity", activation_last="identity"
    )
    trainer = NeurogenesisTrainer(
        ae,
        None,
        thresholds=[0.5],
        max_nodes=[1],
        max_outliers=1,
        objective_mode="paper_level_ae",
        global_coupling_cfg={
            "enabled": True,
            "trigger": "after_class",
            "epochs": 1,
            "lr_ratio": 0.01,
            "scope": "all",
        },
    )
    loader = DataLoader(TensorDataset(torch.randn(4, 4)), batch_size=2)
    seen = {}

    def fake_loop(loader_arg, opt, epochs, **kwargs):
        seen["forward_fn"] = kwargs.get("forward_fn")
        seen["replay"] = kwargs.get("replay")
        replay = kwargs.get("replay")
        if callable(replay):
            seen["replay_shape"] = tuple(replay(2).shape)
        return {"epoch_loss": [0.1]}

    ae._run_epoch_loop = fake_loop

    trainer._run_global_coupling(
        class_id=0,
        loader=loader,
        replay_sampler=lambda n: torch.zeros(n, 4),
        replay_only=False,
        trigger="after_class",
    )

    assert seen["forward_fn"] is None
    assert callable(seen["replay"])
    assert seen["replay_shape"] == (2, 4)


def test_stability_schedule_defaults_to_mixed(dummy_loader):
    ae = DummyAE(hidden_sizes=[2])
    trainer = NeurogenesisTrainer(
        ae,
        None,
        thresholds=[0.5],
        max_nodes=[1],
        max_outliers=1,
    )

    hist = trainer._scheduled_stability_phase(
        loader=dummy_loader,
        level=0,
        lr=1e-3,
        epochs=3,
        old_x=lambda batch_size: torch.zeros(batch_size, 1),
        replay_only=False,
    )

    assert ae.stability_calls == [(0, 3, 1e-3, "callable", False)]
    assert hist["_new_epochs"] == 1


def test_current_then_replay_stability_schedule_splits_phases(dummy_loader):
    ae = DummyAE(hidden_sizes=[2])
    trainer = NeurogenesisTrainer(
        ae,
        None,
        thresholds=[0.5],
        max_nodes=[1],
        max_outliers=1,
        stability_schedule="current_then_replay",
        stability_current_epochs_ratio=0.5,
        stability_replay_epochs_ratio=0.25,
    )

    hist = trainer._scheduled_stability_phase(
        loader=dummy_loader,
        level=0,
        lr=1e-3,
        epochs=10,
        old_x=lambda batch_size: torch.zeros(batch_size, 1),
        replay_only=False,
    )

    assert ae.stability_calls == [
        (0, 5, 1e-3, "none", False),
        (0, 3, 1e-3, "callable", True),
    ]
    assert hist["_schedule"] == ["current", "replay"]
    assert hist["_new_epochs"] == 1


def test_replay_then_current_stability_schedule_splits_phases(dummy_loader):
    ae = DummyAE(hidden_sizes=[2])
    trainer = NeurogenesisTrainer(
        ae,
        None,
        thresholds=[0.5],
        max_nodes=[1],
        max_outliers=1,
        stability_schedule="replay_then_current",
        stability_current_epochs_ratio=0.5,
        stability_replay_epochs_ratio=0.25,
    )

    hist = trainer._scheduled_stability_phase(
        loader=dummy_loader,
        level=0,
        lr=1e-3,
        epochs=10,
        old_x=lambda batch_size: torch.zeros(batch_size, 1),
        replay_only=False,
    )

    assert ae.stability_calls == [
        (0, 3, 1e-3, "callable", True),
        (0, 5, 1e-3, "none", False),
    ]
    assert hist["_schedule"] == ["replay", "current"]
    assert hist["_new_epochs"] == 1



def test_interleave_epochs_stability_schedule_alternates_short_phases(dummy_loader):
    ae = DummyAE(hidden_sizes=[2])
    trainer = NeurogenesisTrainer(
        ae,
        None,
        thresholds=[0.5],
        max_nodes=[1],
        max_outliers=1,
        stability_schedule="interleave_epochs",
        stability_current_epochs_ratio=0.3,
        stability_replay_epochs_ratio=0.2,
    )

    hist = trainer._scheduled_stability_phase(
        loader=dummy_loader,
        level=0,
        lr=1e-3,
        epochs=10,
        old_x=lambda batch_size: torch.zeros(batch_size, 1),
        replay_only=False,
    )

    assert ae.stability_calls == [
        (0, 1, 1e-3, "none", False),
        (0, 1, 1e-3, "callable", True),
        (0, 1, 1e-3, "none", False),
        (0, 1, 1e-3, "callable", True),
        (0, 1, 1e-3, "none", False),
    ]
    assert hist["_schedule"] == ["current", "replay", "current", "replay", "current"]
    assert hist["_new_epochs"] == 3


def test_interleave_batches_stability_schedule_uses_single_mixed_call(dummy_loader):
    ae = DummyAE(hidden_sizes=[2])
    trainer = NeurogenesisTrainer(
        ae,
        None,
        thresholds=[0.5],
        max_nodes=[1],
        max_outliers=1,
        stability_schedule="interleave_batches",
    )

    hist = trainer._scheduled_stability_phase(
        loader=dummy_loader,
        level=0,
        lr=1e-3,
        epochs=10,
        old_x=lambda batch_size: torch.zeros(batch_size, 1),
        replay_only=False,
    )

    assert ae.stability_calls == [(0, 10, 1e-3, "callable", False)]
    assert ae.stability_interleave_flags == [True]
    assert hist["_schedule"] == ["interleave_batches"]
    assert hist["_new_epochs"] == 1

def test_invalid_stability_schedule_rejected():
    with pytest.raises(ValueError):
        NeurogenesisTrainer(
            DummyAE(hidden_sizes=[2]),
            None,
            thresholds=[0.5],
            max_nodes=[1],
            max_outliers=1,
            stability_schedule="shuffle_everything",
        )
