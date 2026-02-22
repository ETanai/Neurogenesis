import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.ng_autoencoder import NGAutoEncoder
from training.neurogenesis_trainer import NeurogenesisTrainer
from utils.intrinsic_replay import IntrinsicReplay


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
        eval_batch=None,
        early_stop_on_eval=False,
        early_stop_cfg=None,
        forward_fn=None,
        epoch_logger=None,
        **kwargs,
    ):
        if callable(old_x):
            replay_descr = "callable"
        elif isinstance(old_x, torch.Tensor):
            replay_descr = f"tensor:{old_x.size(0)}"
        else:
            replay_descr = "none"
        ds_len = len(loader.dataset) if hasattr(loader, "dataset") else -1
        self.stability_calls.append((level, epochs, lr, replay_descr, ds_len))
        return {"epoch_loss": [0.5]}


class DummyIR:
    def __init__(self):
        self.fitted = False
        self._classes: list[int] = []

    def fit(self, loader):
        self.fitted = True
        self._classes = [0]

    def available_classes(self):
        return self._classes

    def sample_images(self, class_id, replay_size):
        # return dummy samples of correct feature size
        return torch.zeros(replay_size, 1)


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


def test_build_replay_sampler_ratio_vs_only(dummy_loader):
    ae = DummyAE(hidden_sizes=[2])
    ir = DummyIR()
    ir._classes = [0]
    trainer_ratio = NeurogenesisTrainer(
        ae,
        ir,
        thresholds=[0.1],
        max_nodes=[1],
        max_outliers=0.1,
        stability_replay_mode="ratio",
        stability_replay_ratio=1.0,
    )
    sampler_ratio, replay_only_ratio = trainer_ratio._build_replay_sampler(torch.device("cpu"), n_old_classes=1)
    assert sampler_ratio is not None
    assert replay_only_ratio is False
    assert sampler_ratio(4).shape[0] == 4

    trainer_only = NeurogenesisTrainer(
        ae,
        ir,
        thresholds=[0.1],
        max_nodes=[1],
        max_outliers=0.1,
        stability_replay_mode="only",
        stability_replay_ratio=1.0,
    )
    sampler_only, replay_only_only = trainer_only._build_replay_sampler(torch.device("cpu"), n_old_classes=1)
    assert sampler_only is not None
    assert replay_only_only is True
    assert sampler_only(4).shape[0] == 4


def test_stability_phase_targets_active_growth_level(dummy_loader):
    ae = DummyAE(hidden_sizes=[1, 1])
    trainer = NeurogenesisTrainer(
        ae,
        ir=None,
        thresholds=[0.5, 0.5],
        max_nodes=[1, 0],
        max_outliers=0.5,
        base_lr=0.1,
        plasticity_epochs=1,
        stability_epochs=1,
        next_layer_epochs=1,
    )

    trainer.learn_class(class_id=1, loader=dummy_loader)

    # First stability call corresponds to the growth round at level 0.
    # A prior bug routed this to the deepest level, increasing forgetting.
    assert ae.stability_calls
    assert ae.stability_calls[0][0] == 0


def test_build_replay_sampler_only_balanced(dummy_loader):
    class BalancedIR:
        def available_classes(self):
            return [1, 7, 9]

        def sample_images(self, cls, replay_size):
            if cls is None:
                raise AssertionError("only_balanced must request per-class samples")
            return torch.full((replay_size, 1), float(cls))

    ae = DummyAE(hidden_sizes=[2])
    ir = BalancedIR()
    trainer = NeurogenesisTrainer(
        ae,
        ir,
        thresholds=[0.1],
        max_nodes=[1],
        max_outliers=0.1,
        stability_replay_mode="only_balanced",
        stability_replay_ratio=1.0,
    )
    sampler, replay_only = trainer._build_replay_sampler(torch.device("cpu"), n_old_classes=3)
    assert sampler is not None
    assert replay_only is True

    batch = sampler(6)
    assert batch is not None
    values = batch.view(-1).tolist()
    counts = {cls: values.count(float(cls)) for cls in [1, 7, 9]}
    assert counts == {1: 2, 7: 2, 9: 2}


def test_growth_condition_modes():
    ae = DummyAE(hidden_sizes=[2])
    trainer_fraction = NeurogenesisTrainer(
        ae,
        ir=None,
        thresholds=[0.1],
        max_nodes=[1],
        max_outliers=0.25,
        max_outliers_mode="fraction",
    )
    assert trainer_fraction._growth_condition(n_outliers=3, total_seen=10, level=0) is True
    assert trainer_fraction._growth_condition(n_outliers=2, total_seen=10, level=0) is False

    trainer_count = NeurogenesisTrainer(
        ae,
        ir=None,
        thresholds=[0.1],
        max_nodes=[1],
        max_outliers=0.25,
        max_outliers_mode="count",
        max_outliers_count=2,
    )
    assert trainer_count._growth_condition(n_outliers=3, total_seen=10, level=0) is True
    assert trainer_count._growth_condition(n_outliers=2, total_seen=10, level=0) is False


def test_carry_forward_adds_next_level(dummy_loader):
    ae = DummyAE(hidden_sizes=[1, 1])
    trainer = NeurogenesisTrainer(
        ae,
        ir=None,
        thresholds=[0.5, 0.5],
        max_nodes=[1, 0],
        max_outliers=0.5,
        base_lr=0.1,
        plasticity_epochs=1,
        stability_epochs=1,
        next_layer_epochs=1,
        carry_forward_min_new_nodes=1,
    )
    trainer.learn_class(class_id=3, loader=dummy_loader)
    # Growth at level 0 should force at least one carry-forward add at level 1.
    assert any(level == 1 and num >= 1 for level, num in ae.added)


def test_growth_budget_caps_added_nodes_per_level(dummy_loader):
    ae = DummyAE(hidden_sizes=[1])
    trainer = NeurogenesisTrainer(
        ae,
        ir=None,
        thresholds=[0.5],
        max_nodes=[3],  # node budget (new semantics)
        max_growth_rounds=[10],
        max_nodes_legacy_round_semantics=False,
        max_outliers=0.5,
        factor_new_nodes=1.0,
        factor_max_new_nodes=1.0,
        plasticity_epochs=1,
        stability_epochs=1,
    )
    trainer.learn_class(class_id=5, loader=dummy_loader)
    total_added = sum(num for level, num in ae.added if level == 0)
    assert total_added == 3


def test_growth_round_cap_limits_iterations(dummy_loader):
    ae = DummyAE(hidden_sizes=[1])
    trainer = NeurogenesisTrainer(
        ae,
        ir=None,
        thresholds=[0.5],
        max_nodes=[10],
        max_growth_rounds=[1],  # single growth iteration even with remaining budget
        max_nodes_legacy_round_semantics=False,
        max_outliers=0.5,
        factor_new_nodes=1.0,
        factor_max_new_nodes=1.0,
        plasticity_epochs=1,
        stability_epochs=1,
    )
    trainer.learn_class(class_id=6, loader=dummy_loader)
    level0_adds = [num for level, num in ae.added if level == 0]
    assert len(level0_adds) == 1


def test_layer_specific_growth_factor_overrides(dummy_loader):
    ae = DummyAE(hidden_sizes=[2, 2])
    trainer = NeurogenesisTrainer(
        ae,
        ir=None,
        thresholds=[0.5, 0.5],
        max_nodes=[10, 10],
        max_growth_rounds=[1, 1],
        max_outliers=0.1,
        factor_new_nodes=0.1,
        factor_new_nodes_by_layer=[0.25, 0.5],
        factor_max_new_nodes=1.0,
        factor_max_new_nodes_by_layer=[1.0, 1.0],
        plasticity_epochs=1,
        stability_epochs=1,
        next_layer_epochs=1,
    )
    trainer.learn_class(class_id=9, loader=dummy_loader)
    lvl0_adds = [num for level, num in ae.added if level == 0]
    lvl1_adds = [num for level, num in ae.added if level == 1]
    assert lvl0_adds and lvl0_adds[0] == 1
    assert lvl1_adds and lvl1_adds[0] == 2


def test_carry_forward_guard_blocks_when_next_level_novelty_low(monkeypatch, dummy_loader):
    ae = DummyAE(hidden_sizes=[1, 1])
    trainer = NeurogenesisTrainer(
        ae,
        ir=None,
        thresholds=[0.5, 0.5],
        max_nodes=[1, 0],
        max_growth_rounds=[1, 0],
        max_outliers=0.1,
        carry_forward_min_new_nodes=1,
        carry_forward_requires_outlier_check=True,
        carry_forward_outlier_fraction_min=0.01,
        plasticity_epochs=1,
        stability_epochs=1,
        next_layer_epochs=1,
    )

    def _fake_get_outliers(loader, level):
        if level == 0:
            return 4, loader, 4
        return 0, loader, 4

    monkeypatch.setattr(trainer, "_get_outliers", _fake_get_outliers)
    trainer.learn_class(class_id=11, loader=dummy_loader)
    # Level 1 additions should be absent because carry-forward guard blocks forced growth.
    assert not any(level == 1 for level, _ in ae.added)


def test_stability_explicit_balanced_loader_includes_replay(dummy_loader):
    ae = DummyAE(hidden_sizes=[1, 1])
    ir = DummyIR()
    ir._classes = [0]
    trainer = NeurogenesisTrainer(
        ae,
        ir=ir,
        thresholds=[0.5, 0.5],
        max_nodes=[1, 0],
        max_outliers=0.1,
        max_growth_rounds=[1, 0],
        stability_dataset_mode="explicit_balanced",
        stability_old_per_class=2,
        plasticity_epochs=1,
        stability_epochs=1,
        next_layer_epochs=1,
    )
    trainer.learn_class(class_id=12, loader=dummy_loader)
    # replay is merged into the loader in explicit_balanced mode (old_x is not used directly)
    assert ae.stability_calls
    first_stability = ae.stability_calls[0]
    assert first_stability[3] == "none"
    assert first_stability[4] >= 6  # 4 new + at least 2 replay samples


def test_deep_cap_hit_dampening_reduces_late_additions(dummy_loader):
    ae = DummyAE(hidden_sizes=[10])
    trainer = NeurogenesisTrainer(
        ae,
        ir=None,
        thresholds=[0.5],
        max_nodes=[20],
        max_growth_rounds=[2],
        max_outliers=0.1,
        factor_new_nodes=1.0,
        factor_max_new_nodes=0.2,
        deep_cap_hit_dampening={"enabled": True, "levels": [0], "factor_decay": 0.1},
        plasticity_epochs=1,
        stability_epochs=1,
    )
    trainer.learn_class(class_id=13, loader=dummy_loader)
    level0_adds = [num for level, num in ae.added if level == 0]
    assert level0_adds[:2] == [2, 1]


def test_explicit_balanced_loader_syncs_intrinsic_replay_latent_dim() -> None:
    model = NGAutoEncoder(
        input_dim=4,
        hidden_sizes=[3, 2],
        activation="sigmoid",
        activation_latent="sigmoid",
        activation_last="sigmoid",
    )
    ir = IntrinsicReplay(model.encoder, model.decoder, device=torch.device("cpu"))
    base_x = torch.rand(6, 1, 2, 2)
    base_y = torch.zeros(6, dtype=torch.long)
    base_loader = DataLoader(TensorDataset(base_x, base_y), batch_size=6, shuffle=False)
    ir.fit(base_loader, class_filter=(0,))

    # Grow deepest level so decoder input width changes.
    model.add_new_nodes(level=1, num_new=1)
    trainer = NeurogenesisTrainer(
        model,
        ir=ir,
        thresholds=[0.0, 0.0],
        max_nodes=[0, 0],
        max_outliers=1.0,
        stability_dataset_mode="explicit_balanced",
        stability_old_per_class=2,
    )

    new_x = torch.rand(4, 1, 2, 2)
    new_y = torch.ones(4, dtype=torch.long)
    new_loader = DataLoader(TensorDataset(new_x, new_y), batch_size=4, shuffle=False)
    stable_loader = trainer._build_explicit_stability_loader(new_loader, device=torch.device("cpu"))
    assert stable_loader is not None
    assert len(stable_loader.dataset) >= len(new_loader.dataset)


def test_explicit_balanced_loader_calls_sync_when_available(dummy_loader):
    class SyncAwareIR:
        def __init__(self):
            self.synced = False

        def available_classes(self):
            return [0]

        def sync_encoder_latent_dim(self):
            self.synced = True

        def sample_images(self, cls, replay_size):
            if not self.synced:
                raise RuntimeError("sync_encoder_latent_dim not called")
            return torch.zeros(replay_size, 1)

    ae = DummyAE(hidden_sizes=[1, 1])
    trainer = NeurogenesisTrainer(
        ae,
        ir=SyncAwareIR(),
        thresholds=[0.5, 0.5],
        max_nodes=[0, 0],
        max_outliers=0.1,
        stability_dataset_mode="explicit_balanced",
        stability_old_per_class=2,
    )
    stable_loader = trainer._build_explicit_stability_loader(dummy_loader, device=torch.device("cpu"))
    assert stable_loader is not None


if __name__ == "__main__":
    pytest.main()
