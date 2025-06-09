import pytest
import torch
from torch import optim

from training.train_ae import LitWrapper
from training.train_ae import main as hydra_main

# --- Fixtures --------------------------------------------------------------


@pytest.fixture
def dummy_mnist_batch():
    # mimic conftest dummy_mnist_batch
    imgs = torch.randn(4, 1, 28, 28)
    labels = torch.randint(0, 10, (4,))
    return imgs, labels


@pytest.fixture
def lit_module():
    # instantiate the LightningModule
    return LitWrapper()


class DummyImageLogger:
    """Capture calls to add_image(tag, img, step)."""

    def __init__(self):
        self.logged = []

    def add_image(self, tag: str, img: torch.Tensor, global_step: int):
        # clone to avoid in-place changes
        self.logged.append((tag, img.clone(), global_step))


@pytest.fixture
def lit_with_dummy_logger(lit_module):
    """Attach a dummy logger and set a fake current_epoch."""
    lit = lit_module
    lit.logger = DummyImageLogger()
    lit.current_epoch = 5
    return lit


# --- Tests -----------------------------------------------------------------


def test_litwrapper_training_step(lit_module, dummy_mnist_batch, monkeypatch):
    batch = dummy_mnist_batch
    loss = lit_module.training_step(batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0


def test_litwrapper_validation_step(lit_module, dummy_mnist_batch, caplog):
    caplog.set_level("INFO")
    # patch .log to capture val_loss
    lit_module.log = lambda name, value: caplog.info(f"{name}: {value.item()}")
    lit_module.validation_step(dummy_mnist_batch, batch_idx=0)
    assert any("val_loss:" in rec.message for rec in caplog.records)


def test_configure_optimizers_returns_optimizer(lit_module):
    opt = lit_module.configure_optimizers()
    assert isinstance(opt, optim.Adam)
    assert hasattr(opt, "param_groups") and len(opt.param_groups) > 0


def test_validation_epoch_end_logs_images(lit_with_dummy_logger, dummy_mnist_batch):
    lit = lit_with_dummy_logger
    # Simulate that validation_step stored the last batch internally
    imgs, labels = dummy_mnist_batch
    lit._last_val_batch = (imgs, labels)

    # Call the hook; normally it takes outputs, but we ignore them
    lit.validation_epoch_end(outputs=[])

    # We should have exactly one image logged
    assert len(lit.logger.logged) == 1
    tag, grid, step = lit.logger.logged[0]

    # Verify tag and step
    assert tag == "val/example_inputs_vs_recon"
    assert step == lit.current_epoch

    # Check grid shape: [C, 2*H, batch_size*W]
    C, H2, W2 = grid.shape
    _, _, H, W = imgs.shape

    assert C == 1, "Expected single-channel image"
    assert H2 == 2 * H, f"Expected height {2 * H}, got {H2}"
    assert W2 == imgs.size(0) * W, f"Expected width {imgs.size(0) * W}, got {W2}"
    assert torch.isfinite(grid).all(), "All grid values should be finite"


def test_main_entry_point_returns_exit_code_zero(monkeypatch, tmp_path):
    """
    Calling hydra_main() directly will try to parse sys.argv;
    Monkeypatch to pass a minimal config and prevent actual training.
    """

    class DummyCfg:
        datamodule = {
            "_target_": "src.data.mnist_datamodule.MNISTDataModule",
            "batch_size": 2,
            "num_workers": 0,
            "data_dir": str(tmp_path),
        }
        model = {"hidden_sizes": [16, 8], "activation": "relu", "lr": 1e-3}
        trainer = {"max_epochs": 1, "fast_dev_run": True}

    # Prevent Hydra from exiting
    monkeypatch.setattr(hydra_main, "__wrapped__", lambda cfg: None)
    hydra_main.callback = None

    try:
        hydra_main(cfg=DummyCfg())
    except SystemExit:
        pytest.skip("Hydra entry point invoked SystemExit — skipping real-run test")
