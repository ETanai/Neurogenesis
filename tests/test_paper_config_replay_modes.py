from pathlib import Path


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_mnist_ir_paper_configs_force_intrinsic_mode():
    cl_ir = _read("config/paper/mnist_cl_ir.yaml")
    ndl_ir = _read("config/paper/mnist_ndl_ir.yaml")
    fig4 = _read("config/paper/figure4.yaml")
    suite = _read("scripts/run_paper_experiments.py")

    assert "replay.mode=intrinsic" in cl_ir
    assert "replay.mode=intrinsic" in ndl_ir
    assert "name: cl_ir" in fig4 and "replay.mode=intrinsic" in fig4
    assert "name: ndl_ir" in fig4 and "replay.mode=intrinsic" in fig4
    assert '"replay.mode=intrinsic"' in suite
