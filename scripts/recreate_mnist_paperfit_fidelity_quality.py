"""Run strict-fidelity MNIST quality-recovery sweep (intrinsic replay only)."""

from pathlib import Path

try:
    from scripts.run_paper_config import run_from_config
except ModuleNotFoundError:  # pragma: no cover - direct script invocation fallback
    from run_paper_config import run_from_config


def main() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "paper"
        / "mnist_paperfit_fidelity_quality.yaml"
    )
    run_from_config(config_path)


if __name__ == "__main__":
    main()
