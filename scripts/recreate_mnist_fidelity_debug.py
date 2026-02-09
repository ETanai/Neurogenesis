"""Run the MNIST fidelity-first debug matrix defined in config/paper/mnist_fidelity_debug.yaml."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_paper_config import run_from_config


def main() -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "config" / "paper" / "mnist_fidelity_debug.yaml"
    )
    run_from_config(config_path)


if __name__ == "__main__":
    main()
