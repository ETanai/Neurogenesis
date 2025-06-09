#!/usr/bin/env python3
from pathlib import Path

SRC = Path("src")
TESTS = Path("tests")

PY_SKELETON = """\
import pytest
import {module_import} as mod

def test_{func_name}_exists():
    assert hasattr(mod, "{func_name}")

# TODO: add more tests for {module_import}
"""


def snake_to_testname(name: str) -> str:
    # e.g. autoencoder.py -> test_autoencoder.py
    return "test_" + name


def main():
    for src_path in SRC.rglob("*.py"):
        # skip __init__.py files
        if src_path.name == "__init__.py":
            continue

        # compute relative path under src/
        rel = src_path.relative_to(SRC)
        # mirror directory in tests/
        test_dir = TESTS / rel.parent
        test_dir.mkdir(parents=True, exist_ok=True)
        # ensure __init__.py so pytest collects packages
        init_file = test_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text("")

        # decide test file path
        test_file = test_dir / snake_to_testname(src_path.name)
        module_import = ".".join(["src"] + list(rel.with_suffix("").parts))
        func_name = rel.stem

        if not test_file.exists():
            # write a basic pytest skeleton
            test_file.write_text(
                PY_SKELETON.format(module_import=module_import, func_name=func_name)
            )
            print(f"Created {test_file}")

    # Optionally, update conftest.py with shared fixtures
    conftest = TESTS / "conftest.py"
    if not conftest.exists():
        conftest.write_text("""\
import pytest
import torch

@pytest.fixture
def dummy_mnist_batch():
    # 4 random 28×28 images + dummy labels
    imgs = torch.randn(4,1,28,28)
    labels = torch.randint(0, 10, (4,))
    return imgs, labels
""")
        print(f"Created shared fixture in {conftest}")


if __name__ == "__main__":
    main()
