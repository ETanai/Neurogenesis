# Neurogenesis

PyTorch research code for reproducing and extending *Neurogenesis Deep
Learning: Extending deep networks to accommodate new classes* (Draelos et al.,
arXiv:1612.03770v2).

The project trains stacked autoencoders incrementally. Samples with excessive
reconstruction error can trigger the addition of plastic neurons; after
training, those neurons are promoted into the mature network. Optional
intrinsic replay rehearses previously learned classes to reduce catastrophic
forgetting.

## Included workflows

- Classical and neurogenesis-enabled autoencoders
- Mature/plastic parameter blocks and dynamic layer growth
- MNIST and NIST SD-19 incremental-learning pipelines
- Intrinsic and dataset replay
- Hydra configuration and MLflow experiment tracking
- Paper-regime configurations and figure-generation scripts
- Unit and integration tests

## Installation

Python 3.11 or 3.12 is recommended. With `uv`:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e '.[dev]'
source .venv/bin/activate
```

Alternatively, create the Conda environment declared in `environment.yml`:

```bash
conda env create -f environment.yml
conda activate neurogenesis
```

PyTorch wheels can be large. Choose a CPU- or CUDA-specific PyTorch index when
appropriate for the target machine.

## Running experiments

Run the default Hydra experiment:

```bash
python scripts/run_experiments.py
```

The paper configurations in `config/paper/` cover classical learning (CL),
neurogenesis deep learning (NDL), and their intrinsic-replay variants for MNIST
and SD-19.

Inspect the reproducibility targets:

```bash
python scripts/recreate_paper.py --list-targets
```

Run a specific target, such as the Figure 4 suite:

```bash
python scripts/recreate_paper.py --only figure4
```

Full experiments download or prepare datasets and may require substantial time,
memory, and storage. Outputs and MLflow data are written beneath the repository
unless overridden by configuration.

## Tests

```bash
pytest -q
```

The test configuration restricts discovery to `tests/`; files under
`notebooks/` are examples rather than test modules.

## Paper source

The original arXiv v2 source bundle is preserved in
`docs/papers/arxiv-1612.03770v2/`. It contains the main
`NeurogenesisDeepLearning.tex`, its generated bibliography (`.bbl`), all paper
figures, and the original compressed source archive.

## Project status and scope

This repository is an independent implementation maintained by ETanai. It is
not the original authors' software release. The implementation includes
paper-oriented defaults as well as explicitly marked diagnostic and ablation
options; exact numerical reproduction can still depend on dataset preparation,
class ordering, hardware, and random seeds. See
`docs/replication_gap_analysis.md` for the current verification status.

No software license is currently declared. Copyright permission for the paper
source and permission to use or redistribute the code should be evaluated
separately; repository ownership alone does not supply a license.
