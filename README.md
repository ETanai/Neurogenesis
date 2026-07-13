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

Dataset replay is intentionally preserved as a clean-data upper bound. Use it
to validate the model, growth, and stability schedule before evaluating
intrinsic replay; a method that fails with original historical samples is not
expected to be rescued by generated replay samples. Dataset-replay and
intrinsic-replay results should always be labeled separately.

The organic-growth screen keeps dataset replay and disables shape pressure:

```bash
python scripts/run_organic_growth_ablation.py --stage screen --seeds 42,43,44
python scripts/run_organic_growth_ablation.py --stage all --seeds 42,43,44
```

Use `--dry-run` to inspect the resolved matrix or `--quick` for a smoke test.
The runner writes JSON, CSV, and Markdown summaries under
`outputs/ablations/organic_growth/`, including cap-invariance, update-coverage,
outlier-stop, foreground-MSE, forgetting, capacity, and runtime evidence.

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
`docs/replication_gap_analysis.md` for the current verification status and
`docs/algorithm_fidelity.md` for the operation-level paper-to-code audit. The
staged experiments, promotion gates, and confirmatory comparisons are defined
in `docs/replication_validation_plan.md`. Tunable variables are divided into
paper-compatible undocumented details and explicit deviations in
`docs/performance_tuning_variables.md`. The iterative paper-compatible search
and renewed congruency audit are specified in
`docs/paper_compatible_optimization_plan.md`.

No software license is currently declared. Copyright permission for the paper
source and permission to use or redistribute the code should be evaluated
separately; repository ownership alone does not supply a license.
