# Replication status

This document records the current state of the Neurogenesis Deep Learning
replication. It replaces an earlier implementation plan whose blockers were
resolved by later commits.

## Implemented

- A rooted, composable Hydra configuration tree (`config/train.yaml`)
- Explicit base pretraining and incremental class orchestration
- Per-class loaders for MNIST and SD-19
- Layer-wise reconstruction thresholds and configurable outlier quotas
- Mature/plastic neuron blocks, bounded growth, promotion, and early stopping
- Paper-style isolated single-hidden-layer autoencoder objectives
- Replay of previously learned classes, including paper-style replay scheduling
- CL, CL+IR, NDL, and NDL+IR experiment configurations
- MNIST and SD-19 paper suites, ablations, diagnostics, MLflow logging, and plots
- A unified paper recreation command (`scripts/recreate_paper.py`)
- Unit and integration coverage for models, trainers, replay, data, and runners

## Verification boundary

The automated suite verifies implementation behavior and short integration
runs. It does not prove that a full experiment reproduces every published
number. Full numerical validation still requires:

1. Running all paper targets with the intended dataset preparation.
2. Recording class orders, random seeds, dependency versions, and hardware.
3. Comparing generated reconstruction curves, neuron counts, and figures with
   the publication.
4. Documenting any deviations caused by modern PyTorch behavior or ambiguous
   details in the paper.

The source contains optional diagnostics and ablations that are not claimed to
be part of the original algorithm. These are disabled in normal paper-oriented
configurations unless explicitly selected.

## Known repository-level limitation

No software license is declared. A license cannot be inferred or safely added
by a contributor; the copyright holder must choose and publish one. The bundled
paper source is a separate work and may have different redistribution terms.
