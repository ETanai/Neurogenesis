# Neurogenesis experiment automation solution blueprint

This document proposes a concrete implementation path that resolves the
blockers called out in the replication gap analysis and delivers a polished,
repeatable workflow for all four experimental regimes described in Draelos et
al. (2017).  Each section ties the design back to the current codebase so that
implementation tasks can be scheduled and verified incrementally.

## 1. Goals and success criteria

1. Provide a single CLI that can execute the entire curriculum (base pretraining
   plus incremental stages) for MNIST and SD-19 with neurogenesis and intrinsic
   replay toggles.
2. Support the four paper variants (CL, NDL, CL+IR, NDL+IR) via configuration so
   experiment sweeps can be run headlessly.
3. Automate data preparation, threshold estimation, replay statistics
   management, and artifact logging (RE curves, neuron growth, reconstructed
   samples) so figures can be regenerated without notebooks.
4. Ship regression coverage (unit + smoke tests) to keep the workflow stable.

Meeting these goals means a researcher can reproduce the published figures with
`python -m scripts.run_experiments experiment=mnist_ndl_ir` (or equivalent) and
obtain both CSV summaries and plotted charts.

## 2. Architecture overview

### 2.1 Hydra configuration tree

Create a rooted Hydra config (`config/train.yaml`) that wires together model,
trainer, data, and experiment definitions.  The current entry point
`train_ng_ae.py` assumes a `train` config but none exists.【F:src/training/train_ng_ae.py†L22-L23】
The new layout should include group defaults for:

- `model`: `ng_autoencoder` with layer sizes, activation, and neurogenesis knobs
  (growth limits, initialization strategies).
- `data`: `mnist` / `sd19` datamodules with per-class loader helpers; these live
  under `src/data` but lack loader factories today.【F:src/data/mnist_datamodule.py†L113-L131】【F:src/data/sd19_datamodule.py†L32-L329】
- `experiment`: wrapper describing class orders, curriculum length, replay mix
  ratios, and evaluation cadence.
- `logging`: output directory, artifact layout, and MLflow tracking settings.

Hydra multirun support then enables class-order sweeps (`hydra.sweeper.ax`) for
SD-19.

### 2.2 Runner modules

Replace the current Lightning-only loop with a layered orchestrator:

1. `BaseTrainer` – handles clean autoencoder pretraining on the current class
   curriculum before handing control to the neurogenesis loop.【F:src/models/ng_autoencoder.py†L59-L96】
2. `NeurogenesisController` – encapsulates threshold derivation, outlier gating,
   neuron addition, and logging.  It replaces the ad-hoc logic in
   `NeurogenesisTrainer` where counts are compared against fractional
   thresholds.【F:src/training/neurogenesis_trainer.py†L71-L259】
3. `ReplayBuffer` – maintains per-class statistics and yields replay batches for
   all learned classes instead of only the newest label.【F:src/utils/intrinsic_replay.py†L28-L87】
4. `IncrementalExperimentRunner` – iterates over class stages, delegates to the
   controller and replay buffer, collects metrics, and persists artifacts.

These components will be wired from the new CLI (`scripts/run_experiments.py`)
and accept Hydra configs for reproducibility.

### 2.3 Data pipeline upgrades

Augment the datamodules to expose loader factories:

- `get_class_loader(class_id, split, batch_size, shuffle)` – wraps the returned
  `Subset` in a `DataLoader`, removing the broken `.to_dataloader(...)`
  expectation.【F:src/training/train_ng_ae.py†L68-L74】
- `get_mixture_loader(class_counts, split, batch_size)` – yields batches across
  multiple classes, used by the replay buffer for stratified rehearsal.
- Dataset preparation hooks to download / preprocess SD-19 via a CLI command.

Ensure dataloaders expose plain image tensors so the autoencoder learns direct
reconstruction without additional input noise, matching the current project
scope.

## 3. Detailed solution plan

### Phase A – CLI foundation

1. **Hydra config scaffolding**: create `config/train.yaml` plus modular
   `config/model/ng_autoencoder.yaml`, `config/data/mnist.yaml`, etc.  Add typed
   config dataclasses (pydantic or OmegaConf structured configs) to guarantee
   schema validation when `train_ng_ae.py` instantiates its config tree.
2. **Runner entry point**: replace the direct `trainer.fit(model)` call with a
   custom loop that wires dataloaders explicitly.  Lightning can still provide
   optimizers and logging, but we will drive per-class steps manually so the
   curriculum is deterministic.
3. **Dataset utilities**: add loader helpers and tests verifying that per-class
   loaders deliver balanced batches for both MNIST and SD-19.

### Phase B – Base-model parity

1. **Clean-input pretraining**: verify `NGAutoEncoder` trains on noise-free
   inputs during the base stage and evaluation, documenting that denoising is
   intentionally omitted to keep the focus on neurogenesis behaviour.
2. **Threshold estimation service**: after base training, compute per-layer
   reconstruction statistics and persist thresholds in the experiment state (e.g.
   JSON).  `NeurogenesisTrainer` currently hard-codes `[0.05, …]` thresholds that
   must be removed.【F:config/neurogenesis.yaml†L1-L7】
3. **Pretraining independence**: decouple base training from the intrinsic
   replay toggle so CL and NDL baselines still receive pretrained weights when
   `cfg.ir.enabled` is false.【F:src/training/train_ng_ae.py†L61-L65】

### Phase C – Neurogenesis control loop

1. **Outlier gating fix**: compute proportions (count / batch size) before
   comparing against `max_outliers`, ensuring neuron growth stops when errors
   fall below threshold.【F:src/training/neurogenesis_trainer.py†L213-L259】
2. **Adaptive thresholds**: consume the measured thresholds from Phase B rather
   than static YAML values; expose logging for when each layer triggers growth.
3. **Growth limits**: add config-driven caps per layer and convergence checks
   (e.g. patience on reconstruction improvements) to prevent runaway expansion.

### Phase D – Intrinsic replay fidelity

1. **Replay statistics store**: redesign `IntrinsicReplay` to keep running
   estimates of means/covariances across sessions and offer
   `sample_batch(batch_size)` that mixes all prior classes using configurable
   class weights.  Maintain numerical stability with covariance shrinkage.
2. **Trainer integration**: during each incremental stage, request replay batches
   for *all* stored classes and interleave them with the real data per the paper
   ratios.  This corrects the current behaviour where only the latest class is
   rehearsed.【F:src/training/neurogenesis_trainer.py†L147-L226】
3. **Statistics refresh**: after training a stage, recompute top-layer moments
   for every class (including replayed ones) so the buffer stays in sync with the
   expanded architecture.

### Phase E – Experiment automation and reporting

1. **Variant toggles**: implement an `ExperimentVariant` enum that enables/disables
   neurogenesis and replay features, ensuring the loop faithfully replicates the
   CL, NDL, CL+IR, and NDL+IR runs.
2. **Class order management**: add utilities for deterministic class sequences,
   random permutations with fixed seeds, and Hydra multiruns that sweep 20 SD-19
   orderings.
3. **Metrics & artifacts**: log per-class reconstruction error, layer-wise error,
   neuron growth, and example reconstructions to disk (CSV + PNG).  Integrate
   the workflow with MLflow for experiment tracking and artifact comparison.
4. **Automation CLI**: new script `scripts/run_experiments.py` orchestrates
   entire sweeps and writes a manifest summarizing run metadata (config hash,
   seed, dataset checksum, artifact paths).

### Phase F – Verification layer

1. **Unit tests**: cover the threshold calculator, replay buffer, and
   neurogenesis gating with deterministic fixtures under `tests/`.
2. **Smoke tests**: add a `pytest` mark that executes a mini curriculum
   (base + one incremental class) for each variant with tiny datasets to ensure
   pipelines stay operational.
3. **CI integration**: configure GitHub Actions (or reuse existing CI) to run the
   smoke tests and static checks on every PR; include optional nightly jobs for
   longer SD-19 sweeps.

## 4. Polished workflow for researchers

Once the above phases are delivered, the experiment lifecycle becomes:

1. **Prepare data**
   ```bash
   python -m scripts.run_experiments task=prepare_data dataset=sd19
   ```
2. **Run MNIST variants**
   ```bash
   python -m scripts.run_experiments experiment=mnist variant=ndl_ir seed=1
   python -m scripts.run_experiments experiment=mnist variant=cl_ir seed=1
   ```
3. **Run SD-19 sweeps**
   ```bash
   python -m scripts.run_experiments experiment=sd19 variant=ndl_ir \
       hydra/sweeper=ax hydra.sweeper.params.seed="range(1,21)"
   ```
4. **Generate reports**
   ```bash
   python -m scripts.generate_report run_dir=outputs/2024-nd-experiments
   ```
   The reporting script consolidates CSV metrics, renders reconstruction plots,
   and exports tables comparing the four regimes.

5. **Validate** – optional CI job or local command:
   ```bash
   pytest -m smoke_incremental
   ```

## 5. Dependencies and migration notes

- Introduce structured configs (pydantic or dataclasses) to catch typos early.
- Adopt a plotting stack (Matplotlib + Seaborn) for report generation.
- Keep notebooks as exploratory tooling but update them to call into the new CLI
  instead of duplicating training logic.

## 6. Risk mitigation

- **Performance**: large SD-19 sweeps may take hours; add resume checkpoints per
  class to continue interrupted runs.
- **Numerical stability**: clamp covariance eigenvalues in the replay buffer to
  avoid singular matrices when sampling.
- **Reproducibility**: log random seeds for data loading, PyTorch, and NumPy;
  store Hydra config snapshots alongside artifacts.

Implementing this blueprint will resolve the currently documented blockers and
provide a reproducible, automation-first path to replicate the Neurogenesis Deep
Learning experiments without reliance on bespoke notebooks.
