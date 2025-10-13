# Neurogenesis Deep Learning replication gaps and implementation plan

The repository already exposes neurogenesis-aware layers, an autoencoder
backbone, intrinsic replay utilities, and some Lightning plumbing, but the
paper’s training loop cannot yet be reproduced end to end.  Below is an updated
gap audit (double-checked against the current code) followed by a concrete plan
for implementing the missing pieces.  A companion blueprint in
`docs/experiment_solution_blueprint.md` translates this audit into a polished
automation workflow and CLI design that resolves every blocker while providing a
repeatable way to run all experiments.

## 1. Revalidated blockers

### 1.1 Configuration and orchestration

- `train_ng_ae.py` is decorated with `@hydra.main(config_name="train")`, yet the
  `config` directory only holds hand-written YAML fragments (no `train.yaml`), so
  the entry point cannot even instantiate its config tree.【F:src/training/train_ng_ae.py†L22-L23】【F:config/config.yaml†L1-L13】
- The script pretrains and sequentially calls `trainer.fit(model)` without ever
  supplying a datamodule or custom loaders, and
  `NeurogenesisLightningModule` does not implement `train_dataloader()`.  The
  resulting Lightning call stack fails immediately instead of driving
  incremental class updates.【F:src/training/train_ng_ae.py†L61-L75】【F:src/training/neurogenesis_lightning_module.py†L68-L109】
- `MNISTDataModule.get_class_dataset` returns a bare `Subset`, but the script
  expects a `.to_dataloader(...)` helper that does not exist, so even a corrected
  training loop cannot obtain per-class loaders without new plumbing.【F:src/training/train_ng_ae.py†L68-L74】【F:src/data/mnist_datamodule.py†L113-L131】

### 1.2 Base-model alignment with the paper

- `NGAutoEncoder.forward` flattens the image and reconstructs deterministically;
  there is no masking or Gaussian corruption even though the paper relies on a
  stacked *denoising* autoencoder.  Adding noise injection during both base and
  incremental phases is still outstanding.【F:src/models/ng_autoencoder.py†L59-L96】
- Thresholds and neurogenesis policies remain configuration constants (`[0.05,
  …]`).  The SD-19 notebook derives thresholds from reconstruction statistics,
  but the production trainer never measures the baseline or updates thresholds
  from data, so it cannot mirror the paper’s adaptive gating.【F:config/neurogenesis.yaml†L1-L7】【F:notebooks/train_nsitsd19.py†L103-L158】【F:src/training/neurogenesis_trainer.py†L19-L220】
- `_get_outliers` returns the **count** of samples above threshold, yet the
  growth loop compares that count to `max_outliers`, which is configured as a
  fraction (e.g. `0.05`).  Once any sample exceeds the threshold the condition is
  perpetually true, so neurogenesis never stops growing.  The comparison needs to
  operate on proportions.【F:src/training/neurogenesis_trainer.py†L71-L259】【F:config/neurogenesis.yaml†L1-L3】
- Base pretraining only fires when `cfg.ir.enabled` is true, so disabling replay
  to run the paper’s baselines also (accidentally) disables the initial training
  stage.  The pretrain loop must be separated from IR toggles to match the paper
  variants.【F:src/training/train_ng_ae.py†L61-L65】

### 1.3 Intrinsic replay gaps

- `NeurogenesisTrainer.learn_class` calls `IntrinsicReplay.fit` on the *current*
  class loader, then immediately samples from `class_id`.  No path ever batches
  replay samples from previously learned ids, so the “stability” phase rehearses
  only the newest class.【F:src/training/neurogenesis_trainer.py†L147-L226】
- `IntrinsicReplay.fit` overwrites the stored mean/covariance for whichever
  labels appear in the loader and offers only `sample_images(cls, n)`.  There is
  no incremental update, mixture sampling, or cap on latent statistics, making
  it impossible to assemble the multi-class replay batches the paper uses.【F:src/utils/intrinsic_replay.py†L28-L87】

### 1.4 Experiment automation

- The codebase only ships the neurogenesis path; there is no runner capable of
  flipping neurogenesis and replay on/off to cover the paper’s four regimes
  (CL, NDL, CL+IR, NDL+IR), nor any harness for multiple class-order
  permutations.【F:src/training/neurogenesis_trainer.py†L19-L283】
- The SD-19 notebook computes thresholds and runs the trainer interactively, but
  there is no CLI automation for dataset preparation, class scheduling, logging
  reconstruction curves, or exporting growth plots—the exact reporting the paper
  highlights.【F:notebooks/train_nsitsd19.py†L103-L188】【F:src/data/sd19_datamodule.py†L32-L329】

## 2. Implementation roadmap

The following milestones will get the repository to a faithful replication.  In
each phase we should land tests or scripts that keep the new surface area
exercised.

### Phase A — Foundation

1. Author a composable Hydra tree (`train.yaml`, model/data/trainer group
   defaults) that covers model hyperparameters, corruption settings, IR knobs,
   and experiment scheduling.
2. Update `train_ng_ae.py` (or successor CLI) to instantiate the datamodule,
   call `setup()`, and explicitly drive base training plus per-class loops with
   real `DataLoader` objects.
3. Extend the MNIST and SD-19 datamodules with helpers that materialize
   per-class and multi-class loaders (supporting stratified replay batches) in
   both training and validation splits.

### Phase B — Base-model parity

1. Introduce configurable corruption modules (e.g. masking/dropout noise) that
   wrap the encoder input during both pretraining and plasticity phases.
2. Collect reconstruction statistics after base training to derive thresholds
   (`mean + k·std` per layer) and persist them for downstream phases.
3. Decouple pretraining from IR toggles so baseline runs (CL / NDL) still train
   the autoencoder before incremental updates.

### Phase C — Neurogenesis control loop

1. Fix the outlier gating to compare proportions (or z-scores) against
   `max_outliers`, and cap growth when thresholds are met.
2. Integrate the new threshold-estimation module so `NeurogenesisTrainer` pulls
   its thresholds from measured stats instead of static config.
3. Add logging around neuron additions, error trajectories, and convergence
   checks to facilitate regression tests.

### Phase D — Intrinsic replay fidelity

1. Extend `IntrinsicReplay` with incremental mean/covariance updates and a
   `sample_mixture(class_counts)` API for multi-class replay batches.
2. Teach `NeurogenesisTrainer` to request replay samples for every cached class
   (respecting configurable mixing ratios) during stability phases.
3. Wire in caps/regularisation for covariance estimates and add unit tests that
   verify replay distributions stay numerically stable.

### Phase E — Experiment harness

1. Build an `IncrementalExperimentRunner` abstraction that can toggle
   neurogenesis and replay, choose class orders, and orchestrate the sequential
   loop for all regimes.
2. Surface experiment sweeps through Hydra (e.g. multirun over class-order
   seeds) and log reconstruction error plus network growth for every stage.
3. Promote the SD-19 notebook logic into a script/CLI that prepares the dataset,
   runs the runner across 20 permutations, and emits the plots/tables reported in
   the paper.

### Phase F — Verification & reporting

1. Add smoke tests or integration tests that execute a short MNIST curriculum
   (e.g. base class + one incremental class) across all four regimes to guard
   the pipeline.
2. Automate artifact generation (plots, CSV summaries) so replication reports can
   be regenerated non-interactively.
3. Document the full workflow (from config selection to analysis scripts) to
   make rerunning the experiments deterministic for future contributors.

Delivering these phases—in order—will give us a runnable training CLI with the
correct denoising behavior, adaptive neurogenesis thresholds, faithful replay
mixing, and the experiment sweeps required to reproduce every figure reported in
Neurogenesis Deep Learning.
