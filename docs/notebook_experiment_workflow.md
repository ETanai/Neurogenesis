# Notebook-driven experiment workflow plan

This guide captures how to reuse the legacy notebook scripts while the main
Lightning CLI is still incomplete.  It combines an orientation to the existing
code with a repeatable workflow for running every experiment variant demanded by
*Neurogenesis Deep Learning* (MNIST and NIST SD-19, with and without intrinsic
replay and neurogenesis).

## 1. Shared prerequisites

1. **Clone / install project deps** – create a Python environment, install the
   repo in editable mode (`pip install -e .`), and make sure PyTorch Lightning
   plus MLflow are available (required by all notebook flows).  The scripts
   import Lightning, MLflow, and our local packages directly.【F:notebooks/notebook_test.py†L1-L52】【F:notebooks/train_nsitsd19.py†L1-L70】
2. **Fix the config paths** – both notebook scripts load YAML files from the
   original author’s absolute Windows directory.  Update the paths (or symlink
   them) so the scripts can resolve the configs in your checkout.【F:notebooks/notebook_test.py†L13-L18】【F:notebooks/train_nsitsd19.py†L29-L34】
3. **Point MLflow to a tracking URI** – the scripts log thresholds and
   reconstructions via an `MLFlowLogger`; configure the URI in the YAML or set
   `MLFLOW_TRACKING_URI` so the runs have a backend store.【F:notebooks/notebook_test.py†L44-L96】【F:notebooks/train_nsitsd19.py†L56-L119】
4. **Stage the datasets** – download MNIST and SD-19 into the data directories
   referenced by the configs.  Each data module calls `prepare_data()`/`setup()`
   when the script starts, so the folders must exist and be writable.【F:notebooks/notebook_test.py†L20-L33】【F:notebooks/train_nsitsd19.py†L40-L70】

## 2. MNIST workflows (`notebook_test.py`)

`notebook_test.py` already mirrors the paper’s MNIST schedule: base pretraining
on two digits, threshold estimation, and sequential neurogenesis across the
remaining classes.【F:notebooks/notebook_test.py†L20-L142】  To extract the four
regimes (CL, NDL, CL+IR, NDL+IR):

1. **Pretraining baseline (all variants)**
   - Run the script once to fit the stacked autoencoder on the pretraining
     digits (default `[1, 7]`).  The Lightning trainer handles this stage and
     logs the base reconstruction losses.【F:notebooks/notebook_test.py†L33-L73】
   - Save the `ae` weights after `trainer.fit` finishes so subsequent variants
     can reload the same baseline (avoid re-pretraining each time).
2. **Threshold estimation**
   - After pretraining, keep the block that invokes
     `trainer_ng.test_all_levels(...)` to compute layer-wise mean / std and set
     thresholds via `mean + 3·std`.【F:notebooks/notebook_test.py†L98-L116】
   - Export the resulting values (e.g., JSON) so the non-neurogenesis baselines
     can reuse them for logging comparisons.
3. **Variant loops**
   - *NDL + IR (paper’s primary method)* – run the script as-is; it calls
     `learn_class` for each digit while intrinsic replay stays enabled on the
     Lightning module.【F:notebooks/notebook_test.py†L117-L142】
   - *NDL only* – disable replay by flipping the config flag (`cfg.ir.enabled`
     or equivalent) before building `NeurogenesisLightningModule`; rerun the
     incremental loop to measure forgetting without intrinsic replay.【F:notebooks/notebook_test.py†L33-L74】
   - *CL + IR* – skip calling `learn_class` on `NeurogenesisTrainer` and instead
     fine-tune the pretrained autoencoder directly with replay batches for each
     new class (modify the script to freeze growth and call into Lightning’s fit
     using class-conditioned loaders).  Record reconstruction curves for
     comparison.
   - *CL only* – run the same loop as above but disable replay and lock the
     architecture; this isolates pure catastrophic forgetting.
4. **Logging & artifacts**
   - For each variant, log per-class reconstruction error, neuron counts, and
     partial reconstructions to MLflow using the helper already wired in.
   - Export CSV/JSON summaries after each run so downstream plotting scripts can
     reproduce the paper’s figures.

## 3. SD-19 workflows (`train_nsitsd19.py`)

`train_nsitsd19.py` mirrors the extended experiment (digits pretraining, then
letters).【F:notebooks/train_nsitsd19.py†L12-L184】  The workflow is analogous to
MNIST but needs additional orchestration:

1. **Dataset verification** – ensure the SD-19 datamodule resolves label
   mappings (digits 0–9, uppercase 10–35, lowercase 36–61).  Adjust
   `build_sd19_class_sequences()` if your dataset uses a different encoding.
2. **Pretraining on digits** – run the script to train the autoencoder on digits
   only (configurable via `cfg.pretraining.classes_pretraining`).  Save the
   checkpoint for reuse across variants.【F:notebooks/train_nsitsd19.py†L38-L112】
3. **Threshold derivation** – keep the `test_all_levels` call on the digit
   dataloader to compute thresholds and log them via MLflow.  Persist these
   metrics for other regimes just like MNIST.【F:notebooks/train_nsitsd19.py†L118-L144】
4. **Incremental sequence** – iterate through uppercase then lowercase letter
   classes, calling `trainer_ng.learn_class`.  Capture neuron growth and error
   curves after each class.【F:notebooks/train_nsitsd19.py†L145-L183】
5. **Variant toggles** – replicate the four regimes by enabling/disabling
   neurogenesis and replay before the loop, mirroring the MNIST setup.
6. **Multiple permutations** – repeat the entire process for ≥20 class-order
   permutations (shuffle uppercase/lowercase separately) and aggregate metrics
   to match the paper’s reporting.

## 4. Post-processing & reporting

1. Collate MLflow metrics and exported CSVs into analysis notebooks or scripts
   that regenerate reconstruction-error plots, neuron-growth charts, and sample
   reconstructions for every variant.
2. Compare MNIST and SD-19 results across the four regimes, highlighting how
   replay and neurogenesis affect forgetting.
3. Version the configs, saved thresholds, and dataset permutations so the
   workflow is repeatable and auditable.
