# Notebook-oriented training scripts overview

This note documents the two legacy scripts that live under `notebooks/` so that
we have a quick reference for what they currently do and how they diverge from
the production training CLI.

## `notebooks/notebook_test.py`

* Loads `config/notebook_config.yaml` directly from an absolute Windows path,
  so it only runs in the original author's environment unless the path is
  patched.【F:notebooks/notebook_test.py†L13-L18】
* Instantiates the `MNISTDataModule` with the pretraining class list, calls
  `prepare_data()` / `setup()`, and prints a sample batch shape as a quick
  sanity check for the notebook workflow.【F:notebooks/notebook_test.py†L20-L31】
* Builds a `NeurogenesisLightningModule`, mirrors its hyperparameters from the
  notebook config, and launches a vanilla Lightning `Trainer.fit` call for
  pretraining only—no incremental loop is executed here.【F:notebooks/notebook_test.py†L33-L52】
* After pretraining, extracts the underlying autoencoder (`ae`) and intrinsic
  replay object (`ir`), generates grouped batches for plotting partial
  reconstructions, and logs them to the hard-coded MLflow tracking server using
  `plot_partial_recon_grid_mlflow`.【F:notebooks/notebook_test.py†L54-L96】
* Wraps the autoencoder and replay components inside `NeurogenesisTrainer`,
  derives reconstruction thresholds from `test_all_levels`, and then iterates
  through the class sequence from the config to invoke `learn_class` for each
  digit.  This is the only place incremental neurogenesis happens in the
  notebook environment.【F:notebooks/notebook_test.py†L98-L142】

## `notebooks/train_mnist_neurogenesis.py`

* Exposes a tiny command-line entry point that trains the
  `NeurogenesisLightningModule` on MNIST with optional handcrafted DataLoaders,
  but the default path uses the `MNISTDataModule` just like the rest of the
  project.【F:notebooks/train_mnist_neurogenesis.py†L1-L72】
* Seeds the Lightning environment, constructs the autoencoder with fixed hidden
  sizes, thresholds, and neurogenesis hyperparameters, and wires up standard
  callbacks such as `ModelCheckpoint` and `LearningRateMonitor`.  The run only
  covers the *pretraining* phase; it never invokes the incremental neurogenesis
  loop.【F:notebooks/train_mnist_neurogenesis.py†L34-L88】
* Depending on the `--use_simple` flag, either hands Lightning explicit
  DataLoaders from `torchvision.datasets.MNIST` or lets the data module supply
  them.  Aside from this toggle, the script simply calls `trainer.fit` for the
  pretraining epochs configured on the module and prints a debug counter when the
  run finishes.【F:notebooks/train_mnist_neurogenesis.py†L12-L77】【F:notebooks/train_mnist_neurogenesis.py†L90-L107】

Together these scripts illustrate the manual workflows the authors used to
prototype neurogenesis experiments before the Lightning CLI existed.  They are
helpful references for expected control flow, but they rely on hard-coded paths
and ad-hoc logging, so they are not currently suitable for automated runs.

For a step-by-step playbook that leverages these scripts to reproduce the
paper’s MNIST and SD-19 experiments (across all four replay/neurogenesis
regimes), see `docs/notebook_experiment_workflow.md`.
