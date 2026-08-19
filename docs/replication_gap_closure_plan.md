# Replication and Data-Gap Closure Plan

## Purpose

This document is the execution plan for turning the existing Neurogenesis Deep
Learning experiments into an auditable replication package. It is intended to
be followed on a freshly cloned GPU machine.

The current portable result bundle supports a strong endpoint-level negative
MNIST result, but it is not a complete experiment archive. Raw run logs,
checkpoints, code and environment provenance, dataset split manifests, uniform
training trajectories, confirmatory intrinsic-replay diagnostics, and the full
SD-19 campaign are absent. Some historical controls are also not exactly
capacity matched.

The goal is not to tune until the published result appears. The goal is to
produce enough preserved evidence to classify every claim as replicated,
contradicted, unresolved, or not tested under a predeclared gate. A robust,
fully documented negative result is a completed replication.

## Current boundary

The following evidence is already preserved under
`docs/figures/replication/data/`:

- 60 MNIST seed-level endpoints covering six conditions and seeds 42--51;
- 600 final MNIST per-class observations;
- nested NDL growth reports for the confirmation runs;
- six SD-19 endpoint rows from a restricted three-seed, 16-class,
  original-data-replay feasibility screen;
- one-seed controlled resource measurements;
- three-seed predictive-coding and global-coupling extension results; and
- portable source tables for the nine existing replication figures.

This is sufficient to support the statement that the archived implementation
had worse final MNIST macro MSE than its CL comparator in all three replay
regimes. It is not sufficient for a completely reproducible confirmation,
process-level forgetting and acquisition claims, qualitative reconstruction
claims, replay-quality explanations, resource generalization, or a full SD-19
replication.

Known issues that the new campaign must correct:

1. Historical original-data CL controls are not exact endpoint matches in four
   of ten seeds.
2. `confirmation_specs()` contains static control widths, and its no-replay CL
   specification is tied to the intrinsic endpoint rather than each paired
   no-replay NDL endpoint.
3. The official MNIST test set is exposed as `val` and evaluated throughout
   training; an untouched final test evaluation is not preserved.
4. `run_paper_config.py` discards most of the rich structure returned by
   `scripts.run_experiments.run()` unless MLflow happens to preserve it.
5. Incremental checkpoint output is atomically overwritten, so all curriculum
   checkpoints do not survive.
6. The portable exporter depends on ignored `outputs/` paths that are absent
   from the repository. A fresh clone currently cannot regenerate the bundle.
7. Failed and interrupted runs are not represented in the portable result
   bundle.
8. Runtime and memory conclusions are based on only one controlled seed.
9. The full 20-order SD-19 result is not present.

## Scientific decision rules

Freeze these rules before launching the final campaign. Store them in a new
machine-readable `config/paper/claims.yaml` and copy that file into every
campaign archive.

### Primary MNIST claim

Primary comparison:

```text
NDL + full-covariance intrinsic replay
minus
exact endpoint-capacity-matched CL + full-covariance intrinsic replay
```

Primary endpoint: final test macro reconstruction MSE.

- Replicated: NDL is better in at least 8/10 paired seeds and the paired 95%
  interval for `NDL - CL` is below zero.
- Contradicted: CL is better in at least 8/10 paired seeds and the paired 95%
  interval excludes a practically negligible reversal.
- Unresolved: neither directional rule is met.

Record the practical threshold in `claims.yaml` before running. The existing
protocol uses a 5% relative reversal as the materiality boundary.

### Secondary MNIST claims

Evaluate, without substituting them for the primary endpoint:

- foreground-pixel MSE;
- positive forgetting and signed backward transfer;
- worst-class final MSE;
- incoming-class acquisition gain;
- preservation of base digits 1 and 7;
- final parameter count and exact capacity equality;
- optimizer updates, runtime, CUDA allocation/reservation, and process RSS;
- quota satisfaction among active growth loops;
- cap exhaustion and unresolved outliers; and
- benefit of dataset or intrinsic replay relative to no replay.

### Growth rule

Do not calculate organicity by mixing loops that required no growth with loops
that grew successfully. Report three distinct fractions:

1. no growth required;
2. active growth stopped at the outlier quota; and
3. active growth exhausted a class or stream allowance with unresolved
   outliers.

The preregistered organicity gate remains at least 80% quota stops among active
growth loops.

### SD-19 rule

The full SD-19 campaign is conditional. The three-seed feasibility screen must
pass its predeclared gate before the 20-order campaign is described as a
promoted replication stage. A forced run after a failed gate is exploratory.

## Required artifact contract

Every launched run, including failures, must have a permanent directory:

```text
outputs/replication/<campaign_id>/<run_id>/
  run_manifest.json
  resolved_config.yaml
  environment.json
  dataset_manifest.json
  evaluation_trajectory.csv
  growth_trajectory.csv
  replay_diagnostics.csv
  resource_metrics.json
  qualitative_manifest.csv
  status.json
  stdout.log
  stderr.log
  checkpoints/
    base.pt
    step_02.pt
    step_03.pt
    ...
    final.pt
  figures/
```

The campaign root must additionally contain:

```text
campaign_manifest.json
claims.yaml
selection_ledger.csv
paired_effects.csv
final_test_metrics.csv
checksums.sha256
README.md
```

### `run_manifest.json`

Required fields:

- campaign ID, run ID, condition, evidence tier, status, and failure traceback;
- seed, class order, paired run ID, and source NDL run ID for a CL control;
- start/end timestamps and elapsed time;
- Git commit and dirty-worktree status;
- resolved configuration path and SHA-256 hash;
- Python, PyTorch, torchvision, Lightning, CUDA, cuDNN, NumPy, and MLflow
  versions;
- operating system, CPU, RAM, GPU model, driver, and device index;
- dataset source/version/archive hash;
- split seed, split index hashes, and train/validation/test counts per class;
- base and incremental checkpoint paths and hashes; and
- concrete replay implementation and old-data-isolation status.

### `evaluation_trajectory.csv`

One row per run, curriculum step, evaluated class, and depth:

```text
run_id, condition, seed, step, new_class, evaluated_class,
learned_status, depth, split, sample_count,
mean_mse, max_mse, std_mse, foreground_mse
```

This table must make final MSE, acquisition gain, forgetting, backward transfer,
base-class retention, and class-step curves recomputable without checkpoints.

### `growth_trajectory.csv`

One row per growth round when possible, otherwise one row per class/depth with
round totals:

```text
run_id, seed, class_id, depth, round,
width_before, width_after, nodes_requested, nodes_accepted,
threshold, initial_outliers, final_outliers, allowed_outliers,
initial_mse, final_mse,
plasticity_updates, stability_updates, next_layer_updates,
stop_reason, budget_exhausted, unresolved_outliers
```

### `replay_diagnostics.csv`

One row per curriculum step and replayed class:

```text
run_id, step, replay_class, real_count, generated_count,
latent_mean_error, covariance_error, round_trip_error,
nearest_neighbor_mean, nearest_neighbor_p95, generated_diversity,
real_error_p50, real_error_p95, generated_error_p50, generated_error_p95
```

Also persist actual replay counts for every old class in every paper-style
stability phase.

### `qualitative_manifest.csv`

Required fields:

```text
figure, row, column, condition, seed,
dataset_split, class_id, source_index, input_sha256,
curriculum_step, last_learned_class, depth, architecture_widths,
full_mse, foreground_mse, input_path, output_path,
checkpoint_path, checkpoint_sha256, config_sha256, git_commit
```

## Implementation work before GPU experiments

### 1. Add a local artifact writer

Extend `scripts/run_paper_config.py` and/or introduce a small shared artifact
writer that serializes the complete object returned by
`scripts.run_experiments.run()`. The archive must not depend on MLflow. MLflow
may mirror the same data, but local files are authoritative.

Reuse existing returned fields:

- `eval_records`;
- `growth_reports`;
- `threshold_history`;
- `training_stats`;
- optimizer diagnostics; and
- model/replay state.

Catch exceptions at the per-run boundary, write `status=failed` plus the full
traceback, update `campaign_manifest.json`, and then decide whether the campaign
should stop. Never silently drop failed seeds.

### 2. Implement train/validation/test separation

Update the MNIST data module so that:

- training and replay statistics use the training subset only;
- a deterministic, stratified portion of the original training set is the
  validation split;
- the official MNIST test set is exposed separately; and
- test evaluation cannot be called by the training loop.

Add configuration similar to:

```yaml
data:
  validation_fraction: 0.1
  split_seed: 20260819
  test_during_training: false
```

Persist exact split indices or a compact index file plus SHA-256 hash and
per-class counts. Apply the same principle to SD-19.

Add a separate, read-only final evaluator. It must load frozen final
checkpoints, refuse to train or refit replay, and write
`final_test_metrics.csv`. Validation trajectories may be collected throughout
training; official test endpoints are evaluated once after the campaign is
frozen.

### 3. Preserve all checkpoints

Replace a single overwritten `incremental_checkpoint_out` with a template or
callback that writes one checkpoint after base training and after every class.
Include the step and learned class in the filename. Retain the atomic temporary
write already used by the checkpoint implementation.

### 4. Make capacity matching dynamic and exact

Create `config/paper/mnist_confirmation.yaml` with six runs ordered as three
NDL-to-CL dependency pairs:

1. NDL + original-data replay;
2. CL + original-data replay with `control_size_from` run 1;
3. NDL + intrinsic replay;
4. CL + intrinsic replay with `control_size_from` run 3;
5. NDL, no replay;
6. CL, no replay with `control_size_from` run 5.

Before each CL run, assert that encoder widths, mirrored decoder widths, and
parameter count equal its source NDL endpoint. Also assert matching replay
regime, seed, split hashes, and class order. Abort the pair on mismatch.

Do not use the static control widths currently present in
`confirmation_specs()` for the new confirmation.

### 5. Export the existing diagnostics as tables

The code already creates fixed visual manifests and IR-quality diagnostics.
Write their scalar values and identities into the local artifact directory in
addition to logging images to MLflow.

### 6. Make resource accounting phase-specific

Record separately:

- base pretraining;
- incremental acquisition;
- neurogenesis plasticity;
- replay stability;
- next-layer propagation;
- replay-statistic refresh;
- validation and diagnostics; and
- figure generation.

Report training-only and total-pipeline time. Reset CUDA peak statistics at a
consistent boundary.

### 7. Make the portable exporter self-contained

Update `scripts/export_replication_figure_data.py` to read the new campaign
archive rather than deleted historical paths. The exporter should verify
checksums, complete seed pairs, exact capacity equality, expected row counts,
and failed-run accounting before producing portable CSV/JSON files.

## Tests required before launching full runs

Add or update automated tests for:

- deterministic and disjoint train/validation/test indices;
- no test access during training;
- no old-class training-data access in intrinsic replay;
- old-class access in the explicitly labelled dataset-replay control;
- exact seed-specific capacity matching;
- complete run-manifest fields;
- failed-run and traceback persistence;
- one checkpoint per completed curriculum step;
- checkpoint-resume equivalence to uninterrupted execution;
- trajectory row counts and metric recomputation;
- replay counts for every stored old class;
- fixed visual sample identities across conditions;
- paired effect and interval calculations;
- portable export from a standalone campaign archive; and
- regeneration of all report figures from portable data.

The gate is a clean full test suite and a successful fresh-directory export:

```bash
uv sync --dev
uv run pytest -q
uv run python scripts/export_replication_figure_data.py \
  --campaign /path/to/smoke-campaign \
  --output /tmp/replication-export
```

The exact new exporter flags should be documented in its `--help` output.

## GPU-machine bootstrap

After the implementation work above has been committed and pushed:

```bash
git clone https://github.com/ETanai/Neurogenesis.git
cd Neurogenesis
git rev-parse HEAD
uv sync --dev
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
uv run pytest -q
```

Record NVIDIA information before the first run:

```bash
nvidia-smi -q > gpu_environment.txt
uname -a > system_environment.txt
uv pip freeze > python_environment.txt
```

Do not run a confirmation from an uncommitted worktree. If an unavoidable local
change is required, commit it and record that commit before training.

MNIST may download through torchvision. For SD-19, verify sufficient disk space
and set stable dataset paths if required:

```bash
export MNIST_DATA_DIR=/data/neurogenesis/MNIST
export SD19_DATA_DIR=/data/neurogenesis/SD19/by_class
export SD19_DOWNLOAD_DIR=/data/neurogenesis/SD19
```

## Execution stages

### Stage A: deterministic smoke campaign

Run `[1,7] -> 0` with at most 16 incoming samples and one or two epochs for:

- NDL, no replay;
- NDL + dataset replay;
- NDL + intrinsic replay; and
- each exact capacity-matched CL control.

Run the same seed twice. Require:

- identical thresholds, widths, metrics, and stop reasons;
- complete artifacts and hashes;
- identical fixed visual source indices;
- checkpoint-resume equivalence;
- correct old-data access behavior;
- successful archive validation; and
- successful figure-data export.

Do not interpret smoke-run performance.

### Stage B: three-seed plumbing pilot

Run the full MNIST curriculum for three paired seeds using the frozen candidate
configuration and the new split boundary. This stage estimates time and storage
and confirms that every planned table and figure can be generated.

Do not use these seeds for post-hoc parameter selection unless the campaign is
explicitly relabelled as validation screening and a different seed set is
reserved for confirmation.

### Stage C: frozen ten-seed MNIST confirmation

Use ten new paired seeds, for example 100--109. Run all six conditions in
`mnist_confirmation.yaml`.

Requirements:

- published curriculum `[1,7] -> [0,2,3,4,5,6,8,9]`;
- one frozen configuration hash;
- exact per-seed capacity matching;
- identical data split hashes and class order within each pair;
- complete validation trajectories and checkpoints;
- full-covariance Gaussian intrinsic replay for the paper-shaped IR condition;
- no official test evaluation during training;
- failed and incomplete rows retained; and
- no tuning after the campaign begins.

The intended command after runner enhancement is:

```bash
uv run python scripts/run_paper_config.py \
  --config config/paper/mnist_confirmation.yaml \
  --output-dir outputs/replication/mnist_confirmation_v1 \
  --resume
```

`--output-dir` and `--resume` need to be added to `run_paper_config.py` before
using this command. The runner must verify that a resumed campaign has the same
Git commit, claims hash, configuration hash, and dataset split hashes.

After all training and artifact validation pass, run the sealed evaluator once:

```bash
uv run python scripts/evaluate_replication_test.py \
  --campaign outputs/replication/mnist_confirmation_v1
```

This evaluator is part of the required implementation work.

### Stage D: paired analysis and figures

Generate at least:

1. paired MNIST macro-MSE, foreground-MSE, and forgetting forest plot;
2. seed-by-digit paired-difference heatmap;
3. validation MSE trajectories over the curriculum;
4. acquisition-gain versus forgetting scatter plot;
5. class-by-depth growth stop-reason heatmap;
6. layer-width trajectories with seed intervals;
7. intrinsic-replay quality dashboard;
8. fixed-sample reconstruction grids;
9. repeated-seed resource comparison; and
10. claim-by-claim verdict table.

Compute paired differences and paired confidence intervals. Marginal condition
confidence intervals are descriptive and are not the primary comparison test.

### Stage E: repeated resource benchmark

Run at least three, preferably five, paired seeds on an otherwise idle fixed
machine. Use identical diagnostics settings and report training-only and total
pipeline costs separately.

An enhanced command should retain the current runner's resume behavior:

```bash
uv run python scripts/benchmark_training_resources.py \
  --output-dir outputs/replication/training_resources_v1 \
  --seeds 100,101,102,103,104 \
  --resume
```

The `--seeds` option and dynamic source-campaign lookup must be added before
using this command. Report medians, ranges, and paired seed-level values rather
than generalizing from one run.

## Replication-gap investigation after confirmation

### If the negative MNIST result remains decisive

If CL remains better in at least 8/10 paired seeds with a paired interval that
excludes a negligible difference:

- classify the tested paper-compatible implementation as a directional
  contradiction;
- do not continue tuning under the label "confirmation";
- use sensitivity analysis only to delimit the effect of paper ambiguities;
- keep global coupling, predictive coding, adaptive thresholds, and shape
  pressure explicitly exploratory; and
- skip the promoted SD-19 campaign or label any forced run exploratory.

This outcome closes the replication gap even though it does not match the
original direction.

### If the MNIST result is uncertain

If seed directions conflict or the paired interval is too wide:

- diagnose failures and pairing first;
- determine a precision-based target before adding seeds;
- normally extend to 20 paired seeds without changing configuration; and
- do not inspect or tune against the official test set.

### If NDL passes the clean-replay gate

Proceed in this order:

1. dataset replay upper bound;
2. literal full-covariance Gaussian intrinsic replay;
3. no-replay lower bound; and
4. diagonal or shrinkage replay variants as diagnostics only.

Require intrinsic replay to beat no replay in at least 8/10 paired seeds,
retain at least 70% of the dataset-replay gain, replay every old class, access
no old images, and avoid systematic cap exhaustion.

## Paper-compatible sensitivity analysis

The original paper leaves optimizer, activation, thresholds, outlier quota,
phase lengths, next-layer scope, and new-node policy underspecified. Resolve
these only on validation data, one mechanism family at a time:

1. optimizer and activation;
2. base-pretraining duration and corruption;
3. threshold percentile;
4. outlier quota;
5. plasticity, stability, and next-layer phase lengths;
6. `paper_columns` versus broad next-layer optimization;
7. absolute versus proportional node addition; and
8. class-local versus stream-level growth allowances.

Use three seeds for initial screens, five for promotion, and successive halving
after digit 0 and digit 3. Joint gates must include acquisition, retention,
outlier convergence, parameter count, and optimizer updates. Do not optimize
for visual proximity to the paper's endpoint widths.

Any promoted alternative requires a new frozen ten-seed campaign. It does not
retroactively replace the primary confirmation.

## Conditional SD-19 campaign

Enhance `scripts/run_sd19_comparison.py` to support the same artifact contract,
full trajectories, exact per-order capacity matching, dataset and intrinsic
replay conditions, class-order manifests, checkpoints, and resume validation.

Execution order:

1. three-seed restricted feasibility screen;
2. three-seed full-class paired comparison;
3. only after the declared gate passes, 20 shuffled class orders; and
4. intrinsic-replay conditions only after the MNIST IR bridge passes, unless
   explicitly exploratory.

Existing screen command pattern:

```bash
uv run python scripts/run_sd19_comparison.py \
  --stage screen \
  --seeds 100,101,102 \
  --output-dir outputs/replication/sd19_screen_v1 \
  --quiet \
  --resume
```

Full comparison, only after a passed screen:

```bash
uv run python scripts/run_sd19_comparison.py \
  --stage full \
  --seeds 100,101,102 \
  --screen-dir outputs/replication/sd19_screen_v1 \
  --output-dir outputs/replication/sd19_full_v1 \
  --quiet \
  --resume
```

For the paper's order/growth claim, retain one row per order seed, curriculum
position, class, and layer containing nodes added, widths, stop reason, MSE, and
forgetting. Plot neurons added versus curriculum position with mean and standard
deviation across the 20 orders.

## Storage, backup, and transfer

The repository intentionally ignores `outputs/`; pushing Git does not preserve
experiment data. Treat result storage as a separate required step.

After each campaign stage:

```bash
cd outputs/replication
find <campaign_id> -type f -print0 | sort -z | xargs -0 sha256sum > <campaign_id>/checksums.sha256
tar --zstd -cf <campaign_id>.tar.zst <campaign_id>
sha256sum <campaign_id>.tar.zst > <campaign_id>.tar.zst.sha256
```

Copy both archive and checksum to durable remote storage before deleting
anything from the GPU host. Suitable destinations include an institutional
object store, an immutable research-data repository, or a GitHub release if
file sizes permit. Do not commit large checkpoints or raw datasets to Git.

The final published record should include:

- Git commit used for training;
- result archive URL and SHA-256;
- dataset acquisition instructions and source hashes, subject to licensing;
- portable CSV/JSON tables needed for every reported figure; and
- a small checked-in manifest that maps every claim and figure to files in the
  external archive.

Verify a restored archive on a different directory or machine:

```bash
sha256sum -c <campaign_id>.tar.zst.sha256
tar --zstd -xf <campaign_id>.tar.zst
cd <campaign_id>
sha256sum -c checksums.sha256
```

## Recommended work order

| Priority | Work | Completion gate |
|---|---|---|
| P0 | Local artifact writer and failure registry | Every smoke run has a complete manifest |
| P0 | Train/validation/test separation | Split tests pass; training cannot access test |
| P0 | Dynamic exact capacity matching | Width and parameter assertions pass for every pair |
| P0 | Per-step trajectories and checkpoints | Metrics recompute and resume equivalence passes |
| P0 | Self-contained portable exporter | Export tests pass without historical `outputs/` |
| P1 | Deterministic smoke campaign | Duplicate run and archive-validation gates pass |
| P1 | Three-seed plumbing pilot | All intended tables and figures can be generated |
| P1 | Ten-seed MNIST confirmation | Frozen campaign and sealed final test complete |
| P1 | Paired statistics and qualitative grids | Claim registry can be resolved from artifacts |
| P2 | Repeated resource benchmark | At least three controlled paired seeds complete |
| P2 | Paper-ambiguity sensitivity | Validation-only screens and decisions preserved |
| P3 | Full SD-19/20-order campaign | Upstream gates pass or run is labelled exploratory |
| P3 | Post-replication extensions | Kept separate from replication verdict |

## Definition of done

The data gap is closed when:

- every number in the protocol can be recomputed from preserved tables;
- every run, including failures, has provenance and status;
- every CL/NDL comparison is exactly seed, split, order, replay, and capacity
  matched;
- acquisition and forgetting are recomputable from class-step trajectories;
- all figures regenerate from archived data;
- qualitative grids regenerate from preserved checkpoints and immutable sample
  manifests;
- resource claims use repeated controlled measurements;
- a fresh clone plus the external result archive can validate checksums and
  rebuild the report; and
- no claim relies solely on an inaccessible MLflow run or deleted log path.

The replication gap is closed when every original claim has one explicit
verdict:

- replicated under the frozen paper-compatible interpretation;
- contradicted under that interpretation;
- unresolved because the publication omits necessary information; or
- not tested because a predeclared upstream gate failed.
