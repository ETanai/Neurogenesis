# Replication validation and ablation plan

This plan tests whether the implementation reproduces the comparative behavior
reported in Draelos et al., *Neurogenesis Deep Learning*. It deliberately tests
clean dataset replay before intrinsic replay: failure with original historical
samples identifies a training or growth problem that generated replay cannot
be expected to solve.

The variable-level boundary between paper-compatible tuning and explicit
non-paper ablations is defined in `docs/performance_tuning_variables.md`.
The iterative congruency audit, sensitivity screen, optimization loop, and
promotion rules are defined in `docs/paper_compatible_optimization_plan.md`.

The paper provides plots and qualitative comparisons but few machine-readable
numbers. The primary replication target is therefore the ordering and shape of
the published effects, accompanied by transparent implementation metrics. This
is not a claim of numerical identity to values that cannot be recovered from
the paper.

## Questions to answer

1. Can the base stacked denoising autoencoder represent its initial classes?
2. Can the incremental trainer learn new classes when supplied perfect replay?
3. Does neurogenesis add capacity only when reconstruction errors demand it?
4. With capacity-matched controls, does NDL improve stability/plasticity over
   conventional learning?
5. How much performance is lost when clean replay is replaced by intrinsic
   replay?
6. Do MNIST and SD-19 reproduce the paper's qualitative reconstruction and
   growth trends across independent seeds and class orders?

## Experimental principles

- Use train data for optimization, validation data for configuration selection,
  and test data once for the final confirmatory comparison.
- Pair all regimes by seed and class order.
- Fit CL controls to the measured final size of the corresponding NDL run.
- Change one mechanism family per ablation stage.
- Retain failed runs and their resolved configurations.
- Report actual optimizer updates and replay sample counts, not only epochs.
- Keep dataset replay, intrinsic replay, and no replay distinctly labeled.
- Do not select a configuration because it visually resembles the published
  test figure; define promotion rules before full test evaluation.

Several existing ablation scripts contain hard-coded metrics from earlier
repository runs. Treat those values only as historical diagnostics: they
predate the optimizer and replay-isolation fixes and must not be used as current
acceptance thresholds. Recompute every baseline under the same commit and data
split used for the new ablations.

## Metrics

### Reconstruction

- Per-class full-autoencoder MSE after every incremental class.
- Per-class partial reconstruction MSE at every level.
- Macro-average MSE across all classes learned so far.
- Incoming-class MSE immediately before and after learning.
- Base-class MSE for MNIST digits 1 and 7 throughout the curriculum.

### Continual-learning summaries

For class `c`, let `best_c` be its lowest validation MSE after it is learned and
`final_c` its MSE after the complete curriculum.

- Forgetting: `final_c - best_c`.
- Relative forgetting: `(final_c - best_c) / max(best_c, eps)`.
- Backward transfer: `best_c - final_c` (positive means later learning helped).
- Acquisition gain: pre-learning MSE minus post-learning MSE.
- Macro-average and worst-class values for all summaries.

### Growth and mechanism diagnostics

- Neurons added per class and per level.
- Cumulative layer sizes and total parameter count.
- Outlier count/fraction before and after every growth round.
- Fraction of levels that terminate below the outlier quota.
- Fraction of levels that terminate by hitting `MaxNodes`.
- Plasticity/stability parameter deltas by mature/plastic and encoder/decoder
  group.
- Optimizer updates, wall time, and peak memory.

### Intrinsic replay quality

- Clean-vs-generated reconstruction-error distributions.
- Latent mean and covariance error.
- Decoder round-trip latent error.
- Nearest-neighbor distance to real samples, reported as a diagnostic rather
  than optimized against the test set.
- Generated-sample diversity and per-class replay counts.
- Performance gap between matched dataset-replay and IR runs.

## Stage 0: deterministic smoke validation

Purpose: reject wiring and logging failures before spending compute.

Run a toy curriculum and a limited MNIST curriculum (`[1,7] -> 0`) for:

1. No replay.
2. Dataset replay.
3. Intrinsic replay.

Use one epoch/phase and at most 16 incoming samples. Verify:

- identical seeds reproduce thresholds, layer sizes, and metrics;
- intrinsic replay never requests loaders for old classes;
- dataset replay does request all configured historical classes;
- plasticity leaves mature encoder weights unchanged;
- stability updates all current-level parameter groups;
- every run emits its resolved configuration and diagnostic artifacts.

Gate: all invariants pass. Do not interpret reconstruction quality at this
stage.

## Stage 1: base autoencoder calibration

Purpose: establish a competent base model before evaluating neurogenesis.

Dataset: MNIST digits 1 and 7. Architecture:
`784-200-100-75-20-75-100-200-784`.

Paper-locked mechanism:

- stacked denoising pretraining;
- no end-to-end fine-tuning in the confirmatory candidate;
- no incremental classes;
- zero weight decay unless explicitly ablated.

Small screening grid:

- learning rate: `1e-4`, `3e-4`, `1e-3`;
- epochs per level: `14`, `50`, `100`;
- masking/dropout corruption: `0.0`, `0.1`, `0.2`;
- seeds: 3 for screening, 5 for promotion.

Primary score: validation macro-MSE on digits 1 and 7. Secondary checks:
balanced per-class MSE, stable thresholds, and absence of collapsed latent
variance.

Promotion gate:

- no seed may produce NaN/Inf or a collapsed representation;
- both base classes must improve substantially over an untrained model;
- the selected candidate must be within 10% of the best screening mean while
  preferring fewer optimizer updates on ties.

Use the `base` stage in `scripts/run_paper_spec_training_ablation.py`; reduce its
large Cartesian grid to the screening values above before launching it.

## Stage 2: clean-replay upper bound on one new class

Purpose: prove that the local objective, growth rule, and stabilization schedule
can learn digit 0 when replay quality is perfect.

Hold the promoted base model fixed. Run `[1,7] -> 0` with dataset replay.

Factor groups, tested sequentially rather than as one Cartesian product:

1. Phase schedule: plasticity/stability/next-level durations and LR ratios.
2. Threshold percentile: `0.95`, `0.975`, `0.985`, `0.995`.
3. Outlier allowance: `0.05`, `0.10`, `0.20`, `0.30`.
4. Growth rule: one node per round versus proportional growth
   (`0.002`, `0.005`, `0.01`).
5. Next-level interpretation: `paper_columns` versus `broad`.

Keep zero weight decay, paper-local SHL-AE objectives, and paper-style
per-class replay fixed.

Promotion gate:

- digit-0 acquisition gain must be positive in every promoted seed;
- final 1/7 macro-MSE may degrade by at most 20% from the base checkpoint;
- at least 80% of level/seed pairs must finish below the outlier quota rather
  than at `MaxNodes`;
- reject configurations whose full reconstruction improves only by explosive
  growth at the deepest level.

Relevant runners:

- `scripts/run_paper_spec_training_ablation.py --stage single0`
- `scripts/run_outlier_quota_ablation.py`
- `scripts/run_growth_shape_ablation.py`

Shape-pressure, adaptive thresholds, global coupling, and end-to-end fine-tuning
are diagnostic extensions. They must not enter the paper-locked candidate
unless the literal mechanism fails and the deviation is reported explicitly.

Before promotion to Stage 3, add an organic-shape gate. Compare the current
cap-driven reference (`0.01` proportional growth with cumulative allowances
`[25,35,8,20]`) against absolute one/two-node growth under loose `2x`/`4x`
cumulative ceilings, small per-class throttles `[4,5,2,3]` and `[8,10,4,6]`,
and a combination of a loose stream ceiling with a small class throttle. Keep
shape pressure off. Repeat each promoted candidate with its stream ceiling
doubled: a demand-emergent result must remain within 10% added width per layer
and 5% macro-MSE, stop at the outlier quota in at least 80% of growth loops,
and continue updating on at least half of the incremental classes. Compare
macro/foreground MSE, forgetting, parameters, update counts, runtime, and final
shape to the current reference. Funnel shape and proximity to
`[225,135,84,40]` are secondary observations and must not be optimization
targets.

## Stage 3: full MNIST clean-replay validation

Purpose: test the entire class sequence before introducing IR.

Run the published curriculum `[1,7] -> [0,2,3,4,5,6,8,9]` using the promoted
Stage 2 settings and dataset replay. Use 5 paired seeds for screening and 10 for
confirmation.

Conditions:

- NDL + dataset replay.
- Capacity-matched CL + dataset replay.
- NDL without replay.
- Capacity-matched CL without replay.

Do not tune after viewing confirmatory test metrics.

Clean-replay gate:

- NDL + dataset replay must have lower paired macro-MSE or lower paired
  forgetting than its capacity-matched CL control on at least 4/5 screening
  seeds;
- the 95% paired bootstrap interval for the promoted comparison should exclude
  a degradation larger than 5%;
- base digits 1 and 7 must remain represented through the final class;
- growth should generally be demand-driven rather than hitting every layer cap.

If this stage fails, stop. Diagnose training, thresholds, and growth; do not
proceed to IR optimization.

## Stage 4: intrinsic replay bridge

Purpose: isolate replay-generator error after the algorithm succeeds with clean
replay.

Freeze all Stage 3 model and training settings. Compare replay only:

1. Dataset replay upper bound.
2. Full-covariance Gaussian IR (paper interpretation).
3. Diagonal Gaussian IR (diagnostic).
4. Shrinkage Gaussian IR with shrinkage `0.1`, `0.25`, `0.5` (diagnostic).
5. No replay lower bound.

For paper fidelity, promote full-covariance Gaussian IR unless it is unstable;
if another sampler is required, label it as an extension.

IR gate:

- IR must outperform no replay on macro-MSE and forgetting in at least 4/5
  paired screening seeds;
- IR should retain at least 70% of the acquisition/retention gain achieved by
  dataset replay relative to no replay;
- all old classes must receive replay samples in every paper-style stability
  phase;
- no old-class original loader may be accessed.

## Stage 5: confirmatory MNIST paper comparison

Purpose: reproduce the qualitative claims behind Figures 4 and 5.

Use `config/paper/figure4.yaml`, which runs grown networks before their
capacity-matched controls. Run 10 paired seeds for final statistics (the current
five repetitions are suitable for screening).

Compare:

- CL;
- NDL;
- CL + IR;
- NDL + IR;
- optional NDL + dataset replay upper bound, displayed separately.

Predeclared claims:

1. Both IR conditions outperform their corresponding no-replay conditions on
   final macro-MSE or forgetting.
2. NDL + IR outperforms capacity-matched CL + IR on paired macro-MSE,
   worst-class MSE, or forgetting.
3. NDL + IR preserves early classes better as later digits are introduced.
4. NDL grows progressively from `[200,100,75,20]` instead of allocating its
   final capacity in advance.

Report effect sizes and paired bootstrap intervals. A qualitative claim is
supported only if its direction holds in at least 80% of paired seeds and the
interval does not include a practically important reversal (>5% relative).

Generate:

- Figure 4-style level-wise reconstruction curves;
- Figure 5-style reconstruction timeline;
- per-class forgetting heatmap;
- layer-growth trajectory with seed intervals;
- table of final MSE, forgetting, parameters, updates, and runtime.

## Stage 6: SD-19 validation

Purpose: test scale and class-order robustness.

First run a smoke curriculum: digits as base classes followed by four letters.
Require successful preprocessing, bounded memory, finite thresholds, and
positive acquisition gain.

Then run the full uppercase/lowercase curriculum under:

- NDL + dataset replay (clean upper bound);
- capacity-matched CL + dataset replay;
- NDL + intrinsic replay;
- capacity-matched CL + intrinsic replay, if compute permits.

Finally run `config/paper/sd19_growth_20.yaml`, which performs the paper's 20
randomly ordered letter curricula with dataset replay.

SD-19 claims:

- final MSE should be below initial MSE for most classes under successful NDL;
- NDL should outperform its capacity-matched CL control across classes;
- mean neurons added per new class should decline over the curriculum;
- the growth trend should remain visible across 20 class orders, with standard
  deviation reported at every step.

Use a paired class-level analysis and a paired seed/order analysis. Report the
fraction of classes and runs supporting each direction, not only a global mean.

## Compute tiers

### Tier A: smoke

- 1 seed, one incoming class, <=16 samples, 1-2 epochs/phase.
- Expected use: CI and wiring checks.

### Tier B: screening

- 3 seeds for base grids; 5 paired seeds for incremental comparisons.
- MNIST validation split only.
- Successive halving: stop a candidate after digit 0 or digit 3 if it is
  dominated on MSE, forgetting, and parameter count.

### Tier C: confirmation

- 10 paired MNIST seeds.
- Full train/validation protocol followed by one test evaluation.
- 20 SD-19 shuffled class orders for the published growth claim.

Record hardware and wall time before extrapolating Tier C cost from Tier B.

## Artifact and manifest requirements

Every run must persist:

- Git commit and dirty-worktree status.
- Fully resolved Hydra configuration.
- Python, PyTorch, CUDA, and dependency versions.
- Dataset source, preprocessing settings, split seed, and sample counts.
- Model checkpoint after base training and every incremental class.
- Thresholds, outlier trajectories, growth requests, and final layer sizes.
- Replay type, per-class counts, and IR statistics/quality diagnostics.
- Per-class/per-level metrics in machine-readable CSV or JSON.
- Actual updates, runtime, and peak memory.
- Failure status and traceback when incomplete.

The aggregate report must identify screening versus confirmatory runs and must
not silently omit failed seeds.

## Decision tree

1. Base model fails -> fix pretraining; stop.
2. Dataset replay fails on digit 0 -> fix local objective/update schedule; stop.
3. Dataset replay succeeds on digit 0 but fails full MNIST -> fix growth and
   stability across classes; stop.
4. Dataset replay succeeds but IR fails -> diagnose replay statistics and
   generated-sample quality.
5. MNIST succeeds but SD-19 fails -> diagnose preprocessing, scale, and
   class-order sensitivity.
6. All stages succeed -> publish confirmatory artifacts and state which claims
   are reproduced, contradicted, or unresolved.

## Current decision-tree position (2026-07-12)

The completed screening campaign is at branch 4: original-data replay succeeds
through the full MNIST curriculum, while the literal full-covariance Gaussian
IR candidate fails its matched no-replay comparison. On seeds 42--44, clean
replay improves macro validation MSE over no replay in `3/3` cases; Gaussian
IR at replay ratio `0.5` degrades it in `3/3` cases. Exact values and the frozen
configuration are recorded in
`docs/paper_compatible_optimization_plan.md`.

This evidence rejects promotion of the tested IR candidate. It does not satisfy
the five-seed IR gate, capacity-matched CL comparison, ten-seed confirmation,
or SD-19 stages, and therefore is not a numerical replication claim.

## Recommended execution order

1. Run the current unit/integration suite.
2. Execute Tier A smoke runs for all replay conditions.
3. Run Stage 1 base screening.
4. Run Stage 2 single-digit clean-replay ablations.
5. Freeze the clean-replay candidate and run Stage 3.
6. Only after Stage 3 passes, run the Stage 4 IR bridge.
7. Freeze the IR candidate and run Stage 5 confirmatory MNIST.
8. Estimate SD-19 cost with the four-letter smoke curriculum.
9. Run Stage 6 and the 20-order growth experiment.
