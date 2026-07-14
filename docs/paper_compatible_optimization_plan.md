# Paper-compatible iterative optimization plan

This protocol finds the best performance obtainable without changing mechanisms
that the Neurogenesis Deep Learning paper explicitly specifies. It begins with a
renewed congruency audit, then uses staged ablations to identify relevant
undocumented variables, optimize them, and confirm the result across paired
seeds.

The clean dataset-replay condition is optimized first as an oracle control.
Intrinsic replay is introduced only after clean replay passes its acquisition,
retention, outlier, and growth gates.

## Fidelity boundary

Variables in the paper-compatible table of
`docs/performance_tuning_variables.md` may be optimized. Paper-specified
variables remain locked during the main search.

The following are locked:

- MNIST base classes `[1,7]` and incremental order `[0,2,3,4,5,6,8,9]`.
- Initial architecture `[200,100,75,20]` and mirrored decoder.
- Stacked denoising layer-wise base training.
- Global reconstruction error for selecting outliers.
- Isolated single-hidden-layer AE objective at each level.
- Mature encoder weights frozen during plasticity.
- New encoder weights at LR and decoder weights at LR/100 during plasticity.
- All current-level weights at LR/100 during stability.
- New lower-level connections propagated into the next level.
- Full-covariance per-class Gaussian intrinsic replay through the decoder.
- No access to old original samples in intrinsic-replay runs.
- Current class plus replay from every old class during stability.
- Capacity-matched CL controls.

Any experiment that changes a locked item belongs in a separate
`explicit_nonpaper_ablation` family and cannot win the paper-compatible search.

## Phase 0: renewed paper/implementation congruency audit

No optimization result is accepted until this audit passes. Record every item
as `match`, `compatible interpretation`, `mismatch`, or `paper unspecified`.

### 0A. Static mapping audit

| Area | Exact check | Failure consequence |
| --- | --- | --- |
| Layer numbering | Paper level L maps to encoder L and decoder N+1-L without an off-by-one error | Wrong local objective and threshold |
| Global reconstruction | `RE_Global,L` encodes through levels 1..L and decodes through the corresponding path to pixels | Growth responds to a different error |
| Local SHL target | Level 1 reconstructs pixels; deeper levels reconstruct the frozen preceding representation | Gradients optimize the wrong space |
| Detachment | Representations below the trained level are detached during local training | Earlier mature features silently change |
| Plasticity mask | Only new encoder rows are trainable at LR | Catastrophic drift in mature features |
| Decoder plasticity | Decoder parameters specified by the paper receive exactly LR/100 | Wrong stability/plasticity balance |
| Stability mask | All parameters of the current SHL-AE train at LR/100 | Incomplete stabilization |
| Structural wiring | Growing level L expands its mirror decoder input and the next encoder input consistently | Shape-correct but disconnected neurons |
| Next-level training | The newly appended columns participate in the next SHL-AE objective | Growth cannot influence deeper levels |
| Promotion timing | Plastic nodes remain plastic through their intended phase and become mature before the next class/level requires them frozen | New neurons freeze too early or train too long |
| Outlier source | Only samples from incoming class `D_U` determine growth | Replay classes incorrectly cause growth |
| Outlier comparison | Per-sample MSE and threshold use identical reduction and pixel scale | Thresholds are dimensionally incompatible |
| Growth stopping | Loop stops only at accepted outlier count or `MaxNodes` in paper-locked runs | Extra stopping changes Algorithm 1 |
| Replay membership | Stability includes current samples and every stored old class | Missing classes bias retention |
| IR statistics | Mean and unbiased full covariance are computed at the top encoder representation | Wrong replay distribution |
| IR sampling equation | Sample is `mu + epsilon @ Cholesky.T`, then decoded once | Transposed/incorrect covariance |
| Old-data isolation | After bootstrap, intrinsic replay fits only the incoming class | Information leakage |
| Clean replay control | Dataset replay retains all historical originals and is labeled oracle/control | Clean replay confused with IR |
| Control capacity | CL/CL+IR use fixed sizes measured from corresponding completed NDL runs | Capacity confounds comparison |

### 0B. Numerical semantics audit

Check details that can create large differences while code still appears
structurally correct:

1. **MSE reduction:** confirm threshold estimation, outlier selection, phase
   loss, evaluation, and reported metrics all use mean squared error per sample,
   not a mixture of sum and mean.
2. **Pixel scale and polarity:** confirm MNIST and resized SD-19 are in `[0,1]`
   with matching foreground/background polarity.
3. **Epoch meaning:** report batches and optimizer steps per epoch for every
   phase. The paper says training phases, but batch size changes the update
   budget.
4. **Learning-rate assignment:** print optimizer parameter names, shapes, LR,
   weight decay, and trainability for plasticity, stability, and next-level
   phases.
5. **Optimizer state after growth:** confirm newly replaced/expanded parameters
   appear in a newly constructed optimizer and no stale parameter references
   remain.
6. **Weight decay:** assert paper-compatible optimizers use the declared value,
   initially zero.
7. **Activation/loss compatibility:** record saturation and dead-unit fractions
   for every level.
8. **Bias treatment:** verify biases follow the same mature/plastic masks as
   their corresponding weights.
9. **Data counts:** report exact train/validation/test samples per class and
   ensure incoming limits are disabled for confirmatory runs.
10. **Randomness:** seed model initialization, data order, corruption, growth
    initialization, and replay sampling.

### 0C. Behavioral conformance probes

Add or run deterministic tests that demonstrate:

- a plasticity step changes new encoder rows and decoder parameters but not old
  encoder rows;
- a stability step changes all current-level groups at exactly LR/100;
- local training at level L cannot change parameters below L;
- sampled IR covariance converges to the stored covariance in a synthetic
  linear encoder/decoder test;
- intrinsic refresh raises an error if an old-class loader is requested;
- all old classes appear in paper-style replay with the expected counts;
- growing each level changes the intended tensor dimensions and produces a
  nonzero gradient through every new connection;
- outlier indices still reference the correct samples when loaders are shuffled
  or backed by `Subset`;
- a saved/reloaded grown model preserves structure and outputs;
- identical seeds reproduce thresholds, growth, and final metrics.

### 0D. Paper ambiguity register

Freeze a written interpretation before tuning for:

- optimizer family;
- hidden/latent/output activations;
- corruption type and amount;
- phase lengths;
- batch size;
- threshold derivation;
- `MaxOutliers` and `MaxNodes`;
- `Nodes_New` calculation;
- next-level trainable subset and learning rate;
- replay samples per old class and minibatch ordering.

Do not silently change an interpretation after seeing test performance. A
change creates a new candidate family and must be documented.

## Phase 1: establish measurement baselines

Run three seeds for each baseline under the same commit and split:

1. Untrained model.
2. Base-only stacked denoising AE.
3. Fixed full-data AE ceiling trained on classes `[1,7,0]`.
4. NDL + clean dataset replay on `[1,7] -> 0`.
5. NDL without replay on `[1,7] -> 0`.

Historical constants embedded in existing scripts are not acceptance criteria;
recompute them after the congruency audit.

Metrics:

- macro and per-class validation MSE;
- incoming-class acquisition gain;
- base-class forgetting;
- final outlier fraction by level;
- cap hits and final layer sizes;
- optimizer updates and runtime.

## Phase 2: screen variable relevance

Use clean replay, one incoming digit (`0`), and three paired seeds. Start from
the best audited base configuration. Change one variable family at a time.

### 2A. Base representation

| Variable | Screening values |
| --- | --- |
| Base LR | `3e-4`, `1e-3`, `3e-3` |
| Epochs per level | `28`, `50`, `100` |
| Dropout corruption | `0.05`, `0.1`, `0.15` |
| Gaussian corruption | `0`, `0.025`, `0.05` |
| Batch size | `64`, `128`, `256` |
| Optimizer | Adam, AdamW with decay zero |

First screen base variables using base-only validation. Promote at most two base
configurations to incremental testing.

### 2B. Paper-compatible NDL schedule

Keep stability LR fixed at the paper value `LR/100`.

| Variable | Screening values |
| --- | --- |
| Plasticity epochs | `50`, `100`, `250`, `500` |
| Stability epochs | `100`, `250`, `500`, `1000` |
| Next-level epochs | `50`, `100`, `250`, `500` |
| Next-level LR ratio | `0.01`, `0.03`, `0.1`, `1.0` |
| Replay per old class | `0.5`, `1.0`, `2.0` current batches |
| Replay schedule | mixed, interleave epochs, interleave batches |

Because the first experiments identified stability as the bottleneck, screen
stability duration first. Test next-level settings only after selecting a
stability duration.

### 2C. Threshold and growth policy

| Variable | Screening values |
| --- | --- |
| Threshold percentile | `0.95`, `0.975`, `0.985`, `0.995` |
| Accepted outlier fraction | `0.05`, `0.10`, `0.20`, `0.30` |
| New nodes per round | absolute `1`, `2`; proportional `0.002`, `0.005`, `0.01` |
| Per-round growth cap | `0.05`, `0.1`, `0.2` of existing nodes |
| Level caps | baseline `[25,35,8,20]`, doubled cap, relaxed level-2 cap |
| `MaxNodes` scope | per incoming class; cumulative across the class stream |

Evaluate threshold and growth variables jointly through Pareto dominance:
lower MSE and forgetting are better; fewer parameters, cap hits, and updates
are better. Do not optimize only for architecture resemblance to a published
plot.

The scope ablation is required by an ambiguity in Algorithm 1. `NewNodes` is
reset inside a single-class invocation, but Figure 4F ends near
`[225,135,84,40]`, approximately the initial `[200,100,75,20]` architecture
plus one `[25,35,~9,20]` allowance. In contrast, treating the vector as a fresh
allowance for every class produced full-stream architectures around
`[275,251,331,340]` and `[284,310,331,300]` in seeds 42 and 43. Architecture
agreement alone is not a promotion criterion, but this order-of-magnitude deep
growth difference makes scope a performance-critical undocumented detail.

### 2D. Organic-shape and cap-invariance study

The current cumulative-cap configuration is a cap-driven reference, not
evidence that the final architecture emerged from demand. With initial widths
`[200,100,75,20]` and allowances `[25,35,8,20]`, exhausting every allowance
forces `[225,135,83,40]`, independently of whether that capacity is optimal.
In the completed runs, most capacity was consumed on digit 0, the remaining
level-2 allowance was consumed on digit 2, and no parameters changed for later
digits. The approximate agreement with Figure 4F must therefore not be scored
as organic growth.

Run the following paired clean-replay conditions with shape pressure disabled.
The maximum is only an emergency ceiling; it must not be used as a desired
width.

| Family | New nodes per round | Growth allowance | Purpose |
| --- | --- | --- | --- |
| Current cap-driven reference | proportional `0.01` | cumulative `[25,35,8,20]` | Reproduce the current result and measure cap saturation |
| Fine-grained cumulative | absolute `1`, `2`; proportional `0.0005` | cumulative `2x` and `4x` current allowances | Test whether small growth steps stop before a loose ceiling |
| Small class-local throttle | absolute `1`, `2` | per-class `[4,5,2,3]` and `[8,10,4,6]` | Permit later classes to request capacity without allowing one class to cause explosive growth |
| Loose ceiling plus class throttle | absolute `1`, `2` | cumulative `4x`, with per-class throttles above | Separate the stream-wide safety ceiling from the amount one class may consume |

For the fine-grained families, test threshold percentiles `0.985` and `0.995`
and accepted outlier fractions `0.10` and `0.20`. Keep the local objective,
paper learning-rate ratios, replay contents, initial architecture, and all other
paper-locked mechanisms fixed. Do not use an explicit monotonic-width rule,
layer-ratio target, architecture loss, or a rule preventing a deeper layer from
becoming wider than its predecessor.

Use paired seeds `42`, `43`, and `44` for `[0,2,3]`, then promote at most three
families to the full digit stream. Confirm the winner on five seeds and at
least five shuffled incremental class orders. Always retain the published
class order as the primary comparison.

Record after every incoming class and level:

- width before and after training, nodes requested/accepted, and cap
  utilization;
- pre/post incoming-class MSE, outlier fraction, and exact stopping reason;
- parameter-update count and norm, including an assertion that later classes
  did not silently receive zero updates;
- macro-MSE, foreground-weighted MSE, per-class forgetting, parameter count,
  runtime, and marginal validation gain per added node;
- monotonic-width violations and distance from the approximate paper endpoint
  `[225,135,84,40]`, reported only after performance-based selection.

An architecture is considered demand-emergent only if all of these hold:

1. At least 80% of level/class growth loops stop at the accepted outlier quota,
   rather than a cap or a round limit.
2. Doubling the loose cumulative ceiling changes each final layer's added width
   by at most 10% and changes macro-MSE by at most 5%.
3. At least half of the incremental classes cause a measurable parameter update
   in two of three screening seeds; growth is not exhausted by digits 0 and 2.
4. The same qualitative funnel appears in at least four of five confirmatory
   seeds and most shuffled orders with no shape pressure.
5. It matches or improves the cap-driven reference on paired macro-MSE and
   forgetting, or stays within 5% while using at least 15% fewer added
   parameters or optimizer updates.

The approximate paper endpoint and the absence of a level-4-over-level-3
inversion are secondary morphology checks. A candidate that resembles the
paper only because it exhausts its allowance fails this study. If no candidate
passes, report that the implementation does not currently recover the paper's
shape organically; do not silently restore the current allowances as if they
were learned.

The implemented runner is `scripts/run_organic_growth_ablation.py`. Its
`screen`, `invariance`, and `full` stages retain original-data replay and emit
the persistent class reports needed by these gates. Run `--stage all` when
paired cap-invariance annotations should be computed in one output set.

## Relevance decision rule

A variable is relevant if, across three paired seeds, at least one tested value:

- changes macro-MSE or forgetting by at least 5% relative;
- changes incoming-class MSE by at least 10%;
- reduces cap hits in at least two seeds without >5% MSE degradation; or
- reduces parameters/updates by at least 15% without >5% MSE degradation.

Also require directional consistency in at least two of three seeds. Variables
that fail this rule are frozen at the cheaper or more paper-conservative value.

Use paired differences, not independent run averages.

## Phase 3: optimize relevant variables

After screening, optimize only relevant variables. Avoid a full Cartesian grid.

### Iteration procedure

1. Rank relevant variables by standardized paired effect and uncertainty.
2. Optimize the strongest variable while holding others at the incumbent.
3. Promote a new incumbent only if it Pareto-dominates or improves the scalar
   validation score by at least 3% in two of three seeds.
4. Re-screen the next variable around the new incumbent.
5. After one pass, test pairwise interactions only among the top three
   variables.
6. Stop after two consecutive iterations fail to improve the incumbent by 3%.

### Validation score

Use normalized validation quantities:

```text
score = macro_mse
      + 0.5 * mean_positive_forgetting
      + 0.25 * worst_class_mse
      + 0.002 * cap_hit_count
      + 1e-5 * added_parameters
```

Report every component. The scalar score selects candidates but never replaces
the Pareto table.

### Successive halving

- Round 1: 3 seeds, digit 0, limited samples if needed.
- Round 2: 3 seeds, digits `[0,2,3]`, all samples.
- Round 3: 5 seeds, full MNIST curriculum.
- Confirmation: 10 paired seeds, frozen configuration, one test evaluation.

Discard a candidate early only when it is dominated on MSE, forgetting, cap
hits, and parameter count.

## Phase 4: clean-replay promotion gate

Before intrinsic replay, the optimized clean-replay candidate must satisfy:

- positive acquisition gain for every incoming digit;
- final base-class MSE no more than 20% above the base checkpoint;
- at least 80% of level/seed pairs terminate below the outlier quota rather
  than at `MaxNodes`;
- no NaN/Inf, collapsed latent representation, or disconnected new node;
- lower paired macro-MSE or forgetting than capacity-matched CL + dataset
  replay in at least four of five seeds;
- no result depends on access to test data during selection.

If LR/100 stability cannot pass after optimizing its undocumented duration,
batch size, replay quantity, threshold, growth, and next-level settings, record
that the literal paper rule is not reproduced. Do not substitute a larger
stability LR inside the paper-compatible candidate.

## Phase 5: optimize intrinsic replay without retuning NDL

Freeze the promoted clean-replay NDL settings. For the main paper-compatible IR
candidate, keep full covariance, noise scale 1, and no filtering.

Tune only undocumented IR details:

- covariance epsilon: `1e-6`, `1e-5`, `1e-4`, `1e-3`;
- latent-stat sample count: `512`, `1024`, `4096`, all;
- replay samples per class: ratios `0.5`, `1`, `2`;
- mixed versus interleaved stability scheduling.

Diagonal covariance, shrinkage, filtering, altered noise scale, and old-data
refresh are separate non-paper diagnostics.

Promotion gate:

- IR outperforms no replay on MSE and forgetting in four of five seeds;
- IR retains at least 70% of clean replay's improvement over no replay;
- every old class receives replay in every paper stability phase;
- no old original loader is accessed;
- Cholesky failures and invalid samples are zero.

## Phase 6: confirmatory comparisons

Freeze all variables before test evaluation.

Run 10 paired MNIST seeds:

- CL;
- NDL;
- CL + IR;
- NDL + IR;
- NDL + dataset replay as a separately labeled oracle.

Then run the SD-19 smoke gate and 20 shuffled class orders only if MNIST passes.

Report paired bootstrap 95% intervals, seed-level direction counts, per-class
MSE/forgetting, growth, parameters, updates, runtime, and memory.

## Separate non-paper diagnostic branch

Maintain a parallel branch for explicitly specified mechanisms whose relaxation
may explain failures:

- stability LR ratios `0.03`, `0.1`, `0.3`, `1.0`;
- decoder plasticity ratios above `0.01`;
- mature encoder updates during plasticity;
- global coupling or end-to-end fine-tuning;
- non-Gaussian or covariance-shrunk replay;
- local or dual growth criteria.

These runs diagnose sensitivity but never promote into the paper-compatible
incumbent. The existing full-LR stability result belongs here.

## Run manifest

Each candidate must record:

- fidelity label and locked-variable checksum;
- Git commit and dirty state;
- resolved Hydra configuration;
- data split, counts, preprocessing, and seed;
- optimizer parameter groups and update counts;
- thresholds and outlier trajectories;
- replay class/sample counts and old-data access audit;
- checkpoints, layer sizes, MSE, forgetting, growth, runtime, and failures.

## Recommended first iteration

The next runs should keep the best screened base settings (`LR=1e-3`, 50
epochs/level, dropout `0.1`) and dataset replay, then proceed in this order:

1. Fix the run evidence: persist per-class architecture, parameter deltas,
   update counts, stopping reasons, and a `cap exhausted with unresolved
   outliers` flag. Fail a paper run when unresolved outliers coexist with zero
   possible updates.
2. Reproduce the current cumulative-cap reference on seeds 42--44 and verify
   the observed early capacity exhaustion.
3. Run the Phase 2D `[0,2,3]` organic-shape screen. Start with absolute `1` and
   `2`, small class-local allowances `[4,5,2,3]`, loose cumulative ceilings at
   `2x` and `4x`, and percentiles `0.985`/`0.995`.
4. Run the explicit cap-invariance pair for each promoted candidate by doubling
   only its stream-wide ceiling. Reject candidates whose shape tracks the cap.
5. Compare promoted organic candidates to the current reference over the full
   curriculum using paired MSE, foreground error, forgetting, update coverage,
   capacity, and runtime. Select by performance and organicity gates, not
   endpoint resemblance.
6. Only after clean replay learns throughout the curriculum, resume schedule
   screens (stability duration and next-level LR) and then diagnose intrinsic
   replay.

## Completed screening campaign (2026-07-12)

This section records the completed validation-screening branch. It is not a
Tier C confirmatory replication: model selection used validation results and
the matched comparison has three seeds. No paper claim should be inferred
beyond that scope.

### Frozen incumbent

The best paper-compatible screening configuration uses base LR `1e-3`, stacked
denoising pretraining for 50 epochs per level with dropout `0.1`, no global
fine-tuning, plasticity/stability/next-level durations `100/500/100`,
next-level LR ratio `0.01`, threshold percentile `0.985`, mixed stability
scheduling, replay ratio `1.0`, and cumulative global growth allowances
`[25,35,8,20]`. All completed full-curriculum runs ended at
`[225,135,83,40]`.

Treating `MaxNodes` as a fresh allowance for every incoming class produced
architectures `[275,251,331,340]` and `[284,310,331,300]` with validation MSE
`0.05677` and `0.06538`. The cumulative interpretation produced clean-replay
MSE `0.03651`, `0.04080`, and `0.04608` for seeds 42--44: mean `0.04113`,
sample SD `0.00479`. On the two paired seeds it improved over the per-class
interpretation by 36.7% on average. This is performance evidence for the
cumulative interpretation, not proof that the paper intended it.

Threshold `0.995` gave seed-42 MSE `0.03680`, 0.8% worse than the `0.985`
incumbent, although it delayed some growth into later classes. Replay ratio
`0.5` was 7.1% worse than ratio `1.0` in the screen, while ratio `2.0` was
50.9% worse. Batch and epoch interleaving were respectively 75.6% and 102%
worse at the digit-0 screen, so mixed scheduling was retained.

### Matched replay-source result

| Source | Seed 42 MSE | Seed 43 MSE | Seed 44 MSE | Mean | Sample SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original-data replay oracle | 0.03651 | 0.04080 | 0.04608 | 0.04113 | 0.00479 |
| No replay | 0.04408 | 0.04659 | 0.06207 | 0.05091 | 0.00974 |
| Full-covariance Gaussian IR, ratio 0.5 | 0.05926 | 0.05574 | 0.06405 | 0.05968 | 0.00417 |

Original-data replay beat no replay in all three paired seeds, by 17.2%, 12.4%,
and 25.8% (mean relative improvement 18.4%). Gaussian IR was worse than no
replay in all three seeds, by 34.4%, 19.6%, and 3.2% (mean degradation 19.1%),
and worse than clean replay by 62.3%, 36.6%, and 39.0% (mean degradation
46.0%). Ratio `1.0` also failed on seed 42 with MSE `0.06144`; ratio `0.5` was
therefore the stronger literal IR setting.

The Phase 5 promotion gate fails: Gaussian IR has a `0/3` direction count
against no replay and retains none of clean replay's improvement. Ratio `2.0`
was not promoted because increased replay was already strongly dominated in
the clean replay screen. Further covariance epsilon/sample-count tuning cannot
be described as justified optimization until replay quality diagnostics expose
a numerical failure rather than a limitation of the paper's Gaussian replay
assumption.

### Stability and remaining scope

The clean-replay and replay-source directions are stable at three-seed
screening strength: all conditions share the same final architecture, clean
replay wins every pair, and IR loses every pair. Three seeds remain too few for
confirmatory statistical evidence. Subsequent trajectory inspection also
showed that these endpoints do not demonstrate successful learning throughout
the curriculum: most growth allowances were exhausted on digit 0, the final
remaining growth occurred by digit 2, and later digits caused no model update.
The final `[225,135,83,40]` shape is therefore currently cap-driven rather than
shown to be demand-emergent. The Phase 2D study is required before interpreting
the layer-growth result.

## Completed confirmation update (2026-07-14)

The capacity-matched controls, five-seed IR gate, and ten-seed MNIST
confirmation are now complete. Learned-class threshold refresh was frozen as
the paper-compatible undocumented-detail candidate after mechanism screens.
All six conditions completed seeds 42--51.

- NDL dataset-oracle MSE: `0.04733` (95% CI `[0.04678,0.04788]`).
- NDL intrinsic-replay MSE: `0.06395` (`[0.06226,0.06565]`).
- NDL no-replay MSE: `0.04944` (`[0.04830,0.05059]`).
- Matched CL dataset-oracle MSE: `0.01508` (`[0.01498,0.01517]`).
- Matched CL intrinsic-replay MSE: `0.02367` (`[0.02329,0.02405]`).
- Matched CL no-replay MSE: `0.01743` (`[0.01730,0.01757]`).

Clean NDL and no-replay NDL produce seed-varying funnels near
`[207,106,78,23]`, but update only 26.3% and 30.0% of incoming classes on
average. Intrinsic NDL instead exhausts all 32 class/level allowances in every
seed and ends exactly at `[232,140,91,44]`. NDL+IR is 170.2% worse in macro MSE
than matched CL+IR, reversing the paper's reported ordering. The MNIST
replication therefore fails. The conditional SD-19 expansion is not activated.

Full estimates and source manifests are in
`outputs/ablations/organic_growth/confirmation_10seed_aggregate/summary.json`.
