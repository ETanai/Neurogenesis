# Algorithm implementation fidelity

This document audits the implementation against Algorithm 1 in Draelos et al.,
*Neurogenesis Deep Learning: Extending deep networks to accommodate new
classes* (arXiv:1612.03770v2). It distinguishes the implemented mechanism from
the settings used by the default experiment presets.

## Verdict

The repository implements the main structure of Algorithm 1. Confirmed
implementation defects found during this audit have been corrected: explicit
optimizer weight decay now defaults to zero, paper IR presets select intrinsic
replay, intrinsic runs cannot reopen old-class data, paper replay composition is
explicit, and paper presets disable the additional phase early stopper.

It is still not possible to call the implementation exact because the paper
does not specify several choices, including the optimizer family, activations,
phase lengths, thresholds, outlier quota, next-layer update scope, and
new-node-count policy.

The paper also leaves important choices unspecified. Without the original
authors' code, optimizer, activations, phase lengths, thresholds, outlier quota,
and new-node-count policy cannot be reconstructed exactly from the publication.

## Operation-by-operation mapping

| Published operation | Implementation | Status |
| --- | --- | --- |
| Compute global reconstruction error at every autoencoder level | `NGAutoEncoder.forward_partial` followed by per-sample MSE | Match |
| Select incoming-class samples whose error exceeds the level threshold | `NeurogenesisTrainer._get_outliers` | Match |
| Continue while outliers exceed `MaxOutliers` and growth remains below `MaxNodes` | Growth loop in `NeurogenesisTrainer.learn_class` | Match |
| Copy level L into an isolated single-hidden-layer autoencoder | `NGAutoEncoder.forward_level_ae` reconstructs the preceding level's representation | Match |
| Add plastic encoder neurons | `NGLinear.add_plastic_nodes` through `NGAutoEncoder.add_new_nodes` | Match |
| During plasticity, update only new encoder weights at LR | Mature encoder weights are frozen; plastic encoder weights use LR | Match |
| During plasticity, update decoder weights at LR/100 | Current-level decoder groups use `LR * plasticity_decoder_lr_ratio`; default ratio is `0.01` | Match at default ratio |
| During stability, train all current-level weights at LR/100 | Mature and plastic encoder/decoder groups use `LR * stability_lr_ratio`; default ratio is `0.01` | Match at default ratio |
| Recalculate global error and repeat growth | Outliers are recomputed after every plasticity/stability round | Match |
| Add random connections from newly grown level L into level L+1 | The next encoder's input columns and mirrored decoder structure are expanded | Match |
| Train level L+1 after lower-level growth | Next-level plasticity and stability phases are run | Partial: scope and LR are implementation choices |
| Generate replay by sampling a class Gaussian in top-level latent space and decoding it | `IntrinsicReplay` stores a mean and Cholesky factor, samples a Gaussian, and decodes it | Match |
| Stabilize with the new class and replay from previous classes | Supported by `stability_replay_mode=paper` | Partial: default mode is `ratio` |
| Operate when original old-class data is unavailable | Supported when old statistics are retained | Default mismatch: statistics are refreshed from old real data |

## Gap remediation record

Dataset replay itself is intentionally retained. It is a useful clean-replay
upper bound: if an NDL training schedule cannot learn and retain classes when
given original historical samples, replacing those samples with imperfect
intrinsic replay cannot be expected to repair the underlying optimizer or
growth behavior. The fidelity issue is therefore not the existence of dataset
replay; it is using or labeling dataset replay as though it were the paper's
intrinsic-replay condition.

### 1. Implicit AdamW weight decay

**Status: resolved.** Neurogenesis optimizers now receive the configured weight
decay explicitly, with `0.0` as the paper-oriented default. A regression test
checks both zero and nonzero configured values.

`NGAutoEncoder._optim_lr_config` constructs `AdamW(param_groups)` without an
explicit `weight_decay`. PyTorch therefore uses AdamW's default weight decay of
`0.01` during plasticity and stability. The configured
`training.weight_decay: 0.0` is not passed into these phase optimizers.

Algorithm 1 specifies LR and LR/100 parameter updates but does not specify
AdamW or decoupled weight decay. The executed update equation is consequently
not a literal translation of the publication.

Resolution:

- Pass an explicit, configured weight decay to all neurogenesis optimizers.
- Use zero for a literal interpretation unless evidence for another value is
  recovered from the authors.

### 2. Non-paper default replay composition

**Status: resolved for paper presets.** General experiments retain configurable
ratio replay, while NDL+IR paper presets explicitly select `paper` mode.

The paper constructs stabilization data from the incoming class and replayed
samples from previously seen classes. The implementation has a `paper` mode
that requests samples from every stored old class, but
`config/neurogenesis/default.yaml` selects `stability_replay_mode: ratio`.

Ratio mode draws a mixed replay batch and does not guarantee representation of
every previous class in each stabilization batch.

Resolution:

- Set `neurogenesis.stability_replay_mode=paper` in literal paper runs.
- Record the per-class sample ratio because the publication does not provide an
  exact value.

### 3. Old-data access during intrinsic-statistic refresh

**Status: resolved.** The runner now structurally limits intrinsic replay
refreshes to the incoming class and preserves prior statistics. Dataset replay
continues to refresh all historical clean samples as its intended upper bound.

In intrinsic-replay runs, `scripts/run_experiments.py` defaults to recomputing
replay statistics for every learned class from the original datasets. This
conflicts with the paper's intended condition that original old-class data is
no longer available and only stored latent statistics remain. It remains valid
and desirable for an explicitly named dataset-replay control.

Resolution:

- Set `replay.reuse_previous_stats=true` for paper-faithful runs.
- Fit statistics only for the incoming class after it has been learned.
- Add an integration test that fails if loaders for an old class are accessed
  after that class leaves an intrinsic-replay stream.
- Preserve unrestricted old-sample access in the dataset-replay upper-bound
  condition.

### 4. Replay type and experiment-label separation

**Status: resolved.** IR paper presets explicitly select intrinsic replay, and
the paper runner rejects IR-labeled configurations that resolve to dataset
replay or allow old-stat refresh. Explicit dataset-replay controls remain.

`config/replay/default.yaml` selects `mode: dataset`. This is appropriate for a
clean-data upper-bound default. However, MNIST presets labeled CL+IR and NDL+IR
enable replay without consistently overriding this mode, so the runner can
instantiate `DatasetReplay` while reporting an intrinsic-replay label.

Resolution:

- Set `replay.mode=intrinsic` explicitly in every paper preset labeled IR.
- Keep separate, explicitly named `*_dataset_replay` presets as diagnostic
  upper bounds.
- Log and validate the concrete replay class at run startup.
- Reject an IR-labeled paper configuration that resolves to dataset replay.

## Replay testing ladder

Replay conditions should be evaluated in the following order so failures can
be localized:

1. **No replay:** establishes catastrophic-forgetting behavior and tests NDL
   growth without stabilization data.
2. **Dataset replay:** uses clean original historical samples as an upper bound
   for the training, growth, and stability implementation.
3. **Intrinsic replay:** replaces historical samples with decoded Gaussian
   latent samples and measures the additional loss caused by replay quality.

Interpretation:

- Failure with dataset replay points to the model, optimizer, growth rule,
  phase schedule, or replay composition—not to IR sampling quality.
- Success with dataset replay but failure with IR isolates the replay generator,
  stored statistics, sampling distribution, or decoder quality.
- Dataset replay results must not be presented as evidence that the paper's IR
  condition has been reproduced.

### 5. Next-layer training is not uniquely determined by the paper

**Status: irreducible paper ambiguity.** Paper presets select and log
`paper_columns`; `broad` remains available for comparative testing.

After level L grows, the publication directs the algorithm to add random
connections into level L+1 and train that next single-hidden-layer
autoencoder. It does not state the exact learning rate or which existing
next-level parameters remain trainable.

The repository supports:

- `broad`: train all current next-level parameters.
- `paper_columns`: train only newly appended encoder columns and matching
  decoder parameters during plasticity.

The default also multiplies the base learning rate by
`next_layer_lr_ratio=0.01`. These are documented interpretations, not confirmed
original settings.

Required handling:

- Treat this as an explicit replication uncertainty.
- Report the selected mode and LR ratio with every result.
- Compare both interpretations rather than describing either as exact.

### 6. Extra phase early stopping

**Status: resolved for paper presets.** Paper NDL configurations set
`neurogenesis.early_stop=null`. Early stopping remains available as an explicit
extension for non-paper and ablation runs.

The published growth loop terminates based on the outlier quota or the
per-level node cap. The implementation also applies loss-delta patience and
phase-specific threshold goals inside plasticity and stability.

These rules do not remove the outer published stopping conditions, but they can
shorten each training phase and therefore change the next outlier measurement.

Resolution:

- Disable phase early stopping for a literal fixed-duration interpretation, or
  identify it clearly as an extension.
- Report actual optimizer steps, not only configured epoch limits.

### 7. New-node count is repository-defined

**Status: irreducible paper ambiguity.** The policy remains configurable and is
logged; it is not described as a formula recovered from the paper.

Algorithm 1 leaves `Nodes_New` unspecified. The default implementation adds a
number proportional to the current outlier count, subject to a cap derived from
the existing layer size and the remaining `MaxNodes` allowance.

This is a reasonable completion of an underspecified step, but it is not a
recoverable exact setting from the paper.

Required handling:

- Record the growth mode, factors, requested node count, and applied node count
  for every round.
- Avoid presenting the proportional policy as a published formula.

## Experiment-protocol gaps that affect algorithm comparisons

These are not defects in the reusable model classes, but they prevent the
shipped paper workflow from establishing numerical replication:

- Figure 4 now runs each NDL source before its corresponding CL control and
  injects the measured final layer sizes into that fixed-capacity control.
- `config/paper/sd19_growth_20.yaml` runs 20 deterministic, independently
  shuffled letter curricula with dataset replay for the Figure 8 protocol.
- The SD-19 NDL+IR preset now selects intrinsic replay; the separate SD-19 NDL
  dataset-replay preset remains the clean-data control.
- No checked-in result manifest demonstrates agreement with the paper's
  reconstruction-error curves, final layer sizes, or figures.

## Closest available configuration

`scripts/run_paper_spec_training_ablation.py` defines a paper-locked baseline
with the following important settings:

```text
neurogenesis.thresholds=null
neurogenesis.objective_mode=paper_level_ae
neurogenesis.next_layer_optimization=paper_columns
neurogenesis.plasticity_decoder_lr_ratio=0.01
neurogenesis.stability_replay_mode=paper
replay.mode=intrinsic
replay.ir_sampling_mode=gaussian_full
replay.ir_cov_shrinkage=0.0
replay.ir_noise_scale=1.0
training.pretrain_mode=stacked_denoising
training.pretrain_finetune_epochs=0
```

For the closest currently expressible run, also set:

```text
replay.reuse_previous_stats=true
```

This uses explicit zero weight decay. Settings omitted from the publication
remain declared replication assumptions.

## Verification status

Focused tests confirm that:

- Plasticity updates new encoder parameters while mature encoder parameters
  remain frozen.
- Stability updates mature and plastic parameters.
- Paper replay mode samples every stored old class.
- Paper-column next-level mode masks existing encoder columns.
- Outlier quotas and node caps control growth.

These are behavioral conformance tests. They do not demonstrate numerical
reproduction of published experiments.

## Criteria for claiming faithful replication

Before describing results as a faithful replication, the repository should:

1. Correct the confirmed optimizer and replay-path gaps above.
2. Freeze and serialize a resolved configuration for every paper figure.
3. Preserve dataset replay as a clean-data control while preventing access to
   original old-class samples in intrinsic-replay runs.
4. Generate capacity-matched controls from completed NDL runs.
5. Execute the full MNIST curriculum and all required SD-19 permutations.
6. Publish seeds, dataset preprocessing, dependency versions, layer growth,
   reconstruction errors, and generated figures.
7. Compare those artifacts quantitatively with values recoverable from the
   publication and label all irreducible ambiguities.
