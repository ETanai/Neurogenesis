# Full replication report: Neurogenesis Deep Learning

**Date:** 2026-07-14  
**Paper:** Draelos et al., *Neurogenesis Deep Learning: Extending deep networks to accommodate new classes*  
**Replication scope:** MNIST, ten independent seeds (`42`--`51`), plus mechanism and growth-policy ablations

## Executive verdict

The paper's central MNIST comparison did **not** replicate. The paper reports
that neurogenesis with intrinsic replay (NDL+IR) slightly outperforms a
capacity-matched conventional learner with intrinsic replay (CL+IR), including
better preservation of old classes. In the replication, the paper-faithful
NDL+IR interpretation has `0.06395` macro reconstruction MSE versus `0.02367`
for matched CL+IR: NDL is **2.70 times worse**. Mean positive forgetting is
`0.01531` versus `0.00742`: NDL forgets **2.06 times more** by this measure.

There is no single honest answer to “which implementation is closest” without
specifying what must be close:

| Meaning of closest | Closest implementation | Verdict |
|---|---|---|
| Published mechanism | **NDL + intrinsic replay** | It implements demand-triggered layer growth, local SHL-AE updates, mature-weight protection, and Gaussian latent replay. It is the primary replication, but it fails the published performance ordering. |
| Published final encoder shape | **Cumulative-cap reference** (`[225,135,83,40]`) | Only one neuron from the visually estimated paper endpoint (`[225,135,~84,40]`), but the shape is imposed by the growth allowance and is not emergent evidence. |
| Best reconstruction performance | **Matched CL + original-data replay** (`0.01508`) | This is a clean-data upper-bound control, not a replication of intrinsic replay. |
| Best condition that retains no old data | **Matched CL without replay** (`0.01743`) | It outperforms both CL+IR and NDL+IR here, contradicting the paper's qualitative replay ordering. |
| Published qualitative result | **None** | No NDL replay regime beats its capacity-matched CL counterpart on macro MSE. |

The strongest defensible conclusion is therefore not that neuron growth can
never work. It is that the benefit claimed in the paper is **not robustly
identified by the publication**: plausible paper-compatible choices reproduce
the network shape but not the performance advantage. The result depends on
undocumented optimizer, activation, threshold, schedule, growth-limit, and
replay details.

## Evidence available from the original work

The repository includes the complete arXiv source bundle, including
[`NeurogenesisDeepLearning.tex`](papers/arxiv-1612.03770v2/NeurogenesisDeepLearning.tex),
the bibliography, all figures, and the original tar archive. Thus the paper is
available as LaTeX. No original implementation, raw experimental table, seed
list, or machine-readable Figure 4 values are included.

The paper specifies:

- a stacked denoising autoencoder initially trained on digits `1` and `7`;
- encoder widths `[200,100,75,20]` and mirrored decoder widths;
- incoming digits in order `0,2,3,4,5,6,8,9`, one complete class at a time;
- outliers selected by reconstruction error at each depth;
- new encoder neurons trained at full learning rate, mature encoder weights
  frozen, and old decoder weights trained at `LR/100` during plasticity;
- stabilization using the new class and intrinsic replay of older classes;
- intrinsic replay produced by sampling a class-conditional Gaussian in the
  top latent layer and decoding it;
- final capacity selected by reconstruction demand, subject to an unspecified
  maximum number of new nodes.

It does not specify exact activations, optimizer, numerical thresholds,
outlier quota, minibatch size, how reported phase counts map to epochs or
updates, maximum-node lifetime, exact next-layer update scope, number of
independent runs, uncertainty, or numeric final errors. These gaps prevent an
exact binary-equivalent reconstruction of the experiment.

### Original published result

![Original paper Figure 4](papers/arxiv-1612.03770v2/Fig4combo_sq.png)

Figure 4 qualitatively places NDL+IR below CL+IR in reconstruction error and
shows a funnel growing to approximately `[225,135,84,40]`. Because its values
are rasterized and no raw data is supplied, the replication does not claim a
numerical distance from the original curves. The statistically valid
comparison is the **direction of the paper's claim**, not fabricated precision
from pixels in a figure.

## Replication design

All confirmatory runs use MNIST, the paper's base classes and learning order,
the same initial funnel, and ten independent seeds. Evaluation uses the held-out
validation data; no test-set result was used for model selection. Reported
intervals are two-sided 95% Student-t confidence intervals over seeds.

The six-condition factorial comparison separates architecture/training from
replay quality:

| Growth/training | Replay regime | Purpose |
|---|---|---|
| Capacity-matched conventional learner (CL) | Original old-class data | Clean upper bound for ordinary training |
| CL | Intrinsic Gaussian replay | Paper's fixed-network IR control |
| CL | None | Sequential-learning baseline |
| Neurogenesis (NDL) | Original old-class data | Clean upper bound for the growth mechanism |
| NDL | Intrinsic Gaussian replay | Closest mechanistic replication of NDL+IR |
| NDL | None | Growth without stabilization replay |

For each replay regime, CL is matched to the corresponding NDL endpoint. The
original-data condition is deliberately retained: if the growth method fails
with perfect historical samples, imperfect generated replay cannot be the sole
cause.

## Main ten-seed results

Lower values are better.

| Condition | Macro MSE, mean [95% CI] | Foreground MSE | Positive forgetting | Updates | Parameters | Final encoder widths |
|---|---:|---:|---:|---:|---:|---|
| CL + original data | **0.01508** [0.01498, 0.01517] | **0.06091** | **0.000009** | 1,113 | 389,343 | `[207,105,77,23]` |
| CL + intrinsic replay | 0.02367 [0.02329, 0.02405] | 0.08410 | 0.00742 | 1,113 | 463,978 | `[232,140,91,44]` |
| CL, no replay | 0.01743 [0.01730, 0.01757] | 0.07606 | 0.00533 | 1,113 | 463,978 | `[232,140,91,44]` |
| NDL + original data | 0.04733 [0.04678, 0.04788] | 0.19388 | 0.000222 | 13,674 | 390,233 | near `[208,105,77,23]` |
| NDL + intrinsic replay | **0.06395** [0.06226, 0.06565] | **0.24402** | **0.01531** | 88,292 | 463,978 | `[232,140,91,44]` |
| NDL, no replay | 0.04944 [0.04830, 0.05059] | 0.19113 | 0.00187 | 14,668 | 390,097 | near `[207,106,78,23]` |

![Performance comparison](figures/replication/performance_comparison.png)

The confidence intervals are narrow and non-overlapping for the main
reconstruction comparisons. These are not isolated unlucky seeds. NDL macro
MSE divided by its matched CL value is:

- `3.14x` with original-data replay;
- `2.70x` with intrinsic replay;
- `2.84x` without replay.

The clean upper bound is especially diagnostic. NDL remains much worse than CL
even when both can use exact old-class images. Therefore replay sample quality
is **not sufficient** to explain the failure; the local growth/training pathway
itself is implicated.

![Claim direction comparison](figures/replication/claim_direction_comparison.png)

For forgetting, NDL without replay is better than its matched CL control
(`0.35x` the positive forgetting), but this does not translate into competitive
reconstruction: its macro MSE remains `2.84x` worse. The paper claims both good
acquisition and retention. Improving one measure while substantially degrading
the other does not reproduce that claim.

## Class-level behavior

![Per-class reconstruction comparison](figures/replication/per_class_comparison.png)

The failure is distributed across the curriculum rather than confined to one
digit. With intrinsic replay, NDL is worse on every final per-class curve than
matched CL. The largest visible discrepancy is the first incoming digit `0`,
which remains close to `0.10` MSE for NDL+IR. Base digits `1` and `7` remain
easier, but NDL still does not achieve the conventional learner's error.

This pattern is consistent with a mismatch between the quantity that triggers
growth and the quantity optimized during growth. Outliers are selected using
global pixel reconstruction error at a particular depth, whereas each growth
phase optimizes an isolated local single-hidden-layer autoencoder. Diagnostics
show that local and global errors can become weakly correlated or negatively
correlated in deeper layers. A local improvement therefore need not remove the
global outliers that keep requesting neurons.

## Does the architecture emerge?

![Capacity and growth mechanism](figures/replication/mechanism_comparison.png)

The experiments distinguish three superficially similar outcomes:

1. **Cap-driven shape.** The cumulative-cap reference reaches
   `[225,135,83,40]`, only `0.21%` relative L1 distance from the approximate
   paper endpoint. It consumes the configured allowance, so this is a
   reproduction of a shape, not evidence that demand discovered it.
2. **Threshold-refresh funnel.** With clean replay, learned-class threshold
   refresh produces about `[207,105,77,23]` and a `0.875` quota-stop fraction.
   However, only `26.3%` of incoming classes update on average. Much of the
   apparent organicity is later classes producing no demand rather than
   successful repeated growth and stopping.
3. **Intrinsic-replay growth.** The closest paper mechanism reaches
   `[232,140,91,44]`, reasonably near the visual paper shape, but every one of
   the 32 class/layer loops exhausts its local allowance in every seed. The
   endpoint is again controlled by limits, while generated replay keeps the
   incoming classes persistently novel.

The NDL+IR run uses about `79.3x` the optimizer updates of matched CL
(`88,292 / 1,113`) without a performance benefit. This is much larger than the
paper's qualitative expectation of a constant-factor processing overhead.

## Ablation lineage

These five-seed full-curriculum ablations explain how the final candidate was
chosen. They are mechanism diagnostics and should not be pooled with the
ten-seed factorial estimates.

| Variant | Paper status | Seeds | Macro MSE | Forgetting | Endpoint-distance | Updates | Growth interpretation |
|---|---|---:|---:|---:|---:|---:|---|
| Cumulative-cap reference | Compatible ambiguity | 5 | 0.04638 | 0.000003 | **0.21%** | 7,180 | Excellent shape match, allowance-driven |
| Learned-class threshold refresh | Compatible ambiguity | 5 | 0.04763 | 0.000183 | 14.83% | 13,443 | Mostly later-class demand stops |
| Refresh + global criterion coupling | Explicit non-paper diagnostic | 5 | **0.04248** | 0.000152 | 14.50% | 15,599 | Better error, but update coverage still weak |

The non-paper coupling result supports the local/global mismatch diagnosis: a
change that couples optimization more directly to the global criterion improves
macro MSE by about `10.8%` relative to ordinary threshold refresh. It still does
not establish organic growth or approach matched CL performance.

Earlier partial-curriculum screens found that sigmoid hidden layers with an
identity latent layer improved the selected reference, and 25 epochs of
end-to-end fine-tuning reduced macro MSE from `0.03519` to `0.02749`. Neither
change made growth stop organically. End-to-end fine-tuning is also not stated
in the paper and is treated as an extension, not evidence for replication.

## Fidelity assessment

| Component | Replication status | Consequence |
|---|---|---|
| Initial classes, order, architecture, stacked denoising pretraining | Matched | Core task setup is aligned |
| Global error/outlier selection at every level | Matched | Growth is driven by the paper's stated signal |
| Local SHL-AE plasticity; mature encoder frozen; decoder at LR/100 | Matched | Main neurogenesis update structure is present |
| Stabilization with new and replayed classes | Matched structurally | Exact replay balance is undocumented |
| Full-covariance class Gaussian at top latent layer, then decode | Matched | Intrinsic replay is literal rather than a modern generative substitute |
| No reopening old data in intrinsic runs | Matched and regression-tested | IR condition is no longer contaminated by clean replay |
| Activations and optimizer | Paper-compatible selection | Publication does not identify them |
| Threshold percentile and outlier quota | Paper-compatible selection | Numerically decisive and undocumented |
| Phase duration interpretation | Paper-compatible selection | Normalized to data epochs/updates; original meaning is unclear |
| Node maximum and scope | Ambiguous | Can determine the final shape directly |
| Next-layer trainable parameter scope | Ambiguous | `paper_columns` is a documented interpretation |
| Global coupling and end-to-end fine-tuning | Non-paper extensions | Diagnostic only; excluded from replication claim |

Corrected implementation defects include accidental AdamW weight decay,
mislabelled dataset replay in IR presets, old-data access during intrinsic
statistic refresh, stale replay in no-replay regimes, shuffled outlier identity,
and a tensor-cloning failure in partial forwards. Regression tests cover these
paths. Remaining differences are publication ambiguities rather than known
silent implementation errors.

## What the result says about the original approach

### Supported observations

- Reconstruction demand can trigger growth at several layers.
- A funnel resembling the published final network can be produced.
- Protecting mature weights can reduce measured forgetting in some regimes.
- Replay quality and threshold provenance materially change growth behavior.

### Observations not supported by this replication

- NDL+IR does not outperform capacity-matched CL+IR.
- NDL does not outperform matched CL even with perfect old-data replay.
- The published-looking funnel is not shown to be an optimal, organically
  discovered capacity allocation.
- Literal Gaussian intrinsic replay does not stabilize growth here; it worsens
  reconstruction by `35.1%` relative to NDL's clean-replay upper bound and
  makes all growth loops exhaust their allowance.
- The additional computation is not rewarded by better reconstruction.

### Interpretation

The biologically inspired idea remains interesting, but the published evidence
is **under-specified and fragile**. Matching the endpoint architecture is not a
validation of the algorithm because the same shape can be induced by limits.
The clean-data result indicates a more fundamental issue: locally training new
neurons under mature-weight constraints is not enough to optimize the global
reconstruction criterion used to judge capacity.

This study does not prove the original reported curves were wrong, and it is
not a universal rejection of dynamic expansion. It shows that the claim cannot
currently be reproduced from the paper alone under a carefully audited,
plausible implementation. Releasing the original code, numeric results, exact
hyperparameters, and seeds would be required to distinguish an implementation
interpretation gap from a fragile original result.

## Limitations

- Original Figure 4 has no raw values or uncertainty, so absolute numeric
  agreement cannot be scored reliably.
- The original reconstruction-error reduction and this report's normalized MSE
  may differ in scaling; directional comparisons are more trustworthy than
  comparing y-axis magnitudes across the two projects.
- Hyperparameters omitted by the paper necessarily remain interpretations.
- The confirmation covers MNIST. The preregistered SD-19 expansion was
  conditional on reproducing the central MNIST direction and was not activated
  after that condition failed.
- Results characterize this implementation and protocol, not all possible
  dynamically expandable autoencoders.

## Reproducibility and artifacts

Primary machine-readable aggregate:

- [`summary.json`](../outputs/ablations/organic_growth/confirmation_10seed_aggregate/summary.json)
- [`summary.csv`](../outputs/ablations/organic_growth/confirmation_10seed_aggregate/summary.csv)
- [`summary.md`](../outputs/ablations/organic_growth/confirmation_10seed_aggregate/summary.md)

Supporting documents and code:

- [`algorithm_fidelity.md`](algorithm_fidelity.md)
- [`organic_growth_ablation_results_2026-07-13.md`](organic_growth_ablation_results_2026-07-13.md)
- [`plot_replication_report.py`](../scripts/plot_replication_report.py)
- [`summarize_confirmation.py`](../scripts/summarize_confirmation.py)

Regenerate the diagrams with:

```bash
.venv/bin/python scripts/plot_replication_report.py
```

The plotting script verifies that each of the six conditions contains exactly
seeds `42`--`51` before producing any figure.

## Final conclusion

**NDL + intrinsic replay is the closest implementation of the published
mechanism, but it does not reproduce the published result.** The cumulative-cap
variant comes closest to the original shape, while capacity-matched CL with
original-data replay gives the best actual performance. Across every replay
regime, conventional learning reconstructs better than neurogenesis. The
original paper therefore remains an interesting proof-of-concept whose claimed
advantage is not independently established by this replication.
