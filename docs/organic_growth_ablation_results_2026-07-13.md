# Organic-growth ablation results — 2026-07-13

## Scope

This campaign tested whether the implementation can recover a useful funnel
architecture from reconstruction demand instead of reproducing a shape imposed
by growth limits. Every run retained original-data replay and disabled shape
pressure. Selection used validation metrics; no test-set result was used.

The initial screen used seed 42 and classes `[0,2,3]`. To make successive
halving tractable, the plasticity/stability/next-level phases were
optimizer-step-normalized with `3/11/3` data epochs (approximately the intended
`100/500/100` minibatch-update scale for these class loaders). These are
screening results, not full-schedule or multi-seed replication estimates.

The preregistered organicity gate requires at least 80% of growth loops to stop
at the accepted outlier quota. Candidates that exhaust a growth allowance with
unresolved outliers are not organic, even when their final shape resembles the
paper.

## Main screen

| Policy | Macro MSE | Foreground MSE | Final widths | Quota-stop fraction | Unresolved exhausted loops | Updated classes | Optimizer steps |
|---|---:|---:|---|---:|---:|---|---:|
| Cumulative-cap reference | 0.03935 | 0.15251 | `[225,135,83,40]` | 0.000 | 12/12 | `0` | 7,100 |
| Absolute 1, stream ceiling 2x | 0.12159 | 0.49373 | `[250,170,91,60]` | 0.083 | 11/12 | `0,2` | 103,947 |
| Absolute 2, stream ceiling 2x | 0.10437 | 0.30728 | `[250,170,91,60]` | 0.083 | 11/12 | `0,2` | 54,275 |
| Absolute 1, class allowance `[4,5,2,3]` | 0.04186 | 0.20573 | `[212,115,81,29]` | 0.000 | 12/12 | `0,2,3` | 32,062 |
| Absolute 2, class allowance `[4,5,2,3]` | 0.04854 | 0.23149 | `[212,115,81,29]` | 0.000 | 12/12 | `0,2,3` | 21,286 |
| Absolute 1, class allowance plus loose stream ceiling | 0.04186 | 0.20573 | `[212,115,81,29]` | 0.000 | 12/12 | `0,2,3` | 32,062 |
| Proportional 0.005, stream ceiling 2x | 0.09573 | 0.42362 | `[250,170,91,60]` | 0.083 | 11/12 | `0,2` | 59,168 |

The hybrid and class-local policies were bit-for-bit identical because the
loose stream ceiling never bound. No policy passed the quota-stop gate, and all
were therefore rejected before multi-seed or full-curriculum promotion.

## Limit sensitivity

Doubling the class-local allowance from `[4,5,2,3]` to `[8,10,4,6]` changed the
endpoint from `[212,115,81,29]` to `[224,130,87,38]`. Both endpoints equal the
initial widths plus the exact per-class allowance for all three incoming
classes. Macro MSE changed only from 0.04186 to 0.04171, foreground MSE worsened
from 0.20573 to 0.20973, and updates increased from 32,062 to 56,816.

This is direct evidence that the apparent funnel is determined by the chosen
allowance, not by a demand-derived stopping point.

## Representation and threshold diagnostics

| Hidden / latent activation and criterion | Policy | Macro MSE | Foreground MSE | Final widths | Quota stops | Unresolved loops |
|---|---|---:|---:|---|---:|---:|
| Sigmoid / sigmoid, percentile 0.985 | Reference | 0.04666 | 0.19852 | `[225,135,83,40]` | 0/12 | 12/12 |
| Sigmoid / sigmoid, percentile 0.985 | Small class allowance | 0.04192 | 0.17864 | `[212,115,81,29]` | 0/12 | 12/12 |
| Sigmoid / identity, percentile 0.985 | Reference | 0.04227 | 0.18564 | `[225,135,83,40]` | 0/12 | 12/12 |
| Sigmoid / identity, percentile 0.985 | Small class allowance | **0.03520** | **0.15521** | `[212,115,81,29]` | 0/12 | 12/12 |
| Sigmoid / identity, percentile 0.995, quota 0.20 | Small class allowance | 0.03571 | 0.15721 | `[212,115,81,29]` | 0/12 | 12/12 |

Sigmoid hidden layers with an identity latent layer are performance-relevant:
the best single-seed macro MSE is 10.6% below the ReLU/identity cumulative-cap
reference (0.03520 versus 0.03935), while foreground MSE is 1.8% higher. The
change does not solve growth, however. The stricter percentile and looser 20%
outlier quota still caused every loop to consume its exact class allowance and
end with unresolved outliers.

## Mechanism audit and corrected diagnostics

The operation-level audit found that the principal paper pathway is wired as
described in Algorithm 1:

- outliers are selected using global pixel-space reconstruction error at the
  current depth;
- plasticity and stability train the corresponding isolated SHL-AE;
- mature encoder weights are frozen during plasticity, decoder weights use
  `LR/100`, and all local weights use `LR/100` during stability;
- newly added node weights and next-level connections are randomly initialized.

Two diagnostic defects were corrected. `forward_partial(..., ret_lat=True)`
used `deepcopy` on a non-leaf tensor, which fails on current PyTorch, and
shuffled subset loaders could map a reconstruction-error mask back to a
different sample ordering. Outlier evaluation now uses one explicit,
deterministic index domain, so the samples trained as outliers are exactly the
samples whose errors exceeded the threshold. The latter affected limited-data
and generic shuffled-loader diagnostics; the full MNIST runner's mutable list
sampler already used a fixed explicit ordering.

Every growth report now persists effective/base thresholds, pixel-error
quantiles, local SHL-AE error statistics, local/global error correlation, and
activation mean, variance, extrema, and saturation fractions. A corrected
smoke run verifies level-0 local/global correlation of approximately `1.0`.
At deeper levels the correlation can become weak or negative. This demonstrates
the central mechanism problem: optimizing the local SHL-AE loss does not
reliably optimize the global pixel error that decides whether growth stops.

## Non-paper post-stack fine-tuning diagnostic

The sigmoid-hidden/identity-latent class-local candidate was rerun with
`0/10/25` end-to-end epochs after stacked pretraining. All runs used seed 42,
classes `[0,2,3]`, original-data replay, no shape pressure, and the normalized
`3/11/3` growth schedule.

| Fine-tune epochs | Macro MSE | Foreground MSE | Quota stops | Unresolved loops | Mean final outlier fraction L0/L1/L2/L3 | Mean final L3 local/global correlation |
|---:|---:|---:|---:|---:|---|---:|
| 0 | 0.03519 | 0.15533 | 0/12 | 12/12 | `0.886/0.863/0.867/0.720` | 0.320 |
| 10 | 0.02810 | 0.12359 | 0/12 | 12/12 | `0.384/0.804/0.923/0.976` | 0.270 |
| 25 | **0.02749** | **0.11630** | 0/12 | 12/12 | `0.341/0.735/0.918/0.992` | 0.218 |

Twenty-five fine-tuning epochs improve macro MSE by 21.9% and foreground MSE
by 25.1% relative to zero fine-tuning. They do not improve organicity. The
base-derived level-3 threshold falls from about `0.0453` at zero epochs to
`0.0178` at 25 epochs while the mean post-growth novel-class error remains
about `0.0423`. Consequently almost every novel sample remains a level-3
outlier even though overall reconstruction improves. Fine-tuning is therefore
a useful explicitly non-paper performance extension, not a fix for emergent
growth.

## Follow-up: learned-class threshold refresh

The planned paper-compatible ambiguity test was implemented and executed. Each
threshold is recalibrated immediately before an incoming class using only
already learned training classes. The incoming class, validation split, and
test split never enter calibration. For intrinsic-replay runs the refresh uses
generated samples from frozen old-class statistics, so original old-class data
is not reopened after base training.

Across ten full-curriculum dataset-oracle seeds, refresh produced a strict
funnel of approximately `[207,105,77,23]`, with only first-layer variation to
208 or 209 nodes. The mean quota-stop fraction was `0.875`, but mean updated
class coverage was only `0.263`: seven later classes usually stopped because
they had no demand, while the early failing class still exhausted its local
allowance. The shape is therefore emergent under the fixed class-local
throttle, but it is not evidence that all classes are being successfully
acquired through growth.

The explicitly non-paper global-coupling diagnostic improved reconstruction
but did not repair update coverage. It remains excluded from the confirmatory
candidate.

## Ten-seed full-curriculum confirmation

All six frozen conditions completed seeds 42--51. `dataset_oracle` retains
original old-class training images as the requested clean upper bound;
`intrinsic` never reopens those images after base training; `no_replay` creates
no replay object. Classical networks are capacity-matched to the corresponding
NDL endpoint. Intervals are two-sided 95% Student-t intervals.

| Condition | Macro MSE (95% CI) | Foreground MSE (95% CI) | Quota stops | Updated classes | Endpoint |
|---|---:|---:|---:|---:|---|
| CL, dataset oracle | 0.01508 [0.01498, 0.01517] | 0.06091 [0.06038, 0.06143] | n/a | n/a | `[207,105,77,23]` |
| CL, intrinsic replay | 0.02367 [0.02329, 0.02405] | 0.08410 [0.08268, 0.08552] | n/a | n/a | `[232,140,91,44]` |
| CL, no replay | 0.02588 [0.02547, 0.02629] | 0.10956 [0.10654, 0.11258] | n/a | n/a | seed-matched near `[207,106,78,23]` |
| NDL, dataset oracle | 0.04733 [0.04678, 0.04788] | 0.19388 [0.18531, 0.20245] | 0.875 | 0.263 | `[207--209,105,77,23]` |
| NDL, intrinsic replay | 0.06395 [0.06226, 0.06565] | 0.24402 [0.23489, 0.25315] | 0.000 | 1.000 | `[232,140,91,44]` |
| NDL, no replay | 0.04944 [0.04830, 0.05059] | 0.19113 [0.17986, 0.20240] | 0.875 | 0.300 | seed-varying near `[207,106,78,23]` |

Intrinsic replay makes NDL macro MSE 35.1% worse than its dataset-oracle upper
bound and causes every one of 32 class/level growth loops to exhaust its local
allowance in all ten seeds. This is not a numerical failure: covariance
factorization, sampling, and decoding all complete. It is a mechanism failure
in which generated replay and replay-sourced thresholds leave every incoming
class persistently novel.

The paper says NDL+IR slightly outperforms capacity-matched CL+IR and better
preserves old representations. This implementation produces the opposite
ordering: NDL+IR macro MSE is 170.2% higher than CL+IR (`0.06395` versus
`0.02367`) and mean positive forgetting is also higher (`0.01531` versus
`0.00742`). NDL also underperforms matched CL with the clean oracle and without
replay. The paper's central comparative MNIST result therefore does **not**
replicate in this implementation.

The no-replay CL row was corrected after the initial report: the first control
used the larger intrinsic-replay endpoint. Ten reruns now use each no-replay
NDL seed's exact endpoint. The correction reduces the NDL/CL macro-MSE ratio
from the confounded `2.84x` to `1.91x` and shows a real retention trade-off:
NDL has `0.21x` the positive forgetting of matched CL, but substantially worse
total reconstruction. Clean-replay and intrinsic-replay comparisons were
already capacity matched and are unchanged.

## Final decision

The implementation follows the main operations in Algorithm 1, after fixes to
outlier identity, tensor cloning, replay isolation, threshold provenance, and
no-replay regime selection. It still cannot be called exact because the paper
does not specify optimizer, activation, thresholds, outlier quota, phase
length interpretation, or node-count policy, and the original authors' code is
not available.

The complete evidence rejects a numerical-replication claim:

1. The clean upper bound works well enough to show that data loading and basic
   reconstruction are functional, but NDL remains much worse than matched CL.
2. Learned-class threshold refresh yields a stable funnel but weak active
   update coverage; its apparent organicity is mostly later-class demand stops.
3. Literal full-covariance intrinsic replay consistently destroys demand-based
   stopping and worsens reconstruction and forgetting.
4. Because the MNIST comparison fails directionally, the conditional SD-19
   expansion is not activated; running it would not rescue the failed MNIST
   replication criterion.

Machine-readable confirmation estimates and source manifests are in
`outputs/ablations/organic_growth/confirmation_10seed_aggregate/summary.json`.
The same directory contains CSV and Markdown tables with uncertainty. Detailed
per-seed diagnostics remain in the referenced source paths recorded inside the
aggregate manifest.
