# Performance tuning variables and paper-fidelity boundary

This document separates performance variables into two groups:

1. **Paper-compatible tuning:** the publication requires the mechanism but does
   not provide the exact value or implementation detail. These variables may be
   tuned while retaining a paper-compatible interpretation, provided every
   resolved value is reported.
2. **Paper-specified variables:** the publication explicitly defines the value,
   rule, architecture, or protocol. Changing one may improve performance, but
   the resulting run must be labeled as an extension or ablation rather than a
   literal replication.

The distinction concerns fidelity, not scientific usefulness. Both groups are
valuable to test, but their results must not be combined under the same label.

## 1. Paper-compatible tuning of undocumented details

| Variable | Current/default value | Suggested values | Expected effect and risk | Selection metric |
| --- | --- | --- | --- | --- |
| Base learning rate | `1e-4` in shared config; best screen used `1e-3` | `1e-4`, `3e-4`, `1e-3`, optionally `3e-3` | Higher values accelerate local learning; excessive values can destabilize layer-wise training | Base validation MSE and variance across seeds |
| Base epochs per level | `14`; best screen used `50` | `14`, `28`, `50`, `100` | More epochs can improve representation but may overfit; 100 epochs was worse than 50 in the first screen | Base 1/7 macro-MSE and optimizer updates |
| Denoising corruption amount | Dropout `0.1`, Gaussian std `0.0` | Dropout `0.05`, `0.1`, `0.15`; std `0.0`, `0.025`, `0.05` | Controls robustness of learned features; both zero and dropout `0.2` performed worse in the first screen | Validation MSE, latent variance, clean/noisy reconstruction gap |
| Batch size | `128` | `64`, `128`, `256` | Changes gradient noise and number of optimizer updates; also changes replay count when replay is expressed per batch | MSE per optimizer update, runtime, memory |
| Optimizer family | Adam for pretraining; AdamW for NDL phases | Adam, AdamW with explicit decay `0`, possibly SGD as a diagnostic | The paper states learning rates but not the optimizer. Optimizer dynamics may explain reproduction differences | Paired MSE/forgetting with equal update budgets |
| Weight decay | Explicit `0.0` | `0`, `1e-6`, `1e-5`, `1e-4` | Small regularization may improve generalization; it can also suppress newly added weights | Base and incoming-class MSE, plastic-weight norms |
| Activation function | ReLU hidden, identity latent, sigmoid output | ReLU, sigmoid/tanh hidden if historically plausible; identity or sigmoid latent | Strongly changes representations and threshold scale. The paper defines a generic activation but not the concrete function | Base MSE, saturation statistics, threshold stability |
| Plasticity phase length | `500` default; validation used `25` and `100` | `50`, `100`, `250`, `500`, `1000` | Longer phases fit new nodes more fully but increase cost and can overfit outliers | Incoming-class gain and final outlier fraction |
| Stability phase length | `500` default; validation used `25` and `100` | `50`, `100`, `250`, `500`, `1000` | More LR/100 updates may compensate for the very small paper-specified stability rate | Retention, acquisition, outliers, actual updates |
| Next-level phase length | `500` | `50`, `100`, `250`, `500`, `1000` | Determines whether new lower-level columns become useful to the next representation | Global error change after next-level training |
| Reconstruction thresholds | Fixed defaults exist; paper-oriented runs estimate from base data | Percentiles `0.95`, `0.975`, `0.985`, `0.995`; optional additive margin | Lower thresholds trigger more growth; overly strict thresholds cause cap hits | Outlier convergence, growth, validation MSE |
| `MaxOutliers` | Default allows 10% | Fractions `0.05`, `0.10`, `0.20`, `0.30`, or documented absolute counts | Strict quotas increase growth and training; loose quotas may underfit new classes | Final outlier fraction, MSE, parameter count |
| `MaxNodes` per level | `[100,100,100,20]`; diagnostic used `[25,35,8,20]` | Small pilot caps followed by higher caps only where needed | Prevents runaway growth but can prematurely terminate learning | Cap-hit rate and marginal MSE per added node |
| Number of nodes added per round (`Nodes_New`) | Proportional to outlier count | Absolute `1`, `2`, `5`; proportional factors `0.002`, `0.005`, `0.01` | Large additions reduce rounds but can overshoot capacity; small additions are slower and more targeted | MSE/growth Pareto frontier |
| New-node initialization | Kaiming uniform, zero bias | Kaiming uniform, Xavier, paper-era sigmoid-compatible initialization | Affects early plasticity and whether new nodes receive useful gradients | First-round loss reduction and dead/saturated nodes |
| Next-layer update interpretation | `paper_columns` for paper presets | `paper_columns` and `broad` as competing interpretations | The paper says to add connections and train the next SHL-AE but does not fully define the trainable subset | Global MSE after propagation, retention |
| Next-layer learning rate | Base LR multiplied by `0.01` | Ratios `0.01`, `0.03`, `0.1`, `1.0` | The exact rate is not stated for this step; too little training can isolate grown lower-level nodes | Global outliers at the next and deeper levels |
| Replay samples per previous class | One current batch per old class in paper mode | Per-class ratios `0.25`, `0.5`, `1`, `2` | More replay improves retention but can suppress acquisition and grows linearly with classes | Acquisition/forgetting balance and runtime |
| Replay scheduling within stability | Mixed batches | Mixed, alternating batches, alternating epochs, current-then-replay | The paper defines the combined stable dataset but not minibatch ordering | Paired acquisition, forgetting, gradient conflict |
| Covariance numerical epsilon | `1e-4` | `1e-6`, `1e-5`, `1e-4`, `1e-3` | Stabilizes Cholesky factorization; large values add unintended latent noise | Cholesky failures and IR distribution error |
| Number of samples used for latent statistics | Up to `4096` per class | `512`, `1024`, `4096`, all available | More samples improve covariance estimation but cost memory/time | Mean/covariance stability and IR quality |
| Random seeds and class-order repetitions | One for most presets | 5 screening seeds, 10 MNIST confirmation seeds, 20 SD-19 orders | Does not change the algorithm, but is essential for reliable estimates | Confidence intervals and directional consistency |

### Priority within the paper-compatible group

Based on the first validation runs, test these first:

1. Stability **duration** at the paper-specified LR/100: `100`, `250`, `500`,
   `1000` epochs.
2. Next-level learning-rate ratio: `0.01`, `0.03`, `0.1`, `1.0`.
3. Threshold percentile and level-2 `MaxNodes`, because level 2 remained at 28%
   outliers and hit its cap under the best diagnostic.
4. Replay samples per old class: `0.5`, `1`, `2`.
5. Node-addition rule: absolute one-node growth versus small proportional growth.

These preserve LR/100 stability while testing whether the paper's unspecified
phase length and growth choices are sufficient.

## 2. Paper-specified variables whose modification is a deviation

| Paper-specified variable | Paper requirement | Current paper-oriented setting | Performance-oriented alternatives | Required label if changed |
| --- | --- | --- | --- | --- |
| Stability learning rate | Update all current-level weights at `LR/100` | `stability_lr_ratio=0.01` | `0.03`, `0.1`, `0.3`, `1.0` | Non-paper stability-LR ablation |
| Plasticity encoder update scope | Update encoder weights connected to new nodes only; old encoder features remain fixed | Mature encoder frozen, plastic encoder at LR | Train mature encoder at small or full LR | Relaxed-freeze extension |
| Plasticity decoder learning rate | Decoder updates at `LR/100` | `plasticity_decoder_lr_ratio=0.01` | `0.03`, `0.1`, `1.0` | Non-paper decoder-LR ablation |
| Stability update scope | Train all weights in the current SHL-AE | Mature and plastic encoder/decoder groups train | Decoder-only, plastic-only, or frozen mature encoder | Restricted-stability extension |
| Level-local objective | Train the copied level as a single-hidden-layer AE | `paper_level_ae` | Full-autoencoder reconstruction or global partial objective | Non-paper objective ablation |
| Outlier criterion | Global reconstruction error at level L | Pixel-space `forward_partial` error | Local representation error or dual quality gate | Non-paper growth criterion |
| Growth signal | Reconstruction error; labels are not required for node creation | Label-free outlier selection | Label-conditioned thresholds/growth | Supervised-growth extension |
| MNIST base classes | Initially train digits 1 and 7 | `[1,7]` | Other digit pairs or all digits | Alternate curriculum |
| MNIST incremental order | `0,2,3,4,5,6,8,9` | Exact published order | Randomized or difficulty-ordered digits | Order-robustness ablation |
| MNIST initial architecture | `784-200-100-75-20-75-100-200-784` | Encoder `[200,100,75,20]` | Wider, deeper, convolutional, or pretrained architecture | Alternate architecture |
| SD-19 initial architecture | `784-1000-500-250-50-250-500-1000-784` | Encoder `[1000,500,250,50]` | Smaller/larger or convolutional architecture | Alternate architecture |
| SD-19 curriculum | Digits first, then uppercase and lowercase letters | Digits base; letters incremental | Different base set or mixed classes | Alternate curriculum |
| Base training form | Stacked denoising autoencoder | `stacked_denoising` | End-to-end-only AE | Non-paper pretraining |
| Post-stack global fine-tuning | Not described as part of the reported base procedure | `pretrain_finetune_epochs=0` | 5-50 end-to-end fine-tuning epochs | Global-finetune extension |
| Intrinsic replay distribution | Per-class top-latent mean and full covariance Cholesky; Gaussian sampling then decode | `gaussian_full`, shrinkage `0`, noise scale `1` | Diagonal/shrunk covariance, filters, mean-only, adjusted noise | Modified-IR ablation |
| Old-data availability in IR | Original samples from old classes are unavailable | Intrinsic refresh fits incoming class only | Refit old statistics from original data | Oracle/leaky IR control |
| Stability replay source | Current class plus replayed previous classes | Paper per-class replay | Replay only, current only, or selected old classes | Modified stability set |
| Dataset replay interpretation | Not the MNIST IR condition; useful as an oracle control | Explicit dataset-replay upper bound | Present dataset replay as IR | Invalid labeling—do not do this |
| Capacity matching of CL controls | CL/CL+IR fixed networks match the corresponding final NDL capacity | Figure 4 runner injects measured NDL sizes | Smaller/larger unmatched controls | Unmatched-capacity control |
| SD-19 order experiment | 20 randomly ordered letter curricula for the growth average | `sd19_growth_20.yaml` | Fewer/fixed orders | Reduced-order diagnostic |

### Evidence from the first stability-LR deviation

The clean dataset-replay digit-0 experiment provides a concrete example of why
the two groups must remain separate:

| Stability ratio | Fidelity | Macro MSE | Digit-0 MSE | Cap hits | Final outlier fractions |
| ---: | --- | ---: | ---: | --- | --- |
| `0.01` | Paper-specified LR/100 | 0.02902 | 0.05044 | 4/4 levels | 0.336, 0.105, 0.664, 0.562 |
| `1.0` | Explicit non-paper diagnostic | 0.01804 | 0.02900 | 1/4 levels | 0.078, 0.014, 0.281, 0.094 |

Full-LR stability clearly improves this implementation, but it cannot be used
as evidence that the literal paper algorithm was reproduced. The next
paper-compatible test is to retain LR/100 and increase the undocumented number
of stability epochs. The separate deviation study should test intermediate
stability ratios and report them as such.

## Reporting rule

Every experiment should include a fidelity field with one of:

- `paper_compatible_undocumented_tuning`
- `paper_locked_confirmatory`
- `explicit_nonpaper_ablation`
- `dataset_replay_oracle_control`

If any variable from Table 2 changes, the run cannot be labeled
`paper_locked_confirmatory`.

