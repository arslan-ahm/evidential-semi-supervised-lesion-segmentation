# Method

## The problem with a confidence threshold

Semi-supervised segmentation works by having a teacher network label the
unlabelled pool for a student. The whole method reduces to one question: *which
of the teacher's pixel predictions should the student be asked to reproduce?*

The dominant answer, inherited from FixMatch, is a confidence threshold. Keep
the pixel if `max_k p_k > 0.95`, discard it otherwise. It is simple and it
works, but it conflates two situations that a single probability cannot
distinguish.

Consider two pixels, both of which the network scores at `P(lesion) = 0.5`:

1. A pixel in a region of the image unlike anything in the labelled set. The
   network has no basis for a prediction, and 0.5 expresses ignorance.
2. A pixel on the boundary of a low-contrast lesion, where the transition
   genuinely spans several pixels. The network has ample evidence, and that
   evidence genuinely supports both classes. 0.5 is the *correct* answer.

A softmax reports these identically. A threshold therefore does the same thing
to both - discards them - and that is the wrong action in both cases. The first
pixel should be ignored because the prediction is worthless. The second carries
real information about where the boundary lies, and discarding it throws away
supervision precisely where segmentation error concentrates.

That is not a marginal concern for lesion segmentation. Dice is dominated by the
lesion interior, which is large and easy; the errors that matter live in a band
a few pixels wide around the contour, and low-contrast lesions have soft
contours by definition. A threshold-based method is structurally blind to its
hardest examples.

## Evidence instead of confidence

Rather than a normalised probability, the network emits non-negative *evidence*
per class, `e = softplus(logits)`, which parameterises a Dirichlet distribution
over the class probabilities with concentration `alpha = e + 1`. With
`S = sum_k alpha_k` for the Dirichlet strength, subjective logic (Josang, 2016)
decomposes the pixel's opinion into three interpretable parts:

| quantity | definition | reading |
|---|---|---|
| belief | `b_k = e_k / S` | evidence accumulated for class `k` |
| **vacuity** | `u = K / S` | *epistemic*: "I have not seen enough to say" |
| **dissonance** | `2 * min(b_0, b_1)` for `K = 2` | *aleatoric*: "the evidence conflicts" |

with the identity `sum_k b_k + u = 1`.

The two pixels above are now separable, and the repository asserts this
numerically rather than claiming it (`tests/test_uncertainty.py::
test_conflicting_evidence_is_dissonance_not_vacuity`):

| case | logits | `P(lesion)` | vacuity | dissonance |
|---|---|---|---|---|
| no evidence | `[-60, -60]` | 0.500 | **1.000** | 0.000 |
| conflicting evidence | `[+30, +30]` | 0.500 | 0.032 | **0.968** |
| confident lesion | `[-30, +30]` | 0.969 | 0.062 | 0.000 |

Identical probability, opposite epistemic state.

### Dissonance in the binary case

Josang's dissonance is

```
diss = sum_k [ b_k * sum_{j != k} b_j * Bal(b_j, b_k) ] / sum_{j != k} b_j
Bal(b_j, b_k) = 1 - |b_j - b_k| / (b_j + b_k)
```

For `K = 2` each inner sum has one term, so the expression telescopes:

```
diss = (b_0 + b_1) * Bal(b_0, b_1)
     = (b_0 + b_1) - |b_0 - b_1|
     = 2 * min(b_0, b_1)
```

which is what `evissl.uncertainty.dissonance` evaluates. The general-`K` branch
is implemented alongside it and the two are asserted equal on the binary case,
so the fast path cannot drift from the definition it claims to implement.

## Vacuity gates, dissonance tempers

The decomposition is only useful if the two components drive different actions.
Two are proposed here. **One is supported by the measurements in
`docs/RESULTS.md` and one is not** — that verdict is stated in each subsection
rather than left to the reader to discover.

**Vacuity gates the weight.** Each unlabelled pixel contributes to the
consistency loss with weight

```
w = (1 - u) * valid
```

`1 - u` is exactly the total belief mass: how much evidence the teacher has
actually accumulated at that pixel. The weight is continuous, bounded in
`[0, 1]`, and requires no threshold to tune. A pixel the teacher knows nothing
about contributes nothing, automatically, and a pixel it knows a little about
contributes a little. `vacuity_cutoff` adds a hard floor, but at its default of
0.80 it only removes pixels whose weight was already below 0.2 - it is a safety
rail, not the mechanism.

**Dissonance tempers the target.** The teacher's probabilities are sharpened
before being used as a target, but with a per-pixel temperature

```
T_eff = T + (1 - T) * diss
```

At `diss = 0` this is the configured `T` (full sharpening: the teacher is
decided, so commit). At `diss = 1` it is exactly 1.0 (no sharpening: leave the
distribution alone). Sharpening a genuinely ambiguous boundary pixel
manufactures a confident label for something the image does not determine, and
the student then learns that fabrication. Tempering by dissonance is meant to
stop that, and it introduces no new hyper-parameter.

FixMatch cannot express either behaviour. Its weight is a step function of a
quantity that does not distinguish the two uncertainty types, and its target is
always a hard one-hot label.

> **The second mechanism does not survive measurement, and this document is not
> going to pretend otherwise.** In the trained model, mean dissonance is 0.018
> and vacuity and dissonance turn out to be nearly collinear, so `T_eff` averages
> 0.5095 against a configured 0.50 — the tempering is effectively inert. The
> component ablation confirms it: disabling tempering entirely does not hurt.
>
> The cause is inside the likelihood, not the regulariser. Minimising
> `log S − log α_target` is achieved by driving the non-target evidence to zero,
> and binary dissonance *is* `2·min(b₀, b₁)`. So the objective that makes vacuity
> informative destroys the quantity dissonance measures. Setting `kl_weight = 0`
> does **not** restore it (0.0259 → 0.0257), which rules out the KL term as the
> culprit; dropping the evidential objective altogether raises dissonance
> six-fold. See `docs/RESULTS.md` for the numbers.
>
> The claim this repository supports is therefore the **vacuity gate**, and only
> that. Dissonance tempering is implemented, unit-tested, motivated — and
> currently unsupported by evidence.

### Why the loss is normalised by weight mass, not pixel count

The consistency term divides by `sum(w)` rather than by the pixel count. This is
a small detail with a large consequence: as training proceeds the teacher
accumulates evidence, so `sum(w)` grows. Dividing by pixel count would let the
term ramp up twice - once from the explicit schedule, once from rising
confidence - which is a documented cause of late-training collapse in
threshold-based methods. Normalising by the mass keeps the effective learning
rate of the branch constant.

## The supervised objective

The evidential head is trained with type-II maximum likelihood under the
Dirichlet,

```
L_nll = sum_k y_k * (log S - log alpha_k)
```

plus a KL term that pulls *misleading* evidence towards the uniform Dirichlet.
Evidence for the correct class is removed first,
`alpha_tilde = y + (1 - y) * alpha`, so only evidence supporting wrong classes
is penalised. The stated motivation is that without it the network could lower
the NLL by inflating evidence everywhere, collapsing vacuity to zero and
degenerating the weighting rule into Mean Teacher.

The `no_kl` row of the component ablation was added to test that motivation, and
it does not hold up at these settings: with `kl_weight = 0`, mean vacuity is
0.2820 against 0.2822 with the term active. The NLL alone already suppresses
non-target evidence, so the KL is largely redundant here — and the ablation
suggests it may be mildly harmful. It is kept in the default config because it
is the formulation the literature specifies and because a longer schedule may
behave differently, but this repository does not claim it helps.

The KL weight is annealed linearly over `kl_anneal_epochs`, because at
initialisation there is no evidence to regularise and a strong KL term simply
locks the model into total vacuity.

A soft Dice term on the Dirichlet mean is added, and it is not optional. Pure
evidential losses are pixel-wise and inherit cross-entropy's indifference to
class imbalance; for lesions covering 4-34% of the frame that collapses the
prediction towards background.

## Architecture

The training ideology is independent of the backbone, but the repository targets
the low-compute regime, so the default network is deliberately small: a
depthwise-separable U-Net with a gated axial bottleneck, **0.96M parameters**
against the 31.0M of the reference U-Net.

Three choices are worth stating because each answers a specific failure mode:

- **GroupNorm, never BatchNorm.** An EMA teacher averages *parameters*;
  BatchNorm running statistics are buffers, so they either leak the student's
  statistics into the teacher or go stale. GroupNorm has no running state and
  the entire bug class disappears. It also behaves better at the 8-16 image
  batches these runs use. `tests/test_models.py::test_no_batchnorm_in_the_proposed_model`
  enforces it.
- **Zero-initialised gates.** The axial-attention and squeeze-excite branches
  are scaled by a parameter starting at zero, so at step 0 the network is
  exactly the plain convolutional net and the extra branches must earn their
  contribution. This removes the warm-up instability that otherwise appears
  when attention is added to a small network.
- **Zeroed output head.** The head starts at zero, so initial evidence is zero
  and initial vacuity is exactly 1. That is the only honest starting point for
  an evidential model: any other initialisation asserts evidence it has not
  seen.

## Experimental design

Three decisions do more for the credibility of the comparison than any
architectural detail.

**One training loop for all four methods.** `evissl/engine/trainer.py` handles
supervised-only, Mean Teacher, FixMatch and the evidential method. If each had
its own script, a difference in results could come from an incidental
difference in the schedule, the augmentation, the EMA handling or the checkpoint
selection. Here `semi.method` is the only thing that varies.

**A fixed number of gradient steps per epoch.** An epoch is
`optim.steps_per_epoch` optimiser steps, not one pass over the labelled set. If
it were one pass, a 5%-labelled run would take 2 steps per epoch and a
50%-labelled run 25 - the labelled-fraction sweep would then vary training
length as well as label count, and the resulting curve would measure a mixture
of the two.

**Nested labelled subsets.** The labelled subset is the head of a fixed shuffle,
so raising `labeled_fraction` strictly *adds* images to the previous set. Each
rung of the sweep is a superset of the last, making it a controlled comparison
rather than a set of unrelated random draws.

**Aligned consistency views.** The geometric transform is applied once and
shared by the weak and strong views; only photometric perturbation and cut-out
differ. A pixel-wise consistency loss between misaligned views is noise, and the
failure is silent - the loss still decreases. `tests/test_data.py::
test_weak_and_strong_views_are_spatially_aligned` measures profile correlation
(0.98 aligned versus 0.58 under a 5-pixel shift) so a regression cannot pass
unnoticed. Cut-out regions are excluded from the consistency term through the
`valid` mask, since the student cannot infer a label for pixels it was never
shown.

## Evaluation

Overlap metrics alone would not show whether any of the above works, so three
families are reported:

- **Overlap** - Dice, IoU, sensitivity, specificity, precision, accuracy.
- **Boundary** - HD95, ASSD and the BF score at a 2-pixel tolerance, computed
  from symmetric surface distances. Where the interesting errors are.
- **Calibration and uncertainty quality** - ECE and its equal-mass variant ACE,
  Brier, NLL, plus **AUSE** and the uncertainty-error AUROC. AUSE measures
  whether the uncertainty *ranks* the errors: discard the most uncertain pixels
  progressively and compare the resulting error curve to an oracle that
  discards the actually-wrong ones. It is invariant to the uncertainty's scale,
  which is what makes vacuity, dissonance, softmax entropy and MC-dropout
  spread directly comparable.

Metrics are computed per image and aggregated afterwards, never pooled over all
pixels first - pooling lets large lesions dominate, so a model could improve its
headline number by getting better at easy big cases while regressing on the
small ones that matter clinically.

Where a metric is genuinely undefined it is reported as `NaN` and excluded, with
the contributing count kept alongside every mean. Surface distances to an empty
prediction have no value; substituting the image diagonal, as some
implementations do, rewards a model for failing completely instead of recording
that it failed.

Differences are established with paired statistics: Wilcoxon signed-rank on
per-image values, a paired bootstrap interval on the mean difference, and
Holm-Bonferroni correction across the whole family of tests. The methods are
evaluated on the same images, so the comparison must be paired; and an ablation
runs many tests, so at `alpha = 0.05` roughly one in twenty would look
significant by chance.

### The unit of analysis is the training run, not the image

This is the trap this project fell into and then caught, and it is worth stating
as a design principle rather than a footnote.

A paired per-image test conditions on **one trained model per configuration**. It
answers "do these two specific sets of weights differ consistently across the
test set?" — correctly, and with a lot of statistical power, because there are
150 images. But the question a paper asks is "is this *method* better?", and
there the sampling unit is the **training run**. With one run per method, `n = 1`
for that question, and no quantity of test images repairs it.

Measured here: re-training the identical configuration with a different seed
moves Dice by a standard deviation of 0.0132 and boundary F1 by **0.0566**. Any
claimed improvement smaller than roughly `√2 ×` those numbers is unsupported no
matter how small its per-image p-value is. `results/tables/seed_variance.csv`
records the measurement, and `docs/RESULTS.md` applies it to every claim.

The practical rule: report per-image tests for what they are — a statement about
two trained models — and treat multi-seed replication as the requirement for any
claim about a method.

## References

- Sensoy, Kaplan & Kandemir. *Evidential Deep Learning to Quantify
  Classification Uncertainty.* NeurIPS 2018.
- Josang. *Subjective Logic: A Formalism for Reasoning Under Uncertainty.* 2016.
- Tarvainen & Valpola. *Mean teachers are better role models.* NeurIPS 2017.
- Sohn et al. *FixMatch: Simplifying Semi-Supervised Learning with Consistency
  and Confidence.* NeurIPS 2020.
- Laine & Aila. *Temporal Ensembling for Semi-Supervised Learning.* ICLR 2017.
- Kendall & Gal. *What Uncertainties Do We Need in Bayesian Deep Learning for
  Computer Vision?* NeurIPS 2017.
- Ho et al. *Axial Attention in Multidimensional Transformers.* 2019.
- Ronneberger, Fischer & Brox. *U-Net: Convolutional Networks for Biomedical
  Image Segmentation.* MICCAI 2015.
