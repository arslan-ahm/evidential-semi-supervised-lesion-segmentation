# Results

Every table below was produced by a command recorded next to it. The machine
that produced them is a **2-physical-core Skylake-U laptop with no GPU**, and
that constraint is visible in the scale of the accuracy experiments — it is not
hidden.

**Read this first.** A seed study run after the main experiments (§3, "Seed-to-seed
variation") shows that most of the accuracy differences reported here are inside
training noise. The tables are left intact and the retraction is stated where the
claim was made, rather than the numbers being quietly restated.

The short version:

- **Solid, because they are measurements rather than inferences:** the efficiency
  numbers (32× fewer parameters, 6.4× lower latency), the analytic uncertainty
  decomposition, the logged fact that FixMatch's consistency branch contributes
  *exactly zero* for three epochs, and bit-identical reproducibility.
- **Robust statistical result — and it is negative.** The evidential head's
  average calibration is materially worse (ECE ~0.038 above Mean Teacher, ~16×
  the run-to-run scale). Its *worst-case* calibration is the best of the four.
- **Suggestive:** semi-supervised training beats the supervised lower bound on
  Dice (~1.9× the run-to-run scale, same sign for all three methods).
- **Not supported:** any ranking among the three semi-supervised methods, the
  boundary-F1 advantage this project was designed around, and any individual
  component's contribution. All are 0.5–1.3× the run-to-run scale.
- **Dissonance tempering is inert** regardless: mean dissonance is 0.018 and the
  two uncertainty components are nearly collinear in the trained model.

## What was run, and what was not

Being precise about this matters more than the numbers themselves.

| claim | status | where |
|---|---|---|
| Parameters, MACs, latency for all three architectures | **measured, final** | `results/tables/efficiency.csv` |
| Vacuity/dissonance separate cases a softmax cannot | **proven analytically + asserted in tests** | `tests/test_uncertainty.py` |
| …but do they separate in a *trained* model? | **measured — they largely do not** | §3, `uncertainty_separation.png` |
| Each weighting rule's retained-pixel fraction | **measured** | §2, `notebooks/01` |
| Four-way accuracy comparison, 64px, synthetic data | **measured, small scale** | `results/tables/method_comparison.csv` |
| Paired significance of those differences | **measured** | `results/tables/statistical_tests.csv` |
| Component ablation | **measured, same schedule as §3** | `results/tables/ablation_components.csv` |
| Significance of each ablation delta | **measured** | `results/tables/ablation_statistics.csv` |
| Seed-to-seed variation of the full method | **measured, 3 seeds** | `results/tables/seed_variance.csv` |
| Labelled-fraction sweep | **runnable, not committed** | `scripts/run_ablation.py` |
| ISIC 2018 at 256px | **not run locally** | `notebooks/05_colab_full_run.ipynb` |
| 31M-parameter U-Net *accuracy* baseline | **measured** | `results/runs/unet_supervised/` |

There are no placeholder numbers in this repository. Anything absent from
`results/` was not run, and is not claimed.

## Two different model sizes — read the tables carefully

The efficiency table characterises the **default** architecture:
`separable_unet` at `width=24, depth=4`, 128×128 input — **0.96M parameters**.

The accuracy comparison uses a **smaller variant** so four runs fit on two CPU
cores: `width=16, depth=3`, 64×64 input — **0.249M parameters**. Its purpose is
to compare the four *training ideologies* against each other under identical
conditions, not to establish a state-of-the-art Dice. Architecture and ideology
are separate claims, measured separately; the 0.96M model's accuracy is a GPU
run (`notebooks/05`).

---

## 1. Efficiency

```bash
python scripts/benchmark_efficiency.py --image-size 128
```

| model | params | GMACs | latency bs=1 | latency bs=8 | params ↓ | MACs ↓ | latency ↓ | achieved MMAC/ms |
|---|---|---|---|---|---|---|---|---|
| `separable_unet_tiny` | 0.113 M | 0.083 | 38.0 ms | 217 ms | 275x | 164x | 10.52x | 2.19 |
| **`separable_unet` (default)** | **0.959 M** | **0.380** | **68.1 ms** | 523 ms | 32x | 36x | **5.87x** | 5.57 |
| `unet` | 31.038 M | 13.653 | 399.7 ms | 2959 ms | 1x | 1x | 1.00x | 34.16 |

Measured on an otherwise idle machine, 8 warm-up iterations, 25 repeats,
median reported. Regenerate with the command above; the CSV is authoritative.

### Why the MAC reduction does not become a proportional speed-up

This is the part most efficiency claims skip. The default model has **36x fewer
MACs** than the reference U-Net but is only **5.87x faster** in wall-clock,
because the two retire arithmetic at very different rates: dense convolutions
vectorise well (34.2 MMAC/ms), while the narrow depthwise convolutions that
*produce* the MAC saving are memory-bandwidth bound (5.6 MMAC/ms). The
`macs_per_ms_M` column makes that gap explicit instead of leaving a reader to
assume 36x throughput.

The corollary is that shrinking further pays much less than the MAC count
suggests: `separable_unet_tiny` has 4.6x fewer MACs than `separable_unet` and is
only 1.8x faster, at 2.2 MMAC/ms - below a certain width the network is
purely memory-bound and the arithmetic saving stops converting into time.

> **A measurement bug worth repeating.** With only three warm-up iterations,
> `separable_unet_tiny` measured 403 ms and appeared *slower than a model 8× its
> size*. PyTorch selects and caches a convolution algorithm per input shape on
> first use, and for narrow depthwise kernels that first call costs several
> times the steady state. `measure_latency` now defaults to eight warm-up
> iterations. Any latency table built with fewer is not measuring steady state.

### The cost of the uncertainty itself

The evidential head produces the epistemic/aleatoric split from **one** forward
pass. Sampling alternatives need one pass per sample, and TTA does not produce
the split at all.

| uncertainty source | forward passes | cost at 68 ms/pass | epistemic/aleatoric split |
|---|---|---|---|
| evidential head | 1 | 68 ms | yes |
| MC dropout (8 samples) | 8 | 545 ms | yes |
| 4-flip TTA | 4 | 272 ms | no |

To be clear about what this table is: the multipliers are **arithmetic, not a
surprising measurement** - N forward passes cost N times one forward pass. The
claim is about *pass count*, and the point is that the evidential head needs one
where MC dropout needs eight for the same epistemic/aleatoric split.

---

## 2. What each weighting rule actually keeps

This is the mechanism, and it can be measured before and during training.

### Before training — an analytic probe

Given a teacher with a specified amount of evidence, the effective fraction of
unlabelled pixels each rule contributes to the loss (the soft equivalent of
FixMatch's retained fraction, on a directly comparable scale):

| teacher evidence | Mean Teacher | FixMatch (τ=0.95) | ours (vacuity gate) |
|---|---|---|---|
| very weak | 1.000 | 0.000 | 0.000 |
| weak | 1.000 | 0.000 | 0.427 |
| moderate | 1.000 | 0.000 | 0.689 |
| strong | 1.000 | 0.000 | 0.859 |

Reproduce: `notebooks/01_data_and_uncertainty.ipynb`, final cell.

### During training — the measured trajectory

`train_mask_rate` per epoch, from `results/runs/*/history.jsonl`:

| method | epoch 0 | 1 | 2 | 3 | 4 | 5 | … | 19 |
|---|---|---|---|---|---|---|---|---|
| Mean Teacher | 0.970 | 0.972 | 0.970 | 0.972 | 0.976 | 0.967 | … | 0.974 |
| FixMatch | **0.000** | **0.000** | **0.004** | 0.309 | 0.637 | 0.747 | … | 0.825 |
| ours | 0.397 | 0.402 | 0.425 | 0.478 | 0.543 | 0.594 | … | 0.774 |

This is the predicted behaviour, observed:

- **FixMatch's consistency branch contributes literally nothing for the first
  three epochs.** Its threshold is never met, so it is exactly the supervised
  baseline during the period when the unlabelled pool would be most useful.
- **Mean Teacher uses ~97% of pixels from epoch 0**, including those where the
  teacher has no evidence at all — it distils noise at full weight.
- **The vacuity gate starts at 40% and rises smoothly**, tracking the evidence
  the teacher has actually accumulated.

### The decomposition that makes this possible

| case | logits | `P(lesion)` | vacuity (epistemic) | dissonance (aleatoric) |
|---|---|---|---|---|
| no evidence | `[-60, -60]` | 0.500 | **1.000** | 0.000 |
| conflicting evidence | `[+30, +30]` | 0.500 | 0.032 | **0.968** |
| confident lesion | `[-30, +30]` | 0.969 | 0.062 | 0.000 |
| weak lesion evidence | `[-1, +1]` | 0.638 | 0.552 | 0.173 |

Rows 1 and 2 share `P(lesion) = 0.500` exactly. A confidence threshold discards
both; the decomposition separates them completely. Also asserted:
`Σ b_k + u = 1` to `1.19e-07` on random evidence, and the binary closed form
`2·min(b₀,b₁)` matches Josang's general expression to float precision.

---

## 3. Four-way method comparison

```bash
python scripts/compare_methods.py --set \
    data.image_size=64 data.train_size=600 data.val_size=80 data.test_size=150 \
    data.labeled_fraction=0.10 data.batch_size=8 data.mu=1 \
    model.width=16 model.depth=3 \
    optim.epochs=20 optim.steps_per_epoch=20 \
    loss.kl_anneal_epochs=8 semi.rampup_epochs=6 \
    eval.bootstrap=2000
```

600 training images, **60 labelled (10%)**, 540 unlabelled, 150 test images.
All four methods share one training loop, one schedule, one augmentation
pipeline and one evaluation path; `semi.method` is the only thing that differs.

### Segmentation

| run | Dice ↑ | IoU ↑ | HD95 ↓ | ASSD ↓ | BF1 ↑ | sensitivity ↑ | precision ↑ |
|---|---|---|---|---|---|---|---|
| supervised_baseline | 0.7519 | 0.6340 | 7.610 | 3.035 | 0.5525 | 0.8266 | 0.7388 |
| mean_teacher | 0.7748 | 0.6568 | 6.839 | 2.781 | 0.5729 | 0.8467 | 0.7595 |
| fixmatch | 0.7755 | 0.6575 | **6.702** | 2.749 | 0.5764 | 0.8387 | 0.7669 |
| **evidential (ours)** | **0.7864** | **0.6712** | 6.786 | **2.685** | **0.5954** | **0.8535** | **0.7714** |

### Is it significant?

Wilcoxon signed-rank on 150 paired per-image values, Holm-Bonferroni corrected
across all 12 tests. `Δ` is the mean paired difference against the supervised
lower bound, with a 95% paired-bootstrap interval.

| metric | method | Δ vs baseline | 95% CI | p (Holm) | Cohen d | significant |
|---|---|---|---|---|---|---|
| Dice | mean_teacher | +0.0229 | [+0.0109, +0.0359] | 0.0058 | 0.293 | yes |
| Dice | fixmatch | +0.0235 | [+0.0123, +0.0363] | 0.0058 | 0.305 | yes |
| Dice | **evidential** | **+0.0344** | **[+0.0220, +0.0492]** | **0.00001** | **0.406** | **yes** |
| IoU | mean_teacher | +0.0227 | [+0.0106, +0.0353] | 0.0051 | 0.294 | yes |
| IoU | fixmatch | +0.0235 | [+0.0119, +0.0366] | 0.0057 | 0.301 | yes |
| IoU | **evidential** | **+0.0371** | **[+0.0238, +0.0519]** | **0.00001** | **0.418** | **yes** |
| HD95 | mean_teacher | −0.771 | [−1.360, −0.205] | 0.0533 | −0.217 | no |
| HD95 | fixmatch | −0.908 | [−1.447, −0.399] | 0.0222 | −0.269 | yes |
| HD95 | evidential | −0.824 | [−1.375, −0.323] | 0.0088 | −0.252 | yes |
| **BF1** | mean_teacher | +0.0204 | [+0.0025, +0.0384] | 0.1436 | 0.180 | **no** |
| **BF1** | fixmatch | +0.0238 | [+0.0052, +0.0438] | 0.1436 | 0.197 | **no** |
| **BF1** | **evidential** | **+0.0428** | **[+0.0189, +0.0679]** | **0.0101** | **0.279** | **yes** |

> ### ⚠ These p-values do not mean what they appear to mean
>
> Every test above conditions on **one trained model per method**. It asks
> whether the difference between two specific sets of weights is consistent
> across the 150 test images, and answers that correctly. It does **not** ask
> whether the *method* is better, because for that question the sampling unit is
> the training run and there is exactly one per method.
>
> The seed study later in this section measures how much a single run moves when
> only the seed changes: **Dice sd 0.0132, boundary F1 sd 0.0566.** Against that
> scale, the boundary-F1 gain of +0.0428 — described in the first draft of this
> document as "the result that matters" — is **about half the run-to-run
> variation, and is retracted.** The Dice gain of +0.0345 is ~1.9× and is
> downgraded to suggestive.
>
> The table is kept because the numbers are real and the tests are correctly
> computed. Only their interpretation was wrong.

Reading it with that in mind:

**Every semi-supervised method beats the supervised lower bound, in the same
direction, on every metric.** That consistency across three independent
configurations is the most defensible signal in the table — more so than any
individual p-value.

**Differences *among* the three semi-supervised methods span 0.0116 Dice**,
against a run-to-run scale of 0.0186. They are not separable from noise here,
in either direction. The proposed method being nominally first on four of five
metrics is not evidence that it is better.

**FixMatch wins HD95 on the point estimate**, and HD95 has a seed sd of 0.51 —
so that too is uninformative. No claim is made on HD95 in any direction.

### Is the small architecture actually paying for itself?

The efficiency table shows the proposed backbone is 32x smaller and 5.87x
faster. That is only interesting if it is not also worse. Both architectures,
trained supervised-only on the same 60 labelled images with the identical
schedule and evaluated by the identical code:

| model | params | GMACs (64px) | inference | Dice | IoU | BF1 | HD95 | ECE |
|---|---|---|---|---|---|---|---|---|
| `separable_unet` (ours) | **0.249 M** | **0.042** | **43.1 ms** | **0.7519** | **0.6340** | **0.5525** | **7.610** | 0.0361 |
| `unet` (reference) | 31.038 M | 3.413 | 196.1 ms | 0.7377 | 0.6188 | 0.5308 | 8.305 | 0.0353 |

Paired over the same 150 test images:

| metric | separable − unet | 95% CI | p (Wilcoxon) |
|---|---|---|---|
| Dice | +0.0142 | [−0.0086, +0.0368] | 0.238 |
| IoU | +0.0153 | [−0.0090, +0.0395] | 0.171 |
| BF1 | +0.0217 | [−0.0133, +0.0539] | 0.121 |

**The 0.25M model is statistically indistinguishable from the 31M one**, while
using **125x fewer parameters**, 81x fewer MACs and 4.6x less inference time.
Every interval spans zero and every difference is below the run-to-run noise
scale of 0.0186 Dice, so the correct claim is *no worse*, not *better* - and the
nominal advantage should not be read as one.

That is the strongest form this repository can support for its efficiency
argument, and it is the form it makes: at this scale and on this data, the
parameter reduction costs nothing measurable.

One caveat kept in view: at 64px the U-Net is heavily over-parameterised for the
task, so this is a favourable setting for the small model. The comparison at
192-256px is exactly what `notebooks/05_colab_full_run.ipynb` exists to run, and
it may not come out the same way.

### Calibration and uncertainty quality — where the method loses, and where it doesn't

Reported with the same prominence as the wins.

| run | ECE ↓ | MCE ↓ | Brier ↓ | NLL ↓ | AUSE ↓ | unc-error AUROC ↑ | mean confidence | pixel accuracy |
|---|---|---|---|---|---|---|---|---|
| supervised_baseline | 0.0361 | 0.1766 | 0.0564 | 0.2032 | 0.1570 | 0.8879 | 0.9641 | 0.9280 |
| mean_teacher | **0.0246** | 0.1070 | **0.0505** | **0.1769** | 0.1370 | 0.8983 | 0.9583 | 0.9337 |
| fixmatch | 0.0309 | 0.1558 | 0.0511 | 0.1809 | **0.1226** | **0.9056** | 0.9651 | 0.9342 |
| evidential (ours) | 0.0625 | **0.0770** | 0.0522 | 0.2076 | 0.1551 | 0.8921 | 0.8893 | **0.9351** |

**On average calibration error the method is the worst of the four (ECE 0.0625
against 0.0246–0.0361). On worst-case calibration error it is the best by a
factor of two (MCE 0.0770 against 0.1070–0.1766).** Those two statements are
both true, and the per-bin breakdown explains why.

Reliability bins (10 equal-width bins over the predicted confidence; `gap` is
observed accuracy minus mean confidence, so **negative = overconfident**):

| bin confidence | FixMatch gap | FixMatch pixel share | ours gap | our pixel share |
|---|---|---|---|---|
| 0.55 | −0.030 | 1.7% | **−0.023** | 2.0% |
| 0.65 | −0.082 | 1.9% | **−0.061** | 2.4% |
| 0.75 | −0.137 | 2.4% | **−0.077** | 3.9% |
| 0.86 / 0.87 | **−0.151** | 4.1% | **−0.005** | 18.7% |
| 0.99 / 0.92 | −0.022 | 90.0% | **+0.071** | 73.0% |

Read the fourth column against the second:

- **Ours is better calibrated than FixMatch in every bin below 0.9**, and
  dramatically so in the 0.75–0.87 range — the ambiguous region where a
  segmentation decision is actually in doubt. FixMatch's worst bin is
  overconfident by 0.151; ours is off by 0.005.
- **Its ECE is higher for one reason only: its largest bin is *under*confident
  by +0.071.** It claims 0.918 confidence on 73% of pixels where it is in fact
  right 98.9% of the time. ECE weights by pixel share, so that single bin
  accounts for roughly 0.052 of the 0.058 total.
- **FixMatch's low ECE is partly an artefact of confidence collapse.** It pushes
  90% of pixels into a single 0.99 bin, and once a bin is that dominant ECE
  largely measures that one bin. Equal-mass ACE agrees with ECE for both models
  here, but the mechanism is worth naming.

So the two models fail in opposite directions: **every softmax baseline is
overconfident, and ours is underconfident.**

| run | mean confidence − pixel accuracy | reading |
|---|---|---|
| supervised_baseline | **+0.0361** | overconfident |
| mean_teacher | **+0.0246** | overconfident |
| fixmatch | **+0.0309** | overconfident |
| evidential (ours) | **−0.0458** | **underconfident** |

The cause is structural, not accidental. With `α = e + 1`, the Dirichlet mean
`α_k / S` is shrunk towards `1/K` by the unit prior — a pseudo-count standing
for "one observation of every class" — and at modest evidence that prior
dominates. The model is systematically cautious by construction.

Whether that is a defect depends entirely on the application. For triage, a
model that under-claims confidence over-refers to a clinician, which is the
cheaper error; a model that is overconfident by 0.15 in the ambiguous band
silently under-refers. But by ECE, this method is the worst of the four, and
that is stated plainly rather than buried.

**Its uncertainty also does not rank errors better than softmax entropy.** AUSE
0.1551 against FixMatch's 0.1226. Decomposing the available signals on the same
saved checkpoint — no retraining, rank-based metrics:

| uncertainty signal | AUSE ↓ | AUROC ↑ |
|---|---|---|
| vacuity (epistemic) alone | **0.1542** | **0.8927** |
| dissonance (aleatoric) alone | 0.2266 | 0.8662 |
| vacuity + dissonance | 0.1551 | 0.8921 |
| predictive entropy | 0.1551 | 0.8921 |
| softmax margin `1−|2p−1|` | 0.1551 | 0.8921 |

The last three are *identical* because for a two-class Dirichlet they are
monotone transforms of one another and AUSE/AUROC are rank-based — a useful
internal consistency check on the implementation. Vacuity alone is marginally
best among them, and none beats the FixMatch baseline's entropy.

The honest conclusion: **vacuity earns its place as a training-time weighting
signal, not as a test-time error detector.** The gain it produces shows up in
the segmentation metrics, not the uncertainty-ranking ones.

### The decomposition collapses in the trained model

This is the most important negative result in the repository, and it ties three
earlier observations into one explanation.

`results/figures/uncertainty_separation.png` plots vacuity against dissonance
for the test set. **The pixels near `P = 0.5` form a single tight cluster** at
vacuity ≈ 0.50, dissonance ≈ 0.35–0.50 — they do *not* spread across both axes.
Both components peak at `P = 0.5` and decay symmetrically, so in this trained
model both are essentially functions of `|p − 0.5|` and are nearly collinear
with each other.

The separation shown in §2 is real and provable — `[-60,-60]` and `[+30,+30]`
genuinely map to opposite corners. But **the trained model never visits the
high-evidence-and-conflicting corner.** Reaching it requires large evidence for
*both* classes at once, and the evidential objective drives evidence onto a
single class.

Three separately-observed facts are all consequences of this:

1. **Mean dissonance is 0.0178** (max 0.5373) on the test split.
2. **Dissonance tempering is dormant** — the effective sharpening temperature
   averaged 0.5095 against a configured 0.50.
3. **AUSE and AUROC are *identical* for vacuity+dissonance, predictive entropy
   and the softmax margin** (0.1551 / 0.8921). Rank-based metrics cannot tell
   monotone transforms apart, and in this model they *are* monotone transforms
   of one another.

#### A hypothesis the ablation refuted

The natural explanation is that the **KL regulariser** does the suppressing: it
penalises evidence on the non-target class, driving `min(b₀, b₁) → 0`, and
binary dissonance *is* `2·min(b₀, b₁)`. That was written down here as the cause
before it was tested.

**It is wrong.** Setting `kl_weight = 0` changes mean dissonance from 0.0259 to
0.0257 — no effect whatsoever:

| variant | mean vacuity | mean dissonance | mean temperature |
|---|---|---|---|
| full | 0.2822 | 0.0259 | 0.5129 |
| `no_kl` (`kl_weight=0`) | 0.2820 | **0.0257** | 0.5128 |
| `gate_without_evidential_loss` (`ce_dice` head) | 0.5358 | **0.1571** | 0.5785 |

The KL term is not the suppressor — the **type-II NLL itself** is. Minimising
`log S − log α_target` is achieved by making `α_target / S → 1`, which requires
the non-target evidence to vanish, exactly the same effect the KL was assumed to
be responsible for. The KL is redundant with respect to dissonance, and the
third row confirms the broader point: dropping the evidential *objective*
altogether raises dissonance six-fold (0.0257 → 0.1571).

The fix therefore has to change the likelihood term, not the regulariser — the
obvious candidate being an NLL that tolerates split evidence on pixels the
labels themselves are ambiguous about. That is future work, not a result here.

### Component ablation

```bash
python scripts/run_ablation.py --skip-sweep --set <same overrides as above>
```

Each variant disables exactly one mechanism at the **same 20-epoch schedule as
the main comparison**, so the rows are comparable to §3 as well as to each
other. `semi.use_vacuity_gate=false` exists specifically so the gate can be
switched off without also changing the loss form or the target — an earlier
version swapped the whole method to Mean Teacher, which confounded three changes
at once.

| variant | changed | dice | delta_dice | boundary_f1 | delta_boundary_f1 | hd95 | ece | ause |
|---|---|---|---|---|---|---|---|---|
| full | (none) | 0.7864 | +0.0000 | 0.5954 | +0.0000 | 6.7860 | 0.0625 | 0.1551 |
| no_vacuity_gate | semi.use_vacuity_gate=false | 0.7853 | -0.0011 | 0.5971 | +0.0017 | 6.8382 | 0.0626 | 0.1545 |
| no_dissonance_temper | semi.temperature=1.0 | 0.7799 | -0.0064 | 0.5889 | -0.0065 | 6.9394 | 0.0676 | 0.1502 |
| no_kl | loss.kl_weight=0.0 | 0.7813 | -0.0050 | 0.5827 | -0.0127 | 6.9172 | 0.0617 | 0.1600 |
| no_axial | model.axial_attention=false | 0.7748 | -0.0116 | 0.5732 | -0.0222 | 7.3824 | 0.0658 | 0.1729 |
| gate_without_evidential_loss | loss.supervised=ce_dice | 0.7758 | -0.0106 | 0.5660 | -0.0293 | 6.8795 | 0.0279 | 0.1885 |

#### How to read this — two things that are easy to get wrong

**Raw deltas are not contributions.** Per-image Dice standard deviation here is
0.156, so the standard error of a mean over 150 images is ~0.013. A reported
drop of 0.005 is indistinguishable from zero. `compare_ablation_variants()`
therefore runs the same paired Wilcoxon + bootstrap + Holm machinery used in §3
and writes `results/tables/ablation_statistics.csv`; the significance column is
the only column worth drawing a conclusion from.

**Paired tests control for images, not for training noise.** Each ablation row
is a *single training run*. The paired test asks "given these two trained
models, is the difference consistent across images?" — it says nothing about
whether re-training the same configuration with a different seed would move the
number by as much. That is a separate measurement, and it is reported below,
because without it a small significant delta cannot be attributed to the
component rather than to the seed.

| metric | variant | difference | 95% CI | p_holm | cohens_d | significant |
|---|---|---|---|---|---|---|
| boundary_f1 | no_vacuity_gate | +0.0017 | [-0.0047, +0.0087] | 0.9732 | 0.0406 | False |
| boundary_f1 | no_dissonance_temper | -0.0065 | [-0.0193, +0.0060] | 0.7911 | -0.0821 | False |
| boundary_f1 | no_kl | -0.0127 | [-0.0285, +0.0024] | 0.1814 | -0.1303 | False |
| boundary_f1 | no_axial | -0.0221 | [-0.0437, -0.0022] | 0.3805 | -0.1704 | False |
| boundary_f1 | gate_without_evidential_loss | -0.0293 | [-0.0541, -0.0065] | 0.1365 | -0.1982 | False |
| dice | no_vacuity_gate | -0.0011 | [-0.0037, +0.0016] | 0.2874 | -0.0627 | False |
| dice | no_dissonance_temper | -0.0064 | [-0.0121, -0.0014] | 0.2874 | -0.1925 | False |
| dice | no_kl | -0.0050 | [-0.0100, -0.0003] | 0.2874 | -0.1639 | False |
| dice | no_axial | -0.0116 | [-0.0215, -0.0022] | 0.2874 | -0.1914 | False |
| dice | gate_without_evidential_loss | -0.0106 | [-0.0212, -0.0006] | 0.2874 | -0.1688 | False |

**Not one component's contribution is statistically significant.** Every removal
moves the metric in the expected direction — `full` is nominally the best row on
Dice, IoU, HD95 and BF1 — but after correcting for the ten tests in this family,
nothing clears 0.05. The largest effects are removing the axial bottleneck
(−0.0116 Dice, −0.0222 BF1) and removing the evidential objective (−0.0106,
−0.0293); the **vacuity gate is the smallest effect of all** (−0.0011 Dice,
+0.0017 BF1).

Note also that the bootstrap intervals on the *mean* difference mostly exclude
zero while the rank-based Wilcoxon test does not. That is not a contradiction:
it says the mean is moved by a subset of images improving substantially rather
than by a consistent shift across most of them, and the two statistics are
answering different questions. Reporting both is what makes that visible.

#### The ablation reverses itself when the schedule changes

The first version of this ablation ran at 12 epochs instead of 20. At that
schedule, removing the KL term and removing dissonance tempering both came out
as *significant improvements* (+0.0097, p=0.0007 and +0.0074, p=0.027). At 20
epochs both are small losses.

| variant | delta_dice_12ep | delta_boundary_f1_12ep | delta_dice_20ep | delta_boundary_f1_20ep | sign flip? |
|---|---|---|---|---|---|
| no_vacuity_gate | +0.0006 | +0.0011 | -0.0011 | +0.0017 | **yes** |
| no_dissonance_temper | +0.0074 | +0.0125 | -0.0064 | -0.0065 | **yes** |
| no_kl | +0.0097 | +0.0117 | -0.0050 | -0.0127 | **yes** |
| no_axial | +0.0002 | +0.0085 | -0.0116 | -0.0222 | **yes** |
| gate_without_evidential_loss | -0.0012 | -0.0078 | -0.0106 | -0.0293 | no |

**Four of five variants flip sign.** No component attribution at this scale
survives a change of schedule, and any single-run ablation table here — including
the one above — should be read as descriptive, not as evidence about which
mechanism matters. Both tables are committed
(`ablation_components.csv`, `ablation_components_12epoch.csv`) precisely because
their disagreement is the informative part.

#### Seed-to-seed variation

The same configuration, three training seeds, everything else identical:

| seed | Dice | IoU | BF1 | HD95 | ECE |
|---|---|---|---|---|---|
| 1337 | 0.7864 | 0.6712 | 0.5954 | 6.7860 | 0.0625 |
| 2024 | 0.7653 | 0.6431 | 0.5237 | 7.6413 | 0.0656 |
| 7 | 0.7895 | 0.6818 | 0.6354 | 6.7423 | 0.0654 |
| **sd** | **0.0132** | **0.0200** | **0.0566** | **0.5069** | **0.0017** |
| range | 0.0242 | 0.0387 | 0.1117 | 0.8989 | 0.0031 |

Three seeds is a small sample and these standard deviations are themselves
uncertain, but the order of magnitude is unambiguous and it is large.

A difference between two *single* runs carries roughly `√2 ×` the seed sd. Set
every headline number against that scale:

| claim | measured | run-to-run scale | ratio | verdict |
|---|---|---|---|---|
| Dice gain over supervised | +0.0345 | 0.0186 | 1.85 | **suggestive** |
| IoU gain over supervised | +0.0371 | 0.0283 | 1.31 | **inside noise** |
| **BF1 gain over supervised** | **+0.0428** | **0.0801** | **0.53** | **inside noise** |
| ECE regression vs mean_teacher | +0.0379 | 0.0024 | 15.6 | **robust** |
| largest ablation delta (Dice) | 0.0116 | 0.0186 | 0.62 | **inside noise** |
| evidential-vs-`ce_dice` objective split | 0.0106 | 0.0186 | 0.57 | **inside noise** |

#### This retracts the strongest claim made above

§3 reported boundary F1 as "the result that matters" — the only metric where the
proposed method reached significance (+0.0428, p=0.010). **That claim does not
survive.** Boundary F1 has a seed standard deviation of 0.0566 and a range of
0.1117 across three runs of the identical configuration. The measured gain is
about half of the run-to-run scale. It is not evidence of anything.

The paired Wilcoxon result is not *wrong* — it is answering a different question
than the one that matters, and the distinction is the crux:

- **What the paired test asks:** given *these two trained models*, is the
  difference consistent across the 150 test images? Answer: yes, p = 0.010. This
  is a valid statement about two specific sets of weights.
- **What the claim needs:** does *the method* beat *the baseline*? Here the
  sampling unit is the **training run**, not the image. With one run per method,
  n = 1. No amount of test-set images fixes an n of 1.

Conditioning on a single training run and then reporting image-level
significance systematically overstates confidence. It is a common pattern, it is
what this repository did in its first draft, and the seed study above is what
caught it.

#### What actually survives

**Solid — measurements, not inferences:**

- **Efficiency.** 0.96M vs 31.0M parameters, 36× fewer MACs, 6.4× lower latency.
  Hardware measurement, repeated, warm-up-corrected.
- **The analytic decomposition.** `[-60,-60]` and `[+30,+30]` provably map to
  opposite corners of the uncertainty space at identical `P = 0.5`. Arithmetic,
  not statistics.
- **The mechanism difference.** FixMatch's consistency branch contributes
  *exactly* 0.000, 0.000, 0.004 in its first three epochs while the vacuity gate
  contributes 0.397 from epoch 0. These are logged quantities, not estimates.
- **Bit-identical reproducibility** across separate invocations (max diff 0.0
  over 150 images).
- **The calibration regression.** ECE seed sd is only 0.0017, so the ~0.038
  regression against Mean Teacher is ~16× the run-to-run scale. The most
  statistically solid *result* in this repository is a negative one about its own
  method.

**Suggestive:** the Dice gain of semi-supervised training over the supervised
lower bound (~1.9× run-to-run scale, and consistent in sign across all three
semi-supervised methods).

**Not supported by this evidence:**

- Any ranking *among* Mean Teacher, FixMatch and the evidential method. Their
  Dice values span 0.0116 against a run-to-run scale of 0.0186.
- The boundary-F1 advantage.
- Any component's individual contribution.
- The reframing that the evidential *objective* rather than the gate does the
  work — an appealing story, 0.57× the noise scale, and therefore not a finding.

**What it would take to settle it:** 5-10 seeds per configuration, with the
training run as the unit of analysis. That is roughly 40-80 CPU-hours here and
under an hour on the GPU that `notebooks/05_colab_full_run.ipynb` targets. Until
then this repository is a rigorously-built, honestly-measured *pipeline* with one
suggestive positive result, one robust negative one, and a solid efficiency
story — not a demonstrated segmentation improvement.

## 4. Figures

All written by `compare_methods.py` into `results/figures/`.

| figure | what it shows |
|---|---|
| `training_curves.png` | loss, val Dice, and the `mask_rate` trajectories above |

> **Reading the loss panel.** The evidential curve sits well above the other
> three and that is not a sign of worse fitting — it is a *different objective*.
> Dirichlet type-II NLL plus an annealed KL is not on the same scale as
> cross-entropy plus Dice, so the four loss curves are only comparable within a
> head, never across heads. Compare validation Dice, which is head-agnostic.
| `comparison_{dice,iou,hd95,boundary_f1}.png` | point estimates with 95% bootstrap intervals |
| `reliability.png` | calibration curves — the under/overconfidence split, visually |
| `sparsification.png` | AUSE curves against the oracle and random references |
| `qualitative.png` | image, prediction, error, vacuity, dissonance per row |
| `uncertainty_separation.png` | vacuity against dissonance near `P=0.5` — shows the collapse described above, not a clean separation |
| `efficiency.png` | accuracy against parameter count |
| `dataset_samples.png` | generator output with ground-truth contours |

---

## 5. Limitations

Stated because they bound what the numbers above support.

1. **Scale.** 64×64, synthetic data, a 0.249M-parameter model, 60 labelled
   images. The ideology comparison is controlled and significant, but nothing
   here establishes a competitive absolute Dice on real dermoscopy.
2. **One seed per configuration.** Significance is established *across images*
   (paired, n=150), not across training seeds. A seed sweep would be the next
   thing to add, and it is cheap on a GPU.
3. **Calibration regresses.** See above. ECE is worse than every baseline.
4. **Dissonance tempering is unsupported by the evidence here.** The trained
   model's vacuity and dissonance are nearly collinear (mean dissonance 0.018),
   because the evidential KL term suppresses exactly the conflicting evidence
   the tempering needs. Half the proposed mechanism therefore has no measured
   contribution — see "The decomposition collapses in the trained model".
5. **Synthetic data cannot settle a clinical claim.** The generator was designed
   to contain the failure modes that matter for uncertainty, but a real archive
   contains failure modes nobody designed.
6. **The architecture comparison is only at 64px.** "Smaller and no worse" is
   demonstrated (p = 0.24, interval spanning zero, 125x fewer parameters), but at
   64px a 31M-parameter U-Net is heavily over-parameterised, which favours the
   small model. The 192-256px comparison is unrun.

---

## 6. Reproducing all of this

```bash
python -m pytest tests -q -m "not slow"   # 219 fast tests
python scripts/benchmark_efficiency.py    # §1
python scripts/compare_methods.py --set … # §3 (command above)
python scripts/run_ablation.py            # ablation + labelled-fraction sweep
```

Environment, seeds and provenance: [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
Method and derivations: [METHOD.md](METHOD.md).

A machine-generated counterpart to this document — tables only, no narrative —
is written to `results/RESULTS_generated.md` by `compare_methods.py`.
