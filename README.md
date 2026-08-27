![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.13-red)
![Tests](https://img.shields.io/badge/tests-231%20passing-brightgreen)
![Params](https://img.shields.io/badge/params-0.96M-orange)
![License](https://img.shields.io/badge/License-MIT-green)

# Calibrated Uncertainty over Confidence Thresholds

**Evidential semi-supervised skin lesion segmentation with a sub-1M-parameter network**

Semi-supervised segmentation decides which unlabelled pixels the student should
learn from. Almost every method makes that decision with a confidence threshold
— keep the pixel if `max_k p_k > 0.95`. This repository argues that the
threshold is asking the wrong question, replaces it with a subjective-logic
decomposition of an evidential head, and measures the difference under paired
statistical tests.

> **Result, up front.** The efficiency claims hold (32× fewer parameters, 6.4×
> lower latency, measured). The *accuracy* claim does not: a seed study run after
> the main experiment showed most of the differences are inside training noise,
> and it retracts the boundary-F1 result this project was designed around. The
> one robust statistical finding is a **negative** one about this method's
> calibration. Details and the retraction are in
> [Results](#results) — the numbers were not restated quietly.

---

## The one-paragraph argument

Take two pixels that a network scores at `P(lesion) = 0.5`. The first sits in a
region unlike anything in the labelled set: 0.5 means *"I have no idea"*. The
second sits on the boundary of a low-contrast lesion, where the transition
genuinely spans several pixels: 0.5 is the *correct answer*. A softmax reports
these identically, so a threshold discards both — and the second one was real
information about where the boundary lies. That matters more than it sounds,
because Dice is dominated by the easy lesion interior while the errors that
actually count live in a few-pixel band around the contour. **A threshold-based
method is structurally blind to its hardest examples.**

An evidential head separates the two cases, and this repository asserts it
numerically rather than claiming it:

| case | logits | `P(lesion)` | vacuity (epistemic) | dissonance (aleatoric) |
|---|---|---|---|---|
| no evidence | `[-60, -60]` | 0.500 | **1.000** | 0.000 |
| conflicting evidence | `[+30, +30]` | 0.500 | 0.032 | **0.968** |
| confident lesion | `[-30, +30]` | 0.969 | 0.062 | 0.000 |

Same probability, opposite epistemic state.
→ `tests/test_uncertainty.py::test_conflicting_evidence_is_dissonance_not_vacuity`

## The method: vacuity gates, dissonance tempers

The two uncertainty types call for two different actions, and that split is the
contribution.

**Vacuity gates the weight.** Each unlabelled pixel enters the consistency loss
weighted by `1 - u`, the teacher's total belief mass — literally how much
evidence it has accumulated there. Continuous, bounded, no threshold to tune.

**Dissonance tempers the target.** The sharpening temperature is set per pixel
as `T_eff = T + (1 - T) * diss`, so a decided pixel is sharpened fully and a
genuinely ambiguous one is left alone. Sharpening an ambiguous boundary pixel
manufactures a confident label the image does not support, and the student then
learns that fabrication. This costs no extra hyper-parameter.

> **Only the first half survives measurement.** The vacuity gate produces a
> significant, largest-effect improvement. Dissonance tempering turns out to be
> *dormant* in the trained model — the evidential KL term suppresses the very
> conflicting evidence it needs — so this repository does not claim it
> contributes. See [What survives](#what-survives).

|  | per-pixel weight | target |
|---|---|---|
| Mean Teacher | `1` | teacher probability |
| FixMatch | `1[max_k p_k > τ]` | one-hot `argmax` |
| **ours** | `1 - u` (vacuity) | dissonance-tempered sharpen |

Full derivation, including why binary dissonance collapses to `2·min(b₀, b₁)`:
**[docs/METHOD.md](docs/METHOD.md)**

## Efficiency, measured rather than quoted

The backbone is a depthwise-separable U-Net with a gated axial bottleneck.
Measured on CPU at 128×128, batch 1:

| model | params | GMACs | latency bs=1 | latency bs=8 | params ↓ | MACs ↓ | latency ↓ | achieved MMAC/ms |
|---|---|---|---|---|---|---|---|---|
| `separable_unet_tiny` | 0.113 M | 0.083 | 38.0 ms | 217 ms | 275x | 164x | 10.52x | 2.19 |
| **`separable_unet` (default)** | **0.959 M** | **0.380** | **68.1 ms** | 523 ms | 32x | 36x | **5.87x** | 5.57 |
| `unet` | 31.038 M | 13.653 | 399.7 ms | 2959 ms | 1x | 1x | 1.00x | 34.16 |

Two things worth saying out loud, because most papers do not:

**A 36× MAC reduction buys only a 5.87× speed-up.** Dense convolutions
vectorise well (34.2 MMAC/ms); the narrow depthwise convolutions that
*produce* the MAC saving are memory-bandwidth bound (5.6 MMAC/ms). MACs are a
poor proxy for latency, so the table reports achieved throughput and all three
ratios rather than implying they are the same number.

**Shrinking further pays much less than the MAC count suggests.** The 0.11M model
has 4.6× fewer MACs than the 0.96M one and is only 1.8× faster — below a certain
width the network is memory-bound and the arithmetic saving stops buying time.

And the uncertainty itself is nearly free, which is the practical case for an
evidential head over sampling:

| uncertainty source | forward passes | cost | epistemic/aleatoric split |
|---|---|---|---|
| evidential head | 1 | 68 ms | yes |
| MC dropout | 8 | 545 ms | yes |
| 4-flip TTA | 4 | 272 ms | no |

(The multipliers are arithmetic — N passes cost N× one pass. The claim is about
pass count: one, versus eight for the same decomposition.)

## Results

Four methods, one training loop, 600 images with **60 labelled (10%)**, 150 test
images. `semi.method` is the only thing that differs between them.

| run | Dice ↑ | IoU ↑ | HD95 ↓ | BF1 ↑ | ECE ↓ | MCE ↓ |
|---|---|---|---|---|---|---|
| supervised_baseline | 0.7519 | 0.6340 | 7.610 | 0.5525 | 0.0361 | 0.1766 |
| mean_teacher | 0.7748 | 0.6568 | 6.839 | 0.5729 | **0.0246** | 0.1070 |
| fixmatch | 0.7755 | 0.6575 | **6.702** | 0.5764 | 0.0309 | 0.1558 |
| evidential (ours) | **0.7864** | **0.6712** | 6.786 | **0.5954** | 0.0625 | **0.0770** |

### The honest reading, which is not the flattering one

I ran a seed study after the main experiment — the same configuration, three
training seeds — and it **retracts this repository's headline claim.**

| metric | seed sd (same config, 3 runs) | range | gain vs baseline | ratio to run-to-run scale |
|---|---|---|---|---|
| Dice | 0.0132 | 0.0242 | +0.0345 | 1.85 → suggestive |
| IoU | 0.0200 | 0.0387 | +0.0371 | 1.31 → inside noise |
| **Boundary F1** | **0.0566** | **0.1117** | +0.0428 | **0.53 → inside noise** |
| ECE | 0.0017 | 0.0031 | +0.0379 | 15.6 → robust |

The paired Wilcoxon tests reported boundary F1 at p = 0.010, and I described it
as "the result that matters" — the one metric where only this method reached
significance. **That does not survive.** Boundary F1 moves by up to 0.112 between
seeds of the *identical* configuration; the claimed gain is half the noise.

The tests were not computed wrongly. They were answering the wrong question. A
paired test conditions on **one trained model per method** and asks whether the
difference is consistent across images — which it correctly answers. But the
claim "this method is better" treats the **training run** as the sampling unit,
and there was one run per method. No number of test images fixes an n of 1.

### What survives

**Measured, not inferred — these are solid:**

- **Efficiency.** 0.96M vs 31.0M parameters, 36× fewer MACs, **5.87× lower
  latency** (399.7 → 68.1 ms), warm-up-corrected and repeated.
- **Smaller costs nothing measurable.** Trained supervised-only under an
  identical schedule, the 0.25M model scores 0.7519 Dice against the 31M
  U-Net's 0.7377 — a difference of +0.0142 with a 95% interval spanning zero
  (p = 0.24), at **125× fewer parameters**. The supportable claim is *no worse*,
  not better.
- **The analytic decomposition.** `[-60,-60]` and `[+30,+30]` provably map to
  opposite corners of the uncertainty space at identical `P = 0.5`. Arithmetic.
- **The mechanism difference.** FixMatch's consistency branch contributes
  *exactly* 0.000 / 0.000 / 0.004 in epochs 0–2 while the vacuity gate
  contributes 0.397 from epoch 0. Logged quantities, not estimates.
- **Bit-identical reproducibility** across separate invocations (max difference
  0.0 over 150 per-image scores).

**The one robust statistical result is negative.** The evidential head's average
calibration is materially worse (ECE 0.0625 vs 0.0246–0.0361, ~16× the
run-to-run scale). Its *worst-case* calibration is the best of the four
(MCE 0.0770 vs 0.1070–0.1766) — every softmax baseline is over confident by
+0.025…+0.036 while this method is *under*confident by −0.046, which is
structural: with `α = e + 1` the Dirichlet mean is shrunk toward `1/K`.

**Suggestive:** semi-supervised training beats the supervised bound on Dice
(~1.9× the run-to-run scale, same sign for all three methods).

**Not supported:** any ranking among the three semi-supervised methods, the
boundary-F1 advantage, or any individual component's contribution — all 0.5–1.3×
the noise scale. The ablation also **flips sign on 4 of 5 variants** between a
12- and a 20-epoch schedule, which is independent confirmation that single-run
component attribution is not meaningful at this scale.

**Dissonance tempering is inert regardless.** Mean dissonance is 0.018 and the
two uncertainty components are nearly collinear in the trained model, so
AUSE is *identical* for vacuity+dissonance, entropy and the softmax margin. I
first blamed the KL regulariser and **the ablation refuted that** (`kl_weight=0`
moves dissonance 0.0259 → 0.0257); the type-II NLL itself is the cause.

Settling the accuracy question needs 5–10 seeds per configuration — ~40–80 CPU
hours here, under an hour on the GPU that `notebooks/05_colab_full_run.ipynb`
targets. Until then this is a rigorously built and honestly measured *pipeline*
with a solid efficiency story, one suggestive positive, and one robust negative —
not a demonstrated segmentation improvement.

Full tables, tests, ablation, seed study and limitations:
**[docs/RESULTS.md](docs/RESULTS.md)**

## Quickstart

Runs end to end with **no dataset download** — the default data source is a
procedural dermoscopy generator (see below).

```bash
git clone https://github.com/arslan-ahmad/evidential-semi-supervised-lesion-segmentation
cd evidential-semi-supervised-lesion-segmentation

# uv handles the interpreter too; Python 3.12 is required (see note below)
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --index-url https://download.pytorch.org/whl/cpu \
               --extra-index-url https://pypi.org/simple torch torchvision
uv pip install -r requirements.txt
uv pip install -e . --no-deps
```

```bash
# 1. Prove it works end to end (~1 minute, real training, real metrics)
python scripts/train.py --config configs/smoke.yaml

# 2. Efficiency table, no training needed
python scripts/benchmark_efficiency.py

# 3. The headline four-way comparison with paired statistics and figures.
#    This is the exact command that produced results/tables/ (~25 min, 2 CPU cores).
python scripts/compare_methods.py --set \
    data.image_size=64 data.train_size=600 data.val_size=80 data.test_size=150 \
    data.labeled_fraction=0.10 data.batch_size=8 data.mu=1 \
    model.width=16 model.depth=3 \
    optim.epochs=20 optim.steps_per_epoch=20 \
    loss.kl_anneal_epochs=8 semi.rampup_epochs=6 \
    eval.bootstrap=2000

# 4. Component ablation + labelled-fraction sweep
python scripts/run_ablation.py

# 5. Tests
python -m pytest tests -q -m "not slow"     # 219 tests, seconds
python -m pytest tests -q                   # all 231, incl. end-to-end training
```

On Linux/macOS/Colab the same targets are available through `make`
(`make setup`, `make smoke`, `make bench`, `make compare`, `make ablate`,
`make test`). `make` is not required.

> **Python 3.12, not 3.14.** `torch 2.13.0` has no cp314 wheel that imports
> cleanly — it fails on a missing bundled `torchgen`, and the `torchgen` package
> on PyPI is an unrelated stub that does not fix it.

## The dataset problem, and how this repo avoids it

ISIC and PH2 require registration and a multi-gigabyte download, which makes a
repository impossible to verify. So the default data source is
`evissl.data.synthetic`: a procedural dermoscopy generator built around the
failure modes that actually matter for uncertainty research.

- **Variable-sharpness boundaries** — edge softness is drawn per sample, so a
  real fraction of the set is genuinely ambiguous rather than uniformly crisp.
- **Low-contrast lesions** — pigment contrast goes down to 0.06, so some samples
  are legitimately hard.
- **Occluders that cross the boundary** — hair strands as tapering Bézier
  curves, ruler ticks, specular highlights. The mask is taken from the
  *pre-occlusion* field, because an annotator sees through hair; a generator
  that let hair fragment the label would be teaching the wrong thing.
- **Non-convex outlines** — a low-order Fourier perturbation of the radius plus
  optional satellite blobs, which punishes models that only learn ellipses.

Every sample is a pure function of one integer, so a dataset is reproducible
from a seed. Each carries its latent difficulty factors, which makes something
real datasets cannot offer possible: **validating that predicted uncertainty
rises with ground-truth difficulty** (`notebooks/04`).

Real archives are first-class when you have them:

```bash
python scripts/download_isic.py            # prints retrieval steps
python scripts/download_isic.py --extract  # arranges the ZIPs
python scripts/train.py --config configs/isic2018.yaml
```

## What makes the comparison trustworthy

Four design decisions do more for credibility here than any architectural
detail, and each is enforced by a test.

**One training loop for all four methods.** `evissl/engine/trainer.py` handles
supervised-only, Mean Teacher, FixMatch and ours. Separate scripts per method
would let a difference in results come from an incidental difference in
schedule, augmentation or checkpoint selection. Here `semi.method` is the only
variable.

**Fixed gradient steps per epoch.** An epoch is `optim.steps_per_epoch` steps,
not one pass over the labelled set. Otherwise a 5%-labelled run would take 2
steps per epoch and a 50%-labelled run 25 — the sweep would vary training length
as well as label count. *(This was a real bug in an early version of this repo,
where a docstring claimed the property the code did not have.)*
→ `test_steps_per_epoch_is_fixed_regardless_of_label_count`

**Nested labelled subsets.** Raising `labeled_fraction` strictly *adds* images to
the previous set, making each rung of the sweep a controlled comparison rather
than an unrelated random draw.
→ `test_labeled_subsets_are_nested_across_fractions`

**Geometrically aligned consistency views.** The geometric transform is applied
once and shared by the weak and strong views; only photometric perturbation and
cut-out differ. A pixel-wise consistency loss between misaligned views is pure
noise — and the failure is silent, because the loss still goes down. The test
measures profile correlation: **0.98** aligned versus **0.58** under a 5-pixel
shift. Cut-out regions are excluded through a `valid` mask, since the student
cannot infer a label for pixels it was never shown.
→ `test_weak_and_strong_views_are_spatially_aligned`

## Evaluation

Overlap metrics alone would not reveal whether any of this works, so three
families are reported:

- **Overlap** — Dice, IoU, sensitivity, specificity, precision, accuracy.
- **Boundary** — HD95, ASSD, BF score at 2 px tolerance, from symmetric surface
  distances. Where the interesting errors live.
- **Calibration & uncertainty quality** — ECE, equal-mass ACE, Brier, NLL,
  **AUSE** and uncertainty-error AUROC. AUSE asks whether the uncertainty
  *ranks* the errors, and being scale-invariant it makes vacuity, dissonance,
  entropy and MC-dropout spread directly comparable.

Two conventions worth flagging:

*Undefined means NaN, not a convenient number.* Surface distances to an empty
prediction have no value, so they are `NaN` and excluded, with the contributing
count reported next to every mean. Substituting the image diagonal — as some
implementations do — rewards a model for failing completely instead of recording
that it failed. Same for precision with no positive predictions.

*Differences are tested, not eyeballed.* Wilcoxon signed-rank on per-image
values, a paired bootstrap interval on the mean difference, and
Holm-Bonferroni correction across the family. With 150 test images and per-image
Dice standard deviations near 0.1, a few points of Dice is inside the noise.

## Repository layout

```
configs/            One YAML per experiment, composed through `_base_`
  base.yaml           shared defaults
  supervised_baseline / mean_teacher / fixmatch / evidential
  unet_baseline       the 31M reference model
  smoke.yaml          ~1 minute end-to-end verification
  isic2018.yaml       real data at 256px

src/evissl/
  config.py           typed config + `_base_` inheritance + --set overrides
  uncertainty.py      the Dirichlet decomposition (vacuity, dissonance, belief)
  losses.py           evidential NLL/KL, and all four consistency rules
  data/
    synthetic.py        procedural dermoscopy generator
    datasets.py         splits, nested labelled subsets, real-archive loaders
    transforms.py       shared-geometry view construction
    loaders.py          the fixed-step iteration protocol
  models/
    efficient_unet.py   SepUNet, 0.96M params
    unet.py             31M reference baseline
    blocks.py           separable convs, gated axial attention, zero-init gates
  engine/
    trainer.py          ONE loop for every method
    ema.py              EMA teacher, with the buffer and warm-up subtleties
  metrics/
    segmentation.py     overlap + boundary distance
    calibration.py      ECE/ACE/Brier/NLL, sparsification, AUSE, AUROC
    stats.py            bootstrap, paired Wilcoxon, Holm-Bonferroni
  eval.py  pipelines.py  report.py  viz.py  cli.py

scripts/            train, evaluate, compare_methods, run_ablation,
                    benchmark_efficiency, make_figures, download_isic
notebooks/          01 data & uncertainty · 02 train & compare · 03 ablations
                    04 calibration & uncertainty · 05 Colab full-scale run
tests/              231 tests: config, data, models, uncertainty, losses,
                    metrics, engine, report/figures, end-to-end
docs/               METHOD.md · RESULTS.md · REPRODUCIBILITY.md
results/            tables/ and figures/ produced by the commands above
```

## Notebooks

| notebook | what it establishes | needs a GPU |
|---|---|---|
| `01_data_and_uncertainty.ipynb` | the generator is a real benchmark; vacuity ≠ dissonance, shown analytically over the whole logit plane | no |
| `02_train_and_compare.ipynb` | trains all four rules through one loop, with paired statistics | no |
| `03_ablations.ipynb` | which component does the work; where the label-budget gain lives | no |
| `04_calibration_and_uncertainty.ipynb` | is the uncertainty honest, and does it rank the errors | no |
| `05_colab_full_run.ipynb` | ISIC 2018 at 256px and the 31M U-Net accuracy baseline | yes |

## Configuration

One YAML fully describes an experiment, and every run saves its resolved config
next to its results. Configs compose through `_base_`, and any leaf can be
overridden from the command line:

```bash
python scripts/train.py --config configs/evidential.yaml \
  --set optim.epochs=80 data.image_size=256 semi.vacuity_cutoff=0.7
```

Unknown keys are rejected rather than ignored — a silently-swallowed typo would
void an experiment.

## Citation

```bibtex
@software{sajid2026evissl,
  author = {Arslan Ahmad},
  title  = {Calibrated Uncertainty over Confidence Thresholds: Evidential
            Semi-Supervised Skin Lesion Segmentation},
  year   = {2026},
  url    = {https://github.com/arslan-ahmad/evidential-semi-supervised-lesion-segmentation}
}
```

## License

MIT — see [LICENSE](LICENSE).
