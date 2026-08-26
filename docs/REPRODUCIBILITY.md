# Reproducibility

Every number in this repository is produced by a command in this file. Nothing
is transcribed from a notebook that no longer exists.

## Environment

The project pins Python 3.12 and installs with [uv](https://docs.astral.sh/uv/).

```bash
uv python install 3.12
uv venv --python 3.12 .venv

# CPU wheels (what the committed results were produced with)
uv pip install --index-url https://download.pytorch.org/whl/cpu \
               --extra-index-url https://pypi.org/simple torch torchvision
uv pip install -r requirements.txt
uv pip install -e . --no-deps
```

Verified on the exact versions below. `torch 2.13.0` has **no cp314 wheel that
imports cleanly** - it fails on a missing bundled `torchgen`, and the `torchgen`
package on PyPI is an unrelated stub that does not fix it. Use 3.12.

```
python 3.12.13
torch 2.13.0+cpu
torchvision 0.28.0+cpu
numpy 2.5.2
scipy 1.18.1
```

## Determinism

`evissl.utils.seed.seed_everything` seeds Python, NumPy and PyTorch (CPU and
CUDA) in one call and returns a separate `torch.Generator` for the data loaders,
so loader shuffling is independent of the global RNG state. Augmentation draws
from `np.random.default_rng((seed, epoch, image_index))`, which makes each
sample's transform a pure function of those three numbers - reproducible without
depending on iteration order.

`run.deterministic: true` (the default) additionally requests deterministic
cuDNN kernels and disables the autotuner.

Same seed, same config, same result:

```bash
pytest tests/test_end_to_end.py::test_run_is_reproducible_from_its_seed -m slow
```

Different seed, different result - a guard against an accidentally constant
pipeline:

```bash
pytest tests/test_end_to_end.py::test_seed_changes_the_result -m slow
```

### Verified on the committed results, not just in a test

The `evidential` run of the method comparison and the `full` row of the
component ablation are the same configuration and seed, launched as two separate
processes about an hour apart. Their per-image Dice values are **bit-identical**
across all 150 test images:

```
results/runs/evidential/per_image.csv     dice mean 0.786363
results/runs/ablation_full/per_image.csv  dice mean 0.786363
max |difference| over 150 images: 0.0
```

That is the determinism claim demonstrated on real artefacts rather than
asserted.

## Data

The default dataset is **procedurally generated** and needs no download, which
is what makes this repository verifiable end to end. `evissl.data.synthetic`
produces dermoscopy-like images with the failure modes that matter for
uncertainty research: variable-sharpness boundaries, low-contrast lesions, hair
and ruler occluders, and non-convex Fourier-perturbed outlines. A dataset is a
pure function of one integer.

Real archives are supported and required for any claim about real data:

```bash
python scripts/download_isic.py            # prints retrieval steps (ISIC needs consent)
python scripts/download_isic.py --extract  # arranges ZIPs into images/ and masks/
python scripts/train.py --config configs/isic2018.yaml
```

PH2 works the same way with `data.name=ph2`, and is intended as an *external*
test set for cross-dataset generalisation.

## Reproducing the results

```bash
make test          # 219 fast tests, seconds
make smoke         # full pipeline end to end, ~1 minute
make bench         # parameters, MACs, latency - no training needed
make compare       # the headline table and figures
make ablate        # component ablation + labelled-fraction sweep
```

`make compare` runs the exact command recorded at the top of
[RESULTS.md](RESULTS.md), including every override, so the committed tables can
be regenerated verbatim.

## Compute and what that means for scale

The committed results were produced on **CPU only**: a 2-physical-core
Skylake-U laptop, no GPU. That constrains scale, and the repository is explicit
about which numbers are which:

- `results/tables/efficiency.csv` - parameters, MACs and latency. Hardware
  measurements, complete and final on this machine.
- `results/tables/method_comparison.csv` - the four-way ideology comparison at
  64px on synthetic data. Real training, real statistics, small scale.
- `results/tables/seed_variance.csv` - three training seeds of one
  configuration. The number that decides which accuracy claims are supportable.
- ISIC 2018 at 256px is **not** run locally; `notebooks/05_colab_full_run.ipynb`
  runs it on a free Colab GPU.

Anything not present in `results/` was not run. There are no placeholder numbers
in this repository.

### Training cost, measured

The claim that a U-Net comparison was infeasible on this machine needed
checking, and it was only half right. Median seconds per training step
(forward + backward + optimiser), 2 threads:

| model | config | params | s / step | one 20-epoch x 20-step run |
|---|---|---|---|---|
| `separable_unet` | 64px, bs 8 (the comparison config) | 0.25 M | **0.45** | 0.05 h |
| `unet` | 64px, bs 8 | 31.04 M | **2.92** | 0.32 h |
| `unet` | 128px, bs 8 | 31.04 M | 9.17 | 1.02 h |
| `unet` | 192px, bs 16 (the GPU config) | 31.04 M | **62.46** | 6.94 h |

Two corrections fall out of this:

- **A U-Net accuracy baseline at the comparison scale is entirely feasible here**
  - about 20 minutes - and an earlier draft of this document wrongly called it
  infeasible. It has since been run and is reported in `docs/RESULTS.md`.
- **The full-scale configuration genuinely is not.** At 192px and batch 16 a
  single 20-epoch run is ~7 hours, and `notebooks/05_colab_full_run.ipynb`
  targets 60 epochs, so ~21 hours for one run of one method. That is what the
  GPU notebook exists for.

Also worth noting for anyone reading the wall-clock numbers in the logs: the
6.5x step-time ratio between the two architectures at 64px is close to the 5.87x
inference-latency ratio in the efficiency table, as it should be.


## Measurement pitfalls found while building this

Recorded because they produced wrong numbers here, and would in any similar
project.

**Latency needs more warm-up than seems reasonable.** PyTorch caches a
convolution algorithm per input shape on first use. With three warm-up
iterations the 0.11M-parameter model measured 403 ms and appeared *slower than a
model 8x its size*; with eight warm-up iterations it measures 49 ms. The default
in `measure_latency` is now 8, and the reason is documented at the call site.

**MACs are a poor proxy for CPU latency.** The proposed model has 36x fewer MACs
than the U-Net baseline but is only about 6x faster in wall-clock. Dense
convolutions vectorise well; the narrow depthwise convolutions that produce the
MAC saving are memory-bandwidth bound. The efficiency table therefore reports
achieved MACs/ms alongside the ratios, so the gap is visible instead of implied.

**Equal-width ECE understates segmentation miscalibration.** Almost every pixel
is confidently background, so nearly all of them land in the top bin and the
estimator collapses to a single average. ACE (equal-mass bins) is reported
alongside it.

## Provenance

Each run writes, under `results/runs/<name>/`:

- `config.yaml` - the fully resolved configuration, after `_base_` inheritance
  and every `--set` override
- `history.jsonl` - one record per epoch, so curves can be replotted without
  re-training
- `per_image.csv` - per-image metrics, which is what the paired statistics need
- `summary.json` - the flat scalar row

Checkpoints in `checkpoints/<name>.pt` carry the student weights, the EMA
teacher weights, the epoch, the monitored metric and the full config dict, so
any saved model can be re-scored without its original command line:

```bash
python scripts/evaluate.py --config configs/evidential.yaml --set eval.tta=true
```
