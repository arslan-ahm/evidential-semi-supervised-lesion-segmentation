# Data

This directory is empty on a fresh clone, and **that is enough to run
everything**. The default data source is procedural, so no download is needed
for any command in the top-level README.

## Default: the procedural generator

`evissl.data.synthetic` synthesises dermoscopy-like images and their masks. It
exists because ISIC and PH2 both require accepting terms of use and downloading
multiple gigabytes, which makes a repository impossible for a reader to verify.

It is not a placeholder. It is built around the four properties that determine
whether an *uncertainty* method works at all:

| property | why it is there |
|---|---|
| edge softness drawn per sample (0.5–7 px blur) | creates genuine aleatoric ambiguity — the regime where a confidence threshold discards usable pixels |
| pigment contrast down to 0.06 | a real fraction of samples is legitimately hard, not uniformly easy |
| hair, ruler ticks, specular highlights | occluders that cross the lesion boundary, the standard dermoscopic nuisance factors |
| Fourier-perturbed radius + satellite blobs | non-convex outlines, which punish models that only learn ellipses |

Two design details matter:

- **The mask comes from the pre-occlusion field.** A human annotator sees
  *through* hair, so hair must not change the label. A generator that let hair
  fragment the mask would be teaching the model something false.
  → `tests/test_data.py::test_occluders_do_not_change_the_label`
- **A sample is a pure function of one integer.** A whole dataset is
  reproducible from a seed, and train/val/test streams are guaranteed disjoint
  by construction (distinct base seeds with a large odd stride).

Each sample also carries its latent difficulty factors (`contrast`,
`edge_softness`, `area_fraction`, `n_hairs`, …) and a composite
`difficulty()` score. That enables a validation a real dataset cannot support:
checking that **predicted uncertainty rises with ground-truth difficulty**
(`notebooks/04_calibration_and_uncertainty.ipynb`).

Preview it:

```bash
python -m evissl.cli data --config configs/base.yaml --n 8
# -> results/figures/dataset_samples.png
```

## Real archives

Both are optional, and both are needed for any claim about real data.

### ISIC 2018 Task 1 (lesion boundary segmentation)

2594 dermoscopic images with expert masks. Requires consent, so it cannot be
fetched non-interactively — the helper prints the steps rather than pretending
otherwise.

```bash
python scripts/download_isic.py            # instructions + current state
# ... download the two ZIPs into data/isic2018/ ...
python scripts/download_isic.py --extract  # arrange into images/ and masks/
python scripts/download_isic.py --verify   # count and pair-check
python scripts/train.py --config configs/isic2018.yaml
```

### PH2

200 dermoscopic images from Hospital Pedro Hispano. Small enough to be
convenient, and used here as an **external** test set: training on ISIC and
evaluating on PH2 measures cross-dataset generalisation, which is a much harder
and more honest question than a within-dataset split.

```bash
python scripts/train.py --config configs/isic2018.yaml --set data.name=ph2
```

## Expected layout

Both real datasets use the same structure. Masks are paired to images **by file
stem**, tolerating the three common naming schemes (`<stem>`,
`<stem>_segmentation` as ISIC ships them, and `<stem>_lesion` as PH2 does).

```
data/
├── README.md              (this file)
├── isic2018/
│   ├── images/            ISIC_0000000.jpg ...
│   ├── masks/             ISIC_0000000_segmentation.png ...
│   └── cache_256.npz      generated on first use
└── ph2/
    ├── images/
    └── masks/
```

## Caching

Decoding and resizing 2594 JPEGs is the slowest step of a real-data run, so the
first run writes `cache_<image_size>.npz` and every later run at that resolution
loads it directly. One cache file per resolution. Disable with
`--set data.cache=false`.

## Split protocol

Fixed and seeded, so a result always traces back to a specific partition.

- 70 / 10 / 20 train / val / test, from a single seeded permutation. (The
  official ISIC 2018 test masks are not public, which is why the training set is
  re-split rather than used whole.)
- The labelled subset is the **head** of a fixed shuffle of the training split.
  Raising `labeled_fraction` therefore strictly *adds* images to the previous
  labelled set, which is what makes the labelled-fraction sweep a controlled
  comparison instead of a series of unrelated random draws.
  → `tests/test_data.py::test_labeled_subsets_are_nested_across_fractions`

## Licensing

Nothing in this directory is committed. ISIC and PH2 carry their own terms —
consult the source archives before redistributing anything derived from them.
The procedural generator produces no real patient data and is covered by this
repository's MIT licence.
