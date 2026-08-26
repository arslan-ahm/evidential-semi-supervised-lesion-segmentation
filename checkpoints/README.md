# Checkpoints

Weights are **not committed** (see `.gitignore`) — they are cheap to reproduce
and would otherwise dominate the repository size. Training writes
`checkpoints/<run.name>.pt` here.

## What a checkpoint contains

Enough to re-score a model without its original command line:

| key | contents |
|---|---|
| `model` | student weights |
| `ema_model` | EMA teacher weights — usually the better model, and what evaluation loads by default |
| `epoch` | epoch the checkpoint was taken at |
| `metrics` | validation metrics at that epoch |
| `config` | the fully resolved config dict, after `_base_` inheritance and every `--set` override |

Only the **best** epoch by `run.monitor` (default `dice`) is kept, so a
checkpoint is not the final epoch unless the final epoch happened to be best.
The monitored metric is tracked in maximise-always form internally, so
`monitor: hd95` selects the *lowest* HD95 rather than silently inverting.

## Re-scoring a saved model

```bash
# The EMA teacher on the test split
python scripts/evaluate.py --config configs/evidential.yaml

# The student instead of the teacher
python scripts/evaluate.py --config configs/evidential.yaml --student

# Validation split, or different evaluation options, without re-training
python scripts/evaluate.py --config configs/evidential.yaml --split val
python scripts/evaluate.py --config configs/evidential.yaml --set eval.tta=true
python scripts/evaluate.py --config configs/evidential.yaml --set eval.mc_dropout=8
```

Note that `eval.tta` and `eval.mc_dropout` average predictions across passes,
which discards the Dirichlet parameterisation that vacuity and dissonance are
defined on. Those two maps are therefore only produced in the single-pass mode;
the uncertainty analysis falls back to the MC-dropout epistemic term or to
predictive entropy in the other modes.

## Loading one by hand

```python
from evissl.config import load_config
from evissl.models import build_model
from evissl.utils.checkpoint import load_checkpoint

cfg = load_config("configs/evidential.yaml")
model = build_model(cfg.model)
payload = load_checkpoint("checkpoints/evidential.pt", model, prefer_ema=True)

print(payload["loaded"])   # "ema_model" or "model"
print(payload["epoch"], payload["metrics"])
```

## Colab

`notebooks/05_colab_full_run.ipynb` zips `results/` at the end so a session's
artefacts survive it. Checkpoints are not included by default — a full-scale
run produces several, and they are large. Copy them to Drive explicitly if you
want to keep them:

```python
!cp checkpoints/*.pt /content/drive/MyDrive/evissl/
```
