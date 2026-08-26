# Development and experiment entry points.
# Every committed number in results/ comes from one of these targets.

PY := ./.venv/Scripts/python.exe
ifeq ($(OS),)
PY := ./.venv/bin/python
endif

# Overrides shared by the CPU-scale experiments. The committed results use
# exactly these; drop them (or use make compare-gpu) for a full-scale run.
CPU_OVERRIDES := data.image_size=64 data.train_size=600 data.val_size=80 \
                 data.test_size=150 data.labeled_fraction=0.10 \
                 data.batch_size=8 data.mu=1 model.width=16 model.depth=3 \
                 optim.epochs=20 optim.steps_per_epoch=20 \
                 loss.kl_anneal_epochs=8 semi.rampup_epochs=6 \
                 eval.bootstrap=2000

.PHONY: help setup test test-all lint fmt smoke data bench compare compare-gpu \
        ablate sweep figures clean clean-results

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

setup:  ## Create the environment with uv and install the package
	uv python install 3.12
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) --index-url https://download.pytorch.org/whl/cpu \
	  --extra-index-url https://pypi.org/simple torch torchvision
	uv pip install --python $(PY) -r requirements.txt
	uv pip install --python $(PY) -e . --no-deps

test:  ## Fast unit tests (seconds)
	$(PY) -m pytest tests -q -m "not slow"

test-all:  ## Everything, including end-to-end training tests (minutes)
	$(PY) -m pytest tests -q

lint:  ## Static checks
	$(PY) -m ruff check src scripts tests

fmt:  ## Auto-fix what ruff can
	$(PY) -m ruff check --fix src scripts tests

smoke:  ## Full pipeline end to end on a tiny config (~1 min)
	$(PY) scripts/train.py --config configs/smoke.yaml

data:  ## Materialise the dataset and write a preview figure
	$(PY) -m evissl.cli data --config configs/base.yaml

bench:  ## Parameters, MACs and latency per architecture (no training)
	$(PY) scripts/benchmark_efficiency.py --image-size 128

compare:  ## The headline four-way comparison at CPU scale
	$(PY) scripts/compare_methods.py --set $(CPU_OVERRIDES)

compare-gpu:  ## Full-scale comparison (needs a GPU)
	$(PY) scripts/compare_methods.py --set run.device=cuda data.image_size=192 \
	  data.train_size=2000 optim.epochs=80

ablate:  ## Component ablation and labelled-fraction sweep
	$(PY) scripts/run_ablation.py --set $(CPU_OVERRIDES)

sweep:  ## Labelled-fraction sweep only
	$(PY) -m evissl.cli sweep --set $(CPU_OVERRIDES)

figures:  ## Redraw figures from artefacts already on disk
	$(PY) scripts/make_figures.py

clean:  ## Remove caches and build artefacts
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

clean-results:  ## Remove run artefacts (keeps committed tables and figures)
	rm -rf results/runs checkpoints/*.pt
