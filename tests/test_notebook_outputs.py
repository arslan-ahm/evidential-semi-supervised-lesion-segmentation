"""The notebooks must not write into the committed evidence directory.

This exists because of a real incident. Executing the notebooks with
``nbconvert --execute`` overwrote ``results/tables/efficiency.csv``,
``results/tables/ablation_components.csv`` and every directory under
``results/runs/`` with notebook-scale numbers, because the pipeline entry points
default to ``out_dir="results"`` and the notebooks did not override it. The
committed tables are the evidence behind every number in the README, so a
notebook run silently replacing them with lower-fidelity output is a
documentation-integrity bug, not a housekeeping one.

``.gitignore`` already reserves ``results/scratch/`` for exactly this. These
tests assert that every notebook that trains or benchmarks anything routes its
output there.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "notebooks"
SCRATCH = "results/scratch"

#: Entry points that persist run artefacts. Each one takes an ``out_dir``.
WRITING_CALLS = (
    "run_training(",
    "run_comparison(",
    "run_component_ablation(",
    "run_labeled_fraction_sweep(",
    "run_efficiency_benchmark(",
    "write_comparison_figures(",
    "write_markdown_report(",
    "write_sweep_figure(",
)

#: 05 is the Colab/GPU notebook. It runs on a fresh machine with no committed
#: results to protect, and it is not executed locally.
LOCAL_NOTEBOOKS = sorted(
    p for p in NOTEBOOK_DIR.glob("0[1-4]*.ipynb") if p.is_file()
)


def _code_source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_local_notebooks_are_discovered() -> None:
    """A glob that silently matches nothing would make every test below vacuous."""
    assert len(LOCAL_NOTEBOOKS) == 4, [p.name for p in LOCAL_NOTEBOOKS]


@pytest.mark.parametrize("path", LOCAL_NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_does_not_write_to_committed_results(path: Path) -> None:
    """No notebook may name the bare committed directory as an output."""
    source = _code_source(path)
    offenders = [
        bad
        for bad in ('out_dir="results"', "out_dir='results'",
                    '"run.out_dir=results"', '(comparison, "results")',
                    'Path("results")', '"results/RESULTS_generated.md"')
        if bad in source
    ]
    assert not offenders, (
        f"{path.name} writes into the committed results/ directory: {offenders}. "
        f"Route it to {SCRATCH}/ instead."
    )


@pytest.mark.parametrize("path", LOCAL_NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_that_trains_pins_scratch_out_dir(path: Path) -> None:
    """If a notebook calls a persisting entry point, it must aim it at scratch."""
    source = _code_source(path)
    calls = [call for call in WRITING_CALLS if call in source]
    if not calls:
        pytest.skip(f"{path.name} persists nothing")
    assert SCRATCH in source, (
        f"{path.name} calls {calls} but never mentions {SCRATCH}; "
        "its output would land on top of the committed evidence."
    )


def test_scratch_is_gitignored() -> None:
    """The scratch directory must not become committed evidence by accident."""
    ignored = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "results/scratch/" in ignored


def test_pipeline_defaults_still_point_at_results() -> None:
    """Guard the assumption this whole module rests on.

    If the library default ever changes to something harmless, these tests stop
    protecting anything and should be revisited rather than left passing.
    """
    source = (REPO_ROOT / "src" / "evissl" / "pipelines.py").read_text(
        encoding="utf-8"
    )
    assert 'out_dir: str | Path = "results"' in source, (
        "the pipeline default changed; re-check whether notebooks can still "
        "clobber committed results"
    )


@pytest.mark.parametrize("path", LOCAL_NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_code_cells_compile(path: Path) -> None:
    """Every code cell must parse.

    Added after a patch to these notebooks introduced
    ``run_comparison(out_dir=..., METHODS)`` - a positional argument after a
    keyword one. It is a SyntaxError, but nothing caught it until nbconvert
    executed the cell six minutes into a run.
    """
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        # IPython magics and shell escapes are not valid Python.
        stripped = "\n".join(
            line
            for line in source.split("\n")
            if not line.strip().startswith(("%", "!"))
        )
        try:
            compile(stripped, f"{path.name}:cell{index}", "exec")
        except SyntaxError as exc:  # pragma: no cover - the assert is the report
            pytest.fail(f"{path.name} cell {index} does not compile: {exc}")


@pytest.mark.parametrize("path", LOCAL_NOTEBOOKS, ids=lambda p: p.name)
def test_every_override_list_pins_scratch(path: Path) -> None:
    """Both arms of a ``QUICK`` switch must redirect run artefacts.

    ``run_comparison(out_dir=...)`` only moves the *tables*; per-run directories
    come from ``cfg.run.out_dir`` inside ``run_training``. So an override list
    that omits ``run.out_dir`` still writes ``results/runs/<name>/`` on top of
    the committed runs - which is exactly what happened, and it was masked
    because only the default branch had been fixed.
    """
    source = _code_source(path)
    if "optim.epochs" not in source:
        pytest.skip(f"{path.name} defines no scale list")
    n_lists = source.count("optim.epochs=")
    n_pins = source.count(f"run.out_dir={SCRATCH}")
    assert n_pins >= n_lists, (
        f"{path.name} has {n_lists} scale list(s) but only {n_pins} "
        f"run.out_dir pin(s); flipping QUICK would clobber committed runs."
    )
