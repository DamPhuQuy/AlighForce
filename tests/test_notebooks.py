"""Automated test suite verifying that all Jupyter notebooks in notebooks/

are self-contained, valid JSON, compile cleanly, and contain ZERO imports from alignforge.
"""

import json
from pathlib import Path
import pytest

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
EXPECTED_NOTEBOOKS = [
    "01_pretrain.ipynb",
    "02_sft.ipynb",
    "03_dpo.ipynb",
    "04_eval.ipynb",
    "pipeline.ipynb",
]


@pytest.mark.parametrize("nb_name", EXPECTED_NOTEBOOKS)
def test_notebook_json_validity(nb_name: str):
    """Verify that notebook is valid JSON and adheres to nbformat v4."""
    nb_path = NOTEBOOKS_DIR / nb_name
    assert nb_path.exists(), f"Notebook {nb_name} does not exist in {NOTEBOOKS_DIR}"

    with open(nb_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "cells" in data, f"{nb_name} missing 'cells' key"
    assert "metadata" in data, f"{nb_name} missing 'metadata' key"
    assert data.get("nbformat") == 4, f"{nb_name} nbformat is not 4"
    assert len(data["cells"]) > 0, f"{nb_name} has no cells"


@pytest.mark.parametrize("nb_name", EXPECTED_NOTEBOOKS)
def test_zero_alignforge_imports(nb_name: str):
    """Verify that NO cell in the notebook imports from alignforge."""
    nb_path = NOTEBOOKS_DIR / nb_name
    with open(nb_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for idx, cell in enumerate(data["cells"]):
        if cell.get("cell_type") == "code":
            source_lines = cell.get("source", [])
            for line_no, line in enumerate(source_lines):
                stripped = line.strip()
                assert not stripped.startswith("import alignforge"), (
                    f"Forbidden import found in {nb_name} cell #{idx} line #{line_no}: '{line.strip()}'"
                )
                assert not stripped.startswith("from alignforge"), (
                    f"Forbidden import found in {nb_name} cell #{idx} line #{line_no}: '{line.strip()}'"
                )


@pytest.mark.parametrize("nb_name", EXPECTED_NOTEBOOKS)
def test_notebook_code_cells_syntax_validity(nb_name: str):
    """Verify that all code cells in the notebook compile cleanly as valid Python code."""
    nb_path = NOTEBOOKS_DIR / nb_name
    with open(nb_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for idx, cell in enumerate(data["cells"]):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            # Filter out IPython magic commands like !pip or %matplotlib
            filtered_lines = []
            for line in source.split("\n"):
                if line.strip().startswith("!") or line.strip().startswith("%"):
                    filtered_lines.append(f"# {line}")
                else:
                    filtered_lines.append(line)
            clean_source = "\n".join(filtered_lines)

            try:
                compile(clean_source, f"{nb_name}_cell_{idx}", "exec")
            except SyntaxError as exc:
                pytest.fail(f"Syntax error in {nb_name} cell #{idx}: {exc}")


def test_master_notebook_contains_all_core_primitives():
    """Verify that the end-to-end master notebook defines all essential pipeline primitives directly."""
    nb_path = NOTEBOOKS_DIR / "pipeline.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    full_code = "\n".join(
        "".join(cell.get("source", []))
        for cell in data["cells"]
        if cell.get("cell_type") == "code"
    )

    expected_primitives = [
        "class ByteLevelBPETokenizer",
        "class RMSNorm",
        "class RotaryEmbedding",
        "class CausalSelfAttention",
        "class CausalTransformerModel",
        "class TensorKVCache",
        "class LoRALinear",
        "class Pretrainer",
        "class SFTTrainer",
        "class DPOTrainer",
        "class CompositeSafetyGuard",
        "def run_full_evaluation_pipeline",
    ]

    for prim in expected_primitives:
        assert prim in full_code, f"Missing primitive '{prim}' in pipeline.ipynb"
