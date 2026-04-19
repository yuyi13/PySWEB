#!/usr/bin/env python3
"""
Script: test_docs_and_notebooks.py
Objective: Verify the run notebook and README files document the canonical package-backed notebook workflow.
Author: Yi Yu
Created: 2026-04-19
Last updated: 2026-04-19
Inputs: Repository README files and the notebooks/01_run_pysweb.ipynb notebook.
Outputs: Pytest assertions.
Usage: python -m pytest tests/package/test_docs_and_notebooks.py -q
Dependencies: json, pathlib, pytest
"""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _read_text(path):
    return (ROOT / path).read_text(encoding = "utf-8")


def _read_notebook_sections(path):
    notebook = json.loads(_read_text(path))

    markdown_text = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    code_text = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    return markdown_text, code_text


def test_run_notebook_uses_package_backed_swb_calls():
    markdown_text, code_text = _read_notebook_sections("notebooks/01_run_pysweb.ipynb")

    assert "pysweb.swb.preprocess(" in code_text
    assert "pysweb.swb.calibrate(" in code_text
    assert "pysweb.swb.run(" in code_text
    assert 'e_var = "E"' in code_text
    assert 'et_var = "ET"' in code_text
    assert 't_var = "T"' in code_text

    assert "SWB preprocess and calibration are still driven by" not in markdown_text
    assert "workflow-script-only" not in markdown_text
    assert "/g/data/ym05/sweb_model/notebook_runs" not in code_text


def test_readmes_list_actual_notebook_files():
    readme_text = _read_text("README.md")
    notebooks_readme_text = _read_text("notebooks/README.md")

    expected_notebooks = (
        "01_run_pysweb.ipynb",
        "02_plot_heatmap.ipynb",
        "03_plot_time_series.ipynb",
    )

    for notebook_name in expected_notebooks:
        assert notebook_name in readme_text
        assert notebook_name in notebooks_readme_text


def test_readmes_point_to_canonical_package_modules():
    readme_text = _read_text("README.md")
    notebooks_readme_text = _read_text("notebooks/README.md")

    assert "pysweb.soil" in readme_text
    assert "pysweb.visualisation" in readme_text
    assert "pysweb.visualisation" in notebooks_readme_text
