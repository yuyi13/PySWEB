#!/usr/bin/env python3
"""
Script: test_docs_and_notebooks.py
Objective: Verify the run notebook and README files document the canonical package-backed notebook workflow.
Author: Yi Yu
Created: 2026-04-19
Last updated: 2026-05-15
Inputs: Repository README files and the notebooks/01_run_pysweb.ipynb notebook.
Outputs: Pytest assertions.
Usage: python -m pytest tests/package/test_docs_and_notebooks.py -q
Dependencies: json, pathlib, re, pytest
"""
import json
from pathlib import Path
import re


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
    assert "ERA5LAND_PRECIP_SOURCE = MET_STACK_DIR" in code_text
    assert "SSEBOP_ET_SOURCE = SSEBOP_OUTPUT_DIR" in code_text
    assert 'rain_var = "precipitation"' in code_text
    assert "SWB_START_DATE" in code_text
    assert "openlandmap_missing_soc_g_per_kg = 5.0" in code_text
    assert "skip_reference_ssm = not include_reference_ssm" in code_text
    assert "include_reference_ssm = RUN_SWB_CALIBRATE and CALIB_SAME_AS_RUN" in code_text

    assert "SWB preprocess and calibration are still driven by" not in markdown_text
    assert "workflow-script-only" not in markdown_text
    assert "/g/data/ym05/sweb_model/notebook_runs" not in code_text
    assert "GEE_PROJECT" in code_text
    assert "GEE_CONFIG" not in code_text
    assert 'DEM_DIR = RUN_DIR / "inputs" / "dem"' in code_text
    assert 'PREPARED_DEM = DEM_DIR / "nasadem.tif"' in code_text
    assert "gee_project = GEE_PROJECT" in code_text
    assert "dem_dir = str(DEM_DIR)" in code_text
    assert "dem = str(PREPARED_DEM)" in code_text
    assert "`SM_RES` is the SWB preprocess target grid resolution." in markdown_text


def test_run_notebook_separates_run_and_calibration_date_ranges():
    _, code_text = _read_notebook_sections("notebooks/01_run_pysweb.ipynb")

    assert re.search(r"(^|\n)DATE_RANGE\s*=", code_text) is None
    assert "RUN_DATE_RANGE =" in code_text
    assert "CALIB_DATE_RANGE =" in code_text
    assert "RUN_START_DATE, RUN_END_DATE = _parse_date_range(RUN_DATE_RANGE)" in code_text
    assert "date_range = RUN_DATE_RANGE" in code_text
    assert "date_range = CALIB_DATE_RANGE" in code_text
    assert "CALIB_START_DATE, CALIB_END_DATE = _parse_date_range(CALIB_DATE_RANGE)" in code_text
    assert code_text.index("if RUN_SWB_CALIBRATE:\n    CALIB_START_DATE") < code_text.index("pysweb.swb.calibrate(")
    assert "date_range = [CALIB_SWB_START_DATE, CALIB_SWB_END_DATE]" in code_text
    assert '"start_date": RUN_SWB_START_DATE' in code_text
    assert '"end_date": RUN_SWB_END_DATE' in code_text


def test_run_notebook_puts_execution_toggles_before_paths():
    _, code_text = _read_notebook_sections("notebooks/01_run_pysweb.ipynb")

    expected_order = [
        "RUN_PREPARE_INPUTS =",
        "RUN_SSEBOP =",
        "RUN_SWB_PREPROCESS =",
        "RUN_SWB_CALIBRATE =",
        "RUN_SWB =",
        "PROJECT_DIR =",
    ]
    positions = [code_text.index(text) for text in expected_order]

    assert positions == sorted(positions)


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

    assert "├── dem/" in readme_text
    assert "│   │   ├── landsat.py" in readme_text
    assert "pysweb.soil" in readme_text
    assert "pysweb.visualisation" in readme_text
    assert "pysweb.visualisation" in notebooks_readme_text
    assert "workflows/6_plot_results.py" in readme_text
    assert "workflows/6_plot_results.py" in notebooks_readme_text
    assert "├── core/" not in readme_text
    assert "├── visualisation/" not in readme_text


def test_readmes_document_explicit_gee_project_and_prepared_dem_contract():
    readme_text = _read_text("README.md")
    notebooks_readme_text = _read_text("notebooks/README.md")

    for text in (readme_text, notebooks_readme_text):
        assert "--gee-project" in text
        assert "--gee-config" not in text
        assert "yiyu-research" not in text
        assert "GEE_PROJECT=your-gee-project bash workflows/ssebop_runner_landsat.sh" in text

    assert "run_inputs/dem/nasadem.tif" in readme_text
    assert "/path/to/run_inputs/dem/nasadem.tif" in notebooks_readme_text


def test_plotting_notebooks_use_package_entrypoints():
    heatmap_markdown_text, heatmap_code_text = _read_notebook_sections("notebooks/02_plot_heatmap.ipynb")
    time_series_markdown_text, time_series_code_text = _read_notebook_sections("notebooks/03_plot_time_series.ipynb")

    heatmap_text = f"{heatmap_markdown_text}\n{heatmap_code_text}"
    time_series_text = f"{time_series_markdown_text}\n{time_series_code_text}"

    assert "pysweb.visualisation.plot_heatmap" in heatmap_text
    assert "visualisation/plot_heatmap.py" not in heatmap_text
    assert 'PYSWEB_DIR = Path("/path/to/PySWEB")' in heatmap_code_text
    assert "--sweb-path" in heatmap_code_text
    assert "--ssebop-path" in heatmap_code_text
    assert "/g/data/ym05/sweb_model/PySWEB" not in heatmap_text
    assert "Boonaldoon" not in heatmap_text

    assert "pysweb.visualisation.plot_time_series" in time_series_text
    assert "visualisation/plot_time_series.py" not in time_series_text
    assert 'PYSWEB_DIR = Path("/path/to/PySWEB")' in time_series_code_text
    assert "--sweb-path" in time_series_code_text
    assert "--ssebop-path" in time_series_code_text
    assert "/g/data/ym05/sweb_model/PySWEB" not in time_series_text
    assert "Boonaldoon" not in time_series_text


def test_readme_drops_stale_wrapper_defaults_guidance():
    readme_text = _read_text("README.md")

    assert "pysweb.visualisation" in readme_text
    assert "environment-specific default paths" not in readme_text
    assert "near the top of each script" not in readme_text
    assert "1b_download_era5land_daily.py" not in readme_text
    assert "1c_stack_era5land_daily.py" not in readme_text
