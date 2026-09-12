"""
Script: test_docs_and_notebooks.py
Objective: Verify documented entrypoints, local links and notebook execution without remote services.
Author: Yi Yu (with assistance from Codex)
Created: 2026-04-19
Last updated: 2026-09-12
Inputs: Repository documentation, notebooks, package source files and a stubbed demo runner.
Outputs: Regression assertions.
Usage: python -m pytest tests/package/test_docs_and_notebooks.py
Dependencies: pytest; pysweb
"""

import ast
import json
from pathlib import Path
import re
import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_documented_local_links_exist():
    for path in [
        ROOT / "README.md",
        *ROOT.glob("docs/*.md"),
        ROOT / "spec/README.md",
        ROOT / "notebooks/README.md",
    ]:
        text = path.read_text()
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if target.startswith(("https:", "http:", "#")):
                continue
            assert (path.parent / target.split("#")[0]).exists(), f"{path}: {target}"
    assert (ROOT / "docs/assets/pysweb-logo.png").exists()


def test_run_notebook_executes_as_offline_script(monkeypatch, tmp_path):
    import pysweb.demo

    called = []
    monkeypatch.setattr(pysweb.demo, "run_demo", lambda output: called.append(output))
    nb = json.loads((ROOT / "notebooks/01_run_pysweb.ipynb").read_text())
    script = "\n".join(
        "".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"
    )
    monkeypatch.chdir(tmp_path)
    exec(compile(script, "run-notebook", "exec"), {})
    assert called == [tmp_path / "examples/outputs/demo"]
    assert not list(tmp_path.iterdir())


def test_notebooks_are_clean_and_python_cells_compile():
    for path in (ROOT / "notebooks").glob("*.ipynb"):
        nb = json.loads(path.read_text())
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                assert not cell.get("outputs"), path
                ast.parse("".join(cell["source"]))


def test_package_has_no_retired_core_imports():
    for path in (ROOT / "pysweb").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("core"), path
            elif isinstance(node, ast.Import):
                assert not any(
                    n.name == "core" or n.name.startswith("core.") for n in node.names
                ), path
    assert not (ROOT / "core").exists()
