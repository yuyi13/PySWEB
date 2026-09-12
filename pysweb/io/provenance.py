"""
Script: provenance.py
Objective: Record input and software identities and validate atomic output writes and cache reuse.
Author: Yi Yu (with assistance from Codex)
Created: 2026-09-12
Last updated: 2026-09-12
Inputs: Input files, workflow settings, package source files and output datasets.
Outputs: SHA-256 manifests, atomic JSON/NetCDF files and cache-validation results.
Usage: from pysweb.io.provenance import input_manifest, atomic_netcdf, validate_existing_output
Dependencies: numpy, xarray; a compatible NetCDF backend; optional git for revision metadata
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
import numpy as np
import xarray as xr


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.generic, np.ndarray)):
        return value.tolist()
    raise TypeError(f"Cannot serialise {type(value).__name__}")


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix="." + path.name, suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                default=_json_default,
                allow_nan=False,
            )
            handle.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def input_manifest(config, paths):
    package_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for source in sorted(package_root.rglob("*.py")):
        digest.update(str(source.relative_to(package_root)).encode())
        digest.update(source.read_bytes())
    code_hash = digest.hexdigest()
    try:
        package_version = version("pysweb")
    except PackageNotFoundError:
        package_version = "source"
    try:
        revision = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=package_root,
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            or None
        )
    except FileNotFoundError:
        revision = None
    dependencies = {}
    for package in (
        "numpy",
        "pandas",
        "xarray",
        "scipy",
        "rasterio",
        "rioxarray",
        "pyproj",
        "netCDF4",
        "earthengine-api",
    ):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = None
    inputs = [
        {
            "path": str(Path(p).resolve()),
            "sha256": sha256(p),
            "size_bytes": Path(p).stat().st_size,
        }
        for p in sorted(set(str(p) for p in paths))
    ]
    effective_config = {
        key: value
        for key, value in config.items()
        if key
        not in {"skip_existing", "workers", "output", "output_dir", "output_file"}
    }
    identity = {
        "config": effective_config,
        "inputs": inputs,
        "code_sha256": code_hash,
        "dependencies": dependencies,
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=_json_default).encode()
    ).hexdigest()
    return {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "config": dict(config),
        "inputs": inputs,
        "software": {
            "version": package_version,
            "git_revision": revision,
            "code_sha256": code_hash,
            "dependencies": dependencies,
        },
    }


def atomic_netcdf(dataset, path, **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix="." + path.name, suffix=".tmp")
    os.close(fd)
    temporary = Path(name)
    try:
        dataset.to_netcdf(temporary, **kwargs)
        with xr.open_dataset(temporary) as check:
            check.load()
            if not check.data_vars or any(
                check[name].size == 0 for name in check.data_vars
            ):
                raise ValueError("Output NetCDF is empty.")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_output_manifest(path, manifest):
    payload = {
        **manifest,
        "output": {"path": str(Path(path).resolve()), "sha256": sha256(path)},
    }
    atomic_json(str(path) + ".manifest.json", payload)


def validate_existing_output(path, manifest):
    sidecar = Path(str(path) + ".manifest.json")
    if not sidecar.exists():
        raise ValueError(
            f"Existing output has no provenance manifest: {path}; rerun without --skip-existing."
        )
    saved = json.loads(sidecar.read_text())
    if saved.get("fingerprint") != manifest["fingerprint"] or saved.get(
        "output", {}
    ).get("sha256") != sha256(path):
        raise ValueError(
            f"Existing output does not match inputs/configuration/code: {path}; rerun without --skip-existing."
        )
    with xr.open_dataset(path) as ds:
        ds.load()
        if "time" not in ds.coords or not ds.sizes.get("time"):
            raise ValueError("Existing output has no valid time dimension.")
