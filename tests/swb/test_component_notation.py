"""
Script: test_component_notation.py
Objective: Verify canonical component I/O, legacy conflicts and prepared-file provenance.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-11
Last updated: 2026-10-11
Inputs: Synthetic component datasets, CLI/API arguments and temporary NetCDF files.
Outputs: Regression assertions for the notation migration.
Usage: python -m pytest tests/swb/test_component_notation.py
Dependencies: numpy, pandas, pytest, xarray; pysweb
"""

from importlib import import_module
import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pysweb.components import (
    COMPONENT_SCHEMA, component_variable, normalise_component_arguments,
    resolve_prepared_component_path,
)
from pysweb.io.provenance import sha256
from pysweb.swb.preprocess import _build_args, _build_target_grid
from pysweb.swb.core import load_forcing


def component(values=(1.0, np.nan), units="mm/day"):
    return xr.DataArray(np.asarray(values), dims="time", coords={"time": pd.date_range("2024-01-01", periods=2)}, attrs={"units": units})


@pytest.mark.parametrize("canonical,legacy", [("E_s", "E"), ("E_s", "e"), ("E_t", "T"), ("E_t", "t"), ("f_t", "Tc")])
def test_variable_aliases_and_identical_duplicates(canonical, legacy):
    da = component(units="1" if canonical == "f_t" else "mm/day")
    with pytest.warns(FutureWarning, match="deprecated"):
        assert component_variable(xr.Dataset({legacy: da}), canonical) == legacy
    ds = xr.Dataset({canonical: da, legacy: da.copy()})
    assert component_variable(ds, canonical) == canonical
    with pytest.warns(FutureWarning):
        assert component_variable(ds, canonical, legacy) == legacy


@pytest.mark.parametrize("mismatch", ["value", "mask", "units", "coordinates", "dimensions"])
def test_conflicting_duplicates_rejected(mismatch):
    left = component()
    right = left.copy(deep=True)
    if mismatch == "value":
        right[0] = 2.0
    elif mismatch == "mask":
        right[1] = 0.0
    elif mismatch == "units":
        right.attrs["units"] = "m/day"
    elif mismatch == "coordinates":
        right = right.assign_coords(height=2.0)
        left = left.assign_coords(depth=1.0)
    else:
        right = right.rename(time="other_time")
    # Coordinates of a Dataset are shared, so use distinct dimension coordinates
    # to represent a coordinate mismatch without xarray merging away the evidence.
    if mismatch == "coordinates":
        right = right.rename(time="other_time").assign_coords(other_time=pd.date_range("2024-02-01", periods=2))
    ds = xr.Dataset({"E_t": left, "T": right})
    with pytest.raises(ValueError, match="Conflicting component"):
        component_variable(ds, "E_t")


def test_explicit_custom_variable_and_temperature_guard():
    ds = xr.Dataset({"uptake": component(), "air_temperature": component(units="K")})
    assert component_variable(ds, "E_t", "uptake") == "uptake"
    with pytest.raises(KeyError):
        component_variable(ds, "E_t")
    with pytest.raises(ValueError, match="Temperature"):
        component_variable(ds.rename(air_temperature="T"), "E_t")
    ds = xr.Dataset({"E_t": component(), "T": component(units="mm day-1")})
    assert component_variable(ds, "E_t") == "E_t"


@pytest.mark.parametrize("module,builder,required", [
    ("run", "_build_run_args", {"precip": "p", "effective_precip": "p", "et": "et", "soil_dir": "soil", "output_dir": "out"}),
    ("calibrate", "_namespace_from_kwargs", {"effective_precip": "p", "et": "et", "soil_dir": "soil", "reference_ssm": "ssm", "output": "out"}),
    ("preprocess", "_build_args", {}),
])
def test_shared_api_alias_policy(module, builder, required):
    build = getattr(import_module("pysweb.swb." + module), builder)
    required = {**required, **({"transpiration": "t.nc"} if module != "preprocess" else {})}
    with pytest.warns(FutureWarning):
        args = build({**required, "t_var": "T"})
    assert args.transpiration_var == "T"
    with pytest.raises(ValueError, match="Conflicting arguments"):
        build({**required, "transpiration_var": "E_t", "t_var": "T"})
    with pytest.warns(FutureWarning):
        args = build({**required, "transpiration_var": "T", "t_var": "T"})
    assert args.transpiration_var == "T"
    assert "t_var" not in vars(args)


@pytest.mark.parametrize("old,new", [("t", "transpiration"), ("e_var", "soil_evaporation_var"), ("tc_var", "transpiration_fraction_var")])
def test_all_argument_alias_conflicts(old, new):
    with pytest.raises(ValueError, match="Conflicting arguments"):
        normalise_component_arguments({old: "old", new: "new"})


@pytest.mark.parametrize("module,required", [
    ("run", ["--precip", "p", "--effective-precip", "p", "--et", "et", "--transpiration", "t", "--output-dir", "out"]),
    ("calibrate", ["--effective-precip", "p", "--et", "et", "--transpiration", "t", "--soil-dir", "soil", "--reference-ssm", "ssm", "--output", "out"]),
    ("preprocess", ["--extent", "148", "-35.1", "148.1", "-35", "--sm-res", "0.1"]),
])
@pytest.mark.parametrize("reverse", [False, True])
def test_cli_alias_conflicts_in_either_order(module, required, reverse):
    parser = import_module("pysweb.swb." + module).build_parser()
    flags = [["--transpiration-var", "E_t"], ["--t-var", "T"]]
    if reverse:
        flags.reverse()
    with pytest.raises(SystemExit):
        parser.parse_args(required + [part for flag in flags for part in flag])
    with pytest.warns(FutureWarning):
        args = parser.parse_args(required + ["--transpiration-var", "E_t", "--t-var", "E_t"])
    assert args.transpiration_var == "E_t"
    assert "_component_arguments_seen" not in vars(args)


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("use_components", [False, True])
def test_preprocess_input_combinations_preserve_values_masks_and_dates(tmp_path, monkeypatch, legacy, use_components):
    module = import_module("pysweb.swb.preprocess")
    dates = pd.date_range("2024-01-01", periods=2)
    values = np.array([[[1.0, np.nan], [2.0, 3.0]], [[0.0, 1.0], [4.0, 2.0]]])
    da = xr.DataArray(values, dims=("time", "lat", "lon"), coords={"time": dates, "lat": [-35.0, -35.1], "lon": [148.0, 148.1]}, attrs={"units": "mm/day"})
    transpiration_name, evaporation_name = ("T", "E") if legacy else ("E_t", "E_s")
    ds = xr.Dataset({transpiration_name: da, evaporation_name: da * 0.5, "ET": da * 1.5})
    path = tmp_path / "components.nc"
    ds.to_netcdf(path)
    args = _build_args({"et_file": str(path), "et_var": "ET", "soil_evaporation_var": "E_s" if use_components else None, "dtype": "float64"})
    grid = _build_target_grid(_build_args({"extent": [148.0, -35.1, 148.1, -35.0], "sm_res": 0.1}))
    monkeypatch.setattr(module, "_reproject_to_template", lambda value, *a, **kw: value)
    result = module.process_et(args, grid, dates)
    assert set(result) == ({"et", "E_s", "E_t"} if use_components else {"et", "E_t"})
    xr.testing.assert_equal(result["E_t"], da.rename("E_t"))
    xr.testing.assert_equal(result["et"], (da * 1.5).rename("et"))
    if use_components:
        np.testing.assert_equal((result["E_s"] + result["E_t"]).values, result["et"].values)


@pytest.mark.parametrize("canonical,prefix,legacy", [("E_t", "transpiration", "t"), ("E_s", "soil_evaporation", "e")])
def test_prepared_path_fallback_requires_verified_manifest(tmp_path, canonical, prefix, legacy):
    tag = "20240101_20240102"
    old = tmp_path / f"{legacy}_daily_{tag}.nc"
    component().rename(legacy).to_netcdf(old)
    new = tmp_path / f"{prefix}_daily_{tag}.nc"
    with pytest.raises(ValueError, match="provenance"):
        resolve_prepared_component_path(new, canonical)
    manifest = tmp_path / f"preprocess_{tag}.manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "outputs": {old.name: {"sha256": sha256(old)}}}))
    with pytest.warns(FutureWarning):
        assert resolve_prepared_component_path(new, canonical) == str(old)
    assert not new.exists() and old.exists()
    old.write_bytes(old.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="provenance"):
        resolve_prepared_component_path(new, canonical)


def test_simulation_and_calibration_readers_share_alias_checks(tmp_path):
    from pysweb.swb.calibrate import _load_single_variable
    path = tmp_path / "input.nc"
    xr.Dataset({"T": component()}).to_netcdf(path)
    with pytest.warns(FutureWarning):
        run = load_forcing(path, "E_t", component="E_t")
    with pytest.warns(FutureWarning):
        calibration = _load_single_variable(path, "E_t", component="E_t")
    xr.testing.assert_identical(run, calibration)
    xr.Dataset({"T": component(), "E_t": component((9.0, np.nan))}).to_netcdf(path)
    for reader in (load_forcing, _load_single_variable):
        with pytest.raises(ValueError, match="Conflicting component"):
            reader(path, "E_t", component="E_t")


def test_public_run_accepts_canonical_input_guard(monkeypatch):
    api = import_module("pysweb.swb.api")
    calls = []
    monkeypatch.setattr(api, "run_swb_workflow", lambda **kwargs: calls.append(kwargs))
    api.run(transpiration="input.nc")
    assert calls == [{"transpiration": "input.nc"}]


def test_old_cached_output_schema_is_rejected(tmp_path):
    from pysweb.io.provenance import input_manifest, save_output_manifest, validate_existing_output
    path = tmp_path / "output.nc"
    component().rename("et").to_netcdf(path)
    manifest = input_manifest({}, [])
    assert manifest["component_schema"] == COMPONENT_SCHEMA
    old = {key: value for key, value in manifest.items() if key != "component_schema"}
    save_output_manifest(path, old)
    with pytest.raises(ValueError, match="component schema"):
        validate_existing_output(path, manifest)
