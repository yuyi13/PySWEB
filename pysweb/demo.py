"""
Script: demo.py
Objective: Run a small offline custom-soil demonstration with synthetic forcing.
Author: Yi Yu (with assistance from Codex)
Created: 2026-09-12
Last updated: 2026-09-12
Inputs: Output directory and worker count; all scientific inputs are synthetic.
Outputs: Synthetic input NetCDFs, aligned forcing, soil states and budget diagnostics.
Usage: pysweb demo
Dependencies: numpy, pandas, xarray, rioxarray, pysweb
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
from pysweb.swb.preprocess import _build_args, _build_target_grid, preprocess_inputs
from pysweb.swb.run import run_swb_workflow


def prepare_demo(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    extent = [148.0, -35.02, 148.02, -35.0]
    grid = _build_target_grid(_build_args({"extent": extent, "sm_res": 0.01}))
    coords = {
        "time": pd.date_range("2024-01-01", periods=31),
        "lat": grid.latitudes,
        "lon": grid.longitudes,
    }
    shape = (31, len(grid.latitudes), len(grid.longitudes))
    rainfall = np.zeros(shape)
    rainfall[[2, 8, 15, 23], :, :] = 12.0

    def daily(values, name):
        return xr.DataArray(
            values,
            dims=("time", "lat", "lon"),
            coords=coords,
            name=name,
            attrs={"units": "mm day-1"},
        ).rio.write_crs("EPSG:4326")

    daily(rainfall, "precipitation").to_dataset().to_netcdf(directory / "rain.nc")
    xr.Dataset(
        {"ET": daily(np.full(shape, 2.0), "ET"), "T": daily(np.full(shape, 1.3), "T")}
    ).to_netcdf(directory / "et.nc")
    soil = {}
    for name, value, unit in [
        ("porosity", 0.45, "m3 m-3"),
        ("wilting_point", 0.12, "m3 m-3"),
        ("available_water_capacity", 0.18, "m3 m-3"),
        ("b_coefficient", 4.0, "dimensionless"),
        ("conductivity_sat", 8.0, "mm day-1"),
    ]:
        soil[name] = xr.DataArray(
            np.full((2, shape[1], shape[2]), value),
            dims=("layer", "lat", "lon"),
            coords={"layer": [1, 2], "lat": grid.latitudes, "lon": grid.longitudes},
            attrs={
                "units": unit,
                "layer_bottoms_mm": [100.0, 400.0],
                "source": "synthetic demonstration",
            },
        ).rio.write_crs("EPSG:4326")
    xr.Dataset(soil).to_netcdf(directory / "custom_soil.nc")
    return extent


def run_demo(output_dir="examples/outputs/demo", workers=1):
    base = Path(output_dir).resolve()
    extent = prepare_demo(base / "source")
    inputs = base / "prepared"
    preprocess_inputs(
        start_date="2024-01-01",
        end_date="2024-01-14",
        extent=extent,
        sm_res=0.01,
        rain_file=str(base / "source/rain.nc"),
        rain_var="precipitation",
        et_file=str(base / "source/et.nc"),
        et_var="ET",
        t_var="T",
        soil_source="custom",
        soil_file=str(base / "source/custom_soil.nc"),
        skip_reference_ssm=True,
        output_dir=str(inputs),
        workers=workers,
    )
    tag = "20240101_20240114"
    result = run_swb_workflow(
        precip=str(inputs / f"rain_daily_{tag}.nc"),
        effective_precip=str(inputs / f"effective_precip_daily_{tag}.nc"),
        et=str(inputs / f"et_daily_{tag}.nc"),
        t=str(inputs / f"t_daily_{tag}.nc"),
        soil_dir=str(inputs),
        output_dir=str(base / "results"),
        start_date="2024-01-01",
        end_date="2024-01-14",
        workers=workers,
    )
    with xr.open_dataset(result) as ds:
        residual = float(np.nanmax(np.abs(ds.balance_residual)))
        if residual > 1e-6 or ds.sizes["time"] != 14:
            raise RuntimeError(
                f"Demo validation failed (water residual={residual} mm)."
            )
    print(f"Offline demo complete: {result}; maximum budget residual={residual:.3g} mm")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a synthetic offline PySWEB custom-soil demo."
    )
    parser.add_argument("--output-dir", default="examples/outputs/demo")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    run_demo(args.output_dir, args.workers)


if __name__ == "__main__":
    main()
