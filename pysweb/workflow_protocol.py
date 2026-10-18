"""
Script: workflow_protocol.py
Objective: Connect configured support, initialization and evaluation gates to native runs.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-17
Last updated: 2026-10-18
Inputs: Native prepared directories, paired workflow paths and scientific protocol settings.
Outputs: Frozen support, explicit initialization files, convergence receipts and evaluation.
Usage: Internal orchestration for pysweb.workflow_native.
Dependencies: numpy, pandas, xarray, netCDF4, pysweb
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import xarray as xr
from pysweb.io.provenance import atomic_json, atomic_netcdf, input_manifest, save_output_manifest
from pysweb.swb.calibrate import SOIL_FILE_STEMS, _load_soil_arrays, _soil_valid_mask
from pysweb.swb.core import infer_layer_bottoms
from pysweb.forcing import complete_support, paired_support


FORCING_FILES = {"effective_precip": ("effective_precip", "effective_precipitation"),
    "et": ("et", "et"), "E_t": ("transpiration", "E_t"),
    "ndvi": ("ndvi", "ndvi"), "precipitation": ("rain", "precipitation"),
    "E_s": ("soil_evaporation", "E_s")}


def load_prepared(directory, start, end, *, stop=None):
    directory = Path(directory)
    tag = f'{start.replace("-", "")}_{end.replace("-", "")}'
    fields = {}
    for name, (stem, variable) in FORCING_FILES.items():
        with xr.open_dataset(directory/f"{stem}_daily_{tag}.nc", decode_coords="all") as ds:
            fields[name] = ds[variable].sel(time=slice(None, stop)).load()
    # Exact coordinate checks prevent xarray's union alignment from hiding errors.
    xr.align(*fields.values(), join="exact")
    return xr.Dataset(fields)


def write_support(directory, start, end, development_end):
    directory = Path(directory)
    forcing = load_prepared(directory, start, end, stop=development_end)
    soils = _load_soil_arrays(directory)
    template = next(iter(soils.values())).isel(layer=0, drop=True)
    mask = template.copy(data=_soil_valid_mask(soils))
    physical = ((soils['wilting_point'] >= 0) & (soils['available_water_capacity'] > 0)
        & (soils['wilting_point']+soils['available_water_capacity'] < soils['porosity'])
        & (soils['porosity'] < 1) & (soils['b_coefficient'] > 0)
        & (soils['conductivity_sat'] > 0)).all('layer')
    mask &= physical
    support = complete_support(forcing, mask, start, development_end,
                               variables=tuple(FORCING_FILES))
    support.attrs.update(development_start=start, development_end=development_end,
                         selection="complete development-only forcing and applicable soil")
    path = directory/"development_support.nc"
    atomic_netcdf(support.astype("uint8"), path)
    inputs = list(directory.glob("*_daily_*.nc"))+[directory/name for name in SOIL_FILE_STEMS.values()]
    save_output_manifest(path, input_manifest(dict(start=start, end=end, development_end=development_end), inputs))
    return path


def initialize_inputs(directory, start, end, scoring_start, development_end,
                      spinup=None, peer_directories=()):
    """Freeze paired support, mask soil and prepend development forcing cycles."""
    from pysweb.swb.spinup import prepend_cycles
    directory = Path(directory)
    masks = []
    for folder in [directory, *map(Path, peer_directories)]:
        with xr.open_dataset(folder/"development_support.nc", decode_coords="all") as ds:
            if ds.development_support.attrs.get("development_end") != development_end:
                raise ValueError("Paired support has a different development boundary")
            masks.append(ds.development_support.load().astype(bool))
    mask = paired_support(masks) if len(masks) > 1 else masks[0].rename("fit_mask")
    if not bool(mask.any()):
        raise ValueError("No complete development forcing cells")
    prepared = directory/"initialized"
    prepared.mkdir(exist_ok=True)
    atomic_netcdf(mask.astype("uint8"), prepared/"fit_mask.nc")
    for name, filename in SOIL_FILE_STEMS.items():
        with xr.open_dataset(directory/filename, decode_coords="all") as ds:
            xr.align(ds[name], mask, join="exact")
            atomic_netcdf(ds.where(mask), prepared/filename)
    actual_tag = f'{start.replace("-", "")}_{end.replace("-", "")}'
    simulation_start = start
    mapping = {}
    if spinup:
        simulation_start = str((pd.Timestamp(start)-pd.Timedelta(days=365*spinup["repeats"])).date())
    for name, (stem, variable) in FORCING_FILES.items():
        source = directory/f"{stem}_daily_{actual_tag}.nc"
        destination = prepared/source.name
        if spinup:
            prepend_cycles(source, destination, start, end, source_year=spinup["source_year"],
                repeats=spinup["repeats"], development_end=development_end)
        else:
            destination = source
        mapping[name] = (str(destination), variable)
    reference = directory/f"reference_ssm_daily_{actual_tag}.nc"
    if reference.exists():
        destination = prepared/reference.name
        if spinup:
            prepend_cycles(reference, destination, start, end, source_year=spinup["source_year"],
                repeats=spinup["repeats"], development_end=development_end,
                reference=True, scoring_start=scoring_start)
        else:
            destination = reference
        mapping["reference_ssm"] = (str(destination), "reference_ssm")
    record = dict(actual_start=start, simulation_start=simulation_start, end=end,
        scoring_start=scoring_start, development_end=development_end,
        spinup_days=(pd.Timestamp(scoring_start)-pd.Timestamp(simulation_start)).days,
        spinup=spinup, cells=int(mask.sum()), peers=list(map(str, peer_directories)), files=mapping)
    sources = [directory/f'{stem}_daily_{actual_tag}.nc' for stem, _ in FORCING_FILES.values()]
    sources += [directory/name for name in SOIL_FILE_STEMS.values()]
    sources += [Path(folder)/'development_support.nc' for folder in [directory, *peer_directories]]
    if reference.exists():
        sources.append(reference)
    record['source_identity'] = input_manifest(dict(start=start, end=end, scoring_start=scoring_start,
        development_end=development_end, spinup=spinup), sources)
    atomic_json(prepared/"initialization.json", record)
    return record, mask


def initialized_forcing(record, directory, *, workers, state_timing, projected=False):
    keys = {"effective_precip": "effective_precip", "et": "et", "E_t": "transpiration", "ndvi": "ndvi"}
    result = dict(soil_dir=str(Path(directory)/"initialized"),
        date_range=[record["simulation_start"], record["end"]], workers=workers,
        state_timing=state_timing, lat_dim="y" if projected else "lat", lon_dim="x" if projected else "lon")
    for name, argument in keys.items():
        result[argument], result[argument+"_var"] = record["files"][name]
    return result


def convergence_gate(directory, record, mask, parameters, config, label, model_options=None, workers=1):
    from pysweb.swb.spinup import verify_convergence
    directory = Path(directory)
    stop = pd.Timestamp(record["scoring_start"])-pd.Timedelta(days=1)
    fields = {}
    for name in ("effective_precip", "et", "E_t", "ndvi"):
        path, variable = record["files"][name]
        with xr.open_dataset(path) as ds:
            # The NDVI cap is based on the whole development record, matching calibration.
            fields[name] = ds[variable].sel(time=slice(None, record["development_end"] if name == "ndvi" else stop)).load()
    arrays = _load_soil_arrays(directory/"initialized")
    soil = xr.Dataset(arrays).assign_coords(layer_depth=("layer", infer_layer_bottoms(arrays, None)))
    # Dataset alignment expands the other histories to NDVI's calendar; verification
    # explicitly selects the initialization prefix again.
    gate, differences = verify_convergence(xr.Dataset(fields), soil, mask, parameters,
        scoring_start=record["scoring_start"], development_end=record["development_end"],
        tolerance=config["tolerance"], minimum_fraction=config["minimum_fraction"],
        solver_backend=config.get("solver_backend", "numba"), model_options=model_options, workers=workers)
    atomic_json(directory/"initialized"/f"{label}_convergence.json", gate)
    atomic_netcdf(differences, directory/"initialized"/f"{label}_convergence.nc")
    if not gate["passed"]:
        raise ValueError("Wet/dry convergence failed; extend development-only cycles and refit")
    return gate


def trim_initialization(source, destination, actual_start):
    """Retain cumulative prefix validity when removing artificial output dates."""
    import netCDF4 as nc
    from pysweb.io.time_reindex import reindex_netcdf_time
    with xr.open_dataset(source) as ds:
        dates = pd.DatetimeIndex(ds.time.values)
        indices = np.flatnonzero(dates >= pd.Timestamp(actual_start))
        flags = [name for name in ds.data_vars if name.startswith("valid_forcing_fraction_")]
        history = {}
        for name in flags:
            valid = np.ones(ds[name].shape[1:], dtype=bool)
            for begin in range(0, int(indices[0]), 32):
                block = ds[name].isel(time=slice(begin, min(begin+32, int(indices[0])))).values
                valid &= (np.isfinite(block) & (block >= 1-1e-8)).all(axis=0)
            history[name.replace("valid_forcing_fraction_", "initial_history_valid_")] = (ds[name].dims[1:], valid)
    reindex_netcdf_time(source, destination, dates[indices], indices)
    with nc.Dataset(destination, "a") as ds:
        for name, (dims, values) in history.items():
            ds.createVariable(name, "u1", dims, zlib=True)[:] = values.astype("uint8")
    save_output_manifest(destination, input_manifest(dict(actual_start=actual_start), [source]))
    return Path(destination)


def evaluate_product(product, config, destination, mask=None):
    from pysweb.evaluation import extract_point, evaluate_series
    destination = Path(destination)
    destination.mkdir(exist_ok=True, parents=True)
    with xr.open_dataset(product) as ds:
        scenarios = ds.attrs["scenarios"].split(",")
    models = {}
    for scenario in scenarios:
        table, record = extract_point(product, longitude=config["longitude"], latitude=config["latitude"],
            depth_interval_mm=config["depth_interval_mm"], admitted_mask=mask, scenario=scenario)
        table.to_csv(destination/f"{scenario}_point.csv")
        atomic_json(destination/f"{scenario}_operator.json", record)
        models[scenario] = table
    if config.get("observations_file"):
        observations = pd.read_csv(config["observations_file"], index_col=0, parse_dates=True)
        regimes = None
        if config.get("regimes_file"):
            regimes = pd.read_csv(config["regimes_file"], index_col=0, parse_dates=True)[config["regime_column"]]
        table = evaluate_series(models, observations, pairs=config["pairs"], phases=config["phases"],
            scoring_starts={key: config["scoring_start"] for key in models}, regimes=regimes)
        table.to_csv(destination/"metrics.csv", index=False)


def default_parameters(model):
    from pysweb.swb.run import _parser_defaults
    defaults = _parser_defaults()
    return {k: float(model.get(k, defaults[k])) for k in ('diff_factor', 'root_beta')}


def calibration_identity(config, directory):
    """Bind acceptance to current development inputs, paired masks and settings."""
    directory = Path(directory)
    peers = [Path(p)/directory.name for p in config.get('forcing', {}).get('peer_runs', [])]
    paths = list(directory.glob('*_daily_*.nc'))
    paths += [directory/name for name in SOIL_FILE_STEMS.values()]
    paths += [folder/'development_support.nc' for folder in [directory, *peers]]
    settings = {key: value for key,value in config.items() if key != 'evaluation'}
    settings['run'] = {key:value for key,value in config['run'].items() if key != 'workers'}
    from pysweb.swb.formulation import settings_identity
    reference = config.get("model", {}).get("reference_file")
    settings["formulation"] = settings_identity(reference, **{
        k: v for k, v in config.get("model", {}).items()
        if k in ("max_root_depth", "drainage_slope", "drainage_upper_limit", "drainage_lower_limit")})
    if reference:
        paths.append(Path(reference))
    return input_manifest(settings, paths)


def reproduce_calibration(directory, record, mask, arguments, model, calibration_file, *, surface_depth=50.):
    """Export coarse states and independently reproduce the cell-day loss."""
    from pysweb.swb.paired import run_default_and_calibrated
    from pysweb.swb.calibrate import _surface_layer_index
    from pysweb.evaluation import cell_day_objective, validate_simulation
    from pysweb.contracts import geographic_weights
    from pysweb.io.provenance import sha256
    directory=Path(directory)
    product=run_default_and_calibrated(**arguments,**model,
        precip=record['files']['precipitation'][0],precip_var=record['files']['precipitation'][1],
        calibration_file=calibration_file,output_dir=directory/'verification',
        auxiliary_inputs={'soil_evaporation_forcing':record['files']['E_s']})
    numerical=validate_simulation(product,mask,development_end=record['development_end'])
    with xr.open_dataset(product,decode_coords='all') as ds, xr.open_dataset(record['files']['reference_ssm'][0],decode_coords='all') as reference:
        index=_surface_layer_index(ds.layer_bottom.values,surface_depth)
        simulated=ds.rzsm_calibrated.isel(layer=index).sel(time=slice(record['scoring_start'],record['development_end'])).load()
        observed=reference.reference_ssm.sel(time=simulated.time).load()
        weights=geographic_weights(observed,*mask.dims)
        result=cell_day_objective(simulated,observed,mask,weights)
    fitted=pd.read_csv(calibration_file).iloc[0]
    result.update(fitted_rmse=float(fitted.rmse),absolute_difference=abs(result['rmse']-float(fitted.rmse)),
        same_pair_count=result['n_cell_days']==int(fitted.n_obs),calibration_sha256=sha256(calibration_file),
        numerical=numerical,coarse_product=str(product),scoring_start=record['scoring_start'],
        scoring_end=record['development_end'])
    result['passed']=bool(result['absolute_difference']<1e-6 and result['same_pair_count'] and numerical['passed'])
    atomic_json(directory/'calibration-reproduction.json',result)
    if not result['passed']:
        raise ValueError('Exported coarse states do not reproduce the accepted calibration loss/support')
    return result
