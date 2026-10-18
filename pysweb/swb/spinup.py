"""
Script: spinup.py
Objective: Assemble development-only forcing cycles and verify initial-state convergence.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-17
Last updated: 2026-10-18
Inputs: Daily forcing calendars, development boundary, hydraulic properties and parameters.
Outputs: Explicit cycle mappings and wet/dry differences on fixed admitted cells.
Usage: from pysweb.swb.spinup import cycle_indices, verify_convergence
Dependencies: numpy, pandas, xarray, pysweb
"""
import numpy as np
import pandas as pd
import xarray as xr


def cycle_indices(dates, actual_start, repeats, source_year, development_end):
    """Repeat a complete non-leap development year ending in the correct phase."""
    dates = pd.DatetimeIndex(dates)
    if dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("Source dates must be unique and increasing")
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    expected = pd.date_range(f"{source_year}-01-01", f"{source_year}-12-31")
    if len(expected) != 365 or expected[-1] > pd.Timestamp(development_end):
        raise ValueError("Spin-up requires a complete non-leap development year")
    indices = dates.get_indexer(expected)
    if np.any(indices < 0):
        raise ValueError("Complete source-year forcing calendar required")
    start = pd.Timestamp(actual_start)
    if start > pd.Timestamp(development_end):
        raise ValueError("Actual record must begin in the development period")
    month_day = "03-01" if (start.month, start.day) == (2, 29) else start.strftime("%m-%d")
    offset = int(np.flatnonzero(expected.strftime("%m-%d") == month_day)[0])
    rotated = np.roll(indices, -offset)
    warm = np.tile(rotated, repeats)
    return warm, dict(source_year=int(source_year), repeats=repeats, days=len(warm),
        source_cycle_first=str(dates[rotated[0]].date()),
        source_cycle_last=str(dates[rotated[-1]].date()), actual_start=str(start.date()),
        development_end=str(pd.Timestamp(development_end).date()),
        limitation="Repeated-year numerical initialization does not establish true historical moisture")


def prepend_cycles(source, destination, actual_start, actual_end, *, source_year,
                   repeats, development_end, reference=False, scoring_start=None):
    """Stream a forcing or reference file, never copying reference into warm-up."""
    from pysweb.io.time_reindex import reindex_netcdf_time
    import json
    with xr.open_dataset(source) as ds:
        dates = pd.DatetimeIndex(ds.time.values)
    actual = pd.date_range(actual_start, actual_end)
    warm_dates = pd.date_range(end=actual[0]-pd.Timedelta(days=1), periods=365*repeats)
    combined = warm_dates.append(actual)
    if reference:
        mapping = dates.get_indexer(combined)
        mapping[(combined < pd.Timestamp(scoring_start or actual_start)) |
                (combined > pd.Timestamp(development_end))] = -1
        metadata = dict(reference_initialization="missing; no repeated observations")
    else:
        warm, metadata = cycle_indices(dates, actual_start, repeats, source_year, development_end)
        observed = dates.get_indexer(actual)
        if np.any(observed < 0):
            raise ValueError("Actual forcing calendar is incomplete")
        mapping = np.concatenate([warm, observed])
    reindex_netcdf_time(source, destination, combined, mapping, chunk_days=32,
                        attributes={"spinup_json": json.dumps(metadata), "actual_start": str(actual_start)})
    return metadata


def verify_convergence(forcing, soil, mask, parameters, *, scoring_start,
                       development_end, tolerance, minimum_fraction,
                       solver_backend="numba", model_options=None, workers=1,
                       mode="wet_dry", cycle_days=None):
    """Run wet and dry histories through the day before scoring on every cell.

    forcing has effective_precip, et and E_t, plus optional ndvi. Soil variables
    have layer followed by the same spatial axes. No missing admitted history is
    accepted, including days with no reference observation.
    """
    from pysweb.swb.calibrate import _extract_props_from_arrays
    from pysweb.swb.formulation import reference_properties, reject_legacy, settings_identity
    reject_legacy(parameters)
    from pysweb.swb.solver import _resolve_sm_bounds, soil_water_balance_1d
    if solver_backend == "numba":
        from pysweb.swb.accelerated import soil_water_balance_1d
    elif solver_backend != "reference":
        raise ValueError("Unknown solver backend")
    if not 0 < tolerance or not 0 < minimum_fraction <= 1:
        raise ValueError("Invalid convergence thresholds")
    if pd.Timestamp(scoring_start) > pd.Timestamp(development_end):
        raise ValueError("Convergence must be checked before the holdout")
    stop = pd.Timestamp(scoring_start)-pd.Timedelta(days=1)
    history = forcing.sel(time=slice(None, stop))
    times = pd.DatetimeIndex(history.time.values)
    if not len(times) or not times.equals(pd.date_range(times[0], stop)):
        raise ValueError("Complete daily initialization history required")
    if mode not in {"wet_dry", "field_capacity_cycles"}:
        raise ValueError("Unknown initialization convergence mode")
    if mode == "field_capacity_cycles" and (
            not isinstance(cycle_days, int) or isinstance(cycle_days, bool)
            or cycle_days < 365 or cycle_days % 365 or cycle_days > len(times)):
        raise ValueError("Field-capacity comparison requires complete repeated-year cycle_days")
    spatial = mask.dims
    if len(spatial) != 2 or not bool(mask.any()):
        raise ValueError("Nonempty two-dimensional admitted mask required")
    options = dict(drainage_slope=.5, drainage_upper_limit=25., drainage_lower_limit=0.,
                   use_ndvi_root_depth=False)
    options.update(model_options or {})
    if options.get("nan_to_zero"):
        raise ValueError("Initialization cannot replace missing forcing with zero")
    values = {k: history[k].transpose("time", *spatial).values for k in ("effective_precip", "et", "E_t")}
    if any(not np.isfinite(v[:, mask.values.astype(bool)]).all() for v in values.values()):
        raise ValueError("Admitted initialization cells have missing forcing")
    bottoms = np.asarray(soil["layer_depth"].values if "layer_depth" in soil.coords
                         else soil.porosity.attrs["layer_bottoms_mm"], dtype=float)
    soils = {k: soil[k].transpose("layer", *spatial).values for k in soil.data_vars}
    ndvi = None
    if options["use_ndvi_root_depth"]:
        ndvi = forcing.ndvi.sel(time=slice(None, development_end)).mean("time").transpose(*spatial).values
    difference = np.full((len(bottoms),)+mask.shape, np.nan)
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError("workers must be a positive integer")
    cells = list(zip(*np.where(mask.values)))
    effective_workers = min(workers, len(cells))
    context = (values, soils, bottoms, ndvi, options, parameters, times,
               solver_backend, reference_properties(options.get("reference_file")), mode, cycle_days)
    if effective_workers == 1:
        for cell in cells:
            y, x, value = _convergence_cell(cell, context)
            difference[:, y, x] = value
    else:
        # Linux fork shares already loaded, read-only histories across workers.
        # Workers return only five final differences, not complete trajectories.
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context
        with ProcessPoolExecutor(max_workers=effective_workers, mp_context=get_context("fork"),
                initializer=_initialize_convergence, initargs=(context,)) as pool:
            for y, x, value in pool.map(_convergence_cell, cells, chunksize=1):
                difference[:, y, x] = value
    admitted = difference[:, mask.values.astype(bool)]
    fraction = float(np.all(np.isfinite(admitted) & (admitted <= tolerance), axis=0).mean())
    record = dict(passed=fraction >= minimum_fraction, pass_fraction=fraction,
        tolerance_m3m3=float(tolerance), minimum_cell_fraction=float(minimum_fraction),
        cells=int(mask.sum()), initialization_days=len(times), scoring_start=str(scoring_start),
        max_by_layer=admitted.max(axis=1).tolist(), p99_by_layer=np.quantile(admitted, .99, axis=1).tolist(),
        workers=effective_workers,
        mode=mode, cycle_days=cycle_days,
        interpretation=("Empty versus saturated initial states" if mode == "wet_dry" else
            "Field-capacity starts with N versus 2N repeated years; does not establish independence from extreme initial states"),
        formulation=settings_identity(options.get("reference_file"),
            max_root_depth=options.get("max_root_depth", 1000.),
            **{k: options[k] for k in ("drainage_slope", "drainage_upper_limit", "drainage_lower_limit")}),
        parameters=dict(parameters), failure_action="Extend development-only initialization and refit")
    data = xr.DataArray(difference, dims=("layer",)+spatial,
        coords={"layer": soil.layer, **{d: mask[d] for d in spatial}}, name="initialization_difference")
    data.attrs["units"] = "m3 m-3"
    return record, data


_CONVERGENCE_CONTEXT = None

def _initialize_convergence(context):
    global _CONVERGENCE_CONTEXT
    _CONVERGENCE_CONTEXT = context


def _convergence_cell(cell, context=None):
    from pysweb.swb.calibrate import _extract_props_from_arrays
    from pysweb.swb.solver import _resolve_sm_bounds, soil_water_balance_1d
    context = context if context is not None else _CONVERGENCE_CONTEXT
    values, soils, bottoms, ndvi, options, parameters, times, backend, references, mode, cycle_days = context
    if backend == "numba":
        from pysweb.swb.accelerated import soil_water_balance_1d
    y, x = cell
    props = _extract_props_from_arrays(soils, bottoms, y, x, parameters["root_beta"],
        ndvi[y, x] if ndvi is not None else np.nan, options["drainage_slope"],
        options["drainage_upper_limit"], options["drainage_lower_limit"],
        options["use_ndvi_root_depth"])
    props.update(references)
    props["max_root_depth"] = options.get("max_root_depth", 1000.)
    final = []
    initial_states = _resolve_sm_bounds(props) if mode == "wet_dry" else (None, None)
    for index, initial in enumerate(initial_states):
        series = {key: values[key][:, y, x] for key in ("effective_precip", "et", "E_t")}
        dates = times
        if mode == "field_capacity_cycles" and index == 1:
            # Duplicate only the artificial prefix; append the real prefix once.
            series = {key: np.concatenate([value[:cycle_days], value]) for key, value in series.items()}
            dates = pd.date_range(end=times[-1], periods=len(times)+cycle_days)
        result = soil_water_balance_1d(series["effective_precip"],
            series["et"], props, dates, initial_soil_moisture=initial,
            diff_factor=parameters["diff_factor"], transpiration_data=series["E_t"],
            state_timing="start")
        if not np.all(result.attrs["forcing_valid"]):
            raise ValueError("Solver rejected initialization forcing")
        final.append(result.attrs["final_state"])
    return y, x, np.abs(final[0]-final[1])
