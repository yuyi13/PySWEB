"""
Script: evaluation.py
Objective: Extract configured observation operators and score explicit paired support.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-17
Last updated: 2026-10-17
Inputs: Consolidated SWB products, fixed site/depth rules and observed daily series.
Outputs: Point series, cumulative-validity flags and annual/pooled paired metrics.
Usage: from pysweb.evaluation import extract_point, evaluate_series
Dependencies: numpy, pandas, pyproj, xarray, pysweb
"""
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer


def depth_weights(layer_bottoms_mm, interval_mm):
    bottoms = np.asarray(layer_bottoms_mm, dtype=float)
    top, bottom = map(float, interval_mm)
    if not len(bottoms) or np.any(np.diff(np.r_[0, bottoms]) <= 0) or not 0 <= top < bottom <= bottoms[-1]:
        raise ValueError("Observation depth interval must lie inside ordered model layers")
    return np.maximum(0, np.minimum(bottoms, bottom)-np.maximum(np.r_[0, bottoms[:-1]], top))


def metrics(model, observed):
    a, b = np.asarray(model, float), np.asarray(observed, float)
    if a.shape != b.shape:
        raise ValueError("Pair shapes differ")
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if not len(a):
        return dict(n=0, rmse=None, mae=None, bias=None, unbiased_rmse=None, pearson_r=None)
    error = a-b
    bias = float(error.mean())
    return dict(n=len(a), rmse=float(np.sqrt(np.mean(error**2))), mae=float(np.mean(np.abs(error))),
        bias=bias, unbiased_rmse=float(np.sqrt(np.mean((error-bias)**2))),
        pearson_r=float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 and np.std(a) > 0 and np.std(b) > 0 else None)


def extract_point(path, *, longitude, latitude, depth_interval_mm,
                  admitted_mask=None, scenario="calibrated", valid_threshold=1-1e-8):
    """Transform the fixed coordinate; retain start and endpoint-mean states.

    Validity starts at the first stored day. Products must retain initialization
    history or supply a verified initial-history validity field when cropped.
    """
    with xr.open_dataset(path, decode_coords="all") as ds:
        if ds.attrs.get("state_timing") != "start":
            raise ValueError("Endpoint operator requires start-of-day states")
        crs = ds.rio.crs
        if crs is None:
            raise ValueError("Point extraction requires explicit output CRS")
        xdim, ydim = ds.rio.x_dim, ds.rio.y_dim
        x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(longitude, latitude)
        left, bottom, right, top = ds.rio.bounds()
        if not left <= x <= right or not bottom <= y <= top:
            raise ValueError("Fixed observation coordinate lies outside the model grid")
        point = ds.sel({xdim: x, ydim: y}, method="nearest").load()
        dates = pd.DatetimeIndex(point.time.values)
        if not len(dates) or not dates.equals(pd.date_range(dates[0], dates[-1])):
            raise ValueError("Daily endpoint operator requires contiguous dates")
        if admitted_mask is not None:
            mask = admitted_mask.sel({xdim: point[xdim], ydim: point[ydim]})
            if not bool(mask):
                raise ValueError("Fixed point is outside the frozen development mask")
        start = point[f"rzsm_{scenario}"].transpose("time", "layer").values
        final = np.array([float(point[f"final_sm_layer_{i}_{scenario}"]) for i in range(1, start.shape[1]+1)])
        following = np.concatenate([start[1:], final[None]], axis=0)
        mean = (start+following)/2
        weights = depth_weights(point.layer_bottom.values, depth_interval_mm)
        table = pd.DataFrame(index=dates)
        for i in range(start.shape[1]):
            table[f"start_layer_{i+1}"] = start[:, i]
            table[f"dailymean_layer_{i+1}"] = mean[:, i]
        table["storage_start_mm"] = start @ weights
        table["storage_dailymean_mm"] = mean @ weights
        flag = point[f"valid_forcing_fraction_{scenario}"].values
        initial = bool(point.get(f"initial_history_valid_{scenario}", xr.DataArray(True)))
        table["continuous_trajectory_valid"] = np.logical_and.accumulate(np.isfinite(flag) & (flag >= valid_threshold)) & initial
        for name in ("actual_et", "storage_adjustment", "drainage", "overflow", "infiltration", "storage_change", "balance_residual"):
            table[name] = point[f"{name}_{scenario}"].values
        for name in ("precipitation", "effective_precipitation", "et_forcing", "transpiration_forcing", "soil_evaporation_forcing"):
            if name in point:
                table[name] = point[name].values
        record = dict(longitude=float(longitude), latitude=float(latitude), crs=str(crs),
            selected_x=float(point[xdim]), selected_y=float(point[ydim]),
            depth_interval_mm=list(depth_interval_mm), model_overlap_mm=weights.tolist(), scenario=scenario,
            state_operator="(start + next_start)/2; final state supplies last endpoint")
    table.index.name = "day_utc"
    return table, record


def aggregate_probe_storage(observations, *, columns, weights_mm, minimum_joint_samples,
                            expected_samples, qc_columns=None):
    """Aggregate fixed probes on joint support, without substitution or filling."""
    if len(columns) != len(weights_mm) or not 1 <= minimum_joint_samples <= expected_samples:
        raise ValueError("Invalid fixed observation operator")
    if not observations.index.is_unique:
        raise ValueError("Duplicate observation timestamps")
    values = observations[columns].astype(float)
    good = np.isfinite(values).all(axis=1)
    if qc_columns is not None:
        if len(qc_columns) != len(columns):
            raise ValueError("One QC column per probe required")
        good &= observations[qc_columns].eq(0).all(axis=1)
    storage = (values @ np.asarray(weights_mm)).where(good)
    count = storage.resample("D").count()
    return pd.DataFrame({"storage_mm": storage.resample("D").mean().where(
        (count >= minimum_joint_samples) & (count <= expected_samples)), "joint_samples": count})


def evaluate_series(models, observations, *, pairs, phases, scoring_starts,
                    regimes=None, validity_column="continuous_trajectory_valid"):
    """Own/common date scores, pooled and annual; no model selection or retuning.

    Model keys are arbitrary workflow/scenario labels. Site comparisons, observed
    columns, units, phases and frozen regime labels are supplied in configuration.
    """
    if not models or set(models) != set(scoring_starts):
        raise ValueError("Provide scoring start for every model")
    for frame in [observations, *models.values()]:
        if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
            raise ValueError("Evaluation dates must be unique and increasing")
    rows = []
    for comparison, definition in pairs.items():
        model_column, observed_column = definition["model"], definition["observed"]
        frames, supports = {}, {}
        for key, frame in models.items():
            joined = frame[[model_column, validity_column]].rename(columns={model_column: "model"}).join(
                observations[[observed_column]].rename(columns={observed_column: "observed"}), how="left")
            eligible = joined[validity_column].eq(True) & np.isfinite(joined.model) & np.isfinite(joined.observed)
            eligible &= joined.index >= pd.Timestamp(scoring_starts[key])
            frames[key], supports[key] = joined, eligible
        common = None
        for key, eligible in supports.items():
            days = eligible.index[eligible]
            common = days if common is None else common.intersection(days)
        for key, frame in frames.items():
            for support_name, supported in (("own", supports[key]), ("common", frame.index.isin(common))):
                for phase, limits in phases.items():
                    first, last = map(pd.Timestamp, limits)
                    selected = supported & (frame.index >= first) & (frame.index <= last)
                    groups = [("pooled", selected)] + [(str(year), selected & (frame.index.year == year))
                        for year in range(first.year, last.year+1)]
                    labels = regimes.reindex(frame.index).fillna("ambiguous") if regimes is not None else None
                    for period, period_mask in groups:
                        for regime in ["all"] + (sorted(set(labels)) if labels is not None else []):
                            use = period_mask if regime == "all" else period_mask & labels.eq(regime)
                            sample = frame.loc[use]
                            rows.append(dict(model=key, comparison=comparison, units=definition.get("units"),
                                support=support_name, phase=phase, period=period, regime=regime,
                                **metrics(sample.model, sample.observed)))
    return pd.DataFrame(rows)


def cell_day_objective(model, reference, mask, weights):
    """Independently reproduce the native calibration loss on fixed support."""
    model, reference, mask, weights = xr.align(model, reference, mask, weights, join="exact")
    support = np.isfinite(reference) & mask.astype(bool) & (weights > 0)
    if bool((support & ~np.isfinite(model)).any()):
        raise ValueError("Model missing on frozen observation support")
    n = int(support.sum())
    if not n:
        raise ValueError("No reference cell-days")
    weighted = weights.broadcast_like(reference).where(support, 0)
    loss = float(np.sqrt((((model-reference)**2).where(support, 0)*weighted).sum()/weighted.sum()))
    return dict(rmse=loss, n_cell_days=n, weight_sum=float(weighted.sum()), objective="area-weighted cell-day")


def validate_simulation(path, mask, *, development_end, closure_tolerance=1e-4,
                        profile_tolerance=1e-3, bound_tolerance=1e-6):
    """Check every admitted development history and exported scenario budgets."""
    reports = {}
    with xr.open_dataset(path, decode_coords='all') as ds:
        spatial = mask.dims
        xr.align(ds.precipitation.isel(time=0,drop=True),mask,join='exact')
        thickness = np.diff(np.r_[0, ds.layer_bottom.values])
        for scenario in ds.attrs['scenarios'].split(','):
            continuous = mask.values.astype(bool).copy()
            development = continuous.copy()
            if f'initial_history_valid_{scenario}' in ds:
                initial = ds[f'initial_history_valid_{scenario}'].values.astype(bool)
                continuous &= initial
                development &= initial
            maximum_closure, maximum_profile, violations, valid_pairs = 0., 0., 0, 0
            for begin in range(0,ds.sizes['time'],32):
                block=ds.isel(time=slice(begin,begin+32))
                flag=block[f'valid_forcing_fraction_{scenario}'].transpose('time',*spatial).values
                valid=np.isfinite(flag) & (flag>=1-1e-8) & mask.values[None]
                continuous &= valid.all(axis=0)
                selected=block.time.values<=np.datetime64(development_end)
                if selected.any():
                    development &= valid[selected].all(axis=0)
                states=block[f'rzsm_{scenario}'].transpose('time','layer',*spatial).values
                upper=ds.soil_porosity.transpose('layer',*spatial).values*float(ds.attrs[f'sm_max_factor_{scenario}'])
                lower=ds.soil_wilting_point.transpose('layer',*spatial).values.copy()
                lower[:2] *= float(ds.attrs[f'sm_min_factor_{scenario}'])
                bad=~np.isfinite(states) | (states>upper[None]+bound_tolerance) | (states<lower[None]-bound_tolerance)
                violations += int((bad & valid[:,None]).sum())
                profile=np.sum(states*thickness[None,:,None,None],axis=1)
                exported=block[f'profile_sm_{scenario}'].transpose('time',*spatial).values
                residual=(block[f'storage_change_{scenario}']-block[f'infiltration_{scenario}']+
                    block[f'actual_et_{scenario}']+block[f'drainage_{scenario}']+
                    block[f'overflow_{scenario}']-block[f'storage_adjustment_{scenario}']).transpose('time',*spatial).values
                valid_pairs += int(valid.sum())
                if valid.any():
                    if not np.isfinite(residual[valid]).all() or not np.isfinite(exported[valid]).all():
                        violations += 1
                    else:
                        maximum_closure=max(maximum_closure,float(np.abs(residual[valid]).max()))
                        maximum_profile=max(maximum_profile,float(np.abs(profile[valid]-exported[valid]).max()))
            passed=(valid_pairs>0 and not violations and int(development.sum())==int(mask.sum()) and
                maximum_closure<=closure_tolerance and maximum_profile<=profile_tolerance)
            reports[scenario]=dict(passed=bool(passed),valid_cell_days=valid_pairs,
                complete_development_cells=int(development.sum()),expected_development_cells=int(mask.sum()),
                complete_trajectory_cells=int(continuous.sum()),state_or_finite_violations=violations,
                maximum_exported_closure_mm=maximum_closure,maximum_profile_reconstruction_mm=maximum_profile)
    return dict(passed=all(r['passed'] for r in reports.values()),scenarios=reports,
        interpretation='Arithmetic closure includes numerical floor additions; it does not establish precipitation-only physical closure')
