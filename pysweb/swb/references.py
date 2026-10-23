"""
Script: references.py
Objective: Freeze native-cell GSSM quantiles against area means of native hydraulic fields on identical support.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-18
Last updated: 2026-10-23
Inputs: Geographic GSSM, frozen complete-cell mask and native layer-1 soil hydraulic rasters.
Outputs: Versioned reference artifact with per-cell statistics and exact spherical area integration.
Usage: from pysweb.swb.references import derive_reference
Dependencies: numpy, pandas, xarray, rioxarray; pysweb
"""
import numpy as np
import pandas as pd
import xarray as xr
from pysweb.io.grids import spatial
from pysweb.swb.formulation import (
    FORMULATION_VERSION, REFERENCE_VERSION, AVAILABLE_REFERENCE_VERSION, canonical_id)


def _rectangles(template):
    template = spatial(template)
    if str(template.rio.crs) != "EPSG:4326":
        raise ValueError("Exact spherical reference integration requires geographic native grids")
    rx, ry = template.rio.resolution()
    x, y = np.meshgrid(template.x.values, template.y.values)
    return x-abs(rx)/2, y-abs(ry)/2, x+abs(rx)/2, y+abs(ry)/2


def _area(left, bottom, right, top):
    return 6371008.8**2 * np.deg2rad(np.maximum(right-left, 0.)) * np.maximum(
        np.sin(np.deg2rad(top))-np.sin(np.deg2rad(bottom)), 0.)


def derive_reference(gssm, mask, hydraulics, *, site, sources, hydraulic_provenance,
                     coverage_policy=None):
    """Temporal linear p01/p99 in each cell, then spatial area means.

    Hydraulic raster values must already have been derived on the native soil
    grid, before any aggregation. Intersections integrate partial native soil
    pixels analytically on a sphere over the union of complete GSSM cells.
    An explicit coverage policy can exclude deficient cells, subject to a
    minimum retained spherical area. The default preserves strict support.
    """
    gssm = spatial(gssm).transpose("time", "y", "x")
    mask = spatial(mask).astype(bool)
    gssm, mask = xr.align(gssm, mask, join="exact")
    times = pd.DatetimeIndex(gssm.time.values)
    if times.has_duplicates or times.hasnans or not times.is_monotonic_increasing:
        raise ValueError("Invalid GSSM time coordinate")
    expected = pd.date_range("2011-01-01", "2020-12-31")
    if not times.isin(expected).all():
        raise ValueError("Reference climatology is fixed at 2011–2020")
    if gssm.attrs.get("units") not in ("m3 m-3", "m3/m3", "m^3 m^-3"):
        raise ValueError("GSSM must be scaled once to volumetric m3 m-3 before derivation")
    values = np.asarray(gssm.values[:, mask.values], dtype=float)
    invalid = np.isfinite(values) & ((values < 0) | (values > 1))
    if invalid.any():
        raise ValueError("Out-of-range GSSM values on frozen reference support")
    counts = np.isfinite(values).sum(axis=0)
    selection = None
    if coverage_policy is None:
        if not len(counts) or np.any(counts < .99*len(expected)):
            raise ValueError("Every frozen GSSM cell requires at least 99% climatology coverage")
    else:
        required = {"minimum_valid_fraction", "minimum_area_fraction"}
        if not isinstance(coverage_policy, dict) or set(coverage_policy) != required:
            raise ValueError("Coverage policy requires minimum_valid_fraction and minimum_area_fraction")
        temporal, spatial_fraction = (coverage_policy[k] for k in
                                      ("minimum_valid_fraction", "minimum_area_fraction"))
        if any(isinstance(v, (bool, str)) or not np.isscalar(v) for v in (temporal, spatial_fraction)):
            raise ValueError("Coverage fractions must be finite numbers")
        if not (np.isfinite(temporal) and 0 < temporal <= 1 and
                np.isfinite(spatial_fraction) and 0 <= spatial_fraction <= 1):
            raise ValueError("Invalid temporal or spatial coverage fraction")
        original_areas = _area(*_rectangles(mask))[mask.values]
        keep = counts >= temporal * len(expected)
        if not len(counts) or not keep.any():
            raise ValueError("No GSSM cells meet the temporal coverage requirement")
        retained_fraction = float(original_areas[keep].sum()/original_areas.sum())
        if retained_fraction + 1e-12 < spatial_fraction:
            raise ValueError(f"Retained GSSM area fraction {retained_fraction:.8f} is below "
                             f"minimum {spatial_fraction:.8f}")
        candidates = []
        for i, (row, col) in enumerate(zip(*np.where(mask.values))):
            candidates.append(dict(longitude=float(mask.x[col]), latitude=float(mask.y[row]),
                area_m2=float(original_areas[i]), valid_days=int(counts[i]), retained=bool(keep[i]),
                reason="retained" if keep[i] else "insufficient_temporal_coverage"))
        selection = dict(policy_version="available-native-cells-v1",
            minimum_valid_fraction=float(temporal), minimum_area_fraction=float(spatial_fraction),
            candidate_cells=candidates, candidate_cell_count=len(counts),
            retained_cell_count=int(keep.sum()), candidate_area_m2=float(original_areas.sum()),
            retained_area_m2=float(original_areas[keep].sum()), retained_area_fraction=retained_fraction,
            missing_values_filled=False, selection_basis="GSSM availability only; no in-situ or fit objective")
        selected_mask = mask.values.copy()
        selected_mask[mask.values] = keep
        mask = mask.copy(data=selected_mask)
        values, counts = values[:, keep], counts[keep]
    quantiles = np.nanquantile(values, [.01, .99], axis=0, method="linear")
    bounds = _rectangles(mask)
    areas = _area(*bounds)[mask.values]
    cells = []
    native = {k: spatial(v).transpose("y", "x") for k, v in hydraulics.items()}
    xr.align(*native.values(), join="exact")
    template = next(iter(native.values()))
    nleft, nbottom, nright, ntop = _rectangles(template)
    native_weights = np.zeros(template.shape)
    for i, (row, col) in enumerate(zip(*np.where(mask.values))):
        left, bottom, right, top = (a[row, col] for a in bounds)
        weights = _area(np.maximum(nleft, left), np.maximum(nbottom, bottom),
                        np.minimum(nright, right), np.minimum(ntop, top))
        if not np.isclose(weights.sum(), areas[i], rtol=1e-10, atol=1e-5):
            raise ValueError("Native hydraulic fields do not cover a complete GSSM cell")
        native_weights += weights
        cells.append(dict(longitude=float(mask.x[col]), latitude=float(mask.y[row]),
            area_m2=float(areas[i]), valid_days=int(counts[i]),
            p01=float(quantiles[0, i]), p99=float(quantiles[1, i])))
    admitted = native_weights > 0
    for key, data in native.items():
        if not np.isfinite(data.values[admitted]).all():
            raise ValueError(f"Missing native hydraulic {key} within frozen footprint")
    wp, por, fc = (native[k].values for k in ("wilting_point", "porosity", "field_capacity"))
    valid = (wp >= 0) & (wp <= fc) & (fc <= por) & (por <= 1) & (por > 0)
    valid &= (native["b_coefficient"].values > 0) & (native["conductivity_sat"].values >= 0)
    if not valid[admitted].all():
        raise ValueError("Invalid hydraulic fields within frozen footprint")
    means = {key: float(np.sum(data.values[admitted]*native_weights[admitted])/areas.sum())
             for key, data in native.items()}
    gd, gw = np.sum(quantiles*areas, axis=1)/areas.sum()
    if means["wilting_point"] <= 0:
        raise ValueError("Positive area-mean WP is required for the dry-reference ratio")
    all_dates = gssm.reindex(time=expected).values[:, mask.values]
    missing_by_cell = [[str(date.date()) for date in expected[~np.isfinite(all_dates[:, i])]]
                       for i in range(len(cells))]
    record = dict(schema_version=REFERENCE_VERSION, formulation_version=FORMULATION_VERSION,
        site=site, dry_reference_factor=float(min(1., gd/means["wilting_point"])),
        wet_reference_factor=float(max(1., gw/means["porosity"])),
        area_mean_gssm_p01=float(gd), area_mean_gssm_p99=float(gw),
        climatology=dict(start="2011-01-01", end="2020-12-31", expected_days=len(expected),
            quantiles=[.01, .99], quantile_method="linear", order="temporal quantiles per native cell, then area mean",
            missing_dates_by_cell=missing_by_cell, qc="finite volumetric values in [0,1]; no filling; >=99% days per cell",
            in_situ_used=False),
        spatial_support=dict(crs="EPSG:4326", cells=cells, area_m2=float(areas.sum()),
            weights="complete native-cell spherical areas; partial soil pixels integrated on the same sphere"),
        hydraulic_support=dict(layer_bottom_mm=50., means=means,
            native_pixels_intersecting=int(admitted.sum()), area_m2=float(native_weights.sum()),
            provenance=hydraulic_provenance, operation="hydraulics on native predictors before any averaging"),
        shallow_layers_mm=[50., 150.], wet_role="diagnostic only; physical saturation capacity unchanged",
        sources=sources)
    if selection is not None:
        record["schema_version"] = AVAILABLE_REFERENCE_VERSION
        record["support_selection"] = selection
        record["climatology"]["qc"] = (
            f"finite volumetric values in [0,1]; no filling; "
            f">={100*selection['minimum_valid_fraction']:g}% days per retained cell; "
            f">={100*selection['minimum_area_fraction']:g}% candidate area retained")
    record["artifact_id"] = canonical_id(record)
    return record


def frozen_mask(template, reference):
    """Locate exactly the recorded complete GSSM cells, rejecting missing coverage."""
    template = spatial(template)
    result = xr.zeros_like(template, dtype=bool)
    for cell in reference["spatial_support"]["cells"]:
        x = np.flatnonzero(np.isclose(template.x.values, cell["longitude"], rtol=0, atol=1e-10))
        y = np.flatnonzero(np.isclose(template.y.values, cell["latitude"], rtol=0, atol=1e-10))
        if len(x) != 1 or len(y) != 1:
            raise ValueError("Prepared grid does not contain the frozen reference cells")
        result.values[y[0], x[0]] = True
    return result


def freeze_prepared_support(directory, reference_file):
    """Keep the agreed reference population; fail if any member loses valid forcing."""
    from pathlib import Path
    from pysweb.swb.formulation import load_reference
    from pysweb.io.provenance import atomic_netcdf, input_manifest, save_output_manifest
    path = Path(directory)/"development_support.nc"
    with xr.open_dataset(path, decode_coords="all") as ds:
        existing = ds.development_support.load()
    reference = load_reference(reference_file)
    selected = frozen_mask(existing, reference)
    if np.any(selected.values & ~existing.values.astype(bool)):
        raise ValueError("A frozen GSSM reference cell has incomplete development forcing or invalid soil")
    result = existing.copy(data=selected.values.astype("uint8"))
    result.attrs.update(reference_artifact_id=reference["artifact_id"],
        selection="Frozen complete GSSM cells; no candidate-dependent population change")
    atomic_netcdf(result, path)
    save_output_manifest(path, input_manifest(dict(reference_artifact_id=reference["artifact_id"]), [reference_file]))
    return path
