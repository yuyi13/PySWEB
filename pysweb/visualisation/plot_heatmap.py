#!/usr/bin/env python3
"""
Script: plot_heatmap.py
Objective: Plot aligned daily water fluxes, layer moisture and soil water storage for a point or domain mean.
Author: Yi Yu (with assistance from Codex)
Created: 2026-02-20
Last updated: 2026-10-26
Inputs: SWEB/SSEBop NetCDF paths (or run_subdir), variable selections, optional lat/lon and date filters.
Outputs: Saved heatmap figure and optional CSV of extracted plotting data.
Usage: pysweb heatmap --help
Dependencies: matplotlib, numpy, pandas, xarray; pysweb
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Optional, Sequence

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

try:
    import xarray as xr
except ModuleNotFoundError:
    xr = None

try:
    from matplotlib.colors import LinearSegmentedColormap
except ModuleNotFoundError:
    LinearSegmentedColormap = None

from pysweb.visualisation.plot_time_series import (
    DEFAULT_SSEBOP_ROOT,
    DEFAULT_SWEB_ROOT,
    SSEBOP_FILE_PATTERN,
    SWEB_FILE_PATTERN,
    _format_var_label,
    _configure_plot_font,
    _parse_timestamp,
    _resolve_product_path,
    extract_product_series,
)

from pysweb.visualisation.heatmap_figure import plot_heatmap

_configure_plot_font(pyplot=plt)


def _require_numpy():
    if np is None:
        raise ModuleNotFoundError("plot_heatmap requires the optional dependency 'numpy'.")
    return np


def _require_xarray():
    if xr is None:
        raise ModuleNotFoundError("plot_heatmap requires the optional dependency 'xarray'.")
    return xr


def _require_matplotlib():
    if plt is None or LinearSegmentedColormap is None:
        raise ModuleNotFoundError("plot_heatmap requires the optional dependency 'matplotlib'.")
    return plt, LinearSegmentedColormap


def _layer_index(name: str) -> int:
    match = re.search(r"(\d+)$", name)
    return int(match.group(1)) if match else 999


def _validate_heatmap_vars(variables: Sequence[str]) -> List[str]:
    layer_vars = [name for name in variables if name.startswith("rzsm_layer_")]
    if not layer_vars:
        raise ValueError(
            "No SWEB layer variables available for heatmap. "
            "Provide --sweb-vars with rzsm_layer_* names."
        )
    return sorted(layer_vars, key=_layer_index)


def _to_float(value: object) -> Optional[float]:
    numpy = _require_numpy()

    try:
        casted = float(value)
    except (TypeError, ValueError):
        return None
    if not numpy.isfinite(casted):
        return None
    return casted


def _layer_layout_from_attrs(nc_path: Path, layer_vars: Sequence[str]) -> tuple[List[str], Optional[np.ndarray]]:
    numpy = _require_numpy()
    xarray = _require_xarray()

    with xarray.open_dataset(nc_path) as ds:
        bottoms = [_to_float(ds[var].attrs.get("depth_bottom_mm")) for var in layer_vars]
        thicknesses = [_to_float(ds[var].attrs.get("layer_thickness_mm")) for var in layer_vars]

    # Depth integration is only valid for a contiguous profile beginning at zero.
    indices = [_layer_index(name) for name in layer_vars]
    if indices != list(range(1, len(layer_vars) + 1)):
        return list(layer_vars), None
    if all(value is not None and value > 0 for value in thicknesses):
        layer_thickness_mm = numpy.asarray(thicknesses, dtype=float)
    elif all(value is not None for value in bottoms):
        layer_thickness_mm = numpy.diff(numpy.array([0.0] + bottoms, dtype=float))
        if numpy.any(layer_thickness_mm <= 0):
            raise ValueError("Layer bottom depths must increase from zero.")
    else:
        return list(layer_vars), None
    y_edges = numpy.concatenate(([0.0], numpy.cumsum(layer_thickness_mm)))
    if all(value is not None for value in bottoms) and not numpy.allclose(y_edges[1:], bottoms):
        raise ValueError("Layer thickness and bottom-depth metadata disagree.")
    labels = [f"{a:g}–{b:g}" for a, b in zip(y_edges[:-1], y_edges[1:])]
    return labels, y_edges


def _get_custom_colormap(cmap_name: str):
    pyplot, color_map_factory = _require_matplotlib()

    custom_cmaps = {
        "nasa_soil": [
            "#8B4513",
            "#D2691E",
            "#DEB887",
            "#F0E68C",
            "#90EE90",
            "#20B2AA",
            "#4169E1",
            "#0000FF",
        ],
        "esa_soil": [
            "#A0522D",
            "#CD853F",
            "#F4A460",
            "#F5DEB3",
            "#98FB98",
            "#87CEEB",
            "#4682B4",
            "#191970",
        ],
        "fao_soil": [
            "#B8860B",
            "#DAA520",
            "#F0E68C",
            "#ADFF2F",
            "#32CD32",
            "#00CED1",
            "#1E90FF",
            "#0000CD",
        ],
        "nasa_precip": [
            "#FFFFFF",
            "#E6F3FF",
            "#87CEEB",
            "#4169E1",
            "#0000FF",
            "#0000CD",
            "#4B0082",
            "#8A2BE2",
        ],
        "esa_precip": [
            "#F0F8FF",
            "#B0E0E6",
            "#87CEEB",
            "#4682B4",
            "#1E90FF",
            "#0000FF",
            "#0000CD",
            "#191970",
        ],
        "fao_precip": [
            "#FFFFFF",
            "#E0F6FF",
            "#87CEEB",
            "#20B2AA",
            "#1E90FF",
            "#0000FF",
            "#00008B",
            "#4B0082",
        ],
        "custom_sm": [
            "#FFFFFF",
            "#CD853F",
            "#FFA500",
            "#FFFF00",
            "#228B22",
            "#00BFFF",
            "#000080",
            "#000000",
        ],
    }

    if cmap_name == "custom_precip":
        red_values = [255, 245, 180, 120, 20, 0, 0, 0, 255, 255, 255, 255, 255, 200, 120, 0]
        green_values = [255, 245, 180, 120, 20, 216, 150, 102, 255, 200, 150, 100, 0, 0, 0, 0]
        blue_values = [255, 255, 255, 255, 255, 195, 144, 102, 0, 0, 0, 0, 0, 0, 0, 0]
        bom_colors = []
        for idx in range(len(red_values)):
            bom_colors.append((red_values[idx] / 255.0, green_values[idx] / 255.0, blue_values[idx] / 255.0))
        return color_map_factory.from_list("custom_precip", bom_colors, N=256)

    if cmap_name in custom_cmaps:
        return color_map_factory.from_list(cmap_name, custom_cmaps[cmap_name], N=256)

    try:
        return pyplot.get_cmap(cmap_name)
    except ValueError:
        print(f"Warning: Colormap '{cmap_name}' not found, using default custom_sm.")
        return color_map_factory.from_list("custom_sm", custom_cmaps["custom_sm"], N=256)


def _read_units(nc_path: Path, variable: str) -> str:
    xarray = _require_xarray()

    with xarray.open_dataset(nc_path) as ds:
        return str(ds[variable].attrs.get("units", "")).strip()


def _apply_time_ticks(ax: plt.Axes, index: pd.DatetimeIndex, center_offset: float = 0.0):
    numpy = _require_numpy()

    n_points = len(index)
    if n_points == 0:
        return

    max_ticks = 10
    if n_points <= max_ticks:
        positions = numpy.arange(n_points)
    else:
        positions = numpy.linspace(0, n_points - 1, max_ticks, dtype=int)
    positions = numpy.unique(positions)

    label_fmt = "%Y-%m-%d" if n_points <= 120 else "%b %Y"
    labels = [index[pos].strftime(label_fmt) for pos in positions]
    ax.set_xticks(positions + center_offset)
    ax.set_xticklabels(labels, rotation=45, ha="right")


def _resolve_optional_ssebop(args: argparse.Namespace) -> Optional[Path]:
    if args.ssebop_path is not None:
        return _resolve_product_path(
            explicit_path=args.ssebop_path,
            root_dir=Path(args.ssebop_root).expanduser(),
            run_subdir=None,
            pattern=SSEBOP_FILE_PATTERN,
            prefer_post_burn=False,
        )

    if args.run_subdir is None:
        return None

    try:
        return _resolve_product_path(
            explicit_path=None,
            root_dir=Path(args.ssebop_root).expanduser(),
            run_subdir=args.run_subdir,
            pattern=SSEBOP_FILE_PATTERN,
            prefer_post_burn=False,
        )
    except FileNotFoundError:
        print("Warning: No SSEBop file found for this run; plotting SWEB heatmap only.")
        return None


def _default_title(args: argparse.Namespace) -> str:
    if args.title:
        return args.title
    if args.run_subdir:
        return f"SWEB layer heatmap ({args.run_subdir})"
    return "SWEB layer heatmap"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot daily fluxes, soil moisture and soil water storage (point mode by default)."
    )
    parser.add_argument("--run-subdir", help="Run subdirectory under default output roots.")

    parser.add_argument("--sweb-path", help="SWEB NetCDF file or run directory.")
    parser.add_argument("--ssebop-path", help="SSEBop NetCDF file or run directory (optional).")
    parser.add_argument("--sweb-root", default=str(DEFAULT_SWEB_ROOT), help="Default SWEB root directory.")
    parser.add_argument("--ssebop-root", default=str(DEFAULT_SSEBOP_ROOT), help="Default SSEBop root directory.")

    parser.add_argument(
        "--sweb-vars",
        nargs="+",
        help="SWEB variables to read before heatmap filtering (default follows rzsm_layer_* outputs).",
    )
    parser.add_argument(
        "--ssebop-var",
        default="ET",
        help="SSEBop variable for the optional top panel (default: ET).",
    )

    parser.add_argument("--lat", type=float, help="Latitude for nearest-cell extraction.")
    parser.add_argument("--lon", type=float, help="Longitude for nearest-cell extraction.")
    parser.add_argument(
        "--domain-mean",
        action="store_true",
        help="Use domain-mean extraction (optional override for default point-based mode).",
    )
    parser.add_argument("--start-date", type=_parse_timestamp, help="Start date filter (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=_parse_timestamp, help="End date filter (YYYY-MM-DD).")

    parser.add_argument("--sample-step", type=int, default=1, help="Keep every Nth timestep for plotting.")
    parser.add_argument(
        "--cmap",
        default="esa_soil",
        help=(
            "Colormap for SWEB heatmap. Supports visualise.py names "
            "(nasa_soil, esa_soil, fao_soil, custom_sm, etc.) and matplotlib colormaps."
        ),
    )
    parser.add_argument("--vmin", type=float, default=0.0, help="Optional lower bound for heatmap colour scale.")
    parser.add_argument("--vmax", type=float, default=0.5, help="Optional upper bound for heatmap colour scale.")

    parser.add_argument("--no-fluxes", action="store_true", help="Omit available effective rainfall and SWEB ET strips.")
    parser.add_argument("--flux-cmap", default="YlGnBu", help="Shared daily-flux colour scheme.")
    parser.add_argument("--flux-vmax", type=float, help="Flux maximum in mm/day (default: data maximum rounded up).")
    parser.add_argument("--flux-gamma", type=float, default=0.5, help="Flux power scale; default 0.5 is square root, 1 is linear.")
    parser.add_argument("--annotations", help="Optional JSON with phases, events and reference_period; no events by default.")
    parser.add_argument("--output", default="sweb_heatmap.png", help="Output figure path.")
    parser.add_argument("--csv-out", help="Optional CSV output for extracted series.")
    parser.add_argument("--title", help="Custom plot title.")
    parser.add_argument("--figsize", nargs=2, type=float, default=(15.0, 8.6), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    parser.add_argument("--show", action="store_true", help="Display plot interactively after saving.")

    args = parser.parse_args(argv)

    if not args.run_subdir and not args.sweb_path:
        parser.error("Provide --run-subdir or --sweb-path.")
    if args.domain_mean:
        if args.lat is not None or args.lon is not None:
            parser.error("--domain-mean cannot be combined with --lat/--lon.")
    else:
        if args.lat is None or args.lon is None:
            parser.error(
                "Point mode is default: provide both --lat and --lon, "
                "or pass --domain-mean for domain-wide extraction."
            )
    if args.start_date is not None and args.end_date is not None and args.end_date < args.start_date:
        parser.error("--end-date must be on or after --start-date.")
    if args.sample_step < 1:
        parser.error("--sample-step must be >= 1.")
    return args


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    numpy = _require_numpy()
    pyplot, _ = _require_matplotlib()
    extract_lat = None if args.domain_mean else args.lat
    extract_lon = None if args.domain_mean else args.lon

    sweb_path = _resolve_product_path(
        explicit_path=args.sweb_path,
        root_dir=Path(args.sweb_root).expanduser(),
        run_subdir=args.run_subdir,
        pattern=SWEB_FILE_PATTERN,
        prefer_post_burn=True,
    )
    ssebop_path = _resolve_optional_ssebop(args)

    sweb = extract_product_series(
        nc_path=sweb_path,
        product="sweb",
        requested_vars=args.sweb_vars,
        start_date=args.start_date,
        end_date=args.end_date,
        lat=extract_lat,
        lon=extract_lon,
    )
    if sweb.data.empty:
        raise ValueError("No SWEB data available after extraction/date filtering.")

    layer_vars = _validate_heatmap_vars(sweb.variables)
    heatmap_df = sweb.data[layer_vars].copy()
    if args.sample_step > 1:
        heatmap_df = heatmap_df.iloc[:: args.sample_step]
    if heatmap_df.empty:
        raise ValueError("No SWEB data available after sample-step filtering.")

    ssebop_series = None
    ssebop_units = ""
    if ssebop_path is not None:
        ssebop = extract_product_series(
            nc_path=ssebop_path,
            product="ssebop",
            requested_vars=[args.ssebop_var],
            start_date=args.start_date,
            end_date=args.end_date,
            lat=extract_lat,
            lon=extract_lon,
        )
        if not ssebop.data.empty:
            column = ssebop.data.columns[0]
            ssebop_series = ssebop.data[column].reindex(heatmap_df.index)
            ssebop_units = _read_units(ssebop_path, column)

    _, y_edges = _layer_layout_from_attrs(sweb_path, layer_vars)
    fluxes = None
    with _require_xarray().open_dataset(sweb_path) as ds:
        state_timing = ds.attrs.get("state_timing")
        flux_names = []
        if not args.no_fluxes:
            for name in ("effective_precipitation", "actual_et"):
                if name not in ds:
                    continue
                units = str(ds[name].attrs.get("units", "")).strip().lower()
                daily_rate = units in {"mm/day", "mm day-1", "mm day^-1", "mm d-1"}
                daily_amount = units == "mm" and ds[name].attrs.get("time_support") == "interval starting at time, duration 1 day"
                if not (daily_rate or daily_amount):
                    raise ValueError(f"Cannot label {name} as mm/day: units/time support are not daily flux metadata.")
                flux_names.append(name)
    if flux_names:
        extracted = extract_product_series(
            nc_path=sweb_path, product="sweb", requested_vars=flux_names,
            start_date=args.start_date, end_date=args.end_date, lat=extract_lat, lon=extract_lon,
        )
        fluxes = extracted.data.reindex(heatmap_df.index).rename(
            columns={"effective_precipitation": "Effective rainfall", "actual_et": "SWEB ET"})
    annotations = {}
    if args.annotations:
        annotations = json.loads(Path(args.annotations).expanduser().read_text())
        allowed = {"phases", "events", "reference_period"}
        if not isinstance(annotations, dict) or set(annotations) - allowed:
            raise ValueError("Annotation JSON accepts only phases, events and reference_period.")
    timing_label = {"start": "Start-of-day states", "end": "End-of-day states"}.get(state_timing)
    fig, _, plotted = plot_heatmap(
        heatmap_df, None if y_edges is None else numpy.diff(y_edges), fluxes=fluxes,
        title=_default_title(args), state_timing=timing_label,
        cmap=args.cmap, soil_limits=(args.vmin, args.vmax), flux_cmap=args.flux_cmap,
        flux_limits=(0, args.flux_vmax), flux_gamma=args.flux_gamma,
        forcing=ssebop_series,
        forcing_label=f"SSEBop {_format_var_label(args.ssebop_var)}" + (f" ({ssebop_units})" if ssebop_units else ""),
        figsize=tuple(args.figsize), dpi=args.dpi, **annotations,
    )

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, facecolor="white")
    print(f"Wrote figure: {output_path}")

    if args.csv_out:
        export_df = plotted[layer_vars].add_prefix("sweb_")
        for name in plotted.columns.difference(layer_vars):
            if name != "ssebop_forcing":
                export_df[name] = plotted[name]
        if ssebop_series is not None:
            export_df.insert(0, f"ssebop_{args.ssebop_var}", ssebop_series.reindex(plotted.index).values)
        csv_path = Path(args.csv_out).expanduser().resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        export_df.to_csv(csv_path, index_label="time")
        print(f"Wrote CSV: {csv_path}")

    print(f"SWEB: {sweb_path}")
    print(f"  mode      : {sweb.mode}")
    print(f"  variables : {', '.join(layer_vars)}")
    print(f"  time span : {heatmap_df.index.min().date()} to {heatmap_df.index.max().date()}")
    print(f"  n points  : {len(heatmap_df)}")
    if ssebop_path is not None:
        print(f"SSEBop: {ssebop_path}")
        print(f"  variable  : {args.ssebop_var}")

    if args.show:
        pyplot.show()
    pyplot.close(fig)


if __name__ == "__main__":
    main()
