#!/usr/bin/env python3
"""
Script: plot_heatmap.py
Objective: Plot SWEB layer heatmaps with an optional SSEBop forcing panel for a point or domain mean.
Author: Yi Yu
Created: 2026-02-20
Last updated: 2026-02-25
Inputs: SWEB/SSEBop NetCDF paths (or run_subdir), variable selections, optional lat/lon and date filters.
Outputs: Saved heatmap figure and optional CSV of extracted plotting data.
Usage: python visualisation/plot_heatmap.py --help
Dependencies: argparse, matplotlib, numpy, pandas, xarray
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap

from plot_time_series import (
    DEFAULT_SSEBOP_ROOT,
    DEFAULT_SWEB_ROOT,
    SSEBOP_FILE_PATTERN,
    SWEB_FILE_PATTERN,
    _format_var_label,
    _parse_timestamp,
    _resolve_product_path,
    extract_product_series,
)


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
    try:
        casted = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(casted):
        return None
    return casted


def _layer_layout_from_attrs(nc_path: Path, layer_vars: Sequence[str]) -> tuple[List[str], np.ndarray]:
    with xr.open_dataset(nc_path) as ds:
        bottoms = [_to_float(ds[var].attrs.get("depth_bottom_mm")) for var in layer_vars]
        thicknesses = [_to_float(ds[var].attrs.get("layer_thickness_mm")) for var in layer_vars]

    if all(value is not None and value > 0 for value in thicknesses):
        layer_thickness_mm = np.asarray(thicknesses, dtype=float)
    elif all(value is not None for value in bottoms):
        derived = np.diff(np.array([0.0] + bottoms, dtype=float))
        if np.any(derived <= 0):
            layer_thickness_mm = np.ones(len(layer_vars), dtype=float)
        else:
            layer_thickness_mm = derived
    else:
        layer_thickness_mm = np.ones(len(layer_vars), dtype=float)

    y_edges = np.concatenate(([0.0], np.cumsum(layer_thickness_mm)))
    labels = [
        f"{int(round(y_edges[idx]))}-{int(round(y_edges[idx + 1]))} mm"
        for idx in range(len(layer_vars))
    ]
    return labels, y_edges


def _get_custom_colormap(cmap_name: str):
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
        return LinearSegmentedColormap.from_list("custom_precip", bom_colors, N=256)

    if cmap_name in custom_cmaps:
        return LinearSegmentedColormap.from_list(cmap_name, custom_cmaps[cmap_name], N=256)

    try:
        return plt.get_cmap(cmap_name)
    except ValueError:
        print(f"Warning: Colormap '{cmap_name}' not found, using default custom_sm.")
        return LinearSegmentedColormap.from_list("custom_sm", custom_cmaps["custom_sm"], N=256)


def _read_units(nc_path: Path, variable: str) -> str:
    with xr.open_dataset(nc_path) as ds:
        return str(ds[variable].attrs.get("units", "")).strip()


def _apply_time_ticks(ax: plt.Axes, index: pd.DatetimeIndex, center_offset: float = 0.0):
    n_points = len(index)
    if n_points == 0:
        return

    max_ticks = 10
    if n_points <= max_ticks:
        positions = np.arange(n_points)
    else:
        positions = np.linspace(0, n_points - 1, max_ticks, dtype=int)
    positions = np.unique(positions)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot SWEB layer heatmaps with optional SSEBop panel (point mode by default)."
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
        default="custom_sm",
        help=(
            "Colormap for SWEB heatmap. Supports visualise.py names "
            "(nasa_soil, esa_soil, fao_soil, custom_sm, etc.) and matplotlib colormaps."
        ),
    )
    parser.add_argument("--vmin", type=float, help="Optional lower bound for heatmap colour scale.")
    parser.add_argument("--vmax", type=float, help="Optional upper bound for heatmap colour scale.")

    parser.add_argument("--output", default="sweb_heatmap.png", help="Output figure path.")
    parser.add_argument("--csv-out", help="Optional CSV output for extracted series.")
    parser.add_argument("--title", help="Custom plot title.")
    parser.add_argument("--figsize", nargs=2, type=float, default=(14.0, 8.0), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    parser.add_argument("--show", action="store_true", help="Display plot interactively after saving.")

    args = parser.parse_args()

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


def main():
    args = parse_args()
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

    layer_labels, y_edges = _layer_layout_from_attrs(sweb_path, layer_vars)
    matrix = heatmap_df.to_numpy(dtype=float).T
    x_edges = np.arange(len(heatmap_df.index) + 1, dtype=float)
    x_centers = x_edges[:-1] + 0.5

    if ssebop_series is not None:
        fig, (ax_top, ax_heat) = plt.subplots(
            2,
            1,
            figsize=tuple(args.figsize),
            sharex=True,
            gridspec_kw={"height_ratios": [1.1, 3.2], "hspace": 0.18},
        )
        valid = ~ssebop_series.isna()
        if valid.any():
            values = ssebop_series.values.astype(float)
            ax_top.plot(
                x_centers,
                values,
                color="#2B5876",
                linewidth=1.8,
                label=_format_var_label(args.ssebop_var),
            )
            ax_top.fill_between(x_centers, values, 0.0, color="#6FA7C9", alpha=0.22)
        else:
            ax_top.text(
                0.02,
                0.86,
                "No SSEBop values in selected period.",
                transform=ax_top.transAxes,
                ha="left",
                va="top",
            )
        unit_text = f" ({ssebop_units})" if ssebop_units else ""
        ax_top.set_ylabel(f"{_format_var_label(args.ssebop_var)}{unit_text}")
        ax_top.set_title("SSEBop forcing panel")
        ax_top.grid(alpha=0.3)
    else:
        fig, ax_heat = plt.subplots(1, 1, figsize=tuple(args.figsize))
        ax_top = None

    heatmap = ax_heat.pcolormesh(
        x_edges,
        y_edges,
        matrix,
        cmap=_get_custom_colormap(args.cmap),
        shading="flat",
        vmin=args.vmin,
        vmax=args.vmax,
    )
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    ax_heat.set_yticks(y_centers)
    ax_heat.set_yticklabels(layer_labels)
    ax_heat.set_ylabel("SWEB soil layers (depth)")
    ax_heat.set_xlabel("Date")
    ax_heat.invert_yaxis()
    _apply_time_ticks(ax_heat, heatmap_df.index, center_offset=0.5)

    cbar = fig.colorbar(heatmap, ax=ax_heat, orientation="horizontal", shrink=0.8, pad=0.16)
    cbar.set_label("RZSM (m3 m-3)")

    stats = (
        f"Points: {matrix.shape[1]}\n"
        f"Layers: {matrix.shape[0]}\n"
        f"Range: {np.nanmin(matrix):.3f} to {np.nanmax(matrix):.3f}"
    )
    if args.sample_step > 1:
        stats += f"\nSample step: every {args.sample_step} points"
    if sweb.mode == "point":
        stats += f"\nNearest: lat={sweb.selected_lat:.6f}, lon={sweb.selected_lon:.6f}"
    else:
        stats += "\nMode: domain mean"
    ax_heat.text(
        0.01,
        0.99,
        stats,
        transform=ax_heat.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    fig.suptitle(_default_title(args), fontsize=13, fontweight="bold")
    if ax_top is not None:
        fig.subplots_adjust(top=0.90, bottom=0.12, hspace=0.20)
    else:
        fig.subplots_adjust(top=0.90, bottom=0.12)

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Wrote figure: {output_path}")

    if args.csv_out:
        export_df = heatmap_df.add_prefix("sweb_")
        if ssebop_series is not None:
            export_df.insert(0, f"ssebop_{args.ssebop_var}", ssebop_series.values)
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
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
