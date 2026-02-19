#!/usr/bin/env python
"""
Extract and plot time series from SSEBop and SWEB NetCDF outputs.

Examples
--------
Domain-mean extraction from a shared run subdirectory:
    python visualisation/plot_time_series.py \
      --run-subdir Boonaldoon \
      --output /g/data/ym05/sweb_model/figures/Boonaldoon_timeseries.png \
      --csv-out /g/data/ym05/sweb_model/figures/Boonaldoon_timeseries.csv

Point extraction (nearest grid cell):
    python visualisation/plot_time_series.py \
      --ssebop-path /g/data/ym05/sweb_model/2_ssebop_outputs/Boonaldoon \
      --sweb-path /g/data/ym05/sweb_model/4_sweb_outputs/Boonaldoon \
      --lat -29.50 --lon 149.39 \
      --sweb-vars rzsm_layer_1 rzsm_layer_2 rzsm_layer_3 \
      --output ./timeseries_point.png
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

DEFAULT_SSEBOP_ROOT = Path("/g/data/ym05/sweb_model/2_ssebop_outputs")
DEFAULT_SWEB_ROOT = Path("/g/data/ym05/sweb_model/4_sweb_outputs")

SSEBOP_FILE_PATTERN = "et_daily_ssebop*.nc"
SWEB_FILE_PATTERN = "SWEB_RZSM*.nc"


@dataclass
class ExtractedSeries:
    data: pd.DataFrame
    variables: List[str]
    mode: str
    selected_lat: Optional[float] = None
    selected_lon: Optional[float] = None


def _parse_timestamp(text: str) -> pd.Timestamp:
    return pd.to_datetime(text, format="%Y-%m-%d")


def _filename_date_key(path: Path) -> Tuple[pd.Timestamp, pd.Timestamp]:
    match = re.search(r"_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.nc$", path.name)
    if not match:
        fallback = pd.Timestamp(1900, 1, 1)
        return fallback, fallback
    start, end = match.groups()
    return _parse_timestamp(start), _parse_timestamp(end)


def _pick_latest_file(candidates: Sequence[Path], prefer_post_burn: bool = False) -> Path:
    if not candidates:
        raise FileNotFoundError("No NetCDF files found.")
    filtered = list(candidates)
    if prefer_post_burn:
        post_burn = [path for path in filtered if "_Burn_In_" not in path.name]
        if post_burn:
            filtered = post_burn
    return max(filtered, key=lambda path: (_filename_date_key(path), path.stat().st_mtime))


def _resolve_product_path(
    explicit_path: Optional[str],
    root_dir: Path,
    run_subdir: Optional[str],
    pattern: str,
    prefer_post_burn: bool = False,
) -> Optional[Path]:
    if explicit_path is None and run_subdir is None:
        return None

    target = Path(explicit_path).expanduser() if explicit_path else (root_dir / run_subdir)
    if not target.exists():
        raise FileNotFoundError(f"Path does not exist: {target}")

    if target.is_file():
        if target.suffix.lower() != ".nc":
            raise ValueError(f"Expected NetCDF file (.nc), got: {target}")
        return target.resolve()

    candidates = sorted(path for path in target.glob(pattern) if path.is_file())
    if not candidates:
        raise FileNotFoundError(f"No files matching '{pattern}' in {target}")
    return _pick_latest_file(candidates, prefer_post_burn=prefer_post_burn).resolve()


def _detect_time_dim(da: xr.DataArray) -> str:
    for name in ("time", "date"):
        if name in da.dims:
            return name
    for dim in da.dims:
        coord = da.coords.get(dim)
        if coord is not None and pd.api.types.is_datetime64_any_dtype(coord.dtype):
            return dim
    raise ValueError(f"Could not detect a datetime dimension for variable '{da.name}'.")


def _detect_spatial_dims(da: xr.DataArray) -> Tuple[Optional[str], Optional[str]]:
    lon_dim = next((name for name in ("lon", "longitude", "x") if name in da.dims), None)
    lat_dim = next((name for name in ("lat", "latitude", "y") if name in da.dims), None)
    return lon_dim, lat_dim


def _format_var_label(name: str) -> str:
    if name in {"ET", "E", "T", "Tc"}:
        return name
    if name == "etf_interp":
        return "ETF"
    if name == "ndvi_interp":
        return "NDVI"
    if name == "profile_sm":
        return "Profile SM"
    if name.startswith("rzsm_layer_"):
        layer = name.rsplit("_", 1)[-1]
        return f"RZSM layer {layer}"
    return name.replace("_", " ")


def _default_variables(ds: xr.Dataset, product: str) -> List[str]:
    available = list(ds.data_vars)
    if product == "ssebop":
        preferred = [name for name in ("ET", "E", "T") if name in ds.data_vars]
        return preferred or available

    layer_vars = sorted(
        (name for name in available if name.startswith("rzsm_layer_")),
        key=lambda name: int(re.search(r"(\d+)$", name).group(1)) if re.search(r"(\d+)$", name) else 999,
    )
    if layer_vars:
        return layer_vars
    if "profile_sm" in ds.data_vars:
        return ["profile_sm"]
    return available


def _resolve_variables(ds: xr.Dataset, requested: Optional[Sequence[str]], product: str) -> List[str]:
    if not requested:
        return _default_variables(ds, product)

    missing = [name for name in requested if name not in ds.data_vars]
    if missing:
        available = ", ".join(ds.data_vars)
        raise ValueError(
            f"{product.upper()} variable(s) not found: {', '.join(missing)}. Available: {available}"
        )
    return list(requested)


def _extract_1d_series(
    da: xr.DataArray,
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
    lat: Optional[float],
    lon: Optional[float],
) -> Tuple[pd.Series, Optional[float], Optional[float]]:
    time_dim = _detect_time_dim(da)
    selected_lat = None
    selected_lon = None

    if lat is not None or lon is not None:
        if lat is None or lon is None:
            raise ValueError("Provide both --lat and --lon for point extraction.")
        lon_dim, lat_dim = _detect_spatial_dims(da)
        if lon_dim is None or lat_dim is None:
            raise ValueError(
                f"Variable '{da.name}' has no recognisable spatial dims for point extraction. Dims: {da.dims}"
            )
        da = da.sel({lon_dim: lon, lat_dim: lat}, method="nearest")
        selected_lon = float(da.coords[lon_dim].item())
        selected_lat = float(da.coords[lat_dim].item())

    reduce_dims = [dim for dim in da.dims if dim != time_dim]
    if reduce_dims:
        da = da.mean(dim=reduce_dims, skipna=True)

    da = da.squeeze(drop=True)
    if tuple(da.dims) != (time_dim,):
        raise ValueError(f"Variable '{da.name}' could not be reduced to a single time series. Dims: {da.dims}")

    series = da.to_series()
    try:
        series.index = pd.to_datetime(series.index)
    except Exception as exc:  # pragma: no cover - defensive parse fallback
        raise ValueError(f"Failed to parse datetime index for variable '{da.name}': {exc}") from exc

    series = series.sort_index()
    if start_date is not None:
        series = series[series.index >= start_date]
    if end_date is not None:
        series = series[series.index <= end_date]

    return series, selected_lat, selected_lon


def extract_product_series(
    nc_path: Path,
    product: str,
    requested_vars: Optional[Sequence[str]],
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
    lat: Optional[float],
    lon: Optional[float],
) -> ExtractedSeries:
    with xr.open_dataset(nc_path) as ds:
        variables = _resolve_variables(ds, requested_vars, product)
        series_dict = {}
        selected_lat = None
        selected_lon = None

        for var_name in variables:
            series, found_lat, found_lon = _extract_1d_series(ds[var_name], start_date, end_date, lat, lon)
            if found_lat is not None and found_lon is not None and selected_lat is None and selected_lon is None:
                selected_lat = found_lat
                selected_lon = found_lon
            series_dict[var_name] = series

    data = pd.DataFrame(series_dict).sort_index()
    mode = "point" if lat is not None and lon is not None else "domain_mean"
    return ExtractedSeries(
        data=data,
        variables=list(data.columns),
        mode=mode,
        selected_lat=selected_lat,
        selected_lon=selected_lon,
    )


def _plot_ssebop(ax: plt.Axes, data: pd.DataFrame):
    for column in data.columns:
        ax.plot(data.index, data[column], linewidth=1.8, alpha=0.9, label=_format_var_label(column))
    ax.set_title("SSEBop time series")
    ax.set_ylabel("Flux / index")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", ncol=2, fontsize=9)


def _plot_sweb(ax: plt.Axes, data: pd.DataFrame):
    rzsm_cols = [name for name in data.columns if name.startswith("rzsm_layer_")]
    other_cols = [name for name in data.columns if name not in rzsm_cols]
    profile_cols = [name for name in other_cols if name == "profile_sm"]
    non_profile_cols = [name for name in other_cols if name != "profile_sm"]

    for column in rzsm_cols + non_profile_cols:
        ax.plot(data.index, data[column], linewidth=1.8, alpha=0.9, label=_format_var_label(column))

    lines, labels = ax.get_legend_handles_labels()
    ax.set_title("SWEB time series")
    ax.set_ylabel("RZSM / soil moisture")
    ax.grid(alpha=0.3)

    if profile_cols:
        profile_ax = ax.twinx()
        for column in profile_cols:
            profile_ax.plot(
                data.index,
                data[column],
                linewidth=1.6,
                linestyle="--",
                alpha=0.9,
                color="#5E3C99",
                label=_format_var_label(column),
            )
        profile_ax.set_ylabel("Profile SM (mm)")
        lines2, labels2 = profile_ax.get_legend_handles_labels()
        lines += lines2
        labels += labels2

    if lines:
        ax.legend(lines, labels, loc="upper right", ncol=2, fontsize=9)


def plot_time_series(
    ssebop: Optional[ExtractedSeries],
    sweb: Optional[ExtractedSeries],
    output_path: Path,
    title: Optional[str],
    figsize: Tuple[float, float],
    dpi: int,
    show: bool,
):
    panels = []
    if ssebop is not None and not ssebop.data.empty:
        panels.append(("ssebop", ssebop.data))
    if sweb is not None and not sweb.data.empty:
        panels.append(("sweb", sweb.data))

    if not panels:
        raise ValueError("No data available to plot after extraction/date filtering.")

    fig, axes = plt.subplots(len(panels), 1, figsize=figsize, sharex=True)
    if len(panels) == 1:
        axes = [axes]

    for ax, (kind, data) in zip(axes, panels):
        if kind == "ssebop":
            _plot_ssebop(ax, data)
        else:
            _plot_sweb(ax, data)

    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    for ax in axes:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
    axes[-1].set_xlabel("Date")

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Wrote figure: {output_path}")

    if show:
        plt.show()
    plt.close(fig)


def _describe_extraction(name: str, path: Path, extracted: Optional[ExtractedSeries]):
    if extracted is None:
        return
    print(f"{name}: {path}")
    print(f"  mode      : {extracted.mode}")
    if extracted.mode == "point":
        print(f"  nearest   : lat={extracted.selected_lat:.6f}, lon={extracted.selected_lon:.6f}")
    print(f"  variables : {', '.join(extracted.variables)}")
    if not extracted.data.empty:
        print(f"  time span : {extracted.data.index.min().date()} to {extracted.data.index.max().date()}")
        print(f"  n points  : {len(extracted.data)}")


def _build_title(args: argparse.Namespace) -> str:
    if args.title:
        return args.title
    base = "SSEBop and SWEB time series"
    if args.run_subdir:
        return f"{base} ({args.run_subdir})"
    return base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot time series from SSEBop and SWEB NetCDF outputs.")
    parser.add_argument("--run-subdir", help="Run subdirectory under default output roots.")

    parser.add_argument("--ssebop-path", help="SSEBop NetCDF file or run directory.")
    parser.add_argument("--sweb-path", help="SWEB NetCDF file or run directory.")
    parser.add_argument("--ssebop-root", default=str(DEFAULT_SSEBOP_ROOT), help="Default SSEBop root directory.")
    parser.add_argument("--sweb-root", default=str(DEFAULT_SWEB_ROOT), help="Default SWEB root directory.")

    parser.add_argument("--ssebop-vars", nargs="+", help="SSEBop variables to plot (default: ET E T).")
    parser.add_argument(
        "--sweb-vars",
        nargs="+",
        help="SWEB variables to plot (default: all rzsm_layer_* variables found).",
    )

    parser.add_argument("--lat", type=float, help="Latitude for nearest-cell extraction.")
    parser.add_argument("--lon", type=float, help="Longitude for nearest-cell extraction.")
    parser.add_argument("--start-date", type=_parse_timestamp, help="Start date filter (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=_parse_timestamp, help="End date filter (YYYY-MM-DD).")

    parser.add_argument("--output", default="timeseries_plot.png", help="Output figure path.")
    parser.add_argument("--csv-out", help="Optional CSV path for extracted series.")
    parser.add_argument("--title", help="Custom plot title.")
    parser.add_argument("--figsize", nargs=2, type=float, default=(14.0, 9.0), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    parser.add_argument("--show", action="store_true", help="Display plot interactively after saving.")

    args = parser.parse_args()
    if not args.run_subdir and not args.ssebop_path and not args.sweb_path:
        parser.error("Provide --run-subdir or at least one of --ssebop-path / --sweb-path.")
    if (args.lat is None) ^ (args.lon is None):
        parser.error("Provide both --lat and --lon for point extraction.")
    if args.start_date is not None and args.end_date is not None and args.end_date < args.start_date:
        parser.error("--end-date must be on or after --start-date.")
    return args


def main():
    args = parse_args()

    ssebop_path = _resolve_product_path(
        explicit_path=args.ssebop_path,
        root_dir=Path(args.ssebop_root).expanduser(),
        run_subdir=args.run_subdir,
        pattern=SSEBOP_FILE_PATTERN,
        prefer_post_burn=False,
    )
    sweb_path = _resolve_product_path(
        explicit_path=args.sweb_path,
        root_dir=Path(args.sweb_root).expanduser(),
        run_subdir=args.run_subdir,
        pattern=SWEB_FILE_PATTERN,
        prefer_post_burn=True,
    )

    ssebop_data = None
    sweb_data = None

    if ssebop_path is not None:
        ssebop_data = extract_product_series(
            nc_path=ssebop_path,
            product="ssebop",
            requested_vars=args.ssebop_vars,
            start_date=args.start_date,
            end_date=args.end_date,
            lat=args.lat,
            lon=args.lon,
        )
    if sweb_path is not None:
        sweb_data = extract_product_series(
            nc_path=sweb_path,
            product="sweb",
            requested_vars=args.sweb_vars,
            start_date=args.start_date,
            end_date=args.end_date,
            lat=args.lat,
            lon=args.lon,
        )

    if ssebop_data is None and sweb_data is None:
        raise ValueError("No SSEBop or SWEB input was resolved. Check paths and run-subdir.")

    _describe_extraction("SSEBop", ssebop_path, ssebop_data)
    _describe_extraction("SWEB", sweb_path, sweb_data)

    if args.csv_out:
        frames = []
        if ssebop_data is not None:
            frames.append(ssebop_data.data.add_prefix("ssebop_"))
        if sweb_data is not None:
            frames.append(sweb_data.data.add_prefix("sweb_"))
        if frames:
            combined = pd.concat(frames, axis=1).sort_index()
            csv_path = Path(args.csv_out).expanduser().resolve()
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            combined.to_csv(csv_path, index_label="time")
            print(f"Wrote CSV: {csv_path}")

    plot_time_series(
        ssebop=ssebop_data,
        sweb=sweb_data,
        output_path=Path(args.output).expanduser().resolve(),
        title=_build_title(args),
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        show=args.show,
    )


if __name__ == "__main__":
    main()
