"""High-level package API for SSEBop workflows."""
from __future__ import annotations

from pysweb.met.era5land import download as era5land_download
from pysweb.met.era5land import stack as era5land_stack
from pysweb.ssebop.inputs import landsat


def prepare_inputs(
    *,
    date_range: str,
    extent: list[float],
    met_source: str,
    landsat_dir: str,
    met_raw_dir: str,
    met_stack_dir: str,
    dem: str,
    gee_config: str,
) -> None:
    start_date, end_date = landsat.parse_date_range(date_range)

    if met_source != "era5land":
        raise NotImplementedError(f"Unsupported met_source: {met_source}")

    landsat.prepare_landsat_inputs(
        date_range = date_range,
        extent = extent,
        gee_config = gee_config,
        out_dir = landsat_dir,
    )

    era5land_download.download_era5land_daily(
        start_date = start_date,
        end_date = end_date,
        extent = extent,
        output_dir = met_raw_dir,
    )
    era5land_stack.stack_era5land_daily_inputs(
        raw_dir = met_raw_dir,
        dem = dem,
        start_date = start_date,
        end_date = end_date,
        output_dir = met_stack_dir,
    )


def run(*args, **kwargs):
    raise NotImplementedError("pysweb.ssebop.run is not wired yet")
