"""
Script: workflow.py
Objective: Run explicitly configured stages with shared paths and model settings.
Author: Yi Yu (with assistance from Codex)
Created: 2026-09-12
Last updated: 2026-10-22
Inputs: Study TOML configuration, requested stages and optional dry-run flag.
Outputs: Resolved configuration and products from the selected model stages.
Usage: pysweb workflow --config examples/workflow.toml --dry-run
Dependencies: pandas; pysweb
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import tomllib
import pandas as pd
from pysweb.contracts import month_bounds
from pysweb.io.provenance import atomic_json

STAGES = ("prepare", "ssebop", "preprocess", "calibrate", "run")


def load_config(path):
    path = Path(path).expanduser().resolve()
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    unknown = set(config) - {"run", "soil", "calibration", "model", "ssebop", "landsat", "spinup", "forcing", "evaluation"}
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    run = config.setdefault("run", {})
    required = {"name", "root", "start", "end"}
    if required - set(run):
        raise ValueError(f"Missing run configuration: {sorted(required - set(run))}")
    allowed = required | {"extent", "center", "chip_size_m", "resolution", "workers", "gee_project", "state_timing", "grid_mode", "sensors", "processing_buffer_m", "holdout_start", "aoi_bounds", "aoi_crs"}
    if set(run) - allowed:
        raise ValueError(f"Unknown run settings: {sorted(set(run) - allowed)}")
    if Path(run["name"]).name != run["name"] or run["name"] in {".", ".."}:
        raise ValueError("run.name must be one directory name.")
    for key in ("start", "end"):
        run[key] = str(pd.Timestamp(run[key]).date())
    if run["end"] < run["start"]:
        raise ValueError("Run end precedes start.")
    from pysweb.ssebop.api import _validate_extent

    run.setdefault("workers", 1)
    run.setdefault("grid_mode", "legacy" if "resolution" in run else "native")
    if run["grid_mode"] not in {"native", "legacy"}:
        raise ValueError("run.grid_mode must be native or legacy")
    if run["grid_mode"] == "native" and "resolution" in run:
        raise ValueError("Native output uses Landsat's grid; remove run.resolution")
    if ("center" in run) == ("extent" in run):
        raise ValueError('Provide exactly one of run.center or run.extent')
    if "center" in run:
        if run['grid_mode'] != 'native' or 'aoi_bounds' in run or 'aoi_crs' in run:
            raise ValueError('run.center requires native mode without explicit aoi_bounds/aoi_crs')
        from pysweb.io.grids import chip_extent
        run.setdefault('chip_size_m', 5100.0)
        run['extent'] = chip_extent(run['center'], run['chip_size_m'])
        run['center'] = [float(value) for value in run['center']]
        run['chip_size_m'] = float(run['chip_size_m'])
    elif "chip_size_m" in run:
        raise ValueError('chip_size_m requires run.center')
    run["extent"] = _validate_extent(run["extent"])
    if ('aoi_bounds' in run) != ('aoi_crs' in run):
        raise ValueError('Provide run.aoi_bounds and run.aoi_crs together')
    if 'aoi_bounds' in run:
        import numpy as np
        from pyproj import CRS
        from rasterio.warp import transform_bounds
        bounds = np.asarray(run['aoi_bounds'],dtype=float)
        if run['grid_mode'] != 'native' or bounds.shape != (4,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
            raise ValueError('Native AOI requires four finite ordered bounds')
        run['aoi_crs'] = CRS.from_user_input(run['aoi_crs']).to_string()
        geographic = transform_bounds(run['aoi_crs'],'EPSG:4326',*bounds,densify_pts=65)
        envelope = run['extent']
        if geographic[0] < envelope[0]-1e-8 or geographic[1] < envelope[1]-1e-8 or geographic[2] > envelope[2]+1e-8 or geographic[3] > envelope[3]+1e-8:
            raise ValueError('run.extent must contain the explicitly referenced AOI')
        run['aoi_bounds'] = bounds.tolist()
    run.setdefault("sensors", ["LC08", "LC09"])
    if not run['sensors'] or len(set(run['sensors'])) != len(run['sensors']) or set(run['sensors']) - {'LC08','LC09','LE07','LT05'}:
        raise ValueError('Choose unique Landsat sensors from LC08, LC09, LE07, LT05')
    if run['grid_mode'] == 'legacy':
        run.setdefault("resolution", 0.00025)
    run.setdefault("state_timing", "start")
    if (
        run["workers"] < 1
        or run.get("resolution", 30) <= 0
        or run["state_timing"] not in {"start", "end"}
    ):
        raise ValueError("Invalid workers, resolution or state_timing.")
    run_root = Path(run["root"]).expanduser()
    run["root"] = str(
        run_root if run_root.is_absolute() else (path.parent / run_root).resolve()
    )
    soil = config.setdefault("soil", {"source": "openlandmap"})
    allowed_soil = {
        "source",
        "file",
        "input_dir",
        "mlcons_dir",
        "layer_bottoms_mm",
        "maximum_organic_matter_percent",
        "reference_file",
        "reference_var",
    }
    if set(soil) - allowed_soil:
        raise ValueError(f"Unknown soil settings: {sorted(set(soil) - allowed_soil)}")
    for key in ("file", "input_dir", "mlcons_dir", "reference_file"):
        if key in soil:
            value = Path(soil[key]).expanduser()
            soil[key] = str(
                value if value.is_absolute() else (path.parent / value).resolve()
            )
    from pysweb.soil.api import validate_soil_source

    validate_soil_source(soil.get("source", "openlandmap"))
    from pysweb.swb.formulation import reject_legacy
    reject_legacy(config.get("model", {}))
    reject_legacy(config.get("calibration", {}))
    if config.get("model", {}).get("reference_file"):
        reference = Path(config["model"]["reference_file"]).expanduser()
        config["model"]["reference_file"] = str(reference if reference.is_absolute() else (path.parent/reference).resolve())
    allowed_model = {
        "reference_file",
        "max_root_depth",
        "diff_factor",
        "root_beta",
        "drainage_slope",
        "drainage_upper_limit",
        "drainage_lower_limit",
        "use_ndvi_root_depth",
        "dtype",
        "skip_existing",
        "nan_to_zero",
        "solver_backend",
    }
    allowed_calibration = {
        "enabled",
        "start",
        "end",
        "seed",
        "max_iter",
        "spinup_days",
        "allow_unconverged",
        "surface_depth",
        "sm_res",
        "diff_bounds",
        "beta_bounds",
        "objective",
        "solver_backend",
        "shared_workers",
        "no_polish",
        "serial_polish",
    }
    from pysweb.ssebop.cli_run import build_parser

    allowed_ssebop = {a.dest for a in build_parser()._actions} - {
        "help",
        "config_pos",
        "config",
        "date_range",
        "silo_dir",
        "met_dir",
        "landsat_dir",
        "dem",
        "output_dir",
        "workers",
        "et_short_crop",
        "tmax",
        "tmin",
        "rs",
        "ea",
    }
    for section, allowed_keys in [
        ("model", allowed_model),
        ("calibration", allowed_calibration),
        ("ssebop", allowed_ssebop),
    ]:
        unknown = set(config.get(section, {})) - allowed_keys
        if unknown:
            raise ValueError(f"Unknown {section} settings: {sorted(unknown)}")
    for section, keys in {
        "landsat": {"manifest_file", "sensor_windows", "wrs", "gapfill"},
        "spinup": {"enabled", "source_year", "source_years", "repeats", "tolerance", "minimum_fraction",
                   "solver_backend", "convergence_mode", "convergence_action"},
        "forcing": {"peer_runs", "reconstruction", "precipitation_qc", "retrieval"},
        "evaluation": {"longitude", "latitude", "depth_interval_mm", "observations_file", "regimes_file",
                       "regime_column", "pairs", "phases", "scoring_start"},
    }.items():
        unknown = set(config.get(section, {})) - keys
        if unknown:
            raise ValueError(f"Unknown {section} settings: {sorted(unknown)}")
    for section, keys in {"landsat": ["manifest_file"], "evaluation": ["observations_file", "regimes_file"],
                          "ssebop": ["landcover"]}.items():
        for key in keys:
            if key in config.get(section, {}):
                value = Path(config[section][key]).expanduser()
                config[section][key] = str(value if value.is_absolute() else (path.parent/value).resolve())
    if "peer_runs" in config.get("forcing", {}):
        config["forcing"]["peer_runs"] = [str(Path(p) if Path(p).is_absolute() else (path.parent/p).resolve())
                                          for p in config["forcing"]["peer_runs"]]
    retrieval = config.get('forcing', {}).get('retrieval', {})
    for key in ('meteorology_raw_dir','meteorology_stack_dir','dem_file','scene_cache_dir','template_file','landcover_file'):
        if key in retrieval:
            value = Path(retrieval[key]).expanduser()
            retrieval[key] = str(value if value.is_absolute() else (path.parent/value).resolve())
    for key, raw in retrieval.get('meteorology_files', {}).items():
        value = Path(raw).expanduser()
        retrieval['meteorology_files'][key] = str(value if value.is_absolute() else (path.parent/value).resolve())
    from pysweb.swb.spinup import enabled_spinup
    if run['grid_mode'] == 'legacy' and (enabled_spinup(config.get('spinup'))
            or any(config.get(k) for k in ('landsat','forcing','evaluation'))):
        raise ValueError('Configured protocol integration requires grid_mode=native')
    if run['grid_mode'] == 'native':
        from pysweb.workflow_native import native_plan
        native_plan(config, STAGES)
    return config


def run_config(config, stages=STAGES, *, dry_run=False):
    import pysweb

    config = copy.deepcopy(config)
    run = config["run"]
    soil = config.get("soil", {"source": "openlandmap"})
    calibration = config.get("calibration", {})
    enabled = bool(calibration.get("enabled", False))
    unknown = set(stages) - (set(STAGES) | {"benchmark"})
    if unknown:
        raise ValueError(f"Unknown stages: {sorted(unknown)}")
    if run.get('grid_mode', 'legacy') == 'native':
        from pysweb.workflow_native import run_native
        return run_native(config, stages, dry_run=dry_run)
    if "benchmark" in stages:
        raise ValueError("The benchmark stage requires grid_mode=native")
    base = Path(run["root"]) / run["name"]
    prepared, ssebop_out, swb_in, swb_out = [
        base / p for p in ("inputs", "ssebop", "swb_inputs", "swb")
    ]
    windows = [(run["start"], run["end"])]
    calib_window = (
        str(pd.Timestamp(calibration.get("start", run["start"])).date()),
        str(pd.Timestamp(calibration.get("end", run["end"])).date()),
    )
    if calib_window[1] < calib_window[0]:
        raise ValueError("Calibration end precedes start.")
    if enabled and calib_window not in windows:
        windows.append(calib_window)
    calibration_file = base / "calibration.csv"
    plan = {
        "stages": list(stages),
        "run_directory": str(base),
        "windows": windows,
        "config": config,
    }
    if dry_run:
        return plan
    base.mkdir(parents=True, exist_ok=True)
    atomic_json(base / "resolved-config.json", plan)
    common = {"workers": run["workers"]}
    for start, end in windows:
        period = f"{start} to {end}"
        if "prepare" in stages:
            pysweb.ssebop.prepare_inputs(
                date_range=period,
                extent=run["extent"],
                met_source="era5land",
                gee_project=run.get("gee_project", ""),
                landsat_dir=str(prepared / "landsat"),
                met_raw_dir=str(prepared / "met/era5land/raw"),
                met_stack_dir=str(prepared / "met/era5land/stack"),
                dem_dir=str(prepared / "dem"),
                landcover_file=config.get('ssebop', {}).get('landcover'),
            )
        if "ssebop" in stages:
            options = {
                "lst_band": "ST_B10",
                "red_band": "SR_B4",
                "nir_band": "SR_B5",
                "ndvi_band": "ndvi",
            }
            options.update(config.get("ssebop", {}))
            if options.get('tcold_method', 'worldcover_fano') == 'worldcover_fano':
                options.setdefault('landcover', str(prepared/'dem/worldcover_v200_2021.tif'))
            pysweb.ssebop.run(
                date_range=period,
                landsat_dir=str(prepared / "landsat"),
                met_dir=str(prepared / "met/era5land/stack"),
                dem=str(prepared / "dem/nasadem.tif"),
                output_dir=str(ssebop_out),
                **common,
                **options,
            )
        if "preprocess" in stages:
            first, last = month_bounds(start, end)
            soil_args = {"soil_source": soil.get("source", "openlandmap")}
            for key in ("file", "input_dir", "mlcons_dir", "layer_bottoms_mm"):
                if key in soil:
                    soil_args["soil_" + key] = soil[key]
            for key in ("reference_file", "reference_var"):
                if key in soil:
                    soil_args[key] = soil[key]
            pysweb.swb.preprocess(
                start_date=start,
                end_date=end,
                extent=run["extent"],
                sm_res=run["resolution"],
                gee_project=run.get("gee_project"),
                rain_file=str(
                    prepared
                    / f"met/era5land/stack/precipitation_daily_{first:%Y-%m-%d}_{last:%Y-%m-%d}.nc"
                ),
                rain_var="precipitation",
                et_file=str(ssebop_out / f"et_daily_ssebop_{start}_{end}.nc"),
                et_var="ET",
                soil_evaporation_var="E_s",
                transpiration_var="E_t",
                output_dir=str(swb_in),
                skip_reference_ssm=not (enabled and (start, end) == calib_window),
                **common,
                **soil_args,
            )

    def forcing(window):
        tag = "_".join(x.replace("-", "") for x in window)
        return {
            "effective_precip": str(swb_in / f"effective_precip_daily_{tag}.nc"),
            "et": str(swb_in / f"et_daily_{tag}.nc"),
            "transpiration": str(swb_in / f"transpiration_daily_{tag}.nc"),
            "soil_dir": str(swb_in),
            "date_range": list(window),
            "state_timing": run["state_timing"],
            **common,
        }

    if "calibrate" in stages and enabled:
        options = {
            key: value
            for key, value in calibration.items()
            if key not in {"enabled", "start", "end"}
        }
        # Explicit legacy configurations retain their original fitting target.
        options.setdefault("objective", "domain_mean")
        # The calibration and final simulation must use the same process switches.
        for key in (
            "reference_file",
            "max_root_depth",
            "drainage_slope",
            "drainage_upper_limit",
            "drainage_lower_limit",
            "use_ndvi_root_depth",
        ):
            if key in config.get("model", {}):
                options[key] = config["model"][key]
        tag = "_".join(x.replace("-", "") for x in calib_window)
        if options.get("use_ndvi_root_depth"):
            options["ndvi"] = str(swb_in / f"ndvi_daily_{tag}.nc")
        pysweb.swb.calibrate(
            **forcing(calib_window),
            reference_ssm=str(swb_in / f"reference_ssm_daily_{tag}.nc"),
            output=str(calibration_file),
            **options,
        )
    if "run" in stages:
        options = dict(config.get("model", {}))
        if enabled:
            options["calibration_file"] = str(calibration_file)
        tag = "_".join(x.replace("-", "") for x in windows[0])
        if options.get("use_ndvi_root_depth"):
            options["ndvi"] = str(swb_in / f"ndvi_daily_{tag}.nc")
        pysweb.swb.run(
            **forcing(windows[0]),
            precip=str(swb_in / f"rain_daily_{tag}.nc"),
            output_dir=str(swb_out),
            **options,
        )
    return plan


def main(argv=None):
    import json

    parser = argparse.ArgumentParser(
        description="Execute configured PySWEB stages; paths resolve relative to the TOML file."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stages",
        default=",".join(STAGES),
        help="Comma-separated prepare,ssebop,preprocess,benchmark,calibrate,run; benchmark is optional",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and print the plan without creating outputs.",
    )
    args = parser.parse_args(argv)
    plan = run_config(
        load_config(args.config), args.stages.split(","), dry_run=args.dry_run
    )
    if args.dry_run:
        print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
