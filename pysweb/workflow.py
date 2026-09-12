"""
Script: workflow.py
Objective: Run explicitly configured stages with shared paths and model settings.
Author: Yi Yu (with assistance from Codex)
Created: 2026-09-12
Last updated: 2026-09-12
Inputs: Study TOML configuration, requested stages and optional dry-run flag.
Outputs: Resolved configuration and products from the selected model stages.
Usage: pysweb workflow --config study.toml --dry-run
Dependencies: argparse, pathlib, tomllib, pandas, pysweb
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
    unknown = set(config) - {"run", "soil", "calibration", "model", "ssebop"}
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    run = config.setdefault("run", {})
    required = {"name", "root", "start", "end", "extent"}
    if required - set(run):
        raise ValueError(f"Missing run configuration: {sorted(required - set(run))}")
    allowed = required | {"resolution", "workers", "gee_project", "state_timing"}
    if set(run) - allowed:
        raise ValueError(f"Unknown run settings: {sorted(set(run) - allowed)}")
    if Path(run["name"]).name != run["name"] or run["name"] in {".", ".."}:
        raise ValueError("run.name must be one directory name.")
    for key in ("start", "end"):
        run[key] = str(pd.Timestamp(run[key]).date())
    if run["end"] < run["start"]:
        raise ValueError("Run end precedes start.")
    from pysweb.ssebop.api import _validate_extent

    run["extent"] = _validate_extent(run["extent"])
    run.setdefault("workers", 1)
    run.setdefault("resolution", 0.00025)
    run.setdefault("state_timing", "start")
    if (
        run["workers"] < 1
        or run["resolution"] <= 0
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
    allowed_model = {
        "diff_factor",
        "sm_max_factor",
        "sm_min_factor",
        "root_beta",
        "drainage_slope",
        "drainage_upper_limit",
        "drainage_lower_limit",
        "use_ndvi_root_depth",
        "dtype",
        "skip_existing",
        "nan_to_zero",
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
        "sm_max_bounds",
        "sm_min_bounds",
        "beta_bounds",
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
        "landcover",
    }
    for section, allowed_keys in [
        ("model", allowed_model),
        ("calibration", allowed_calibration),
        ("ssebop", allowed_ssebop),
    ]:
        unknown = set(config.get(section, {})) - allowed_keys
        if unknown:
            raise ValueError(f"Unknown {section} settings: {sorted(unknown)}")
    return config


def run_config(config, stages=STAGES, *, dry_run=False):
    import pysweb

    config = copy.deepcopy(config)
    run = config["run"]
    soil = config.get("soil", {"source": "openlandmap"})
    calibration = config.get("calibration", {})
    enabled = bool(calibration.get("enabled", False))
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise ValueError(f"Unknown stages: {sorted(unknown)}")
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
            )
        if "ssebop" in stages:
            options = {
                "lst_band": "ST_B10",
                "red_band": "SR_B4",
                "nir_band": "SR_B5",
                "ndvi_band": "ndvi",
            }
            options.update(config.get("ssebop", {}))
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
                e_var="E",
                t_var="T",
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
            "t": str(swb_in / f"t_daily_{tag}.nc"),
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
        # The calibration and final simulation must use the same process switches.
        for key in (
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
        help="Comma-separated prepare,ssebop,preprocess,calibrate,run",
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
