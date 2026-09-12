# Local soil workflows

`spec/` is maintained for users with local soil data. It contains thin compatibility entrypoints; all calculations live in `pysweb`.

- `3_sweb_mlcons_preprocess_inputs.py` forwards to `pysweb.swb.preprocess`, defaulting to `--soil-source mlcons`. Supply `--soil-mlcons-dir /path/to/data`. For measured hydraulic NetCDFs, pass `--soil-source custom --soil-file soil.nc`.
- `5_sweb_mlcons_run_model.py` forwards to the standard SWB run interface.
- `sweb_mlcons_runner.sh --config study.toml` uses the shared workflow. Set `[soil] source = "mlcons"` or `"custom"` in that file.

Use `--help` for complete arguments. Preprocessing also requires full-month rainfall, ET/T inputs, dates, extent and output path. To remain offline, skip the reference with `--skip-reference-ssm` or supply `--reference-file`.

See [the soil schema](../docs/soil-inputs.md), [custom configuration](../examples/custom-soil.toml) and [migration notes](../docs/migration.md). Historical duplicated code remains in Git history. New scientific changes belong in the package and need regression tests across soil backends.
