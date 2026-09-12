# Contributing

Keep model implementation in `pysweb/`; workflow and `spec/` entrypoints should delegate to it. Use Python 3.12+ and install `python -m pip install -e ".[dev]"`.

```bash
python -m pytest -q
python -m build
pysweb demo --output-dir examples/outputs/demo
```

CI runs tests on Linux and macOS and installs the built wheel before running the demo outside the checkout. Add focused numerical tests for changes to units, timestamps, layer geometry, hydraulic equations, calibration masks or output identity. Explain intended numerical differences and scientific assumptions in the change description. Preserve source credits and script headers.

Update README examples, configuration and migration guidance together when interfaces change. Keep generated rasters, NetCDFs, notebooks with outputs, credentials, local environments and institution-specific paths out of Git. Use small synthetic fixtures unless redistribution is clearly permitted. Report the commit, configuration, input provenance and smallest reproducing example with an issue.
