# Visualisation Post-Step Workflow Design

Date: 2026-05-03

## Goal

Make result visualisation feel like a post-processing step in the existing PySWEB workflow while keeping `pysweb.visualisation` as the canonical package implementation.

This refactor should reduce confusion between the top-level `visualisation/` wrappers and the installable package modules without breaking existing local usage immediately.

## Current Context

PySWEB currently has:

- Canonical plotting implementations under `pysweb/visualisation/`.
- Thin legacy wrappers under top-level `visualisation/`.
- Workflow scripts under `workflows/` for staged model execution.
- Notebook examples under `notebooks/` that already prefer package imports.

The model workflow is staged through input preparation, SSEBop execution, SWB preprocessing, calibration, and SWB execution. Plotting should become the next optional workflow step after model outputs exist.

## Approved Approach

Add a single post-step workflow script:

```bash
python workflows/6_plot_results.py heatmap ...
python workflows/6_plot_results.py time-series ...
```

The workflow script is only a dispatcher. It forwards subcommand arguments to the existing package CLIs:

```bash
python -m pysweb.visualisation.plot_heatmap
python -m pysweb.visualisation.plot_time_series
```

This avoids duplicating plotting options in `workflows/6_plot_results.py` and keeps parser behavior owned by the package modules.

## Package API Changes

Update the visualisation CLI modules so they can be reused programmatically by wrappers:

- `pysweb.visualisation.plot_heatmap.parse_args(argv=None)`
- `pysweb.visualisation.plot_heatmap.main(argv=None)`
- `pysweb.visualisation.plot_time_series.parse_args(argv=None)`
- `pysweb.visualisation.plot_time_series.main(argv=None)`

The default `argv=None` behavior should preserve the current command-line behavior.

## Compatibility

Keep the top-level `visualisation/plot_heatmap.py` and `visualisation/plot_time_series.py` wrappers for now as compatibility shims.

They should remain thin, import the package `main`, and be documented as deprecated. Documentation should point users to:

- `python workflows/6_plot_results.py ...` for workflow-oriented post-processing.
- `python -m pysweb.visualisation...` for package-oriented usage.

The top-level `visualisation/` directory should not be expanded with new logic.

## Documentation

Update documentation so plotting is described as optional Step 6/post-processing:

- README repository structure and workflow section.
- Notebook documentation that explains where the plotting notebooks fit.
- Any existing visualisation references that currently imply top-level `visualisation/` is the main entry point.

The docs should make clear that the package modules remain canonical.

## Testing

Add or update focused tests for:

- `python workflows/6_plot_results.py --help` exits successfully.
- `python workflows/6_plot_results.py heatmap --help` exits successfully.
- `python workflows/6_plot_results.py time-series --help` exits successfully.
- The workflow dispatcher forwards subcommand arguments to the correct package `main(argv)` function.
- Existing legacy visualisation wrappers still expose the package `main` and keep `--help` behavior working.

Existing plotting tests should continue to verify that notebooks and docs prefer `pysweb.visualisation` over direct top-level wrapper usage.

## Out Of Scope

This change will not:

- Delete the top-level `visualisation/` directory.
- Add plotting flags to the model runner shell scripts.
- Redesign the plotting products or change their output semantics.
- Add new package console scripts.

Those can be considered after the workflow wrapper is in place and the package layout is cleaned further.

