# Notebooks

- [01_run_pysweb.ipynb](01_run_pysweb.ipynb): runnable offline demo by default; optional regional stages use the same TOML configuration as the CLI. Imports and toggles precede execution for exported PBS scripts.
- [02_plot_heatmap.ipynb](02_plot_heatmap.ipynb): heatmap example; set paths to your prepared outputs.
- [03_plot_time_series.ipynb](03_plot_time_series.ipynb): time-series example; set paths to your prepared outputs.

Install PySWEB before opening notebooks. Plotting uses `pysweb.visualisation` and the `plot` extra; Jupyter/IPython are needed only for interactive notebooks. `workflows/6_plot_results.py` provides a script alternative. The run notebook can be exported to a script; diagnostic imports occur only when enabled. See [workflow guidance](../docs/workflows.md) for Earth Engine and offline PBS requirements.
