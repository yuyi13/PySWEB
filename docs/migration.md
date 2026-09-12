# Migration to the maintained package

## September 2026 baseline

GitHub and Gadi shared commit `e96879a`; the local checkout was four commits behind. Gadi additionally contained notebook/PBS improvements. Those two files were committed as `3fb0f23`, then GitHub, local and Gadi were fast-forwarded to that baseline before this update. This preserves the latest development rather than replacing it with an older checkout.

## Architecture and paths

| Previous surface | Current surface |
|---|---|
| Top-level `core.*` | `pysweb.io.gee_downloader`, `pysweb.met.*`, `pysweb.ssebop.*`, `pysweb.swb.solver` |
| Numbered workflows with implementation | Thin `workflows/` wrappers and installed `pysweb` commands |
| Duplicated `spec` preprocessing/solver | `pysweb.soil.mlcons` / `pysweb.soil.custom` and canonical SWB workflow |
| Shell-specific run settings and notebook copies | Shared TOML configuration and `pysweb.workflow` |
| `docs/superpowers` plans | Maintained methods, data, workflow and migration documentation |
| Root `SWEB_logo.png` | `docs/assets/pysweb-logo.png` |

The former top-level `core/` had already been removed from the latest baseline. This update closes the retirement plan through installed-wheel verification outside the checkout, canonical imports and direct solver regression tests. Package modules named `core.py` are maintained implementation modules within `pysweb`; they are unrelated to the retired top-level namespace. The small deprecated `pysweb.ssebop.inputs` compatibility namespace remains available.

Historical solver explanations already moved into `pysweb.swb.solver` remain there. This update retains the existing hydraulic equations, root distribution, diffusion limiter, ET floor relaxation, OpenLandMap mapping and local cold-temperature/FANO behavior. Current assumptions and unresolved scientific checks are summarized in `docs/methods.md`. Older implementation notes and code remain available through Git history at `3fb0f23` and its predecessors.

## Intentional behavior changes

- Shell launchers now require `--config study.toml`; old positional date/extent conventions and embedded paths are replaced by explicit configuration. The numbered Python CLIs remain available.
- `spec/3_sweb_mlcons_preprocess_inputs.py` defaults to the MLConstraints adapter and otherwise uses the canonical preprocess arguments. Set local source paths explicitly. `spec/5_sweb_mlcons_run_model.py` delegates to the package solver.
- Full calendar-month rainfall is required for monthly effective-precipitation calculations. Previously produced partial-month effective rain should be regenerated.
- Missing daily timestamps and mismatched grids raise errors. Present-but-missing forcing has explicit QC; zero filling remains an opt-in compatibility choice.
- The default start-of-day state convention is preserved. End-of-day reporting is optional, and final states plus water budgets are now retained.
- One-layer bottom drainage bookkeeping is corrected. Saturation overflow and floor adjustments are exposed; these changes can alter one-layer historical results.
- Calibration uses matched spatial support and geographic area weights, native resolution by default, a reproducible seed and explicit convergence checks. Old parameter estimates should be re-evaluated before reuse.
- User-provided soils are supported through one hydraulic contract. Historical MLConstraints pedotransfer conventions are preserved and identified in metadata.
- Old cached model outputs without a matching manifest are rejected by `skip_existing`. Rerun without that option to regenerate a product from verified inputs.

The update adds packaging, MIT licensing, CI and examples. It does not publish to PyPI or establish a new scientifically validated model release.
