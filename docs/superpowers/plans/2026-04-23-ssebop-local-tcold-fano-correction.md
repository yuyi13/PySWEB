# PySWEB Local Tcold/FANO Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current same-pixel `Tcold` shortcut with a local multiscale FANO analog so PySWEB SSEBop regains thermal sensitivity while keeping the existing GEE-download plus local-processing workflow and full-scene outputs.

**Architecture:** Add a new local `tcold_fano_local_xr()` builder in `pysweb.ssebop.core` that reconstructs `Tcold` from a full-scene vegetated anchor pool, coarse aggregation, and a scene-level fallback. Then thread a typed `LocalFanoConfig` through the package run path and CLI surface, while keeping the existing `dT`, ETf, interpolation, and NetCDF output logic unchanged.

**Tech Stack:** Python 3.12, xarray, rioxarray, numpy, rasterio, pytest, stdlib `dataclasses`

---

All new Python files in this plan must use the standard script header from `/home/603/yy4778/.codex/docs/script_header_standard.md`.

## File Structure

### Create

- None

### Modify

- `pysweb/ssebop/core.py`
- `pysweb/ssebop/api.py`
- `core/ssebop_au.py`
- `workflows/2_ssebop_run_model.py`
- `tests/ssebop/test_core.py`
- `tests/ssebop/test_api_run.py`
- `tests/workflows/test_2_ssebop_run_model.py`

### Keep As-Is

- `pysweb/ssebop/landcover.py`
- `pysweb/ssebop/landsat.py`
- `workflows/1_ssebop_prepare_inputs.py`
- `workflows/ssebop_runner_landsat.sh`
- `tests/ssebop/test_api_prepare_inputs.py`

---

### Task 1: Add core regressions for thermal sensitivity and fallback behavior

**Files:**
- Modify: `tests/ssebop/test_core.py`
- Test: `tests/ssebop/test_core.py`

- [ ] **Step 1: Write the failing multiscale `Tcold` regression tests**

```python
# tests/ssebop/test_core.py
from pysweb.ssebop.core import (
    LocalFanoConfig,
    build_doy_climatology,
    compute_dt_daily,
    daily_et_from_etf,
    et_fraction_xr,
    tcold_fano_local_xr,
)


def _spatial_raster(values: np.ndarray, name: str) -> xr.DataArray:
    rows, cols = values.shape
    x = np.arange(cols, dtype=float) * 30.0 + 15.0
    y = np.arange(rows, dtype=float)[::-1] * 30.0 + 15.0
    data = xr.DataArray(
        values,
        coords = {"y": y, "x": x},
        dims = ("y", "x"),
        name = name,
    )
    data = data.rio.write_crs("EPSG:32755")
    data = data.rio.write_transform(
        Affine.translation(0.0, rows * 30.0) * Affine.scale(30.0, -30.0)
    )
    return data


def test_tcold_fano_local_xr_preserves_lst_sensitivity():
    config = LocalFanoConfig(
        anchor_ndvi_threshold = 0.4,
        fine_scale_m = 60.0,
        coarse_scale_m = 120.0,
        smooth_scale_m = 60.0,
    )
    lst = _spatial_raster(
        np.array(
            [
                [300.0, 302.0, 304.0, 306.0],
                [301.0, 303.0, 305.0, 307.0],
                [302.0, 304.0, 306.0, 308.0],
                [303.0, 305.0, 307.0, 309.0],
            ],
            dtype = float,
        ),
        "lst",
    )
    ndvi = _spatial_raster(np.full((4, 4), 0.6, dtype=float), "ndvi")
    dt = _spatial_raster(np.full((4, 4), 10.0, dtype=float), "dt")

    tcold = tcold_fano_local_xr(lst, ndvi, dt, config = config)
    etf = et_fraction_xr(lst, tcold, dt)
    collapsed = np.clip(1.25 * ndvi.values - 0.125, 0.0, 1.0)

    assert np.isfinite(tcold.values).all()
    assert etf.values[0, 0] > etf.values[-1, -1]
    assert not np.allclose(etf.values, collapsed)


def test_tcold_fano_local_xr_falls_back_without_anchor_pixels():
    config = LocalFanoConfig(
        anchor_ndvi_threshold = 0.4,
        fine_scale_m = 60.0,
        coarse_scale_m = 120.0,
        smooth_scale_m = 60.0,
    )
    lst = _spatial_raster(
        np.array(
            [
                [312.0, 313.0, 314.0, 315.0],
                [311.0, 312.0, 313.0, 314.0],
                [310.0, 311.0, 312.0, 313.0],
                [309.0, 310.0, 311.0, 312.0],
            ],
            dtype = float,
        ),
        "lst",
    )
    ndvi = _spatial_raster(np.full((4, 4), 0.2, dtype=float), "ndvi")
    dt = _spatial_raster(np.full((4, 4), 12.0, dtype=float), "dt")

    tcold = tcold_fano_local_xr(lst, ndvi, dt, config = config)
    etf = et_fraction_xr(lst, tcold, dt)

    assert tcold.shape == lst.shape
    assert etf.shape == lst.shape
    assert np.isfinite(tcold.values).all()
    assert np.isfinite(etf.values).all()
```

- [ ] **Step 2: Run the new core tests to verify the new builder does not exist yet**

Run:

```bash
cd /g/data/ym05/github/yuyi13/PySWEB && python -m pytest tests/ssebop/test_core.py -q
```

Expected: FAIL with `ImportError` because `LocalFanoConfig` and `tcold_fano_local_xr` do not exist yet.

- [ ] **Step 3: Keep the existing ETf clamp regression intact**

```python
# tests/ssebop/test_core.py
def test_et_fraction_xr_clamps_and_masks():
    lst = xr.DataArray(np.array([[305.0]], dtype=float), dims=("y", "x"))
    tcold = xr.DataArray(np.array([[300.0]], dtype=float), dims=("y", "x"))
    dt = xr.DataArray(np.array([[4.0]], dtype=float), dims=("y", "x"))

    result = et_fraction_xr(lst, tcold, dt, clamp_max = 1.0, mask_max = 2.0)

    np.testing.assert_allclose(result.values, [[0.0]])
```

- [ ] **Step 4: Re-run the narrowed failure to confirm the failure is still only the missing builder surface**

Run:

```bash
cd /g/data/ym05/github/yuyi13/PySWEB && python -m pytest tests/ssebop/test_core.py::test_tcold_fano_local_xr_preserves_lst_sensitivity tests/ssebop/test_core.py::test_tcold_fano_local_xr_falls_back_without_anchor_pixels -q
```

Expected: FAIL with `ImportError` for the missing core symbols.

- [ ] **Step 5: Commit the test-only regression scaffold**

```bash
cd /g/data/ym05/github/yuyi13/PySWEB && git add tests/ssebop/test_core.py && git commit -m "test: add local tcold regression coverage"
```

---

### Task 2: Implement the local multiscale `Tcold` builder and compatibility config

**Files:**
- Modify: `pysweb/ssebop/core.py`
- Modify: `core/ssebop_au.py`
- Test: `tests/ssebop/test_core.py`

- [ ] **Step 1: Extend the compatibility config contract before touching the core implementation**

```python
# core/ssebop_au.py
@dataclass
class SsebopAuConfig:
    """Configuration hints for AU SSEBop processing."""

    et_reference_type: str = "alfalfa"
    et_reference_unit: str = "mm/day"
    dt_coeff: float = 0.125
    high_ndvi_threshold: float = 0.9
    anchor_ndvi_threshold: float = 0.4
    fine_scale_m: float = 240.0
    coarse_scale_m: float = 4800.0
    smooth_scale_m: float = 240.0
    etf_clamp_max: float = 1.0
    etf_mask_max: float = 2.0
    worldcover_path: str = AU_SSEBOP_SOURCE_CANDIDATES["landcover"]["local"]

    @property
    def veg_ndvi_threshold(self) -> float:
        return self.anchor_ndvi_threshold
```

- [ ] **Step 2: Add the new typed config surface and helper exports**

```python
# pysweb/ssebop/core.py
from dataclasses import dataclass
from typing import Protocol

__all__ = [
    "LocalFanoConfig",
    "build_doy_climatology",
    "compute_dt_daily",
    "daily_et_from_etf",
    "dt_fao56_xr",
    "et_fraction_xr",
    "tcold_fano_local_xr",
    "tcold_fano_simple_xr",
]


class TcoldConfig(Protocol):
    """Structural config interface for local FANO tuning."""

    dt_coeff: float
    high_ndvi_threshold: float
    anchor_ndvi_threshold: float
    fine_scale_m: float
    coarse_scale_m: float
    smooth_scale_m: float


@dataclass(frozen = True)
class LocalFanoConfig:
    """Runtime configuration for the local multiscale FANO cold-boundary builder."""

    dt_coeff: float = 0.125
    high_ndvi_threshold: float = 0.9
    anchor_ndvi_threshold: float = 0.4
    fine_scale_m: float = 240.0
    coarse_scale_m: float = 4800.0
    smooth_scale_m: float = 240.0
```

- [ ] **Step 3: Replace the same-pixel builder with a multiscale local FANO analog**

```python
# pysweb/ssebop/core.py
def _config_value(config: TcoldConfig | None, name: str, default: float) -> float:
    if config is None:
        return default
    return float(getattr(config, name, default))


def _window_from_scale(data: xr.DataArray, scale_m: float) -> tuple[int, int]:
    if scale_m <= 0:
        raise ValueError("scale_m must be > 0")
    if data.rio.crs is None:
        raise ValueError("Input raster must have a CRS for local FANO window sizing")

    x_res, y_res = data.rio.resolution()
    x_window = max(1, int(round(scale_m / abs(float(x_res)))))
    y_window = max(1, int(round(scale_m / abs(float(y_res)))))
    return y_window, x_window


def _rolling_mean_2d(data: xr.DataArray, y_window: int, x_window: int) -> xr.DataArray:
    return data.rolling(y = y_window, x = x_window, center = True, min_periods = 1).mean()


def _coarsen_mean_2d(data: xr.DataArray, y_window: int, x_window: int) -> xr.DataArray:
    return data.coarsen(y = y_window, x = x_window, boundary = "pad").mean(skipna = True)


def _interp_like_2d(source: xr.DataArray, match: xr.DataArray) -> xr.DataArray:
    source_xy = source.sortby("y").sortby("x")
    target_y = np.sort(match["y"].values)
    target_x = np.sort(match["x"].values)
    interp = source_xy.interp(y = target_y, x = target_x, method = "linear")
    interp = interp.reindex(y = match["y"].values, x = match["x"].values)
    return interp.transpose(*match.dims)


def tcold_fano_local_xr(
    lst_k: xr.DataArray,
    ndvi: xr.DataArray,
    dt_k: xr.DataArray,
    config: TcoldConfig | None = None,
) -> xr.DataArray:
    """Build a local multiscale FANO-style cold boundary for full-scene ETf."""
    dt_coeff = _config_value(config, "dt_coeff", 0.125)
    high_ndvi_threshold = _config_value(config, "high_ndvi_threshold", 0.9)
    anchor_ndvi_threshold = _config_value(
        config,
        "anchor_ndvi_threshold",
        _config_value(config, "veg_ndvi_threshold", 0.4),
    )
    fine_scale_m = _config_value(config, "fine_scale_m", 240.0)
    coarse_scale_m = _config_value(config, "coarse_scale_m", 4800.0)
    smooth_scale_m = _config_value(config, "smooth_scale_m", fine_scale_m)

    valid = np.isfinite(lst_k) & np.isfinite(ndvi) & np.isfinite(dt_k)
    tc_fine = lst_k - (dt_coeff * dt_k * (high_ndvi_threshold - ndvi) * 10.0)

    fine_y, fine_x = _window_from_scale(lst_k, fine_scale_m)
    coarse_y, coarse_x = _window_from_scale(lst_k, coarse_scale_m)
    smooth_y, smooth_x = _window_from_scale(lst_k, smooth_scale_m)

    ndvi_support = _rolling_mean_2d(ndvi.where(valid), fine_y, fine_x)
    anchor_tc = tc_fine.where(valid & (ndvi_support >= anchor_ndvi_threshold))
    tc_coarse = _coarsen_mean_2d(anchor_tc, coarse_y, coarse_x)

    tc_scene = tc_fine.where(valid).mean()
    tc_coarse = tc_coarse.fillna(tc_scene)

    tcold = _interp_like_2d(tc_coarse, lst_k)
    tcold = _rolling_mean_2d(tcold, smooth_y, smooth_x)
    return tcold.where(valid).rename("tcold")


def tcold_fano_simple_xr(
    lst_k: xr.DataArray,
    ndvi: xr.DataArray,
    dt_k: xr.DataArray,
    config: TcoldConfig | None = None,
) -> xr.DataArray:
    """Compatibility wrapper over the multiscale local FANO cold-boundary builder."""
    return tcold_fano_local_xr(lst_k, ndvi, dt_k, config = config)
```

- [ ] **Step 4: Run the core regression suite**

Run:

```bash
cd /g/data/ym05/github/yuyi13/PySWEB && python -m pytest tests/ssebop/test_core.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the core implementation**

```bash
cd /g/data/ym05/github/yuyi13/PySWEB && git add pysweb/ssebop/core.py core/ssebop_au.py tests/ssebop/test_core.py && git commit -m "feat: add local multiscale tcold builder"
```

---

### Task 3: Thread the new `Tcold` builder and config knobs through the package run workflow

**Files:**
- Modify: `pysweb/ssebop/api.py`
- Modify: `workflows/2_ssebop_run_model.py`
- Modify: `tests/ssebop/test_api_run.py`
- Modify: `tests/workflows/test_2_ssebop_run_model.py`
- Test: `tests/ssebop/test_api_run.py`
- Test: `tests/workflows/test_2_ssebop_run_model.py`

- [ ] **Step 1: Write the failing forwarding and CLI tests**

```python
# tests/ssebop/test_api_run.py
def test_ssebop_run_forwards_tcold_fano_kwargs_intact(monkeypatch):
    recorded = {}

    def fake_run_ssebop_workflow(**kwargs):
        recorded.update(kwargs)

    monkeypatch.setattr(ssebop_api, "run_ssebop_workflow", fake_run_ssebop_workflow)

    run(
        date_range = "2024-01-01 to 2024-01-03",
        landsat_dir = "/tmp/landsat",
        dem = "/tmp/dem.tif",
        output_dir = "/tmp/out",
        et_short_crop = "/tmp/eto.nc",
        tmax = "/tmp/tmax.nc",
        tmin = "/tmp/tmin.nc",
        rs = "/tmp/rs.nc",
        ea = "/tmp/ea.nc",
        tcold_dt_coeff = 0.15,
        tcold_high_ndvi_threshold = 0.85,
        tcold_anchor_ndvi_threshold = 0.35,
        tcold_fine_scale_m = 240.0,
        tcold_coarse_scale_m = 4800.0,
        tcold_smooth_scale_m = 240.0,
        workers = 2,
    )

    assert recorded["tcold_dt_coeff"] == 0.15
    assert recorded["tcold_high_ndvi_threshold"] == 0.85
    assert recorded["tcold_anchor_ndvi_threshold"] == 0.35
    assert recorded["tcold_fine_scale_m"] == 240.0
    assert recorded["tcold_coarse_scale_m"] == 4800.0
    assert recorded["tcold_smooth_scale_m"] == 240.0
```

```python
# tests/workflows/test_2_ssebop_run_model.py
def test_workflow_help_includes_tcold_fano_options(monkeypatch, capsys):
    workflow_module = _load_workflow_module(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["2_ssebop_run_model.py", "--help"])

    with pytest.raises(SystemExit) as exc:
        workflow_module.main()

    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "--tcold-anchor-ndvi-threshold" in captured.out
    assert "--tcold-coarse-scale-m" in captured.out


def test_workflow_main_forwards_tcold_fano_args(monkeypatch):
    workflow_module = _load_workflow_module(monkeypatch)
    recorded = {}

    def fake_run_ssebop_workflow(**kwargs):
        recorded.update(kwargs)

    monkeypatch.setattr(workflow_module, "run_ssebop_workflow", fake_run_ssebop_workflow)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "2_ssebop_run_model.py",
            "--date-range",
            "2024-01-01 to 2024-01-03",
            "--landsat-dir",
            "/tmp/landsat",
            "--met-dir",
            "/tmp/met",
            "--dem",
            "/tmp/dem.tif",
            "--output-dir",
            "/tmp/out",
            "--tcold-dt-coeff",
            "0.15",
            "--tcold-high-ndvi-threshold",
            "0.85",
            "--tcold-anchor-ndvi-threshold",
            "0.35",
            "--tcold-fine-scale-m",
            "240",
            "--tcold-coarse-scale-m",
            "4800",
            "--tcold-smooth-scale-m",
            "240",
        ],
    )

    workflow_module.main()

    assert recorded["tcold_dt_coeff"] == 0.15
    assert recorded["tcold_high_ndvi_threshold"] == 0.85
    assert recorded["tcold_anchor_ndvi_threshold"] == 0.35
    assert recorded["tcold_fine_scale_m"] == 240.0
    assert recorded["tcold_coarse_scale_m"] == 4800.0
    assert recorded["tcold_smooth_scale_m"] == 240.0
```

- [ ] **Step 2: Run the forwarding tests to verify the workflow surface does not expose the new controls yet**

Run:

```bash
cd /g/data/ym05/github/yuyi13/PySWEB && python -m pytest tests/ssebop/test_api_run.py tests/workflows/test_2_ssebop_run_model.py -q
```

Expected: FAIL because the workflow help and parsed-argument surface do not expose the new `Tcold` configuration fields yet.

- [ ] **Step 3: Build and thread `LocalFanoConfig` through the package run path**

```python
# pysweb/ssebop/api.py
from pysweb.ssebop.core import (
    LocalFanoConfig,
    build_doy_climatology,
    compute_dt_daily,
    et_fraction_xr,
    tcold_fano_local_xr,
)


def process_landsat_scene(
    tif_path: str,
    lst_band: str,
    ndvi_band: str,
    red_band: str,
    nir_band: str,
    apply_water_mask: bool,
    water_mask: Optional[xr.DataArray],
    dt_clim: xr.DataArray,
    tcold_config: LocalFanoConfig,
    template_crs: Optional[CRS],
    etf_dir: str,
    ndvi_dir: str,
) -> Tuple[np.datetime64, str, str]:
    date_str = parse_date_from_filename(tif_path)
    doy = datetime.fromisoformat(date_str).timetuple().tm_yday
    bands = read_geotiff_bands(tif_path)

    if lst_band not in bands:
        raise ValueError(f"LST band '{lst_band}' not found in {tif_path}")
    lst = scale_landsat_st(bands[lst_band])

    if ndvi_band in bands:
        ndvi = bands[ndvi_band]
    else:
        if red_band not in bands or nir_band not in bands:
            raise ValueError(f"NDVI or red/nir bands missing in {tif_path}")
        red = scale_landsat_sr(bands[red_band])
        nir = scale_landsat_sr(bands[nir_band])
        ndvi = (nir - red) / (nir + red)

    if apply_water_mask and water_mask is not None:
        lst = lst.where(water_mask == 0)
        ndvi = ndvi.where(water_mask == 0)

    dt = reproject_match(dt_clim.sel(dayofyear = doy), lst, resampling = "bilinear")
    tcold = tcold_fano_local_xr(lst, ndvi, dt, config = tcold_config)
    etf = et_fraction_xr(lst, tcold, dt).rename("etf")
    ts = np.datetime64(date_str, "ns")
    etf = etf.assign_coords(time = ts).expand_dims("time")
    ndvi_out = ndvi.rename("ndvi").assign_coords(time = ts).expand_dims("time")

    for coord in ("dayofyear", "band"):
        if coord in etf.coords and coord not in etf.dims:
            etf = etf.drop_vars(coord)
        if coord in ndvi_out.coords and coord not in ndvi_out.dims:
            ndvi_out = ndvi_out.drop_vars(coord)

    out_etf = os.path.join(etf_dir, f"etf_{date_str}.tif")
    etf_single = etf.squeeze("time", drop = True)
    etf_single.rio.write_crs(template_crs, inplace = True)
    etf_single.rio.to_raster(out_etf)

    out_ndvi = os.path.join(ndvi_dir, f"ndvi_{date_str}.tif")
    ndvi_single = ndvi_out.squeeze("time", drop = True)
    ndvi_single.rio.write_crs(template_crs, inplace = True)
    ndvi_single.rio.to_raster(out_ndvi)
    return ts, out_etf, out_ndvi


def run_ssebop_workflow(
    date_range: Optional[str] = None,
    landsat_dir: Optional[str] = None,
    met_dir: Optional[str] = None,
    dem: Optional[str] = None,
    output_dir: Optional[str] = None,
    **kwargs,
) -> None:
    tcold_dt_coeff = float(_cfg_value(cfg, args, "tcold_dt_coeff", default = 0.125))
    tcold_high_ndvi_threshold = float(
        _cfg_value(cfg, args, "tcold_high_ndvi_threshold", default = 0.9)
    )
    tcold_anchor_ndvi_threshold = float(
        _cfg_value(cfg, args, "tcold_anchor_ndvi_threshold", default = 0.4)
    )
    tcold_fine_scale_m = float(_cfg_value(cfg, args, "tcold_fine_scale_m", default = 240.0))
    tcold_coarse_scale_m = float(_cfg_value(cfg, args, "tcold_coarse_scale_m", default = 4800.0))
    tcold_smooth_scale_m = float(_cfg_value(cfg, args, "tcold_smooth_scale_m", default = 240.0))

    tcold_config = LocalFanoConfig(
        dt_coeff = tcold_dt_coeff,
        high_ndvi_threshold = tcold_high_ndvi_threshold,
        anchor_ndvi_threshold = tcold_anchor_ndvi_threshold,
        fine_scale_m = tcold_fine_scale_m,
        coarse_scale_m = tcold_coarse_scale_m,
        smooth_scale_m = tcold_smooth_scale_m,
    )
    if workers == 1:
        for tif_path in landsat_files:
            scene_outputs.append(
                process_landsat_scene(
                    tif_path = tif_path,
                    lst_band = lst_band,
                    ndvi_band = ndvi_band,
                    red_band = red_band,
                    nir_band = nir_band,
                    apply_water_mask = apply_water_mask,
                    water_mask = water_mask,
                    dt_clim = dt_clim,
                    tcold_config = tcold_config,
                    template_crs = template_crs,
                    etf_dir = etf_dir,
                    ndvi_dir = ndvi_dir,
                )
            )
    else:
        scene_context: Dict[str, object] = {
            "lst_band": lst_band,
            "ndvi_band": ndvi_band,
            "red_band": red_band,
            "nir_band": nir_band,
            "apply_water_mask": apply_water_mask,
            "water_mask": water_mask,
            "dt_clim": dt_clim,
            "tcold_config": tcold_config,
            "template_crs": template_crs,
            "etf_dir": etf_dir,
            "ndvi_dir": ndvi_dir,
        }
```

```python
# pysweb/ssebop/api.py
def _process_landsat_scene_worker(tif_path: str) -> Tuple[np.datetime64, str, str]:
    if _SCENE_WORKER_CONTEXT is None:
        raise RuntimeError("Scene worker context has not been initialised")

    return process_landsat_scene(
        tif_path = tif_path,
        lst_band = str(_SCENE_WORKER_CONTEXT["lst_band"]),
        ndvi_band = str(_SCENE_WORKER_CONTEXT["ndvi_band"]),
        red_band = str(_SCENE_WORKER_CONTEXT["red_band"]),
        nir_band = str(_SCENE_WORKER_CONTEXT["nir_band"]),
        apply_water_mask = bool(_SCENE_WORKER_CONTEXT["apply_water_mask"]),
        water_mask = _SCENE_WORKER_CONTEXT["water_mask"],
        dt_clim = _SCENE_WORKER_CONTEXT["dt_clim"],
        tcold_config = _SCENE_WORKER_CONTEXT["tcold_config"],
        template_crs = _SCENE_WORKER_CONTEXT["template_crs"],
        etf_dir = str(_SCENE_WORKER_CONTEXT["etf_dir"]),
        ndvi_dir = str(_SCENE_WORKER_CONTEXT["ndvi_dir"]),
    )
```

```python
# workflows/2_ssebop_run_model.py
parser.add_argument("--tcold-dt-coeff", type = float, default = 0.125)
parser.add_argument("--tcold-high-ndvi-threshold", type = float, default = 0.9)
parser.add_argument("--tcold-anchor-ndvi-threshold", type = float, default = 0.4)
parser.add_argument("--tcold-fine-scale-m", type = float, default = 240.0)
parser.add_argument("--tcold-coarse-scale-m", type = float, default = 4800.0)
parser.add_argument("--tcold-smooth-scale-m", type = float, default = 240.0)
```

- [ ] **Step 4: Run the workflow-surface regression suite**

Run:

```bash
cd /g/data/ym05/github/yuyi13/PySWEB && python -m pytest tests/ssebop/test_api_run.py tests/workflows/test_2_ssebop_run_model.py tests/ssebop/test_core.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the workflow wiring**

```bash
cd /g/data/ym05/github/yuyi13/PySWEB && git add pysweb/ssebop/api.py workflows/2_ssebop_run_model.py tests/ssebop/test_api_run.py tests/workflows/test_2_ssebop_run_model.py && git commit -m "feat: wire local tcold config through ssebop workflow"
```

---

### Task 4: Run the focused verification sweep before merge or further tuning

**Files:**
- Modify: None
- Test: `tests/ssebop/test_core.py`
- Test: `tests/ssebop/test_api_run.py`
- Test: `tests/workflows/test_2_ssebop_run_model.py`

- [ ] **Step 1: Run the exact SSEBop regression sweep**

Run:

```bash
cd /g/data/ym05/github/yuyi13/PySWEB && python -m pytest \
  tests/ssebop/test_core.py \
  tests/ssebop/test_api_run.py \
  tests/workflows/test_2_ssebop_run_model.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run the package-backed workflow smoke tests that should stay unchanged**

Run:

```bash
cd /g/data/ym05/github/yuyi13/PySWEB && python -m pytest \
  tests/ssebop/test_api_prepare_inputs.py \
  tests/workflows/test_runners.py \
  -q
```

Expected: PASS.

- [ ] **Step 3: Record the scientific acceptance checklist in the commit message body**

```text
- ETf changes when LST changes at fixed NDVI and dT.
- The no-anchor fallback produces finite full-scene Tcold and ETf.
- The outer run workflow and CLI remain stable aside from new optional knobs.
```

- [ ] **Step 4: Create the final integration commit**

```bash
cd /g/data/ym05/github/yuyi13/PySWEB && git commit --allow-empty -m "test: verify local tcold correction regressions" -m "- ETf changes when LST changes at fixed NDVI and dT." -m "- The no-anchor fallback produces finite full-scene Tcold and ETf." -m "- The outer run workflow and CLI remain stable aside from new optional knobs."
```

- [ ] **Step 5: Capture the post-implementation sanity command for reviewers**

Run:

```bash
cd /g/data/ym05/github/yuyi13/PySWEB && python workflows/2_ssebop_run_model.py --help
```

Expected: help output includes the new `--tcold-*` options and still documents the existing Landsat/met/dem inputs.
