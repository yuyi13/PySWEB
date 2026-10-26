"""
Script: test_heatmap.py
Objective: Verify daily heatmap data semantics, optional context and NetCDF CLI compatibility.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-26
Last updated: 2026-10-26
Inputs: Synthetic daily layer, flux and NetCDF fixtures.
Outputs: Regression assertions and temporary figures.
Usage: python -m pytest tests/visualisation/test_heatmap.py
Dependencies: pytest, numpy, pandas, xarray, matplotlib; pysweb
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pysweb.visualisation.plot_heatmap import main, plot_heatmap


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def layers():
    return pd.DataFrame({"rzsm_layer_1": [0.1, 0.2, np.nan],
                         "rzsm_layer_2": [0.3, 0.4, 0.5]},
                        index=pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-04"]))


def test_missing_days_and_layers_remain_missing_in_storage_and_heatmap():
    fig, axes, data = plot_heatmap(layers(), [100, 900])
    np.testing.assert_allclose(data.soil_water_storage_mm, [280, np.nan, 380, np.nan], equal_nan=True)
    assert len(data) == 4
    assert np.ma.getmaskarray(axes["soil"].collections[0].get_array()).sum() == 3
    assert not axes["storage"].patches  # No implicit drought or Black Summer.
    assert "0–1 m" in axes["storage"].get_ylabel()
    fig.canvas.draw()


def test_fluxes_share_scale_and_events_affect_only_storage():
    frame = layers()
    fluxes = pd.DataFrame({"Effective rainfall": [50, 0, 2], "SWEB ET": [3, 4, 1]}, index=frame.index)
    event = {"start": "2020-01-01", "end": "2020-01-03", "label": "Drought", "colour": "#E5E5E5"}
    _, axes, data = plot_heatmap(frame, [100, 900], fluxes=fluxes, events=[event], flux_limits=(0, 50))
    mesh = axes["flux"].collections[0]
    assert mesh.cmap.name == "YlGnBu"
    np.testing.assert_allclose(mesh.norm([0, 2, 8, 18, 32, 50]), [0, 0.2, 0.4, 0.6, 0.8, 1])
    np.testing.assert_allclose(mesh.colorbar.get_ticks(), [0, 2, 10, 50])
    np.testing.assert_allclose(mesh.get_array().filled(np.nan),
                               [[50, np.nan, 0, 2], [3, np.nan, 4, 1]], equal_nan=True)
    assert len(axes["storage"].patches) == 1
    assert not axes["soil"].patches and not axes["flux"].patches
    patch = axes["storage"].patches[0]
    np.testing.assert_allclose(patch.get_facecolor(), [229 / 255] * 3 + [1])
    assert patch.get_x() == mdates.date2num(pd.Timestamp(event["start"]))
    assert patch.get_width() == 2
    assert data.loc["2020-01-01", "flux_SWEB ET"] == 3


def test_depth_unknown_never_invents_storage_or_millimetres():
    _, axes, data = plot_heatmap(layers())
    assert "storage" not in axes and "soil_water_storage_mm" not in data
    assert axes["soil"].get_ylabel() == "Soil layer"
    with pytest.raises(ValueError, match="requires known layer thicknesses"):
        plot_heatmap(layers(), events=[{"start": "2020-01-01", "end": "2020-02-01"}])


def test_colourbar_identifies_values_beyond_selected_soil_limits():
    _, axes, data = plot_heatmap(layers(), [100, 900], soil_limits=(0.2, 0.4))
    assert axes["soil"].collections[0].colorbar.extend == "both"
    assert data.loc["2020-01-01", "rzsm_layer_1"] == 0.1
    assert data.loc["2020-01-04", "rzsm_layer_2"] == 0.5


def test_phases_use_full_years_and_reference_period_is_explicit():
    fig, axes, _ = plot_heatmap(layers(), [100, 900],
                              phases=[("2016-01-01", "2021-01-01", "Development")],
                              reference_period=(2011, 2020))
    assert axes["phases"].texts[0].get_text() == "2016–2020\nDevelopment"
    assert any("2011–2020" in text.get_text() for text in fig.texts)


@pytest.mark.parametrize("thickness", [[100], [100, -1], [100, np.nan]])
def test_invalid_depths_rejected(thickness):
    with pytest.raises(ValueError, match="positive finite thickness"):
        plot_heatmap(layers(), thickness)


def test_invalid_dates_and_flux_scale_rejected():
    frame = layers()
    duplicate = pd.concat([frame, frame.iloc[:1]])
    with pytest.raises(ValueError, match="unique"):
        plot_heatmap(duplicate)
    fluxes = pd.DataFrame({"SWEB ET": [2, 3, 60]}, index=frame.index)
    with pytest.raises(ValueError, match="exceed flux_limits"):
        plot_heatmap(frame, [100, 900], fluxes=fluxes, flux_limits=(0, 50))


def write_netcdf(path, *, metadata=True, fluxes=True):
    frame = layers()
    ds = xr.Dataset({name: ("time", frame[name]) for name in frame}, coords={"time": frame.index})
    if metadata:
        for name, thickness, bottom in zip(frame.columns, [100, 900], [100, 1000]):
            ds[name].attrs.update(units="m3 m-3", layer_thickness_mm=thickness, depth_bottom_mm=bottom)
    if fluxes:
        ds["effective_precipitation"] = ("time", [5., 0., 2.], {"units": "mm day-1"})
        ds["actual_et"] = ("time", [1., 2., 1.], {"units": "mm", "time_support": "interval starting at time, duration 1 day"})
    ds.attrs["state_timing"] = "start"
    ds.to_netcdf(path)


def test_cli_extracts_actual_fluxes_and_keeps_ssebop_distinct(tmp_path):
    source = tmp_path / "sweb.nc"
    write_netcdf(source)
    ssebop = tmp_path / "ssebop.nc"
    xr.Dataset({"ET": ("time", [10., 11., 12.], {"units": "mm/day"})},
               coords={"time": layers().index}).to_netcdf(ssebop)
    out = tmp_path / "heatmap.png"
    csv = tmp_path / "data.csv"
    main(["--sweb-path", str(source), "--ssebop-path", str(ssebop), "--domain-mean",
          "--output", str(out), "--csv-out", str(csv), "--dpi", "60"])
    data = pd.read_csv(csv)
    np.testing.assert_allclose(data["flux_SWEB ET"], [1, np.nan, 2, 1], equal_nan=True)
    np.testing.assert_allclose(data["ssebop_ET"], [10, np.nan, 11, 12], equal_nan=True)
    assert out.stat().st_size > 1000


def test_cli_legacy_metadata_and_no_flux_options(tmp_path):
    source = tmp_path / "sweb.nc"
    write_netcdf(source, metadata=False, fluxes=False)
    csv = tmp_path / "data.csv"
    main(["--sweb-path", str(source), "--domain-mean", "--no-fluxes", "--sample-step", "2",
          "--output", str(tmp_path / "legacy.png"), "--csv-out", str(csv), "--dpi", "60"])
    data = pd.read_csv(csv)
    assert list(data.columns) == ["time", "sweb_rzsm_layer_1", "sweb_rzsm_layer_2"]
    assert data.loc[1:2, "sweb_rzsm_layer_2"].isna().all()
