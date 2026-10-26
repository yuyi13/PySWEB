# Daily water-balance heatmaps

`pysweb heatmap` uses aligned panels for available daily water fluxes, layered
soil moisture and soil water storage. The defaults are Arial (when installed),
`esa_soil` for moisture, and `YlGnBu` with a shared square-root scale for fluxes.
Use `--cmap custom_sm` to retain the earlier soil palette. There is no drought,
fire or study-period annotation unless explicitly supplied.
Soil colourbar extensions identify values outside the selected `--vmin`/`--vmax`
range (defaults: 0–0.5 m³/m³); the exported data retain their original values.

```bash
pysweb heatmap --sweb-path outputs/SWEB_RZSM.nc --domain-mean \
  --output figures/heatmap.png --csv-out figures/heatmap_data.csv
```

Point extraction remains the default: supply `--lat` and `--lon`, or explicitly
select `--domain-mean`. The latter uses the existing extraction weights.
The output suffix selects the format; use `.pdf` for a vector export with
rasterised heatmap cells. The default page is 15 × 8.6 inches at 300 dpi.

The CLI reads `effective_precipitation` and `actual_et` when available, labels
them **Effective rainfall** and **SWEB ET**, and checks their units/time support.
A daily amount in mm is equivalent to mm/day only when its metadata identifies
a one-day interval. `--no-fluxes` omits both strips. `--flux-vmax` selects a
shared upper limit (otherwise rounded up from the data); values outside an
explicit flux scale cause an error. `--flux-gamma 1` selects a linear scale.
An explicitly supplied SSEBop file still gets an independent forcing line
panel in its own units; it is never relabelled as SWEB ET.

Soil water storage is the sum of each plotted layer's volumetric moisture
multiplied by its thickness in mm. The axis reports the actual integrated
depth, such as **Soil water storage, 0–1 m (mm)**. Missing layers produce missing
storage. Unknown or non-contiguous depth metadata yields a layer-only panel,
without inventing depths or storage. Missing dates remain visible as gaps;
`--sample-step` leaves skipped dates missing rather than extending a sampled
daily value over intervening days. The CSV contains the actual plotted daily
grid, including these gaps, available fluxes and derived storage.

## Optional phases and events

Pass `--annotations annotations.json` to add study phases, reference dates or
event shading. Interval starts are inclusive and ends exclusive. Reference
period endpoints are inclusive. Phase labels automatically include full years.
For example, the Yanco illustration uses:

```json
{
  "phases": [
    ["2014-01-01", "2016-01-01", "Spin-up"],
    ["2016-01-01", "2021-01-01", "Development"],
    ["2021-01-01", "2026-01-01", "Evaluation"]
  ],
  "reference_period": [2011, 2020],
  "events": [
    {"start": "2017-01-01", "end": "2019-11-01", "label": "NSW drought", "colour": "#E5E5E5"},
    {"start": "2019-11-01", "end": "2020-02-01", "label": "NSW Black Summer", "colour": "#BC3C29", "alpha": 0.14, "text_colour": "#BC3C29", "leader": true}
  ]
}
```

`#E5E5E5` is R/X11 `grey90`. Events shade only the storage panel, preserving
heatmap colours. They require known layer thicknesses. A custom `text_x` date
can move an event label; `leader: true` points back to the event midpoint.
Inspect labels for overlap when supplying many or closely spaced events.
No location, event dates, reference dates or phase names are hard-coded in the
renderer. These dates describe the requested Yanco illustration.

## Python figure API

```python
from pysweb.visualisation.plot_heatmap import plot_heatmap

# layers: daily DatetimeIndex DataFrame in m3/m3, ordered from surface downward.
# fluxes: daily DataFrame in mm/day, with column names used as strip labels.
fig, axes, plotted = plot_heatmap(
    layers, layer_thickness_mm=[50, 100, 150, 300, 400], fluxes=fluxes,
    title="Yanco · AU-Ync", state_timing="Start-of-day states · UTC days",
    phases=[("2014-01-01", "2016-01-01", "Spin-up"),
            ("2016-01-01", "2021-01-01", "Development"),
            ("2021-01-01", "2026-01-01", "Evaluation")],
    reference_period=(2011, 2020), flux_limits=(0, 50),
)
fig.savefig("heatmap.png", dpi=300)
fig.savefig("heatmap.pdf")
```

Pass `events=[...]` only when desired. Named axes include `flux`, `soil` and
`storage` when present, plus their colourbars. `plotted` contains the daily
layer values, `flux_<label>` columns and `soil_water_storage_mm`. Input indices
must be unique, timezone-naive daily timestamps in the caller's chosen day
convention; the API does not silently convert local days to UTC. No reference
period or state timing is inferred by this API.
