"""
Script: heatmap_figure.py
Objective: Draw aligned daily water-flux, soil-moisture and soil-water-storage panels.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-26
Last updated: 2026-10-26
Inputs: Daily layer water contents, optional layer thicknesses, fluxes, phases and events.
Outputs: Matplotlib figure, named axes and exact daily plotting data.
Usage: from pysweb.visualisation.plot_heatmap import plot_heatmap
Dependencies: matplotlib, numpy, pandas; pysweb
"""
from __future__ import annotations


def plot_heatmap(
    layers, layer_thickness_mm=None, *, fluxes=None, phases=(), events=(),
    title="SWEB water balance", subtitle="", reference_period=None,
    state_timing=None, cmap="esa_soil", soil_limits=(0, 0.5),
    flux_cmap="YlGnBu", flux_limits=(0, None), flux_gamma=0.5,
    storage_limits=None, storage_ticks=None, forcing=None, forcing_label="SSEBop forcing",
    figsize=(15, 8.6), dpi=300, font_size=11, title_size=22,
):
    """Return ``(figure, axes, data)`` for daily volumetric layer moisture.

    ``layers`` is a DatetimeIndex DataFrame in m3/m3, ordered from the surface
    downward. Positive ``layer_thickness_mm`` defines contiguous layers from
    zero; omit it when depth is unknown (no storage panel is then inferred).
    ``fluxes`` is an optional daily DataFrame in mm/day whose columns supply
    the strip labels. Missing dates are inserted as NaN, never interpolated.
    Storage is sum(theta * thickness), with any missing layer making it NaN.

    ``phases`` contains (start, end, label) tuples. ``events`` contains dicts
    with start/end dates and optional label, colour, alpha and text_colour.
    All intervals are start-inclusive/end-exclusive. Events are off by default
    and shade only the storage panel; no region-specific events are implicit.
    Optional event text_x sets its date position and leader=True adds a leader.
    Phase labels receive full year ranges automatically. ``reference_period``
    is an inclusive pair of dates or years for the footer.

    ``forcing`` preserves the independent SSEBop line panel in its supplied
    units; it is never substituted for SWEB ET. No flux/state timing is assumed
    beyond the supplied daily index and optional ``state_timing`` description.
    """
    import warnings
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib import font_manager
    from matplotlib.colors import PowerNorm
    import numpy as np
    import pandas as pd
    from pysweb.visualisation.plot_heatmap import _get_custom_colormap
    from pysweb.visualisation.plot_time_series import _configure_plot_font

    def daily(frame, name):
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is not None:
            raise ValueError(f"{name} needs a timezone-naive DatetimeIndex in the intended day convention.")
        if frame.index.has_duplicates or frame.index.hasnans:
            raise ValueError(f"{name} dates must be unique and finite.")
        if not frame.index.equals(frame.index.normalize()):
            raise ValueError(f"{name} dates must identify daily midnight intervals.")
        return frame.sort_index().astype(float)

    layers = daily(layers, "layers")
    if layers.empty or not layers.columns.is_unique:
        raise ValueError("Provide at least one layer and date, with unique column names.")
    dates = pd.date_range(layers.index[0], layers.index[-1], freq="D")
    layers = layers.reindex(dates)
    thickness = None
    if layer_thickness_mm is not None:
        thickness = np.asarray(layer_thickness_mm, dtype=float)
        if thickness.shape != (layers.shape[1],) or not np.all(np.isfinite(thickness) & (thickness > 0)):
            raise ValueError("layer_thickness_mm must contain one positive finite thickness per layer.")
    if events and thickness is None:
        raise ValueError("Event shading requires known layer thicknesses and a storage panel.")
    if flux_gamma <= 0 or not np.isfinite(flux_gamma):
        raise ValueError("flux_gamma must be positive and finite.")
    flux_data = None if fluxes is None else daily(fluxes, "fluxes").reindex(dates)
    if flux_data is not None and (flux_data.shape[1] == 0 or not flux_data.columns.is_unique):
        raise ValueError("Provide uniquely named flux columns or omit fluxes.")
    if forcing is not None:
        forcing = daily(forcing, "forcing").reindex(dates)
    data = layers.copy()
    if flux_data is not None:
        for column in flux_data:
            data[f"flux_{column}"] = flux_data[column]
    if thickness is not None:
        data["soil_water_storage_mm"] = layers.to_numpy() @ thickness
    if forcing is not None:
        data["ssebop_forcing"] = forcing

    def interval(start, end):
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        if pd.isna(start) or pd.isna(end) or start >= end or start.tz is not None or end.tz is not None:
            raise ValueError("Intervals require finite, timezone-naive start < end dates.")
        return start, end

    phases = [(*interval(start, end), label) for start, end, label in phases]
    prepared_events = []
    for event in events:
        start, end = interval(event["start"], event["end"])
        prepared_events.append(dict(event, start=start, end=end))
    events = prepared_events
    x_edges = mdates.date2num(dates.append(pd.DatetimeIndex([dates[-1] + pd.Timedelta(days=1)])))
    x_start, x_end = x_edges[0], x_edges[-1]
    _configure_plot_font(pyplot=plt)
    try:
        font_manager.findfont("Arial", fallback_to_default=False)
        family = "Arial"
    except ValueError:
        warnings.warn("Arial is unavailable; install Arial for publication exports. Using DejaVu Sans.", stacklevel=2)
        family = "DejaVu Sans"
    style = {
        "font.family": family, "font.size": font_size, "font.weight": "normal",
        "axes.labelsize": font_size, "axes.titlesize": font_size,
        "xtick.labelsize": font_size, "ytick.labelsize": font_size,
        "legend.fontsize": font_size, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.fontset": "custom", "mathtext.default": "regular", "mathtext.fallback": None,
        **{f"mathtext.{key}": family for key in ("rm", "it", "bf", "sf", "tt", "cal")},
    }
    plt.rcParams.update(style)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    axes = {}
    left, right, gap, bottom = 0.15, 0.865, 0.012, 0.105
    fig.text(left, 0.97, title, fontsize=title_size, va="top")
    fig.text(right, 0.965, subtitle, ha="right", va="top")
    top = 0.853 if phases else 0.92
    if phases:
        phase_ax = fig.add_axes([left, 0.865, right - left, 0.065])
        axes["phases"] = phase_ax
        phase_ax.set(xlim=(x_start, x_end), ylim=(0, 1))
        phase_ax.set_axis_off()
        for i, (start, end, label) in enumerate(phases):
            x0, x1 = mdates.date2num([start, end])
            if x1 <= x_start or x0 >= x_end:
                continue
            phase_ax.axvspan(x0, x1, facecolor=("#F0F0F0" if i % 2 == 0 else "#F8F8F8"),
                             edgecolor="white", linewidth=1.5)
            last = end - pd.Timedelta(nanoseconds=1)
            years = str(start.year) if start.year == last.year else f"{start.year}–{last.year}"
            phase_ax.text((max(x0, x_start) + min(x1, x_end)) / 2, 0.5,
                          f"{years}\n{label}", ha="center", va="center", linespacing=1.4)
    panels = []
    if forcing is not None:
        panels.append(("forcing", 0.126))
    if flux_data is not None:
        panels.append(("flux", 0.126))
    panels.append(("soil", 0.398))
    if thickness is not None:
        panels.append(("storage", 0.200))
    available = top - bottom - gap * (len(panels) - 1)
    scale = available / sum(weight for _, weight in panels)
    y_top = top
    for i, (key, weight) in enumerate(panels):
        height = weight * scale
        axes[key] = fig.add_axes([left, y_top - height, right - left, height])
        fig.text(0.022, y_top, f"({chr(97 + i)})", fontsize=title_size, va="top")
        y_top -= height + gap

    def colourbar(mesh, key, label, ticks=None, extend="neither"):
        bounds = axes[key].get_position()
        cax = fig.add_axes([0.886, bounds.y0 + 0.007, 0.014, bounds.height - 0.014])
        axes[f"{key}_colourbar"] = cax
        bar = fig.colorbar(mesh, cax=cax, ticks=ticks, extend=extend)
        bar.set_label(label, labelpad=9)
        bar.ax.minorticks_off()
        bar.outline.set_linewidth(0.7)

    if flux_data is not None:
        values = flux_data.to_numpy().T
        finite = values[np.isfinite(values)]
        low, high = flux_limits
        if high is None:
            high = max(1.0, float(np.ceil(finite.max()))) if finite.size else 1.0
        if not np.isfinite(low) or not np.isfinite(high) or low < 0 or low >= high:
            raise ValueError("Flux limits must be finite with 0 <= minimum < maximum.")
        if finite.size and (finite.min() < low or finite.max() > high):
            raise ValueError("Flux values exceed flux_limits; expand the scale rather than clip them.")
        colours = plt.get_cmap(flux_cmap).copy()
        colours.set_bad("#E5E5E5")
        ax = axes["flux"]
        mesh = ax.pcolormesh(x_edges, np.arange(len(values) + 1), np.ma.masked_invalid(values),
                            cmap=colours, norm=PowerNorm(flux_gamma, vmin=low, vmax=high),
                            shading="flat", rasterized=True)
        ax.set_ylim(len(values), 0)
        ax.set_yticks(np.arange(len(values)) + 0.5, list(flux_data.columns))
        ax.tick_params(axis="y", length=0, pad=8)
        for boundary in range(1, len(values)):
            ax.axhline(boundary, color="white", lw=1.5)
        ticks = [0, 2, 10, 50] if low == 0 and high == 50 and flux_gamma == 0.5 else None
        colourbar(mesh, "flux", r"Flux (mm day$^{-1}$)", ticks)
    edges = np.r_[0, np.cumsum(thickness)] if thickness is not None else np.arange(layers.shape[1] + 1)
    colours = _get_custom_colormap(cmap).copy()
    colours.set_bad("#E5E5E5")
    ax = axes["soil"]
    mesh = ax.pcolormesh(x_edges, edges, np.ma.masked_invalid(layers.to_numpy().T),
                        cmap=colours, vmin=soil_limits[0], vmax=soil_limits[1],
                        shading="flat", rasterized=True)
    ax.set_ylim(edges[-1], 0)
    labels = [f"{a:g}–{b:g}" for a, b in zip(edges[:-1], edges[1:])] if thickness is not None else list(layers.columns)
    ax.set_yticks((edges[:-1] + edges[1:]) / 2, labels)
    ax.set_ylabel("Soil depth (mm)" if thickness is not None else "Soil layer", labelpad=10)
    finite_soil = layers.to_numpy()[np.isfinite(layers.to_numpy())]
    below = finite_soil.size > 0 and finite_soil.min() < mesh.norm.vmin
    above = finite_soil.size > 0 and finite_soil.max() > mesh.norm.vmax
    extension = "both" if below and above else ("min" if below else "max" if above else "neither")
    colourbar(mesh, "soil", r"Soil moisture (m$^3$ m$^{-3}$)", extend=extension)
    if forcing is not None:
        axes["forcing"].plot(dates, forcing, color="#0072B5", lw=1.05)
        axes["forcing"].set_ylabel(forcing_label)
    if thickness is not None:
        ax = axes["storage"]
        values = data.soil_water_storage_mm.to_numpy()
        ax.plot(dates, values, color="#E18727", lw=1.05)
        ax.set_ylabel(f"Soil water storage\n0–{edges[-1] / 1000:g} m (mm)", labelpad=11, linespacing=1.5)
        if storage_limits is not None:
            ax.set_ylim(*storage_limits)
        else:
            finite = values[np.isfinite(values)]
            low, high = (finite.min(), finite.max()) if finite.size else (0, edges[-1])
            pad = max((high - low) * 0.12, 1)
            ax.set_ylim(max(0, low - pad), high + pad * (2 if events else 1))
        if storage_ticks is not None:
            ax.set_yticks(storage_ticks)
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.5, zorder=0)
        for event in events:
            start, end = event["start"], event["end"]
            if mdates.date2num(end) <= x_start or mdates.date2num(start) >= x_end:
                continue
            ax.axvspan(start, end, color=event.get("colour", "#E5E5E5"),
                       alpha=event.get("alpha", 1), linewidth=0, zorder=0)
            midpoint = max(start, dates[0]) + (min(end, dates[-1] + pd.Timedelta(days=1)) - max(start, dates[0])) / 2
            text_x = pd.Timestamp(event.get("text_x", midpoint))
            ax.annotate(event.get("label", ""), xy=(midpoint, 0.72), xycoords=("data", "axes fraction"),
                        xytext=(text_x, 0.983), textcoords=("data", "axes fraction"),
                        ha="center", va="top", color=event.get("text_colour", "black"),
                        arrowprops=({"arrowstyle": "-", "color": event.get("text_colour", "black"), "lw": 0.8}
                                    if event.get("leader", False) else None))
    for key, _ in panels:
        ax = axes[key]
        ax.set_xlim(x_start, x_end)
        boundaries = {date for start, end, _ in phases for date in (start, end)}
        for date in sorted(boundaries):
            if x_start < mdates.date2num(date) < x_end:
                ax.axvline(date, color="#333333", lw=0.8)
        if key != panels[-1][0]:
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax = axes[panels[-1][0]]
    if len(dates) >= 730:
        year_step = max(1, int(np.ceil((dates[-1].year - dates[0].year + 1) / 14)))
        years = pd.date_range(f"{dates[0].year}-01-01", dates[-1], freq=f"{year_step}YS")
        years = years[years >= dates[0]]
        ax.set_xticks(years, [str(date.year) for date in years])
        ax.set_xlabel("Year", labelpad=8)
    else:
        locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.set_xlabel("Date", labelpad=8)
    footer = []
    if state_timing:
        footer.append(state_timing)
    if flux_data is not None:
        scale_label = "square-root" if flux_gamma == 0.5 else ("linear" if flux_gamma == 1 else f"power {flux_gamma:g}")
        footer.append(f"Daily fluxes ({scale_label} colour scale)")
    if reference_period is not None:
        years = [pd.Timestamp(str(date)).year for date in reference_period]
        if len(years) != 2 or years[0] > years[1]:
            raise ValueError("reference_period must contain an ordered pair of dates or years.")
        footer.append(f"Wet–dry references: {years[0]}–{years[1]}")
    fig.text(left, 0.018, " · ".join(footer), va="center", gid="sweb-footer")
    return fig, axes, data
