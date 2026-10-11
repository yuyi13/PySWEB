"""
Script: components.py
Objective: Resolve canonical ET component names and validate deprecated input contracts.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-11
Last updated: 2026-10-11
Inputs: CLI/API arguments, component datasets and prepared forcing paths.
Outputs: Canonical arguments, validated variable selections and source paths.
Usage: from pysweb.components import component_variable, normalise_component_arguments
Dependencies: xarray; Python standard library
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import xarray as xr

COMPONENT_SCHEMA = "sweb-components-v2"
COMPONENT_ALIASES = {"E_s": ("E", "e"), "E_t": ("T", "t"), "f_t": ("Tc",)}
ARGUMENT_ALIASES = {
    "t": "transpiration",
    "t_var": "transpiration_var",
    "e_var": "soil_evaporation_var",
    "tc_var": "transpiration_fraction_var",
}


def _deprecated(old, new):
    warnings.warn(f"{old} is deprecated; use {new}.", FutureWarning, stacklevel=3)


def normalise_component_arguments(arguments):
    """Normalise only supplied values, before adding parser defaults."""
    result = dict(arguments)
    for old, new in ARGUMENT_ALIASES.items():
        if old not in result:
            continue
        value = result.pop(old)
        if value is None:
            result.setdefault(new, None)
            continue
        if new in result and result[new] is not None and str(result[new]) != str(value):
            raise ValueError(f"Conflicting arguments: {new} and deprecated {old}.")
        result[new] = value
        _deprecated(old, new)
    return result


class ComponentArgumentParser(argparse.ArgumentParser):
    """Reject conflicting repeated canonical/legacy flags without last-value wins."""

    def parse_known_args(self, args=None, namespace=None):
        result, remaining = super().parse_known_args(args, namespace)
        vars(result).pop("_component_arguments_seen", None)
        return result, remaining


class ComponentArgument(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        seen = vars(namespace).setdefault("_component_arguments_seen", {})
        if self.dest in seen and str(seen[self.dest]) != str(values):
            raise argparse.ArgumentError(self, f"Conflicting values for {self.dest}.")
        seen[self.dest] = values
        setattr(namespace, self.dest, values)
        if option_string != self.option_strings[0]:
            _deprecated(option_string, self.option_strings[0])


def _units(da):
    units = str(da.attrs.get("units", "")).strip()
    if units in {"mm/day", "mm day-1", "mm d-1", "mm day^-1"}:
        return "mm/day"
    if units in {"1", "dimensionless"}:
        return "1"
    return units


def component_variable(dataset, canonical, requested=None):
    """Select explicit names first; automatic fallback is limited to known aliases.

    All coexisting canonical/legacy fields must agree exactly in coordinates,
    dimensions, masks and values, with equivalent units. Temperature names are
    never candidates for automatic transpiration selection.
    """
    names = (canonical, *COMPONENT_ALIASES[canonical])
    present = [name for name in names if name in dataset.data_vars]
    for field, aliases in COMPONENT_ALIASES.items():
        duplicates = [name for name in (field, *aliases) if name in dataset.data_vars]
        for name in duplicates[1:]:
            first = dataset[duplicates[0]]
            other = dataset[name]
            try:
                xr.testing.assert_equal(first.rename(field), other.rename(field))
                if _units(first) != _units(other):
                    raise AssertionError("units differ")
            except AssertionError as exc:
                raise ValueError(f"Conflicting component variables: {duplicates[0]} and {name}.") from exc
    selected = requested or canonical
    if selected not in dataset.data_vars:
        if selected == canonical and present:
            selected = present[0]
        else:
            raise KeyError(f"Component variable '{selected}' not found; expected {canonical}.")
    attrs = dataset[selected].attrs
    temperature_units = {"k", "kelvin", "degc", "celsius", "degrees_celsius", "degree_celsius", "°c"}
    if canonical in {"E_s", "E_t"} and (
        _units(dataset[selected]).lower() in temperature_units
        or "temperature" in str(attrs.get("standard_name", "")).lower()
        or "temperature" in str(attrs.get("long_name", "")).lower()
    ):
        raise ValueError(f"Temperature variable {selected} cannot supply {canonical}.")
    if selected in COMPONENT_ALIASES[canonical]:
        _deprecated(f"component variable {selected}", canonical)
    return selected


def resolve_prepared_component_path(path, canonical="E_t"):
    """Use an old prepared filename only when its recorded checksum is valid.

    Explicitly supplied legacy source files remain ordinary inputs. Automatic
    fallback from a canonical filename requires a supported preprocessing
    manifest; callers record the resolved source path in new provenance.
    """
    candidate = Path(path).expanduser()
    prefix, legacy = {"E_t": ("transpiration_daily_", "t_daily_"),
                      "E_s": ("soil_evaporation_daily_", "e_daily_")}[canonical]
    if candidate.exists() or not candidate.name.startswith(prefix):
        return path
    tag = candidate.name[len(prefix):]
    old = candidate.with_name(legacy + tag)
    if not old.exists():
        return path
    sidecar = candidate.with_name("preprocess_" + tag.removesuffix(".nc") + ".manifest.json")
    from pysweb.io.provenance import sha256
    saved = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    entry = saved.get("outputs", {}).get(old.name, {})
    if (saved.get("schema_version") != 1
            or saved.get("component_schema") not in (None, COMPONENT_SCHEMA)
            or entry.get("sha256") != sha256(old)):
        raise ValueError(f"Legacy prepared file lacks supported provenance: {old}; rerun preprocessing.")
    # Validate the actual schema as well as its checksum before reusing it.
    with xr.open_dataset(old, decode_coords="all") as dataset:
        component_variable(dataset, canonical)
    _deprecated(f"prepared file {old.name}", candidate.name)
    return str(old)
