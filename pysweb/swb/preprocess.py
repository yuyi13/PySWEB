#!/usr/bin/env python3
"""
Script: preprocess.py
Objective: Provide the package-owned SWB preprocess workflow entry point.
Author: Yi Yu
Created: 2026-04-19
Last updated: 2026-04-19
Inputs: Command-line arguments or keyword arguments for SWB preprocessing.
Outputs: Parsed arguments forwarded to the preprocess implementation.
Usage: Imported as `pysweb.swb.preprocess` or run as a module entry point.
Dependencies: argparse, sys, types
"""
from __future__ import annotations

import argparse
import sys
import types
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description = "Preprocess spatial datasets into aligned SWB NetCDF files."
    )
    parser.add_argument("--output-dir", required = True)
    parser.add_argument("--reference-source", default = "gssm1km")
    parser.add_argument("--reference-ssm-asset", default = "users/qianrswaterr/GlobalSSM1km0509")
    parser.add_argument("--gee-project", default = "yiyu-research")
    parser.add_argument("--skip-reference-ssm", action = "store_true")
    return parser


def preprocess_inputs(**kwargs):
    return kwargs


def main(argv: Sequence[str] | None = None):
    args = build_parser().parse_args(argv)
    return preprocess_inputs(**vars(args))


class _CallablePreprocessModule(types.ModuleType):
    def __call__(self, **kwargs):
        return preprocess_inputs(**kwargs)


sys.modules[__name__].__class__ = _CallablePreprocessModule


if __name__ == "__main__":
    raise SystemExit(main())
