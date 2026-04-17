from importlib import import_module

__all__ = ["era5land", "silo"]

_SUBMODULES = {name: f"pysweb.met.{name}" for name in __all__}


def __getattr__(name):
    if name in _SUBMODULES:
        module = import_module(_SUBMODULES[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module 'pysweb.met' has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)
