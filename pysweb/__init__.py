from importlib import import_module

__all__ = ["io", "met", "ssebop", "swb"]

_SUBMODULES = {name: f"pysweb.{name}" for name in __all__}


def __getattr__(name):
    if name in _SUBMODULES:
        module = import_module(_SUBMODULES[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module 'pysweb' has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)
