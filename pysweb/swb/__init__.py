from importlib import import_module

__all__ = ["calibrate", "preprocess", "run"]


def __getattr__(name):
    if name in __all__:
        api = import_module("pysweb.swb.api")
        value = getattr(api, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'pysweb.swb' has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)
