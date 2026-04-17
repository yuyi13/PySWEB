from importlib import import_module

__all__ = ["prepare_inputs", "run"]


def __getattr__(name):
    if name in __all__:
        api = import_module("pysweb.ssebop.api")
        value = getattr(api, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'pysweb.ssebop' has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)
