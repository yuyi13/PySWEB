"""Package-level exports for Google Earth Engine download helpers."""

__all__ = ["GEEDownloader", "_safe_mkdir"]


def __getattr__(name):
    if name in __all__:
        from core.gee_downloader import GEEDownloader, _safe_mkdir

        exports = {
            "GEEDownloader": GEEDownloader,
            "_safe_mkdir": _safe_mkdir,
        }
        value = exports[name]
        globals()[name] = value
        return value
    raise AttributeError(f"module 'pysweb.io.gee' has no attribute {name!r}")
