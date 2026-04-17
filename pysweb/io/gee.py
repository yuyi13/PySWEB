"""Package-level exports for Google Earth Engine download helpers."""

from __future__ import annotations

from typing import Any

__all__ = ["GEEDownloader", "_safe_mkdir"]


def _load_legacy():
    from core.gee_downloader import GEEDownloader as LegacyGEEDownloader
    from core.gee_downloader import _safe_mkdir as legacy_safe_mkdir

    return LegacyGEEDownloader, legacy_safe_mkdir


class GEEDownloader:
    """Thin stable adapter over the legacy core downloader."""

    def __init__(self, config_path: str, *args: Any, **kwargs: Any) -> None:
        legacy_cls, _ = _load_legacy()
        self._legacy = legacy_cls(config_path, *args, **kwargs)

    def run(self) -> Any:
        return self._legacy.run()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._legacy, name)


def _safe_mkdir(path: str) -> None:
    _, legacy_safe_mkdir = _load_legacy()
    legacy_safe_mkdir(path)
