"""Reusable Leica file browsing and metadata helpers."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("convertleica-file-browser")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0+unknown"

__all__ = ["__version__"]

