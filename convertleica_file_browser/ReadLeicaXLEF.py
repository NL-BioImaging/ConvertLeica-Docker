"""Compatibility import for the packaged ReadLeicaXLEF module."""

from importlib import import_module
import sys

sys.modules[__name__] = import_module("ReadLeicaXLEF")

