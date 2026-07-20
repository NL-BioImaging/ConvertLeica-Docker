"""Compatibility import for the packaged ReadLeicaLIF module."""

from importlib import import_module
import sys

sys.modules[__name__] = import_module("ReadLeicaLIF")

