"""Compatibility import for the packaged Leica browser helpers."""

from importlib import import_module
import sys

sys.modules[__name__] = import_module("ci_leica_converters_helpers")

