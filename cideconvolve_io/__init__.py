"""Standalone microscopy image writers adapted from CIDeconvolve.

The original implementation is MIT licensed and was imported from
CIDeconvolve commit 570dcd157ad6b7029328ee8af343122aa8a4e22b.  This local
copy deliberately has no dependency on CIDeconvolve's processing core.
"""

from .ome_tiff_io import TiledOmeTiffSink, TiledRgbOmeTiffSink

__all__ = ["TiledOmeTiffSink", "TiledRgbOmeTiffSink"]
