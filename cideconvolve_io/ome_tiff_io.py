"""Tiled, pyramidal OME-TIFF writers adapted from CIDeconvolve.

Unlike the deconvolution writer, these sinks preserve the integer container
type produced by Leica instruments and can embed an already-complete OME-XML
document.  Data are staged in local memmaps so arbitrary source tiles can be
written without holding the complete image in RAM.
"""

from __future__ import annotations

import math
import shutil
import tempfile
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np


_OME_NS = "http://www.openmicroscopy.org/Schemas/OME/2016-06"


def _merge_ome_xml(generated_xml: str, rich_xml: str) -> str:
    """Merge Leica metadata into tifffile's reader-tested OME structure.

    The generated document remains authoritative for Pixels dimensions,
    channel/sample layout, and TiffData/SubIFD mapping. Rich acquisition and
    instrument fields are copied without disturbing that structural contract.
    """
    generated = ET.fromstring(generated_xml)
    rich = ET.fromstring(rich_xml)
    q = lambda name: f"{{{_OME_NS}}}{name}"
    image = generated.find(q("Image"))
    rich_image = rich.find(q("Image"))
    if image is None or rich_image is None:
        return generated_xml

    if rich_image.get("Name"):
        image.set("Name", rich_image.get("Name", ""))
    pixels = image.find(q("Pixels"))
    rich_pixels = rich_image.find(q("Pixels"))
    if pixels is not None and rich_pixels is not None:
        for key in (
            "PhysicalSizeX", "PhysicalSizeXUnit", "PhysicalSizeY", "PhysicalSizeYUnit",
            "PhysicalSizeZ", "PhysicalSizeZUnit", "SignificantBits",
        ):
            if rich_pixels.get(key) is not None:
                pixels.set(key, rich_pixels.get(key, ""))
        channels = pixels.findall(q("Channel"))
        rich_channels = rich_pixels.findall(q("Channel"))
        for index, channel in enumerate(channels):
            if index >= len(rich_channels):
                break
            source = rich_channels[index]
            for key, value in source.attrib.items():
                if key not in {"ID", "SamplesPerPixel"}:
                    channel.set(key, value)
            for detector in source.findall(q("DetectorSettings")):
                channel.append(detector)

    insert_at = list(generated).index(image)
    for instrument in list(generated.findall(q("Instrument"))):
        generated.remove(instrument)
    for instrument in rich.findall(q("Instrument")):
        generated.insert(insert_at, instrument)
        insert_at += 1

    existing_image_tags = {child.tag for child in image}
    pixels_index = list(image).index(pixels) if pixels is not None else len(image)
    for name in ("AcquisitionDate", "Description", "ExperimenterRef", "InstrumentRef", "ObjectiveSettings"):
        child = rich_image.find(q(name))
        if child is not None and child.tag not in existing_image_tags:
            image.insert(pixels_index, child)
            pixels_index += 1
            existing_image_tags.add(child.tag)

    for annotations in [*rich.findall(q("StructuredAnnotations")), *rich_image.findall(q("StructuredAnnotations"))]:
        generated.append(annotations)

    ET.register_namespace("", _OME_NS)
    ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")
    return '<?xml version="1.0" encoding="UTF-8"?>' + ET.tostring(generated, encoding="unicode")


def _tile_dim(requested: int, extent: int) -> int:
    requested = max(16, int(requested))
    aligned_extent = max(16, int(math.ceil(max(1, extent) / 16.0) * 16))
    return min(requested, aligned_extent)


def _default_levels(yx: tuple[int, int]) -> int:
    levels = 1
    y, x = (int(yx[0]), int(yx[1]))
    while max(y, x) > 512 and levels < 8:
        y, x = max(1, math.ceil(y / 2)), max(1, math.ceil(x / 2))
        levels += 1
    return levels


def _downsample_xy(source: np.ndarray, destination: np.ndarray) -> None:
    """Write a 2x mean XY reduction, retaining odd edge rows and columns."""
    is_integer = np.issubdtype(destination.dtype, np.integer)
    for index in np.ndindex(source.shape[:-2]):
        plane = source[index]
        sums = np.asarray(plane[0::2, 0::2], dtype=np.float32).copy()
        counts = np.ones(sums.shape, dtype=np.uint8)
        for dy, dx in ((0, 1), (1, 0), (1, 1)):
            part = np.asarray(plane[dy::2, dx::2], dtype=np.float32)
            py, px = part.shape
            sums[:py, :px] += part
            counts[:py, :px] += 1
        values = sums / counts
        destination[index] = np.rint(values) if is_integer else values


def _downsample_rgb_xy(source: np.ndarray, destination: np.ndarray) -> None:
    is_integer = np.issubdtype(destination.dtype, np.integer)
    for index in np.ndindex(source.shape[:-3]):
        plane = source[index]
        sums = np.asarray(plane[0::2, 0::2, :], dtype=np.float32).copy()
        counts = np.ones(sums.shape[:2], dtype=np.uint8)
        for dy, dx in ((0, 1), (1, 0), (1, 1)):
            part = np.asarray(plane[dy::2, dx::2, :], dtype=np.float32)
            py, px = part.shape[:2]
            sums[:py, :px, :] += part
            counts[:py, :px] += 1
        values = sums / counts[..., np.newaxis]
        destination[index] = np.rint(values) if is_integer else values


class _BaseTiffSink:
    photometric = "minisblack"

    def __init__(
        self,
        path: str | Path,
        *,
        shape: tuple[int, ...],
        dtype: np.dtype | str,
        ome_xml: str,
        tile_yx: tuple[int, int] = (512, 512),
        levels: int | None = None,
        compression: str | None = "lzw",
        temp_dir: str | Path | None = None,
        level0: np.ndarray | None = None,
    ) -> None:
        self.path = Path(path)
        self.shape = tuple(int(v) for v in shape)
        self.dtype = np.dtype(dtype)
        xml = str(ome_xml)
        root = '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
        if root in xml:
            replacement = (
                '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" '
                'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
                'Creator="ConvertLeica" '
                f'UUID="urn:uuid:{uuid.uuid4()}" '
                'xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 '
                'http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">'
            )
            xml = xml.replace(root, replacement, 1)
        # TIFF ImageDescription is an ASCII field. XML character references
        # preserve units such as µm and arbitrary Leica text losslessly.
        self.ome_xml = xml.encode("ascii", "xmlcharrefreplace").decode("ascii")
        self.tile_yx = tuple(max(16, int(v)) for v in tile_yx)
        base_y, base_x = self._yx(self.shape)
        self._tiff_tile = (_tile_dim(self.tile_yx[0], base_y), _tile_dim(self.tile_yx[1], base_x))
        self.levels = int(levels) if levels is not None else _default_levels(self._yx(self.shape))
        self.compression = compression
        self.path.parent.mkdir(parents=True, exist_ok=True)
        root = str(temp_dir) if temp_dir else None
        if root:
            Path(root).mkdir(parents=True, exist_ok=True)
        self._tmpdir = Path(tempfile.mkdtemp(prefix="convertleica_tiff_", dir=root))
        self._owns_level0 = level0 is None
        if level0 is None:
            self._level0 = np.memmap(self._tmpdir / "level0.dat", dtype=self.dtype, mode="w+", shape=self.shape)
        else:
            if tuple(level0.shape) != self.shape or np.dtype(level0.dtype) != self.dtype:
                raise ValueError(
                    f"External level 0 is {level0.shape}/{level0.dtype}, expected {self.shape}/{self.dtype}"
                )
            self._level0 = level0
        self._pyramids: list[np.memmap] = []
        self._closed = False

    @staticmethod
    def _yx(shape: tuple[int, ...]) -> tuple[int, int]:
        return int(shape[-2]), int(shape[-1])

    def _pyramid_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        return (*shape[:-2], max(1, math.ceil(shape[-2] / 2)), max(1, math.ceil(shape[-1] / 2)))

    def _downsample(self, source: np.ndarray, destination: np.ndarray) -> None:
        _downsample_xy(source, destination)

    def build_pyramids(self) -> None:
        current: np.ndarray = self._level0
        self._pyramids = []
        for level in range(1, max(1, self.levels)):
            shape = self._pyramid_shape(tuple(current.shape))
            pyramid = np.memmap(self._tmpdir / f"level{level}.dat", dtype=self.dtype, mode="w+", shape=shape)
            self._downsample(current, pyramid)
            pyramid.flush()
            self._pyramids.append(pyramid)
            current = pyramid

    def validate(self) -> None:
        if tuple(self._level0.shape) != self.shape:
            raise ValueError(f"Staging shape {self._level0.shape} does not match {self.shape}")
        if not self.ome_xml.lstrip().startswith("<?xml"):
            raise ValueError("A complete OME-XML document is required")

    def _write_kwargs(self, array: np.ndarray, *, reduced: bool = False) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "photometric": self.photometric,
            "tile": self._tiff_tile,
        }
        if reduced:
            kwargs["subfiletype"] = 1
        if self.compression:
            kwargs["compression"] = self.compression
            kwargs["predictor"] = True
        return kwargs

    def _tifffile_metadata(self) -> dict[str, Any]:
        return self._metadata_from_rich_xml("TCZYX")

    def _metadata_from_rich_xml(self, axes: str) -> dict[str, Any]:
        root = ET.fromstring(self.ome_xml)
        q = lambda name: f"{{{_OME_NS}}}{name}"
        image = root.find(q("Image"))
        pixels = image.find(q("Pixels")) if image is not None else None
        metadata: dict[str, Any] = {"axes": axes}
        if image is not None and image.get("Name"):
            metadata["Name"] = image.get("Name")
        if pixels is None:
            return metadata
        for key in (
            "PhysicalSizeX", "PhysicalSizeXUnit", "PhysicalSizeY", "PhysicalSizeYUnit",
            "PhysicalSizeZ", "PhysicalSizeZUnit", "SignificantBits",
        ):
            value = pixels.get(key)
            if value is None:
                continue
            metadata[key] = int(value) if key == "SignificantBits" else value
        channels = pixels.findall(q("Channel"))
        if channels:
            channel_meta: dict[str, Any] = {
                "Name": [channel.get("Name") or f"Channel {i + 1}" for i, channel in enumerate(channels)]
            }
            for key in ("Color", "EmissionWavelength", "EmissionWavelengthUnit",
                        "ExcitationWavelength", "ExcitationWavelengthUnit", "PinholeSize", "PinholeSizeUnit"):
                values = [channel.get(key) for channel in channels]
                if all(value is not None for value in values):
                    if key == "Color":
                        channel_meta[key] = [int(value) for value in values]
                    elif key in {"EmissionWavelength", "ExcitationWavelength", "PinholeSize"}:
                        channel_meta[key] = [float(value) for value in values]
                    else:
                        channel_meta[key] = values
            metadata["Channel"] = channel_meta
        return metadata

    def _write_once(self, compression: str | None) -> None:
        import tifffile

        old_compression, self.compression = self.compression, compression
        try:
            if self.path.exists():
                self.path.unlink()
            flush = getattr(self._level0, "flush", None)
            if flush is not None:
                flush()
            kwargs = self._write_kwargs(self._level0)
            kwargs["metadata"] = self._tifffile_metadata()
            kwargs["extratags"] = [(65000, "s", 0, self.ome_xml, True)]
            if self._pyramids:
                kwargs["subifds"] = len(self._pyramids)
            with tifffile.TiffWriter(self.path, bigtiff=True, ome=True) as tif:
                tif.write(self._level0, **kwargs)
                for pyramid in self._pyramids:
                    tif.write(pyramid, **self._write_kwargs(pyramid, reduced=True))
        finally:
            self.compression = old_compression

    def close(self) -> None:
        if self._closed:
            return
        self.validate()
        try:
            try:
                self._write_once(self.compression)
            except Exception:
                if not self.compression:
                    raise
                self._write_once(None)
            if not self.path.is_file() or self.path.stat().st_size == 0:
                raise OSError(f"OME-TIFF was not created: {self.path}")
            self._closed = True
        finally:
            self._dispose_staging()

    def _dispose_staging(self) -> None:
        self._pyramids.clear()
        level0 = getattr(self, "_level0", None)
        if level0 is not None and self._owns_level0:
            try:
                level0.flush()
            except Exception:
                pass
        self._level0 = None  # type: ignore[assignment]
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def abort(self) -> None:
        self._dispose_staging()
        if self.path.exists() and not self._closed:
            self.path.unlink(missing_ok=True)


class TiledOmeTiffSink(_BaseTiffSink):
    """Incremental sink for canonical ``T,C,Z,Y,X`` image data."""

    def __init__(self, path: str | Path, *, shape: tuple[int, int, int, int, int], **kwargs: Any) -> None:
        super().__init__(path, shape=shape, **kwargs)

    def write_tile(self, *, t: int, c: int, z: int | slice, y: slice, x: slice, data: np.ndarray) -> None:
        value = np.asarray(data, dtype=self.dtype)
        self._level0[int(t), int(c), z, y, x] = value


class TiledRgbOmeTiffSink(_BaseTiffSink):
    """Incremental sink for true interleaved ``T,Z,Y,X,S`` RGB data."""

    photometric = "rgb"

    def __init__(self, path: str | Path, *, shape: tuple[int, int, int, int, int], **kwargs: Any) -> None:
        if len(shape) != 5 or int(shape[-1]) != 3:
            raise ValueError(f"RGB sink expects TZYXS with S=3, got {shape}")
        super().__init__(path, shape=shape, **kwargs)

    @staticmethod
    def _yx(shape: tuple[int, ...]) -> tuple[int, int]:
        return int(shape[-3]), int(shape[-2])

    def _pyramid_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        return (*shape[:-3], max(1, math.ceil(shape[-3] / 2)), max(1, math.ceil(shape[-2] / 2)), shape[-1])

    def _downsample(self, source: np.ndarray, destination: np.ndarray) -> None:
        _downsample_rgb_xy(source, destination)

    def _write_kwargs(self, array: np.ndarray, *, reduced: bool = False) -> dict[str, Any]:
        kwargs = super()._write_kwargs(array, reduced=reduced)
        kwargs["planarconfig"] = "contig"
        return kwargs

    def _tifffile_metadata(self) -> dict[str, Any]:
        return self._metadata_from_rich_xml("TZYXS")

    def write_tile(self, *, t: int, z: int | slice, y: slice, x: slice, data: np.ndarray) -> None:
        value = np.asarray(data, dtype=self.dtype)
        self._level0[int(t), z, y, x, :] = value
