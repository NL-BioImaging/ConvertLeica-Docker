import gc
import json
import math
import os
import shutil
import uuid
from html import escape as escape_xml_chars
from pathlib import Path

import cv2
import numpy as np

from ci_leica_converters_helpers import (
    color_name_to_decimal,
    compute_channel_intensity_stats,
    decimal_to_rgb,
    prepare_temp_source,
    cleanup_temp_source,
    print_progress_bar,
    read_image_metadata,
)


def _import_zarr_modules():
    try:
        import zarr
        from numcodecs import Blosc
    except Exception as exc:
        raise RuntimeError(
            "OME-Zarr support requires the 'zarr' and 'numcodecs' packages."
        ) from exc
    return zarr, Blosc


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(val) for val in value]
    if isinstance(value, tuple):
        return [_json_safe(val) for val in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _ome_zarr_name(save_child_name: str, suffix: str = ".ome.zarr") -> str:
    return f"{save_child_name}{suffix}"


def _color_to_hex(color_name: str) -> str:
    try:
        rgb_int = color_name_to_decimal(color_name)
    except Exception:
        rgb_int = color_name_to_decimal("white")
    r, g, b = decimal_to_rgb(rgb_int)
    return f"{r:02x}{g:02x}{b:02x}"


def _channel_labels(meta: dict, channel_count: int, is_rgb: bool) -> list[str]:
    if is_rgb:
        return ["Red", "Green", "Blue"]

    candidates = meta.get("channel_names")
    if isinstance(candidates, list) and any(candidates):
        labels = [str(item) if item else f"Channel {idx + 1}" for idx, item in enumerate(candidates)]
    else:
        candidates = meta.get("filterblock")
        if isinstance(candidates, list) and any(candidates):
            labels = [str(item) if item else f"Channel {idx + 1}" for idx, item in enumerate(candidates)]
        else:
            labels = [f"Channel {idx + 1}" for idx in range(channel_count)]

    if len(labels) < channel_count:
        labels.extend(f"Channel {idx + 1}" for idx in range(len(labels), channel_count))
    return labels[:channel_count]


def _channel_colors(meta: dict, channel_count: int, is_rgb: bool) -> list[str]:
    if is_rgb:
        return ["ff0000", "00ff00", "0000ff"]

    lutnames = meta.get("lutname")
    if isinstance(lutnames, list) and lutnames:
        colors = [_color_to_hex(str(name or "white")) for name in lutnames]
    else:
        palette = ["ffffff", "00ff00", "ff00ff", "00ffff", "ffff00", "ff0000", "0000ff"]
        colors = [palette[idx % len(palette)] for idx in range(channel_count)]

    if len(colors) < channel_count:
        colors.extend("ffffff" for _ in range(channel_count - len(colors)))
    return colors[:channel_count]


def _channel_windows(meta: dict, channel_count: int, dtype: np.dtype) -> list[dict]:
    max_value = int(np.iinfo(dtype).max)
    stats = {}
    try:
        stats = compute_channel_intensity_stats(meta, sample_fraction=0.05, use_memmap=True)
    except Exception:
        stats = {}

    mins = stats.get("channel_min", []) if isinstance(stats, dict) else []
    maxs = stats.get("channel_max", []) if isinstance(stats, dict) else []
    windows = []
    for idx in range(channel_count):
        start = int(mins[idx]) if idx < len(mins) else 0
        end = int(maxs[idx]) if idx < len(maxs) else max_value
        if end <= start:
            end = max_value
        windows.append({"min": max(0, start), "max": min(max_value, end), "start": max(0, start), "end": min(max_value, end)})
    return windows


def _axes_definition() -> list[dict]:
    return [
        {"name": "t", "type": "time", "unit": "second"},
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]


def _build_omero_attrs(meta: dict, channel_count: int, dtype: np.dtype, is_rgb: bool) -> dict:
    labels = _channel_labels(meta, channel_count, is_rgb)
    colors = _channel_colors(meta, channel_count, is_rgb)
    windows = _channel_windows(meta, channel_count, dtype)
    channels = []
    for idx in range(channel_count):
        channels.append(
            {
                "label": labels[idx],
                "color": colors[idx],
                "active": True,
                "window": windows[idx],
            }
        )
    return {
        "id": 1,
        "name": meta.get("save_child_name") or meta.get("name") or "Leica image",
        "version": "0.5",
        "channels": channels,
        "rdefs": {"model": "color" if channel_count > 1 else "greyscale"},
    }


def _axis_scale_um(meta: dict, normalized_key: str, raw_key: str, fallback: float) -> float:
    normalized = meta.get(normalized_key)
    try:
        normalized_value = float(normalized)
        if normalized_value > 0:
            return normalized_value
    except Exception:
        pass

    raw_value = meta.get(raw_key)
    try:
        raw_float = float(raw_value)
        if raw_float > 0:
            return raw_float * 1_000_000.0
    except Exception:
        pass

    return float(fallback)


def _dataset_scales(level_count: int, meta: dict) -> list[list[float]]:
    time_scale = float(meta.get("tres") or meta.get("timeres") or 1.0)
    x_scale = _axis_scale_um(meta, "xres2", "xres", 1.0)
    y_scale = _axis_scale_um(meta, "yres2", "yres", x_scale)
    z_scale = _axis_scale_um(meta, "zres2", "zres", min(x_scale, y_scale))
    scales = []
    for level in range(level_count):
        factor = float(2 ** level)
        scales.append([time_scale, 1.0, z_scale, y_scale * factor, x_scale * factor])
    return scales


def _multiscales_attrs(meta: dict, level_count: int) -> list[dict]:
    datasets = []
    for idx, scale in enumerate(_dataset_scales(level_count, meta)):
        datasets.append(
            {
                "path": str(idx),
                "coordinateTransformations": [
                    {"type": "scale", "scale": scale}
                ],
            }
        )
    return [
        {
            "version": "0.5",
            "name": meta.get("save_child_name") or meta.get("name") or "Leica image",
            "axes": _axes_definition(),
            "datasets": datasets,
        }
    ]


def _pixel_type_and_bits(meta: dict, channel_count: int) -> tuple[str, int]:
    resolutions = meta.get("channelResolution", [16])
    if not isinstance(resolutions, list) or not resolutions or not all(isinstance(item, int) for item in resolutions):
        resolutions = [16] * max(1, channel_count)
    significant_bits = max(resolutions) if resolutions else 16
    if significant_bits > 8:
        return "uint16", significant_bits
    return "uint8", significant_bits


def _ome_xml_physical_sizes(meta: dict) -> tuple[float, float, float]:
    x_scale = _axis_scale_um(meta, "xres2", "xres", 1.0)
    y_scale = _axis_scale_um(meta, "yres2", "yres", x_scale)
    z_scale = _axis_scale_um(meta, "zres2", "zres", 1.0)
    return x_scale, y_scale, z_scale


def _generate_ome_xml_companion(meta: dict, output_name: str, is_rgb: bool, include_original_metadata: bool) -> str:
    channel_count = 3 if is_rgb else int(meta.get("channels", 1))
    pixel_type, significant_bits = _pixel_type_and_bits(meta, channel_count)
    physical_x, physical_y, physical_z = _ome_xml_physical_sizes(meta)
    acquisition_date = meta.get("experiment_datetime") or meta.get("experiment_datetime_str") or "Unknown"
    image_name = meta.get("save_child_name") or meta.get("name") or os.path.splitext(output_name)[0]
    image_name = escape_xml_chars(str(image_name))
    size_x = int(meta.get("xs", 1))
    size_y = int(meta.get("ys", 1))
    size_z = int(meta.get("zs", 1))
    size_t = int(meta.get("ts", 1))
    objective_name = escape_xml_chars(str(meta.get("objective") or "Unknown Objective"))
    microscope_model = escape_xml_chars(
        f"{meta.get('SystemTypeName', '')} {meta.get('MicroscopeModel', '')}".strip()
    )
    ome_uuid = uuid.uuid4()
    channel_labels = _channel_labels(meta, channel_count, is_rgb)
    excitations = meta.get("excitation") if isinstance(meta.get("excitation"), list) else []
    emissions = meta.get("emission") if isinstance(meta.get("emission"), list) else []

    xml = [
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
        "<OME xmlns=\"http://www.openmicroscopy.org/Schemas/OME/2016-06\"",
        "     xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"",
        "     xsi:schemaLocation=\"http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd\"",
        f"     UUID=\"urn:uuid:{ome_uuid}\">",
        "  <Instrument ID=\"Instrument:0\">",
        f"    <Microscope ID=\"Microscope:0\" Manufacturer=\"Leica\" Model=\"{microscope_model}\"/>",
        f"    <Objective ID=\"Objective:0\" Description=\"{objective_name}\"/>",
        "  </Instrument>",
        f"  <Image ID=\"Image:0\" Name=\"{image_name}\">",
        f"    <AcquisitionDate>{escape_xml_chars(str(acquisition_date))}</AcquisitionDate>",
        "    <InstrumentRef ID=\"Instrument:0\"/>",
        "    <ObjectiveSettings ID=\"Objective:0\"/>",
        f"    <Pixels ID=\"Pixels:0\" DimensionOrder=\"XYZCT\" Type=\"{pixel_type}\" SignificantBits=\"{significant_bits}\" SizeX=\"{size_x}\" SizeY=\"{size_y}\" SizeZ=\"{size_z}\" SizeC=\"{channel_count}\" SizeT=\"{size_t}\" PhysicalSizeX=\"{physical_x}\" PhysicalSizeY=\"{physical_y}\" PhysicalSizeZ=\"{physical_z}\" PhysicalSizeXUnit=\"µm\" PhysicalSizeYUnit=\"µm\" PhysicalSizeZUnit=\"µm\" Interleaved=\"false\">",
    ]

    for channel_index, channel_label in enumerate(channel_labels):
        channel_attrs = [
            f"ID=\"Channel:0:{channel_index}\"",
            f"Name=\"{escape_xml_chars(str(channel_label))}\"",
            'SamplesPerPixel=\"1\"',
        ]
        if channel_index < len(excitations) and isinstance(excitations[channel_index], (int, float)) and excitations[channel_index] > 0:
            channel_attrs.append(f"ExcitationWavelength=\"{float(excitations[channel_index])}\"")
            channel_attrs.append('ExcitationWavelengthUnit=\"nm\"')
        if channel_index < len(emissions) and isinstance(emissions[channel_index], (int, float)) and emissions[channel_index] > 0:
            channel_attrs.append(f"EmissionWavelength=\"{float(emissions[channel_index])}\"")
            channel_attrs.append('EmissionWavelengthUnit=\"nm\"')
        xml.append(f"      <Channel {' '.join(channel_attrs)}/>")

    if include_original_metadata and meta.get("xmlElement"):
        annotation_value = escape_xml_chars(str(meta.get("xmlElement")))
        xml.extend([
            "    </Pixels>",
            "    <StructuredAnnotations>",
            "      <XMLAnnotation ID=\"Annotation:0\">",
            "        <Value>",
            f"          {annotation_value}",
            "        </Value>",
            "      </XMLAnnotation>",
            "    </StructuredAnnotations>",
            "  </Image>",
            "</OME>",
        ])
        return "\n".join(xml)

    xml.extend([
        "    </Pixels>",
        "  </Image>",
        "</OME>",
    ])
    return "\n".join(xml)


def _write_ome_xml_companion(out_path: str, meta: dict, output_name: str, is_rgb: bool, include_original_metadata: bool) -> None:
    ome_dir = os.path.join(out_path, "OME")
    os.makedirs(ome_dir, exist_ok=True)
    ome_xml_path = os.path.join(ome_dir, "METADATA.ome.xml")
    with open(ome_xml_path, "w", encoding="utf-8") as fh:
        fh.write(_generate_ome_xml_companion(meta, output_name, is_rgb, include_original_metadata))


def _read_rows(
    base_file: str,
    base_pos: int,
    xs: int,
    row_start: int,
    row_end: int,
    skip: int,
    channel: int,
    bits: int,
    zbytes: int,
    cbytesinc: list[int],
    zs: int = 1,
    *,
    target_z: int | None = None,
    timepoint: int = 0,
    tbytes: int = 0,
    ts: int = 1,
) -> np.ndarray:
    dtype = np.uint16 if bits == 16 else np.uint8
    bpp = bits // 8
    out_rows = (row_end - row_start + skip - 1) // skip
    arr = np.zeros((out_rows, xs), dtype=dtype)
    line_bytes = xs * bpp

    with open(base_file, "rb") as fh:
        ch_offset = cbytesinc[channel]
        idx = 0
        z = 0 if target_z is None else int(target_z)
        plane_start_offset = timepoint * tbytes + z * zbytes + ch_offset
        plane_start_pos = base_pos + plane_start_offset

        for row_idx in range(row_start, row_end, skip):
            pos = plane_start_pos + row_idx * line_bytes
            fh.seek(pos)
            row = fh.read(line_bytes)
            if len(row) < line_bytes:
                read_count = len(row) // bpp
                if read_count > 0:
                    arr[idx, :read_count] = np.frombuffer(row, dtype=dtype, count=read_count)
            else:
                arr[idx] = np.frombuffer(row, dtype=dtype, count=xs)
            idx += 1
            if idx >= out_rows:
                break
    return arr


def _read_interleaved_rgb_plane(
    base_file: str,
    base_pos: int,
    xs: int,
    ys: int,
    bits: int,
    zbytes: int,
    tbytes: int,
    target_z: int,
    timepoint: int,
    zs: int,
    ts: int,
) -> np.ndarray:
    dtype = np.uint16 if bits == 16 else np.uint8
    bpp = bits // 8
    plane_bytes = xs * ys * 3 * bpp
    pos = base_pos + timepoint * tbytes + target_z * zbytes

    with open(base_file, "rb") as fh:
        fh.seek(pos)
        plane_data = fh.read(plane_bytes)

    if len(plane_data) < plane_bytes:
        buffer = bytearray(plane_bytes)
        buffer[:len(plane_data)] = plane_data
        plane_data = buffer

    return np.frombuffer(plane_data, dtype=dtype).reshape((ys, xs, 3))


def _create_level_shapes(ts: int, channel_count: int, zs: int, ys: int, xs: int) -> list[tuple[int, int, int, int, int]]:
    shapes = [(ts, channel_count, zs, ys, xs)]
    current_y = ys
    current_x = xs
    while max(current_y, current_x) > 512:
        current_y = max(1, math.ceil(current_y / 2))
        current_x = max(1, math.ceil(current_x / 2))
        shapes.append((ts, channel_count, zs, current_y, current_x))
    return shapes


def _chunk_shape(shape: tuple[int, int, int, int, int]) -> tuple[int, int, int, int, int]:
    return (1, 1, 1, min(512, shape[-2]), min(512, shape[-1]))


def _resize_plane(src_plane: np.ndarray, out_height: int, out_width: int) -> np.ndarray:
    if src_plane.shape[0] == out_height and src_plane.shape[1] == out_width:
        return np.asarray(src_plane)
    resized = cv2.resize(src_plane, (out_width, out_height), interpolation=cv2.INTER_AREA)
    if resized.dtype != src_plane.dtype:
        resized = resized.astype(src_plane.dtype, copy=False)
    return resized


def _write_pyramids(arrays: list, show_progress: bool, prefix: str) -> None:
    if len(arrays) <= 1:
        return

    level_count = len(arrays)
    total_chunks = 0
    for level_idx in range(1, level_count):
        dest = arrays[level_idx]
        chunk_y = dest.chunks[-2]
        chunk_x = dest.chunks[-1]
        total_chunks += dest.shape[0] * dest.shape[1] * dest.shape[2] * math.ceil(dest.shape[-2] / chunk_y) * math.ceil(dest.shape[-1] / chunk_x)

    processed = 0
    for level_idx in range(1, level_count):
        src = arrays[level_idx - 1]
        dest = arrays[level_idx]
        chunk_y = dest.chunks[-2]
        chunk_x = dest.chunks[-1]
        for t_idx in range(dest.shape[0]):
            for c_idx in range(dest.shape[1]):
                for z_idx in range(dest.shape[2]):
                    for y0 in range(0, dest.shape[-2], chunk_y):
                        out_h = min(chunk_y, dest.shape[-2] - y0)
                        src_y0 = y0 * 2
                        src_y1 = min(src.shape[-2], (y0 + out_h) * 2)
                        for x0 in range(0, dest.shape[-1], chunk_x):
                            out_w = min(chunk_x, dest.shape[-1] - x0)
                            src_x0 = x0 * 2
                            src_x1 = min(src.shape[-1], (x0 + out_w) * 2)
                            src_plane = np.asarray(src[t_idx, c_idx, z_idx, src_y0:src_y1, src_x0:src_x1])
                            dest[t_idx, c_idx, z_idx, y0:y0 + out_h, x0:x0 + out_w] = _resize_plane(src_plane, out_h, out_w)
                            processed += 1
                            if show_progress and (processed % 16 == 0 or processed == total_chunks):
                                progress = 85.0 + (processed / max(1, total_chunks)) * 15.0
                                print_progress_bar(progress, prefix=prefix, suffix=f"Building pyramid L{level_idx}/{level_count - 1}")


def _copytree_overwrite(src_path: str, dest_root: str, show_progress: bool) -> str:
    os.makedirs(dest_root, exist_ok=True)
    dest_path = os.path.join(dest_root, os.path.basename(src_path))
    if os.path.isdir(dest_path):
        shutil.rmtree(dest_path)
    if show_progress:
        print_progress_bar(0, prefix="Copying output:", suffix=os.path.basename(src_path))
    shutil.copytree(src_path, dest_path)
    if show_progress:
        print_progress_bar(100, prefix="Copying output:", suffix="Copy complete", final_call=True)
    return dest_path


def _write_group_attrs(group, meta: dict, dtype: np.dtype, level_count: int, is_rgb: bool) -> None:
    channel_count = 3 if is_rgb else int(meta.get("channels", 1))
    metadata_copy = dict(meta)
    metadata_copy.pop("xmlElement", None)
    x_scale = _axis_scale_um(meta, "xres2", "xres", 1.0)
    y_scale = _axis_scale_um(meta, "yres2", "yres", x_scale)
    z_scale = _axis_scale_um(meta, "zres2", "zres", min(x_scale, y_scale))
    group.attrs["multiscales"] = _multiscales_attrs(meta, level_count)
    group.attrs["omero"] = _build_omero_attrs(meta, channel_count, dtype, is_rgb)
    group.attrs["leica"] = {
        "source_filetype": meta.get("filetype"),
        "image_uuid": meta.get("uuid") or meta.get("UniqueID"),
        "save_child_name": meta.get("save_child_name"),
        "pixel_size_x_um": x_scale,
        "pixel_size_y_um": y_scale,
        "pixel_size_z_um": z_scale,
        "metadata": _json_safe(metadata_copy),
    }


def _create_zarr_arrays(group, shapes: list[tuple[int, int, int, int, int]], dtype: np.dtype):
    _, Blosc = _import_zarr_modules()
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
    arrays = []
    axis_names = ["t", "c", "z", "y", "x"]
    for level_idx, shape in enumerate(shapes):
        array = group.create_array(
            str(level_idx),
            shape=shape,
            chunks=_chunk_shape(shape),
            dtype=np.dtype(dtype),
            compressor=compressor,
            fill_value=0,
            overwrite=True,
        )
        array.attrs["_ARRAY_DIMENSIONS"] = axis_names
        arrays.append(array)
    return arrays


def _prepare_output_path(outputfolder: str, output_name: str) -> str:
    out_path = os.path.join(outputfolder, output_name)
    if os.path.isdir(out_path):
        shutil.rmtree(out_path)
    elif os.path.exists(out_path):
        os.remove(out_path)
    return out_path


def _open_output_group(out_path: str):
    zarr, _ = _import_zarr_modules()
    return zarr.open_group(store=out_path, mode="w", zarr_format=2)


def _write_non_rgb_level0(arr0, meta: dict, base_file: str, base_pos: int, bits: int, show_progress: bool) -> None:
    xs_orig = int(meta["xs"])
    ys_orig = int(meta["ys"])
    zs = int(meta.get("zs", 1))
    channels = int(meta["channels"])
    ts = int(meta.get("ts", 1))
    tiles = int(meta.get("tiles", 1))
    tile_positions = meta.get("tile_positions", []) or []
    is_tilescan = tiles > 1 and bool(tile_positions)
    overlap_x = float(meta.get("OverlapPercentageX", 0.0) or 0.0)
    overlap_y = float(meta.get("OverlapPercentageY", 0.0) or 0.0)
    do_swapxy = meta.get("swapxy", 0)
    cbytesinc = meta.get("channelbytesinc")
    zbytesinc = meta.get("zbytesinc")
    tbytesinc = meta.get("tbytesinc")
    tile_width = xs_orig
    tile_height = ys_orig
    planes_total = channels * zs * ts
    progress_per_plane = 80.0 / max(1, planes_total)
    plane_idx = 0

    if show_progress:
        print_progress_bar(5, prefix="Converting to OME-Zarr:", suffix="Reading raw data")

    for t_idx in range(ts):
        for c_idx in range(channels):
            for z_idx in range(zs):
                curr_progress = 5.0 + plane_idx * progress_per_plane
                suffix = f"T={t_idx + 1}/{ts} C={c_idx + 1}/{channels} Z={z_idx + 1}/{zs}"
                if is_tilescan:
                    num_tiles = len(tile_positions)
                    update_interval = 1 if num_tiles <= 20 else math.ceil(num_tiles / 20.0)
                    for pos_idx, pos_data in enumerate(tile_positions):
                        tile_num = pos_data.get("num")
                        if tile_num is None:
                            continue
                        tile_off = (tile_num - 1) * meta.get("tilesbytesinc", 0)
                        base_pos_tile = base_pos + tile_off
                        read_w, read_h = tile_width, tile_height
                        if do_swapxy:
                            read_w, read_h = tile_height, tile_width
                        slab = _read_rows(
                            base_file=base_file,
                            base_pos=base_pos_tile,
                            xs=read_w,
                            row_start=0,
                            row_end=read_h,
                            skip=1,
                            channel=c_idx,
                            bits=bits,
                            zbytes=zbytesinc,
                            cbytesinc=cbytesinc,
                            zs=zs,
                            target_z=z_idx,
                            timepoint=t_idx,
                            tbytes=tbytesinc,
                            ts=ts,
                        )
                        tile_do_flipx = meta.get("tilescan_flipx", 0)
                        tile_do_flipy = meta.get("tilescan_flipy", 0)
                        tile_do_swapxy = meta.get("tilescan_swapxy", 0)
                        if tile_do_swapxy:
                            if tile_do_flipy:
                                slab = slab[::-1, :]
                            if tile_do_flipx:
                                slab = slab[:, ::-1]
                            slab = slab.T
                        else:
                            if tile_do_flipy:
                                slab = slab[::-1, :]
                            if tile_do_flipx:
                                slab = slab[:, ::-1]

                        current_step_x = tile_width * (1.0 - overlap_x)
                        current_step_y = tile_height * (1.0 - overlap_y)
                        xstart = int(pos_data.get("FieldX", 0) * current_step_x)
                        ystart = int(pos_data.get("FieldY", 0) * current_step_y)
                        xend = min(arr0.shape[-1], xstart + tile_width)
                        yend = min(arr0.shape[-2], ystart + tile_height)
                        if ystart < arr0.shape[-2] and xstart < arr0.shape[-1]:
                            arr0[t_idx, c_idx, z_idx, ystart:yend, xstart:xend] = slab[:yend - ystart, :xend - xstart]

                        if show_progress and ((pos_idx + 1) % update_interval == 0 or (pos_idx + 1) == num_tiles):
                            tile_progress = curr_progress + ((pos_idx + 1) / max(1, num_tiles)) * progress_per_plane
                            print_progress_bar(tile_progress, prefix="Converting to OME-Zarr:", suffix=f"{suffix} Tile={pos_idx + 1}/{num_tiles}")
                else:
                    slab = _read_rows(
                        base_file=base_file,
                        base_pos=base_pos,
                        xs=xs_orig,
                        row_start=0,
                        row_end=ys_orig,
                        skip=1,
                        channel=c_idx,
                        bits=bits,
                        zbytes=zbytesinc,
                        cbytesinc=cbytesinc,
                        zs=zs,
                        target_z=z_idx,
                        timepoint=t_idx,
                        tbytes=tbytesinc,
                        ts=ts,
                    )
                    arr0[t_idx, c_idx, z_idx, :ys_orig, :xs_orig] = slab
                    if show_progress:
                        print_progress_bar(curr_progress + progress_per_plane, prefix="Converting to OME-Zarr:", suffix=f"Finished {suffix}")
                plane_idx += 1


def _write_rgb_level0(arr0, meta: dict, base_file: str, base_pos: int, bits: int, show_progress: bool) -> None:
    xs = int(meta["xs"])
    ys = int(meta["ys"])
    zs = int(meta.get("zs", 1))
    ts = int(meta.get("ts", 1))
    tiles = int(meta.get("tiles", 1))
    tile_positions = meta.get("tile_positions", []) or []
    is_tilescan = tiles > 1 and bool(tile_positions)
    overlap_x = float(meta.get("OverlapPercentageX", 0.0) or 0.0)
    overlap_y = float(meta.get("OverlapPercentageY", 0.0) or 0.0)
    zbytesinc = meta.get("zbytesinc")
    tbytesinc = meta.get("tbytesinc")
    tilesbytesinc = meta.get("tilesbytesinc")
    tile_width = xs if not is_tilescan else int(meta.get("tile_width", meta["xs"]))
    tile_height = ys if not is_tilescan else int(meta.get("tile_height", meta["ys"]))
    planes_total = zs * ts
    progress_per_plane = 80.0 / max(1, planes_total)
    plane_idx = 0

    if show_progress:
        print_progress_bar(5, prefix="Converting to OME-Zarr:", suffix="Reading raw RGB data")

    for t_idx in range(ts):
        for z_idx in range(zs):
            curr_progress = 5.0 + plane_idx * progress_per_plane
            suffix = f"T={t_idx + 1}/{ts} Z={z_idx + 1}/{zs}"
            if is_tilescan:
                num_tiles = len(tile_positions)
                update_interval = 1 if num_tiles <= 20 else math.ceil(num_tiles / 20.0)
                for pos_idx, tile_info in enumerate(tile_positions):
                    tile_num = tile_info.get("num")
                    if tile_num is None or tile_num < 1:
                        continue
                    tile_offset = (tile_num - 1) * tilesbytesinc
                    base_pos_for_tile = base_pos + tile_offset
                    tile_plane_data = _read_interleaved_rgb_plane(
                        base_file=base_file,
                        base_pos=base_pos_for_tile,
                        xs=tile_width,
                        ys=tile_height,
                        bits=bits,
                        zbytes=zbytesinc,
                        tbytes=tbytesinc,
                        target_z=z_idx,
                        timepoint=t_idx,
                        zs=zs,
                        ts=ts,
                    )
                    channels_first = np.moveaxis(tile_plane_data, -1, 0)
                    xstart = int(tile_info.get("FieldX", 0) * (tile_width - tile_width * overlap_x))
                    ystart = int(tile_info.get("FieldY", 0) * (tile_height - tile_height * overlap_y))
                    xend = min(arr0.shape[-1], xstart + tile_width)
                    yend = min(arr0.shape[-2], ystart + tile_height)
                    if ystart < arr0.shape[-2] and xstart < arr0.shape[-1]:
                        arr0[t_idx, :, z_idx, ystart:yend, xstart:xend] = channels_first[:, :yend - ystart, :xend - xstart]
                    if show_progress and ((pos_idx + 1) % update_interval == 0 or (pos_idx + 1) == num_tiles):
                        tile_progress = curr_progress + ((pos_idx + 1) / max(1, num_tiles)) * progress_per_plane
                        print_progress_bar(tile_progress, prefix="Converting to OME-Zarr:", suffix=f"{suffix} Tile={pos_idx + 1}/{num_tiles}")
            else:
                plane_data = _read_interleaved_rgb_plane(
                    base_file=base_file,
                    base_pos=base_pos,
                    xs=xs,
                    ys=ys,
                    bits=bits,
                    zbytes=zbytesinc,
                    tbytes=tbytesinc,
                    target_z=z_idx,
                    timepoint=t_idx,
                    zs=zs,
                    ts=ts,
                )
                arr0[t_idx, :, z_idx, :ys, :xs] = np.moveaxis(plane_data, -1, 0)
                if show_progress:
                    print_progress_bar(curr_progress + progress_per_plane, prefix="Converting to OME-Zarr:", suffix=f"Finished {suffix}")
            plane_idx += 1


def _convert_leica_to_omezarr_impl(
    inputfile: str,
    *,
    image_uuid: str,
    outputfolder: str | None,
    show_progress: bool,
    altoutputfolder: str | None,
    include_original_metadata: bool,
    tempfolder: str | None,
    save_child_name: str | None,
    expect_rgb: bool,
) -> str | None:
    meta = read_image_metadata(inputfile, image_uuid)
    if bool(meta.get("isrgb")) != bool(expect_rgb):
        return None
    if meta.get("OverlapIsNegative"):
        print(f"Image UUID {image_uuid} overlap is Negative - skipping OME-Zarr conversion.")
        return None

    if outputfolder is None:
        outputfolder = os.path.dirname(inputfile)
    os.makedirs(outputfolder, exist_ok=True)
    if altoutputfolder:
        os.makedirs(altoutputfolder, exist_ok=True)

    xs_orig = int(meta["xs"])
    ys_orig = int(meta["ys"])
    tiles = int(meta.get("tiles", 1))
    tile_positions = meta.get("tile_positions", []) or []
    is_tilescan = tiles > 1 and bool(tile_positions)
    overlap_x = float(meta.get("OverlapPercentageX", 0.0) or 0.0)
    overlap_y = float(meta.get("OverlapPercentageY", 0.0) or 0.0)
    tile_width = xs_orig
    tile_height = ys_orig
    canvas_xs = xs_orig
    canvas_ys = ys_orig
    if is_tilescan:
        xlist = [pos.get("FieldX", 0) for pos in tile_positions]
        ylist = [pos.get("FieldY", 0) for pos in tile_positions]
        xdim = (max(xlist) + 1) if xlist else 0
        ydim = (max(ylist) + 1) if ylist else 0
        step_x = tile_width * (1.0 - overlap_x)
        step_y = tile_height * (1.0 - overlap_y)
        canvas_xs = int((xdim - 1) * step_x + tile_width) if xdim > 0 else tile_width
        canvas_ys = int((ydim - 1) * step_y + tile_height) if ydim > 0 else tile_height
        if expect_rgb:
            meta["tile_width"] = tile_width
            meta["tile_height"] = tile_height

    meta = dict(meta)
    meta["xs"] = canvas_xs
    meta["ys"] = canvas_ys
    if save_child_name:
        meta["save_child_name"] = save_child_name

    res = meta.get("channelResolution", [16])
    if not isinstance(res, list) or not res or not all(isinstance(item, int) for item in res):
        res = [16]
    bits = 16 if max(res) > 8 else 8
    dtype = np.uint16 if bits == 16 else np.uint8
    channel_count = 3 if expect_rgb else int(meta.get("channels", 1))
    ts = int(meta.get("ts", 1))
    zs = int(meta.get("zs", 1))
    level_shapes = _create_level_shapes(ts, channel_count, zs, canvas_ys, canvas_xs)

    temp_source_cleanup = None
    try:
        base_file, base_pos, temp_source_cleanup = prepare_temp_source(
            inputfile=inputfile,
            image_uuid=image_uuid,
            metadata=meta,
            tempfolder=tempfolder,
            show_progress=show_progress,
        )

        effective_save_child_name = save_child_name if save_child_name else meta.get("save_child_name", f"omezarr_output_{image_uuid}")
        out_name = _ome_zarr_name(effective_save_child_name)
        out_path = _prepare_output_path(outputfolder, out_name)
        group = _open_output_group(out_path)
        arrays = _create_zarr_arrays(group, level_shapes, dtype)
        _write_group_attrs(group, meta, dtype, len(level_shapes), expect_rgb)
        _write_ome_xml_companion(out_path, meta, out_name, expect_rgb, include_original_metadata)
        if include_original_metadata and meta.get("xmlElement"):
            leica_attrs = dict(group.attrs["leica"])
            leica_attrs["original_xml"] = str(meta.get("xmlElement"))
            group.attrs["leica"] = leica_attrs

        if expect_rgb:
            _write_rgb_level0(arrays[0], meta, base_file, base_pos, bits, show_progress)
        else:
            _write_non_rgb_level0(arrays[0], meta, base_file, base_pos, bits, show_progress)

        if show_progress and len(arrays) > 1:
            print_progress_bar(85, prefix="Converting to OME-Zarr:", suffix="Building pyramid levels")
        _write_pyramids(arrays, show_progress, "Converting to OME-Zarr:")
        if show_progress and len(arrays) <= 1:
            print_progress_bar(100, prefix="Converting to OME-Zarr:", suffix="Processing complete", final_call=True)
        elif show_progress:
            print_progress_bar(100, prefix="Converting to OME-Zarr:", suffix="Processing complete", final_call=True)

        if altoutputfolder:
            _copytree_overwrite(out_path, altoutputfolder, show_progress)

        return out_name
    finally:
        gc.collect()
        cleanup_temp_source(temp_source_cleanup, show_progress=show_progress)


def convert_leica_to_omezarr(
    inputfile: str,
    *,
    image_uuid: str = "n/a",
    outputfolder: str | None = None,
    show_progress: bool = True,
    altoutputfolder: str | None = None,
    include_original_metadata: bool = False,
    tempfolder: str | None = None,
    save_child_name: str | None = None,
) -> str | None:
    return _convert_leica_to_omezarr_impl(
        inputfile,
        image_uuid=image_uuid,
        outputfolder=outputfolder,
        show_progress=show_progress,
        altoutputfolder=altoutputfolder,
        include_original_metadata=include_original_metadata,
        tempfolder=tempfolder,
        save_child_name=save_child_name,
        expect_rgb=False,
    )


def convert_leica_rgb_to_omezarr(
    inputfile: str,
    *,
    image_uuid: str = "n/a",
    outputfolder: str | None = None,
    show_progress: bool = True,
    altoutputfolder: str | None = None,
    include_original_metadata: bool = False,
    tempfolder: str | None = None,
    save_child_name: str | None = None,
) -> str | None:
    return _convert_leica_to_omezarr_impl(
        inputfile,
        image_uuid=image_uuid,
        outputfolder=outputfolder,
        show_progress=show_progress,
        altoutputfolder=altoutputfolder,
        include_original_metadata=include_original_metadata,
        tempfolder=tempfolder,
        save_child_name=save_child_name,
        expect_rgb=True,
    )