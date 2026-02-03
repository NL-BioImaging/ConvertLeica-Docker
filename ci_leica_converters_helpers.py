"""
Helpers for reading Leica LIF/LOF/XLEF metadata and pixels and for producing
derived values used by the ConvertLeica tools (CLI, GUI, and server).

Overview
--------
- File readers: read_leica_file(...), read_image_metadata(...)
- Metadata helpers: get_image_metadata(...), get_image_metadata_LOF(...)
- Intensity stats: compute_channel_intensity_stats(...) — fast, approximate
    per-channel min/max using subsampling and numpy.memmap; understands Leica
    planar-versus-interleaved layouts and byte offsets
    (channelbytesinc/zbytesinc/tbytesinc/tilesbytesinc).
- UI: print_progress_bar(...)
- Colors: decimal_to_rgb(...), color_name_to_decimal(...), decimal_to_ome_color(...)
- OME schema: parse_ome_xsd(...), validate_metadata(...)

Supported file types
--------------------
.lif, .lof, .xlef (with linked .xlcf/.xlif). The module works when imported as
part of a package (relative imports) or directly from a folder (fallback imports).

Expected metadata keys
----------------------
Many helpers accept a Leica image metadata dict produced by ReadLeica* modules.
Commonly used keys include:
- xs, ys, zs, ts, tiles: dimensions
- isrgb (bool), channels (int), channelResolution (per-channel bit depth)
- channelbytesinc, zbytesinc, tbytesinc, tilesbytesinc: byte offsets
- LIFFile, LOFFilePath, Position, filetype
- blackvalue, whitevalue: viewer display levels (either 0..1 floats or container-range ints)

Error handling
--------------
- read_leica_file/read_image_metadata: raise ValueError for unsupported types
    or unknown image UUIDs.
- compute_channel_intensity_stats: returns display-only defaults when raw data
    cannot be read (missing files, offsets, or memmap failures).

Networking note
---------------
On import, the OME-XML schema is downloaded (parse_ome_xsd) to build
`metadata_schema` used by validate_metadata for normalizing certain string
fields. If offline, callers can ignore `metadata_schema` and skip validation.
"""

import os
import json
import tempfile
import shutil
import time
import uuid as uuid_module
import numpy as np
import xml.etree.ElementTree as ET
import urllib.request
import math
from typing import Dict, List, Tuple

try:
    # Package context (e.g., inside omero_biomero.leica_file_browser)
    from .ReadLeicaLIF import read_leica_lif
    from .ReadLeicaLOF import read_leica_lof
    from .ReadLeicaXLEF import read_leica_xlef
except ImportError:  # pragma: no cover - fallback for script usage
    # Script context (running from a plain folder)
    from ReadLeicaLIF import read_leica_lif
    from ReadLeicaLOF import read_leica_lof
    from ReadLeicaXLEF import read_leica_xlef

dtype_to_format = {
    np.uint8: "uchar",
    np.int8: "char",
    np.uint16: "ushort",
    np.int16: "short",
    np.uint32: "uint",
    np.int32: "int",
    np.float32: "float",
    np.float64: "double",
    np.complex64: "complex",
    np.complex128: "dpcomplex",
}


def read_leica_file(file_path, include_xmlelement=False, image_uuid=None, folder_uuid=None):
    """
    Read Leica LIF, XLEF, or LOF file.

    Parameters:
    - file_path: path to the LIF, XLEF, or LOF file
    - include_xmlelement: whether to include the XML element in the lifinfo dictionary
    - image_uuid: optional UUID of an image
    - folder_uuid: optional UUID of a folder/collection

    Returns:
    - If image_uuid is provided:
        - Returns the lifinfo dictionary for the matching image, including detailed metadata.
    - Else if folder_uuid is provided:
        - Returns a single-level XML tree (as a string) of that folder (its immediate children only).
    - Else (no image_uuid or folder_uuid):
        - Returns a single-level XML tree (as a string) of the root/top-level folder(s) or items.
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == '.lif':
        return read_leica_lif(file_path, include_xmlelement, image_uuid, folder_uuid)
    elif ext == '.xlef':
        return read_leica_xlef(file_path, folder_uuid)
    elif ext == '.lof':
        return read_leica_lof(file_path, include_xmlelement)
    else:
        raise ValueError('Unsupported file type: {}'.format(ext))


def get_image_metadata_LOF(folder_metadata, image_uuid):
    folder_metadata_dict = json.loads(folder_metadata)
    image_metadata_dict = next((img for img in folder_metadata_dict["children"] if img["uuid"] == image_uuid), None)
    image_metadata = read_leica_file(image_metadata_dict['lof_file_path'])
    return image_metadata


def get_image_metadata(folder_metadata, image_uuid):
    folder_metadata_dict = json.loads(folder_metadata)
    image_metadata_dict = next((img for img in folder_metadata_dict["children"] if img["uuid"] == image_uuid), None)
    image_metadata = json.dumps(image_metadata_dict, indent=2)
    return image_metadata


def _as_int_list(value, length: int, default: int) -> List[int]:
    """Normalize metadata values to a per-channel integer list of a given length."""
    if isinstance(value, list):
        arr = []
        for i in range(length):
            v = value[i] if i < len(value) else None
            if v is None:
                arr.append(int(default))
            else:
                try:
                    arr.append(int(round(float(v))))
                except Exception:
                    arr.append(int(default))
        return arr
    try:
        v = int(round(float(value)))
    except Exception:
        v = int(default)
    return [v for _ in range(length)]


def _resolve_bits_per_channel(meta: dict, channels: int, isrgb: bool) -> List[int]:
    """Return bits per channel list, using channelResolution when available, else default heuristics."""
    res = meta.get("channelResolution")
    if isinstance(res, list) and res:
        bits_list = []
        fallback = 16
        for i in range(channels):
            b = res[i] if i < len(res) else None
            try:
                b = int(b) if b is not None else fallback
            except Exception:
                b = fallback
            if b not in (8, 12, 14, 15, 16, 32):
                # Container will be 8 or 16 for our raw stream; clamp to typical values
                b = 8 if b and b <= 8 else 16
            bits_list.append(b)
        return bits_list
    if isinstance(res, (int, float)):
        b = int(res)
        b = 8 if b and b <= 8 else 16
        return [b] * channels
    # Heuristic when missing
    if isrgb:
        # Most Leica RGB exports are 8-bit; fall back to 16 if metadata indicates higher container
        return [8, 8, 8]
    return [16] * channels


def _dtype_from_bits(bits: int):
    return (np.uint8, 1, 255) if bits <= 8 else (np.uint16, 2, 65535)


def compute_channel_intensity_stats(metadata: dict, sample_fraction: float = 0.1, use_memmap: bool = True) -> Dict[str, List[int]]:
    """
    Fast approximate per-channel intensity stats using subsampling and memmap.

    Strategy:
    - Choose center Z, center T, center tile (if present) to avoid reading full volumes/mosaics.
    - Subsample rows with a stride computed from sample_fraction (e.g., 0.1 -> every 10th row).
    - For RGB: read a single interleaved slice (ys, xs, 3) and compute per-channel min/max.
    - For multi-channel: read per-channel planar slices using channelbytesinc offsets.

    Returns dict with keys:
      - channel_mins: int list per channel (length 3 for RGB)
      - channel_maxs: int list per channel
      - channel_display_black_values: scaled to container (int) per channel
      - channel_display_white_values: scaled to container (int) per channel
    """
    # Ensure dict input
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dict (use read_image_metadata first)")

    filetype = metadata.get("filetype")
    if filetype not in (".lif", ".xlef", ".lof"):
        return {
            "channel_mins": [],
            "channel_maxs": [],
            "channel_display_black_values": [],
            "channel_display_white_values": [],
        }

    # Determine file name and base offset similar to CreatePreview
    if filetype == ".lif":
        file_name = metadata.get("LIFFile") or metadata.get("LOFFilePath")
        base_pos = int(metadata.get("Position", 0) or 0)
    else:  # .xlef / .lof images read from LOF
        file_name = metadata.get("LOFFilePath")
        base_pos = 62  # LOF header size used in codebase

    if not file_name or not os.path.exists(file_name):
        # Can't read pixels; just scale display min/max if available
        return _fallback_only_display(metadata)

    xs = int(metadata.get("xs", 1) or 1)
    ys = int(metadata.get("ys", 1) or 1)
    zs = int(metadata.get("zs", 1) or 1)
    ts = int(metadata.get("ts", 1) or 1)
    tiles = int(metadata.get("tiles", 1) or 1)
    isrgb = bool(metadata.get("isrgb", False))

    channels = 3 if isrgb else int(metadata.get("channels", 1) or 1)
    channelbytesinc = metadata.get("channelbytesinc") or [0] * channels
    zbytesinc = int(metadata.get("zbytesinc", 0) or 0)
    tbytesinc = int(metadata.get("tbytesinc", 0) or 0)
    tilesbytesinc = int(metadata.get("tilesbytesinc", 0) or 0)

    # Select center t, tile, and z
    t_sel = ts // 2 if ts and ts > 1 else 0
    s_sel = tiles // 2 if tiles and tiles > 1 else 0
    z_sel = zs // 2 if zs and zs > 1 else 0

    # Adjust base position
    base = base_pos + (t_sel * tbytesinc) + (s_sel * tilesbytesinc) + (z_sel * zbytesinc)

    # Resolve bits per channel and container dtype
    bits_per_ch = _resolve_bits_per_channel(metadata, channels, isrgb)
    # Use first channel bit depth to decide container dtype for reading (stream container)
    dtype, bpp, container_max_val = _dtype_from_bits(bits_per_ch[0])

    # Determine sampling stride
    if sample_fraction <= 0 or sample_fraction > 1:
        sample_fraction = 0.1
    step = max(1, int(round(1.0 / sample_fraction)))

    ch_mins: List[int] = []
    ch_maxs: List[int] = []

    try:
        if isrgb:
            # Interleaved RGB slice (ys, xs, 3)
            shape = (ys, xs, 3)
            if use_memmap:
                arr = np.memmap(file_name, dtype=dtype, mode="r", offset=base, shape=shape, order="C")
                sample = arr[::step, :, :]
            else:
                # Fallback: read in row strides
                sample = _read_rows_strided(file_name, base, ys, xs, 3, bpp, step, dtype)
            # Compute per-channel min/max
            ch_mins = sample.reshape(-1, 3).min(axis=0).astype(int).tolist()
            ch_maxs = sample.reshape(-1, 3).max(axis=0).astype(int).tolist()
        else:
            # Planar per-channel slices
            for c in range(channels):
                c_off = base + int(channelbytesinc[c] if c < len(channelbytesinc) and channelbytesinc[c] is not None else 0)
                shape = (ys, xs)
                if use_memmap:
                    arr = np.memmap(file_name, dtype=dtype, mode="r", offset=c_off, shape=shape, order="C")
                    sample = arr[::step, :]
                else:
                    sample = _read_rows_strided(file_name, c_off, ys, xs, 1, bpp, step, dtype)
                ch_mins.append(int(sample.min()))
                ch_maxs.append(int(sample.max()))
    except Exception:
        # If anything fails, fall back to display-only values
        return _fallback_only_display(metadata)

    # Display black/white values scaled to significant bits (channelResolution) per channel
    channel_resolution = metadata.get("channelResolution", [16] * channels)
    black_vals = _scale_display_values(metadata.get("blackvalue"), channel_resolution, channels)
    white_vals = _scale_display_values(metadata.get("whitevalue"), channel_resolution, channels)

    return {
        "channel_mins": ch_mins,
        "channel_maxs": ch_maxs,
        "channel_display_black_values": black_vals,
        "channel_display_white_values": white_vals,
    }


def _read_rows_strided(file_name: str, offset: int, ys: int, xs: int, chans: int, bpp: int, step: int, dtype) -> np.ndarray:
    """Slow-path reader: read every `step`th row into an ndarray of shape (ceil(ys/step), xs[, chans])."""
    import io
    rows = int(math.ceil(ys / step))
    if chans == 3:
        out = np.empty((rows, xs, 3), dtype=dtype)
        stride_bytes = xs * bpp * 3
    else:
        out = np.empty((rows, xs), dtype=dtype)
        stride_bytes = xs * bpp
    with open(file_name, "rb", buffering=io.DEFAULT_BUFFER_SIZE) as f:
        for i in range(rows):
            r_start = i * step
            if r_start >= ys:
                out = out[:i]
                break
            f.seek(offset + r_start * stride_bytes, os.SEEK_SET)
            buf = f.read(stride_bytes)
            if len(buf) < stride_bytes:
                out = out[:i]
                break
            arr = np.frombuffer(buf, dtype=dtype)
            if chans == 3:
                out[i, :, :] = arr.reshape((xs, 3))
            else:
                out[i, :] = arr.reshape((xs,))
    return out


def _fallback_only_display(meta: dict) -> Dict[str, List[int]]:
    channels = 3 if meta.get("isrgb") else int(meta.get("channels", 1) or 1)
    bits_per_ch = _resolve_bits_per_channel(meta, channels, bool(meta.get("isrgb", False)))
    # Use container from first channel for min/max range
    _, _, container_max_val = _dtype_from_bits(bits_per_ch[0])
    # Use channelResolution (significant bits) for display values scaling
    channel_resolution = meta.get("channelResolution", [16] * channels)
    black_vals = _scale_display_values(meta.get("blackvalue"), channel_resolution, channels)
    white_vals = _scale_display_values(meta.get("whitevalue"), channel_resolution, channels)
    return {
        "channel_mins": [0] * channels,
        "channel_maxs": [container_max_val] * channels,
        "channel_display_black_values": black_vals,
        "channel_display_white_values": white_vals,
    }


def _scale_display_values(values, channel_resolution: List[int], channels: int) -> List[int]:
    """Scale viewer black/white values (often 0..1 floats) to significant bits range per channel.

    Uses channelResolution (significant bits, e.g., 12) to compute the max value (e.g., 4095),
    not the container bits (8 or 16).

    If `values` length mismatches `channels`, pad/repeat as needed.
    If values seem already in integer range (>1), clamp and cast.
    """
    # Normalize input list length
    if isinstance(values, list) and values:
        vals = values[:]
    elif values is None:
        vals = [0.0] * channels
    else:
        # Scalar
        try:
            vals = [float(values)] * channels
        except Exception:
            vals = [0.0] * channels

    # Pad or trim
    if len(vals) < channels:
        vals = vals + [vals[-1] if vals else 0.0] * (channels - len(vals))
    else:
        vals = vals[:channels]

    # Normalize channel_resolution list
    if isinstance(channel_resolution, list) and channel_resolution:
        ch_res = channel_resolution[:]
    else:
        ch_res = [16] * channels  # Default to 16-bit if not provided
    
    # Pad or trim channel_resolution to match channels
    if len(ch_res) < channels:
        ch_res = ch_res + [ch_res[-1] if ch_res else 16] * (channels - len(ch_res))
    else:
        ch_res = ch_res[:channels]

    out: List[int] = []
    for i in range(channels):
        v = vals[i]
        try:
            v = float(v)
        except Exception:
            v = 0.0
        
        # Get max value from significant bits (channelResolution)
        sig_bits = ch_res[i] if ch_res[i] is not None else 16
        sig_max_val = (1 << sig_bits) - 1  # e.g., 12 bits -> 4095
        
        # If already in integer range, clamp to significant bits max
        if v > 1.0:
            out.append(int(max(0, min(sig_max_val, round(v)))))
        else:
            # Assume normalized [0,1] -> scale to significant bits range
            out.append(int(max(0, min(sig_max_val, round(v * sig_max_val)))))
    return out


# Global timer tracking for progress bars
_progress_start_times = {}  # Dict to track start times by prefix/phase
_total_process_start_time = None  # Track overall process start


def format_elapsed_time(seconds: float) -> str:
    """Format elapsed seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {mins}m {secs:.0f}s"


def start_progress_timer(phase: str = "default") -> None:
    """Start a timer for a specific progress phase."""
    global _progress_start_times, _total_process_start_time
    _progress_start_times[phase] = time.time()
    if _total_process_start_time is None:
        _total_process_start_time = time.time()


def get_elapsed_time(phase: str = "default") -> float:
    """Get elapsed seconds for a specific phase."""
    if phase in _progress_start_times:
        return time.time() - _progress_start_times[phase]
    return 0.0


def get_total_elapsed_time() -> float:
    """Get total elapsed seconds since first timer started."""
    global _total_process_start_time
    if _total_process_start_time is not None:
        return time.time() - _total_process_start_time
    return 0.0


def reset_progress_timers() -> None:
    """Reset all progress timers."""
    global _progress_start_times, _total_process_start_time
    _progress_start_times = {}
    _total_process_start_time = None


# Consistent prefix width for aligned progress bars
# Based on longest prefix: "Converting RGB to OME-TIFF:" = 27 chars
PROGRESS_PREFIX_WIDTH = 27

# Minimum suffix padding to ensure previous line content is cleared
MIN_SUFFIX_WIDTH = 20


def print_progress_bar(progress: float, *, total: float = 100.0, prefix: str = "Progress:",
                       suffix: str = "Complete", length: int = 40, fill: str = "█",
                       final_call: bool = False, phase: str = None) -> None:
    """Draw an in-place ASCII progress bar."""
    global _max_suffix_len  # pylint: disable=global-statement

    if "_max_suffix_len" not in globals():
        _max_suffix_len = 0

    # Auto-start timer on first call for this phase
    timer_phase = phase if phase else prefix
    if timer_phase not in _progress_start_times and progress < total:
        start_progress_timer(timer_phase)

    progress = min(progress, total)
    _max_suffix_len = max(_max_suffix_len, len(suffix), MIN_SUFFIX_WIDTH)
    padded_suffix = suffix.ljust(_max_suffix_len)
    padded_prefix = prefix.ljust(PROGRESS_PREFIX_WIDTH)

    percent = progress / total
    filled = int(length * percent)
    bar = fill * filled + "-" * (length - filled)
    print(f"\r  {padded_prefix} |{bar}| {percent:.1%} {padded_suffix}", end="", flush=True)

    if final_call:
        elapsed = get_elapsed_time(timer_phase)
        print(f" [{prefix.strip().rstrip(':')} {format_elapsed_time(elapsed)}]")
        _max_suffix_len = 0


# Track suffix length separately for save progress bar
_save_max_suffix_len = 0

def print_save_progress_bar(progress: float, *, total: float = 100.0, prefix: str = "Saving:",
                            suffix: str = "Complete", length: int = 40, fill: str = "▓",
                            empty: str = "░", final_call: bool = False, phase: str = None) -> None:
    """Draw an in-place ASCII progress bar with distinct style for save operations.
    
    Uses a different fill character and format to visually distinguish from the 
    main conversion progress bar.
    """
    global _save_max_suffix_len  # pylint: disable=global-statement

    # Auto-start timer on first call for this phase
    timer_phase = phase if phase else prefix
    if timer_phase not in _progress_start_times and progress < total:
        start_progress_timer(timer_phase)

    progress = min(progress, total)
    _save_max_suffix_len = max(_save_max_suffix_len, len(suffix), MIN_SUFFIX_WIDTH)
    padded_suffix = suffix.ljust(_save_max_suffix_len)
    padded_prefix = prefix.ljust(PROGRESS_PREFIX_WIDTH)

    percent = progress / total
    filled = int(length * percent)
    bar = fill * filled + empty * (length - filled)
    # Use a distinct format: angle brackets and different structure
    print(f"\r  {padded_prefix} <{bar}> {percent:.1%} - {padded_suffix}", end="", flush=True)

    if final_call:
        elapsed = get_elapsed_time(timer_phase)
        print(f" [{prefix.strip().rstrip(':')} {format_elapsed_time(elapsed)}]")
        _save_max_suffix_len = 0


# Track suffix length for copy progress bar
_copy_max_suffix_len = 0

def print_copy_progress_bar(progress: float, *, total: float = 100.0, prefix: str = "Copying:",
                            suffix: str = "Complete", length: int = 40, fill: str = "▒",
                            empty: str = "░", final_call: bool = False, phase: str = None) -> None:
    """Draw an in-place ASCII progress bar for file copy operations.
    
    Uses a different fill character to visually distinguish from save and conversion bars.
    """
    global _copy_max_suffix_len  # pylint: disable=global-statement

    # Auto-start timer on first call for this phase
    timer_phase = phase if phase else prefix
    if timer_phase not in _progress_start_times and progress < total:
        start_progress_timer(timer_phase)

    progress = min(progress, total)
    _copy_max_suffix_len = max(_copy_max_suffix_len, len(suffix), MIN_SUFFIX_WIDTH)
    padded_suffix = suffix.ljust(_copy_max_suffix_len)
    padded_prefix = prefix.ljust(PROGRESS_PREFIX_WIDTH)

    percent = progress / total
    filled = int(length * percent)
    bar = fill * filled + empty * (length - filled)
    # Use curly brackets for copy operations
    print(f"\r  {padded_prefix} {{{bar}}} {percent:.1%} - {padded_suffix}", end="", flush=True)

    if final_call:
        elapsed = get_elapsed_time(timer_phase)
        print(f" [{prefix.strip().rstrip(':')} {format_elapsed_time(elapsed)}]")
        _copy_max_suffix_len = 0


# Buffer size for file copy with progress (4MB chunks for good performance)
COPY_BUFFER_SIZE = 4 * 1024 * 1024


def copy_file_with_progress(
    src_path: str,
    dest_path: str,
    show_progress: bool = False,
    prefix: str = "Copying source:",
    buffer_size: int = COPY_BUFFER_SIZE
) -> bool:
    """
    Copy a file with progress bar display.
    
    Uses chunked reading/writing to show progress during potentially long copies
    from network locations.
    
    Parameters:
    - src_path: Source file path
    - dest_path: Destination file path
    - show_progress: If True, display progress bar
    - prefix: Prefix text for progress bar
    - buffer_size: Size of chunks to copy at a time (default 4MB)
    
    Returns:
    - True on success
    
    Raises:
    - OSError/IOError on failure
    """
    src_size = os.path.getsize(src_path)
    
    # Ensure destination directory exists
    dest_dir = os.path.dirname(dest_path)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)
    
    # Remove existing destination if present
    if os.path.exists(dest_path):
        try:
            os.remove(dest_path)
        except OSError:
            pass
    
    copied = 0
    last_percent = -1
    
    with open(src_path, 'rb') as fsrc:
        with open(dest_path, 'wb') as fdest:
            while True:
                buf = fsrc.read(buffer_size)
                if not buf:
                    break
                fdest.write(buf)
                copied += len(buf)
                
                if show_progress and src_size > 0:
                    percent = int(100 * copied / src_size)
                    # Update every 2% to avoid too many updates
                    if percent >= last_percent + 2 or copied >= src_size:
                        last_percent = percent
                        # Format size nicely
                        if src_size >= 1024 * 1024 * 1024:
                            size_str = f"{copied / (1024*1024*1024):.1f}/{src_size / (1024*1024*1024):.1f} GB"
                        elif src_size >= 1024 * 1024:
                            size_str = f"{copied / (1024*1024):.0f}/{src_size / (1024*1024):.0f} MB"
                        else:
                            size_str = f"{copied / 1024:.0f}/{src_size / 1024:.0f} KB"
                        
                        is_final = copied >= src_size
                        print_copy_progress_bar(
                            percent, prefix=prefix, suffix=size_str, 
                            final_call=is_final, phase=prefix
                        )
    
    # Copy file metadata (permissions, times)
    shutil.copystat(src_path, dest_path)
    
    # Verify size
    dest_size = os.path.getsize(dest_path)
    if dest_size != src_size:
        raise OSError(f"Size mismatch after copy: source={src_size}, dest={dest_size}")
    
    return True


def _find_image_hierarchical_path(xlef_path, image_uuid):
    """
    Recursively traverse the XLEF/XLCF/XLIF hierarchy to build a hierarchical name
    for the image with the given UUID. Returns a single underscore-joined string
    like "Root_Collection1_Collection2_Image" (without any ".xlef/.xlcf" redundancy),
    or None if not found.
    """
    import xml.etree.ElementTree as ET
    from urllib.parse import unquote
    import os

    def _traverse(file_path, target_uuid, parent_names, visited):
        if not os.path.exists(file_path) or file_path in visited:
            return None
        visited.add(file_path)
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
        except Exception:
            return None
        element = root.find(".//Element")
        if element is None:
            return None
        raw_name = element.get("Name", "")
        this_uuid = element.get("UniqueID", "")
        ext = file_path.lower().split('.')[-1]
        # Normalize folder names: prefer the on-disk base name and strip extensions
        if ext in ("xlef", "xlcf"):
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            xml_base = os.path.splitext(raw_name)[0]
            folder_name = base_name or xml_base or raw_name
            # Avoid duplicate consecutive names (e.g., Root then Root.xlef)
            if not parent_names or parent_names[-1] != folder_name:
                current_names = parent_names + [folder_name]
            else:
                current_names = parent_names[:]
        else:
            current_names = parent_names[:]
        # If this is the image node
        if ext == 'xlif' and this_uuid == target_uuid:
            image_name = raw_name or os.path.splitext(os.path.basename(file_path))[0]
            return current_names + [image_name]
        # If this is a folder (XLEF/XLCF), search children
        child_elem = element.find("Children")
        if child_elem is not None:
            for ref in child_elem.findall("Reference"):
                ref_file = unquote(ref.get("File") or "")
                ref_file = ref_file.replace("\\", "/")
                ref_file = os.path.normpath(os.path.join(os.path.dirname(file_path), ref_file))
                ref_uuid = ref.get("UUID") or ""
                ext2 = ref_file.lower().split('.')[-1]
                # If this is the image node
                if ext2 == 'xlif' and ref_uuid == target_uuid:
                    # Get the name from the referenced file
                    try:
                        tree2 = ET.parse(ref_file)
                        root2 = tree2.getroot()
                        el2 = root2.find(".//Element")
                        name2_raw = el2.get("Name", "") if el2 is not None else ""
                        name2 = name2_raw or os.path.splitext(os.path.basename(ref_file))[0]
                    except Exception:
                        name2 = ""
                    return current_names + [name2]
                # Otherwise, recurse
                result = _traverse(ref_file, target_uuid, current_names, visited)
                if result:
                    return result
        return None

    root_name = os.path.splitext(os.path.basename(xlef_path))[0]
    parts = _traverse(xlef_path, image_uuid, [root_name], set())
    if parts:
        return "_".join([p for p in parts if p])
    return None


def _read_xlef_image(xlef_path: str, image_uuid: str) -> dict:
    """Return metadata dict for *one* image UUID inside an XLEF experiment."""
    # Load and search lazily across potentially linked XLEFs

    def walk(node: dict) -> dict | None:
        if node.get("uuid") == image_uuid and node.get("type", "").lower() == "image":
            return node
        for child in node.get("children", []):
            found = walk(child)
            if found:
                return found
        return None

    # breadth-first search through linked folders
    queue: list[str] = [xlef_path]
    processed_paths = set()  # Avoid infinite loops with circular links
    while queue:
        current = queue.pop(0)
        if current in processed_paths:
            continue
        processed_paths.add(current)

        try:
            meta = json.loads(read_leica_xlef(current))
        except Exception as e:
            print(f"Warning: Could not read linked XLEF '{current}': {e}")
            continue  # Skip unreadable files

        maybe = walk(meta)
        if maybe:
            # Preserve original save_child_name if merging LOF
            original_save_child_name = maybe.get("save_child_name")
            if "lof_file_path" in maybe and maybe["lof_file_path"]:
                try:
                    # merge LOF metadata if present
                    lof_meta = json.loads(read_leica_lof(maybe["lof_file_path"], include_xmlelement=True))
                    maybe.update(lof_meta)
                    # Restore original name if it was overwritten by LOF merge
                    if original_save_child_name is not None:
                        maybe["save_child_name"] = original_save_child_name
                except Exception as e:
                    print(f"Warning: Could not read/merge LOF metadata from '{maybe['lof_file_path']}': {e}")
            # Ensure essential fields exist after potential merge
            maybe.setdefault("filetype", ".xlef")
            maybe.setdefault("LOFFilePath", maybe.get("lof_file_path", current))  # Best guess if LOF failed
            return maybe

        for child in meta.get("children", []):
            if child.get("type", "").lower() == "folder" and child.get("file_path"):
                # Ensure the path is absolute or relative to the current XLEF
                child_path = child["file_path"]
                if not os.path.isabs(child_path):
                    child_path = os.path.join(os.path.dirname(current), child_path)
                if os.path.exists(child_path):  # Check if linked file exists
                    queue.append(os.path.normpath(child_path))
                else:
                    print(f"Warning: Linked XLEF folder path not found: '{child_path}'")

    raise ValueError(f"Image UUID {image_uuid} not found in {xlef_path} or linked XLEFs")


def read_image_metadata(file_path: str, image_uuid: str) -> dict:
    """Front-end that works for .lif / .xlef / .lof."""
    if file_path.endswith(".lif"):
        meta_str = read_leica_lif(file_path, include_xmlelement=True, image_uuid=image_uuid)
        if not meta_str:
            raise ValueError(f"Image UUID {image_uuid} not found in LIF file {file_path}")
        meta = json.loads(meta_str)
        # Ensure essential fields exist
        meta.setdefault("filetype", ".lif")
        meta.setdefault("LIFFile", file_path)
        return meta
    if file_path.endswith(".lof"):
        meta_str = read_leica_lof(file_path, include_xmlelement=True)
        if not meta_str:
            raise ValueError(f"Could not read LOF file {file_path}")
        meta = json.loads(meta_str)
        meta.setdefault("filetype", ".lof")
        meta.setdefault("LOFFilePath", file_path)
        return meta
    if file_path.endswith(".xlef"):
        return _read_xlef_image(file_path, image_uuid)
    raise ValueError(f"Unsupported file type: {file_path}")


def decimal_to_rgb(value: int) -> tuple[int, int, int]:
    r = (value >> 16) & 0xFF   # top byte
    g = (value >> 8)  & 0xFF   # middle byte
    b =  value        & 0xFF   # bottom byte
    return (r, g, b)


def color_name_to_decimal(name: str) -> int:
    css_colors = {
        "aqua": (0, 255, 255),
        "azure": (240, 255, 255),
        "beige": (245, 245, 220),
        "black": (0, 0, 0),
        "blue": (0, 0, 255),
        "blueviolet": (138, 43, 226),
        "brown": (165, 42, 42),
        "cyan": (0, 255, 255),
        "darkblue": (0, 0, 139),
        "darkcyan": (0, 139, 139),
        "darkgray": (169, 169, 169),  "darkgrey": (169, 169, 169),
        "darkgreen": (0, 100, 0),
        "darkmagenta": (139, 0, 139),
        "darkorange": (255, 140, 0),
        "darkred": (139, 0, 0),
        "dimgray": (105, 105, 105),   "dimgrey": (105, 105, 105),
        "gray": (128, 128, 128),       "grey": (128, 128, 128),
        "greenyellow": (173, 255, 47),
        "green": (0, 128, 0),
        "indigo": (75, 0, 130),
        "lightblue": (173, 216, 230),
        "lightcyan": (224, 255, 255),
        "lightgray": (211, 211, 211),  "lightgrey": (211, 211, 211),
        "lightgreen": (144, 238, 144),
        "lightyellow": (255, 255, 224),
        "lime": (0, 255, 0),
        "limegreen": (50, 205, 50),
        "magenta": (255, 0, 255),
        "mediumblue": (0, 0, 205),
        "mediumpurple": (147, 112, 219),
        "orange": (255, 165, 0),
        "orangered": (255, 69, 0),
        "pink": (255, 192, 203),
        "purple": (128, 0, 128),
        "red": (255, 0, 0),
        "silver": (192, 192, 192),
        "tomato": (255, 99, 71),
        "turquoise": (64, 224, 208),
        "violet": (238, 130, 238),
        "white": (255, 255, 255),
        "yellow": (255, 255, 0),
        "yellowgreen": (154, 205, 50)
    }
    r, g, b = css_colors[name.lower()]       # KeyError if unknown
    return (r << 16) | (g << 8) | b


def decimal_to_ome_color(rgb_int: int, alpha: int = 255) -> int:
    r = (rgb_int >> 16) & 0xFF
    g = (rgb_int >> 8)  & 0xFF
    b =  rgb_int        & 0xFF
    unsigned_rgba = (r << 24) | (g << 16) | (b << 8) | (alpha & 0xFF)
    if unsigned_rgba >= 0x80000000:
        signed_rgba = unsigned_rgba - 0x100000000  # Subtract 2**32
    else:
        signed_rgba = unsigned_rgba
    return signed_rgba


XS_NS = {"xs": "http://www.w3.org/2001/XMLSchema"}


def _download(url: str) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False)
    urllib.request.urlretrieve(url, tmp.name)
    return tmp.name


def _load_schema_tree(url: str, seen: set[str]) -> list[ET.ElementTree]:
    if url in seen:
        return []
    seen.add(url)
    filename = _download(url)
    tree = ET.parse(filename)
    trees = [tree]
    for include in tree.findall(".//xs:include|.//xs:import", XS_NS):
        loc = include.get("schemaLocation")
        if not loc:
            continue
        full_url = urllib.parse.urljoin(url, loc)
        trees.extend(_load_schema_tree(full_url, seen))
    return trees


def parse_ome_xsd(xsd_url: str) -> dict[str, dict]:
    trees = _load_schema_tree(xsd_url, seen=set())
    simple_type_enums: dict[str, list[str]] = {}
    metadata: dict[str, dict] = {}
    for tree in trees:
        for s_type in tree.findall(".//xs:simpleType[@name]", XS_NS):
            name = s_type.get("name")
            enum_vals = [
                e.get("value")
                for e in s_type.findall(".//xs:enumeration", XS_NS)
            ]
            if enum_vals:
                simple_type_enums[name] = enum_vals
    for tree in trees:
        for attr in tree.findall(".//xs:attribute", XS_NS):
            attr_name = attr.get("name")
            if not attr_name:
                continue
            typeref = attr.get("type")
            if typeref in simple_type_enums:
                metadata[attr_name] = {
                    "type": "string",
                    "values": simple_type_enums[typeref],
                }
                continue
            inline_enum = [
                e.get("value")
                for e in attr.findall(".//xs:enumeration", XS_NS)
            ]
            if inline_enum:
                metadata[attr_name] = {
                    "type": "string",
                    "values": inline_enum,
                }
        for c_type in tree.findall(".//xs:complexType[@name]", XS_NS):
            name = c_type.get("name")
            attrs = {
                a.get("name"): a.get("type", "string")
                for a in c_type.findall(".//xs:attribute", XS_NS)
                if a.get("name")
            }
            if attrs:
                metadata[name] = {"type": "complex", "attributes": attrs}
    return metadata


def validate_metadata(value: str, field: str, schema: dict) -> str:
    spec = schema.get(field)
    if not spec or "values" not in spec:
        return "Other"
    cleaned = value.strip().lower()
    for canonical in spec["values"]:
        if cleaned == canonical.lower():
            return canonical
    return "Other"


# -----------------------------------------------------------------------------
# Robust file copy with size verification and retry logic
# -----------------------------------------------------------------------------

def robust_file_copy(src_path: str, dest_path: str, max_retries: int = 10,
                     show_progress: bool = False, prefix: str = "Copying output:",
                     buffer_size: int = COPY_BUFFER_SIZE) -> bool:
    """
    Copy a file from src_path to dest_path with size verification, retry logic,
    and optional progress display.
    
    On failure, retries up to max_retries times with progressive backoff:
    - Retry 1: wait 1 minute
    - Retry 2: wait 2 minutes after retry 1
    - Retry N: wait N minutes after retry N-1
    
    Parameters:
    - src_path: Source file path (should be a local file)
    - dest_path: Destination file path (may be on a network drive)
    - max_retries: Maximum number of retry attempts (default: 10)
    - show_progress: If True, display progress bar during copy
    - prefix: Prefix text for progress bar
    - buffer_size: Size of chunks to copy at a time (default 4MB)
    
    Returns:
    - True on success
    
    Raises:
    - OSError: If all retries are exhausted
    """
    src_size = os.path.getsize(src_path)
    
    for attempt in range(max_retries + 1):
        try:
            # Ensure destination directory exists
            dest_dir = os.path.dirname(dest_path)
            if dest_dir:
                os.makedirs(dest_dir, exist_ok=True)
            
            # Remove destination if it exists (partial copy from previous attempt)
            if os.path.exists(dest_path):
                try:
                    os.remove(dest_path)
                except OSError:
                    pass  # Will fail on copy if still locked
            
            # Copy the file with optional progress display
            copied = 0
            last_percent = -1
            
            with open(src_path, 'rb') as fsrc:
                with open(dest_path, 'wb') as fdest:
                    while True:
                        buf = fsrc.read(buffer_size)
                        if not buf:
                            break
                        fdest.write(buf)
                        copied += len(buf)
                        
                        if show_progress and src_size > 0:
                            percent = int(100 * copied / src_size)
                            # Update every 2% to avoid too many updates
                            if percent >= last_percent + 2 or copied >= src_size:
                                last_percent = percent
                                # Format size nicely
                                if src_size >= 1024 * 1024 * 1024:
                                    size_str = f"{copied / (1024*1024*1024):.1f}/{src_size / (1024*1024*1024):.1f} GB"
                                elif src_size >= 1024 * 1024:
                                    size_str = f"{copied / (1024*1024):.0f}/{src_size / (1024*1024):.0f} MB"
                                else:
                                    size_str = f"{copied / 1024:.0f}/{src_size / 1024:.0f} KB"
                                
                                is_final = copied >= src_size
                                print_copy_progress_bar(
                                    percent, prefix=prefix, suffix=size_str, 
                                    final_call=is_final, phase=prefix
                                )
            
            # Copy file metadata (permissions, times)
            shutil.copystat(src_path, dest_path)
            
            # Verify by size comparison
            if os.path.exists(dest_path):
                dest_size = os.path.getsize(dest_path)
                if dest_size == src_size:
                    return True
                else:
                    raise OSError(f"Size mismatch: source={src_size}, dest={dest_size}")
            else:
                raise OSError("Destination file does not exist after copy")
                
        except (OSError, IOError, shutil.Error) as e:
            if attempt < max_retries:
                wait_minutes = attempt + 1
                print(f"\nWarning: Copy attempt {attempt + 1} failed: {e}")
                print(f"  Retrying in {wait_minutes} minute(s)... ({max_retries - attempt} retries remaining)")
                time.sleep(wait_minutes * 60)
            else:
                raise OSError(
                    f"Failed to copy file after {max_retries + 1} attempts. "
                    f"Source: {src_path}, Destination: {dest_path}. Last error: {e}"
                )
    
    # Should not reach here, but just in case
    return False


def safe_tiffsave(img, out_path: str, altoutputfolder: str | None = None,
                  show_progress: bool = False, progress_callback: callable = None,
                  tempfolder: str | None = None,
                  **tiffsave_kwargs) -> str:
    """
    Save a pyvips image to a TIFF file using a temp-first approach for reliability.
    
    This function:
    1. Saves the image to a temporary file in the system temp directory (or custom tempfolder)
    2. Copies the temp file to out_path with verification and retry logic
    3. If altoutputfolder is provided, copies temp file there too
    4. Removes the temp file only after all copies are verified
    
    Parameters:
    - img: pyvips.Image object to save
    - out_path: Final destination path for the TIFF file
    - altoutputfolder: Optional alternative output folder for a second copy
    - show_progress: If True, display console progress bar during save
    - progress_callback: Optional callable(phase: str, percent: float) for programmatic progress
                        phase is one of: "writing", "copying", "copying_alt"
    - tempfolder: Optional custom temp folder path. If None, uses system temp directory.
    - **tiffsave_kwargs: Additional arguments passed to img.tiffsave()
    
    Returns:
    - The filename (basename) of the saved file
    
    Raises:
    - Various exceptions if saving or copying fails after all retries
    """
    # Generate a unique temp file path in system temp directory or custom tempfolder
    temp_dir = tempfolder if tempfolder else tempfile.gettempdir()
    # Ensure temp directory exists
    os.makedirs(temp_dir, exist_ok=True)
    temp_filename = f"{uuid_module.uuid4()}.tmp.tiff"
    temp_path = os.path.join(temp_dir, temp_filename)
    
    ome_name = os.path.basename(out_path)
    has_alt = altoutputfolder is not None
    
    # Progress tracking for pyvips signals
    last_reported_percent = [-1]  # Use list to allow modification in nested function
    saving_completed = [False]  # Track if we've already printed the final saving line
    
    def _report_progress(phase: str, percent: float):
        """Internal helper to report progress via both callback and console."""
        if progress_callback is not None:
            try:
                progress_callback(phase, percent)
            except Exception:
                pass  # Don't let callback errors break the save
        
        if show_progress and phase == "writing":
            # Only show the Saving progress bar for the writing phase
            # Copying phases have their own progress bars via robust_file_copy
            is_final = percent >= 100 and not saving_completed[0]
            print_save_progress_bar(percent, prefix="Saving:", suffix="Writing TIFF", final_call=is_final)
            if is_final:
                saving_completed[0] = True
    
    try:
        # Set up pyvips progress reporting
        def on_eval(image, progress):
            # Only report every 2% to avoid too many updates, and skip if already completed
            if saving_completed[0]:
                return
            current_percent = int(progress.percent)
            if current_percent >= last_reported_percent[0] + 2 or current_percent >= 100:
                last_reported_percent[0] = current_percent
                _report_progress("writing", progress.percent)
        
        img.set_progress(True)
        img.signal_connect("eval", on_eval)
        
        if show_progress:
            _report_progress("writing", 0)
        
        # Save to temp location first
        img.tiffsave(temp_path, **tiffsave_kwargs)
        
        # Ensure 100% is reported for writing phase (only if not already done by callback)
        if not saving_completed[0]:
            _report_progress("writing", 100)
        
        # Verify temp file was created
        if not os.path.exists(temp_path):
            raise OSError(f"Temp file was not created: {temp_path}")
        
        temp_size = os.path.getsize(temp_path)
        if temp_size == 0:
            raise OSError(f"Temp file is empty: {temp_path}")
        
        # Copy to primary destination with retry logic and progress display
        robust_file_copy(temp_path, out_path, show_progress=show_progress, 
                        prefix="Copying output:")
        
        # Copy to alternative destination if specified
        if altoutputfolder:
            alt_out_path = os.path.join(altoutputfolder, ome_name)
            robust_file_copy(temp_path, alt_out_path, show_progress=show_progress,
                           prefix="Copying alt output:")
        
        # Print total elapsed time if we were showing progress
        if show_progress:
            total_elapsed = get_total_elapsed_time()
            print(f"  Total time: {format_elapsed_time(total_elapsed)}")
        
        return ome_name
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                if show_progress:
                    print(f"  Cleaned up temp output file")
            except OSError as e:
                print(f"\nWarning: Could not remove temporary TIFF file {temp_path}: {e}")


# -----------------------------------------------------------------------------
# Robust source file reading - copy source to temp before processing
# -----------------------------------------------------------------------------

def prepare_temp_source(
    inputfile: str,
    image_uuid: str,
    metadata: dict,
    tempfolder: str | None = None,
    show_progress: bool = False,
    max_retries: int = 10
) -> tuple[str, int, str | None]:
    """
    Prepare a local temp copy of the source data for robust reading from network files.
    
    For LOF files: copies the entire file to temp folder.
    For LIF files: extracts the single image to a temp LIF file using existing singlelif code.
    
    Parameters:
    - inputfile: Original source file path (.lif, .lof, .xlef)
    - image_uuid: UUID of the image to extract (for LIF files)
    - metadata: Pre-read metadata dict containing filetype, LIFFile, LOFFilePath, Position, etc.
    - tempfolder: Optional custom temp folder. If None, uses system temp.
    - show_progress: If True, display progress during copy
    - max_retries: Maximum retry attempts for file operations
    
    Returns:
    - Tuple of (temp_file_path, base_position, cleanup_path)
      - temp_file_path: Path to the temp file to read from
      - base_position: Byte position offset for reading (62 for LOF-style, or from metadata for LIF)
      - cleanup_path: Path to remove after processing (may differ for LIF extraction)
    
    Raises:
    - OSError: If copy/extraction fails after all retries
    """
    # Import here to avoid circular imports
    try:
        from .ci_leica_converters_single_lif import convert_leica_to_singlelif_temp
    except ImportError:
        from ci_leica_converters_single_lif import convert_leica_to_singlelif_temp
    
    temp_dir = tempfolder if tempfolder else tempfile.gettempdir()
    os.makedirs(temp_dir, exist_ok=True)
    
    filetype = metadata.get("filetype", "").lower()
    
    if filetype in [".lof", ".xlef"]:
        # For LOF/XLEF: copy the entire LOF file to temp
        source_file = metadata.get("LOFFilePath")
        if not source_file or not os.path.exists(source_file):
            raise FileNotFoundError(f"LOF source file not found: {source_file}")
        
        # Generate unique temp filename
        temp_filename = f"{uuid_module.uuid4()}.tmp.lof"
        temp_path = os.path.join(temp_dir, temp_filename)
        
        # Use robust copy with progress and retry logic
        src_size = os.path.getsize(source_file)
        
        for attempt in range(max_retries + 1):
            try:
                # Use copy with progress bar
                copy_file_with_progress(
                    source_file, temp_path,
                    show_progress=show_progress,
                    prefix="Copying source:"
                )
                
                # Verify is done inside copy_file_with_progress, but double-check
                if os.path.exists(temp_path):
                    dest_size = os.path.getsize(temp_path)
                    if dest_size == src_size:
                        # LOF data starts at position 62
                        return (temp_path, 62, temp_path)
                    else:
                        raise OSError(f"Size mismatch: source={src_size}, temp={dest_size}")
                else:
                    raise OSError("Temp file does not exist after copy")
                    
            except (OSError, IOError, shutil.Error) as e:
                if attempt < max_retries:
                    wait_minutes = attempt + 1
                    if show_progress:
                        print(f"\nWarning: Source copy attempt {attempt + 1} failed: {e}")
                        print(f"  Retrying in {wait_minutes} minute(s)...")
                    time.sleep(wait_minutes * 60)
                else:
                    # Clean up partial temp file if exists
                    if os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except OSError:
                            pass
                    raise OSError(
                        f"Failed to copy source file after {max_retries + 1} attempts. "
                        f"Source: {source_file}. Last error: {e}"
                    )
    
    elif filetype == ".lif":
        # For LIF: extract single image to temp LIF file
        source_file = metadata.get("LIFFile")
        if not source_file or not os.path.exists(source_file):
            raise FileNotFoundError(f"LIF source file not found: {source_file}")
        
        # Extract to temp folder - returns (temp_lif_path, new_position)
        # The singlelif_temp function has its own progress bar
        temp_lif_path, new_position = convert_leica_to_singlelif_temp(
            inputfile=source_file,
            image_uuid=image_uuid,
            tempfolder=temp_dir,
            show_progress=show_progress,
            max_retries=max_retries
        )
        
        return (temp_lif_path, new_position, temp_lif_path)
    
    else:
        raise ValueError(f"Unsupported filetype for temp source preparation: {filetype}")


def cleanup_temp_source(cleanup_path: str | None, show_progress: bool = False) -> None:
    """
    Remove a temporary source file created by prepare_temp_source.
    
    Parameters:
    - cleanup_path: Path to the temp file to remove (or None to skip)
    - show_progress: If True, print cleanup messages
    """
    if cleanup_path and os.path.exists(cleanup_path):
        try:
            os.remove(cleanup_path)
            if show_progress:
                print(f"  Cleaned up temp source file")
        except OSError as e:
            print(f"\nWarning: Could not remove temp source file {cleanup_path}: {e}")


xsd_url = "http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd"
metadata_schema = parse_ome_xsd(xsd_url)
