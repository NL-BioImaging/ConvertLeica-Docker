import os
import json
import uuid
import tempfile
import numpy as np
import xml.etree.ElementTree as ET  # For escaping XML
from html import escape as escape_xml_chars  # More robust XML escaping
import gc
import re
import math # Added for math.ceil
import sys

# Import helpers from the dedicated module
from ci_leica_converters_helpers import (
    dtype_to_format,
    print_progress_bar,
    read_image_metadata,
    color_name_to_decimal,
    decimal_to_ome_color,
    validate_metadata,
    metadata_schema, # This already contains the parsed schema
    safe_tiffsave,
    prepare_temp_source,
    cleanup_temp_source
)

if sys.platform.startswith("win"):
    vips_bin_dir = r"C:\bin\vips\bin"
    os.environ["PATH"] = os.pathsep.join((vips_bin_dir, os.environ["PATH"]))

import pyvips

# -----------------------------------------------------------------------------
# Low-level reader - pull a range of rows out of a *single* channel / Z plane
# -----------------------------------------------------------------------------

# Rename cbytes parameter back to cbytesinc for clarity
def read_rows(base_file: str, base_pos: int, xs: int,
              row_start: int, row_end: int, skip: int,
              channel: int, bits: int, zbytes: int, cbytesinc: list[int],
              zs: int = 1, *, project: bool = False, target_z: int | None = None,
              timepoint: int = 0, tbytes: int = 0, ts: int = 1) -> np.ndarray: # Default tbytes to 0
    """Return a 2D numpy (row, col) slice from Leica raw data blocks, supporting time dimension."""

    dtype = np.uint16 if bits == 16 else np.uint8
    bpp = bits // 8
    out_rows = (row_end - row_start + skip - 1) // skip
    arr = np.zeros((out_rows, xs), dtype=dtype)
    # Revert: Calculate line_bytes without padding
    line_bytes = xs * bpp

    if not isinstance(base_pos, int) or base_pos < 0:
        raise ValueError(f"Invalid base position offset: {base_pos}")
    if tbytes is None: # Handle None case if passed
        tbytes = 0
    if zbytes is None:
        zbytes = 0

    try:
        with open(base_file, "rb") as fh:
            if not (0 <= channel < len(cbytesinc)):
                raise IndexError(f"Channel index {channel} out of range for cbytesinc (length {len(cbytesinc)})")
            # Get the base offset for this channel relative to the start of its Z/T plane
            ch_offset = cbytesinc[channel]
            idx = 0

            if project and zs > 1:
                 # Projection logic needs careful review if dimension order changed significantly.
                 # Assuming T > Z > C > Y > X for projection for now, might need adjustment.
                for r in range(row_start, row_end, skip):
                    zbuf = np.zeros((zs, xs), dtype=dtype)
                    for z_proj in range(zs):
                        # Calculate position assuming T > Z > C > Y > X for projection
                        # Use original line_bytes for position calculation
                        pos = base_pos + timepoint * tbytes + z_proj * zbytes + ch_offset + r * line_bytes
                        fh.seek(pos)
                        row = fh.read(line_bytes) # Read original line bytes
                        # Use original line_bytes for buffer
                        if len(row) < line_bytes:
                            read_count = len(row) // bpp
                            if read_count > 0:
                                zbuf[z_proj, :read_count] = np.frombuffer(row, dtype=dtype, count=read_count)
                        else:
                            zbuf[z_proj] = np.frombuffer(row, dtype=dtype, count=xs)
                    arr[idx] = zbuf.max(axis=0)
                    idx += 1
                    if idx >= out_rows:
                        break
            else:
                z = 0 if target_z is None else int(target_z)
                if not (0 <= z < zs):
                    raise IndexError(f"Target Z index {z} out of range for zs size {zs}")
                if not (0 <= timepoint < ts):
                     raise IndexError(f"Timepoint index {timepoint} out of range for ts size {ts}")

                # Calculate the start position of the specific C, Z, T plane relative to base_pos
                # Assumes T > Z > C order for the start of the plane
                plane_start_offset = timepoint * tbytes + z * zbytes + ch_offset
                plane_start_pos = base_pos + plane_start_offset

                for r in range(row_start, row_end, skip):
                    # Position is the start of the plane + row offset (using original line_bytes)
                    pos = plane_start_pos + r * line_bytes
                    fh.seek(pos)
                    row = fh.read(line_bytes) # Read original line bytes
                    # Use original line_bytes for buffer
                    if len(row) < line_bytes:
                        read_count = len(row) // bpp
                        if read_count > 0:
                            arr[idx, :read_count] = np.frombuffer(row, dtype=dtype, count=read_count)
                    else:
                        arr[idx] = np.frombuffer(row, dtype=dtype, count=xs)
                    idx += 1
                    if idx >= out_rows:
                        break
    except FileNotFoundError:
        raise FileNotFoundError(f"Base data file not found: {base_file}")
    except OSError as e:
        raise OSError(f"Error reading from base file {base_file} at offset {pos}: {e}")

    return arr

# -----------------------------------------------------------------------------
# Generate OME-XML (XYZCT) including original Leica metadata annotation
# -----------------------------------------------------------------------------

def generate_ome_xml(meta: dict, filename: str, *, include_original_metadata: bool = False) -> str:
    """Return OME-XML including original Leica XML as an annotation."""

    xs, ys = meta["xs"], meta["ys"]
    zs, channels = meta.get("zs", 1), meta["channels"]
    xres = meta.get("xres", 0.0)  # Physical size in METERS from Leica readers
    yres = meta.get("yres", 0.0)
    zres = meta.get("zres", 0.0)
    ts = meta.get("ts", 1)
    # Get acquisition date from metadata, default to "Unknown"
    acquisition_date = meta.get("experiment_datetime_str", "Unknown")
    if not acquisition_date: # Handle empty string case
        acquisition_date = "Unknown"

    # Convert meters to micrometers for OME standard units
    xres_um = xres * 1_000_000 if xres > 0 else 1.0  # Default to 1.0 if zero/invalid
    yres_um = yres * 1_000_000 if yres > 0 else 1.0
    zres_um = zres * 1_000_000 if zres > 0 else 1.0

    # Determine bit depth and significant bits
    res = meta.get("channelResolution", [16]) # Default to 16-bit if not specified
    if not isinstance(res, list) or not res or not all(isinstance(r, int) for r in res):
        print(f"\nWarning: Invalid channelResolution format: {res}. Defaulting to 16-bit.")
        res = [16] * channels # Assume 16-bit for all channels if invalid
    # Use the maximum resolution found
    sbits = max(res) if res else 16 # Store original significant bits
    # Determine pixel type based on container size (8 or 16)
    bits = 16 if sbits > 8 else 8
    pixel_type = "uint16" if bits == 16 else "uint8"


    # --- Instrument & Channel Metadata Extraction ---
    objective_name = meta.get("objective", "Unknown Objective")
    objective_na = meta.get("na", 0.0)
    objective_model=objective_name
    objective_refractive_index = meta.get("refractiveindex", 1.0) # Default to 1.0 if not specified

    # Prioritize magnification from metadata if available and valid
    objective_mag = meta.get("magnification")
    if not isinstance(objective_mag, (int, float)) or objective_mag <= 0:
        # If not available or invalid, try to extract from name
        objective_mag = None # Reset to None before trying regex
        try:
            # Find digits followed by 'x' (case-insensitive)
            match = re.search(r'(\d+)\s*x', objective_name, re.IGNORECASE)
            if match:
                objective_mag = int(match.group(1))
        except Exception as e:
            # Handle potential errors during regex or int conversion gracefully
            objective_mag = None  # Ensure it's None if parsing fails
    else:
        # Ensure it's an integer if it came from metadata
        objective_mag = int(objective_mag)

    # --- Determine Immersion ---
    # Use immersion directly from metadata if available, otherwise determine based on name/NA
    immersion_type = meta.get("immersion") # Get from parser first
    if immersion_type is None or not isinstance(immersion_type, str) or immersion_type.upper() not in ["OIL", "WATER", "AIR", "GLYCEROL", "MULTI", "OTHER"]:
        # If not found or invalid, determine based on name/NA
        immersion_type = "Other" # Default OME value
        if isinstance(objective_name, str):
            objective_name_lower = objective_name.lower()
            if "oil" in objective_name_lower:
                immersion_type = "Oil"
            elif "water" in objective_name_lower or "wasser" in objective_name_lower: # Add German 'Wasser'
                immersion_type = "Water"
            elif "air" in objective_name_lower or "dry" in objective_name_lower: # Add 'Dry'
                immersion_type = "Air"
            elif "glyc" in objective_name_lower: # Check for Glycerol/Glycerin
                 immersion_type = "Glycerol"
            else:
                # Fallback based on NA if name doesn't specify
                if isinstance(objective_na, (int, float)) and objective_na > 0:
                    if objective_na > 0.75:
                        immersion_type = "Oil" # Guess Oil for high NA
                    else:
                        immersion_type = "Air" # Guess Air for low NA
                # If NA is invalid/zero, keep default "Other"
    else:
        # Standardize case if found in metadata
        immersion_type = immersion_type.capitalize()
        if immersion_type == "Dry": immersion_type = "Air" # Map Dry to Air

    # Validate metadata fields before XML generation
    immersion_type = validate_metadata(immersion_type, "Immersion", metadata_schema)

    mic_model = meta.get("mic_type2", "Unknown Microscope") # Using mic_type2 as requested
    pinhole_size_um = meta.get("pinholesize_um") # Get pinhole size from parser

    # Ensure channel-specific lists match channel count
    def _pad_list(data_list, default_val, count):
        if not isinstance(data_list, list):
            return [default_val] * count
        padded = list(data_list)
        padded.extend([default_val] * (count - len(padded)))
        return padded[:count] # Ensure it's not longer

    lut = meta.get("lutname", ["white"] * channels)
    lut = _pad_list(lut, "white", channels)
    lut_names = [str(l) if l is not None else "white" for l in lut]

    filter_blocks = _pad_list(meta.get("filterblock"), "Unknown Filter", channels)
    excitations = _pad_list(meta.get("excitation"), None, channels)
    emissions = _pad_list(meta.get("emission"), None, channels)
    contrast_methods = _pad_list(meta.get("contrastmethod"), "Unknown", channels)
    # Map contrast methods to OME terms
    contrast_methods_ome = []
    for cm in contrast_methods:
        cm_upper = cm.upper() if isinstance(cm, str) else "UNKNOWN"
        if cm_upper == "FLUO":
            contrast_methods_ome.append("Fluorescence")
        elif cm_upper == "BF":
            contrast_methods_ome.append("Brightfield")
        elif cm_upper == "DIC":
            contrast_methods_ome.append("DIC")
        elif cm_upper == "PH":
            contrast_methods_ome.append("Phase")
        # Add other mappings as needed
        else:
            contrast_methods_ome.append(cm if cm else "Other") # Keep original if not mapped or map to Other

     # --- Determine Channel Display Names (Prioritize Filter Block) ---
    channel_display_names = []
    for c in range(channels):
        fb_name = filter_blocks[c]
        lut_name = lut_names[c]
        # Use filter block name if it's valid and not generic
        if fb_name and fb_name != "Unknown Filter":
            channel_display_names.append(fb_name)
        else:
            # Fallback to LUT name
            channel_display_names.append(lut_name)
    # --- End Channel Display Names ---

    # --- End Metadata Extraction ---

    ome_uuid = uuid.uuid4()
    # Use filename derived from save_child_name if available
    image_name = meta.get("save_child_name", os.path.splitext(filename)[0])
    # Sanitize image name for XML
    image_name = escape_xml_chars(image_name)


    xml = [
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
        "<OME xmlns=\"http://www.openmicroscopy.org/Schemas/OME/2016-06\"",
        "     xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"",
        "     xsi:schemaLocation=\"http://www.openmicroscopy.org/Schemas/OME/2016-06 "
        "http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd\"",
        f"     UUID=\"urn:uuid:{ome_uuid}\">",
    ]

    # --- Instrument Block ---
    xml.append("  <Instrument ID=\"Instrument:0\">")
    manufacturer_info = f"{escape_xml_chars(meta.get('SystemTypeName', ''))} {escape_xml_chars(meta.get('MicroscopeModel', ''))}".strip()
    xml.append(f"    <Microscope ID=\"Microscope:0\" Manufacturer=\"Leica\" Model=\"{escape_xml_chars( manufacturer_info)}\"/>")

    obj_attrs = f"ID=\"Objective:0\""
    # Add Model attribute
    if objective_model:
        obj_attrs += f" Model=\"{escape_xml_chars(objective_model)}\""
    if objective_mag is not None:
        obj_attrs += f" NominalMagnification=\"{objective_mag}\""
    if isinstance(objective_na, (int, float)) and objective_na > 0:
        obj_attrs += f" LensNA=\"{objective_na}\""
    # Add RefractiveIndex attribute if valid
    if isinstance(objective_refractive_index, (int, float)) and objective_refractive_index > 0:
        obj_attrs += f" RefractiveIndex=\"{objective_refractive_index}\""
    # Add determined Immersion type
    obj_attrs += f" Immersion=\"{immersion_type}\""
    # Include original name as description if available
    obj_attrs += f" Description=\"{escape_xml_chars(objective_name)}\""
    xml.append(f"    <Objective {obj_attrs}/>")

    xml.append("  </Instrument>")
    # --- End Instrument Block ---


    xml.extend([
        f"  <Image ID=\"Image:0\" Name=\"{image_name}\">",
        f"    <AcquisitionDate>{escape_xml_chars(acquisition_date)}</AcquisitionDate>", # Use parsed date
        f"    <InstrumentRef ID=\"Instrument:0\"/>", # Link Image to Instrument
        f"    <ObjectiveSettings ID=\"Objective:0\"/>", # Link Image to Objective used
        f"    <Pixels ID=\"Pixels:0\" DimensionOrder=\"XYZCT\" Type=\"{pixel_type}\"",
        f"            SignificantBits=\"{sbits}\"", # Add SignificantBits attribute
        f"            SizeX=\"{xs}\" SizeY=\"{ys}\" SizeZ=\"{zs}\" SizeC=\"{channels}\" SizeT=\"{ts}\"",
        f"            PhysicalSizeX=\"{xres_um}\" PhysicalSizeY=\"{yres_um}\" PhysicalSizeZ=\"{zres_um}\" "
        f"PhysicalSizeXUnit=\"µm\" PhysicalSizeYUnit=\"µm\" PhysicalSizeZUnit=\"µm\"",
        "            Interleaved=\"false\">",
    ])

    for c in range(channels):
        # Use the determined display name
        channel_name_escaped = escape_xml_chars(channel_display_names[c])
        # Use lut_names for color determination as before
        lut_name_for_color = lut_names[c]
        channel_attrs = f"ID=\"Channel:0:{c}\" Name=\"{channel_name_escaped}\" SamplesPerPixel=\"1\""

        # Add Color attribute (signed int) - based on original lut_name
        if lut_name_for_color.lower() in ["gray", "grey"]:
            channel_attrs += " Color=\"-1\"" # OME standard for grayscale
        else:
            try:
                # Use original lut_name for color lookup
                col_int_rgb = decimal_to_ome_color(color_name_to_decimal(lut_name_for_color))
                channel_attrs += f" Color=\"{col_int_rgb}\""
            except KeyError:
                 channel_attrs += " Color=\"-1\"" # Default to gray if color name unknown

        # Determine IlluminationType and AcquisitionMode based on mic_type2 and contrast method
        illumination_type_ome = "Other"
        acquisition_mode_ome = "Other"
        contrast_method_ome = contrast_methods_ome[c]  # Use mapped value

        # Map "TL-BF" to "Brightfield"
        if contrast_method_ome == "TL-BF":
            contrast_method_ome = "Brightfield"

        if mic_model.lower() == "confocal":
            illumination_type_ome = "Epifluorescence"
            acquisition_mode_ome = "LaserScanningConfocalMicroscopy"
        elif mic_model.lower() == "camera":
            acquisition_mode_ome = "WideField"
            # Refine IlluminationType based on contrast
            if contrast_method_ome == "Fluorescence":
                illumination_type_ome = "Epifluorescence"
            elif contrast_method_ome in ["Brightfield", "DIC", "Phase"]:
                illumination_type_ome = "Transmitted"
            # Add more conditions if needed (e.g., Oblique)

        # Validate metadata fields against the schema
        illumination_type_ome = validate_metadata(illumination_type_ome, "IlluminationType", metadata_schema)
        acquisition_mode_ome = validate_metadata(acquisition_mode_ome, "AcquisitionMode", metadata_schema)
        contrast_method_ome = validate_metadata(contrast_method_ome, "ContrastMethod", metadata_schema)

        # Add validated metadata
        channel_attrs += f" IlluminationType=\"{illumination_type_ome}\""
        channel_attrs += f" AcquisitionMode=\"{acquisition_mode_ome}\""
        channel_attrs += f" ContrastMethod=\"{contrast_method_ome}\""

        excitation_wl = excitations[c]
        if isinstance(excitation_wl, (int, float)) and excitation_wl > 0:
            channel_attrs += f" ExcitationWavelength=\"{excitation_wl}\" ExcitationWavelengthUnit=\"nm\""

        emission_wl = emissions[c]
        if isinstance(emission_wl, (int, float)) and emission_wl > 0:
            channel_attrs += f" EmissionWavelength=\"{emission_wl}\" EmissionWavelengthUnit=\"nm\""

        # Add PinholeSize if available and valid
        if pinhole_size_um is not None and isinstance(pinhole_size_um, (int, float)) and pinhole_size_um > 0:
             channel_attrs += f" PinholeSize=\"{pinhole_size_um}\" PinholeSizeUnit=\"µm\""

        xml.append(f"      <Channel {channel_attrs}/>")

    # One <TiffData> + <Plane> per IFD (order T slowest, C middle, Z fastest) for XYZCT
    ifd = 0
    for t in range(ts):        # T slowest
        for c in range(channels): # C middle
            for z in range(zs):   # Z fastest
                # TheC, TheZ, TheT must be zero-based indices
                xml.append(f"      <TiffData IFD=\"{ifd}\" FirstZ=\"{z}\" FirstC=\"{c}\" FirstT=\"{t}\">")
                xml.append(f"          <Plane TheZ=\"{z}\" TheC=\"{c}\" TheT=\"{t}\"/>")
                xml.append(f"      </TiffData>")
                ifd += 1

    xml.append("    </Pixels>")

    # Add original Leica XML metadata if available and requested
    if include_original_metadata:
        original_xml = meta.get("xmlElement")
        if original_xml and isinstance(original_xml, str):
            # Basic check if it looks like XML
            if original_xml.strip().startswith("<"):
                # Escape the original XML content thoroughly for embedding
                escaped_original_xml = escape_xml_chars(original_xml)
                xml.extend([
                    "    <StructuredAnnotations>",
                    "      <XMLAnnotation ID=\"Annotation:OriginalMetadata\" Namespace=\"leica.microsystems.com/TiffMetaData\">",
                    "        <Value>",
                    "          <OriginalMetadata>",
                    f"            {escaped_original_xml}",
                    "          </OriginalMetadata>",
                    "        </Value>",
                    "      </XMLAnnotation>",
                    "    </StructuredAnnotations>",
                ])

    xml.extend([
        "  </Image>",
        "</OME>",
    ])
    return "\n".join(xml)


# -----------------------------------------------------------------------------
# MAIN - Convert Leica raw into tiled, pyramidal OME-TIFF
# -----------------------------------------------------------------------------


def convert_leica_to_ometiff(inputfile: str, *, image_uuid: str = "n/a",
                       outputfolder: str | None = None, show_progress: bool = True,
                       altoutputfolder: str | None = None,
                       include_original_metadata: bool = False,
                       tempfolder: str | None = None) -> str | None:
    """High-level wrapper - multi-channel, multi-Z Leica → OME-TIFF.
    Handles tiled scans with positive overlap by stitching them into single planes.
    Handles image orientation metadata (flip/swap) for tiles.

    *RGB Leica images are skipped (function returns ``None``).*
    *Negative overlap images are skipped (function returns ``None``).*
    
    Parameters:
    - tempfolder: Optional custom temp folder for intermediate files. If None, uses system temp.
    """

    if pyvips is None:
        raise RuntimeError("pyvips is required for OME-TIFF conversion, but could not be imported.")

    try:
        meta = read_image_metadata(inputfile, image_uuid)
    except (ValueError, FileNotFoundError, IndexError, KeyError, json.JSONDecodeError) as e:
        print(f"\nError reading metadata for UUID {image_uuid} from {inputfile}: {e}")
        return None

    # with open(r"c:\xml3.xml", "w") as f:
    #     f.write(meta["xmlElement"])

    if meta.get("isrgb"):
        print(f"Image UUID {image_uuid} is RGB - skipping OME-TIFF conversion.")
        return None

    if meta.get("OverlapIsNegative"):
        print(f"Image UUID {image_uuid} overlap is Negative - skipping OME-TIFF conversion.")
        return None

    # Get orientation flags from metadata, defaulting to 0 (False)
    do_swapxy = meta.get("swapxy", 0)

    required_fields = ["xs", "ys", "channels", "filetype"]
    if meta["filetype"].lower() == ".lif":
        required_fields.extend(["LIFFile", "Position"])
    elif meta["filetype"].lower() in [".lof", ".xlef"]:
        required_fields.extend(["LOFFilePath"])
    # Add tile fields if it's potentially a tilescan
    if meta.get("tiles", 1) > 1:
        required_fields.extend(["tiles", "tile_positions"])


    missing_fields = [field for field in required_fields if field not in meta or meta[field] is None]
    if missing_fields:
        print(f"\nError: Missing essential metadata fields: {', '.join(missing_fields)}")
        return None

    xs_orig, ys_orig = meta["xs"], meta["ys"]
    zs, channels = meta.get("zs", 1), meta["channels"]
    ts = meta.get("ts", 1)

    # --- tilescan stitching setup ---
    tiles = meta.get("tiles", 1)
    tile_positions = meta.get("tile_positions", [])
    is_tilescan = tiles > 1 and bool(tile_positions)
    tile_width, tile_height = xs_orig, ys_orig # Dimensions of a tile slot on the canvas
    # OverlapPercentageX/Y can be negative, indicating a gap between tiles.
    # The formulas used for step_x/step_y and stitched dimensions correctly handle this.
    overlap_x = meta.get("OverlapPercentageX", 0.0)
    overlap_y = meta.get("OverlapPercentageY", 0.0)

    # canvas_xs, canvas_ys are the dimensions of the full image plane before global orientation changes
    canvas_xs, canvas_ys = xs_orig, ys_orig
    if is_tilescan:
        # compute grid size
        xlist = [pos.get("FieldX", 0) for pos in tile_positions]
        ylist = [pos.get("FieldY", 0) for pos in tile_positions]
        xdim, ydim = (max(xlist) + 1) if xlist else 0, (max(ylist) + 1) if ylist else 0
        
        # Calculate stitched dimensions considering overlap
        # step_x/y is the distance moved from the top-left of one tile to the top-left of the next
        step_x = tile_width * (1.0 - overlap_x)
        step_y = tile_height * (1.0 - overlap_y)
        
        stitched_xs = int((xdim - 1) * step_x + tile_width) if xdim > 0 else 0
        stitched_ys = int((ydim - 1) * step_y + tile_height) if ydim > 0 else 0
        
        # Ensure dimensions are at least tile_width/height if xdim/ydim is 1
        if xdim == 1: stitched_xs = tile_width
        if ydim == 1: stitched_ys = tile_height
        if xdim == 0: stitched_xs = 0
        if ydim == 0: stitched_ys = 0


        print(f"Stitched dimensions: {stitched_xs} x {stitched_ys}")
        # Update meta for OME-XML with stitched dimensions (pre-swap)
        # These will be further updated if global swapxy is applied
        meta["xs"], meta["ys"] = stitched_xs, stitched_ys # Store pre-transform canvas size
        canvas_xs, canvas_ys = stitched_xs, stitched_ys
    else:
        # meta["xs"], meta["ys"] are already xs_orig, ys_orig
        canvas_xs, canvas_ys = xs_orig, ys_orig

    # --- Dimension Validation ---
    if not (isinstance(canvas_xs, int) and canvas_xs > 0 and isinstance(canvas_ys, int) and canvas_ys > 0 and
            isinstance(zs, int) and zs > 0 and isinstance(channels, int) and channels > 0 and
            isinstance(ts, int) and ts > 0):
        print(f"\nError: Invalid canvas dimensions (canvas_xs={canvas_xs}, canvas_ys={canvas_ys}, zs={zs}, channels={channels}, ts={ts})")
        return None

    res = meta.get("channelResolution", [16])
    if not isinstance(res, list) or not all(isinstance(r, int) for r in res):
        print(f"\nWarning: Invalid channelResolution format: {res}. Defaulting to 16-bit.")
        res = [16] * channels

    bits = 16 if max(res) > 8 else 8
    dtype = np.uint16 if bits == 16 else np.uint8
    vips_format = dtype_to_format[dtype]

    # --- Prepare temp source for robust reading from network files ---
    temp_source_cleanup = None
    try:
        temp_source_path, temp_base_pos, temp_source_cleanup = prepare_temp_source(
            inputfile=inputfile,
            image_uuid=image_uuid,
            metadata=meta,
            tempfolder=tempfolder,
            show_progress=show_progress
        )
        base_file = temp_source_path
        base_pos = temp_base_pos
    except Exception as e:
        print(f"\nError preparing temp source: {e}")
        return None

    # --- Get Byte Increments ---
    cbytesinc = meta.get("channelbytesinc") 
    zbytesinc = meta.get("zbytesinc")
    tbytesinc = meta.get("tbytesinc")

    try:
        os.makedirs(outputfolder, exist_ok=True)
        if altoutputfolder:
            os.makedirs(altoutputfolder, exist_ok=True)
    except OSError as e:
        print(f"\nError creating output directory: {e}")
        return None

    ome_name = f"{meta.get('save_child_name', f'ometiff_output_{image_uuid}')}.ome.tiff"
    out_path = os.path.join(outputfolder, ome_name)

    final_planar_height = channels * zs * ts * canvas_ys
    mmap_path = ""
    planar = None
    img = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".planar.mmap", delete=False) as tmp_f:
            mmap_path = tmp_f.name
        if not os.path.exists(mmap_path):
            open(mmap_path, 'w').close()

        planar = np.memmap(mmap_path, dtype=dtype, mode="w+", shape=(final_planar_height, canvas_xs))

        planes_total = channels * zs * ts
        progress_per_plane = 80.0 / max(1, planes_total) # Total progress contribution for one C,Z,T plane (5% to 85%)
        plane_idx = 0 # Counter for fully processed planes

        if show_progress:
            print_progress_bar(5, prefix="Converting to OME-TIFF:", suffix="Reading raw data")

        for t in range(ts):
            for c in range(channels):
                for z in range(zs):
                    # curr_progress_before_this_plane is the progress % achieved *before* starting work on the current plane
                    curr_progress_before_this_plane = 5 + plane_idx * progress_per_plane
                    
                    plane_identity_suffix = f"T={t + 1}/{ts} C={c + 1}/{channels} Z={z + 1}/{zs}"

                    planar_start_row = (t * channels * zs + c * zs + z) * canvas_ys

                    # tilescan branch
                    if is_tilescan:
                        num_tiles_in_plane = len(tile_positions)
                        # progress_increment_per_tile is the fraction of progress_per_plane that each tile contributes
                        progress_increment_per_tile = progress_per_plane / max(1, num_tiles_in_plane)
                        
                        # Determine update interval for progress bar during tile stitching
                        if num_tiles_in_plane <= 20:
                            update_interval_for_tiles = 1  # Update after every tile
                        else:
                            update_interval_for_tiles = math.ceil(num_tiles_in_plane / 20.0)


                        for pos_idx, pos_data in enumerate(tile_positions):
                            num = pos_data.get("num")
                            if num is None:
                                print(f"\nWarning: Tile at index {pos_idx} missing 'num'. Skipping.")
                                continue
                            tile_off = (num - 1) * meta.get("tilesbytesinc", 0)
                            base_pos_tile = base_pos + tile_off

                            # Determine read dimensions for this tile based on tilescan_swapxy
                            # tile_width and tile_height are the dimensions of the slot on the canvas
                            read_w, read_h = tile_width, tile_height
                            if do_swapxy: # Note: This is global do_swapxy, not tilescan_swapxy for read_rows
                                read_w, read_h = tile_height, tile_width
                            
                            try:
                                slab = read_rows(
                                    base_file=base_file,
                                    base_pos=base_pos_tile,
                                    xs=read_w, # Use potentially swapped width for reading
                                    row_start=0, row_end=read_h, skip=1, # Use potentially swapped height for reading
                                    channel=c, bits=bits,
                                    zbytes=zbytesinc,
                                    cbytesinc=cbytesinc,
                                    zs=zs, project=False, target_z=z, # Pass correct zs and ts for read_rows
                                    timepoint=t, tbytes=tbytesinc, ts=ts
                                )
                                # slab shape is (read_h, read_w)

                                # Apply tile-specific transformations (using meta.get('tilescan_flipx') etc.)
                                tile_do_flipx = meta.get("tilescan_flipx", 0)
                                tile_do_flipy = meta.get("tilescan_flipy", 0)
                                tile_do_swapxy = meta.get("tilescan_swapxy", 0)

                                if tile_do_swapxy:
                                    if tile_do_flipy: slab = slab[::-1, :] # Applied to original width dimension
                                    if tile_do_flipx: slab = slab[:, ::-1] # Applied to original height dimension
                                    slab = slab.T 
                                else:
                                    if tile_do_flipy: slab = slab[::-1, :]
                                    if tile_do_flipx: slab = slab[:, ::-1]
                                # Now slab shape is (tile_height, tile_width) matching canvas slot

                            except (IndexError, ValueError, OSError, FileNotFoundError) as e:
                                print(f"\nError reading or transforming tile {num} (Z={z}, T={t}): {e}")
                                del planar
                                planar = None
                                gc.collect()
                                if os.path.exists(mmap_path):
                                    os.remove(mmap_path)
                                return None
                            
                            # placement using same formula as RGB version
                            # tile_width and tile_height here refer to the canvas slot dimensions
                            # step_x/y already account for overlap for positioning tile origins
                            current_step_x = tile_width * (1.0 - overlap_x)
                            current_step_y = tile_height * (1.0 - overlap_y)
                            xstart = int(pos_data.get("FieldX", 0) * current_step_x)
                            ystart_plane = int(pos_data.get("FieldY", 0) * current_step_y)
                            
                            y_abs = planar_start_row + ystart_plane
                            # The slab is now (tile_height, tile_width)
                            # So it fits into a region of tile_height rows and tile_width columns
                            xend = xstart + tile_width 
                            yend = y_abs + tile_height

                            # Bounds check against the (potentially stitched) planar dimensions
                            if y_abs < final_planar_height and xstart < canvas_xs:
                                # Calculate actual region to write to, respecting planar boundaries
                                y_slice = slice(y_abs, min(yend, final_planar_height))
                                x_slice = slice(xstart, min(xend, canvas_xs))
                                
                                # Calculate corresponding slice from slab
                                slab_y_len = y_slice.stop - y_slice.start
                                slab_x_len = x_slice.stop - x_slice.start

                                if slab.shape[0] == tile_height and slab.shape[1] == tile_width:
                                    planar[y_slice, x_slice] = slab[:slab_y_len, :slab_x_len]
                                else:
                                    print(f"\nWarning: Tile {num} transformed shape ({slab.shape}) does not match slot shape ({tile_height}, {tile_width}). Skipping placement.")
                            else:
                                print(f"\nWarning: Tile {num} placement start ({y_abs}, {xstart}) out of bounds for planar array ({final_planar_height}, {canvas_xs}). Skipping.")

                            # Update progress after each tile is processed, but only at intervals
                            if show_progress:
                                # Calculate accurate progress regardless of update display
                                progress_made_by_tiles_so_far_in_plane = (pos_idx + 1) * progress_increment_per_tile
                                overall_progress_at_this_tile = curr_progress_before_this_plane + progress_made_by_tiles_so_far_in_plane
                                
                                # Check if it's time to update the progress bar
                                if (pos_idx + 1) % update_interval_for_tiles == 0 or (pos_idx + 1) == num_tiles_in_plane:
                                    tile_specific_suffix = f"{plane_identity_suffix} Tile={pos_idx + 1}/{num_tiles_in_plane}"
                                    print_progress_bar(overall_progress_at_this_tile, prefix="Converting to OME-TIFF:", suffix=tile_specific_suffix)
                    else: # Not a tilescan
                        try:
                            # For non-tilescan, read dimensions are xs_orig, ys_orig
                            slab = read_rows(
                                base_file=base_file,
                                base_pos=base_pos,
                                xs=xs_orig, 
                                row_start=0, row_end=ys_orig, skip=1,
                                channel=c,
                                bits=bits,
                                zbytes=zbytesinc,
                                cbytesinc=cbytesinc,
                                zs=zs, 
                                project=False,
                                target_z=z,
                                timepoint=t,
                                tbytes=tbytesinc,
                                ts=ts
                            )

                        except (IndexError, ValueError, OSError, FileNotFoundError) as e:
                            print(f"\nError reading data for T={t}, C={c}, Z={z}: {e}")
                            del planar
                            planar = None
                            gc.collect()
                            if os.path.exists(mmap_path):
                                os.remove(mmap_path)
                            return None

                        try:
                            if slab.shape[0] == ys_orig and slab.shape[1] == xs_orig:
                                planar[planar_start_row : planar_start_row + ys_orig, 0:xs_orig] = slab
                            else:
                                # This case indicates a mismatch after transformations, should be an error or warning.
                                print(f"\nError: Slab shape {slab.shape} does not match target planar slice shape ({target_h_slice}, {target_w_slice}) for non-tilescan.")
                                # Handle error appropriately: skip, raise, etc. For now, printing.
                                raise ValueError("Slab shape mismatch for non-tilescan placement.")

                        except ValueError as place_err:
                            print(f"\nError placing plane data into planar array (T={t}, C={c}, Z={z}). Shape mismatch?")
                            print(f"  Target planar slice shape: ({ys_orig}, {xs_orig})")
                            print(f"  Source slab shape: {slab.shape}")
                            print(f"  Original error: {place_err}")
                            del planar
                            planar = None
                            gc.collect()
                            if os.path.exists(mmap_path):
                                os.remove(mmap_path)
                            return None
                        
                        # After slab is read and placed, the work for this plane is done.
                        if show_progress:
                            progress_after_this_plane_completed = curr_progress_before_this_plane + progress_per_plane
                            print_progress_bar(progress_after_this_plane_completed, prefix="Converting to OME-TIFF:", suffix=f"Finished {plane_identity_suffix}")

                    plane_idx += 1

        planar.flush()

        if show_progress:
            print_progress_bar(90, prefix="Converting to OME-TIFF:", suffix="Creating pyvips image")

        img = pyvips.Image.new_from_memory(planar, canvas_xs, final_planar_height, 1, vips_format)

        del planar
        planar = None
        gc.collect()

        meta["xs"] = canvas_xs
        meta["ys"] = canvas_ys

        if show_progress:
            print_progress_bar(95, prefix="Converting to OME-TIFF:", suffix="Embedding OME-XML")

        ome_xml = generate_ome_xml(meta, ome_name, include_original_metadata=include_original_metadata)
        img = img.copy()
        img.set_type(pyvips.GValue.gstr_type, "image-description", ome_xml.encode("utf-8"))

        if show_progress:
            print_progress_bar(100, prefix="Converting to OME-TIFF:", suffix="Processing complete", final_call=True)

        # Use safe_tiffsave for robust saving to potentially network-mounted destinations
        # This has its own progress bar for the save phase
        safe_tiffsave(
            img, out_path, altoutputfolder=altoutputfolder,
            show_progress=show_progress,
            tempfolder=tempfolder,
            tile=True, tile_width=512, tile_height=512,
            pyramid=True, subifd=True, compression="lzw",
            page_height=canvas_ys,  # Use final plane height (after global swap) for IFD separation
            bigtiff=True
        )

        img = None
        gc.collect()

        return ome_name

    except pyvips.Error as e:
        print(f"\nError during pyvips processing: {e}")
        img = None
        planar = None
        gc.collect()
        return None
    except MemoryError:
        print("\nError: Insufficient memory to process the image.")
        img = None
        planar = None
        gc.collect()
        return None
    except Exception as e:
        import traceback
        print(f"\nAn unexpected error occurred during OME-TIFF conversion:")
        print(traceback.format_exc())
        img = None
        planar = None
        gc.collect()
        return None
    finally:
        gc.collect()
        if planar is not None:
            del planar
        if img is not None:
            del img
        if mmap_path and os.path.exists(mmap_path):
            try:
                os.remove(mmap_path)
            except OSError as e:
                print(f"\nWarning: Could not remove temporary file {mmap_path}: {e}")
        # Clean up temp source file
        cleanup_temp_source(temp_source_cleanup, show_progress=show_progress)

