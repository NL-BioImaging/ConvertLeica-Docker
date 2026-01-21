import os
import json
import uuid
import tempfile
import numpy as np
import gc
import re
from html import escape as escape_xml_chars
import math # Added for math.ceil
import sys

# Import helpers from the dedicated module
from ci_leica_converters_helpers import (
    dtype_to_format,
    print_progress_bar,
    read_image_metadata,
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
 # Low-level reader - pull a *single* interleaved RGB Z/T plane 
 # ----------------------------------------------------------------------------- 
 

def read_interleaved_rgb_plane(base_file: str, base_pos: int, xs: int, ys: int,
                               bits: int, zbytes: int, tbytes: int,
                               target_z: int, timepoint: int,
                               zs: int, ts: int) -> np.ndarray:
     """Return a 3-D numpy (row, col, channel) slice for one Z/T plane from Leica raw RGB data."""

     dtype = np.uint16 if bits == 16 else np.uint8
     bpp = bits // 8
     bytes_per_pixel = 3 * bpp # 3 channels interleaved
     plane_bytes = xs * ys * bytes_per_pixel
     arr = np.zeros((ys, xs, 3), dtype=dtype)

     if not isinstance(base_pos, int) or base_pos < 0:
         raise ValueError(f"Invalid base position offset: {base_pos}")
     if not (0 <= target_z < zs):
         raise IndexError(f"Target Z index {target_z} out of range for zs size {zs}")
     if not (0 <= timepoint < ts):
         raise IndexError(f"Timepoint index {timepoint} out of range for ts size {ts}")
     if tbytes is None: tbytes = 0 # Handle None
     if zbytes is None: zbytes = 0 # Handle None

     try:
         with open(base_file, "rb") as fh:
             # Calculate position based on T > Z > YX(interleaved C) layout
             # base_pos is the start of the specific tile's data block
             pos = base_pos + timepoint * tbytes + target_z * zbytes
             fh.seek(pos)
             plane_data = fh.read(plane_bytes)

             if len(plane_data) < plane_bytes:
                 print(f"Warning: Read fewer bytes ({len(plane_data)}) than expected ({plane_bytes}) for plane Z={target_z}, T={timepoint}. Padding with zeros.")
                 # Create a buffer of the expected size filled with zeros
                 full_plane_buffer = bytearray(plane_bytes)
                 # Copy the read data into the beginning of the buffer
                 full_plane_buffer[:len(plane_data)] = plane_data
                 # Convert the potentially padded buffer
                 arr = np.frombuffer(full_plane_buffer, dtype=dtype).reshape((ys, xs, 3))
             else:
                 arr = np.frombuffer(plane_data, dtype=dtype).reshape((ys, xs, 3))

     except FileNotFoundError:
         raise FileNotFoundError(f"Base data file not found: {base_file}")
     except OSError as e:
         raise OSError(f"Error reading from base file {base_file} at offset {pos}: {e}")
     except ValueError as e:
         # Catch potential reshape errors if data size is wrong
         raise ValueError(f"Error reshaping data for plane Z={target_z}, T={timepoint}. Check dimensions and data format. Original error: {e}")

     return arr

 # ----------------------------------------------------------------------------- 
 # Generate OME-XML (XYZCT) including original Leica metadata annotation 
 # ----------------------------------------------------------------------------- 


# Note: This function now relies on validate_metadata and metadata_schema imported from helpers
def generate_ome_xml(meta: dict, filename: str, *, include_original_metadata: bool = False) -> str:
    """Return OME-XML including original Leica XML as an annotation. Assumes RGB input."""

    xs, ys = meta["xs"], meta["ys"]
    zs = meta.get("zs", 1)
    xres = meta.get("xres", 0.0)  # Physical size in METERS from Leica readers
    yres = meta.get("yres", 0.0)
    zres = meta.get("zres", 0.0)
    ts = meta.get("ts", 1)
    # Get acquisition date from metadata, default to "Unknown"
    acquisition_date = meta.get("experiment_datetime_str", "Unknown")
    if not acquisition_date: # Handle empty string case
        acquisition_date = "Unknown"

    # --- Determine Actual Channel Count and Pixel Type ---
    channels = 3 # Explicitly 3 channels for RGB

    # Determine bit depth from metadata, default to 16 if missing/invalid
    res = meta.get("channelResolution", [16]) # Default to 16-bit if not specified
    if not isinstance(res, list) or not res or not all(isinstance(r, int) for r in res):
        print(f"\nWarning: Invalid channelResolution format for RGB: {res}. Defaulting to 16-bit.")
        res = [16] * channels # Assume 16-bit for all RGB channels if invalid
    # Use the maximum resolution found if multiple are listed (though unlikely for RGB)
    bits = max(res) if res else 16
    sbits=bits # Store original significant bits for later use in TIFF save
    # Ensure bits is either 8 or 16, default to 16 otherwise
    if bits > 8:
        bits = 16
        pixel_type = "uint16"
    else:
        bits = 8
        pixel_type = "uint8"

    # Convert meters to micrometers for OME standard units
    xres_um = xres * 1_000_000 if xres > 0 else 1.0  # Default to 1.0 if zero/invalid
    yres_um = yres * 1_000_000 if yres > 0 else 1.0
    zres_um = zres * 1_000_000 if zres > 0 else 1.0

    # --- Instrument & Channel Metadata Extraction ---
    objective_name = meta.get("objective", "Unknown Objective")
    objective_na = meta.get("na", 0.0)
    objective_model = objective_name # Use name as model for now
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
    xml.append(f"    <Microscope ID=\"Microscope:0\" Manufacturer=\"Leica\" Model=\"{escape_xml_chars(manufacturer_info)}\"/>") # Changed Model source

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
    ])

    # ---------------------------------------------------------------
    # --- <Pixels> (base-resolution image, interleaved RGB) ---------
    # ---------------------------------------------------------------
    pixels_attrs = [
        'ID="Pixels:0"',
        'DimensionOrder="XYZCT"',
        f'Type="{pixel_type}"', # Use determined pixel type
        f'SignificantBits="{sbits}"', # Add SignificantBits attribute
        f'SizeX="{xs}" SizeY="{ys}" SizeZ="{zs}"',
        'SizeC="1"',                        # ⇦ one logical channel
        f'SizeT="{ts}"',
        f'PhysicalSizeX="{xres_um}" PhysicalSizeY="{yres_um}" PhysicalSizeZ="{zres_um}"',
        'PhysicalSizeXUnit="µm" PhysicalSizeYUnit="µm" PhysicalSizeZUnit="µm"',
        'Interleaved="true"'
    ]
    xml.append(f'    <Pixels {" ".join(pixels_attrs)}>')

    # --- one Channel, SamplesPerPixel = TIFF SamplesPerPixel (3) ---
    channel_attrs = [
        'ID="Channel:0:0"',
        'Name="RGB"',                      # any label you like
        'SamplesPerPixel="3"',             # ⇦ matches TIFF tag
        'IlluminationType="Transmitted"',
        'AcquisitionMode="WideField"',
        'ContrastMethod="Brightfield"'
    ]
    xml.append(f'      <Channel {" ".join(channel_attrs)}/>')

    # ---------------------------------------------------------------
    # --- <TiffData> map --------------------------------------------
    # one entry, PlaneCount = zs*ts  (all planes are contiguous)
    # ---------------------------------------------------------------
    plane_count = zs * ts
    xml.append(
        f'      <TiffData IFD="0" FirstZ="0" FirstC="0" FirstT="0" '
        f'PlaneCount="{plane_count}"/>'
    )

    xml.append('    </Pixels>')

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
 # MAIN - Convert Leica *RGB* raw into tiled, pyramidal OME-TIFF 
 # ----------------------------------------------------------------------------- 


def convert_leica_rgb_to_ometiff(inputfile: str, *, image_uuid: str = "n/a",
                                outputfolder: str | None = None, show_progress: bool = True,
                                altoutputfolder: str | None = None,
                                include_original_metadata: bool = False,
                                tempfolder: str | None = None) -> str | None:
    """High-level wrapper - Leica RGB (interleaved) data → OME-TIFF.
    Handles tiled scans by stitching them into a single plane, using byte increments.

    *Multi-channel non-RGB images are skipped (function returns ``None``).*
    
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

    # --- This function ONLY handles RGB images ---
    if not meta.get("isrgb"):
        print(f"Image UUID {image_uuid} is not RGB - skipping RGB OME-TIFF conversion.")
        return None

    required_fields = ["xs", "ys", "filetype", "isrgb"] # No 'channels' needed explicitly for RGB check
    if meta["filetype"].lower() == ".lif":
        required_fields.extend(["LIFFile", "Position"])
    elif meta["filetype"].lower() in [".lof", ".xlef"]:
        required_fields.extend(["LOFFilePath"]) # Position might be missing, default later
    # Add tile fields if it's a tilescan
    if meta.get("tiles", 1) > 1:
        required_fields.extend(["tiles", "tile_positions"])


    missing_fields = [field for field in required_fields if field not in meta or meta[field] is None]
    if missing_fields:
        print(f"\nError: Missing essential metadata fields for RGB conversion: {', '.join(missing_fields)}")
        return None

    xs, ys = meta["xs"], meta["ys"]
    zs = meta.get("zs", 1)
    ts = meta.get("ts", 1)
    channels = 3 # Hardcoded for RGB
    tiles = meta.get("tiles", 1)
    tile_positions = meta.get("tile_positions", [])

    is_tilescan = tiles > 1 and tile_positions

    tile_width = meta["xs"]
    tile_height = meta["ys"]
    stitched_xs, stitched_ys = xs, ys # Initialize stitched dimensions
    xdim, ydim = 1, 1 # Initialize tile grid dimensions

    if is_tilescan:
        print(f"Detected tilescan ({tiles} tiles). Preparing for stitching.")
        # Get original grid dimensions
        xlist = [pos.get("FieldX", 0) for pos in tile_positions]
        ylist = [pos.get("FieldY", 0) for pos in tile_positions]
        if not xlist or not ylist:
            print("\nError: Tile positions are missing FieldX or FieldY information. Cannot stitch.")
            return None
        xdim = max(xlist) + 1
        ydim = max(ylist) + 1

        # Calculate potentially swapped stitched dimensions
        stitched_xs = xdim * tile_width
        stitched_ys = ydim * tile_height
        print(f"Stitched dimensions: {stitched_xs} x {stitched_ys} ({xdim} x {ydim} tiles)")

        # Update metadata dimensions for OME-XML and TIFF saving
        meta["xs"] = stitched_xs
        meta["ys"] = stitched_ys
        # Update local xs, ys variables used later for planar array shape etc.
        xs, ys = stitched_xs, stitched_ys

    if not (isinstance(xs, int) and xs > 0 and isinstance(ys, int) and ys > 0 and
            isinstance(zs, int) and zs > 0 and isinstance(ts, int) and ts > 0):
        print(f"\nError: Invalid dimensions (xs={xs}, ys={ys}, zs={zs}, ts={ts}) for RGB image.")
        return None

    # Determine bit depth from metadata, default to 16 if missing/invalid
    res = meta.get("channelResolution", [16]) # Default to 16-bit if not specified
    if not isinstance(res, list) or not res or not all(isinstance(r, int) for r in res):
        print(f"\nWarning: Invalid channelResolution format for RGB: {res}. Defaulting to 16-bit.")
        res = [16] * channels # Assume 16-bit for all RGB channels if invalid
    # Use the maximum resolution found if multiple are listed (though unlikely for RGB)
    bits = max(res) if res else 16
    # Ensure bits is either 8 or 16, default to 16 otherwise
    if bits > 8:
        bits = 16
        dtype = np.uint16
    else:
        bits = 8
        dtype = np.uint8
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
    zbytesinc = meta.get("zbytesinc")
    tbytesinc = meta.get("tbytesinc")
    tilesbytesinc = meta.get("tilesbytesinc")

    overlap_x = meta.get("OverlapPercentageX", 0.0)
    overlap_y = meta.get("OverlapPercentageY", 0.0)

    if outputfolder is None:
        outputfolder = os.path.dirname(inputfile)
    try:
        os.makedirs(outputfolder, exist_ok=True)
        if altoutputfolder:
            os.makedirs(altoutputfolder, exist_ok=True)
    except OSError as e:
        print(f"\nError creating output directory: {e}")
        return None

    ome_name = f"{meta.get('save_child_name', f'ometiff_rgb_output_{image_uuid}')}.ome.tiff"
    out_path = os.path.join(outputfolder, ome_name)

    # Use potentially swapped STITCHED dimensions for the final planar array
    final_height = zs * ts * ys # ys is potentially stitched_ys
    mmap_path = ""
    planar = None
    img = None
    try:
        # Create a memmap file for the entire image stack (T*Z*Y, X, C)
        with tempfile.NamedTemporaryFile(suffix=".rgb_planar.mmap", delete=False) as tmp_f:
            mmap_path = tmp_f.name
        if not os.path.exists(mmap_path):
            open(mmap_path, 'w').close() # Ensure file exists before memmap

        # Shape uses potentially swapped dimensions: (total_rows, width, bands)
        planar = np.memmap(mmap_path, dtype=dtype, mode="w+", shape=(final_height, xs, 3)) # xs is potentially stitched_xs

        planes_total = zs * ts
        progress_per_plane = 80.0 / max(1, planes_total) # Total progress contribution for one Z,T plane (5% to 85%)
        plane_idx = 0 # Counter for fully processed planes

        if show_progress:
            print_progress_bar(5, prefix="Converting RGB to OME-TIFF:", suffix="Reading raw data")

        # Loop through timepoints and Z-slices
        for t in range(ts):
            for z in range(zs):
                curr_progress_before_this_plane = 5 + plane_idx * progress_per_plane
                plane_identity_suffix = f"T={t + 1}/{ts} Z={z + 1}/{zs}" # RGB implies C=1/1 effectively

                # Calculate the starting row in the large planar array for this Z/T plane
                stitched_plane_start_row = (t * zs + z) * ys # ys is potentially stitched_ys

                if is_tilescan:
                    num_tiles_in_plane = len(tile_positions)
                    progress_increment_per_tile = progress_per_plane / max(1, num_tiles_in_plane)
                    
                    if num_tiles_in_plane <= 20:
                        update_interval_for_tiles = 1
                    else:
                        update_interval_for_tiles = math.ceil(num_tiles_in_plane / 20.0)

                    # Loop through tiles for the current Z/T plane
                    for pos_idx, tile_info in enumerate(tile_positions):
                        tile_num = tile_info.get("num")
                        if tile_num is None or tile_num < 1:
                            print(f"\nError: Invalid or missing 'num' in tile_positions. Cannot calculate tile offset.")
                            # Clean up memmap before returning
                            del planar # Releases lock
                            planar = None
                            gc.collect() # Try to force collection
                            if os.path.exists(mmap_path):
                                os.remove(mmap_path)
                            return None

                        # --- Calculate base position for this specific tile's data block ---
                        tile_offset = (tile_num - 1) * tilesbytesinc
                        base_pos_for_tile = base_pos + tile_offset
                        # --- End Tile Position Calculation ---

                        try:
                            # Read the single TILE's interleaved RGB plane using ORIGINAL dimensions
                            read_xs = tile_width 
                            read_ys = tile_height
                            tile_plane_data = read_interleaved_rgb_plane(
                                base_file=base_file,
                                base_pos=base_pos_for_tile,
                                xs=read_xs, # Use original width for reading
                                ys=read_ys, # Use original height for reading
                                bits=bits,
                                zbytes=zbytesinc,
                                tbytes=tbytesinc,
                                target_z=z,
                                timepoint=t,
                                zs=zs,
                                ts=ts
                            )
                        except (IndexError, ValueError, OSError, FileNotFoundError) as e:
                            print(f"\nError reading RGB tile data for Tile {tile_num} (Calc Pos:{base_pos_for_tile}), Z={z}, T={t}: {e}")
                            # Clean up memmap before returning
                            del planar # Releases lock
                            planar = None
                            gc.collect() # Try to force collection
                            if os.path.exists(mmap_path):
                                os.remove(mmap_path)
                            return None

                        # Calculate target coordinates in the stitched planar array, accounting for swapxy
                        tile_x_idx_orig = tile_info.get("FieldX")
                        tile_y_idx_orig = tile_info.get("FieldY")
                        if tile_x_idx_orig is None or tile_y_idx_orig is None:
                             print(f"\nError: Tile {tile_num} missing FieldX/FieldY. Skipping placement.")
                             continue

                        xstart = int(tile_x_idx_orig * (tile_width - tile_width * overlap_x))
                        xend = xstart + tile_width
                        ystart_in_plane = int(tile_y_idx_orig * (tile_height - tile_height * overlap_y))
                        yend_in_plane = ystart_in_plane + tile_height

                        # Calculate absolute y start/end in the full memmap
                        ystart_abs = stitched_plane_start_row + ystart_in_plane
                        yend_abs = stitched_plane_start_row + yend_in_plane

                        # Check bounds before placing data (using potentially swapped xs, final_height)
                        if yend_abs <= final_height and xend <= xs:
                            try:
                                # Place the transformed tile data
                                planar[ystart_abs:yend_abs, xstart:xend, :] = tile_plane_data
                            except ValueError as place_err:
                                print(f"\nError placing RGB tile {tile_num} data into planar array (Z={z}, T={t}). Shape mismatch?")
                                print(f"  Target planar slice shape: ({yend_abs-ystart_abs}, {xend-xstart}, 3) = ({ys}, {xs}, 3)")
                                print(f"  Source tile_plane_data shape: {tile_plane_data.shape}")
                                print(f"  Original error: {place_err}")
                                # Clean up and return None
                                del planar
                                planar = None
                                gc.collect()
                                if os.path.exists(mmap_path):
                                    os.remove(mmap_path)
                                return None
                        else:
                            print(f"\nWarning: Tile {tile_num} placement ({ystart_abs}:{yend_abs}, {xstart}:{xend}) out of bounds for planar array ({final_height}, {xs}). Skipping.")
                        
                        if show_progress:
                            progress_made_by_tiles_so_far = (pos_idx + 1) * progress_increment_per_tile
                            overall_progress_at_this_tile = curr_progress_before_this_plane + progress_made_by_tiles_so_far
                            if (pos_idx + 1) % update_interval_for_tiles == 0 or (pos_idx + 1) == num_tiles_in_plane:
                                tile_specific_suffix = f"{plane_identity_suffix} Tile={pos_idx + 1}/{num_tiles_in_plane}"
                                print_progress_bar(overall_progress_at_this_tile, prefix="Converting RGB to OME-TIFF:", suffix=tile_specific_suffix)

                else: # Not a tilescan, read the single plane directly
                    try:
                        # Read the single plane using potentially swapped dimensions
                        plane_data = read_interleaved_rgb_plane(
                            base_file=base_file,
                            base_pos=base_pos,
                            xs=xs, # Use current width for reading
                            ys=ys, # Use current height for reading
                            bits=bits,
                            zbytes=zbytesinc,
                            tbytes=tbytesinc,
                            target_z=z,
                            timepoint=t,
                            zs=zs,
                            ts=ts
                        )
                    except (IndexError, ValueError, OSError, FileNotFoundError) as e:
                        print(f"\nError reading RGB data for Z={z}, T={t}: {e}")
                        # Clean up memmap before returning
                        del planar # Releases lock
                        planar = None
                        gc.collect() # Try to force collection
                        if os.path.exists(mmap_path):
                            os.remove(mmap_path)
                        return None

                    try:
                        # Place the transformed plane data (ys is potentially swapped height)
                        start_row = stitched_plane_start_row
                        planar[start_row : start_row + ys, :, :] = plane_data
                    except ValueError as place_err:
                        print(f"\nError placing non-tile RGB plane data into planar array (Z={z}, T={t}). Shape mismatch?")
                        print(f"  Target planar slice shape: ({ys}, {xs}, 3)") # Use current xs/ys
                        print(f"  Source plane_data shape: {plane_data.shape}")
                        print(f"  Original error: {place_err}")
                        # Clean up and return None
                        del planar
                        planar = None
                        gc.collect()
                        if os.path.exists(mmap_path):
                            os.remove(mmap_path)
                        return None

                    if show_progress:
                        progress_after_this_plane_completed = curr_progress_before_this_plane + progress_per_plane
                        print_progress_bar(progress_after_this_plane_completed, prefix="Converting RGB to OME-TIFF:", suffix=f"Finished {plane_identity_suffix}")

                plane_idx += 1

        planar.flush() # Ensure all data is written to the memmap file before pyvips reads it

        if show_progress:
            print_progress_bar(90, prefix="Converting RGB to OME-TIFF:", suffix="Creating pyvips image")

        # Create pyvips image from the complete memmap array
        # Height = final_height (zs * ts * stitched_ys), Width = stitched_xs, Bands = 3
        img = pyvips.Image.new_from_memory(planar, xs, final_height, 3, vips_format)

        # Explicitly delete the memmap object reference BEFORE deleting the file
        del planar
        planar = None
        gc.collect() # Encourage garbage collection

        if show_progress:
            print_progress_bar(95, prefix="Converting RGB to OME-TIFF:", suffix="Embedding OME-XML")

        # Generate OME-XML specifically for this RGB image (using updated meta['xs'], meta['ys'])
        ome_xml = generate_ome_xml(meta, ome_name, include_original_metadata=include_original_metadata)
        img = img.copy()
        img.set_type(pyvips.GValue.gstr_type, "image-description", ome_xml.encode("utf-8"))

        if show_progress:
            print_progress_bar(100, prefix="Converting RGB to OME-TIFF:", suffix="Processing complete", final_call=True)

        # Use safe_tiffsave for robust saving to potentially network-mounted destinations
        # This has its own progress bar for the save phase
        # Save as tiled, pyramidal OME-TIFF
        # page_height=ys ensures each Z/T plane becomes a separate IFD (uses stitched_ys)
        safe_tiffsave(
            img, out_path, altoutputfolder=altoutputfolder,
            show_progress=show_progress,
            tempfolder=tempfolder,
            tile=True, tile_width=512, tile_height=512,
            pyramid=True, subifd=True, compression="lzw",  # Use LZW for RGB
            page_height=ys,  # Critical for correct Z/T plane separation (uses stitched_ys)
            bigtiff=True
        )

        img = None # Release pyvips image object
        # No need to delete mmap_path here, finally block handles it

        gc.collect()

        return ome_name

    except pyvips.Error as e:
        print(f"\nError during pyvips processing for RGB image: {e}")
        img = None
        planar = None
        gc.collect()
        return None
    except MemoryError:
        print("\nError: Insufficient memory to process the RGB image.")
        img = None
        planar = None
        gc.collect()
        return None
    except Exception as e:
        import traceback
        print(f"\nAn unexpected error occurred during RGB OME-TIFF conversion:")
        print(traceback.format_exc())
        img = None
        planar = None
        gc.collect()
        return None
    finally:
        # Ensure cleanup happens
        # Make sure references are gone before trying to delete the file
        if planar is not None:
            del planar # Release lock if not already deleted
            planar = None
        if img is not None:
            del img
            img = None
    
        gc.collect() # Force garbage collection again before file deletion

        if mmap_path and os.path.exists(mmap_path):
            try:
                os.remove(mmap_path)
            except OSError as e:
                print(f"\nWarning: Could not remove temporary file {mmap_path}: {e}") # Report error if deletion fails
        
        # Clean up temp source file
        cleanup_temp_source(temp_source_cleanup, show_progress=show_progress)
