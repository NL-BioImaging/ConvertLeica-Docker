import os
import json
import shutil

from ci_leica_converters_single_lif import convert_leica_to_singlelif
from ci_leica_converters_ometiff import convert_leica_to_ometiff
from ci_leica_converters_ometiff_rgb import convert_leica_rgb_to_ometiff
from ci_leica_converters_helpers import read_image_metadata, _read_xlef_image, _find_image_hierarchical_path, compute_channel_intensity_stats


def _format_display_name(inputfile: str, save_child_name: str | None, fallback_name: str) -> str:
    """
    Format the display name for the JSON output.
    
    For files with a hierarchical path (save_child_name), formats as:
        "basename.ext [path_inside_file]"
    e.g., "RGB-Multilevel.lif [collection1_collection2_LargeImage]"
    
    Args:
        inputfile: Path to the input LIF/XLEF file
        save_child_name: Hierarchical name like "RGB-Multilevel-lif_collection1_collection2_LargeImage"
        fallback_name: Name to use if save_child_name is not available
    
    Returns:
        Formatted display name
    """
    if not save_child_name:
        return fallback_name
    
    # Get the base filename and extension
    input_basename = os.path.splitext(os.path.basename(inputfile))[0]
    input_ext = os.path.splitext(inputfile)[1].lower()  # e.g., ".lif" or ".xlef"
    ext_suffix = input_ext.lstrip(".")  # e.g., "lif" or "xlef"
    
    # The save_child_name now includes the extension: "basename-ext_path"
    # Check if save_child_name starts with the input basename + extension pattern
    prefix_with_ext = f"{input_basename}-{ext_suffix}_"
    if save_child_name.startswith(prefix_with_ext):
        # Extract the path inside the file (everything after "basename-ext_")
        path_inside = save_child_name[len(prefix_with_ext):]
        return f"{input_basename}{input_ext} [{path_inside}]"
    else:
        # Fallback: just use save_child_name as-is
        return save_child_name


def _format_save_child_name_with_ext(inputfile: str, save_child_name: str | None) -> str | None:
    """
    Insert the file extension into the save_child_name for the output filename.
    
    Transforms: "RGB-Multilevel_Collection1_Collection2_SmallImage"
    To: "RGB-Multilevel-lif_Collection1_Collection2_SmallImage" (for .lif files)
    Or: "RGB-Multilevel-xlef_Collection1_Collection2_SmallImage" (for .xlef files)
    
    Args:
        inputfile: Path to the input LIF/XLEF file
        save_child_name: Hierarchical name like "RGB-Multilevel_collection1_collection2_LargeImage"
    
    Returns:
        Modified save_child_name with extension inserted, or None if save_child_name is None
    """
    if not save_child_name:
        return None
    
    # Get the base filename and extension
    input_basename = os.path.splitext(os.path.basename(inputfile))[0]
    input_ext = os.path.splitext(inputfile)[1].lower().lstrip(".")  # e.g., "lif" or "xlef"
    
    # Check if save_child_name starts with the input basename followed by underscore
    if save_child_name.startswith(input_basename + "_"):
        # Insert the extension after the basename
        # "RGB-Multilevel_rest" -> "RGB-Multilevel-lif_rest"
        rest = save_child_name[len(input_basename):]  # "_rest"
        return f"{input_basename}-{input_ext}{rest}"
    else:
        # Fallback: just return as-is
        return save_child_name


def _add_optional_metadata(kv: dict, metadata: dict) -> None:
    """
    Add optional metadata fields to keyvalues dict if they are available in metadata.
    Only adds fields that have meaningful values (not None, empty, or default zeros).
    Filters out fluorescence-specific fields for RGB images.
    
    Args:
        kv: The keyvalues dictionary to update
        metadata: The image metadata dictionary
    """
    # Check if this is an RGB image (brightfield/transmitted light)
    is_rgb = metadata.get("isrgb", False)
    # Check microscope type for context
    mic_type = metadata.get("mic_type", "")
    is_confocal = mic_type == "IncohConfMicr"
    is_widefield = mic_type == "IncohWFMicr"
    
    # LUT names (channel color names) - relevant for all image types
    lutnames = metadata.get("lutname")
    if lutnames and isinstance(lutnames, list) and any(lutnames):
        kv["lut_names"] = lutnames
    
    # Channel names (user-defined names like DAPI, GFP, etc.) - relevant for all image types
    channel_names = metadata.get("channel_names")
    if channel_names and isinstance(channel_names, list) and any(channel_names):
        kv["channel_names"] = channel_names
    
    # Fluorescence-specific fields - only for non-RGB images
    if not is_rgb:
        # Excitation wavelengths
        excitation = metadata.get("excitation")
        if excitation and isinstance(excitation, list) and any(e and e > 0 for e in excitation):
            kv["excitation_wavelengths"] = [int(e) if e else 0 for e in excitation]
        
        # Emission wavelengths
        emission = metadata.get("emission")
        if emission and isinstance(emission, list) and any(e and e > 0 for e in emission):
            kv["emission_wavelengths"] = [int(e) if e else 0 for e in emission]
        
        # Filter block names
        filterblock = metadata.get("filterblock")
        if filterblock and isinstance(filterblock, list) and any(f and f != "Unknown" for f in filterblock):
            kv["filter_blocks"] = filterblock
        
        # Contrast methods (for widefield fluorescence)
        contrastmethod = metadata.get("contrastmethod")
        if contrastmethod and isinstance(contrastmethod, list) and any(contrastmethod):
            kv["contrast_methods"] = contrastmethod
    
    # Experiment name
    exp_name = metadata.get("experiment_name")
    if exp_name:
        kv["experiment_name"] = exp_name
    
    # Experiment datetime
    exp_datetime = metadata.get("experiment_datetime") or metadata.get("experiment_datetime_str")
    if exp_datetime:
        kv["experiment_datetime"] = exp_datetime
    
    # Microscope type
    mic_type = metadata.get("mic_type")
    if mic_type and mic_type != "unknown":
        kv["microscope_type"] = mic_type
    
    # System type name and microscope model
    system_type = metadata.get("SystemTypeName")
    if system_type:
        kv["system_type_name"] = system_type
    
    mic_model = metadata.get("MicroscopeModel")
    if mic_model:
        kv["microscope_model"] = mic_model
    
    # System serial number (microscope identification)
    system_serial = metadata.get("system_serial_number")
    if system_serial:
        kv["system_serial_number"] = system_serial
    
    # Gamma values for display (per channel)
    gamma_values = metadata.get("gammavalue")
    if gamma_values and isinstance(gamma_values, list) and any(g != 1 for g in gamma_values):
        kv["gamma_values"] = gamma_values
    
    # Objective info
    objective = metadata.get("objective")
    if objective:
        kv["objective"] = objective
    
    # Numerical aperture
    na = metadata.get("na")
    if na and na > 0:
        kv["numerical_aperture"] = na
    
    # Magnification
    mag = metadata.get("magnification")
    if mag and mag > 0:
        kv["magnification"] = mag
    
    # Immersion
    immersion = metadata.get("immersion")
    if immersion:
        kv["immersion"] = immersion
    
    # Refractive index
    ri = metadata.get("refractiveindex")
    if ri and ri > 0:
        kv["refractive_index"] = ri
    
    # Pinhole size (in micrometers) - confocal only
    if is_confocal:
        pinhole = metadata.get("pinholesize_um")
        if pinhole and pinhole > 0:
            kv["pinhole_size_um"] = pinhole
        
        # Pinhole in Airy units (more meaningful, objective-independent)
        pinhole_airy = metadata.get("pinhole_airy")
        if pinhole_airy and pinhole_airy > 0:
            kv["pinhole_airy"] = pinhole_airy
    
    # Scan settings (confocal only)
    if is_confocal:
        zoom = metadata.get("zoom")
        if zoom and zoom > 0:
            kv["zoom"] = zoom
        
        scan_speed = metadata.get("scan_speed")
        if scan_speed and scan_speed > 0:
            kv["scan_speed"] = scan_speed
        
        scan_direction = metadata.get("scan_direction")
        if scan_direction:
            kv["scan_direction"] = scan_direction
        
        # Bidirectional phase correction
        phase_x = metadata.get("phase_x")
        if phase_x is not None:
            kv["phase_x"] = phase_x
        
        # Rotator/scan angle
        rotator_angle = metadata.get("rotator_angle")
        if rotator_angle is not None:
            kv["rotator_angle"] = rotator_angle
        
        is_resonant = metadata.get("is_resonant_scanner")
        if is_resonant is not None:
            kv["is_resonant_scanner"] = is_resonant
        
        scan_mode = metadata.get("scan_mode")
        if scan_mode:
            kv["scan_mode"] = scan_mode
        
        pixel_dwell = metadata.get("pixel_dwell_time_us")
        if pixel_dwell and pixel_dwell > 0:
            kv["pixel_dwell_time_us"] = pixel_dwell
        
        line_time = metadata.get("line_time_us")
        if line_time and line_time > 0:
            kv["line_time_us"] = line_time
        
        frame_time = metadata.get("frame_time_ms")
        if frame_time and frame_time > 0:
            kv["frame_time_ms"] = frame_time
        
        # Averaging/Accumulation settings
        frame_avg = metadata.get("frame_average")
        if frame_avg and frame_avg > 1:
            kv["frame_average"] = frame_avg
        
        line_avg = metadata.get("line_average")
        if line_avg and line_avg > 1:
            kv["line_average"] = line_avg
        
        frame_acc = metadata.get("frame_accumulation")
        if frame_acc and frame_acc > 1:
            kv["frame_accumulation"] = frame_acc
        
        line_acc = metadata.get("line_accumulation")
        if line_acc and line_acc > 1:
            kv["line_accumulation"] = line_acc
        
        # Detector settings per channel (confocal only)
        detector_types = metadata.get("detector_types")
        if detector_types and isinstance(detector_types, list) and any(detector_types):
            kv["detector_types"] = detector_types
        
        detector_gains = metadata.get("detector_gains")
        if detector_gains and isinstance(detector_gains, list) and any(g is not None for g in detector_gains):
            kv["detector_gains"] = detector_gains
        
        detector_offsets = metadata.get("detector_offsets")
        if detector_offsets and isinstance(detector_offsets, list) and any(o is not None for o in detector_offsets):
            kv["detector_offsets"] = detector_offsets
        
        # Z-stack bounds (absolute positions)
        zstack_begin = metadata.get("zstack_begin_m")
        if zstack_begin is not None:
            kv["zstack_begin_m"] = zstack_begin
        
        zstack_end = metadata.get("zstack_end_m")
        if zstack_end is not None:
            kv["zstack_end_m"] = zstack_end
        
        zstack_sections = metadata.get("zstack_sections")
        if zstack_sections and zstack_sections > 1:
            kv["zstack_sections"] = zstack_sections
    
    # Stage position (in meters)
    stage_x = metadata.get("stage_pos_x_m")
    if stage_x is not None:
        kv["stage_pos_x_m"] = stage_x
    
    stage_y = metadata.get("stage_pos_y_m")
    if stage_y is not None:
        kv["stage_pos_y_m"] = stage_y
    
    # Camera settings (widefield only)
    if is_widefield:
        exposure_times = metadata.get("exposure_times_s")
        if exposure_times and isinstance(exposure_times, list) and any(e and e > 0 for e in exposure_times):
            kv["exposure_times_s"] = exposure_times
        
        camera_gains = metadata.get("camera_gains")
        if camera_gains and isinstance(camera_gains, list) and any(g is not None for g in camera_gains):
            kv["camera_gains"] = camera_gains
        
        camera_name = metadata.get("camera_name")
        if camera_name:
            kv["camera_name"] = camera_name
        
        binning = metadata.get("binning")
        if binning and binning > 0:
            kv["binning"] = binning
    
    # Climate control settings (live cell imaging)
    temperature = metadata.get("temperature_c")
    if temperature is not None:
        kv["temperature_c"] = temperature
    
    co2 = metadata.get("co2_percent")
    if co2 is not None:
        kv["co2_percent"] = co2
    
    obj_heater = metadata.get("objective_heater_c")
    if obj_heater is not None:
        kv["objective_heater_c"] = obj_heater
    
    # Coverslip thickness - relevant for all microscopy types
    coverslip = metadata.get("coverslip_thickness_um")
    if coverslip and coverslip > 0:
        kv["coverslip_thickness_um"] = coverslip
    
    # Illumination intensities - only for non-RGB fluorescence images
    if not is_rgb:
        if is_confocal:
            laser_intensities = metadata.get("laser_intensities")
            if laser_intensities and isinstance(laser_intensities, list) and len(laser_intensities) > 0:
                kv["laser_intensities"] = laser_intensities
        
        if is_widefield:
            led_intensities = metadata.get("led_intensities")
            if led_intensities and isinstance(led_intensities, list) and len(led_intensities) > 0:
                kv["led_intensities"] = led_intensities


def convert_leica(
    inputfile: str = '',
    image_uuid: str = 'n/a',
    show_progress: bool = True,
    outputfolder: str | None = None,
    altoutputfolder: str | None = None,
    xy_check_value: int = 3192,
    get_image_metadata: bool = False,
    get_image_xml: bool = False,
    tempfolder: str | None = None,
):
    """
    Converts Leica LIF, LOF, or XLEF files to OME-TIFF, .LOF, or single-image .LIF based on metadata and specific rules.

    Args:
        inputfile (str): Path to the input LIF/LOF/XLEF file.
        image_uuid (str, optional): UUID of the image. Defaults to 'n/a'.
        show_progress (bool, optional): Enable progress bar during conversion. Defaults to True.
        outputfolder (str, optional): Output directory for converted files. Defaults to None.
        altoutputfolder (str, optional): Optional alternative second output folder. Defaults to None.
        xy_check_value (int, optional): Threshold for XY dimensions to determine conversion type. Defaults to 3192.
        get_image_metadata (bool, optional): When True, include full image metadata JSON under keyvalues.image_metadata_json. Defaults to False.
        get_image_xml (bool, optional): When True, include raw image XML string under keyvalues.image_xml (empty if unavailable). Defaults to False.
        tempfolder (str, optional): Custom temp folder for intermediate files. If None, uses system temp directory. Defaults to None.

    Returns:
        str: JSON array string with conversion results. Each element is a dict with keys:
            - name: base name of the created or relevant file (without extension)
            - full_path: absolute path to the output file (OME-TIFF, .LOF, or .LIF)
            - alt_path: absolute path to the file in altoutputfolder (if used and file exists), else None
        Returns an empty JSON array string ("[]") if no conversion is applicable or an error occurs.
    """
    created_filename = None

    try:
        if show_progress:
            # Construct the processing message
            processing_msg = f"Processing: {os.path.basename(inputfile)}"
            # Append UUID if it's provided and not the default 'n/a'
            if image_uuid != 'n/a':
                processing_msg += f" (UUID: {image_uuid})"
            print(processing_msg + "...") 

        metadata = read_image_metadata(inputfile, image_uuid)
        filetype = metadata.get("filetype", "").lower()
        xs = metadata.get("xs", 0)
        ys = metadata.get("ys", 0)
        tiles = metadata.get("tiles", 0)
        isrgb = metadata.get("isrgb", False)
        overlap_is_negative = metadata.get("OverlapIsNegative", False)
        lof_path = metadata.get("LOFFilePath")
        save_child_name = metadata.get("save_child_name")
        # For XLEF/LOF images, reconstruct full hierarchical save_child_name if possible
        if inputfile.lower().endswith(".xlef") and image_uuid and image_uuid != 'n/a':
            try:
                full_name = _find_image_hierarchical_path(inputfile, image_uuid)
                if full_name:
                    save_child_name = full_name
                    # Update metadata so conversion functions can use the hierarchical name
                    metadata["save_child_name"] = full_name
            except Exception as e:
                if show_progress:
                    print(f"Warning: Could not get hierarchical save_child_name from XLEF: {e}")

        # Add the file extension to the save_child_name for output filename
        save_child_name = _format_save_child_name_with_ext(inputfile, save_child_name)

        if filetype == ".lif":
            if tiles>1 and overlap_is_negative:
                if show_progress:
                    print(f"  Detected a Tilescan with OverlapIsNegative. Calling convert_leica_to_singlelif...")
                created_filename = convert_leica_to_singlelif(
                    inputfile=inputfile,
                    image_uuid=image_uuid,
                    outputfolder=outputfolder,
                    show_progress=show_progress,
                    altoutputfolder=altoutputfolder
                )
                if created_filename:
                    # Compute per-channel stats once
                    stats = compute_channel_intensity_stats(metadata, sample_fraction=0.1, use_memmap=True)
                    kv = dict(stats)
                    _add_optional_metadata(kv, metadata)
                    if get_image_metadata:
                        kv["image_metadata_json"] = metadata
                    if get_image_xml:
                        kv["image_xml"] = metadata.get("xmlElement") or ""
                    # Format the display name
                    fallback_name = os.path.splitext(os.path.basename(created_filename))[0]
                    if fallback_name.endswith('.ome'):
                        fallback_name = fallback_name[:-4]
                    name = _format_display_name(inputfile, save_child_name, fallback_name)
                    full_path = os.path.join(outputfolder, os.path.basename(created_filename))
                    full_path = os.path.normpath(full_path)
                    alt_path = None
                    if altoutputfolder:
                        alt_candidate = os.path.join(altoutputfolder, os.path.basename(created_filename))
                        alt_candidate = os.path.normpath(alt_candidate)
                        if os.path.exists(alt_candidate):
                            alt_path = alt_candidate
                    result = [{
                        "name": name,
                        "full_path": full_path,
                        "alt_path": alt_path,
                        "keyvalues": [kv]
                    }]
                    if show_progress: print(f"  Finished convert_leica_to_singlelif.")
                    return json.dumps(result)
                else:
                    if show_progress: print(f"  convert_leica_to_singlelif failed.")
                    return json.dumps([])
            else:
                # Large LIF, not OverlapIsNegative: OME-TIFF
                if isrgb:
                    if show_progress: print(f"  Detected RGB LIF. Calling convert_leica_rgb_to_ometiff...")
                    created_filename = convert_leica_rgb_to_ometiff(
                        inputfile=inputfile,
                        image_uuid=image_uuid,
                        outputfolder=outputfolder,
                        show_progress=show_progress,
                        altoutputfolder=altoutputfolder,
                        tempfolder=tempfolder,
                        save_child_name=save_child_name
                    )
                else:
                    if show_progress: print(f"  Detected (Multi/Single) Channel LIF. Calling convert_leica_to_ometiff...")
                    created_filename = convert_leica_to_ometiff(
                        inputfile=inputfile,
                        image_uuid=image_uuid,
                        outputfolder=outputfolder,
                        show_progress=show_progress,
                        altoutputfolder=altoutputfolder,
                        tempfolder=tempfolder,
                        save_child_name=save_child_name
                    )
                if created_filename:
                    stats = compute_channel_intensity_stats(metadata, sample_fraction=0.1, use_memmap=True)
                    kv = dict(stats)
                    _add_optional_metadata(kv, metadata)
                    if get_image_metadata:
                        kv["image_metadata_json"] = metadata
                    if get_image_xml:
                        kv["image_xml"] = metadata.get("xmlElement") or ""
                    # Format the display name
                    fallback_name = os.path.splitext(os.path.basename(created_filename))[0]
                    if fallback_name.endswith('.ome'):
                        fallback_name = fallback_name[:-4]
                    name = _format_display_name(inputfile, save_child_name, fallback_name)
                    full_path = os.path.join(outputfolder, os.path.basename(created_filename))
                    full_path = os.path.normpath(full_path)
                    alt_path = None
                    if altoutputfolder:
                        alt_candidate = os.path.join(altoutputfolder, os.path.basename(created_filename))
                        alt_candidate = os.path.normpath(alt_candidate)
                        if os.path.exists(alt_candidate):
                            alt_path = alt_candidate
                    result = [{
                        "name": name,
                        "full_path": full_path,
                        "alt_path": alt_path,
                        "keyvalues": [kv]
                    }]
                    if show_progress: print(f"  Finished OME-TIFF conversion.")
                    return json.dumps(result)
                else:
                    if show_progress: print(f"  OME-TIFF conversion failed.")
                    return json.dumps([])

        elif filetype in [".xlef", ".lof"]:
            relevant_path = lof_path if lof_path else inputfile
            if ((xs <= xy_check_value and ys <= xy_check_value) or (tiles>1 and overlap_is_negative)):
                stats = compute_channel_intensity_stats(metadata, sample_fraction=0.1, use_memmap=True)
                kv = dict(stats)
                _add_optional_metadata(kv, metadata)
                if get_image_metadata:
                    kv["image_metadata_json"] = metadata
                if get_image_xml:
                    kv["image_xml"] = metadata.get("xmlElement") or ""
                # Format the display name
                filename = os.path.basename(relevant_path)
                fallback_name = os.path.splitext(filename)[0]
                if fallback_name.endswith('.ome'):
                    fallback_name = fallback_name[:-4]
                name = _format_display_name(inputfile, save_child_name, fallback_name)
                full_path = os.path.normpath(relevant_path)
                alt_path = None
                # Copy the file to altoutputfolder and set alt_path
                if altoutputfolder:
                    dest_path = os.path.join(altoutputfolder, os.path.basename(full_path))
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    shutil.copy2(full_path, dest_path)
                    alt_path = dest_path
                result = [{
                    "name": name,
                    "full_path": full_path,
                    "alt_path": alt_path,
                    "keyvalues": [kv]
                }]
                if show_progress:
                    print(f"  No conversion needed for small/OverlapIsNegative {filetype}. Returning path: {relevant_path}")
                return json.dumps(result)
            else:
                # Large XLEF/LOF, not OverlapIsNegative: OME-TIFF
                if isrgb:
                    if show_progress: print(f"  Detected RGB {filetype}. Calling convert_leica_rgb_to_ometiff...")
                    created_filename = convert_leica_rgb_to_ometiff(
                        inputfile=inputfile,
                        image_uuid=image_uuid,
                        outputfolder=outputfolder,
                        show_progress=show_progress,
                        altoutputfolder=altoutputfolder,
                        tempfolder=tempfolder,
                        save_child_name=save_child_name
                    )
                else:
                    if show_progress: print(f"  Calling convert_leica_to_ometiff...")
                    created_filename = convert_leica_to_ometiff(
                        inputfile=inputfile,
                        image_uuid=image_uuid,
                        outputfolder=outputfolder,
                        show_progress=show_progress,
                        altoutputfolder=altoutputfolder,
                        tempfolder=tempfolder,
                        save_child_name=save_child_name
                    )
                if created_filename:
                    stats = compute_channel_intensity_stats(metadata, sample_fraction=0.1, use_memmap=True)
                    kv = dict(stats)
                    _add_optional_metadata(kv, metadata)
                    if get_image_metadata:
                        kv["image_metadata_json"] = metadata
                    if get_image_xml:
                        kv["image_xml"] = metadata.get("xmlElement") or ""
                    # Format the display name
                    fallback_name = os.path.splitext(os.path.basename(created_filename))[0]
                    if fallback_name.endswith('.ome'):
                        fallback_name = fallback_name[:-4]
                    name = _format_display_name(inputfile, save_child_name, fallback_name)
                    full_path = os.path.join(outputfolder, os.path.basename(created_filename))
                    full_path = os.path.normpath(full_path)
                    alt_path = None
                    if altoutputfolder:
                        alt_candidate = os.path.join(altoutputfolder, os.path.basename(created_filename))
                        alt_candidate = os.path.normpath(alt_candidate)
                        if os.path.exists(alt_candidate):
                            alt_path = alt_candidate
                    result = [{
                        "name": name,
                        "full_path": full_path,
                        "alt_path": alt_path,
                        "keyvalues": [kv]
                    }]
                    if show_progress: print(f"  Finished OME-TIFF conversion.")
                    return json.dumps(result)
                else:
                    if show_progress: print(f"  OME-TIFF conversion failed.")
                    return json.dumps([])

        else:
            if show_progress:
                print(f"  No applicable conversion rule for {filetype}.")
            return json.dumps([])

    except Exception as e:
        # Print newline to avoid messing up progress bar if error occurs mid-conversion
        print(f"\nError during convert_leica processing for {inputfile}: {str(e)}")
        return json.dumps([])

