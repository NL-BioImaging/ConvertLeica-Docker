import os
import json
import shutil

from ci_leica_converters_single_lif import convert_leica_to_singlelif
from ci_leica_converters_ometiff import convert_leica_to_ometiff
from ci_leica_converters_ometiff_rgb import convert_leica_rgb_to_ometiff
from ci_leica_converters_helpers import read_image_metadata, _read_xlef_image, _find_image_hierarchical_path, compute_channel_intensity_stats

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
            except Exception as e:
                if show_progress:
                    print(f"Warning: Could not get hierarchical save_child_name from XLEF: {e}")

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
                    if get_image_metadata:
                        kv["image_metadata_json"] = metadata
                    if get_image_xml:
                        kv["image_xml"] = metadata.get("xmlElement") or ""
                    # Use save_child_name from metadata if available, else fallback to filename logic
                    if save_child_name:
                        name = save_child_name
                    else:
                        name = os.path.splitext(os.path.basename(created_filename))[0]
                        if name.endswith('.ome'):
                            name = name[:-4]
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
                        tempfolder=tempfolder
                    )
                else:
                    if show_progress: print(f"  Detected (Multi/Single) Channel LIF. Calling convert_leica_to_ometiff...")
                    created_filename = convert_leica_to_ometiff(
                        inputfile=inputfile,
                        image_uuid=image_uuid,
                        outputfolder=outputfolder,
                        show_progress=show_progress,
                        altoutputfolder=altoutputfolder,
                        tempfolder=tempfolder
                    )
                if created_filename:
                    stats = compute_channel_intensity_stats(metadata, sample_fraction=0.1, use_memmap=True)
                    kv = dict(stats)
                    if get_image_metadata:
                        kv["image_metadata_json"] = metadata
                    if get_image_xml:
                        kv["image_xml"] = metadata.get("xmlElement") or ""
                    if save_child_name:
                        name = save_child_name
                    else:
                        name = os.path.splitext(os.path.basename(created_filename))[0]
                        if name.endswith('.ome'):
                            name = name[:-4]
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
                if get_image_metadata:
                    kv["image_metadata_json"] = metadata
                if get_image_xml:
                    kv["image_xml"] = metadata.get("xmlElement") or ""
                if save_child_name:
                    name = save_child_name
                else:
                    filename = os.path.basename(relevant_path)
                    name = os.path.splitext(filename)[0]
                    if name.endswith('.ome'):
                        name = name[:-4]
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
                        tempfolder=tempfolder
                    )
                else:
                    if show_progress: print(f"  Calling convert_leica_to_ometiff...")
                    created_filename = convert_leica_to_ometiff(
                        inputfile=inputfile,
                        image_uuid=image_uuid,
                        outputfolder=outputfolder,
                        show_progress=show_progress,
                        altoutputfolder=altoutputfolder,
                        tempfolder=tempfolder
                    )
                if created_filename:
                    stats = compute_channel_intensity_stats(metadata, sample_fraction=0.1, use_memmap=True)
                    kv = dict(stats)
                    if get_image_metadata:
                        kv["image_metadata_json"] = metadata
                    if get_image_xml:
                        kv["image_xml"] = metadata.get("xmlElement") or ""
                    if save_child_name:
                        name = save_child_name
                    else:
                        name = os.path.splitext(os.path.basename(created_filename))[0]
                        if name.endswith('.ome'):
                            name = name[:-4]
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

