import xml.etree.ElementTree as ET

###############################################################################
# Shared metadata parser for images
###############################################################################
def parse_image_xml(xml_element):
    """
    Parses a Leica image XML element to extract image metadata such as pixel sizes, dimensions, color LUTs, channel names, and more.

    Args:
        xml_element (xml.etree.ElementTree.Element): XML element containing Leica image metadata.

    Returns:
        dict: Dictionary with extracted metadata fields (e.g., xs, ys, zs, channels, isrgb, resolutions, LUTs, etc.).
    """

    # with open(r"c:\xml.xml", "w") as f:
    #     f.write(ET.tostring(xml_element).decode())

    metadata = {}
    metadata['UniqueID'] = None  # Initialize UniqueID
    metadata['ElementName'] = None

    # Initialize metadata with default values
    metadata['xs'] = 1  # x size
    metadata['ys'] = 1  # y size
    metadata['zs'] = 1  # z size (slices)
    metadata['ts'] = 1  # time
    metadata['tiles'] = 1  # tiles
    metadata['channels'] = 1
    metadata['isrgb'] = False
    metadata['xres'] = 0.0
    metadata['yres'] = 0.0
    metadata['zres'] = 0.0
    metadata['tres'] = 0.0  # time resolution
    metadata['resunit'] = ''
    metadata['xres2'] = 0.0  # x resolution in micrometers
    metadata['yres2'] = 0.0  # y resolution in micrometers
    metadata['zres2'] = 0.0  # z resolution in micrometers
    metadata['resunit2'] = '' # resolution unit after conversion (should be micrometer)
    metadata['lutname'] = []
    metadata['channelResolution'] = []
    metadata['channelbytesinc'] = []
    metadata['xbytesinc'] = 0 # x dimension bytes increment
    metadata['ybytesinc'] = 0 # y dimension bytes increment
    metadata['zbytesinc'] = 0 # z dimension bytes increment
    metadata['tbytesinc'] = 0 # t dimension bytes increment
    metadata['tilesbytesinc'] = 0 # tiles dimension bytes increment
    metadata['blackvalue'] = []
    metadata['whitevalue'] = []
    metadata['flipx'] = 0
    metadata['flipy'] = 0
    metadata['swapxy'] = 0
    metadata['tilescan_flipx'] = 0
    metadata['tilescan_flipy'] = 0
    metadata['tilescan_swapxy'] = 0
    metadata['tile_positions'] = []
    metadata['objective'] = ''
    metadata['na'] = None
    metadata['refractiveindex'] = None
    metadata['mic_type'] = ''
    metadata['mic_type2'] = ''
    metadata['filterblock'] = []
    metadata['excitation'] = []
    metadata['emission'] = []
    metadata['contrastmethod'] = []
    metadata['immersion'] = None
    metadata['pinholesize_um'] = None
    metadata['magnification'] = None
    metadata['SystemTypeName'] = ''
    metadata['MicroscopeModel'] = ''
    metadata['OverlapPercentageX'] = 0.0
    metadata['OverlapPercentageY'] = 0.0
    metadata['stitching_settings_found'] = False
    metadata['OverlapIsNegative'] = False # True if any overlap percentage is negative
    metadata['dimensions'] = { # Consolidated dimensions
        'x': 1, 'y': 1, 'z': 1, 'c': 1, 't': 1, 's': 1, 'isrgb': False
    }
    
    # Scan settings (confocal)
    metadata['zoom'] = None
    metadata['scan_speed'] = None
    metadata['scan_direction'] = None
    metadata['is_resonant_scanner'] = None
    metadata['scan_mode'] = None
    metadata['pixel_dwell_time_us'] = None
    metadata['line_time_us'] = None
    metadata['frame_time_ms'] = None
    metadata['frame_average'] = None
    metadata['line_average'] = None
    metadata['frame_accumulation'] = None
    metadata['line_accumulation'] = None
    
    # Stage position
    metadata['stage_pos_x_m'] = None
    metadata['stage_pos_y_m'] = None
    
    # Camera settings (widefield)
    metadata['exposure_times_s'] = []
    metadata['camera_gains'] = []
    metadata['camera_name'] = None
    metadata['binning'] = None
    
    # Channel names (user-defined names like DAPI, GFP, etc.)
    metadata['channel_names'] = []
    
    # Climate control settings (live cell imaging)
    metadata['temperature_c'] = None
    metadata['co2_percent'] = None
    metadata['objective_heater_c'] = None
    
    # Coverslip thickness (optical corrections)
    metadata['coverslip_thickness_um'] = None
    
    # Illumination intensities
    metadata['laser_intensities'] = []  # List of {wavelength, intensity_percent}
    metadata['led_intensities'] = []    # List of {wavelength, intensity_percent}
    
    # Detector settings (confocal) - per channel
    metadata['detector_types'] = []      # e.g., ['PMT', 'HyD', 'HPD']
    metadata['detector_gains'] = []      # Gain values per channel
    metadata['detector_offsets'] = []    # Offset values per channel
    
    # Additional confocal settings
    metadata['pinhole_airy'] = None      # Pinhole in Airy units (objective-independent)
    metadata['phase_x'] = None           # Bidirectional phase correction
    metadata['rotator_angle'] = None     # Scan rotation angle in degrees
    metadata['system_serial_number'] = None  # Microscope serial number
    
    # Z-stack bounds (absolute positions)
    metadata['zstack_begin_m'] = None    # Z-stack start position in meters
    metadata['zstack_end_m'] = None      # Z-stack end position in meters
    metadata['zstack_sections'] = None   # Planned number of sections
    
    # Gamma values for display
    metadata['gammavalue'] = []

    # Temporary storage for overlap values from XML
    xml_overlap_x_value = None
    xml_overlap_y_value = None

    if xml_element.tag == 'Element':
        metadata['UniqueID'] = xml_element.attrib.get('UniqueID')
        metadata['ElementName'] = xml_element.attrib.get('Name', '')
    else:
        metadata['UniqueID'] = 'none (LOF)'
        metadata['ElementName'] = 'none (LOF)'

    memory_block = xml_element.find('.//Memory/Block')
    if memory_block is not None:
        block_file = memory_block.attrib.get('File')
        if block_file and block_file.lower().endswith('.lof'):
            metadata['LOFFile'] = block_file
        pass # Added pass to make block valid

    # Extract ImageDescription
    image_description = xml_element.find('.//ImageDescription')
    if image_description is not None:
        # Extract Channels
        channels_element = image_description.find('Channels')
        if channels_element is not None:
            channel_descriptions = channels_element.findall('ChannelDescription')
            metadata['channels'] = len(channel_descriptions)
            if metadata['channels'] > 1:
                channel_tag = channel_descriptions[0].attrib.get('ChannelTag')
                if channel_tag and int(channel_tag) != 0:
                    metadata['isrgb'] = True
            for channel_desc in channel_descriptions:
                bytes_inc = channel_desc.attrib.get('BytesInc')
                resolution = channel_desc.attrib.get('Resolution')
                lut_name = channel_desc.attrib.get('LUTName')
                metadata['channelbytesinc'].append(int(bytes_inc) if bytes_inc else None)
                metadata['channelResolution'].append(int(resolution) if resolution else None)
                metadata['lutname'].append(lut_name.lower() if lut_name else '')
                
                # Extract channel name from ChannelProperty with Key="DyeName"
                channel_name = None
                channel_properties = channel_desc.findall('ChannelProperty')
                for prop in channel_properties:
                    key_elem = prop.find('Key')
                    value_elem = prop.find('Value')
                    if key_elem is not None and key_elem.text and key_elem.text.strip() == 'DyeName':
                        if value_elem is not None and value_elem.text:
                            channel_name = value_elem.text.strip()
                            break
                metadata['channel_names'].append(channel_name if channel_name else '')
        else:
            # Single channel, handle separately
            channel_desc = image_description.find('.//ChannelDescription')
            if channel_desc is not None:
                bytes_inc = channel_desc.attrib.get('BytesInc')
                resolution = channel_desc.attrib.get('Resolution')
                lut_name = channel_desc.attrib.get('LUTName')
                metadata['channelbytesinc'].append(int(bytes_inc) if bytes_inc else None)
                metadata['channelResolution'].append(int(resolution) if resolution else None)
                metadata['lutname'].append(lut_name.lower() if lut_name else '')
                metadata['channels'] = 1
                
                # Extract channel name from ChannelProperty with Key="DyeName"
                channel_name = None
                channel_properties = channel_desc.findall('ChannelProperty')
                for prop in channel_properties:
                    key_elem = prop.find('Key')
                    value_elem = prop.find('Value')
                    if key_elem is not None and key_elem.text and key_elem.text.strip() == 'DyeName':
                        if value_elem is not None and value_elem.text:
                            channel_name = value_elem.text.strip()
                            break
                metadata['channel_names'].append(channel_name if channel_name else '')
        pass # Added pass

        # Extract Dimensions
        dimensions_element = image_description.find('Dimensions')
        if dimensions_element is not None:
            dim_descriptions = dimensions_element.findall('DimensionDescription')
            for dim_desc in dim_descriptions:
                dim_id = int(dim_desc.attrib.get('DimID', '0'))
                num_elements = int(dim_desc.attrib.get('NumberOfElements', '0'))
                length = float(dim_desc.attrib.get('Length', '0'))
                bytes_inc = int(dim_desc.attrib.get('BytesInc', '0'))
                unit = dim_desc.attrib.get('Unit', '')
                if unit:
                    metadata['resunit'] = unit

                # Compute resolution
                if num_elements > 1:
                    res = length / (num_elements - 1)
                else:
                    res = 0

                if dim_id == 1:
                    metadata['xs'] = num_elements
                    metadata['xres'] = res
                    metadata['xbytesinc'] = bytes_inc
                elif dim_id == 2:
                    metadata['ys'] = num_elements
                    metadata['yres'] = res
                    metadata['ybytesinc'] = bytes_inc
                elif dim_id == 3:
                    metadata['zs'] = num_elements
                    metadata['zres'] = res
                    metadata['zbytesinc'] = bytes_inc
                elif dim_id == 4:
                    metadata['ts'] = num_elements
                    metadata['tres'] = res
                    metadata['tbytesinc'] = bytes_inc
                elif dim_id == 10:
                    metadata['tiles'] = num_elements
                    metadata['tilesbytesinc'] = bytes_inc

        # Extract ViewerScaling (black and white values)
        attachments = xml_element.findall('.//Attachment')
        viewer_scaling = None
        for attachment in attachments:
            if attachment.attrib.get('Name') == 'ViewerScaling':
                viewer_scaling = attachment
                break
        if viewer_scaling is not None:
            channel_scaling_infos = viewer_scaling.findall('ChannelScalingInfo')
            if channel_scaling_infos:
                for csi in channel_scaling_infos:
                    black_value = float(csi.attrib.get('BlackValue', '0'))
                    white_value = float(csi.attrib.get('WhiteValue', '1'))
                    gamma_value = float(csi.attrib.get('GammaValue', '1'))
                    metadata['blackvalue'].append(black_value)
                    metadata['whitevalue'].append(white_value)
                    metadata['gammavalue'].append(gamma_value)
            else:
                csi = viewer_scaling.find('ChannelScalingInfo')
                if csi is not None:
                    black_value = float(csi.attrib.get('BlackValue', '0'))
                    white_value = float(csi.attrib.get('WhiteValue', '1'))
                    gamma_value = float(csi.attrib.get('GammaValue', '1'))
                    metadata['blackvalue'].append(black_value)
                    metadata['whitevalue'].append(white_value)
                    metadata['gammavalue'].append(gamma_value)
        else:
            # Default black/white/gamma
            for _ in range(metadata['channels']):
                metadata['blackvalue'].append(0)
                metadata['whitevalue'].append(1)
                metadata['gammavalue'].append(1)


        # Extract HardwareSetting
        hardware_setting = None
        for attachment in attachments:
            if attachment.attrib.get('Name') == 'HardwareSetting':
                hardware_setting = attachment
                break
        if hardware_setting is not None:
            data_source_type_name = hardware_setting.attrib.get('DataSourceTypeName', '')
            metadata['mic_type2'] = data_source_type_name.lower()
            if data_source_type_name == 'Confocal':
                metadata['mic_type'] = 'IncohConfMicr'
                # Confocal settings
                confocal_setting = hardware_setting.find('ATLConfocalSettingDefinition')
                if confocal_setting is not None:
                    attributes = confocal_setting.attrib
                    metadata['SystemTypeName'] = attributes.get('SystemTypeName', '')
                    metadata['MicroscopeModel'] = attributes.get('MicroscopeModel', '')                    
                    metadata['objective'] = attributes.get('ObjectiveName', '')
                    # ... existing attribute parsing (NA, RefractiveIndex, Immersion, Magnification, Pinhole) ...
                    metadata['na'] = float(attributes.get('NumericalAperture', '0'))
                    metadata['refractiveindex'] = float(attributes.get('RefractionIndex', '0'))
                    # Extract Immersion
                    metadata['immersion'] = attributes.get('Immersion')
                    # Extract Magnification
                    mag_str = attributes.get('Magnification')
                    if mag_str:
                        try:
                            metadata['magnification'] = float(mag_str)
                        except ValueError:
                            print(f"Warning: Could not parse Magnification value '{mag_str}'")
                            metadata['magnification'] = None
                    # Extract Pinhole size (convert meters to micrometers)
                    pinhole_m_str = attributes.get('Pinhole')
                    if pinhole_m_str:
                        try:
                            pinhole_m = float(pinhole_m_str)
                            metadata['pinholesize_um'] = pinhole_m * 1_000_000
                        except ValueError:
                            print(f"Warning: Could not parse Pinhole value '{pinhole_m_str}'")
                            metadata['pinholesize_um'] = None  # Ensure it's None if parsing fails
                    
                    # Pinhole in Airy units (objective-independent, more meaningful)
                    pinhole_airy_str = attributes.get('PinholeAiry')
                    if pinhole_airy_str:
                        try:
                            metadata['pinhole_airy'] = float(pinhole_airy_str)
                        except ValueError:
                            pass
                    
                    # Z-stack bounds (absolute positions in meters)
                    zstack_begin_str = attributes.get('Begin')
                    if zstack_begin_str:
                        try:
                            metadata['zstack_begin_m'] = float(zstack_begin_str)
                        except ValueError:
                            pass
                    
                    zstack_end_str = attributes.get('End')
                    if zstack_end_str:
                        try:
                            metadata['zstack_end_m'] = float(zstack_end_str)
                        except ValueError:
                            pass
                    
                    zstack_sections_str = attributes.get('Sections')
                    if zstack_sections_str:
                        try:
                            metadata['zstack_sections'] = int(zstack_sections_str)
                        except ValueError:
                            pass

                    # Extract scan settings
                    zoom_str = attributes.get('Zoom')
                    if zoom_str:
                        try:
                            metadata['zoom'] = float(zoom_str)
                        except ValueError:
                            pass
                    
                    scan_speed_str = attributes.get('ScanSpeed')
                    if scan_speed_str:
                        try:
                            metadata['scan_speed'] = int(float(scan_speed_str))
                        except ValueError:
                            pass
                    
                    metadata['scan_direction'] = attributes.get('ScanDirectionXName')
                    
                    # Bidirectional phase correction (only relevant for bidirectional scanning)
                    phase_x_str = attributes.get('PhaseX')
                    if phase_x_str:
                        try:
                            metadata['phase_x'] = float(phase_x_str)
                        except ValueError:
                            pass
                    
                    # Rotator/scan angle
                    rotator_str = attributes.get('RotatorAngle')
                    if rotator_str:
                        try:
                            metadata['rotator_angle'] = float(rotator_str)
                        except ValueError:
                            pass
                    
                    # System serial number
                    metadata['system_serial_number'] = attributes.get('SystemSerialNumber')
                    
                    is_resonant_str = attributes.get('IsResonantConfocalScanner')
                    if is_resonant_str:
                        metadata['is_resonant_scanner'] = is_resonant_str == '1'
                    
                    metadata['scan_mode'] = attributes.get('ScanMode')
                    
                    # Pixel dwell time (convert seconds to microseconds)
                    pixel_dwell_str = attributes.get('PixelDwellTime')
                    if pixel_dwell_str:
                        try:
                            metadata['pixel_dwell_time_us'] = float(pixel_dwell_str) * 1_000_000
                        except ValueError:
                            pass
                    
                    # Line time (convert seconds to microseconds)
                    line_time_str = attributes.get('LineTime')
                    if line_time_str:
                        try:
                            metadata['line_time_us'] = float(line_time_str) * 1_000_000
                        except ValueError:
                            pass
                    
                    # Frame time (convert seconds to milliseconds)
                    frame_time_str = attributes.get('FrameTime')
                    if frame_time_str:
                        try:
                            metadata['frame_time_ms'] = float(frame_time_str) * 1000
                        except ValueError:
                            pass
                    
                    # Averaging/Accumulation settings
                    frame_avg_str = attributes.get('FrameAverage')
                    if frame_avg_str:
                        try:
                            metadata['frame_average'] = int(frame_avg_str)
                        except ValueError:
                            pass
                    
                    line_avg_str = attributes.get('LineAverage')
                    if line_avg_str:
                        try:
                            metadata['line_average'] = int(line_avg_str)
                        except ValueError:
                            pass
                    
                    frame_acc_str = attributes.get('FrameAccumulation')
                    if frame_acc_str:
                        try:
                            metadata['frame_accumulation'] = int(frame_acc_str)
                        except ValueError:
                            pass
                    
                    line_acc_str = attributes.get('Line_Accumulation')
                    if line_acc_str:
                        try:
                            metadata['line_accumulation'] = int(line_acc_str)
                        except ValueError:
                            pass
                    
                    # Stage position (in meters)
                    stage_x_str = attributes.get('StagePosX')
                    if stage_x_str:
                        try:
                            metadata['stage_pos_x_m'] = float(stage_x_str)
                        except ValueError:
                            pass
                    
                    stage_y_str = attributes.get('StagePosY')
                    if stage_y_str:
                        try:
                            metadata['stage_pos_y_m'] = float(stage_y_str)
                        except ValueError:
                            pass
                    
                    # Climate control settings (confocal)
                    temp_str = attributes.get('IncubatorTemperature')
                    if temp_str:
                        try:
                            metadata['temperature_c'] = float(temp_str)
                        except ValueError:
                            pass
                    
                    co2_str = attributes.get('IncubatorCO2Percentage')
                    if co2_str:
                        try:
                            metadata['co2_percent'] = float(co2_str)
                        except ValueError:
                            pass
                    
                    obj_heater_str = attributes.get('ObjectiveHeaterTemperature')
                    if obj_heater_str:
                        try:
                            metadata['objective_heater_c'] = float(obj_heater_str)
                        except ValueError:
                            pass
                    
                    # Coverslip thickness (convert meters to micrometers)
                    coverslip_str = attributes.get('CoverGlassThickness')
                    if coverslip_str:
                        try:
                            metadata['coverslip_thickness_um'] = float(coverslip_str) * 1_000_000
                        except ValueError:
                            pass

                    # Extract FlipX, FlipY, SwapXY
                    metadata['flipx'] = int(attributes.get('FlipX', '0'))
                    metadata['flipy'] = int(attributes.get('FlipY', '0'))
                    metadata['swapxy'] = int(attributes.get('SwapXY', '0'))

                    # --- New Logic for Excitation/Emission ---
                    # 1. Extract available laser lines from the LaserArray
                    active_lasers = []
                    laser_array = confocal_setting.find('LaserArray')
                    if laser_array is not None:
                        for laser in laser_array.findall('Laser'):
                            wavelength_str = laser.attrib.get('Wavelength')
                            if wavelength_str:
                                try:
                                    wavelength = float(wavelength_str)
                                    if wavelength not in active_lasers: # Avoid duplicates
                                        active_lasers.append(wavelength)
                                except ValueError:
                                    print(f"Warning: Could not parse laser wavelength '{wavelength_str}' from LaserArray")
                    
                    active_lasers.sort() # Sort for easier searching
                    
                    # Extract laser intensities
                    if laser_array is not None:
                        for laser in laser_array.findall('Laser'):
                            wavelength_str = laser.attrib.get('Wavelength')
                            intensity_str = laser.attrib.get('IntensityDev') or laser.attrib.get('Intensity')
                            if wavelength_str and intensity_str:
                                try:
                                    wl = float(wavelength_str)
                                    intensity = float(intensity_str)
                                    # Only add if intensity > 0 (laser is active)
                                    if intensity > 0:
                                        metadata['laser_intensities'].append({
                                            'wavelength_nm': int(wl),
                                            'intensity_percent': round(intensity * 100, 2) if intensity <= 1 else round(intensity, 2)
                                        })
                                except ValueError:
                                    pass
                    
                    # Extract detector settings per channel from DetectorList
                    # For sequential scans, we need to look at LDM_Block_Sequential_List
                    # Each sequence has its own active detector(s)
                    
                    # First, check for sequential scan list
                    sequential_list = hardware_setting.find('.//LDM_Block_Sequential_List')
                    if sequential_list is not None:
                        # Sequential scan mode - extract from each sequence
                        seq_settings = sequential_list.findall('ATLConfocalSettingDefinition')
                        for seq_setting in seq_settings:
                            seq_detector_list = seq_setting.find('DetectorList')
                            if seq_detector_list is not None:
                                for detector in seq_detector_list.findall('Detector'):
                                    is_active = detector.attrib.get('IsActive', '0')
                                    if is_active == '1':
                                        det_type = detector.attrib.get('Type', '')
                                        metadata['detector_types'].append(det_type)
                                        
                                        # Get gain/offset from master settings since sequential entries don't have them
                                        det_name = detector.attrib.get('Name', '')
                                        det_channel = detector.attrib.get('Channel', '')
                                        
                                        # Look up full detector info from master DetectorList
                                        master_detector_list = confocal_setting.find('DetectorList')
                                        gain_val = None
                                        offset_val = None
                                        if master_detector_list is not None:
                                            for master_det in master_detector_list.findall('Detector'):
                                                if master_det.attrib.get('Name') == det_name or master_det.attrib.get('Channel') == det_channel:
                                                    gain_str = master_det.attrib.get('Gain')
                                                    if gain_str:
                                                        try:
                                                            gain_val = float(gain_str)
                                                        except ValueError:
                                                            pass
                                                    offset_str = master_det.attrib.get('Offset')
                                                    if offset_str:
                                                        try:
                                                            offset_val = float(offset_str)
                                                        except ValueError:
                                                            pass
                                                    break
                                        
                                        metadata['detector_gains'].append(gain_val)
                                        metadata['detector_offsets'].append(offset_val)
                    else:
                        # Non-sequential scan - use the main DetectorList
                        detector_list = confocal_setting.find('DetectorList')
                        if detector_list is not None:
                            for detector in detector_list.findall('Detector'):
                                # Only include active detectors
                                is_active = detector.attrib.get('IsActive', '0')
                                if is_active == '1':
                                    det_type = detector.attrib.get('Type', '')
                                    metadata['detector_types'].append(det_type)
                                    
                                    gain_str = detector.attrib.get('Gain')
                                    if gain_str:
                                        try:
                                            metadata['detector_gains'].append(float(gain_str))
                                        except ValueError:
                                            metadata['detector_gains'].append(None)
                                    else:
                                        metadata['detector_gains'].append(None)
                                    
                                    offset_str = detector.attrib.get('Offset')
                                    if offset_str:
                                        try:
                                            metadata['detector_offsets'].append(float(offset_str))
                                        except ValueError:
                                            metadata['detector_offsets'].append(None)
                                    else:
                                        metadata['detector_offsets'].append(None)

                    # Clear existing default emission/excitation lists
                    metadata['emission'] = []
                    metadata['excitation'] = []

                    # 2. Process Spectro/MultiBand and match excitation
                    # For sequential scans, only extract for channels that have active detectors
                    active_channels = set()
                    if sequential_list is not None:
                        # Build set of active channels from sequential list
                        seq_settings = sequential_list.findall('ATLConfocalSettingDefinition')
                        for seq_setting in seq_settings:
                            seq_detector_list = seq_setting.find('DetectorList')
                            if seq_detector_list is not None:
                                for detector in seq_detector_list.findall('Detector'):
                                    if detector.attrib.get('IsActive', '0') == '1':
                                        channel_num = detector.attrib.get('Channel')
                                        if channel_num:
                                            try:
                                                active_channels.add(int(channel_num))
                                            except ValueError:
                                                pass
                    
                    spectro = confocal_setting.find('Spectro')
                    if spectro is not None:
                        multiband = spectro.findall('MultiBand')
                        for mb in multiband:
                            # For sequential scans, only process MultiBand entries for active channels
                            mb_channel_str = mb.attrib.get('Channel')
                            if active_channels and mb_channel_str:
                                try:
                                    mb_channel = int(mb_channel_str)
                                    if mb_channel not in active_channels:
                                        continue  # Skip channels without active detectors
                                except ValueError:
                                    pass  # If we can't parse channel, include it anyway
                            
                            left_world_str = mb.attrib.get('LeftWorld', '0')
                            right_world_str = mb.attrib.get('RightWorld', '0')
                            dye_name = mb.attrib.get('DyeName', '') # Also grab DyeName for filterblock
                            if dye_name:
                                metadata['filterblock'].append(dye_name)

                            try:
                                left_world = float(left_world_str)
                                right_world = float(right_world_str)
                                emission = left_world + (right_world - left_world) / 2
                                metadata['emission'].append(int(round(emission))) # Round to nearest integer

                                # Find the closest lower active laser wavelength
                                best_excitation = 0.0 # Default if no suitable laser found
                                possible_excitations = [laser for laser in active_lasers if laser <= emission]
                                if possible_excitations:
                                    best_excitation = max(possible_excitations) # Use the highest laser wavelength that is <= emission
                                
                                metadata['excitation'].append(best_excitation)

                            except ValueError:
                                print(f"Warning: Could not parse MultiBand LeftWorld/RightWorld: '{left_world_str}', '{right_world_str}'")
                                # Append defaults or skip? Let's append defaults to maintain list length consistency
                                metadata['emission'].append(500) # Default emission
                                metadata['excitation'].append(480) # Default excitation
                                if dye_name and dye_name not in metadata['filterblock']: # Avoid duplicate default filterblock entry
                                     metadata['filterblock'].append(dye_name)

                    # --- Added XYStageConfiguratorSettings Parsing ---
                    xy_stage_config = confocal_setting.find('.//XYStageConfiguratorSettings')
                    if xy_stage_config is not None:
                        stitching_settings = xy_stage_config.find('StitchingSettings')
                        if stitching_settings is not None:
                            metadata['stitching_settings_found'] = True # Set flag
                            overlap_percentage_x_str = stitching_settings.attrib.get('OverlapPercentageX')
                            overlap_percentage_y_str = stitching_settings.attrib.get('OverlapPercentageY')
                            
                            if overlap_percentage_x_str is not None:
                                try:
                                    xml_overlap_x_value = float(overlap_percentage_x_str)
                                except ValueError:
                                    print(f"Warning: Could not parse OverlapPercentageX value '{overlap_percentage_x_str}' from Confocal XML.")
                                    xml_overlap_x_value = None # Ensure None on error
                            else:
                                xml_overlap_x_value = None # Attribute not found

                            if overlap_percentage_y_str is not None:
                                try:
                                    xml_overlap_y_value = float(overlap_percentage_y_str)
                                except ValueError:
                                    print(f"Warning: Could not parse OverlapPercentageY value '{overlap_percentage_y_str}' from Confocal XML.")
                                    xml_overlap_y_value = None # Ensure None on error
                            else:
                                xml_overlap_y_value = None # Attribute not found
                    # --- End Added XYStageConfiguratorSettings Parsing ---

            elif data_source_type_name == 'Camera':
                metadata['mic_type'] = 'IncohWFMicr'
                # Camera settings
                camera_setting = hardware_setting.find('ATLCameraSettingDefinition') # Initial find

                # If SEE_SEQUENTIAL_BLOCK is present in the initial ATLCameraSettingDefinition,
                # the actual ATLCameraSettingDefinition to parse is located within LDM_Block_Sequential_Master.
                if camera_setting is not None and  camera_setting.find('SEE_SEQUENTIAL_BLOCK') is not None:
                    
                    # This is the special sequential case.
                    # The 'camera_setting' found above is potentially just a pointer/shell.
                    # We must find the real one inside LDM_Block_Sequential_Master.
                    
                    sequential_master_block = hardware_setting.find('.//LDM_Block_Sequential_Master') # Changed to find descendant
                    if sequential_master_block is not None:
                        actual_camera_setting_definition = sequential_master_block.find('ATLCameraSettingDefinition')
                        if actual_camera_setting_definition is not None:
                            camera_setting = actual_camera_setting_definition # Update to the real one
                        else:
                            # Real one not found within LDM_Block_Sequential_Master
                            print("Warning: LDM_Block_Sequential_Master found, but it does not contain the nested ATLCameraSettingDefinition for sequential mode. Camera settings may be incomplete.")
                            camera_setting = None # Prevent parsing the shell
                    else:
                        # LDM_Block_Sequential_Master itself not found
                        print("Warning: Expected LDM_Block_Sequential_Master not found for sequential camera settings, despite SEE_SEQUENTIAL_BLOCK. Camera settings may be incomplete.")
                        camera_setting = None # Prevent parsing the shell
                
                # Proceed with parsing if camera_setting is valid (either original or the one from LDM_Block_Sequential_Master)
                if camera_setting is not None:
                    attributes = camera_setting.attrib
                    metadata['SystemTypeName'] = attributes.get('SystemTypeName', '')
                    metadata['MicroscopeModel'] = attributes.get('MicroscopeModel', '')
                    metadata['objective'] = attributes.get('ObjectiveName', '')
                    metadata['na'] = float(attributes.get('NumericalAperture', '0'))
                    metadata['refractiveindex'] = float(attributes.get('RefractionIndex', '0'))
                    # Extract Immersion
                    metadata['immersion'] = attributes.get('Immersion')
                    # Extract Magnification
                    mag_str = attributes.get('Magnification')
                    if mag_str:
                        try:
                            metadata['magnification'] = float(mag_str)
                        except ValueError:
                            print(f"Warning: Could not parse Magnification value '{mag_str}'")
                            metadata['magnification'] = None
                    
                    # Extract FlipX, FlipY, SwapXY
                    metadata['flipx'] = int(attributes.get('FlipX', '0'))
                    metadata['flipy'] = int(attributes.get('FlipY', '0'))
                    metadata['swapxy'] = int(attributes.get('SwapXY', '0'))

                    xy_stage_config = camera_setting.find('.//XYStageConfiguratorSettings')
                    if xy_stage_config is not None:
                        # Find StitchingSettings
                        stitching_settings = xy_stage_config.find('StitchingSettings')
                        if stitching_settings is not None:
                            metadata['stitching_settings_found'] = True # Set flag
                            # Extract OverlapPercentageX and OverlapPercentageY
                            overlap_percentage_x_str = stitching_settings.attrib.get('OverlapPercentageX')
                            overlap_percentage_y_str = stitching_settings.attrib.get('OverlapPercentageY')

                            if overlap_percentage_x_str is not None:
                                try:
                                    xml_overlap_x_value = float(overlap_percentage_x_str)
                                except ValueError:
                                    print(f"Warning: Could not parse OverlapPercentageX value '{overlap_percentage_x_str}' from Camera XML.")
                                    xml_overlap_x_value = None # Ensure None on error
                            else:
                                xml_overlap_x_value = None # Attribute not found
                            
                            if overlap_percentage_y_str is not None:
                                try:
                                    xml_overlap_y_value = float(overlap_percentage_y_str)
                                except ValueError:
                                    print(f"Warning: Could not parse OverlapPercentageY value '{overlap_percentage_y_str}' from Camera XML.")
                                    xml_overlap_y_value = None # Ensure None on error
                            else:
                                xml_overlap_y_value = None # Attribute not found
                    wf_channel_infos = camera_setting.findall('WideFieldChannelInfo')
                    for wfci in wf_channel_infos:
                        fluo_cube_name = wfci.attrib.get('FluoCubeName', '')
                        contrast_method_name = wfci.attrib.get('ContrastingMethodName', '')
                        metadata['contrastmethod'].append(contrast_method_name)
                        
                        # Extract user-defined channel name
                        user_def_name = wfci.attrib.get('UserDefName', '')
                        if user_def_name:
                            metadata['channel_names'].append(user_def_name)
                        ex_name = fluo_cube_name
                        if fluo_cube_name == 'QUAD-S':
                            ex_name = wfci.attrib.get('FFW_Excitation1FilterName', '')
                        elif fluo_cube_name == 'DA/FI/TX':
                            ex_name = wfci.attrib.get('LUT', '')
                        if fluo_cube_name!=ex_name or fluo_cube_name=='':
                            metadata['filterblock'].append(f"{fluo_cube_name}: {ex_name}")

                        ex_em_wavelengths = {
                            'DAPI': (355, 460),
                            'DAP': (355, 460),
                            'A': (355, 460),
                            'Blue': (355, 460),
                            'L5': (480, 527),
                            'I5': (480, 527),
                            'Green': (480, 527),
                            'FITC': (480, 527),
                            'N3': (545, 605),
                            'N2.1': (545, 605),
                            'TRITC': (545, 605),
                            '488': (488, 525),
                            '532': (532, 550),
                            '642': (642, 670),
                            'Red': (545, 605),
                            'Y3': (545, 605),
                            'I3': (545, 605),
                            'Y5': (590, 700),
                        }
                        ex_em = ex_em_wavelengths.get(ex_name, (0, 0))
                        metadata['excitation'].append(ex_em[0])
                        metadata['emission'].append(ex_em[1])
                    
                    # Extract camera-specific settings
                    # Camera name from WideFieldChannelConfigurator
                    wf_configurator = camera_setting.find('.//WideFieldChannelConfigurator')
                    if wf_configurator is not None:
                        metadata['camera_name'] = wf_configurator.attrib.get('CameraName')
                    
                    # Binning from CameraFormat
                    camera_format = camera_setting.find('.//CameraFormat')
                    if camera_format is not None:
                        binning_str = camera_format.attrib.get('Binning')
                        if binning_str:
                            try:
                                metadata['binning'] = int(binning_str)
                            except ValueError:
                                pass
                    
                    # Exposure times and gains from IndividualCameraInfo
                    individual_cameras = camera_setting.findall('.//IndividualCameraInfo')
                    for cam_info in individual_cameras:
                        exp_time_str = cam_info.attrib.get('ExposureTime')
                        if exp_time_str:
                            try:
                                metadata['exposure_times_s'].append(float(exp_time_str))
                            except ValueError:
                                pass
                        gain_str = cam_info.attrib.get('Gain')
                        if gain_str:
                            try:
                                metadata['camera_gains'].append(float(gain_str))
                            except ValueError:
                                pass
                    
                    # Stage position (in meters) for camera mode
                    stage_x_str = attributes.get('StagePosX')
                    if stage_x_str:
                        try:
                            metadata['stage_pos_x_m'] = float(stage_x_str)
                        except ValueError:
                            pass
                    
                    stage_y_str = attributes.get('StagePosY')
                    if stage_y_str:
                        try:
                            metadata['stage_pos_y_m'] = float(stage_y_str)
                        except ValueError:
                            pass
                    
                    # Climate control settings (camera/widefield)
                    temp_str = attributes.get('IncubatorTemperature')
                    if temp_str:
                        try:
                            metadata['temperature_c'] = float(temp_str)
                        except ValueError:
                            pass
                    
                    co2_str = attributes.get('IncubatorCO2Percentage')
                    if co2_str:
                        try:
                            metadata['co2_percent'] = float(co2_str)
                        except ValueError:
                            pass
                    
                    obj_heater_str = attributes.get('ObjectiveHeaterTemperature')
                    if obj_heater_str:
                        try:
                            metadata['objective_heater_c'] = float(obj_heater_str)
                        except ValueError:
                            pass
                    
                    # Coverslip thickness (convert meters to micrometers)
                    coverslip_str = attributes.get('CoverGlassThickness')
                    if coverslip_str:
                        try:
                            metadata['coverslip_thickness_um'] = float(coverslip_str) * 1_000_000
                        except ValueError:
                            pass
            else:
                metadata['mic_type'] = 'unknown'
                metadata['mic_type2'] = 'generic'
        else:
            metadata['mic_type'] = 'unknown'
            metadata['mic_type2'] = 'generic'


        # Extract TileScanInfo
        tile_scan_info = None
        for attachment in attachments:
            if attachment.attrib.get('Name') == 'TileScanInfo':
                tile_scan_info = attachment
                break
        if tile_scan_info is not None:
            metadata['tilescan_flipx'] = int(tile_scan_info.attrib.get('FlipX', '0'))
            metadata['tilescan_flipy'] = int(tile_scan_info.attrib.get('FlipY', '0'))
            metadata['tilescan_swapxy'] = int(tile_scan_info.attrib.get('SwapXY', '0'))
            tiles = tile_scan_info.findall('Tile')
            for i, tile in enumerate(tiles):
                tile_info = {
                    'num': i + 1,
                    'FieldX': int(tile.attrib.get('FieldX', '0')),
                    'FieldY': int(tile.attrib.get('FieldY', '0')),
                    'PosX': float(tile.attrib.get('PosX', '0')),
                    'PosY': float(tile.attrib.get('PosY', '0')),
                }
                metadata['tile_positions'].append(tile_info)            

        # Handle STELLARIS or AF 6000LX (Thunder) - Check if filterblock needs adjustment based on confocal logic
        if hardware_setting is not None:
            system_type_name = hardware_setting.attrib.get('SystemTypeName', '')

            # --- Existing STELLARIS logic ---
            # Note: Confocal logic above might already populate filterblock from DyeName.
            # Decide if this STELLARIS-specific logic is still needed or redundant.
            # Keeping it for now, might need review based on STELLARIS XML examples.
            if 'STELLARIS' in system_type_name and not metadata['filterblock']: # Only run if confocal didn't populate it
                channels_element = image_description.find('Channels')
                if channels_element is not None:
                    channel_descriptions = channels_element.findall('ChannelDescription')
                    for ch_desc in channel_descriptions:
                        channel_properties = ch_desc.findall('ChannelProperty')
                        for prop in channel_properties:
                            key = prop.find('Key')
                            value = prop.find('Value')
                            if key is not None and key.text.strip() == 'DyeName' and value is not None:
                                metadata['filterblock'].append(value.text.strip())
                                break
            elif 'AF 6000LX' in system_type_name:
                 # Clear potentially incorrect defaults from Camera logic if it ran before Thunder logic
                 if metadata['mic_type'] == 'IncohWFMicr': # Check if Camera logic ran
                     metadata['excitation'] = []
                     metadata['emission'] = []
                     metadata['filterblock'] = []
                     metadata['contrastmethod'] = []

                 # Grab ALL WideFieldChannelConfigurator blocks
                 wf_channel_config_list = hardware_setting.findall('.//WideFieldChannelConfigurator')
                 for wf_channel_config in wf_channel_config_list:
                     # Skip if it's the HS autofocus instance
                     if wf_channel_config.attrib.get('ThisIsHSAutofocusInstance', '0') == '1':
                         continue

                     # Now parse the actual WideFieldChannelInfo blocks
                     wf_channel_infos = wf_channel_config.findall('WideFieldChannelInfo')
                     for wfci in wf_channel_infos:
                         fluo_cube_name = wfci.attrib.get('FluoCubeName', '')
                         emission_str = wfci.attrib.get('EmissionWavelength', '0')
                         try:
                             emission_val = float(emission_str)
                         except ValueError:
                             emission_val = 0.0

                         # Find the highest ILLEDWavelength_i where ILLEDActiveState_i="1"
                         valid_excitation_wavelength = 0.0
                         for i in range(8):
                             active_state = wfci.attrib.get(f'ILLEDActiveState{i}', '0')
                             if active_state == '1':
                                 w_str = wfci.attrib.get(f'ILLEDWavelength{i}', '0')
                                 try:
                                     w_val = float(w_str)
                                 except ValueError:
                                     w_val = 0.0
                                 valid_excitation_wavelength = w_val

                         # Append to metadata fields
                         metadata['excitation'].append(valid_excitation_wavelength)
                         metadata['emission'].append(emission_val)

                         # Build filterblock as "FluoCubeName + emission"
                         block_label = f"{fluo_cube_name} {int(emission_val)}"
                         metadata['filterblock'].append(block_label)

                         # Also store contrast method if wanted
                         contrast_method_name = wfci.attrib.get('ContrastingMethodName', '')
                         metadata['contrastmethod'].append(contrast_method_name)
                         
                         # Extract LED intensities
                         for i in range(8):
                             active_state = wfci.attrib.get(f'ILLEDActiveState{i}', '0')
                             if active_state == '1':
                                 wl_str = wfci.attrib.get(f'ILLEDWavelength{i}', '0')
                                 intensity_str = wfci.attrib.get(f'ILLEDIntensity{i}', '0')
                                 try:
                                     wl = float(wl_str)
                                     intensity = float(intensity_str)
                                     if wl > 0:
                                         metadata['led_intensities'].append({
                                             'wavelength_nm': int(wl),
                                             'intensity_percent': round(intensity * 100, 2) if intensity <= 1 else round(intensity, 2)
                                         })
                                 except ValueError:
                                     pass


    # Convert resolution units to micrometers
    unit = metadata['resunit'].lower()
    factor = 1.0 # Default factor
    if unit in ['meter', 'm']:
        factor = 1e6
    elif unit == 'centimeter':
        factor = 1e4
    elif unit == 'inch':
        factor = 25400
    elif unit == 'millimeter':
        factor = 1e3
    elif unit == 'micrometer':
        factor = 1
    else:
        factor = 1  # Default to micrometers
    metadata['xres2'] = metadata['xres'] * factor
    metadata['yres2'] = metadata['yres'] * factor
    metadata['zres2'] = metadata['zres'] * factor
    metadata['resunit2'] = 'micrometer'

    # Define the conversion factor for tile positions (PosX, PosY) from meters to micrometers.
    tile_pos_mm_to_um_factor = 1000.0*1000.0
    # Define a tolerance for comparing tile positions (e.g., for Y when checking X-adjacency).
    # This value is in the same units as PosX/PosY from the XML (assumed mm).
    position_comparison_tolerance = 0.0000001

    # Flags to track if overlap was successfully calculated from tiles
    calculated_overlap_x_from_tiles = False
    calculated_overlap_y_from_tiles = False

    # Calculate OverlapPercentageX from tile positions if not found in StitchingSettings
    if metadata.get('tiles', 1) > 1 and \
       metadata['xs'] > 0 and metadata.get('xres2', 0) > 0:
        
        tile_width_um = metadata['xs'] * metadata['xres2'] # tile_width_um is already in micrometers
        if tile_width_um > 0: # Proceed only if tile width is positive
            min_delta_pos_x_orig_units = float('inf')
            found_x_delta = False
            
            tile_positions = metadata['tile_positions']
            # This loop structure assumes len(tile_positions) > 1 from the outer if condition.
            for i in range(len(tile_positions)):
                for j in range(i + 1, len(tile_positions)): # Compare each pair once
                    tile_a = tile_positions[i]
                    tile_b = tile_positions[j]
                    
                    # Check if tiles are in the same row (approximately)
                    if abs(tile_a['PosY'] - tile_b['PosY']) < position_comparison_tolerance:
                        delta_x = abs(tile_a['PosX'] - tile_b['PosX']) # This is in original units (assumed mm)
                        # Ensure delta_x is significant (greater than tolerance) and the smallest found so far
                        if delta_x > position_comparison_tolerance and delta_x < min_delta_pos_x_orig_units:
                            min_delta_pos_x_orig_units = delta_x
                            found_x_delta = True
            
            if found_x_delta:
                # Convert delta_pos_x from original units (assumed mm) to micrometers
                delta_pos_x_um = min_delta_pos_x_orig_units * tile_pos_mm_to_um_factor
                overlap_x_um = tile_width_um - delta_pos_x_um
                if tile_width_um > 0: # Ensure no division by zero
                    metadata['OverlapPercentageX'] = (overlap_x_um / tile_width_um)
                    calculated_overlap_x_from_tiles = True

    # Calculate OverlapPercentageY from tile positions if not found in StitchingSettings
    if metadata.get('tiles', 1) > 1 and \
       metadata['ys'] > 0 and metadata.get('yres2', 0) > 0:

        tile_height_um = metadata['ys'] * metadata['yres2'] # tile_height_um is already in micrometers
        if tile_height_um > 0: # Proceed only if tile height is positive
            min_delta_pos_y_orig_units = float('inf')
            found_y_delta = False

            tile_positions = metadata['tile_positions']
            # This loop structure assumes len(tile_positions) > 1 from the outer if condition.
            for i in range(len(tile_positions)):
                for j in range(i + 1, len(tile_positions)): # Compare each pair once
                    tile_a = tile_positions[i]
                    tile_b = tile_positions[j]

                    # Check if tiles are in the same column (approximately)
                    if abs(tile_a['PosX'] - tile_b['PosX']) < position_comparison_tolerance:
                        delta_y = abs(tile_a['PosY'] - tile_b['PosY']) # This is in original units (assumed mm)
                        # Ensure delta_y is significant (greater than tolerance) and the smallest found so far
                        if delta_y > position_comparison_tolerance and delta_y < min_delta_pos_y_orig_units:
                            min_delta_pos_y_orig_units = delta_y
                            found_y_delta = True
            
            if found_y_delta:
                # Convert delta_pos_y from original units (assumed mm) to micrometers
                delta_pos_y_um = min_delta_pos_y_orig_units * tile_pos_mm_to_um_factor
                overlap_y_um = tile_height_um - delta_pos_y_um
                if tile_height_um > 0: # Ensure no division by zero
                    metadata['OverlapPercentageY'] = (overlap_y_um / tile_height_um)
                    calculated_overlap_y_from_tiles = True

    # Fallback to XML values if tile calculation was not successful
    if not calculated_overlap_x_from_tiles and metadata['stitching_settings_found'] and xml_overlap_x_value is not None:
        metadata['OverlapPercentageX'] = xml_overlap_x_value
    
    if not calculated_overlap_y_from_tiles and metadata['stitching_settings_found'] and xml_overlap_y_value is not None:
        metadata['OverlapPercentageY'] = xml_overlap_y_value

    # Determine if overlap is negative after final values are set
    overlap_x_final = metadata.get("OverlapPercentageX", 0.0)
    overlap_y_final = metadata.get("OverlapPercentageY", 0.0)
    metadata["OverlapIsNegative"] = (overlap_x_final < 0) or (overlap_y_final < 0)

    # Defaults if empty - Only apply fluorescence defaults for non-RGB images
    # RGB images are typically brightfield/transmitted light and don't have fluorescence metadata
    channels_count = metadata.get('channels', 1)
    if not metadata['isrgb']:
        if not metadata['emission']:
            metadata['emission'] = [500] * channels_count
        if not metadata['excitation']:
            metadata['excitation'] = [480] * channels_count
        # Ensure filterblock has the right number of entries if empty
        if not metadata['filterblock']:
            metadata['filterblock'] = ['Unknown'] * channels_count
        elif len(metadata['filterblock']) < channels_count:
            metadata['filterblock'].extend(['Unknown'] * (channels_count - len(metadata['filterblock'])))


    # Consolidate dimensions
    metadata['dimensions'] = {
        'x': metadata['xs'],
        'y': metadata['ys'],
        'z': metadata['zs'],
        'c': metadata['channels'],
        't': metadata['ts'],
        's': metadata['tiles'],
        'isrgb': metadata['isrgb'],
    }

    return metadata