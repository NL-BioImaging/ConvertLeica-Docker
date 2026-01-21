import uuid
import xml.etree.ElementTree as ET
import os
import shutil
import tempfile
import time
from ci_leica_converters_helpers import print_progress_bar, read_image_metadata, print_copy_progress_bar

def convert_leica_to_singlelif(inputfile, image_uuid, outputfolder=None, show_progress=True, altoutputfolder=None):
    """
    Creates a LIF file from a single image within an existing LIF file,
    using the image's UUID to extract its metadata.

    Args:
        inputfile (str): Path to the original LIF file.
        image_uuid (str): UUID of the image to extract.
        outputfolder (str, optional): Full path to output folder. If None, same directory as the input file is used.
        show_progress (bool): Whether to show progress (default=True).
        altoutputfolder (str, optional): Optional alternative second output folder. Defaults to None.

    Returns:
        str: The filename of the created LIF file (without path), or None if an error occurred.
    """
    try:
        if show_progress:
            print_progress_bar(5.0, prefix='Creating Single LIF:', suffix='Reading metadata')

        metadata = read_image_metadata(inputfile, image_uuid)

        if show_progress:
            print_progress_bar(10.0, prefix='Creating Single LIF:', suffix='Processing metadata')

        xml_element = metadata.get('xmlElement')
        save_child_name = metadata.get('save_child_name')
        name = metadata.get('name')

        if xml_element:
            root = ET.fromstring(xml_element)
            for el in root.iter('Element'):
                if el.get('Name') == name:
                    el.set('Name', save_child_name)
            xml_element = ET.tostring(root, encoding='unicode')

        BlockID = metadata.get('BlockID')
        memory_size = metadata.get('MemorySize')
        image_data_path = inputfile
        image_data_position = metadata.get('Position')

        if outputfolder is None:
            outputfolder = os.path.dirname(inputfile)
        
        os.makedirs(outputfolder, exist_ok=True)
        
        if altoutputfolder is not None:
            os.makedirs(altoutputfolder, exist_ok=True)

        base_lif_filename = save_child_name + ".lif"
        lif_filepath = os.path.join(outputfolder, base_lif_filename)

        if show_progress:
            print_progress_bar(30.0, prefix='Creating Single LIF:', suffix='Creating LIF file header')

        outxml = '<LMSDataContainerHeader Version="2"><Element CopyOption="1" Name="_name_" UniqueID="_uuid_" Visibility="1"> <Data><Experiment IsSavedFlag="1" Path="_path_"/></Data><Memory MemoryBlockID="MemBlock_221" Size="0"/><Children>_element_</Children></Element></LMSDataContainerHeader>'
        outxml = outxml.replace('_name_', save_child_name)
        outxml = outxml.replace('_path_', lif_filepath) 
        outxml = outxml.replace('_uuid_', str(uuid.uuid4()))
        outxml = outxml.replace('_element_', xml_element)

        outxml = outxml.replace('\n', '')
        outxml = ' '.join(outxml.split())
        outxml = outxml.replace('</Data>', '</Data>\r\n')
        outxml = outxml.replace('</LMSDataContainerHeader>', '</LMSDataContainerHeader>\r\n')
        outxml16 = outxml.encode('utf-16')[2:]

        if show_progress:
            print_progress_bar(40.0, prefix='Creating Single LIF:', suffix='Writing LIF file structure')

        with open(lif_filepath, 'wb') as fid:
            fid.write(int(0x70).to_bytes(4, 'little'))
            fid.write(int(len(outxml16) + 1 + 4).to_bytes(4, 'little'))
            fid.write(int(0x2A).to_bytes(1, 'little'))
            fid.write(int(len(outxml16) // 2).to_bytes(4, 'little'))
            fid.write(outxml16)

            elementMemID = "MemBlock_221"
            msize = 0
            mdescription = f"{elementMemID}".encode('utf-16')[2:]        
            fid.write(int(0x70).to_bytes(4, 'little'))
            fid.write(int(len(mdescription) + 1 + 8 + 1 + 4).to_bytes(4, 'little'))
            fid.write(int(0x2A).to_bytes(1, 'little'))
            fid.write(int(msize).to_bytes(8, 'little'))
            fid.write(int(0x2A).to_bytes(1, 'little'))
            fid.write(int(len(mdescription) // 2).to_bytes(4, 'little'))
            fid.write(mdescription)

            elementMemID = BlockID
            msize = memory_size
            mdescription = f"{elementMemID}".encode('utf-16')[2:]        
            fid.write(int(0x70).to_bytes(4, 'little'))
            fid.write(int(len(mdescription) + 1 + 8 + 1 + 4).to_bytes(4, 'little'))
            fid.write(int(0x2A).to_bytes(1, 'little'))
            fid.write(int(msize).to_bytes(8, 'little'))
            fid.write(int(0x2A).to_bytes(1, 'little'))
            fid.write(int(len(mdescription) // 2).to_bytes(4, 'little'))
            fid.write(mdescription)

            if image_data_path and msize > 0:
                copy_memory_block_with_text_progress(image_data_path, fid, msize, image_data_position, show_progress)

        if show_progress:
            print_progress_bar(100.0, prefix='Creating Single LIF:', suffix='Complete', final_call=True)
        
        print(f"LIF file created: {lif_filepath}") 

        if altoutputfolder is not None:
            alt_out_path = os.path.join(altoutputfolder, base_lif_filename)
            shutil.copy2(lif_filepath, alt_out_path)
            print(f"LIF file also copied to: {alt_out_path}")

        return base_lif_filename 
        
    except ValueError as ve:
        print(f"\nError processing metadata for UUID {image_uuid}: {str(ve)}")
        return None
    except Exception as e:
        print(f"\nError creating LIF file: {str(e)}") 
        return None

def copy_memory_block(input_file, output_file, memory_size, offset):
    """
    Original function - Copies a memory block from an input file to an output file
    in chunks, starting from a specific offset.
    """
    block_size = 25600000
    num_full_blocks = memory_size // block_size
    final_block_size = memory_size % block_size

    with open(input_file, 'rb') as fid:
        fid.seek(offset, os.SEEK_SET)

        for i in range(num_full_blocks):
            memblock = fid.read(block_size)
            output_file.write(memblock)

        if final_block_size > 0:
            memblock = fid.read(final_block_size)
            output_file.write(memblock)

def copy_memory_block_with_text_progress(input_file, output_file, memory_size, offset, show_progress):
    """
    Console progress version - Copies a memory block from an input file to an output file
    in chunks, updating progress via console bar, spanning from 40% to 95% of the overall task.
    """
    block_size = 25600000
    num_full_blocks = memory_size // block_size
    final_block_size = memory_size % block_size

    total_blocks = num_full_blocks + (1 if final_block_size > 0 else 0)
    overall_progress_start = 40.0
    overall_progress_end = 95.0
    overall_progress_span = overall_progress_end - overall_progress_start

    if total_blocks == 0:
        if show_progress:
            print_progress_bar(overall_progress_end, prefix='Creating Single LIF:', suffix="No data to copy")
        return

    progress_increment = overall_progress_span / total_blocks

    with open(input_file, 'rb') as fid:
        fid.seek(offset, os.SEEK_SET)

        for i in range(num_full_blocks):
            memblock = fid.read(block_size)
            output_file.write(memblock)
            if show_progress:
                current_progress = min(overall_progress_end, overall_progress_start + ((i + 1) * progress_increment))
                print_progress_bar(current_progress, prefix='Creating Single LIF:', 
                                   suffix=f"Copying data: block {i + 1}/{total_blocks}")

        if final_block_size > 0:
            memblock = fid.read(final_block_size)
            output_file.write(memblock)
            if show_progress:
                print_progress_bar(overall_progress_end, prefix='Creating Single LIF:', suffix="Data copy complete")


def copy_memory_block_with_retry(input_file, output_file, memory_size, offset, 
                                  show_progress=False, max_retries=10, prefix="Copying source:"):
    """
    Robust version - Copies a memory block from an input file to an output file
    in chunks with retry logic for network reliability.
    
    Parameters:
    - input_file: Source file path
    - output_file: Open file handle to write to
    - memory_size: Total bytes to copy
    - offset: Starting offset in source file
    - show_progress: If True, display progress
    - max_retries: Maximum retry attempts per block
    - prefix: Progress bar prefix text
    
    Returns position in output file after writing.
    
    Raises OSError if all retries exhausted.
    """
    block_size = 25600000  # ~25MB chunks
    num_full_blocks = memory_size // block_size
    final_block_size = memory_size % block_size
    
    total_blocks = num_full_blocks + (1 if final_block_size > 0 else 0)
    
    if total_blocks == 0:
        if show_progress:
            print_copy_progress_bar(100, prefix=prefix, suffix="No data to copy", 
                                   final_call=True, phase=prefix)
        return output_file.tell()
    
    bytes_written = 0
    
    # Format size nicely for display
    def format_size(bytes_val, total):
        if total >= 1024 * 1024 * 1024:
            return f"{bytes_val / (1024*1024*1024):.1f}/{total / (1024*1024*1024):.1f} GB"
        elif total >= 1024 * 1024:
            return f"{bytes_val / (1024*1024):.0f}/{total / (1024*1024):.0f} MB"
        else:
            return f"{bytes_val / 1024:.0f}/{total / 1024:.0f} KB"
    
    with open(input_file, 'rb') as fid:
        fid.seek(offset, os.SEEK_SET)
        
        for block_num in range(num_full_blocks):
            current_block_size = block_size
            block_written = False
            
            for attempt in range(max_retries + 1):
                try:
                    # Remember position for potential retry
                    read_pos = offset + block_num * block_size
                    fid.seek(read_pos, os.SEEK_SET)
                    
                    memblock = fid.read(current_block_size)
                    if len(memblock) != current_block_size:
                        raise IOError(f"Short read: expected {current_block_size}, got {len(memblock)}")
                    
                    output_file.write(memblock)
                    bytes_written += len(memblock)
                    block_written = True
                    
                    if show_progress:
                        pct = 100.0 * bytes_written / memory_size
                        size_str = format_size(bytes_written, memory_size)
                        print_copy_progress_bar(pct, prefix=prefix, suffix=size_str, phase=prefix)
                    break
                    
                except (OSError, IOError) as e:
                    if attempt < max_retries:
                        wait_minutes = attempt + 1
                        if show_progress:
                            print(f"\nWarning: Read block {block_num + 1} attempt {attempt + 1} failed: {e}")
                            print(f"  Retrying in {wait_minutes} minute(s)...")
                        time.sleep(wait_minutes * 60)
                    else:
                        raise OSError(
                            f"Failed to read block {block_num + 1} after {max_retries + 1} attempts. "
                            f"Last error: {e}"
                        )
            
            if not block_written:
                raise OSError(f"Block {block_num + 1} was not written")
        
        # Handle final partial block
        if final_block_size > 0:
            for attempt in range(max_retries + 1):
                try:
                    read_pos = offset + num_full_blocks * block_size
                    fid.seek(read_pos, os.SEEK_SET)
                    
                    memblock = fid.read(final_block_size)
                    if len(memblock) != final_block_size:
                        raise IOError(f"Short read: expected {final_block_size}, got {len(memblock)}")
                    
                    output_file.write(memblock)
                    bytes_written += len(memblock)
                    
                    if show_progress:
                        size_str = format_size(bytes_written, memory_size)
                        print_copy_progress_bar(100, prefix=prefix, suffix=size_str, 
                                               final_call=True, phase=prefix)
                    break
                    
                except (OSError, IOError) as e:
                    if attempt < max_retries:
                        wait_minutes = attempt + 1
                        if show_progress:
                            print(f"\nWarning: Read final block attempt {attempt + 1} failed: {e}")
                            print(f"  Retrying in {wait_minutes} minute(s)...")
                        time.sleep(wait_minutes * 60)
                    else:
                        raise OSError(
                            f"Failed to read final block after {max_retries + 1} attempts. "
                            f"Last error: {e}"
                        )
    
    return output_file.tell()


def convert_leica_to_singlelif_temp(inputfile, image_uuid, tempfolder=None, 
                                     show_progress=False, max_retries=10):
    """
    Creates a temporary single-image LIF file from an image within an existing LIF file.
    Uses robust chunked reading with retry logic for network reliability.
    
    This is used by prepare_temp_source to create a reliable local copy of LIF image data.
    
    Args:
        inputfile (str): Path to the original LIF file.
        image_uuid (str): UUID of the image to extract.
        tempfolder (str, optional): Temp folder path. If None, uses system temp.
        show_progress (bool): Whether to show progress (default=False).
        max_retries (int): Maximum retry attempts for each block (default=10).
    
    Returns:
        tuple: (temp_lif_path, position) where:
            - temp_lif_path: Path to the created temp LIF file
            - position: Byte position of the image data in the temp LIF (for reading)
    
    Raises:
        ValueError: If metadata cannot be read
        OSError: If file operations fail after all retries
    """
    temp_dir = tempfolder if tempfolder else tempfile.gettempdir()
    os.makedirs(temp_dir, exist_ok=True)
    
    # Generate unique temp filename
    temp_filename = f"{uuid.uuid4()}.tmp.lif"
    temp_lif_path = os.path.join(temp_dir, temp_filename)
    
    try:
        if show_progress:
            print_progress_bar(5.0, prefix='Extracting to temp:', suffix='Reading metadata')
        
        metadata = read_image_metadata(inputfile, image_uuid)
        
        if show_progress:
            print_progress_bar(10.0, prefix='Extracting to temp:', suffix='Processing metadata')
        
        xml_element = metadata.get('xmlElement')
        save_child_name = metadata.get('save_child_name', f'image_{image_uuid}')
        name = metadata.get('name')
        
        if xml_element:
            root = ET.fromstring(xml_element)
            for el in root.iter('Element'):
                if el.get('Name') == name:
                    el.set('Name', save_child_name)
            xml_element = ET.tostring(root, encoding='unicode')
        
        BlockID = metadata.get('BlockID')
        memory_size = metadata.get('MemorySize')
        image_data_path = inputfile
        image_data_position = metadata.get('Position')
        
        if not memory_size or memory_size <= 0:
            raise ValueError(f"Invalid MemorySize in metadata: {memory_size}")
        
        if show_progress:
            print_progress_bar(15.0, prefix='Extracting to temp:', suffix='Creating LIF header')
        
        # Build LIF structure
        outxml = '<LMSDataContainerHeader Version="2"><Element CopyOption="1" Name="_name_" UniqueID="_uuid_" Visibility="1"> <Data><Experiment IsSavedFlag="1" Path="_path_"/></Data><Memory MemoryBlockID="MemBlock_221" Size="0"/><Children>_element_</Children></Element></LMSDataContainerHeader>'
        outxml = outxml.replace('_name_', save_child_name)
        outxml = outxml.replace('_path_', temp_lif_path)
        outxml = outxml.replace('_uuid_', str(uuid.uuid4()))
        outxml = outxml.replace('_element_', xml_element if xml_element else '')
        
        outxml = outxml.replace('\n', '')
        outxml = ' '.join(outxml.split())
        outxml = outxml.replace('</Data>', '</Data>\r\n')
        outxml = outxml.replace('</LMSDataContainerHeader>', '</LMSDataContainerHeader>\r\n')
        outxml16 = outxml.encode('utf-16')[2:]
        
        if show_progress:
            print_progress_bar(20.0, prefix='Extracting to temp:', suffix='Writing LIF structure')
        
        with open(temp_lif_path, 'wb') as fid:
            # Write header block marker
            fid.write(int(0x70).to_bytes(4, 'little'))
            fid.write(int(len(outxml16) + 1 + 4).to_bytes(4, 'little'))
            fid.write(int(0x2A).to_bytes(1, 'little'))
            fid.write(int(len(outxml16) // 2).to_bytes(4, 'little'))
            fid.write(outxml16)
            
            # Write empty memory block for container
            elementMemID = "MemBlock_221"
            msize = 0
            mdescription = f"{elementMemID}".encode('utf-16')[2:]
            fid.write(int(0x70).to_bytes(4, 'little'))
            fid.write(int(len(mdescription) + 1 + 8 + 1 + 4).to_bytes(4, 'little'))
            fid.write(int(0x2A).to_bytes(1, 'little'))
            fid.write(int(msize).to_bytes(8, 'little'))
            fid.write(int(0x2A).to_bytes(1, 'little'))
            fid.write(int(len(mdescription) // 2).to_bytes(4, 'little'))
            fid.write(mdescription)
            
            # Write actual data memory block header
            elementMemID = BlockID
            msize = memory_size
            mdescription = f"{elementMemID}".encode('utf-16')[2:]
            fid.write(int(0x70).to_bytes(4, 'little'))
            fid.write(int(len(mdescription) + 1 + 8 + 1 + 4).to_bytes(4, 'little'))
            fid.write(int(0x2A).to_bytes(1, 'little'))
            fid.write(int(msize).to_bytes(8, 'little'))
            fid.write(int(0x2A).to_bytes(1, 'little'))
            fid.write(int(len(mdescription) // 2).to_bytes(4, 'little'))
            fid.write(mdescription)
            
            # Record position where image data starts in temp LIF
            data_position = fid.tell()
            
            if show_progress:
                print_progress_bar(25.0, prefix='Extracting to temp:', suffix='Copying image data')
            
            # Copy image data with retry logic
            if image_data_path and msize > 0:
                copy_memory_block_with_retry(
                    image_data_path, fid, msize, image_data_position,
                    show_progress=show_progress, max_retries=max_retries,
                    prefix='Extracting to temp:'
                )
        
        if show_progress:
            print_progress_bar(100.0, prefix='Extracting to temp:', suffix='Complete', final_call=True)
        
        return (temp_lif_path, data_position)
        
    except Exception as e:
        # Clean up partial temp file on error
        if os.path.exists(temp_lif_path):
            try:
                os.remove(temp_lif_path)
            except OSError:
                pass
        raise