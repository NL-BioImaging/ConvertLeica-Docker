#!/usr/bin/env python3
"""
Validate OME-TIFF files for OMERO compatibility.
Usage: python validate_ometiff.py <path_to_ome.tiff>
"""

import sys
import os
import xml.etree.ElementTree as ET

def validate_ometiff(filepath):
    """Validate an OME-TIFF file structure and metadata."""
    print(f"\n{'='*60}")
    print(f"Validating: {filepath}")
    print(f"{'='*60}\n")
    
    errors = []
    warnings = []
    
    try:
        import tifffile
    except ImportError:
        print("ERROR: tifffile not installed. Run: pip install tifffile")
        return False
    
    # 1. Basic TIFF structure
    print("1. Checking TIFF structure...")
    try:
        with tifffile.TiffFile(filepath) as tif:
            print(f"   - Number of pages (IFDs): {len(tif.pages)}")
            print(f"   - Is OME-TIFF: {tif.is_ome}")
            print(f"   - Is BigTIFF: {tif.is_bigtiff}")
            
            # Check first page for basic info
            page0 = tif.pages[0]
            print(f"   - Image shape (first page): {page0.shape}")
            print(f"   - Data type: {page0.dtype}")
            print(f"   - Compression: {page0.compression}")
            print(f"   - Photometric: {page0.photometric}")
            
            # 2. OME-XML validation
            print("\n2. Checking OME-XML metadata...")
            if tif.ome_metadata:
                ome_xml = tif.ome_metadata
                print(f"   - OME-XML length: {len(ome_xml)} characters")
                
                # Parse XML
                try:
                    # Remove namespace for easier parsing
                    ome_xml_clean = ome_xml.replace('xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"', '')
                    root = ET.fromstring(ome_xml_clean)
                    
                    # Check for Image element
                    image = root.find('.//Image')
                    if image is not None:
                        print(f"   - Image Name: {image.get('Name', 'N/A')}")
                        
                        pixels = image.find('.//Pixels')
                        if pixels is not None:
                            size_x = pixels.get('SizeX')
                            size_y = pixels.get('SizeY')
                            size_z = pixels.get('SizeZ')
                            size_c = pixels.get('SizeC')
                            size_t = pixels.get('SizeT')
                            pixel_type = pixels.get('Type')
                            dim_order = pixels.get('DimensionOrder')
                            
                            print(f"   - Dimensions: X={size_x}, Y={size_y}, Z={size_z}, C={size_c}, T={size_t}")
                            print(f"   - Pixel Type: {pixel_type}")
                            print(f"   - Dimension Order: {dim_order}")
                            
                            # Check TiffData elements
                            tiffdata_elements = pixels.findall('.//TiffData')
                            print(f"   - TiffData elements: {len(tiffdata_elements)}")
                            
                            # Check Channel elements
                            channel_elements = pixels.findall('.//Channel')
                            print(f"   - Channel elements: {len(channel_elements)}")
                            
                            # Validate counts
                            expected_pages = int(size_z or 1) * int(size_c or 1) * int(size_t or 1)
                            actual_pages = len(tif.pages)
                            
                            print(f"\n3. Validating consistency...")
                            print(f"   - Expected IFDs (Z*C*T): {expected_pages}")
                            print(f"   - Actual IFDs in file: {actual_pages}")
                            
                            if expected_pages != actual_pages:
                                errors.append(f"IFD count mismatch: expected {expected_pages}, got {actual_pages}")
                            
                            if len(tiffdata_elements) != actual_pages:
                                errors.append(f"TiffData count mismatch: {len(tiffdata_elements)} TiffData elements but {actual_pages} IFDs")
                            
                            if int(size_c or 1) != len(channel_elements):
                                warnings.append(f"Channel count mismatch: SizeC={size_c} but {len(channel_elements)} Channel elements")
                            
                            # Check pixel dimensions match actual data
                            if page0.shape[0] != int(size_y) or page0.shape[1] != int(size_x):
                                errors.append(f"Image dimensions mismatch: OME says {size_x}x{size_y}, actual is {page0.shape[1]}x{page0.shape[0]}")
                            
                        else:
                            errors.append("No Pixels element found in OME-XML")
                    else:
                        errors.append("No Image element found in OME-XML")
                        
                except ET.ParseError as e:
                    errors.append(f"OME-XML parse error: {e}")
                    
            else:
                errors.append("No OME-XML metadata found in file")
            
            # 3. Check for common issues
            print("\n4. Checking for common issues...")
            
            # Check if all pages are readable
            print("   - Testing page readability...")
            try:
                for i, page in enumerate(tif.pages[:5]):  # Test first 5 pages
                    data = page.asarray()
                    if data is None:
                        errors.append(f"Page {i} returned None")
                print(f"   - First 5 pages readable: OK")
            except Exception as e:
                errors.append(f"Error reading pages: {e}")
            
            # Check compression compatibility
            for i, page in enumerate(tif.pages[:1]):
                comp = str(page.compression)
                if 'JPEG' in comp.upper():
                    warnings.append(f"JPEG compression detected - may cause issues with some readers")
                    
    except Exception as e:
        errors.append(f"Failed to open file: {e}")
        import traceback
        traceback.print_exc()
    
    # Print summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    if errors:
        print(f"\n❌ ERRORS ({len(errors)}):")
        for err in errors:
            print(f"   - {err}")
    else:
        print("\n✓ No errors found")
    
    if warnings:
        print(f"\n⚠ WARNINGS ({len(warnings)}):")
        for warn in warnings:
            print(f"   - {warn}")
    
    print()
    return len(errors) == 0


def dump_ome_xml(filepath, output_path=None):
    """Extract and save OME-XML from a TIFF file for inspection."""
    try:
        import tifffile
        with tifffile.TiffFile(filepath) as tif:
            if tif.ome_metadata:
                if output_path is None:
                    output_path = filepath.replace('.tiff', '_ome.xml').replace('.tif', '_ome.xml')
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(tif.ome_metadata)
                print(f"OME-XML saved to: {output_path}")
                return output_path
    except Exception as e:
        print(f"Error extracting OME-XML: {e}")
    return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_ometiff.py <path_to_ome.tiff> [--dump-xml]")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        sys.exit(1)
    
    # Dump XML if requested
    if "--dump-xml" in sys.argv:
        dump_ome_xml(filepath)
    
    # Validate
    is_valid = validate_ometiff(filepath)
    sys.exit(0 if is_valid else 1)
