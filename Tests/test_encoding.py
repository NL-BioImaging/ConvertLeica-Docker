"""Test µm encoding for OME-TIFF"""

# Test µm encoding
um = '\u00B5m'
print(f'Character: {um}')
print(f'Unicode codepoints: {[hex(ord(c)) for c in um]}')
print(f'UTF-8 bytes: {um.encode("utf-8")}')
print(f'Length: {len(um)} chars, {len(um.encode("utf-8"))} bytes')

# Simulate what pyvips should see
xml_snippet = f'PhysicalSizeXUnit="{um}"'
print(f'\nXML snippet: {xml_snippet}')
print(f'Bytes: {xml_snippet.encode("utf-8")}')

# Check if double encoding happens (this is what was broken before)
bad = um.encode('utf-8').decode('latin-1')  # This simulates double-encoding
print(f'\n--- Encoding comparison ---')
print(f'Good (correct): "{um}"')
print(f'Bad (double-encoded): "{bad}"')

# The fix: pyvips expects a Python string, NOT bytes
# Before: img.set_type(..., ome_xml.encode("utf-8"))  <- WRONG, causes double encoding
# After:  img.set_type(..., ome_xml)  <- CORRECT, pyvips handles encoding

print(f'\n--- Verification ---')
if um == 'µm':
    print('✓ µm character is correct')
else:
    print('✗ µm character is wrong')

# Test that the string can be written to a file correctly
test_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<Pixels PhysicalSizeXUnit="{um}" PhysicalSizeYUnit="{um}" PhysicalSizeZUnit="{um}"/>
'''
print(f'\nTest XML:\n{test_xml}')
