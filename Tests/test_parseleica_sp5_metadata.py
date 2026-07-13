import os
import sys
import unittest
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ParseLeicaImageXML import parse_image_xml


def _legacy_sp5_image():
    root = ET.Element('Element', {'Name': 'Job 3_009'})
    image = ET.SubElement(ET.SubElement(root, 'Data'), 'Image')
    description = ET.SubElement(image, 'ImageDescription')
    channels = ET.SubElement(description, 'Channels')
    ET.SubElement(channels, 'ChannelDescription', {
        'Resolution': '12', 'LUTName': 'Green', 'BytesInc': '0', 'ChannelTag': '0',
    })
    dimensions = ET.SubElement(description, 'Dimensions')
    for dim_id, count, origin, length, unit in (
        ('1', '1024', '0', '0.00009372629', 'm'),
        ('2', '1024', '0', '0.00009372629', 'm'),
        ('3', '32', '0.01222694', '-0.0000091884', 'm'),
        ('4', '16', '0', '900.05', 's'),
    ):
        ET.SubElement(dimensions, 'DimensionDescription', {
            'DimID': dim_id, 'NumberOfElements': count, 'Origin': origin,
            'Length': length, 'Unit': unit, 'BytesInc': '1',
        })

    hardware = ET.SubElement(image, 'Attachment', {'Name': 'HardwareSettingList'})
    scanner = ET.SubElement(hardware, 'ScannerSetting')
    scanner_values = {
        'SystemType': 'TCS SP5', 'csScanMode': 'xyzt', 'dblZoom': '2.5',
        'eDirectional': '1', 'nAccumulation': '1', 'nAverageFrame': '1',
        'nAverageLine': '1', 'nLineAccumulation': '3', 'dblPinhole': '0.000156',
        'dblPinholeAiry': '1.0', 'nSections': '32',
    }
    for identifier, value in scanner_values.items():
        ET.SubElement(scanner, 'ScannerSettingRecord', {
            'Identifier': identifier, 'Variant': value, 'Unit': '', 'Description': '',
        })

    filters = ET.SubElement(hardware, 'FilterSetting')
    records = (
        ('Hardware Tree (5100001274)', 'CFolderHardwareTree', 'System_Number', '5100001274', ''),
        ('DM6000', 'CMicroscopeStand', 'TLD_Settings', '100', ''),
        ('DM6000 Turret', 'CTurret', 'Objective', 'HCX APO L U-V-I  63.0x0.90 WATER  UV', ''),
        ('DM6000 Turret', 'CTurret', 'OrderNumber', '11506148', ''),
        ('DM6000 Turret', 'CTurret', 'NumericalAperture', '0.9', ''),
        ('DM6000 Turret', 'CTurret', 'RefractionIndex', '1.33', ''),
        ('Visible AOTF-7', 'CAotf', 'Intensity', '14.997253', 'AOTF (514)'),
        ('Laser (Argon, visible)', 'CLaser', 'Wavelength', '458', ''),
        ('Laser (Argon, visible)', 'CLaser', 'Power State', 'On', ''),
        ('Laser (Argon, visible)', 'CLaser', 'Output Power', '50', ''),
        ('HyD 1', 'CDetectionUnit', 'State', 'Active', ''),
        ('HyD 1', 'CDetectionUnit', 'HighVoltage', '10', ''),
        ('HyD 1', 'CDetectionUnit', 'AcquisitionMode', 'PhotonCounting', ''),
        ('Scan Head', 'CScanCtrlUnit', 'Speed', '8000', ''),
        ('Scan Head', 'CScanCtrlUnit', 'Phase', '3.682', ''),
        ('Reson. Galvo Pan', 'CFilterWheel', 'Filter', 'Galvo X Pan Center', ''),
        ('Scan Field Rotator MA4', 'CRotator', 'Scan Rotation', '0', ''),
        ('DM6000 Stage', 'CXYZStage', 'XPos', '0.0068', ''),
        ('DM6000 Stage', 'CXYZStage', 'YPos', '0.0054', ''),
        ('SP Mirror Channel 1', 'CSpectrophotometerUnit', 'Wavelength', '524', ''),
        ('SP Mirror Channel 1', 'CSpectrophotometerUnit', 'Wavelength', '649.441552', ''),
    )
    for object_name, class_name, attribute, value, record_description in records:
        ET.SubElement(filters, 'FilterSettingRecord', {
            'ObjectName': object_name, 'ClassName': class_name, 'Attribute': attribute,
            'Variant': value, 'Description': record_description,
        })
    return root


class ParseLeicaSP5MetadataTests(unittest.TestCase):
    def test_maps_legacy_sp5_objective_microscope_and_acquisition_metadata(self):
        metadata = parse_image_xml(_legacy_sp5_image())

        self.assertEqual(metadata['SystemTypeName'], 'TCS SP5')
        self.assertEqual(metadata['MicroscopeModel'], 'DM6000')
        self.assertEqual(metadata['mic_type'], 'IncohConfMicr')
        self.assertEqual(metadata['mic_type2'], 'confocal')
        self.assertEqual(metadata['system_serial_number'], '5100001274')
        self.assertEqual(metadata['objective'], 'HCX APO L U-V-I  63.0x0.90 WATER  UV')
        self.assertEqual(metadata['objective_order_number'], '11506148')
        self.assertEqual(metadata['magnification'], 63.0)
        self.assertEqual(metadata['na'], 0.9)
        self.assertEqual(metadata['refractiveindex'], 1.33)
        self.assertEqual(metadata['immersion'], 'Water')
        self.assertAlmostEqual(metadata['pinholesize_um'], 156.0)
        self.assertEqual(metadata['scan_mode'], 'xyzt')
        self.assertEqual(metadata['zoom'], 2.5)
        self.assertEqual(metadata['scan_speed'], 8000)
        self.assertTrue(metadata['is_resonant_scanner'])
        self.assertEqual(metadata['line_accumulation'], 3)
        self.assertEqual(metadata['stage_pos_x_m'], 0.0068)
        self.assertAlmostEqual(metadata['zstack_end_m'], 0.0122177516)

    def test_maps_legacy_sp5_laser_detector_and_spectral_metadata(self):
        metadata = parse_image_xml(_legacy_sp5_image())

        self.assertEqual(metadata['laser_intensities'], [
            {'wavelength_nm': 514, 'intensity_percent': 15.0},
        ])
        self.assertEqual(metadata['lasers'], [{
            'name': 'Laser (Argon, visible)', 'wavelength_nm': 458.0,
            'power_state': 'On', 'output_power_percent': 50.0,
        }])
        self.assertEqual(metadata['excitation'], [514.0])
        self.assertEqual(metadata['emission'], [587])
        self.assertEqual(metadata['detector_types'], ['HyD'])
        self.assertEqual(metadata['detector_gains'], [10.0])
        self.assertEqual(metadata['detector_acquisition_modes'], ['PhotonCounting'])
        self.assertAlmostEqual(metadata['spectral_windows'][0]['right_nm'], 649.441552)
        self.assertIn('SystemType', metadata['legacy_scanner_settings'])
        self.assertGreater(len(metadata['legacy_hardware_settings']), 10)


if __name__ == '__main__':
    unittest.main()
