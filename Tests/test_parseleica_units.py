import os
import sys
import unittest
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ParseLeicaImageXML import parse_image_xml
from ParseLeicaImageXMLLite import parse_image_xml_lite


def _build_image_xml(unit='meters'):
    root = ET.Element('Element', {'Name': 'img', 'UniqueID': '1'})
    data = ET.SubElement(root, 'Data')
    image = ET.SubElement(data, 'Image')
    description = ET.SubElement(image, 'ImageDescription')
    dimensions = ET.SubElement(description, 'Dimensions')

    ET.SubElement(dimensions, 'DimensionDescription', {
        'DimID': '1',
        'NumberOfElements': '3',
        'Length': '0.000002',
        'Unit': unit,
        'BytesInc': '1',
    })
    ET.SubElement(dimensions, 'DimensionDescription', {
        'DimID': '2',
        'NumberOfElements': '3',
        'Length': '0.000004',
        'Unit': unit,
        'BytesInc': '1',
    })
    ET.SubElement(dimensions, 'DimensionDescription', {
        'DimID': '3',
        'NumberOfElements': '2',
        'Length': '0.000003',
        'Unit': unit,
        'BytesInc': '1',
    })
    return root


def _build_reverse_z_image_xml(unit='meters'):
    root = _build_image_xml(unit)
    z_dimension = root.find('.//DimensionDescription[@DimID="3"]')
    z_dimension.set('NumberOfElements', '32')
    z_dimension.set('Length', '-0.0000091884')
    dimensions = root.find('.//Dimensions')
    ET.SubElement(dimensions, 'DimensionDescription', {
        'DimID': '4',
        'NumberOfElements': '16',
        'Length': '900.05',
        'Unit': 's',
        'BytesInc': '1',
    })
    return root


class ParseLeicaUnitsTests(unittest.TestCase):
    def test_parse_image_xml_converts_plural_meter_units(self):
        meta = parse_image_xml(_build_image_xml(unit='meters'))

        self.assertEqual(meta['resunit2'], 'micrometer')
        self.assertAlmostEqual(meta['xres'], 1e-6)
        self.assertAlmostEqual(meta['xres2'], 1.0)
        self.assertAlmostEqual(meta['yres2'], 2.0)
        self.assertAlmostEqual(meta['zres2'], 3.0)

    def test_parse_image_xml_lite_converts_plural_meter_units(self):
        meta = parse_image_xml_lite(_build_image_xml(unit='meters'))

        self.assertEqual(meta['resunit2'], 'micrometer')
        self.assertAlmostEqual(meta['xres'], 1e-6)
        self.assertAlmostEqual(meta['xres2'], 1.0)
        self.assertAlmostEqual(meta['yres2'], 2.0)
        self.assertAlmostEqual(meta['zres2'], 3.0)

    def test_full_parser_treats_reverse_z_length_as_positive_spacing(self):
        meta = parse_image_xml(_build_reverse_z_image_xml())

        self.assertAlmostEqual(meta['zres'], 2.964e-7)
        self.assertAlmostEqual(meta['zres2'], 0.2964)

    def test_lite_parser_treats_reverse_z_length_as_positive_spacing(self):
        meta = parse_image_xml_lite(_build_reverse_z_image_xml())

        self.assertAlmostEqual(meta['zres'], 2.964e-7)
        self.assertAlmostEqual(meta['zres2'], 0.2964)


if __name__ == '__main__':
    unittest.main()
