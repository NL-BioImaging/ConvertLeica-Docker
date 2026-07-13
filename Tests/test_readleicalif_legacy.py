import json
import os
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ReadLeicaLIF import read_leica_lif


class LegacyLIFIdentifiersTests(unittest.TestCase):
    def setUp(self):
        # LAS AF files from older SP5 systems may have no UniqueID attributes.
        # Their zero-sized folder and image elements are instead identified by
        # MemoryBlockID, as in the reported SP5 time-series file.
        xml = (
            '<LMSDataContainerHeader Version="2">'
            '<Element Name="Experiment"><Memory Size="0" MemoryBlockID="MemBlock_17"/>'
            '<Children><Element Name="Sequence">'
            '<Memory Size="0" MemoryBlockID="MemBlock_30"/><Children>'
            '<Element Name="Job 2_008"><Memory Size="4" MemoryBlockID="MemBlock_31"/></Element>'
            '<Element Name="Job 3_009"><Memory Size="4" MemoryBlockID="MemBlock_32"/></Element>'
            '</Children></Element></Children></Element>'
            '</LMSDataContainerHeader>'
        )
        encoded_xml = xml.encode('utf-16')
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tempdir.name, 'legacy-sp5.lif')
        with open(self.path, 'wb') as lif:
            lif.write(struct.pack('i', 112))
            lif.write(struct.pack('i', 0))
            lif.write(struct.pack('B', 42))
            lif.write(struct.pack('i', len(encoded_xml) // 2))
            lif.write(encoded_xml)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_uses_memory_block_id_for_folder_without_unique_id(self):
        root = json.loads(read_leica_lif(self.path))

        self.assertEqual(root['children'][0]['name'], 'Sequence')
        self.assertEqual(root['children'][0]['uuid'], 'MemBlock_30')

    def test_expands_legacy_folder_and_exposes_image_ids(self):
        folder = json.loads(read_leica_lif(self.path, folder_uuid='MemBlock_30'))

        self.assertEqual(folder['uuid'], 'MemBlock_30')
        self.assertEqual(
            [(child['name'], child['uuid']) for child in folder['children']],
            [('Job 2_008', 'MemBlock_31'), ('Job 3_009', 'MemBlock_32')],
        )
        self.assertTrue(all(child['datatype'] == 'Image' for child in folder['children']))


if __name__ == '__main__':
    unittest.main()
