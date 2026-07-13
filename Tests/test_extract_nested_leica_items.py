import json
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ci_leica_converters_helpers import extract_nested_leica_items


class ExtractNestedLeicaItemsTests(unittest.TestCase):
    def _nested_reader(self, file_path, include_xmlelement=False, image_uuid=None, folder_uuid=None):
        del include_xmlelement, image_uuid
        trees = {
            None: {
                'type': 'File', 'children': [
                    {'type': 'Folder', 'name': 'Sequence', 'uuid': 'folder-1', 'children': []},
                ],
            },
            'folder-1': {
                'type': 'Folder', 'uuid': 'folder-1', 'children': [
                    {'type': 'Image', 'name': 'Image 1', 'uuid': 'image-1'},
                    {'type': 'Folder', 'name': 'Nested', 'uuid': 'folder-2', 'children': []},
                ],
            },
            'folder-2': {
                'type': 'Folder', 'uuid': 'folder-2', 'children': [
                    {'datatype': 'Image', 'ElementName': 'Image 2', 'BlockID': 'legacy-image-2'},
                    # A cycle must not cause an infinite traversal.
                    {'type': 'Folder', 'name': 'Sequence', 'uuid': 'folder-1', 'children': []},
                ],
            },
        }
        return json.dumps(trees[folder_uuid])

    @patch('ci_leica_converters_helpers.read_leica_file')
    def test_extracts_nested_lif_images_and_legacy_block_ids(self, read_mock):
        read_mock.side_effect = self._nested_reader

        items = extract_nested_leica_items('sample.lif')

        self.assertEqual(items, [
            {'localPath': 'sample.lif', 'uuid': 'image-1', 'name': 'Image 1'},
            {'localPath': 'sample.lif', 'uuid': 'legacy-image-2', 'name': 'Image 2'},
        ])

    @patch('ci_leica_converters_helpers.read_leica_file')
    def test_expands_nested_xlef_folders(self, read_mock):
        read_mock.side_effect = self._nested_reader

        items = extract_nested_leica_items('sample.xlef')

        self.assertEqual([item['uuid'] for item in items], ['image-1', 'legacy-image-2'])

    @patch('ci_leica_converters_helpers.read_leica_file')
    def test_returns_single_lof_image(self, read_mock):
        read_mock.return_value = json.dumps({
            'UniqueID': 'lof-image-id',
            'ElementName': 'LOF image',
            'filetype': '.lof',
        })

        self.assertEqual(extract_nested_leica_items('sample.lof'), [{
            'localPath': 'sample.lof',
            'uuid': 'lof-image-id',
            'name': 'LOF image',
        }])


if __name__ == '__main__':
    unittest.main()
