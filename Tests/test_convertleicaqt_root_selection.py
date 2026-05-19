import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ConvertLeicaQT import _root_has_single_image, parse_progress_text


class ConvertLeicaQTRootSelectionTests(unittest.TestCase):
    def test_root_with_single_visible_image_is_auto_selectable(self):
        meta = {
            'children': [
                {'name': 'Metadata', 'type': 'folder'},
                {'name': 'Image 1', 'type': 'image', 'uuid': 'img-1'},
            ]
        }

        self.assertTrue(_root_has_single_image(meta))

    def test_root_with_folder_is_not_auto_selectable(self):
        meta = {
            'children': [
                {'name': 'Folder 1', 'type': 'folder', 'uuid': 'folder-1'},
            ]
        }

        self.assertFalse(_root_has_single_image(meta))

    def test_root_with_multiple_visible_children_is_not_auto_selectable(self):
        meta = {
            'children': [
                {'name': 'Image 1', 'type': 'image', 'uuid': 'img-1'},
                {'name': 'Image 2', 'type': 'image', 'uuid': 'img-2'},
            ]
        }

        self.assertFalse(_root_has_single_image(meta))

    def test_extracting_to_temp_progress_is_labeled_as_preparing_input(self):
        parsed = parse_progress_text(
            'Extracting to temp: |████████████████████████████████████████| 100.0% Temp file ready, starting conversion'
        )

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed['phase'], 'Preparing input')
        self.assertEqual(parsed['suffix'], 'Temp file ready, starting conversion')


if __name__ == '__main__':
    unittest.main()
