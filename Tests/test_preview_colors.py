import os
import shutil
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from CreatePreview import _preview_cache_filename, _write_preview_png


class PreviewColorTests(unittest.TestCase):
    def test_preview_png_writer_preserves_rgb_display_colors(self):
        temp_dir = tempfile.mkdtemp()
        try:
            output_path = os.path.join(temp_dir, 'preview.png')
            rgb_image = np.array([[[255, 0, 0]]], dtype=np.uint8)

            _write_preview_png(rgb_image, output_path)

            stored_bgr = cv2.imread(output_path, cv2.IMREAD_COLOR)
            self.assertIsNotNone(stored_bgr)
            self.assertEqual(stored_bgr[0, 0].tolist(), [0, 0, 255])
        finally:
            shutil.rmtree(temp_dir)

    def test_preview_cache_filename_includes_version(self):
        self.assertEqual(_preview_cache_filename('abc', 256), 'abc_pv2_h256.png')


if __name__ == '__main__':
    unittest.main()