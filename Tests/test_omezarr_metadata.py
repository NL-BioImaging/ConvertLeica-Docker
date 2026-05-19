import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ci_leica_converters_omezarr import _create_zarr_arrays, _dataset_scales, _open_output_group, _write_group_attrs, _write_ome_xml_companion


class OMEZarrMetadataTests(unittest.TestCase):
    def test_dataset_scales_prefer_normalized_micrometer_values(self):
        meta = {
            'xres': 4.541556558935361e-07,
            'yres': 4.541320101238844e-07,
            'zres': 0.0,
            'xres2': 0.4541556558935361,
            'yres2': 0.4541320101238844,
            'zres2': 0.0,
            'tres': 0.0,
        }

        scales = _dataset_scales(2, meta)

        self.assertEqual(scales[0], [1.0, 1.0, 0.4541320101238844, 0.4541320101238844, 0.4541556558935361])
        self.assertEqual(scales[1], [1.0, 1.0, 0.4541320101238844, 0.9082640202477688, 0.9083113117870723])

    def test_writer_adds_array_dimensions_and_pixel_size_attrs(self):
        meta = {
            'name': 'Example',
            'channels': 3,
            'filetype': '.lif',
            'uuid': 'img-1',
            'save_child_name': 'example',
            'xres2': 0.25,
            'yres2': 0.5,
            'zres2': 1.5,
        }
        temp_dir = tempfile.mkdtemp(suffix='.ome.zarr')
        try:
            group = _open_output_group(temp_dir)
            arrays = _create_zarr_arrays(group, [(1, 3, 1, 16, 16)], 'uint8')
            _write_group_attrs(group, meta, 'uint8', 1, False)

            self.assertEqual(arrays[0].attrs['_ARRAY_DIMENSIONS'], ['t', 'c', 'z', 'y', 'x'])
            self.assertEqual(group.attrs['leica']['pixel_size_x_um'], 0.25)
            self.assertEqual(group.attrs['leica']['pixel_size_y_um'], 0.5)
            self.assertEqual(group.attrs['leica']['pixel_size_z_um'], 1.5)
        finally:
            shutil.rmtree(temp_dir)

    def test_writer_creates_ome_xml_companion_with_physical_sizes(self):
        meta = {
            'name': 'Example',
            'save_child_name': 'example',
            'channels': 2,
            'xs': 64,
            'ys': 32,
            'zs': 1,
            'ts': 1,
            'xres2': 0.25,
            'yres2': 0.5,
            'zres2': 1.5,
            'channelResolution': [16, 16],
            'channel_names': ['A', 'B'],
        }
        temp_dir = tempfile.mkdtemp(suffix='.ome.zarr')
        try:
            _write_ome_xml_companion(temp_dir, meta, 'example.ome.zarr', False, False)

            ome_xml_path = os.path.join(temp_dir, 'OME', 'METADATA.ome.xml')
            self.assertTrue(os.path.exists(ome_xml_path))
            with open(ome_xml_path, 'r', encoding='utf-8') as handle:
                ome_xml = handle.read()

            self.assertIn('PhysicalSizeX="0.25"', ome_xml)
            self.assertIn('PhysicalSizeY="0.5"', ome_xml)
            self.assertIn('PhysicalSizeXUnit="µm"', ome_xml)
            self.assertIn('SizeC="2"', ome_xml)
        finally:
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()