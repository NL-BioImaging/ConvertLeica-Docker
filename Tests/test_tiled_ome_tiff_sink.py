import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cideconvolve_io import TiledOmeTiffSink, TiledRgbOmeTiffSink
from tools.omero_import_metadata_probe.omero_import_metadata_probe import ome_xml


class TiledOmeTiffSinkTests(unittest.TestCase):
    def test_integer_multichannel_tiff_is_tiled_pyramidal_and_metadata_roundtrips(self):
        shape = (2, 2, 2, 35, 37)
        data = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "multi.ome.tiff"
            sink = TiledOmeTiffSink(
                path, shape=shape, dtype=data.dtype,
                ome_xml=ome_xml(name="Unit multichannel", shape=shape, dtype="uint16"),
                tile_yx=(32, 32), levels=2, level0=data,
            )
            sink.build_pyramids()
            sink.close()
            with tifffile.TiffFile(path) as tif:
                self.assertTrue(tif.is_ome)
                self.assertTrue(tif.is_bigtiff)
                self.assertTrue(tif.pages[0].is_tiled)
                self.assertEqual(tif.pages[0].dtype, np.dtype("uint16"))
                self.assertEqual(len(tif.pages[0].pages), 1)
                self.assertEqual(len(tif.series), 1)
                self.assertEqual(len(tif.series[0].levels), 2)
                self.assertEqual(tif.ome_metadata.count("<Image"), 1)
                pixels = next(node for node in ET.fromstring(tif.ome_metadata).iter() if node.tag.endswith("Pixels"))
                self.assertEqual(pixels.attrib["PhysicalSizeX"], "0.125")
                self.assertEqual(pixels.attrib["PhysicalSizeXUnit"], "µm")
                np.testing.assert_array_equal(tif.asarray(), data)

    def test_true_rgb_retains_interleaved_samples_and_odd_edges(self):
        shape = (1, 2, 35, 37, 3)
        data = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
        xml = ome_xml(name="Unit RGB", shape=(1, 1, 2, 35, 37), dtype="uint8", rgb=True)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rgb.ome.tiff"
            sink = TiledRgbOmeTiffSink(path, shape=shape, dtype=data.dtype, ome_xml=xml,
                                       tile_yx=(32, 32), levels=2, level0=data)
            sink.build_pyramids()
            sink.close()
            with tifffile.TiffFile(path) as tif:
                page = tif.pages[0]
                self.assertEqual(page.samplesperpixel, 3)
                self.assertEqual(int(page.photometric), 2)
                self.assertEqual(len(page.pages), 1)
                self.assertEqual(len(tif.series), 1)
                self.assertEqual(len(tif.series[0].levels), 2)
                self.assertEqual(tif.ome_metadata.count("<Image"), 1)
                self.assertEqual(tuple(page.pages[0].shape), (18, 19, 3))
                np.testing.assert_array_equal(tif.asarray(), data[0])


if __name__ == "__main__":
    unittest.main()
