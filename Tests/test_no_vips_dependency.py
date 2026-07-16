import sys
import unittest
from pathlib import Path


class NoVipsDependencyTests(unittest.TestCase):
    def test_runtime_and_packaging_are_vips_free(self):
        root = Path(__file__).resolve().parents[1]
        checked = [
            *root.glob("*.py"),
            *root.glob("requirements-*.txt"),
            root / "Dockerfile",
            root / "ConvertLeicaGUI.spec",
        ]
        forbidden = "py" + "vips"
        native = "lib" + "vips"
        failures = []
        for path in checked:
            text = path.read_text(encoding="utf-8", errors="replace").lower()
            if forbidden in text or native in text:
                failures.append(path.name)
        self.assertEqual(failures, [])

    def test_tiff_modules_import_without_vips_module(self):
        sys.modules.pop("py" + "vips", None)
        __import__("ci_leica_converters_ometiff")
        __import__("ci_leica_converters_ometiff_rgb")


if __name__ == "__main__":
    unittest.main()
