import importlib
from pathlib import Path
import urllib.request

import ci_leica_converters_helpers


def test_helpers_import_does_not_access_network(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("module import attempted network access")

    monkeypatch.setattr(urllib.request, "urlretrieve", fail_if_called)
    monkeypatch.setattr(urllib.request, "urlopen", fail_if_called)
    helpers = importlib.reload(ci_leica_converters_helpers)

    assert Path(helpers.xsd_url).is_file()
    assert helpers.metadata_schema == helpers.parse_ome_xsd()
    assert helpers.validate_metadata("oil", "Immersion", helpers.metadata_schema) == "Oil"
    assert (
        helpers.validate_metadata(
            "laserScanningConfocalMicroscopy",
            "AcquisitionMode",
            helpers.metadata_schema,
        )
        == "LaserScanningConfocalMicroscopy"
    )
