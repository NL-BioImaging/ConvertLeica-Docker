#!/usr/bin/env python
"""Generate and probe ConvertLeica OME outputs against an OMERO importer."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import posixpath
import shutil
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cideconvolve_io import TiledOmeTiffSink, TiledRgbOmeTiffSink


def ome_xml(*, name: str, shape: tuple[int, int, int, int, int], dtype: str, rgb: bool = False) -> str:
    t, c, z, y, x = shape
    channels = (
        '<Channel ID="Channel:0:0" Name="RGB" SamplesPerPixel="3" '
        'IlluminationType="Transmitted" AcquisitionMode="WideField" ContrastMethod="Brightfield"/>'
        if rgb else
        '<Channel ID="Channel:0:0" Name="DAPI" SamplesPerPixel="1" Color="65535" '
        'ExcitationWavelength="405" ExcitationWavelengthUnit="nm" EmissionWavelength="460" EmissionWavelengthUnit="nm"/>'
        '<Channel ID="Channel:0:1" Name="GFP" SamplesPerPixel="1" Color="16711935" '
        'ExcitationWavelength="488" ExcitationWavelengthUnit="nm" EmissionWavelength="520" EmissionWavelengthUnit="nm"/>'
    )
    size_c = 3 if rgb else c
    mappings = []
    if rgb:
        mappings.append(f'<TiffData IFD="0" FirstZ="0" FirstC="0" FirstT="0" PlaneCount="{t*z}"/>')
    else:
        ifd = 0
        for ti in range(t):
            for ci in range(c):
                for zi in range(z):
                    mappings.append(f'<TiffData IFD="{ifd}" FirstZ="{zi}" FirstC="{ci}" FirstT="{ti}"/>')
                    ifd += 1
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
        '<Instrument ID="Instrument:0"><Microscope Manufacturer="Leica Microsystems" Model="Probe"/>'
        '<Objective ID="Objective:0" LensNA="1.4" NominalMagnification="63" Immersion="Oil"/></Instrument>'
        f'<Image ID="Image:0" Name="{name}"><AcquisitionDate>2026-01-02T03:04:05</AcquisitionDate>'
        '<InstrumentRef ID="Instrument:0"/><ObjectiveSettings ID="Objective:0"/>'
        f'<Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="{dtype}" SignificantBits="{8 if dtype == "uint8" else 16}" '
        f'SizeX="{x}" SizeY="{y}" SizeZ="{z}" SizeC="{size_c}" SizeT="{t}" '
        f'PhysicalSizeX="0.125" PhysicalSizeY="0.125" PhysicalSizeZ="0.5" '
        f'PhysicalSizeXUnit="µm" PhysicalSizeYUnit="µm" PhysicalSizeZUnit="µm" Interleaved="{str(rgb).lower()}">'
        f'{channels}{"".join(mappings)}</Pixels></Image></OME>'
    )


def generate_fixtures(folder: Path) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    shape = (2, 2, 3, 259, 275)
    yy, xx = np.indices(shape[-2:])
    data = np.empty(shape, dtype=np.uint16)
    for t in range(shape[0]):
        for c in range(shape[1]):
            for z in range(shape[2]):
                data[t, c, z] = t * 10000 + c * 3000 + z * 500 + yy * 2 + xx
    tiff_path = folder / "multichannel_tiled_pyramid.ome.tiff"
    sink = TiledOmeTiffSink(tiff_path, shape=shape, dtype=data.dtype,
                            ome_xml=ome_xml(name="Probe multichannel", shape=shape, dtype="uint16"),
                            tile_yx=(128, 128), levels=3, level0=data)
    sink.build_pyramids(); sink.close(); paths.append(tiff_path)

    rgb_shape = (2, 2, 259, 275, 3)
    rgb = np.empty(rgb_shape, dtype=np.uint8)
    for t in range(2):
        for z in range(2):
            rgb[t, z, ..., 0] = (xx + 20 * t) % 256
            rgb[t, z, ..., 1] = (yy + 30 * z) % 256
            rgb[t, z, ..., 2] = (xx + yy + 10 * t + 15 * z) % 256
    rgb_path = folder / "rgb_tiled_pyramid.ome.tiff"
    sink = TiledRgbOmeTiffSink(rgb_path, shape=rgb_shape, dtype=rgb.dtype,
                               ome_xml=ome_xml(name="Probe RGB", shape=(2, 1, 2, 259, 275), dtype="uint8", rgb=True),
                               tile_yx=(128, 128), levels=3, level0=rgb)
    sink.build_pyramids(); sink.close(); paths.append(rgb_path)

    try:
        import zarr
        zarr_path = folder / "multichannel_multiscale.ome.zarr"
        root = zarr.open_group(str(zarr_path), mode="w", zarr_format=2)
        levels = [data, data[..., ::2, ::2], data[..., ::4, ::4]]
        for index, level in enumerate(levels):
            root.create_array(str(index), data=level, chunks=(1, 1, 1, 128, 128), overwrite=True)
        root.attrs["multiscales"] = [{"version": "0.4", "name": "Probe multichannel", "axes": [
            {"name": "t", "type": "time"}, {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"}],
            "datasets": [{"path": str(i), "coordinateTransformations": [{"type": "scale", "scale": [1, 1, .5, .125 * 2**i, .125 * 2**i]}]} for i in range(3)]}]
        root.attrs["omero"] = {"name": "Probe multichannel", "channels": [
            {"label": "DAPI", "color": "0000FF", "active": True, "excitation_wavelength": 405.0, "emission_wavelength": 460.0,
             "window": {"min": 0, "max": 65535, "start": 0, "end": 12000}},
            {"label": "GFP", "color": "00FF00", "active": True, "excitation_wavelength": 488.0, "emission_wavelength": 520.0,
             "window": {"min": 0, "max": 65535, "start": 0, "end": 15000}}],
            "rdefs": {"defaultT": 0, "defaultZ": 1, "model": "color"}}
        ome_dir = zarr_path / "OME"; ome_dir.mkdir()
        (ome_dir / ".zgroup").write_text('{"zarr_format": 2}', encoding="ascii")
        (ome_dir / "METADATA.ome.xml").write_text(ome_xml(name="Probe multichannel", shape=shape, dtype="uint16"), encoding="utf-8")
        paths.append(zarr_path)
    except ImportError:
        pass
    return paths


def inspect(path: Path) -> dict[str, Any]:
    if path.is_dir():
        import zarr
        root = zarr.open_group(str(path), mode="r")
        levels = sorted(root.array_keys(), key=lambda value: int(value) if str(value).isdigit() else str(value))
        return {"path": str(path), "kind": "ome-zarr", "levels": levels,
                "arrays": {key: {"shape": list(root[key].shape), "dtype": str(root[key].dtype), "chunks": list(root[key].chunks)} for key in levels},
                "multiscales": dict(root.attrs).get("multiscales"), "omero": dict(root.attrs).get("omero")}
    import tifffile
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        root = ET.fromstring(tif.ome_metadata or "<OME/>")
        pixels = next((node for node in root.iter() if node.tag.endswith("Pixels")), None)
        channels = [dict(node.attrib) for node in root.iter() if node.tag.endswith("Channel")]
        return {"path": str(path), "kind": "ome-tiff", "is_ome": tif.is_ome, "is_bigtiff": tif.is_bigtiff,
                "pages": len(tif.pages), "shape": list(page.shape), "dtype": str(page.dtype),
                "tiled": page.is_tiled, "tile": [page.tilelength, page.tilewidth],
                "compression": str(page.compression), "predictor": str(page.predictor),
                "subifds": len(page.pages), "pixels": dict(pixels.attrib) if pixels is not None else {}, "channels": channels}


CONTAINER_CODE = r'''
import json, os, sys, uuid
from biomero_importer.utils.initialize import load_settings
from biomero_importer.utils.importer import DataPackageImporter
from omero.gateway import BlitzGateway

p=json.load(sys.stdin); settings=load_settings('/auto-importer/config/settings.yml')
package={'Group':p['group'],'Username':p['user'],'UUID':p['uuid'],'DestinationID':p['target_id'],'DestinationType':'Dataset','Files':[p['path']],'FileNames':[os.path.basename(p['path'].rstrip('/'))]}
imp=DataPackageImporter(settings, package)
if p['mode']=='biomero':
    successful, failed, import_failed=imp.import_data_package(); ids=[row[3] for row in successful if isinstance(row[3],int)]
    if import_failed or failed: raise RuntimeError('BIOMERO import failed: '+repr(failed))
else:
    if os.path.isdir(p['path']) and imp.use_register_zarr: ids,_=imp.import_zarr(uri=p['path'],target=p['target_id'])
    else: ids=imp.import_dataset(target=p['path'],dataset=p['target_id'],transfer=p.get('transfer','upload'))
root=BlitzGateway(os.environ['OMERO_USER'],os.environ['OMERO_PASSWORD'],host=os.environ['OMERO_HOST'],port=int(os.environ.get('OMERO_PORT','4064')),secure=True); root.connect()
import ezomero; conn=root.suConn(p['user'],ttl=600000); conn.setGroupForSession(ezomero.get_group_id(root,p['group']))
objects=[]
for iid in ids:
    image=conn.getObject('Image',int(iid)); px=image.getPrimaryPixels(); channels=[]
    for ch in image.getChannels():
        channels.append({'name':ch.getName(),'emission':str(ch.getEmissionWave()),'excitation':str(ch.getExcitationWave())})
    objects.append({'id':int(iid),'name':image.getName(),'size_x':px.getSizeX(),'size_y':px.getSizeY(),'size_z':px.getSizeZ(),'size_c':px.getSizeC(),'size_t':px.getSizeT(),'type':str(px.getPixelsType().getValue()),'physical_x':str(px.getPhysicalSizeX()),'physical_y':str(px.getPhysicalSizeY()),'physical_z':str(px.getPhysicalSizeZ()),'channels':channels})
if p.get('cleanup'):
    conn.deleteObjects('Image',[int(i) for i in ids],deleteAnns=True,deleteChildren=True,wait=True)
conn.close(); root.close(); print(json.dumps({'ids':ids,'objects':objects},default=str))
'''


def import_one(path: Path, args: argparse.Namespace, mode: str) -> dict[str, Any]:
    # /data is shared with the OMERO server, which is required when the
    # BIOMERO importer is configured for symlink/in-place transfer.
    stage = f"/data/.convertleica_probe_{uuid.uuid4().hex}/{path.name}"
    stage_parent = posixpath.dirname(stage)
    subprocess.run(["docker", "exec", args.importer_container, "mkdir", "-p", stage_parent], check=True)
    subprocess.run(["docker", "cp", str(path), f"{args.importer_container}:{stage}"], check=True)
    target_type, target_id = args.target.split(":", 1)
    if target_type.lower() != "dataset":
        raise ValueError("Generated image probes require a Dataset target")
    payload = {"path": stage, "mode": mode, "target_id": int(target_id), "user": args.user,
               "group": args.group, "uuid": f"convertleica-probe-{uuid.uuid4()}",
               "cleanup": args.cleanup != "never", "transfer": args.transfer}
    proc = subprocess.run(["docker", "exec", "-i", args.importer_container, "python", "-c", CONTAINER_CODE],
                          input=json.dumps(payload), text=True, capture_output=True, timeout=3600)
    cleanup = subprocess.run(
        ["docker", "exec", args.importer_container, "rm", "-rf", stage_parent],
        capture_output=True,
    )
    if cleanup.returncode:
        # Registered Zarr data can contain files owned by another container user.
        # This path is generated above and is therefore safe to remove as root.
        subprocess.run(
            ["docker", "exec", "-u", "0", args.importer_container,
             "rm", "-rf", stage_parent],
            capture_output=True,
        )
    if proc.returncode:
        return {"error": proc.stderr or proc.stdout, "stdout": proc.stdout}
    lines = [line for line in proc.stdout.splitlines() if line.lstrip().startswith("{")]
    return json.loads(lines[-1]) if lines else {"error": "Importer returned no JSON", "stdout": proc.stdout}


def compare(source: dict[str, Any], imported: dict[str, Any]) -> dict[str, Any]:
    if imported.get("error"):
        return {"Import": {"expected": "success", "actual": imported["error"], "status": "changed"}}
    objects = imported.get("objects") or []
    if not objects:
        return {"Image": {"expected": "one imported image", "actual": "none", "status": "missing"}}
    actual = objects[0]
    if source["kind"] == "ome-tiff":
        pixels = source.get("pixels") or {}
        expected = {
            "SizeX": int(pixels.get("SizeX", 0)), "SizeY": int(pixels.get("SizeY", 0)),
            "SizeZ": int(pixels.get("SizeZ", 0)), "SizeC": int(pixels.get("SizeC", 0)),
            "SizeT": int(pixels.get("SizeT", 0)), "Type": pixels.get("Type"),
            "ChannelNames": [channel.get("Name") for channel in source.get("channels", [])],
            "ChannelEmission": [channel.get("EmissionWavelength") for channel in source.get("channels", [])],
            "ChannelExcitation": [channel.get("ExcitationWavelength") for channel in source.get("channels", [])],
            "PhysicalSizeX": pixels.get("PhysicalSizeX"), "PhysicalSizeY": pixels.get("PhysicalSizeY"),
            "PhysicalSizeZ": pixels.get("PhysicalSizeZ"),
        }
    else:
        first = source["arrays"][source["levels"][0]]
        shape = first["shape"]
        zarr_channels = (source.get("omero") or {}).get("channels", [])
        scale = source["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"]
        expected = {"SizeX": shape[-1], "SizeY": shape[-2], "SizeZ": shape[-3],
                    "SizeC": shape[-4], "SizeT": shape[-5], "Type": first["dtype"],
                    "ChannelNames": [channel.get("label") for channel in zarr_channels],
                    "ChannelEmission": [channel.get("emission_wavelength") for channel in zarr_channels],
                    "ChannelExcitation": [channel.get("excitation_wavelength") for channel in zarr_channels],
                    "PhysicalSizeX": scale[-1], "PhysicalSizeY": scale[-2], "PhysicalSizeZ": scale[-3]}
    observed = {"SizeX": actual.get("size_x"), "SizeY": actual.get("size_y"),
                "SizeZ": actual.get("size_z"), "SizeC": actual.get("size_c"),
                "SizeT": actual.get("size_t"), "Type": actual.get("type"),
                "ChannelNames": [channel.get("name") for channel in actual.get("channels", [])],
                "ChannelEmission": [channel.get("emission") for channel in actual.get("channels", [])],
                "ChannelExcitation": [channel.get("excitation") for channel in actual.get("channels", [])],
                "PhysicalSizeX": actual.get("physical_x"), "PhysicalSizeY": actual.get("physical_y"),
                "PhysicalSizeZ": actual.get("physical_z")}
    result = {}
    for key, value in expected.items():
        got = observed.get(key)
        if key == "ChannelNames" and len(value) == 1 and isinstance(got, list) and got and all(item == got[0] for item in got):
            matched = value[0] == got[0]
        elif key.startswith("PhysicalSize"):
            try:
                matched = abs(float(value) - float(str(got).split()[0])) < 1e-9
            except (TypeError, ValueError):
                matched = value is None and got in (None, "None")
        elif key in {"ChannelEmission", "ChannelExcitation"}:
            def numeric_list(values):
                return [None if item in (None, "None") else float(str(item).split()[0]) for item in values]
            try:
                expected_values, actual_values = numeric_list(value), numeric_list(got)
                if len(expected_values) == 1 and actual_values and all(item == actual_values[0] for item in actual_values):
                    matched = expected_values[0] == actual_values[0]
                else:
                    matched = expected_values == actual_values
            except (TypeError, ValueError):
                matched = False
        else:
            matched = str(value).lower() == str(got).lower()
        result[key] = {"expected": value, "actual": got, "status": "matched" if matched else "changed"}
    return result


def markdown(report: dict[str, Any]) -> str:
    lines = ["# ConvertLeica OMERO Probe Report", "", f"Created: `{report['created']}`", "", "## Inputs", ""]
    for item in report["inputs"]:
        lines.append(f"- `{item['path']}`: {item['kind']}, dtype `{item.get('dtype', '')}`, levels/SubIFDs `{item.get('subifds', len(item.get('levels', [])))}`")
    lines += ["", "## Imports", ""]
    for path, modes in report.get("imports", {}).items():
        lines.append(f"### {Path(path).name}"); lines.append("")
        for mode, result in modes.items():
            lines.append(f"- {mode}: {'FAILED' if result.get('error') else 'OK'} `{result.get('ids', [])}`")
            comparison = report.get("comparisons", {}).get(path, {}).get(mode, {})
            changed = [key for key, row in comparison.items() if row.get("status") != "matched"]
            lines.append(f"  Metadata: {'changed: ' + ', '.join(changed) if changed else 'matched'}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true", help="Generate deterministic TIFF and Zarr fixtures.")
    parser.add_argument("--input", action="append", default=[], help="Existing OME-TIFF or OME-Zarr path; repeatable.")
    parser.add_argument("--out", help="Report directory.")
    parser.add_argument("--importer-container", help="Running biomero-importer container; omit for inspection only.")
    parser.add_argument("--target", help="OMERO Dataset target, e.g. Dataset:123.")
    parser.add_argument("--user"); parser.add_argument("--group")
    parser.add_argument("--mode", choices=["direct", "biomero", "both"], default="both")
    parser.add_argument("--cleanup", choices=["success", "always", "never"], default="success")
    parser.add_argument("--transfer", choices=["upload", "ln_s", "ln", "cp", "ln_rm"], default="upload")
    args = parser.parse_args(argv)
    if not args.generate and not args.input:
        parser.error("Use --generate and/or --input")
    out = Path(args.out) if args.out else Path(__file__).parent / "reports" / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    paths = generate_fixtures(out / "fixtures") if args.generate else []
    paths += [Path(value).resolve() for value in args.input]
    report: dict[str, Any] = {"created": dt.datetime.now().isoformat(), "report_dir": str(out), "inputs": [inspect(path) for path in paths], "imports": {}, "comparisons": {}}
    if args.importer_container:
        if not args.target or not args.user or not args.group:
            parser.error("--target, --user, and --group are required for import")
        modes = ["direct", "biomero"] if args.mode == "both" else [args.mode]
        for path in paths:
            report["imports"][str(path)] = {mode: import_one(path, args, mode) for mode in modes}
            source = next(item for item in report["inputs"] if item["path"] == str(path))
            report["comparisons"][str(path)] = {
                mode: compare(source, result) for mode, result in report["imports"][str(path)].items()
            }
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out / "report.md").write_text(markdown(report), encoding="utf-8")
    print(out / "report.md")
    import_failed = any(result.get("error") for modes in report["imports"].values() for result in modes.values())
    metadata_changed = any(row.get("status") != "matched" for paths in report["comparisons"].values() for modes in paths.values() for row in modes.values())
    return 2 if import_failed or metadata_changed else 0


if __name__ == "__main__":
    raise SystemExit(main())
