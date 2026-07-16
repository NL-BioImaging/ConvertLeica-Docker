# OMERO Import Metadata Probe

This host-side tool generates deterministic tiled/pyramidal OME-TIFF and
multiscale OME-Zarr images, inspects their metadata, and can import them through
the `biomero-importer` container. Reports are written as JSON and Markdown.

Generate and inspect fixtures without OMERO:

```cmd
tools\omero_import_metadata_probe\run.cmd --generate
```

Import generated fixtures through both importer paths:

```cmd
tools\omero_import_metadata_probe\run.cmd --generate ^
  --importer-container biomero-importer ^
  --target Dataset:123 --user root --group system --mode both
```

Use `--input PATH` (repeatable) to inspect/import existing OME-TIFF or
OME-Zarr outputs. `--cleanup success` is the default and removes successfully
imported probe images after reporting. The report folder contains
`report.json`, `report.md`, generated fixtures, and captured container logs.

