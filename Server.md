# ConvertLeica Web Server (server.py + index.html)

This document explains how the web server and single-page client work together, with a deep dive on the progressive preview generation and disk caching pipeline.

## Requirements

Install with:

```sh
pip install -r requirements-server.txt
```

The server uses the same core dependencies as the converter (numpy, pyvips, opencv-python). All HTTP/server functionality uses Python standard library modules only.

## Overview

- server.py: A small HTTP server exposing JSON API endpoints and serving static files (index.html, style.css, preview.png).
- index.html: A vanilla JS client that browses Leica folders/files, previews images, and streams convert progress.
- CreatePreview.py + helpers: Do the metadata parsing and preview rendering used by server-side APIs.

## Quick start

- Edit ROOT_DIR in server.py to your top-level data folder.
- Run the server (Python 3.8+):
  - On Windows: python server.py
- A browser tab opens at http://localhost:8000. Use the left folder tree and image list to navigate.

## Architecture

### API endpoints

- GET /api/config
  - Returns server configuration: rootDir, maxXYSize (MAX_XY_SIZE), previewSize (PREVIEW_SIZE), previewSteps (PREVIEW_STEPS), previewCacheMax (PREVIEW_CACHE_MAX).
- GET /api/list?dir=<path>[&folder_uuid=<uuid>]
  - Dir listing: folders and .lif/.xlef files. When dir points at a .lif/.xlef file, returns the metadata “children” (images/folders) instead.
  - Special filtering to hide non-image metadata like _environmentalgraph, .lifext, etc. If any .xlef exists in a directory, only .xlef are listed.
- POST /api/lof_metadata
  - Returns the JSON metadata for a .lof (or file-like) item.
- POST /api/preview
  - Body: { filePath, image_uuid, folder_metadata, preview_height }
  - Returns a base64 data URL (PNG) for the requested preview height and the image metadata used.
- POST /api/preview_status
  - Body: { filePath, image_uuid, folder_metadata }
  - Returns { maxCached, xs, ys } to inform the client about existing cached previews and native image size.
- POST /api/convert_leica (Server-Sent Events)
  - Streams progress lines and a final JSON result for the convert_leica call into an output folder (OUTPUT_SUBFOLDER).

### Static serving

- Any non-/api path is served via SimpleHTTPRequestHandler (e.g. index.html, style.css, preview.png).

## Data model & metadata

- LIF/XLEF: Trees with folders (nodes) and images. server.py uses read_leica_file(file) to get the root tree JSON; for images, get_image_metadata(folder_metadata, image_uuid).
- LOF: Flat file with image-like metadata derived by get_image_metadata_LOF().
- Metadata fields used in previews:
  - UniqueID: stable key used to derive cache filenames.
  - dimensions: { x, y, z, c, t, s }
  - xs/ys: optional direct pixel dimensions; when missing, fallback to dimensions.x/.y
  - save_child_name: used for display in the UI when present.

## Progressive previews and caching (deep dive)

### Terms

- PREVIEW_STEPS (server.py): e.g. [24, 112, 256]. Heights (pixels) at which PNG previews are rendered/cached.
- PREVIEW_SIZE (server.py): UI’s fixed preview box height (px). Also used as a single-shot height when PREVIEW_STEPS is empty.
- PREVIEW_CACHE_MAX (server.py): Max number of cached preview files the server keeps. Older/pre-existing items may be evicted inside CreatePreview code.
- Cache directory: %TEMP%/leica_preview_cache (Windows) – see get_cache_dir().
- Cache filename convention: {UniqueID}_h{height}.png, e.g. e5a9…_h256.png

### Client flow (index.html)

1) When user clicks an image, loadPreview(item) runs.
2) The client asks the server for status first:
   - POST /api/preview_status with { filePath, image_uuid, folder_metadata }.
   - The server parses metadata to get UniqueID and pixel dimensions (xs/ys), then inspects the cache directory for files named {uid}_h{h}.png for every h in PREVIEW_STEPS. The largest found is maxCached.
   - Response contains { maxCached, xs, ys }.
3) The client decides between single-shot or progressive:
   - SMALL_LIMIT = 2048. If xs <= 2048 and ys <= 2048, or if maxCached already equals the maximum configured step, it fetches just once at the max step height.
   - Else, it constructs a list of progressive steps strictly larger than maxCached (so it never re-requests already cached sizes) and downloads them in increasing order.
   - If PREVIEW_STEPS is empty, the client falls back to a single request at PREVIEW_SIZE.
4) Progressive loop (when used):
   - For each step h, POST /api/preview with { filePath, image_uuid, folder_metadata, preview_height: h }.
   - The server generates (or reuses) the cached {uid}_h{h}.png and returns a base64 data URL. The client updates the preview image immediately and waits for it to paint (ensureImagePaint) before requesting the next step. If the user clicks away, the loop aborts.
5) Metadata is updated as each /api/preview response arrives so the right-side panel stays in sync.

This approach yields a snappy UX:
- Small images show the full-resolution preview at once.
- Large images become progressively sharper using an ascending ladder of heights, reusing any cached levels.

### Server preview & cache flow (server.py)

- handle_preview:
  1) Compute image metadata:
     - .lof -> read_leica_file(filePath)
     - .xlef -> merge get_image_metadata(folder_metadata, image_uuid) + get_image_metadata_LOF(folder_metadata, image_uuid) to pick the right fields.
     - .lif -> get_image_metadata(folder_metadata, image_uuid)
  2) Parse the metadata JSON and extract UniqueID. If provided, build cache path {tmp}/leica_preview_cache/{UniqueID}_h{height}.png
  3) Call create_preview_image(image_metadata, cache_dir, preview_height=int(preview_height), use_memmap=True, max_cache_size=PREVIEW_CACHE_MAX)
     - This function renders and saves the PNG preview (if not present) and may prune older entries based on PREVIEW_CACHE_MAX.
  4) Read the PNG bytes, base64-encode, and return in { src: dataUrl, metadata: image_metadata, height, cached: bool }.

- handle_preview_status:
  - Recomputes metadata quickly, extracts UniqueID and xs/ys.
  - Scans PREVIEW_STEPS to compute maxCached by checking existence of {uid}_h{h}.png for each h. Returns { maxCached, xs, ys }.

### Cache hygiene

- Cache key uniqueness depends on metadata UniqueID; if not stable, multiple previews could collide or fail to reuse.
- To clear cache: exit the server and delete %TEMP%/leica_preview_cache.

## Robust file saving during conversion

The converter uses a **temp-first approach** when saving OME-TIFF files for reliability on network drives:

1. **Temp save**: The TIFF is written to the system temp directory first
2. **Verified copy**: The file is copied to the output folder with size verification
3. **Retry logic**: On copy failure, retries up to 10 times with progressive backoff (1 min, 2 min, ... 10 min)
4. **Cleanup**: Temp file is removed only after successful copy

This prevents corrupted files when the output folder is on an unreliable network mount.

### Progress reporting

With `--show_progress`, two progress bars are displayed during conversion:

- **Processing bar** (`|███|`): Reading data, stitching tiles, creating pyvips image (0-100%)
- **Saving bar** (`<▓▓▓>`): Writing TIFF to temp, copying to output (0-100%)

The SSE stream from `/api/convert_leica` includes both progress bars in the streamed output.

## Browsing, breadcrumbs, and folder_uuid

- /api/list behavior:
  - If dir is a directory: returns folders and .lif/.xlef files (with optional xlef-only filter if any .xlef exists there).
  - If dir is a file (.lif/.xlef): returns the JSON children of that file (virtual folders/images). Each child has a name, type, and carries path to the parent file. Some children (folders) can also have a uuid used to dive into nested groups.
- index.html maintains a breadcrumbHistory array of crumbs: { name, path, uuid, uniquePath }, where uniquePath concatenates path and uuid (path#uuid). Clicking a crumb calls loadDir(path, uuid, name) and fully re-renders the tree and list.

## Conversion endpoint (SSE)

- The /api/convert_leica endpoint streams Server-Sent Events:
  - type: "progress" lines mirror print() calls from convert_leica.
  - type: "result" contains the parsed JSON (converted files list) at the end.
  - type: "error" + a final type: "end" message.
- Output folder is {dirname(inputfile)}/_c (OUTPUT_SUBFOLDER), created automatically.

## Configuration knobs (server.py)

- ROOT_DIR: Top-level folder users see upon loading the page.
- OUTPUT_SUBFOLDER: Where converted files are placed.
- DEFAULT_PORT: HTTP port.
- MAX_XY_SIZE: Threshold to annotate tile scans (UI hints only).
- PREVIEW_SIZE: UI preview box height; also used as single-shot height when PREVIEW_STEPS is empty.
- PREVIEW_STEPS: Heights for progressive preview rendering and caching.
- PREVIEW_CACHE_MAX: Cache size cap (number of preview files).

Note: The client (index.html) trusts values from /api/config and does not hardcode preview steps. Adjust PREVIEW_STEPS and PREVIEW_SIZE only in server.py.

## Troubleshooting

- Previews don’t appear:
  - Check /api/preview responses in DevTools (Network tab). Look for errors parsing metadata or invalid filePath/image_uuid.
- Progressive steps don’t advance:
  - Ensure previewSteps are non-empty in /api/config. If empty, the client intentionally uses a single fetch at PREVIEW_SIZE.
- Breadcrumbs don’t navigate:
  - The anchors call loadDir(path, uuid, name). If a name contains quotes, ensure proper escaping or consider moving to addEventListener-based handlers.
- Cache not reused:
  - Confirm metadata UniqueID is present and stable; cache filenames depend on it.
- Out-of-date previews:
  - Clear %TEMP%/leica_preview_cache, restart the server.

## Extending

- Change progressive quality ladder: update PREVIEW_STEPS in server.py; the UI automatically adapts.
- Increase preview box size: adjust PREVIEW_SIZE. Consider adding an extra step equal to PREVIEW_SIZE.
- Different cache location: tweak get_cache_dir() to point elsewhere.
