# ConvertLeicaQT — Desktop GUI for Leica LIF/XLEF/LOF

ConvertLeicaQT is a PyQt6 desktop application to browse Leica files, preview images quickly, and convert to OME‑TIFF using the same logic as the project’s CLI/server. It mirrors the web app’s behavior for progressive previews and disk caching, but runs locally as a native GUI.

## What it does

- Browse folders and Leica files (.lif, .xlef, .lof) with a left file tree.
- Inspect the structure of a selected Leica file in a right tree (folders/images), with lazy loading for nested XLEF structures.
- Show an image preview (center Z/T/tile) with progressive loading and disk caching.
- Display a compact, readable metadata summary and open the full JSON for folders or images.
- Convert the selected image to OME‑TIFF (or other formats depending on metadata) with a live log.
- Apply a size threshold rule during conversion (XY > threshold → OME‑TIFF; otherwise copy-through for LOF/XLEF).

## Key files and modules

- `ConvertLeicaQT.py` — This GUI.
- `CreatePreview.py` — Generates previews and manages the preview disk cache.
- `ci_leica_converters_helpers.py` — Reads Leica metadata (LIF/XLEF/LOF) and utilities.
- `leica_converter.py` — Conversion coordinator calling the concrete converters.
- `server.py` — Source of shared constants (preview steps, cache size) when available.
- `styles/darktheme.css` and `images/` — Styling and icons.

## UI layout

- Left: Folder/file tree (filters noisy items; when a folder has any .xlef, only .xlef files are shown in that folder).
- Right split:
  - Left: Leica file content tree (root → folders → images). Expands lazily; shows Folder JSON and Image JSON.
  - Right: Preview panel with a large image area, a short metadata summary, and a log.
- Bottom: Output folder picker, optional “Only convert LOF/XLEF with XY >” threshold, and a Convert button.

## Preview: progressive + cache (server parity)

- Steps: The GUI uses `server.PREVIEW_STEPS` when available (defaults to `[24, 112, 256]`).
- Cache dir: `%TEMP%/leica_preview_cache` (same as the server). Previews are cached as PNGs.
- Cache size: Uses `server.PREVIEW_CACHE_MAX` when available (default 500 files). Oldest items are evicted.
- Cache naming: `<UniqueID>_h<height>.png`. The code prefers `UniqueID`, then `uuid`, then `ImageUUID`. If none exist, it derives a stable-ish key from file path + dims.
- Generator: All previews go through `CreatePreview.create_preview_image(...)`, which:
  - Emits a cached file path if it already exists.
  - Otherwise generates via memory‑mapped reads when possible, adjusts contrast, writes to cache, and trims the cache.
- Small-image rule: If the image is ≤ 2048×2048, the GUI requests only the largest preview step.
- Skip smaller when largest cached: If the largest step image is already cached, the GUI skips requesting the smaller steps and immediately loads the largest.
- Diagnostics: The log prints “Preview [height]px: cache hit/miss” so you can verify cache usage.

## Image selection → which metadata is used?

- LIF: `get_image_metadata(folder_metadata, uuid)` returns the image block.
- XLEF: Merges the image node from the XLEF tree with the corresponding LOF‑like metadata (`get_image_metadata_LOF`). Preserves `save_child_name` if overwritten.
- LOF: Reads the LOF file’s metadata directly.

## Conversion flow

- Click “Convert selected image → OME‑TIFF”. Output goes to the chosen folder (defaults to `…/_c` near the file).
- The GUI spawns a worker that calls `leica_converter.convert_leica(...)` and streams `print()` output into the log.
- Rules (simplified):
  - LIF: If tilescan with `OverlapIsNegative`, create a single‑image `.lif`. Otherwise create OME‑TIFF (RGB vs multi‑channel handled by separate converters).
  - XLEF/LOF: If XY ≤ threshold (checkbox), return/copy original path; else convert to OME‑TIFF (RGB handled accordingly).
- The result dialog lists created paths; the log shows detailed progress.
### Robust file saving

The converter uses a **temp-first approach** for reliable saving:

1. TIFF is saved to the system temp directory first
2. File is copied to output folder with size verification
3. On failure, retries up to 10 times with progressive backoff (1 min, 2 min, ... 10 min)
4. Temp file is cleaned up only after all copies succeed

### Dual progress bars

With progress enabled, two distinct progress indicators appear in the log:

```
Converting to OME-TIFF: |██████████████████████████████████████████████████| 100.0% Processing complete
  Saving: <▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓> 100.0% - Copying to output
```

- **Processing bar** (`|███|`): Data reading, stitching, pyvips image creation
- **Saving bar** (`<▓▓▓>`): TIFF writing and file copying phases
## Configuration and defaults

- Initial root folder: If `server.ROOT_DIR` exists and is accessible, it’s used as the starting point. Otherwise the current working directory.
- Preview settings: Imported from `server.py` when present. Fallbacks inside the GUI:
  - `PREVIEW_STEPS`: `[24, 112, 256]`
  - `PREVIEW_CACHE_MAX`: `500`
- Style: `styles/darktheme.css` is applied if present; it references icons in `images/`.

## Performance notes

- Previews read only every Nth row (based on target height) and downscale via OpenCV; channels are colored by LUT names.
- Contrast is auto‑stretched using robust percentiles.
- Memory‑mapped slices avoid loading full volumes.
- Disk cache makes repeat previews near‑instant.

## Troubleshooting

- No cache hits: Ensure metadata includes a stable `UniqueID`/`uuid`/`ImageUUID`. If missing, the app derives a fallback key that may not be stable across different sources.
- Slow previews: Verify `opencv-python` and `numpy` are installed and hardware isn’t constrained. Large multi‑channel datasets will be slower on first render (cold cache).
- Empty trees: Some noise files/folders are filtered; ensure you double‑click a .lif/.xlef/.lof to populate the right content tree.
- Styling missing: Check `styles/darktheme.css` and that `images/` exists.- Network save failures: The converter retries up to 10 times with increasing delays. Check the log for retry messages. If failures persist, try using a local output folder.
- Temp folder issues: By default, the system temp directory is used. On constrained systems, ensure sufficient temp disk space for large TIFF files.
## Requirements

Install with:

```sh
pip install -r requirements-qt6ui.txt
```

Core runtime for previews and conversion:

- numpy
- opencv-python
- pyvips

GUI runtime:

- PyQt6 (included in requirements-qt6ui.txt)

## Running (example)

```bash
# Windows (cmd.exe) example
python ConvertLeicaQT.py
```

## FAQ

- Why do I sometimes see only one preview request?  
  If the image is ≤ 2048×2048 or the largest step is already cached, the app requests only the largest image.
- Where is the cache?  
  `%TEMP%/leica_preview_cache` (shared with the web server for parity).
- Can I clear the cache?  
  You can delete the cache folder. The app will recreate it on demand.
