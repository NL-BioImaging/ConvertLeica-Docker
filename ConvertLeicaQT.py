"""PyQt6 GUI to browse Leica files (LIF/XLEF/LOF), preview images, and convert to OME-TIFF.

Features:
- Folder browser with filtering similar to the web server.
- Lists images within selected .lif/.xlef/.lof using existing helpers.
- Preview of the selected image (center Z/T/tile) leveraging CreatePreview.
- Convert selected image via convert_leica with live progress messages.
- Dark theme and basic styling cues inspired by MultiRepAnalysisQT.
- Help dialog that loads ConvertLeicaQTHelp.html.
"""
from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from dataclasses import dataclass

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QListWidget, QListWidgetItem, QLabel, QFileDialog, QTextEdit, QSplitter,
    QSizePolicy, QLineEdit, QMessageBox, QDialog, QDialogButtonBox, QTextBrowser,
    QTreeWidget, QTreeWidgetItem, QStyle, QCheckBox, QSpinBox, QProgressBar
)
from PyQt6.QtGui import QIcon, QPixmap, QPalette, QColor
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import traceback
from datetime import datetime

# Internal helpers from the repo
from ci_leica_converters_helpers import read_leica_file, get_image_metadata, get_image_metadata_LOF
from CreatePreview import create_preview_image
from leica_converter import convert_leica
import tempfile
import re


# ----------------------------- Progress Parsing Utilities -----------------------------
def parse_progress_text(text: str) -> dict | None:
    """
    Parse progress bar text to extract percentage and phase information.
    
    Returns dict with keys: 'percent', 'phase', 'suffix' or None if not a progress line.
    
    Handles formats like:
    - 'Converting to OME-TIFF: |████----| 45.0% Finished T=5/10 C=1/1 Z=1/1'
    - 'Copying source: {▒▒░░} 25.0% - 100/400 MB'
    - 'Saving: <▓▓░░> 50.0% - Writing TIFF'
    """
    if not text:
        return None
    
    # Match percentage patterns: XX.X% or XX%
    percent_match = re.search(r'(\d+(?:\.\d+)?)%', text)
    if not percent_match:
        return None
    
    try:
        percent = float(percent_match.group(1))
    except ValueError:
        return None
    
    # Determine phase from prefix (before the progress bar)
    phase = 'Converting'
    text_lower = text.lower()
    if 'copying source' in text_lower or 'copying output' in text_lower:
        phase = 'Copying'
    elif 'saving' in text_lower:
        phase = 'Saving'
    elif 'converting' in text_lower:
        if 'rgb' in text_lower:
            phase = 'Converting RGB'
        else:
            phase = 'Converting'
    elif 'creating single lif' in text_lower:
        phase = 'Creating LIF'
    
    # Extract suffix (text after percentage)
    suffix = ''
    suffix_match = re.search(r'\d+(?:\.\d+)?%\s*[-–]?\s*(.+)', text)
    if suffix_match:
        suffix = suffix_match.group(1).strip()
    
    return {'percent': percent, 'phase': phase, 'suffix': suffix}


# ----------------------------- Theme (inspired by MultiRepAnalysisQT) -----------------------------
def apply_dark_theme(app: QApplication) -> None:
    app.setStyle('Fusion')
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(32, 32, 32))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Base, QColor(45, 45, 45))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(60, 60, 60))
    palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.ColorRole.Button, QColor(50, 50, 50))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(127, 127, 127))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(127, 127, 127))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(127, 127, 127))
    app.setPalette(palette)


# ----------------------------- Worker to run conversion with progress -----------------------------
class StdoutSignalEmitter:
    """A lightweight stdout replacement to route print() to a Qt signal."""
    def __init__(self, signal, parsed_signal=None):
        self.signal = signal
        self.parsed_signal = parsed_signal
        self._buffer = ""

    def write(self, s: str):
        self._buffer += s
        # Handle carriage return (in-place updates) - emit the last line segment
        while '\r' in self._buffer or '\n' in self._buffer:
            # Find the first occurrence of either
            cr_pos = self._buffer.find('\r')
            nl_pos = self._buffer.find('\n')
            
            if nl_pos >= 0 and (cr_pos < 0 or nl_pos < cr_pos):
                # Newline first - emit the complete line
                line, self._buffer = self._buffer.split('\n', 1)
                self._emit_line(line.strip())
            elif cr_pos >= 0:
                # Carriage return - emit and continue (in-place update)
                line, self._buffer = self._buffer.split('\r', 1)
                self._emit_line(line.strip())
            else:
                break
    
    def _emit_line(self, line: str):
        if not line:
            return
        self.signal.emit(line)
        # Also emit parsed progress if available
        if self.parsed_signal:
            parsed = parse_progress_text(line)
            if parsed:
                self.parsed_signal.emit(
                    int(parsed['percent']),
                    parsed['phase'],
                    parsed['suffix']
                )

    def flush(self):
        if self._buffer.strip():
            self._emit_line(self._buffer.strip())
        self._buffer = ""


class ConvertWorker(QThread):
    progress = pyqtSignal(str)
    progressParsed = pyqtSignal(int, str, str)  # percent (0-100), phase, suffix
    finished = pyqtSignal(bool, object)  # success, result(list/dict/None)

    def __init__(self, inputfile: str, image_uuid: str, outputfolder: str, xy_check_value: int = 3192):
        super().__init__()
        self.inputfile = inputfile
        self.image_uuid = image_uuid
        self.outputfolder = outputfolder
        self.xy_check_value = int(xy_check_value)

    def run(self):  # noqa: D401
        """Run convert_leica and emit progress lines captured from print()."""
        orig_stdout = sys.stdout
        sys.stdout = StdoutSignalEmitter(self.progress, self.progressParsed)
        try:
            result_json = convert_leica(
                inputfile=self.inputfile,
                image_uuid=self.image_uuid,
                outputfolder=self.outputfolder,
                show_progress=True,
                xy_check_value=self.xy_check_value,
            )
            try:
                result = json.loads(result_json)
            except Exception:
                result = []
            self.progress.emit("Conversion finished.")
            self.finished.emit(bool(result), result)
        except Exception as e:  # noqa: BLE001
            self.progress.emit(f"Error: {e}")
            self.finished.emit(False, None)
        finally:
            sys.stdout = orig_stdout


# ----------------------------- Progressive Preview Worker -----------------------------
class PreviewWorker(QThread):
    """Generate progressively larger previews for an image metadata dict.

    Emits previewReady(job_id, height, path) for each generated PNG file path.
    The main thread must load the pixmap and delete the temp file.
    """
    previewReady = pyqtSignal(int, int, str)  # job_id, height, cached_png_path
    error = pyqtSignal(int, str)              # job_id, message
    cacheInfo = pyqtSignal(int, int, bool)    # job_id, height, cached_before

    def __init__(self, job_id: int, meta: dict, heights: list[int], cache_dir: str, max_cache_size: int,
                 use_memmap: bool = True, pause_ms: int = 120):
        super().__init__()
        self.job_id = int(job_id)
        self.meta = meta
        self.heights = list(heights)
        self.cache_dir = cache_dir
        self.max_cache_size = int(max_cache_size)
        self.use_memmap = bool(use_memmap)
        self.pause_ms = int(pause_ms)

    def run(self) -> None:  # noqa: D401
        try:
            for h in self.heights:
                if self.isInterruptionRequested():
                    break
                # Generate this step
                try:
                    # Check if cached file exists before triggering generation
                    cached_before = False
                    uid = (
                        self.meta.get("UniqueID")
                        or self.meta.get("uuid")
                        or self.meta.get("ImageUUID")
                    )
                    cache_path = None
                    if uid:
                        cache_path = os.path.join(self.cache_dir, f"{uid}_h{int(h)}.png")
                        cached_before = os.path.exists(cache_path)
                    self.cacheInfo.emit(self.job_id, int(h), bool(cached_before))

                    cached_png = create_preview_image(
                        self.meta,
                        self.cache_dir,
                        preview_height=int(h),
                        use_memmap=self.use_memmap,
                        max_cache_size=self.max_cache_size,
                    )
                except Exception as e:  # noqa: BLE001
                    # Send full traceback with file and line numbers
                    try:
                        tb = traceback.format_exc()
                    except Exception:
                        tb = f"{type(e).__name__}: {e}"
                    self.error.emit(self.job_id, tb)
                    break
                # Deliver to UI
                self.previewReady.emit(self.job_id, int(h), cached_png)
                # Brief pause to allow UI to react and for possible cancellation before next step
                if self.pause_ms > 0:
                    QThread.msleep(self.pause_ms)
        except Exception as e:  # noqa: BLE001
            try:
                tb = traceback.format_exc()
            except Exception:
                tb = f"{type(e).__name__}: {e}"
            self.error.emit(self.job_id, tb)


# ----------------------------- Data types -----------------------------
@dataclass
class ImageItem:
    name: str
    uuid: str
    meta: dict


# ----------------------------- Main Window -----------------------------
class ConvertLeicaApp(QMainWindow):
    VERSION = "1.0.0"

    # Try importing server constants for parity; provide sane fallbacks
    try:
        from server import PREVIEW_STEPS as _SERVER_PREVIEW_STEPS  # type: ignore
    except Exception:
        _SERVER_PREVIEW_STEPS = [24, 112, 256]
    try:
        from server import PREVIEW_CACHE_MAX as _SERVER_PREVIEW_CACHE_MAX  # type: ignore
    except Exception:
        _SERVER_PREVIEW_CACHE_MAX = 500

    @staticmethod
    def get_cache_dir() -> str:
        d = os.path.join(tempfile.gettempdir(), "leica_preview_cache")
        os.makedirs(d, exist_ok=True)
        return d

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"Convert Leica to OME-TIFF v{self.VERSION}")
        # Optional icon if available
        icon_path = Path(__file__).with_name('images').joinpath('app-icon.png')
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        self.resize(1200, 800)

        self.current_dir = self._default_root()
        self.folder_metadata_json: str | None = None  # Cache of current file's folder metadata (JSON string)
        self.current_file: str | None = None
        self.selected_image: ImageItem | None = None
        # Progressive preview state
        self._preview_job_id: int = 0
        self._preview_worker: PreviewWorker | None = None

        central = QWidget(self)
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)

        # Top bar: root path, browse, help
        top = QHBoxLayout()
        self.lbl_root = QLabel(f"Root: {self.current_dir}")
        self.btn_browse_root = QPushButton("Browse…")
        self.btn_browse_root.setFixedWidth(100)
        self.btn_browse_root.clicked.connect(self.choose_root)
        self.btn_help = QPushButton("Help")
        self.btn_help.setFixedWidth(90)
        self.btn_help.clicked.connect(self.show_help)
        top.addWidget(self.btn_browse_root)
        top.addWidget(self.lbl_root)
        top.addStretch(1)
        top.addWidget(self.btn_help)
        outer.addLayout(top)

        # Splitter: left (filesystem tree) | right (content tree + preview)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget(); left_layout = QVBoxLayout(left); left_layout.setContentsMargins(0, 0, 0, 0)
        # Keep a reference so we can align its height with the header buttons on the right
        self.lbl_folders = QLabel("Folders and Leica files:")
        left_layout.addWidget(self.lbl_folders)
        self.tree_fs = QTreeWidget(); self.tree_fs.setHeaderHidden(True)
        self.tree_fs.itemExpanded.connect(self.on_fs_item_expanded)
        self.tree_fs.itemDoubleClicked.connect(self.on_fs_item_double_clicked)
        self.tree_fs.itemSelectionChanged.connect(self.on_fs_selection_changed)
        left_layout.addWidget(self.tree_fs, 1)
        splitter.addWidget(left)

        # Right side split: left = content tree, right = preview + controls
        right_split = QSplitter(Qt.Orientation.Horizontal)

        right_left = QWidget(); right_left_layout = QVBoxLayout(right_left); right_left_layout.setContentsMargins(0, 0, 0, 0)
        # Header row with label and JSON buttons
        header_row = QHBoxLayout()
        # Keep a reference to the label so we can align button heights to it
        self.lbl_contents = QLabel("Contents of selected Leica file:")
        header_row.addWidget(self.lbl_contents)
        header_row.addStretch(1)
        self.btn_show_folder_json = QPushButton("Folder JSON")
        self.btn_show_folder_json.setToolTip("Show the JSON for the selected folder (or file root)")
        self.btn_show_folder_json.clicked.connect(self.show_folder_json)
        self.btn_show_folder_json.setEnabled(False)
        self.btn_show_image_json = QPushButton("Image JSON")
        self.btn_show_image_json.setToolTip("Show the JSON metadata for the selected image")
        self.btn_show_image_json.clicked.connect(self.show_image_json)
        self.btn_show_image_json.setEnabled(False)
        # Align label heights to the typical button height for a clean, readable header
        try:
            _h_btn = max(self.btn_show_folder_json.sizeHint().height(), self.btn_show_image_json.sizeHint().height())
            if _h_btn and _h_btn > 0:
                self.lbl_contents.setFixedHeight(_h_btn)
                # Also align the left-pane section label
                if hasattr(self, 'lbl_folders') and self.lbl_folders is not None:
                    self.lbl_folders.setFixedHeight(_h_btn)
        except Exception:
            pass
        header_row.addWidget(self.btn_show_folder_json)
        header_row.addWidget(self.btn_show_image_json)
        right_left_layout.addLayout(header_row)
        self.tree_images = QTreeWidget(); self.tree_images.setHeaderHidden(True)
        self.tree_images.itemSelectionChanged.connect(self.on_image_selection_changed)
        self.tree_images.itemExpanded.connect(self.on_tree_item_expanded)
        right_left_layout.addWidget(self.tree_images, 1)
        right_split.addWidget(right_left)

        right_right = QWidget(); right_right_layout = QVBoxLayout(right_right); right_right_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_label = QLabel("Preview will appear here")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview_label.setMinimumSize(360, 300)
        right_right_layout.addWidget(self.preview_label, 1)
        # Metadata summary under preview
        self.meta_text = QTextEdit(); self.meta_text.setReadOnly(True)
        self.meta_text.setPlaceholderText("Image metadata will appear here")
        self.meta_text.setMaximumHeight(140)
        right_right_layout.addWidget(self.meta_text)

    # Progress bar section
        progress_widget = QWidget()
        progress_layout = QVBoxLayout(progress_widget)
        progress_layout.setContentsMargins(0, 4, 0, 4)
        progress_layout.setSpacing(2)
        
        # Phase label and progress bar in a row
        progress_row = QHBoxLayout()
        self.lbl_progress_phase = QLabel("")
        self.lbl_progress_phase.setMinimumWidth(120)
        progress_row.addWidget(self.lbl_progress_phase)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setMinimumHeight(20)
        progress_row.addWidget(self.progress_bar, 1)
        
        progress_layout.addLayout(progress_row)
        
        # Status/suffix label
        self.lbl_progress_suffix = QLabel("")
        self.lbl_progress_suffix.setStyleSheet("color: #888; font-size: 11px;")
        progress_layout.addWidget(self.lbl_progress_suffix)
        
        progress_widget.setVisible(False)  # Hidden by default
        self.progress_widget = progress_widget
        right_right_layout.addWidget(progress_widget)

    # Log output
        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setMinimumHeight(120)
        right_right_layout.addWidget(self.log)

        right_split.addWidget(right_right)
        right_split.setStretchFactor(0, 3)
        right_split.setStretchFactor(1, 4)

        splitter.addWidget(right_split)
        # Make the folder/file tree start ~2x wider than the right side
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        outer.addWidget(splitter, 1)

        # Bottom pane: output folder + convert button
        bottom = QWidget(); bottom_layout = QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(6, 6, 6, 6)
        bottom_layout.addWidget(QLabel("Output folder:"))
        self.btn_out = QPushButton("Browse…")
        self.btn_out.clicked.connect(self.choose_output)
        bottom_layout.addWidget(self.btn_out)
        self.edit_out = QLineEdit()
        self.edit_out.setPlaceholderText("Choose output folder…")
        bottom_layout.addWidget(self.edit_out, 1)
        # XY threshold controls
        self.chk_large_only = QCheckBox("Only convert LOF files with XY >")
        self.chk_large_only.setChecked(False)
        self.chk_large_only.setToolTip("When checked, only convert LOF/XLEF images with XY dimensions greater than the threshold; otherwise, copy through.")
        bottom_layout.addWidget(self.chk_large_only)
        self.spin_xy_threshold = QSpinBox()
        self.spin_xy_threshold.setRange(1, 4096)
        self.spin_xy_threshold.setValue(3192)
        self.spin_xy_threshold.setToolTip("XY dimension threshold (pixels)")
        bottom_layout.addWidget(self.spin_xy_threshold)
        self.btn_convert = QPushButton("Convert selected image → OME-TIFF")
        self.btn_convert.clicked.connect(self.convert_selected)
        self.btn_convert.setEnabled(False)
        bottom_layout.addWidget(self.btn_convert)
        outer.addWidget(bottom, 0)

    # Initial load
        self.refresh_dir()
        # Stretch factors above set initial proportions; no pixel sizes needed here

    # (Styling is applied in main() via application stylesheet.)

    # ----------------------------- Root helpers -----------------------------
    def _default_root(self) -> str:
        # Try to reuse server.py ROOT_DIR; fall back to cwd
        try:
            from server import ROOT_DIR as SERVER_ROOT  # type: ignore
            if SERVER_ROOT and os.path.exists(SERVER_ROOT):
                return os.path.normpath(SERVER_ROOT)
        except Exception:
            pass
        return os.getcwd()

    def choose_root(self):
        d = QFileDialog.getExistingDirectory(self, "Choose root folder", self.current_dir)
        if d:
            self.current_dir = os.path.normpath(d)
            self.lbl_root.setText(f"Root: {self.current_dir}")
            self.refresh_dir()

    def refresh_dir(self):
        # repopulate filesystem tree to new root
        self.populate_fs_root()
        # Reset right side when changing dir
        self.tree_images.clear(); self._clear_preview(); self._cancel_preview_worker()
        self.folder_metadata_json = None
        self.current_file = None
        self.selected_image = None
        self.btn_convert.setEnabled(False)
        self.btn_show_folder_json.setEnabled(False)
        self.btn_show_image_json.setEnabled(False)
        self.meta_text.clear()
        # Default output folder for this directory
        self.edit_out.setText(os.path.join(self.current_dir, "_c"))

    # ----------------------------- Filesystem tree helpers -----------------------------
    def populate_fs_root(self):
        self.tree_fs.clear()
        root_item = QTreeWidgetItem([self.current_dir])
        root_item.setIcon(0, self.icon_folder())
        root_item.setData(0, Qt.ItemDataRole.UserRole, self.current_dir)
        # add placeholder and expand lazily
        root_item.addChild(QTreeWidgetItem(["…"]))
        self.tree_fs.addTopLevelItem(root_item)
        # Expand root; on_fs_item_expanded will populate once due to placeholder check
        self.tree_fs.expandItem(root_item)

    def on_fs_item_expanded(self, item: QTreeWidgetItem):
        # Populate only once per node
        if item.childCount() == 1 and item.child(0).text(0) == "…":
            self._populate_fs_children(item)

    def _populate_fs_children(self, parent_item: QTreeWidgetItem):
        # remove placeholder
        if parent_item.childCount() == 1 and parent_item.child(0).text(0) == "…":
            parent_item.removeChild(parent_item.child(0))
        parent_path = parent_item.data(0, Qt.ItemDataRole.UserRole)
        if not parent_path or not os.path.isdir(parent_path):
            return
        try:
            entries = sorted(os.listdir(parent_path))
        except Exception:
            return
    # If any .xlef exists, prefer microscopy files: allow .xlef, hide others
        has_xlef = any(os.path.splitext(n)[1].lower() == ".xlef" for n in entries)
        for name in entries:
            low = name.lower()
            if ("metadata" in low or "_pmd_" in low or "_histo" in low or
                "_environmetalgraph" in low or low.endswith(".lifext") or
                low in ("iomanagerconfiguation", "iomanagerconfiguration")):
                continue
            full = os.path.join(parent_path, name)
            ext = os.path.splitext(name)[1].lower()
            if os.path.isdir(full):
                item = QTreeWidgetItem([name])
                item.setIcon(0, self.icon_folder())
                item.setData(0, Qt.ItemDataRole.UserRole, full)
                item.addChild(QTreeWidgetItem(["…"]))
                parent_item.addChild(item)
            else:
                if has_xlef and ext not in (".xlef",):
                    continue
            if ext in (".lif", ".xlef", ".lof"):
                item = QTreeWidgetItem([name])
                item.setIcon(0, self.icon_for_file(ext))
                item.setData(0, Qt.ItemDataRole.UserRole, full)
                parent_item.addChild(item)

    def on_fs_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        path = item.data(0, Qt.ItemDataRole.UserRole)
        if not path:
            return
        if os.path.isdir(path):
            self.current_dir = os.path.normpath(path)
            self.lbl_root.setText(f"Root: {self.current_dir}")
            self.refresh_dir()
        else:
            self.load_file_images(path)

    def on_fs_selection_changed(self):
        items = self.tree_fs.selectedItems()
        if not items:
            return
        path = items[0].data(0, Qt.ItemDataRole.UserRole)
        if os.path.isfile(path):
            self.load_file_images(path)

    def load_file_images(self, filepath: str):
        self.current_file = filepath
        self.tree_images.clear(); self._clear_preview(); self._cancel_preview_worker()
        self.folder_metadata_json = None
        ext = os.path.splitext(filepath)[1].lower()
        try:
            if ext in (".lif", ".xlef"):
                meta_json = read_leica_file(filepath)  # returns tree JSON string
                self.folder_metadata_json = meta_json
                root_item = QTreeWidgetItem([os.path.basename(filepath)])
                root_item.setData(0, Qt.ItemDataRole.UserRole + 1, "root")
                root_item.setData(0, Qt.ItemDataRole.UserRole + 2, filepath)
                root_item.setData(0, Qt.ItemDataRole.UserRole + 4, meta_json)
                root_item.setIcon(0, self.icon_for_file(ext))
                self.tree_images.addTopLevelItem(root_item)
                self.populate_children(root_item, meta_json)
                self.tree_images.expandItem(root_item)
            elif ext == ".lof":
                root_item = QTreeWidgetItem([os.path.basename(filepath)])
                root_item.setData(0, Qt.ItemDataRole.UserRole + 1, "root")
                root_item.setData(0, Qt.ItemDataRole.UserRole + 2, filepath)
                root_item.setIcon(0, self.icon_for_file(ext))
                # Attach root JSON so Folder JSON button works
                try:
                    lof_root_json = read_leica_file(filepath)
                    root_item.setData(0, Qt.ItemDataRole.UserRole + 4, lof_root_json)
                    self.folder_metadata_json = lof_root_json
                except Exception:
                    pass
                img_item = QTreeWidgetItem([os.path.basename(filepath)])
                img_item.setData(0, Qt.ItemDataRole.UserRole + 1, "image")
                img_item.setData(0, Qt.ItemDataRole.UserRole + 2, filepath)
                img_item.setData(0, Qt.ItemDataRole.UserRole + 3, "__LOF__")
                img_item.setIcon(0, self.icon_image())
                root_item.addChild(img_item)
                self.tree_images.addTopLevelItem(root_item)
                self.tree_images.expandItem(root_item)
            else:
                QMessageBox.information(self, "Unsupported", f"Unsupported file type: {ext}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read file metadata:\n{e}")

        # Default output folder near this file
        outdir = os.path.join(os.path.dirname(filepath), "_c")
        self.edit_out.setText(outdir)
        # Enable folder JSON only for Leica types
        self.btn_show_folder_json.setEnabled(ext in (".lif", ".xlef", ".lof"))

    # ----------------------------- Image selection, preview -----------------------------
    def on_image_selection_changed(self):
        items = self.tree_images.selectedItems()
        if not items or not self.current_file:
            self._cancel_preview_worker()
            self.selected_image = None
            self.btn_convert.setEnabled(False)
            self.btn_show_image_json.setEnabled(False)
            self.meta_text.clear()
            return
        item = items[0]
        kind = item.data(0, Qt.ItemDataRole.UserRole + 1)
        if kind != "image":
            self._cancel_preview_worker()
            self._clear_preview()
            self.selected_image = None
            self.btn_convert.setEnabled(False)
            self.btn_show_image_json.setEnabled(False)
            self.meta_text.clear()
            return

        name = item.text(0)
        uuid = item.data(0, Qt.ItemDataRole.UserRole + 3)
        ext = os.path.splitext(self.current_file)[1].lower()

        # Find closest ancestor carrying folder_metadata
        folder_meta = None
        ancestor = item.parent()
        while ancestor is not None and folder_meta is None:
            folder_meta = ancestor.data(0, Qt.ItemDataRole.UserRole + 4)
            ancestor = ancestor.parent()
        if folder_meta is None:
            folder_meta = self.folder_metadata_json

        try:
            ext = os.path.splitext(self.current_file)[1].lower()
            if ext == ".lof" or uuid == "__LOF__":
                # LOF: full image metadata is the file itself
                image_metadata = read_leica_file(self.current_file)
                meta = json.loads(image_metadata)
            elif ext == ".xlef":
                # XLEF: like server.py, combine folder image entry with LOF image metadata
                image_metadata_f = json.loads(get_image_metadata(folder_meta, uuid))
                lof_like = json.loads(get_image_metadata_LOF(folder_meta, uuid))
                if "save_child_name" in image_metadata_f:
                    lof_like["save_child_name"] = image_metadata_f["save_child_name"]
                meta = lof_like
            else:
                # LIF: fetch full metadata for preview (includes Position, LUTs, etc.)
                meta = json.loads(read_leica_file(self.current_file, image_uuid=uuid))

            # Start progressive previews 
            self._start_progressive_preview(meta)

            self.selected_image = ImageItem(name=name, uuid=uuid, meta=meta)
            self.btn_convert.setEnabled(True)
            self.btn_show_image_json.setEnabled(True)
            # Store and show metadata summary
            try:
                self.last_image_meta_json = json.dumps(meta, indent=2)
            except Exception:
                self.last_image_meta_json = json.dumps({"error": "Could not serialize metadata"}, indent=2)
            self.meta_text.setPlainText(self.format_meta_summary(meta))
        except Exception as e:
            try:
                tb = traceback.format_exc()
            except Exception:
                tb = f"{type(e).__name__}: {e}"
            self.preview_label.setText("Preview error. See log for details.")
            self.append_log(tb)
            self.selected_image = None
            self.btn_convert.setEnabled(False)
            self.btn_show_image_json.setEnabled(False)
            self.meta_text.clear()

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        # Rescale preview when window changes size
        self._update_preview_scaled()

    # ----------------------------- Output & Convert -----------------------------
    def choose_output(self):
        base = os.path.dirname(self.current_file) if self.current_file else self.current_dir
        d = QFileDialog.getExistingDirectory(self, "Choose output folder", base)
        if d:
            self.edit_out.setText(os.path.normpath(d))

    def convert_selected(self):
        if not self.current_file or not self.selected_image:
            return
        outdir = self.edit_out.text().strip()
        if not outdir:
            QMessageBox.warning(self, "Missing output", "Please choose an output folder.")
            return
        os.makedirs(outdir, exist_ok=True)

        self.log.clear()
        self.append_log(f"Converting {os.path.basename(self.current_file)} / {self.selected_image.name}\n")
        self.btn_convert.setEnabled(False)

        # Determine xy_check_value based on checkbox
        xy_value = self.spin_xy_threshold.value() if self.chk_large_only.isChecked() else 1
        self.worker = ConvertWorker(self.current_file, self.selected_image.uuid, outdir, xy_check_value=xy_value)
        self.worker.progress.connect(self.append_log)
        self.worker.progressParsed.connect(self.on_progress_update)
        self.worker.finished.connect(self.on_convert_finished)
        
        # Show and reset progress bar
        self.progress_widget.setVisible(True)
        self.progress_bar.setValue(0)
        self.lbl_progress_phase.setText("Starting...")
        self.lbl_progress_suffix.setText("")
        
        self.worker.start()

    def on_progress_update(self, percent: int, phase: str, suffix: str):
        """Update the progress bar with parsed progress data."""
        self.progress_bar.setValue(percent)
        self.lbl_progress_phase.setText(phase)
        self.lbl_progress_suffix.setText(suffix)

    def append_log(self, text: str):
        # Don't append pure progress bar updates to the log (they clutter it)
        # Only append lines that are not progress bars
        parsed = parse_progress_text(text)
        if parsed:
            # Skip adding progress bar text to log - it's shown in the progress bar
            return
        self.log.append(text)
        self.log.ensureCursorVisible()

    def on_convert_finished(self, success: bool, result):
        self.btn_convert.setEnabled(True)
        
        # Hide progress bar and show completion
        self.progress_bar.setValue(100)
        self.lbl_progress_phase.setText("Complete" if success else "Failed")
        self.lbl_progress_suffix.setText("")
        # Hide progress bar after a short delay (or keep visible to show final state)
        
        # Show the JSON string in the log panel and in a dialog box after converting
        import json
        if success and result:
            try:
                items = result if isinstance(result, list) else [result]
                lines = []
                for it in items:
                    name = it.get('name')
                    full = it.get('full_path')
                    alt = it.get('alt_path')
                    lines.append(f"✓ {name}: {full}" + (f" (alt: {alt})" if alt else ""))
                summary = "\n".join(lines)
                # Show summary and JSON string in dialog
                json_str = json.dumps(result, indent=2, ensure_ascii=False)
                self.append_log("Conversion result (JSON):\n" + json_str)
                # QMessageBox.information(self, "Done", summary + "\n\nJSON Result:\n" + json_str)
            except Exception:
                QMessageBox.information(self, "Done", "Conversion completed.")
        else:
            QMessageBox.warning(self, "Conversion failed", "No output was produced. See log for details.")

    # ----------------------------- Help -----------------------------
    def show_help(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Convert Leica — Help")
        v = QVBoxLayout(dlg)
        browser = QTextBrowser(dlg); browser.setOpenExternalLinks(True)
        try:
            html_path = Path(__file__).with_name("ConvertLeicaQTHelp.html")
            if html_path.exists():
                browser.setHtml(html_path.read_text(encoding='utf-8'))
            else:
                browser.setHtml("<h2>Help file not found</h2><p>Expected ConvertLeicaQTHelp.html next to the application.</p>")
        except Exception as e:
            browser.setHtml(f"<h2>Error loading help</h2><pre>{e}</pre>")
        v.addWidget(browser)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, parent=dlg)
        btns.rejected.connect(dlg.reject)
        v.addWidget(btns)
        dlg.resize(900, 700)
        dlg.exec()

    # ----------------------------- JSON dialogs -----------------------------
    def show_text_dialog(self, title: str, text: str):
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        v = QVBoxLayout(dlg)
        browser = QTextBrowser(dlg)
        browser.setOpenExternalLinks(False)
        # Wrap JSON in <pre> to preserve formatting
        escaped = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        browser.setHtml(f"<pre style='font-family: Consolas, monospace; font-size: 12px;'>{escaped}</pre>")
        v.addWidget(browser)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, parent=dlg)
        btns.rejected.connect(dlg.reject)
        v.addWidget(btns)
        dlg.resize(900, 700)
        dlg.exec()

    def show_folder_json(self):
        if not self.current_file:
            return
        # Prefer selected folder's JSON; fallback to root file JSON
        items = self.tree_images.selectedItems()
        meta_json = None
        if items:
            sel = items[0]
            kind = sel.data(0, Qt.ItemDataRole.UserRole + 1)
            file_path = sel.data(0, Qt.ItemDataRole.UserRole + 2) or self.current_file
            if kind == "folder":
                uuid = sel.data(0, Qt.ItemDataRole.UserRole + 3)
                meta_json = sel.data(0, Qt.ItemDataRole.UserRole + 4)
                if not meta_json and uuid:
                    try:
                        meta_json = read_leica_file(file_path, folder_uuid=uuid)
                        sel.setData(0, Qt.ItemDataRole.UserRole + 4, meta_json)
                    except Exception:
                        meta_json = None
            elif kind == "root":
                meta_json = sel.data(0, Qt.ItemDataRole.UserRole + 4)
        if not meta_json:
            meta_json = self.folder_metadata_json
            if not meta_json:
                try:
                    meta_json = read_leica_file(self.current_file)
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Could not load folder JSON:\n{e}")
                    return
        self.show_text_dialog("Folder JSON", meta_json)

    def show_image_json(self):
        # Show last selected image's metadata JSON
        if getattr(self, 'last_image_meta_json', None):
            self.show_text_dialog("Image JSON", self.last_image_meta_json)
        else:
            QMessageBox.information(self, "Image JSON", "No image is selected.")

    # ----------------------------- Metadata formatting -----------------------------
    def format_meta_summary(self, meta: dict) -> str:
        def pick(*keys, default=None):
            for k in keys:
                if k in meta and meta[k] is not None:
                    return meta[k]
            return default

        # Name and ID
        name = pick('save_child_name', 'name', 'ElementName', default='(unnamed)')
        uuid = pick('uuid', 'UniqueID', 'ImageUUID', default='')

        # Dimensions: prefer explicit xs/ys/zs/ts/channels, fallback to meta['dimensions']
        xs = pick('xs', default=None)
        ys = pick('ys', default=None)
        zs = pick('zs', default=None)
        ts = pick('ts', default=None)
        cs = pick('channels', default=None)
        if xs is None or ys is None:
            dims_dict = meta.get('dimensions') or {}
            xs = xs if xs is not None else dims_dict.get('x')
            ys = ys if ys is not None else dims_dict.get('y')
            zs = zs if zs is not None else dims_dict.get('z')
            ts = ts if ts is not None else dims_dict.get('t')
            cs = cs if cs is not None else dims_dict.get('c')

        dims_parts = []
        if xs and ys: dims_parts.append(f"{xs}×{ys}")
        if zs: dims_parts.append(f"Z={zs}")
        if ts: dims_parts.append(f"T={ts}")
        if cs: dims_parts.append(f"C={cs}")

        # Voxel size: use xres2/yres2/zres2 with unit resunit2 when available
        vx = pick('xres2', default=None)
        vy = pick('yres2', default=None)
        vz = pick('zres2', default=None)
        vunit = pick('resunit2', default='µm')
        scale_parts = []
        def fmt2(val):
            try:
                return f"{float(val):.2f}"
            except Exception:
                return str(val)
        if vx: scale_parts.append(f"X={fmt2(vx)} {vunit}")
        if vy: scale_parts.append(f"Y={fmt2(vy)} {vunit}")
        if vz: scale_parts.append(f"Z={fmt2(vz)} {vunit}")

        # Pixel type: infer from channelResolution (bits) and isrgb
        isrgb = bool(pick('isrgb', default=False))
        ch_res = meta.get('channelResolution') or []
        bit_depth = None
        if isinstance(ch_res, list) and ch_res:
            try:
                # Use first channel resolution; if mixed, indicate mixed
                first = ch_res[0]
                if all(v == first for v in ch_res if v is not None):
                    bit_depth = f"{first}-bit"
                else:
                    bit_depth = "mixed-bit"
            except Exception:
                bit_depth = None
        pixel_type = (bit_depth + (" RGB" if isrgb else "")) if bit_depth else ("RGB" if isrgb else None)

        # Optional extras
        exp_name = pick('experiment_name', default=None)
        exp_dt = pick('experiment_datetime', 'experiment_datetime_str', default=None)

        lines = [
            f"Name: {name}",
            f"UUID: {uuid}" if uuid else "UUID: (n/a)",
            f"Dimensions: {'  '.join(dims_parts)}" if dims_parts else "Dimensions: (n/a)",
            f"Voxel size: {', '.join(scale_parts)}" if scale_parts else "Voxel size: (n/a)",
            f"Pixel type: {pixel_type}" if pixel_type else "Pixel type: (n/a)",
        ]
        if exp_name:
            lines.append(f"Experiment: {exp_name}")
        if exp_dt:
            # Normalize ISO timestamps like 2025-07-30T12:50:59.777814+00:00 or ...Z
            try:
                iso = str(exp_dt).strip().replace('Z', '+00:00')
                dt = datetime.fromisoformat(iso)
                pretty = dt.strftime('%Y-%m-%d %H:%M:%S')
            except Exception:
                # Fallback: best-effort trim microseconds and timezone
                pretty = str(exp_dt).strip().replace('T', ' ')
                if '.' in pretty:
                    pretty = pretty.split('.', 1)[0]
                # Remove timezone offset part if present (e.g., +00:00 or -05:00)
                for sep in ['+', '-']:
                    pos = pretty.find(sep, 10)
                    if pos != -1:
                        pretty = pretty[:pos]
                        break
            lines.append(f"Date: {pretty}")
        return "\n".join(lines)

    # ----------------------------- Tree helpers -----------------------------
    def populate_children(self, parent_item: QTreeWidgetItem, folder_meta_json: str):
        """Populate children under parent_item from a folder metadata JSON string."""
        try:
            meta = json.loads(folder_meta_json)
        except Exception:
            return
        children = meta.get("children", [])
        for ch in children:
            name = ch.get("name", "") or ch.get("ElementName", "")
            # Apply same filtering rules as filesystem tree
            low = name.lower()
            if ("metadata" in low or "_pmd_" in low or "_histo" in low or
                "_environmetalgraph" in low or low.endswith(".lifext") or
                low in ("iomanagerconfiguation", "iomanagerconfiguration")):
                continue
            uuid = ch.get("uuid") or ""
            ctype = (ch.get("type") or "").lower()
            filetype = ch.get("filetype") or ""
            item = QTreeWidgetItem([name])
            if ctype in ("folder", "file"):
                item.setData(0, Qt.ItemDataRole.UserRole + 1, "folder")
                item.setIcon(0, self.icon_folder())
                # placeholder to expand lazily
                item.addChild(QTreeWidgetItem(["…"]))
            else:
                item.setData(0, Qt.ItemDataRole.UserRole + 1, "image")
                item.setIcon(0, self.icon_image())
            item.setData(0, Qt.ItemDataRole.UserRole + 2, self.current_file)  # file path
            item.setData(0, Qt.ItemDataRole.UserRole + 3, uuid)               # image/folder uuid
            item.setData(0, Qt.ItemDataRole.UserRole + 5, filetype)
            parent_item.addChild(item)

    def on_tree_item_expanded(self, item: QTreeWidgetItem):
        kind = item.data(0, Qt.ItemDataRole.UserRole + 1)
        if kind != "folder":
            return
        # If we already loaded children for this folder, skip
        if item.childCount() == 1 and item.child(0).text(0) == "…":
            file_path = item.data(0, Qt.ItemDataRole.UserRole + 2)
            uuid = item.data(0, Qt.ItemDataRole.UserRole + 3)
            try:
                meta_json = read_leica_file(file_path, folder_uuid=uuid)
                item.setData(0, Qt.ItemDataRole.UserRole + 4, meta_json)
                # replace placeholder
                item.removeChild(item.child(0))
                self.populate_children(item, meta_json)
            except Exception as e:
                self.append_log(f"Error expanding folder: {e}")

    # ----------------------------- Icon helpers -----------------------------
    def _icon(self, rel: str, fallback: QStyle.StandardPixmap | None = None) -> QIcon:
        p = Path(__file__).with_name("images").joinpath(rel)
        if p.exists():
            return QIcon(str(p))
        if fallback is not None:
            return self.style().standardIcon(fallback)
        return QIcon()

    def icon_folder(self) -> QIcon:
        return self._icon("folder.svg", QStyle.StandardPixmap.SP_DirIcon)

    def icon_image(self) -> QIcon:
        return self._icon("image.svg", QStyle.StandardPixmap.SP_FileIcon)

    def icon_for_file(self, ext: str) -> QIcon:
        ext = ext.lower().lstrip('.')
        mapping = {
            'lif': "file-lif.svg",
            'xlef': "file-xlef.svg",
            'lof': "file-lof.svg",
        }
        fname = mapping.get(ext, "file.svg")
        return self._icon(fname, QStyle.StandardPixmap.SP_FileIcon)

    # ----------------------------- Preview helpers -----------------------------
    def _clear_preview(self):
        self._preview_pixmap = None  # type: ignore[attr-defined]
        self.preview_label.setPixmap(QPixmap())
        self.preview_label.setText("Preview will appear here")

    def _update_preview_scaled(self):
        try:
            if getattr(self, "_preview_pixmap", None):
                self.preview_label.setPixmap(self._preview_pixmap.scaled(
                    self.preview_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
                ))
        except Exception:
            pass

    def _cancel_preview_worker(self):
        try:
            if self._preview_worker is not None:
                self._preview_worker.requestInterruption()
                # Do not wait() here to avoid blocking the UI; worker will finish soon.
        except Exception:
            pass
        finally:
            self._preview_worker = None

    def _start_progressive_preview(self, meta: dict):
        """Kick off a new progressive preview job for the given metadata.

        Cancels any previous job; only applies previews for the latest job id.
        """
        # Cancel any previous worker and bump job id
        self._cancel_preview_worker()
        self._preview_job_id += 1
        job_id = self._preview_job_id

        # Show immediate loading hint
        self.preview_label.setText("Loading preview…")

        # Use same progressive steps as server when possible
        steps = list(self._SERVER_PREVIEW_STEPS)
        # Apply 2048x2048 small-image rule: if image small, fetch only the max step
        xs = meta.get("xs") or (meta.get("dimensions") or {}).get("x")
        ys = meta.get("ys") or (meta.get("dimensions") or {}).get("y")
        try:
            xi = int(xs) if xs is not None else None
            yi = int(ys) if ys is not None else None
        except Exception:
            xi = yi = None
        if xi is not None and yi is not None and xi <= 2048 and yi <= 2048 and steps:
            heights = [max(steps)]
        else:
            # If the largest step is already cached, skip smaller steps and use only the largest
            if steps:
                max_step = max(steps)
                uid = meta.get("UniqueID") or meta.get("uuid") or meta.get("ImageUUID")
                cache_dir = self.get_cache_dir()
                if uid:
                    largest_cached_path = os.path.join(cache_dir, f"{uid}_h{int(max_step)}.png")
                    if os.path.exists(largest_cached_path):
                        heights = [max_step]
                    else:
                        heights = steps
                else:
                    heights = steps
            else:
                heights = steps
        cache_dir = self.get_cache_dir()
        worker = PreviewWorker(job_id, meta, heights, cache_dir, self._SERVER_PREVIEW_CACHE_MAX,
                    use_memmap=True, pause_ms=100)
        worker.previewReady.connect(self._on_preview_ready)
        worker.error.connect(self._on_preview_error)
        worker.cacheInfo.connect(self._on_cache_info)
        self._preview_worker = worker
        worker.start()

    def _on_preview_ready(self, job_id: int, height: int, temp_png: str):  # slot
        # Ignore stale jobs
        if job_id != self._preview_job_id:
            try:
                if temp_png and os.path.exists(temp_png):
                    os.remove(temp_png)
            except Exception:
                pass
            return
        try:
            pix = QPixmap(temp_png)
            if not pix.isNull():
                self._preview_pixmap = pix
                self._update_preview_scaled()
        except Exception:
            pass

    def _on_preview_error(self, job_id: int, message: str):  # slot
        if job_id != self._preview_job_id:
            return
        self.preview_label.setText(f"Preview error: {message}")

    def _on_cache_info(self, job_id: int, height: int, cached_before: bool):  # slot
        if job_id != self._preview_job_id:
            return
        # Lightweight trace in log; helps confirm cache hits
        # self.append_log(f"Preview {height}px: {'cache hit' if cached_before else 'cache miss'}")


def main() -> None:
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    # Optional stylesheet
    css_file = Path(__file__).parent / "styles" / "darktheme.css"
    if css_file.exists():
        try:
            css = css_file.read_text(encoding='utf-8')
            css = css.replace('$$IMAGES_DIR$$', (Path(__file__).parent / 'images').as_posix())
            app.setStyleSheet(css)
        except Exception:
            pass
    win = ConvertLeicaApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":  # pragma: no cover
    main()
