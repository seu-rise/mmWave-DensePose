"""Standalone read-only multimodal sequence viewer.

Run: python -m mmwave_densepose.gui.dataset_viewer --file /path/to/sequence.pkl
"""

import argparse
import json
from pathlib import Path
import re
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QFontDatabase, QImage, QPixmap
from PyQt5.QtWidgets import (
    QAction, QApplication, QCheckBox, QComboBox, QFileDialog, QFrame,
    QDoubleSpinBox, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QSizePolicy, QSlider,
    QSpinBox, QSplitter, QVBoxLayout, QWidget, QAbstractSpinBox,
)
from matplotlib import colormaps, font_manager
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from mmwave_densepose.data.inspection import (
    IMAGE_KEYS, SKELETON_EDGES_19, LoadCancelled, load_sequence,
)

COLORS = {"background": "#edf1f3", "surface": "#ffffff", "ink": "#20333e",
          "muted": "#647985", "border": "#d4dfe4", "accent": "#087f8c",
          "radar": "#6AB4E6", "joint": "#00BAAF", "bone": "#545454",
          "error": "#b54935", "image": "#15232c"}
IMAGE_TITLES = dict(zip(IMAGE_KEYS, ("Color", "Depth", "Infrared", "Body index")))
SUPPORTED = {".pkl", ".pickle", ".npz", ".npy"}


def configure_fonts(app):
    """Use bundled Matplotlib fonts when Qt's offscreen font database is empty."""
    families = set(QFontDatabase().families())
    for family in ("DejaVu Sans", "DejaVu Sans Mono", "DejaVu Serif"):
        if family not in families:
            QFontDatabase.addApplicationFont(font_manager.findfont(family))
    families = set(QFontDatabase().families())
    app.setFont(QFont("Times New Roman" if "Times New Roman" in families else "DejaVu Serif", 10))


def decode_image(data):
    """Decode original encoded bytes for display, without re-encoding source data."""
    if not data:
        return QImage()
    try:
        return QImage.fromData(bytes(data))
    except (TypeError, ValueError):
        return QImage()


def label(text="", object_name=None):
    widget = QLabel(text)
    widget.setTextFormat(Qt.PlainText)
    if object_name:
        widget.setObjectName(object_name)
    return widget


def timestamp_text(value):
    return "not stored" if value is None else f"{value:,.1f} ms"


class ImagePanel(QFrame):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setObjectName("imagePanel")
        self._pixmap = QPixmap()
        self.has_image = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        header = QVBoxLayout()
        header.setSpacing(2)
        header.addWidget(label(title, "panelTitle"))
        self.time_label = label("not stored", "imageTime")
        self.time_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.time_label.setToolTip("Original stored image timestamp in milliseconds")
        header.addWidget(self.time_label)
        layout.addLayout(header)
        self.image_label = label("No image loaded", "imageCanvas")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setWordWrap(True)
        self.image_label.setMinimumSize(90, 90)
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        layout.addWidget(self.image_label, 1)

    def set_payload(self, data, timestamp=None):
        image = decode_image(data)
        self.has_image = not image.isNull()
        self._pixmap = QPixmap.fromImage(image) if self.has_image else QPixmap()
        self.time_label.setText(timestamp_text(timestamp))
        if not self.has_image:
            self.image_label.clear()
            self.image_label.setText("Unreadable image data" if data else "No image for this frame")
        else:
            self._rescale()

    def _rescale(self):
        if self.has_image:
            self.image_label.setPixmap(self._pixmap.scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()


class SequenceLoader(QThread):
    loaded = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()
    progress_update = pyqtSignal(int, str)

    def __init__(self, path, trusted, checksum, parent=None):
        super().__init__(parent)
        self.path, self.trusted, self.checksum = path, trusted, checksum

    def run(self):
        try:
            record = load_sequence(self.path, trusted=self.trusted,
                expected_sha256=self.checksum, progress=self.progress_update.emit,
                cancelled=self.isInterruptionRequested)
            bounds = {}
            for index in range(record.frame_count):
                if self.isInterruptionRequested():
                    raise LoadCancelled()
                points = record.frame(index)["radar"]
                for channel in (3, 4):
                    if points.shape[1] > channel:
                        values = points[:, channel]
                        finite = values[np.isfinite(values)]
                        if len(finite):
                            low, high = float(finite.min()), float(finite.max())
                            previous = bounds.get(channel, (low, high))
                            bounds[channel] = (min(previous[0], low), max(previous[1], high))
            record.summary["color_bounds"] = bounds
            self.loaded.emit(record)
        except LoadCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class DatasetViewer(QMainWindow):
    """Open one sequence at a time, with independent frame and camera controls."""

    sequence_loaded = pyqtSignal(object)
    load_failed = pyqtSignal(str)

    def __init__(self, data_root=None, trust_pickle=False):
        super().__init__()
        configure_fonts(QApplication.instance())
        self.data_root = Path(data_root or Path(__file__).resolve().parents[2] / "data/mmwave_densepose")
        self.trust_pickle = trust_pickle
        self.record, self.worker, self.frame_index = None, None, 0
        self._closing = False
        self._artists, self._colorbar = [], None
        self._files = []
        self._listing_directory = None
        self._subject_id = "not stored"
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._advance)
        self.setWindowTitle("mmWave-DensePose | Dataset Viewer")
        self.resize(1480, 900)
        self.setMinimumSize(1020, 700)
        self._build_ui()
        self._apply_style()
        self._set_busy(False)
        self._populate_files(self._browse_directory(self.data_root))

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)
        header = QHBoxLayout()
        header.addWidget(label("mmWave-DensePose · Dataset Viewer", "title"), 1)
        header.addWidget(label("READ ONLY", "badge"))
        layout.addLayout(header)

        selection = QGroupBox("File Selection")
        selection_layout = QVBoxLayout(selection)
        selection_layout.setContentsMargins(10, 12, 10, 10)
        folder_row = QHBoxLayout()
        folder_row.addWidget(label("Data Folder"))
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        folder_row.addWidget(self.path_edit, 1)
        self.open_button = QPushButton("Open file…")
        self.folder_button = QPushButton("Select folder…")
        self.open_button.clicked.connect(self.choose_file)
        self.folder_button.clicked.connect(self.choose_folder)
        folder_row.addWidget(self.folder_button)
        folder_row.addWidget(self.open_button)
        selection_layout.addLayout(folder_row)
        files_row = QHBoxLayout()
        files_row.addWidget(label("Sequence"))
        self.file_combo = QComboBox()
        self.file_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.file_combo.setMinimumContentsLength(16)
        self.file_combo.activated.connect(self._file_selected)
        files_row.addWidget(self.file_combo, 1)
        files_row.addWidget(label("Subject ID"))
        self.subject_id_edit = QLineEdit("not stored")
        self.subject_id_edit.setReadOnly(True)
        self.subject_id_edit.setMaximumWidth(110)
        files_row.addWidget(self.subject_id_edit)
        self.previous_file_button = QPushButton("Previous File")
        self.next_file_button = QPushButton("Next File")
        self.previous_file_button.clicked.connect(lambda: self._step_file(-1))
        self.next_file_button.clicked.connect(lambda: self._step_file(1))
        files_row.addWidget(self.previous_file_button)
        files_row.addWidget(self.next_file_button)
        selection_layout.addLayout(files_row)
        self.summary_label = label("Open an anonymized PKL to inspect original images, radar points and ground truth.", "muted")
        self.summary_label.setWordWrap(True)
        selection_layout.addWidget(self.summary_label)
        layout.addWidget(selection)

        self.splitter = QSplitter(Qt.Horizontal)
        viewport = QGroupBox("3D Radar Point Cloud + Skeleton")
        viewport.setObjectName("viewport")
        view = QVBoxLayout(viewport)
        view.setContentsMargins(8, 12, 8, 8)
        self.frame_info = label("Open a sequence. Drag the 3D view to rotate.", "muted")
        self.frame_info.setWordWrap(True)
        self.frame_info.setMinimumHeight(35)
        view.addWidget(self.frame_info)
        self.figure = Figure(figsize=(6, 6), facecolor=COLORS["surface"])
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.axes = self.figure.add_subplot(111, projection="3d")
        self.figure.subplots_adjust(left=0.06, right=0.77, bottom=0.09, top=0.98)
        self._colorbar_axes = self.figure.add_axes([0.88, 0.22, 0.026, 0.56])
        self._colorbar_axes.set_visible(False)
        font = "Times New Roman" if "Times New Roman" in QFontDatabase().families() else "DejaVu Serif"
        self.axes.set(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)")
        self.axes.tick_params(labelsize=8)
        for axis in (self.axes.xaxis, self.axes.yaxis, self.axes.zaxis):
            axis.set_major_locator(MaxNLocator(nbins=3))
            axis.label.set_fontname(font)
            for tick in axis.get_ticklabels():
                tick.set_fontname(font)
        self.axes.set_facecolor(COLORS["surface"])
        self.reset_view()
        view.addWidget(self.canvas, 1)

        toggles = QHBoxLayout()
        self.radar_check = QCheckBox("Show Radar")
        self.skeleton_check = QCheckBox("Show Skeleton")
        self.boundary_check = QCheckBox("Show Boundary")
        self.images_check = QCheckBox("Show Images")
        for checkbox in (self.radar_check, self.skeleton_check, self.boundary_check, self.images_check):
            checkbox.setChecked(True)
            toggles.addWidget(checkbox)
        toggles.addStretch(1)
        self.fit_button = QPushButton("Fit Frame")
        self.fit_button.clicked.connect(self.fit_view)
        toggles.addWidget(self.fit_button)
        view.addLayout(toggles)
        tools = QHBoxLayout()
        tools.addWidget(label("Radar Render"))
        self.color_mode = QComboBox()
        self.color_mode.addItems(["Solid", "Doppler", "SNR"])
        tools.addWidget(self.color_mode)
        tools.addWidget(label("Doppler"))
        self.doppler_min, self.doppler_max = QDoubleSpinBox(), QDoubleSpinBox()
        for spin, value in ((self.doppler_min, -150), (self.doppler_max, 150)):
            spin.setRange(-1000000, 1000000)
            spin.setDecimals(1)
            spin.setValue(value)
            spin.setMaximumWidth(85)
            spin.valueChanged.connect(self._render_frame)
        tools.addWidget(self.doppler_min)
        tools.addWidget(label("to"))
        tools.addWidget(self.doppler_max)
        tools.addStretch(1)
        self.reset_button = QPushButton("Reset View")
        self.reset_button.clicked.connect(self.reset_view)
        tools.addWidget(self.reset_button)
        view.addLayout(tools)
        self.splitter.addWidget(viewport)

        self.image_area = QWidget()
        self.image_area.setMinimumWidth(320)
        images_layout = QVBoxLayout(self.image_area)
        images_layout.setContentsMargins(0, 0, 0, 0)
        images_layout.setSpacing(6)
        images_layout.addWidget(label("Original Image Streams", "panelTitle"))
        self.images = {IMAGE_KEYS[0]: ImagePanel(IMAGE_TITLES[IMAGE_KEYS[0]])}
        images_layout.addWidget(self.images[IMAGE_KEYS[0]], 2)
        thumbnails = QHBoxLayout()
        thumbnails.setSpacing(6)
        for key in IMAGE_KEYS[1:]:
            panel = ImagePanel(IMAGE_TITLES[key])
            self.images[key] = panel
            thumbnails.addWidget(panel, 1)
        images_layout.addLayout(thumbnails, 1)
        image_note = label("Stored frame association. No resynchronization or re-encoding.", "muted")
        image_note.setWordWrap(True)
        images_layout.addWidget(image_note)
        self.splitter.addWidget(self.image_area)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        self.splitter.setSizes([850, 560])
        layout.addWidget(self.splitter, 1)

        transport = QFrame()
        transport.setObjectName("transport")
        row = QHBoxLayout(transport)
        row.setContentsMargins(12, 10, 12, 10)
        self.previous_button = QPushButton("Previous")
        self.play_button = QPushButton("Play")
        self.next_button = QPushButton("Next")
        row.addWidget(self.previous_button)
        row.addWidget(self.play_button)
        row.addWidget(self.next_button)
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        row.addWidget(self.frame_slider, 1)
        row.addWidget(label("Frame"))
        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(0, 0)
        row.addWidget(self.frame_spin)
        self.frame_total = label("/ —", "muted")
        row.addWidget(self.frame_total)
        row.addWidget(label("FPS"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(10)
        self.fps_spin.valueChanged.connect(lambda value: self.timer.setInterval(round(1000 / value)))
        row.addWidget(self.fps_spin)
        layout.addWidget(transport)
        status = QHBoxLayout()
        self.status_label = label("Choose a file, or select a sequence above.", "muted")
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        status.addWidget(self.status_label, 1)
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(180)
        self.cancel_button = QPushButton("Cancel load")
        self.cancel_button.clicked.connect(self.cancel_load)
        status.addWidget(self.progress_bar)
        status.addWidget(self.cancel_button)
        layout.addLayout(status)
        self.previous_button.clicked.connect(lambda: self.set_frame(self.frame_index - 1))
        self.next_button.clicked.connect(lambda: self.set_frame(self.frame_index + 1))
        self.play_button.clicked.connect(self.toggle_playback)
        self.frame_slider.valueChanged.connect(self.set_frame)
        self.frame_spin.valueChanged.connect(self.set_frame)
        self.radar_check.toggled.connect(self._render_frame)
        self.skeleton_check.toggled.connect(self._render_frame)
        self.boundary_check.toggled.connect(self._render_frame)
        self.images_check.toggled.connect(self.image_area.setVisible)
        self.color_mode.currentIndexChanged.connect(self._render_frame)
        open_action = QAction("Open file…", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.choose_file)
        self.addAction(open_action)
        self.open_action = open_action

    def _apply_style(self):
        self.setFont(QApplication.instance().font())
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background: {COLORS['background']}; color: {COLORS['ink']}; }}
            QGroupBox, QFrame#transport, QFrame#imagePanel {{
                background: {COLORS['surface']}; border: 1px solid {COLORS['border']}; border-radius: 5px; }}
            QGroupBox {{ margin-top: 8px; font-weight: 600; }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}
            QLabel {{ background: transparent; border: none; }}
            QLabel#title {{ font-size: 20px; font-weight: 700; letter-spacing: 1px; }}
            QLabel#section {{ font-size: 11px; font-weight: 700; color: {COLORS['muted']}; }}
            QLabel#muted, QLabel#imageTime {{ color: {COLORS['muted']}; font-size: 11px; }}
            QLabel#imageTime {{ font-family: 'DejaVu Sans Mono'; font-size: 9px; }}
            QLabel#panelTitle {{ font-weight: 600; }}
            QLabel#badge {{ color: {COLORS['accent']}; background: #dfefed; padding: 6px 10px; border-radius: 4px; font-size: 10px; }}
            QLabel#imageCanvas {{ background: {COLORS['image']}; color: #bacbd4; border-radius: 3px; }}
            QPushButton {{ background: white; border: 1px solid {COLORS['border']}; border-radius: 4px; padding: 7px 10px; }}
            QPushButton:hover {{ border-color: {COLORS['accent']}; color: {COLORS['accent']}; }}
            QPushButton:disabled {{ color: #a2afb6; background: #f3f5f6; }}
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ background: white; border: 1px solid {COLORS['border']}; border-radius: 3px; padding: 4px; }}
            QSlider::groove:horizontal {{ height: 5px; background: #d4dfe4; border-radius: 2px; }}
            QSlider::handle:horizontal {{ background: {COLORS['accent']}; width: 14px; margin: -5px 0; border-radius: 3px; }}
            QProgressBar {{ border: 1px solid {COLORS['border']}; border-radius: 3px; text-align: center; }}
            QProgressBar::chunk {{ background: {COLORS['accent']}; }}
        """)

    @staticmethod
    def _browse_directory(root):
        root = Path(root)
        for name in ("raw_face_anonymized", "raw"):
            if (root / name).is_dir():
                return root / name
        return root

    def _populate_files(self, directory):
        directory = Path(directory)
        try:
            self._listing_directory = directory.resolve()
            self._files = sorted((p.resolve() for p in directory.iterdir() if p.is_file()
                                  and p.suffix.lower() in SUPPORTED),
                                 key=lambda p: [int(x) if x.isdigit() else x.lower()
                                                for x in re.split(r"(\d+)", p.name)]) if directory.is_dir() else []
        except OSError as exc:
            self._files = []
            self.status_label.setText(f"Unable to list folder: {exc}")
        self.path_edit.setText(str(directory))
        self.file_combo.blockSignals(True)
        self.file_combo.clear()
        for path in self._files:
            self.file_combo.addItem(path.name, str(path))
            self.file_combo.setItemData(self.file_combo.count() - 1, str(path), Qt.ToolTipRole)
        current = self.file_combo.findData(str(self.record.path)) if self.record else -1
        self.file_combo.setCurrentIndex(current)
        self.file_combo.blockSignals(False)
        self._update_file_navigation()

    def _file_selected(self, index):
        path = self.file_combo.itemData(index)
        if path and not self.open_file(path):
            current = self.file_combo.findData(str(self.record.path)) if self.record else -1
            self.file_combo.setCurrentIndex(current)

    def _step_file(self, direction):
        index = self.file_combo.currentIndex() + direction
        if 0 <= index < self.file_combo.count():
            self._file_selected(index)

    def _update_file_navigation(self):
        index = self.file_combo.currentIndex()
        ready = self.worker is None
        self.previous_file_button.setEnabled(ready and index > 0)
        self.next_file_button.setEnabled(ready and index < self.file_combo.count() - 1)

    def choose_file(self):
        if self.worker is not None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open sequence file",
            str(self._browse_directory(self.data_root)), "Sequence files (*.pkl *.pickle *.npz *.npy)")
        if path:
            self.open_file(path)

    def choose_folder(self):
        if self.worker is not None:
            return
        path = QFileDialog.getExistingDirectory(self, "Choose dataset or sequence folder", str(self.data_root))
        if path:
            self.data_root = Path(path)
            self._populate_files(self._browse_directory(self.data_root))

    def _expected_checksum(self, path):
        self._subject_id = "not stored"
        directories = [path.parent, path.parent.parent]
        if path.parent.name == "raw_face_anonymized":
            directories.append(path.parent.parent / "raw")  # Subject metadata only, never unmasked images.
        directories.append(self.data_root)
        for directory in dict.fromkeys(directories):
            manifest_path = directory / "manifest.json"
            if not manifest_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                raise ValueError("manifest root must be an object")
            sequences = manifest.get("sequences")
            if not isinstance(sequences, dict):
                raise ValueError("manifest sequences must be an object")
            if manifest.get("dataset_kind") != "full_multimodal":
                # The original viewer stores filename-to-subject mappings only.
                subject = sequences.get(path.name)
                if subject is not None:
                    if not isinstance(subject, str):
                        raise ValueError("subject manifest entry must be a string")
                    self._subject_id = subject or "not stored"
                continue
            for entry in sequences.values():
                if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
                    raise ValueError("manifest sequence must contain a file path")
                if (directory / entry["file"]).resolve() == path:
                    checksum = entry.get("sha256")
                    if not isinstance(checksum, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", checksum):
                        raise ValueError("manifest sequence must contain a SHA256 checksum")
                    self._subject_id = str(entry.get("subject_id") or "not stored")
                    return checksum
        return None

    def open_file(self, path):
        if self.worker is not None:
            return False
        path = Path(path).expanduser().resolve()
        if path.suffix.lower() in (".pkl", ".pickle") and not self.trust_pickle:
            answer = QMessageBox.warning(self, "Open trusted pickle only",
                f"PKL files can execute code. Open this file only if you trust its source.\n\n{path}\n\n"
                "This viewer does not anonymize images. Prefer the supplied anonymized dataset.",
                QMessageBox.Open | QMessageBox.Cancel, QMessageBox.Cancel)
            if answer != QMessageBox.Open:
                return False
        self._stop_playback()
        try:
            checksum = self._expected_checksum(path)
        except (OSError, ValueError, KeyError) as exc:
            self._load_error(f"Invalid dataset manifest: {exc}")
            return False
        self.record = None
        self._clear_display()
        self._set_busy(True)
        self.status_label.setText(f"Opening {path.name}…")
        self.status_label.setStyleSheet("")
        self.worker = SequenceLoader(path, True, checksum, self)
        self.worker.progress_update.connect(self._load_progress)
        self.worker.loaded.connect(self.set_sequence)
        self.worker.failed.connect(self._load_error)
        self.worker.cancelled.connect(lambda: self.status_label.setText("Loading cancelled."))
        self.worker.finished.connect(lambda worker=self.worker: self._load_finished(worker))
        self.worker.start()
        return True

    def _load_progress(self, percent, message):
        self.progress_bar.setValue(percent)
        self.status_label.setText(message)

    def _load_finished(self, worker):
        if self.worker is not worker:
            worker.deleteLater()
            return
        self.worker = None
        worker.deleteLater()
        self._set_busy(False)
        if self._closing:
            self.close()

    def cancel_load(self):
        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption()
            self.cancel_button.setEnabled(False)
            self.status_label.setText("Cancelling… an active pickle read must finish safely first.")

    def _load_error(self, message):
        self.status_label.setText(f"Unable to open sequence: {message}")
        self.status_label.setStyleSheet(f"color: {COLORS['error']};")
        self.load_failed.emit(message)

    def set_sequence(self, record):
        if self._closing:
            return
        self._stop_playback()
        self.record = record
        self.frame_index = 0
        for control in (self.frame_slider, self.frame_spin):
            control.blockSignals(True)
            control.setRange(0, record.frame_count - 1)
            control.setValue(0)
            control.blockSignals(False)
        self.frame_total.setText(f"/ {record.frame_count - 1:,}")
        modalities = record.summary.get("modalities", [])
        image_count = sum(key in modalities for key in IMAGE_KEYS)
        self.summary_label.setText(f"{record.frame_count:,} frames  ·  {image_count} image streams  ·  "
            f"{record.summary['source_kind']}  ·  XYZ in metres  ·  Original frame order")
        self.subject_id_edit.setText(self._subject_id)
        self.status_label.setText(str(record.path))
        self.status_label.setStyleSheet("")
        if record.path.parent != self._listing_directory:
            self._populate_files(record.path.parent)
        self.file_combo.setCurrentIndex(self.file_combo.findData(str(record.path)))
        self._update_file_navigation()
        self._set_busy(self.worker is not None)
        self._render_frame()
        self.reset_view()
        self.sequence_loaded.emit(record)

    def _clear_display(self):
        for artist in self._artists:
            artist.remove()
        self._artists.clear()
        self._remove_colorbar()
        for panel in self.images.values():
            panel.set_payload(None)
        self.frame_info.setText("Loading a new sequence; previous sequence released.")
        self.summary_label.setText("Loading…")
        self.subject_id_edit.setText("not stored")
        self.canvas.draw_idle()

    def _remove_colorbar(self):
        if self._colorbar:
            self._colorbar.remove()
            self._colorbar = None
            self._colorbar_axes = self.figure.add_axes([0.88, 0.22, 0.026, 0.56])
        self._colorbar_axes.set_visible(False)

    def _set_busy(self, busy):
        for widget in (self.open_button, self.folder_button, self.file_combo):
            widget.setEnabled(not busy)
        self.open_action.setEnabled(not busy)
        self.progress_bar.setVisible(busy)
        self.cancel_button.setVisible(busy)
        self.cancel_button.setEnabled(busy)
        ready = self.record is not None and not busy
        for widget in (self.frame_slider, self.frame_spin, self.play_button,
                       self.reset_button, self.fit_button, self.color_mode, self.radar_check, self.skeleton_check,
                       self.boundary_check, self.images_check):
            widget.setEnabled(ready)
        for spin in (self.doppler_min, self.doppler_max):
            spin.setEnabled(ready and self.color_mode.currentIndex() == 1)
        self.previous_button.setEnabled(ready and self.frame_index > 0)
        self.next_button.setEnabled(ready and self.frame_index < self.record.frame_count - 1)
        self._update_file_navigation()

    def set_frame(self, index):
        if self.record is None:
            return
        self.frame_index = max(0, min(int(index), self.record.frame_count - 1))
        for control in (self.frame_slider, self.frame_spin):
            control.blockSignals(True)
            control.setValue(self.frame_index)
            control.blockSignals(False)
        if self.frame_index == self.record.frame_count - 1:
            self._stop_playback()
        self.previous_button.setEnabled(self.frame_index > 0)
        self.next_button.setEnabled(self.frame_index < self.record.frame_count - 1)
        self._render_frame()

    def _stop_playback(self):
        self.timer.stop()
        self.play_button.setText("Play")

    def toggle_playback(self):
        if self.record is None:
            return
        if self.timer.isActive():
            self._stop_playback()
        else:
            if self.frame_index == self.record.frame_count - 1:
                self.set_frame(0)
            self.play_button.setText("Pause")
            self.timer.start(round(1000 / self.fps_spin.value()))

    def _advance(self):
        self.set_frame(self.frame_index + 1)

    def _render_frame(self, *_):
        if self.record is None:
            return
        for spin in (self.doppler_min, self.doppler_max):
            spin.setEnabled(self.worker is None and self.color_mode.currentIndex() == 1)
        frame = self.record.frame(self.frame_index)
        for artist in self._artists:
            artist.remove()
        self._artists.clear()
        points = frame["radar"]
        points = points[np.isfinite(points[:, :3]).all(axis=1)]
        joints = frame["joints"]
        mode = self.color_mode.currentIndex()
        channel = {1: 3, 2: 4}.get(mode)
        feature_available = channel is not None and points.shape[1] > channel and len(points)
        if feature_available:
            values = points[:, channel]
            valid = values[np.isfinite(values)]
            low, high = self.record.summary.get("color_bounds", {}).get(channel,
                (float(valid.min()), float(valid.max())) if len(valid) else (0., 1.))
            if channel == 3:
                low, high = sorted((self.doppler_min.value(), self.doppler_max.value()))
            if high <= low:
                high = low + 1
            norm = Normalize(low, high)
            cmap = colormaps["coolwarm" if channel == 3 else "viridis"]
            colors = cmap(norm(np.nan_to_num(values, nan=low)))
            colors[~np.isfinite(values)] = (0.5, 0.5, 0.5, 1.)
            if not self._colorbar:
                self._colorbar_axes.set_visible(True)
                self._colorbar = self.figure.colorbar(ScalarMappable(norm=norm, cmap=cmap),
                    cax=self._colorbar_axes)
            else:
                self._colorbar.mappable.set_norm(norm)
                self._colorbar.mappable.set_cmap(cmap)
            self._colorbar.set_label(self.color_mode.currentText(), fontsize=8)
            self._colorbar.ax.tick_params(labelsize=7)
        else:
            colors = COLORS["radar"]
            self._remove_colorbar()
        if self.radar_check.isChecked() and len(points):
            self._artists.append(self.axes.scatter(*points[:, :3].T, s=26, c=colors, alpha=0.92,
                edgecolors=COLORS["bone"], linewidths=0.3, depthshade=False))
        joint_count = 0
        if joints is not None:
            valid_joints = np.isfinite(joints).all(axis=1)
            joint_count = int(valid_joints.sum())
            if self.skeleton_check.isChecked():
                for start, end in SKELETON_EDGES_19 if len(joints) == 19 else ():
                    if valid_joints[start] and valid_joints[end]:
                        self._artists.extend(self.axes.plot(*joints[[start, end]].T,
                            color=COLORS["bone"], linewidth=2))
                self._artists.append(self.axes.scatter(*joints[valid_joints].T, s=22,
                    c=COLORS["joint"], edgecolors="white", linewidths=0.5, depthshade=False))
            if self.boundary_check.isChecked() and joint_count:
                low = joints[valid_joints].min(axis=0) - 0.2
                high = joints[valid_joints].max(axis=0) + 0.2
                corners = np.array([[low[0] if not i & 1 else high[0],
                                     low[1] if not i & 2 else high[1],
                                     low[2] if not i & 4 else high[2]] for i in range(8)])
                for start in range(8):
                    for bit in (1, 2, 4):
                        end = start ^ bit
                        if start < end:
                            self._artists.extend(self.axes.plot(*corners[[start, end]].T,
                                color="#94A3B8", linewidth=0.8, linestyle="--", alpha=0.8))
        for key, panel in self.images.items():
            panel.set_payload(frame["images"][key], frame["image_timestamps"][key])
        note = f"  |  {self.color_mode.currentText()} unavailable; solid color" if mode and not feature_available else ""
        self.frame_info.setText(f"Frame {self.frame_index:,}  |  {len(points):,} finite radar points  |  {joint_count} joints{note}\n"
            f"Radar: {timestamp_text(frame['radar_timestamp'])}    Kinect: {timestamp_text(frame['kinect_timestamp'])}")
        self.canvas.draw_idle()

    def reset_view(self):
        """Restore the source viewer's fixed physical range and camera angle."""
        self.axes.set(xlim=(-1.5, 1.5), ylim=(1., 4.), zlim=(-0.1, 2.))
        self.axes.set_xticks([-1., 0., 1.])
        self.axes.set_yticks([1., 2., 3., 4.])
        self.axes.set_zticks([0., 1., 2.])
        self.axes.set_box_aspect((3., 3., 2.1))
        self.axes.view_init(elev=16, azim=-60)
        self.canvas.draw_idle()

    def fit_view(self):
        if self.record is None:
            return
        frame = self.record.frame(self.frame_index)
        arrays = [frame["radar"][:, :3]]
        if frame["joints"] is not None:
            arrays.append(frame["joints"])
        points = np.concatenate(arrays)
        points = points[np.isfinite(points).all(axis=1)]
        if not len(points):
            points = np.array([[-1., -1., 0.], [1., 1., 2.]])
        low, high = points.min(axis=0), points.max(axis=0)
        if self.boundary_check.isChecked() and frame["joints"] is not None:
            valid = frame["joints"][np.isfinite(frame["joints"]).all(axis=1)]
            if len(valid):
                low = np.minimum(low, valid.min(axis=0) - 0.2)
                high = np.maximum(high, valid.max(axis=0) + 0.2)
        margin = np.maximum((high - low) * 0.15, 0.15)
        low, high = low - margin, high + margin
        self.axes.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]))
        for axis in (self.axes.xaxis, self.axes.yaxis, self.axes.zaxis):
            axis.set_major_locator(MaxNLocator(nbins=3))
        self.axes.set_box_aspect(high - low)
        self.axes.view_init(elev=16, azim=-60)
        self.canvas.draw_idle()

    def keyPressEvent(self, event):
        editor = isinstance(self.focusWidget(), (QLineEdit, QAbstractSpinBox))
        if not editor and self.record is not None:
            if event.key() == Qt.Key_Space:
                self.toggle_playback()
                return
            if event.key() in (Qt.Key_Left, Qt.Key_Right):
                self.set_frame(self.frame_index + (1 if event.key() == Qt.Key_Right else -1))
                return
            toggles = {Qt.Key_R: self.radar_check, Qt.Key_K: self.skeleton_check,
                       Qt.Key_B: self.boundary_check, Qt.Key_M: self.images_check}
            if event.key() in toggles:
                toggles[event.key()].toggle()
                return
            if event.key() == Qt.Key_D:
                self.color_mode.setCurrentIndex((self.color_mode.currentIndex() + 1) % self.color_mode.count())
                return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        self._stop_playback()
        if self.worker and self.worker.isRunning():
            self._closing = True
            self.cancel_load()
            event.ignore()
        else:
            event.accept()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, help="Open a complete sequence PKL or numeric NPZ/NPY")
    parser.add_argument("--data-root", type=Path, help="Initial dataset or sequence directory")
    parser.add_argument("--trust-pickle", action="store_true", help="Explicitly trust selected PKLs; pickle can execute code")
    args = parser.parse_args(argv)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    configure_fonts(app)
    window = DatasetViewer(args.data_root, args.trust_pickle)
    window.show()
    if args.file:
        QTimer.singleShot(0, lambda: window.open_file(args.file))
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
