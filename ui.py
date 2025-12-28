from __future__ import annotations

import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from numba import get_num_threads

from worker import (
    BuddhabrotWorker,
    GenerationParams,
    STAGE_BUDDHABROT,
    STAGE_COMBINE,
    STAGE_IDLE,
    STAGE_MANDELBROT,
    STAGE_REFINE,
    STAGE_SAMPLE_PAIRS,
)


def _apply_colormap(gray: np.ndarray, name: str) -> np.ndarray:
    try:
        import matplotlib.cm as cm

        cmap = cm.get_cmap(name)
        rgb = cmap(gray, bytes=False)[..., :3]
        return rgb
    except Exception:
        if name != "gray":
            return np.stack([gray, gray, gray], axis=-1)

    return np.stack([gray, gray, gray], axis=-1)


def _gaussian_kernel(sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return np.array([1.0], dtype=np.float32)
    radius = max(1, int(3.0 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


def _gaussian_blur_separable(
    image: np.ndarray, sigma_x: float, sigma_y: float
) -> np.ndarray:
    output = image
    if sigma_x > 0.01:
        kernel_x = _gaussian_kernel(sigma_x)
        pad = len(kernel_x) // 2
        padded = np.pad(output, ((pad, pad), (0, 0)), mode="edge")
        output = np.apply_along_axis(
            lambda m: np.convolve(m, kernel_x, mode="valid"), 0, padded
        )
    if sigma_y > 0.01:
        kernel_y = _gaussian_kernel(sigma_y)
        pad = len(kernel_y) // 2
        padded = np.pad(output, ((0, 0), (pad, pad)), mode="edge")
        output = np.apply_along_axis(
            lambda m: np.convolve(m, kernel_y, mode="valid"), 1, padded
        )
    return output


class ImageView(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setCursor(QtCore.Qt.ArrowCursor)
        self.setFocusPolicy(QtCore.Qt.ClickFocus)

        self._base_pixmap: QtGui.QPixmap | None = None
        self._zoom = 1.0
        self._fit_to_view = True
        self._offset = QtCore.QPointF(0, 0)
        self._dragging = False
        self._drag_start = QtCore.QPointF()
        self._offset_start = QtCore.QPointF()
        self._status_text = ""

    def set_base_pixmap(self, pixmap: QtGui.QPixmap, reset_fit: bool = True) -> None:
        self._base_pixmap = pixmap
        if reset_fit:
            self._fit_to_view = True
            self._apply_fit()
        self._status_text = ""
        self.update()

    def set_status(self, text: str) -> None:
        if self._base_pixmap is None:
            self._status_text = text
            self.update()

    def clear(self, text: str = "") -> None:
        self._base_pixmap = None
        self._fit_to_view = True
        self._zoom = 1.0
        self._offset = QtCore.QPointF(0, 0)
        self._status_text = text
        self.update()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if self._base_pixmap is None:
            return

        delta = event.angleDelta().y()
        if delta == 0:
            return

        if self._fit_to_view:
            self._apply_fit()
            self._fit_to_view = False

        factor = 1.25 if delta > 0 else 0.8
        new_zoom = max(0.1, min(20.0, self._zoom * factor))
        if new_zoom == self._zoom:
            return

        pos = event.position()
        before = (pos - self._offset) / self._zoom
        self._zoom = new_zoom
        self._offset = pos - before * self._zoom
        self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            self._dragging = True
            self._drag_start = event.position()
            self._offset_start = QtCore.QPointF(self._offset.x(), self._offset.y())
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._dragging:
            delta = event.position() - self._drag_start
            self._offset = self._offset_start + delta
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            self._dragging = False
            self.setCursor(QtCore.Qt.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._fit_to_view:
            self._apply_fit()
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), self.palette().window())
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, False)

        if self._base_pixmap is None:
            if self._status_text:
                painter.setPen(self.palette().text().color())
                painter.drawText(self.rect(), QtCore.Qt.AlignCenter, self._status_text)
            return

        painter.translate(self._offset)
        painter.scale(self._zoom, self._zoom)
        painter.drawPixmap(0, 0, self._base_pixmap)

    def _fit_scale(self) -> float:
        if self._base_pixmap is None:
            return 1.0
        size = self.size()
        if size.width() == 0 or size.height() == 0:
            return 1.0
        scale_w = size.width() / self._base_pixmap.width()
        scale_h = size.height() / self._base_pixmap.height()
        return min(scale_w, scale_h)

    def _apply_fit(self) -> None:
        if self._base_pixmap is None:
            return
        self._zoom = self._fit_scale()
        width = self._base_pixmap.width() * self._zoom
        height = self._base_pixmap.height() * self._zoom
        self._offset = QtCore.QPointF(
            (self.width() - width) * 0.5,
            (self.height() - height) * 0.5,
        )


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Buddhabrot Explorer")

        self._images_dir = Path("images")
        self._presets_dir = Path("presets")
        self._ensure_save_dirs()

        self._raw_histogram: np.ndarray | None = None
        self._display_buffer: np.ndarray | None = None
        self._display_hist_buffer: np.ndarray | None = None
        self._hist_pixmap: QtGui.QPixmap | None = None
        self._reset_zoom_on_next_image = True

        self._thread: QtCore.QThread | None = None
        self._worker: BuddhabrotWorker | None = None
        self._progress_timer: QtCore.QTimer | None = None
        self._progress_stage: np.ndarray | None = None
        self._progress_steps: np.ndarray | None = None
        self._progress_samples: np.ndarray | None = None
        self._cancel_flag: np.ndarray | None = None
        self._progress_totals: dict[str, int] | None = None

        self._build_ui()
        self._load_presets()

    def _build_ui(self) -> None:
        container = QtWidgets.QWidget()
        root = QtWidgets.QHBoxLayout(container)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setSpacing(8)

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignLeft)

        self.samples_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.samples_slider.setRange(0, 1000)
        self.samples_slider.valueChanged.connect(self._on_samples_changed)

        self.samples_value = QtWidgets.QLabel("50,000")
        self.samples_value.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        samples_box = QtWidgets.QVBoxLayout()
        samples_box.addWidget(QtWidgets.QLabel("Samples"))
        samples_box.addWidget(self.samples_slider)
        samples_box.addWidget(self.samples_value)

        self.nearby_points_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.nearby_points_slider.setRange(0, 1000)
        self.nearby_points_slider.valueChanged.connect(self._on_nearby_points_changed)

        self.nearby_points_value = QtWidgets.QLabel("0")
        self.nearby_points_value.setAlignment(
            QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter
        )

        nearby_points_box = QtWidgets.QVBoxLayout()
        nearby_points_box.addWidget(QtWidgets.QLabel("Nearby points / boundary"))
        nearby_points_box.addWidget(self.nearby_points_slider)
        nearby_points_box.addWidget(self.nearby_points_value)

        self.nearby_distance_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.nearby_distance_slider.setRange(0, 1000)
        self.nearby_distance_slider.valueChanged.connect(
            self._on_nearby_distance_changed
        )

        self.nearby_distance_value = QtWidgets.QLabel("10")
        self.nearby_distance_value.setAlignment(
            QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter
        )

        nearby_distance_box = QtWidgets.QVBoxLayout()
        nearby_distance_box.addWidget(QtWidgets.QLabel("Nearby distance (slider)"))
        nearby_distance_box.addWidget(self.nearby_distance_slider)
        nearby_distance_box.addWidget(self.nearby_distance_value)

        self.iter_spin = QtWidgets.QSpinBox()
        self.iter_spin.setRange(2, 1_000_000)
        self.iter_spin.setValue(1_000)

        self.res_spin = QtWidgets.QSpinBox()
        self.res_spin.setRange(128, 8000)
        self.res_spin.setValue(800)

        self.mandel_res_spin = QtWidgets.QSpinBox()
        self.mandel_res_spin.setRange(6, 8000)
        self.mandel_res_spin.setValue(400)

        self.refine_steps_spin = QtWidgets.QSpinBox()
        self.refine_steps_spin.setRange(0, 10_000)
        self.refine_steps_spin.setValue(80)

        self.soft_points_check = QtWidgets.QCheckBox("Gaussian blur")
        self.soft_points_check.toggled.connect(self._on_soft_points_toggled)

        self.soft_strength_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.soft_strength_slider.setRange(10, 200000)
        self.soft_strength_slider.setValue(200)
        self.soft_strength_slider.valueChanged.connect(self._on_soft_strength_changed)

        self.soft_strength_value = QtWidgets.QLabel("2.00")
        self.soft_strength_value.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        self.soft_strength_widget = QtWidgets.QWidget()
        soft_layout = QtWidgets.QVBoxLayout(self.soft_strength_widget)
        soft_layout.setContentsMargins(0, 0, 0, 0)
        soft_layout.setSpacing(4)
        soft_layout.addWidget(QtWidgets.QLabel("Blur (1/stddev)"))
        soft_layout.addWidget(self.soft_strength_slider)
        soft_layout.addWidget(self.soft_strength_value)

        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setRange(0, 2_147_483_647)
        self.seed_spin.setValue(0)

        self.seed_button = QtWidgets.QPushButton("Random seed")
        self.seed_button.clicked.connect(self._on_random_seed)

        seed_row = QtWidgets.QHBoxLayout()
        seed_row.addWidget(self.seed_spin)
        seed_row.addWidget(self.seed_button)

        self.xmin_spin = QtWidgets.QDoubleSpinBox()
        self.xmin_spin.setRange(-4.0, 4.0)
        self.xmin_spin.setDecimals(4)
        self.xmin_spin.setSingleStep(0.1)
        self.xmin_spin.setValue(-2.0)

        self.xmax_spin = QtWidgets.QDoubleSpinBox()
        self.xmax_spin.setRange(-4.0, 4.0)
        self.xmax_spin.setDecimals(4)
        self.xmax_spin.setSingleStep(0.1)
        self.xmax_spin.setValue(1.0)

        self.ymin_spin = QtWidgets.QDoubleSpinBox()
        self.ymin_spin.setRange(-4.0, 4.0)
        self.ymin_spin.setDecimals(4)
        self.ymin_spin.setSingleStep(0.1)
        self.ymin_spin.setValue(-1.5)

        self.ymax_spin = QtWidgets.QDoubleSpinBox()
        self.ymax_spin.setRange(-4.0, 4.0)
        self.ymax_spin.setDecimals(4)
        self.ymax_spin.setSingleStep(0.1)
        self.ymax_spin.setValue(1.5)

        form.addRow("Max iter", self.iter_spin)
        form.addRow("Resolution", self.res_spin)
        form.addRow("Mandelbrot res", self.mandel_res_spin)
        form.addRow("Refine steps", self.refine_steps_spin)
        form.addRow("Seed", seed_row)
        form.addRow("X min", self.xmin_spin)
        form.addRow("X max", self.xmax_spin)
        form.addRow("Y min", self.ymin_spin)
        form.addRow("Y max", self.ymax_spin)

        self.generate_button = QtWidgets.QPushButton("Generate")
        self.generate_button.clicked.connect(self._start_generation)

        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop_generation)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.currentIndexChanged.connect(self._on_preset_selected)

        self.preset_name_edit = QtWidgets.QLineEdit()
        self.preset_name_edit.setPlaceholderText("Preset name")

        self.save_preset_button = QtWidgets.QPushButton("Save preset")
        self.save_preset_button.clicked.connect(self._save_preset)

        self.save_image_button = QtWidgets.QPushButton("Save image")
        self.save_image_button.clicked.connect(self._save_image)

        preset_layout = QtWidgets.QVBoxLayout()
        preset_layout.addWidget(QtWidgets.QLabel("Presets"))
        preset_layout.addWidget(self.preset_combo)
        preset_layout.addWidget(self.preset_name_edit)
        preset_layout.addWidget(self.save_preset_button)
        preset_layout.addWidget(self.save_image_button)

        left_layout.addLayout(samples_box)
        left_layout.addLayout(nearby_points_box)
        left_layout.addLayout(nearby_distance_box)
        left_layout.addLayout(form)
        left_layout.addWidget(self.generate_button)
        left_layout.addWidget(self.stop_button)
        left_layout.addWidget(self.progress_bar)
        left_layout.addLayout(preset_layout)
        left_layout.addStretch()

        self.image_view = ImageView()
        self.image_view.setMinimumSize(400, 400)
        self.image_view.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.image_view.set_status("Generate a histogram to begin")

        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setSpacing(8)

        post_form = QtWidgets.QFormLayout()

        self.scale_combo = QtWidgets.QComboBox()
        self.scale_combo.addItems(["linear", "nth root", "log"])
        self.scale_combo.currentIndexChanged.connect(self._on_scale_changed)

        self.root_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.root_slider.setRange(110, 2000)
        self.root_slider.setValue(200)
        self.root_slider.valueChanged.connect(self._on_root_changed)

        self.root_value = QtWidgets.QLabel("2.00")
        self.root_value.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        self.root_widget = QtWidgets.QWidget()
        root_layout = QtWidgets.QVBoxLayout(self.root_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(4)
        root_title = QtWidgets.QLabel("Root (n)")
        root_layout.addWidget(root_title)
        root_layout.addWidget(self.root_slider)
        root_layout.addWidget(self.root_value)

        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(
            [
                "gray",
                "hot",
                "magma",
                "viridis",
                "plasma",
                "inferno",
                "cividis",
                "turbo",
                "twilight",
                "cubehelix",
                "Spectral",
                "coolwarm",
                "RdYlBu",
                "YlGnBu",
            ]
        )
        self.cmap_combo.currentIndexChanged.connect(self._update_display)

        self.pre_invert_check = QtWidgets.QCheckBox("Pre-invert")
        self.pre_invert_check.toggled.connect(self._update_display)

        self.post_invert_check = QtWidgets.QCheckBox("Post-invert")
        self.post_invert_check.toggled.connect(self._update_display)

        self.min_clip_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.min_clip_slider.setRange(0, 1000)
        self.min_clip_slider.setValue(0)
        self.min_clip_slider.valueChanged.connect(self._on_clip_changed)

        self.max_clip_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.max_clip_slider.setRange(0, 1000)
        self.max_clip_slider.setValue(1000)
        self.max_clip_slider.valueChanged.connect(self._on_clip_changed)

        self.min_clip_value = QtWidgets.QLabel("0.000")
        self.max_clip_value = QtWidgets.QLabel("1.000")

        self.min_clip_value.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.max_clip_value.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        min_clip_title = QtWidgets.QLabel("Min clip")
        max_clip_title = QtWidgets.QLabel("Max clip")

        min_clip_box = QtWidgets.QVBoxLayout()
        min_clip_box.addWidget(min_clip_title)
        min_clip_box.addWidget(self.min_clip_slider)
        min_clip_box.addWidget(self.min_clip_value)

        max_clip_box = QtWidgets.QVBoxLayout()
        max_clip_box.addWidget(max_clip_title)
        max_clip_box.addWidget(self.max_clip_slider)
        max_clip_box.addWidget(self.max_clip_value)

        post_form.addRow("Scaling", self.scale_combo)
        post_form.addRow("Colormap", self.cmap_combo)
        post_form.addRow("Pre-invert", self.pre_invert_check)
        post_form.addRow("Post-invert", self.post_invert_check)

        self.histogram_label = QtWidgets.QLabel()
        self.histogram_label.setMinimumSize(220, 120)
        self.histogram_label.setAlignment(QtCore.Qt.AlignCenter)
        self.histogram_label.setText("Histogram")

        right_layout.addLayout(post_form)
        right_layout.addWidget(self.soft_points_check)
        right_layout.addWidget(self.soft_strength_widget)
        right_layout.addWidget(self.root_widget)
        right_layout.addLayout(min_clip_box)
        right_layout.addLayout(max_clip_box)
        right_layout.addWidget(self.histogram_label)
        right_layout.addStretch()

        root.addWidget(left_panel)
        root.addWidget(self.image_view, stretch=1)
        root.addWidget(right_panel)

        self.setCentralWidget(container)
        self._set_samples_slider(50_000)
        self._set_nearby_points_slider(0)
        self._set_nearby_distance_slider(1000)
        self._sync_root_visibility()
        self._sync_soft_visibility()

    def _ensure_save_dirs(self) -> None:
        self._images_dir.mkdir(parents=True, exist_ok=True)
        self._presets_dir.mkdir(parents=True, exist_ok=True)

    def _load_presets(self) -> None:
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItem("Select preset...")
        for path in sorted(self._presets_dir.glob("*.json")):
            self.preset_combo.addItem(path.stem, str(path))
        self.preset_combo.blockSignals(False)

    def _on_preset_selected(self) -> None:
        if self.preset_combo.currentIndex() <= 0:
            return
        path_str = self.preset_combo.currentData()
        if not path_str:
            return
        path = Path(path_str)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "Preset error", f"Failed to load preset: {exc}"
            )
            return
        self._apply_preset(payload)

    def _sanitize_preset_name(self, name: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
        cleaned = cleaned.strip("_")
        return cleaned

    def _unique_preset_path(self, name: str) -> Path:
        base = self._sanitize_preset_name(name)
        if not base:
            base = f"preset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        path = self._presets_dir / f"{base}.json"
        if not path.exists():
            return path
        counter = 1
        while True:
            candidate = self._presets_dir / f"{base}_{counter}.json"
            if not candidate.exists():
                return candidate
            counter += 1

    def _save_preset(self) -> None:
        data = self._current_preset_data()
        name = self.preset_name_edit.text().strip()
        path = self._unique_preset_path(name)
        try:
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "Preset error", f"Failed to save preset: {exc}"
            )
            return
        self.preset_name_edit.clear()
        self._load_presets()
        for idx in range(1, self.preset_combo.count()):
            if Path(self.preset_combo.itemData(idx)).resolve() == path.resolve():
                self.preset_combo.setCurrentIndex(idx)
                break

    def _save_image(self) -> None:
        if self.image_view._base_pixmap is None:
            QtWidgets.QMessageBox.information(
                self, "Save image", "No image available to save."
            )
            return
        filename = f"buddhabrot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        path = self._images_dir / filename
        if not self.image_view._base_pixmap.save(str(path)):
            QtWidgets.QMessageBox.warning(
                self, "Save image", "Failed to save the image."
            )

    def _current_preset_data(self) -> dict[str, object]:
        return {
            "samples": self._samples_from_slider(),
            "nearby_points": self._nearby_points_from_slider(),
            "nearby_distance": self._nearby_distance_from_slider(),
            "max_iter": int(self.iter_spin.value()),
            "resolution": int(self.res_spin.value()),
            "mandelbrot_resolution": int(self.mandel_res_spin.value()),
            "refine_steps": int(self.refine_steps_spin.value()),
            "seed": int(self.seed_spin.value()),
            "xmin": float(self.xmin_spin.value()),
            "xmax": float(self.xmax_spin.value()),
            "ymin": float(self.ymin_spin.value()),
            "ymax": float(self.ymax_spin.value()),
            "scale_mode": self.scale_combo.currentText(),
            "root": float(self.root_slider.value() / 100.0),
            "cmap": self.cmap_combo.currentText(),
            "pre_invert": self.pre_invert_check.isChecked(),
            "post_invert": self.post_invert_check.isChecked(),
            "min_clip": self.min_clip_slider.value() / 1000.0,
            "max_clip": self.max_clip_slider.value() / 1000.0,
            "soft_points": self.soft_points_check.isChecked(),
            "soft_strength": int(self.soft_strength_slider.value()),
        }

    def _apply_preset(self, data: dict[str, object]) -> None:
        if "samples" in data:
            self._set_samples_slider(int(data["samples"]))
        if "nearby_points" in data:
            self._set_nearby_points_slider(int(data["nearby_points"]))
        if "nearby_distance" in data:
            self._set_nearby_distance_slider(float(data["nearby_distance"]))
        if "max_iter" in data:
            self.iter_spin.setValue(int(data["max_iter"]))
        if "resolution" in data:
            self.res_spin.setValue(int(data["resolution"]))
        if "mandelbrot_resolution" in data:
            self.mandel_res_spin.setValue(int(data["mandelbrot_resolution"]))
        if "refine_steps" in data:
            self.refine_steps_spin.setValue(int(data["refine_steps"]))
        if "seed" in data:
            self.seed_spin.setValue(int(data["seed"]))
        if "xmin" in data:
            self.xmin_spin.setValue(float(data["xmin"]))
        if "xmax" in data:
            self.xmax_spin.setValue(float(data["xmax"]))
        if "ymin" in data:
            self.ymin_spin.setValue(float(data["ymin"]))
        if "ymax" in data:
            self.ymax_spin.setValue(float(data["ymax"]))
        if "scale_mode" in data:
            self.scale_combo.setCurrentText(str(data["scale_mode"]))
        if "root" in data:
            root_val = float(data["root"])
            self.root_slider.setValue(int(round(root_val * 100.0)))
        if "cmap" in data:
            self.cmap_combo.setCurrentText(str(data["cmap"]))
        if "pre_invert" in data:
            self.pre_invert_check.setChecked(bool(data["pre_invert"]))
        if "post_invert" in data:
            self.post_invert_check.setChecked(bool(data["post_invert"]))
        if "min_clip" in data:
            self.min_clip_slider.setValue(int(round(float(data["min_clip"]) * 1000)))
        if "max_clip" in data:
            self.max_clip_slider.setValue(int(round(float(data["max_clip"]) * 1000)))
        if "soft_points" in data:
            self.soft_points_check.setChecked(bool(data["soft_points"]))
        if "soft_strength" in data:
            self.soft_strength_slider.setValue(int(data["soft_strength"]))
    def _collect_generation_params(self) -> GenerationParams:
        return GenerationParams(
            samples=self._samples_from_slider(),
            max_iter=int(self.iter_spin.value()),
            resolution=int(self.res_spin.value()),
            mandelbrot_resolution=int(self.mandel_res_spin.value()),
            refine_steps=int(self.refine_steps_spin.value()),
            nearby_points_per_boundary=self._nearby_points_from_slider(),
            nearby_distance_value=self._nearby_distance_from_slider(),
            seed=int(self.seed_spin.value()),
            xmin=float(self.xmin_spin.value()),
            xmax=float(self.xmax_spin.value()),
            ymin=float(self.ymin_spin.value()),
            ymax=float(self.ymax_spin.value()),
        )

    def _start_generation(self) -> None:
        if self._thread and self._thread.isRunning():
            return

        params = self._collect_generation_params()
        self.progress_bar.setValue(0)
        self._set_generating(True)
        self._reset_zoom_on_next_image = True
        self.image_view.clear("Generating...")
        self._init_progress_tracking(params)
        self._start_progress_timer()

        self._thread = QtCore.QThread(self)
        self._worker = BuddhabrotWorker(
            params,
            self._progress_stage,
            self._progress_steps,
            self._progress_samples,
            self._cancel_flag,
        )
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.finished.connect(self._on_histogram_ready)
        self._worker.error.connect(self._on_worker_error)
        self._worker.canceled.connect(self._on_worker_canceled)

        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._worker.canceled.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.canceled.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._on_thread_finished)

        self._thread.start()

    def _stop_generation(self) -> None:
        if self._thread and self._thread.isRunning():
            self._thread.requestInterruption()
            self.stop_button.setEnabled(False)
            self.image_view.clear("Stopping...")
            if self._cancel_flag is not None:
                self._cancel_flag[0] = 1

    def _set_generating(self, is_generating: bool) -> None:
        self.generate_button.setEnabled(not is_generating)
        self.stop_button.setEnabled(is_generating)
        if not is_generating:
            self.generate_button.setText("Generate")

    def _on_thread_finished(self) -> None:
        self._set_generating(False)
        self._stop_progress_timer()
        if self._thread is not None:
            self._thread.deleteLater()
        self._thread = None
        self._worker = None
        self._progress_stage = None
        self._progress_steps = None
        self._progress_samples = None
        self._cancel_flag = None
        self._progress_totals = None

    def _on_worker_error(self, message: str) -> None:
        self._set_generating(False)
        self.progress_bar.setValue(0)
        self.image_view.clear("Generation failed")
        QtWidgets.QMessageBox.critical(self, "Generation error", message)

    def _on_worker_canceled(self) -> None:
        self._set_generating(False)
        self.progress_bar.setValue(0)
        self.image_view.clear("Generation stopped")

    def _on_histogram_ready(self, histogram: np.ndarray) -> None:
        self._raw_histogram = histogram.astype(np.float32)
        self._set_generating(False)
        self.progress_bar.setValue(100)
        self._update_display()

    def _init_progress_tracking(self, params: GenerationParams) -> None:
        n_threads = get_num_threads()
        self._progress_stage = np.zeros(1, dtype=np.int32)
        self._progress_steps = np.zeros(1, dtype=np.int32)
        self._progress_samples = np.zeros(n_threads, dtype=np.int64)
        self._cancel_flag = np.zeros(1, dtype=np.uint8)
        total_samples = int(
            params.samples * (1 + max(0, params.nearby_points_per_boundary))
        )
        self._progress_totals = {
            "steps": int(params.refine_steps),
            "samples": total_samples,
        }

    def _start_progress_timer(self) -> None:
        if self._progress_timer is None:
            self._progress_timer = QtCore.QTimer(self)
            self._progress_timer.setInterval(200)
            self._progress_timer.timeout.connect(self._update_generation_progress)
        self._progress_timer.start()

    def _stop_progress_timer(self) -> None:
        if self._progress_timer is not None:
            self._progress_timer.stop()

    def _on_worker_progress(self, value: int) -> None:
        self.progress_bar.setValue(value)

    def _update_generation_progress(self) -> None:
        if self._progress_stage is None or self._progress_totals is None:
            return

        stage = int(self._progress_stage[0])
        stage_name = {
            STAGE_MANDELBROT: "Mandelbrot",
            STAGE_SAMPLE_PAIRS: "Sampling",
            STAGE_REFINE: "Refine",
            STAGE_BUDDHABROT: "Buddhabrot",
            STAGE_COMBINE: "Combine",
        }.get(stage, "Generating")

        fraction = 0.0
        if stage == STAGE_REFINE and self._progress_steps is not None:
            total = self._progress_totals.get("steps", 0)
            done = int(self._progress_steps[0])
            if total > 0:
                fraction = min(1.0, done / total)
        elif stage == STAGE_BUDDHABROT and self._progress_samples is not None:
            total = self._progress_totals.get("samples", 0)
            done = int(self._progress_samples.sum())
            if total > 0:
                fraction = min(1.0, done / total)

        if stage != STAGE_IDLE:
            percent = int(round(fraction * 100))
            self.generate_button.setText(f"{stage_name} {percent}%")
            if stage in (STAGE_REFINE, STAGE_BUDDHABROT):
                ranges = {
                    STAGE_MANDELBROT: (0.0, 25.0),
                    STAGE_SAMPLE_PAIRS: (25.0, 45.0),
                    STAGE_REFINE: (45.0, 65.0),
                    STAGE_BUDDHABROT: (65.0, 90.0),
                    STAGE_COMBINE: (90.0, 100.0),
                }
                start, end = ranges.get(stage, (0.0, 100.0))
                overall = start + (end - start) * fraction
                self.progress_bar.setValue(int(round(overall)))

    def _update_display(self) -> None:
        if self._raw_histogram is None:
            return

        raw = self._raw_histogram
        processed = raw
        if self.soft_points_check.isChecked():
            inv_std = self.soft_strength_slider.value() / 100.0
            std_units = 1.0 / max(inv_std, 1e-6)
            height, width = processed.shape
            dx = (self.xmax_spin.value() - self.xmin_spin.value()) / width
            dy = (self.ymax_spin.value() - self.ymin_spin.value()) / height
            sigma_x = std_units / dx
            sigma_y = std_units / dy
            processed = _gaussian_blur_separable(processed, sigma_x, sigma_y)

        scaled = processed
        if self.pre_invert_check.isChecked():
            max_raw = float(processed.max())
            if max_raw > 0:
                scaled = max_raw - processed

        scale_mode = self.scale_combo.currentText()
        if scale_mode == "nth root":
            root = max(1.1, self.root_slider.value() / 100.0)
            scaled = np.power(scaled, 1.0 / root)
        elif scale_mode == "log":
            scaled = np.log1p(scaled)

        max_val = float(scaled.max())
        if max_val <= 0:
            return

        normalized = scaled / max_val
        min_clip, max_clip = self._clip_values()
        if max_clip <= min_clip:
            max_clip = min_clip + 1e-6

        normalized = (normalized - min_clip) / (max_clip - min_clip)
        normalized = np.clip(normalized, 0.0, 1.0)
        if self.post_invert_check.isChecked():
            normalized = 1.0 - normalized

        # transpose so width/height map to image coordinates
        img = normalized
        rgb = _apply_colormap(img, self.cmap_combo.currentText())
        rgb_uint8 = (rgb * 255).astype(np.uint8)

        self._display_buffer = rgb_uint8
        height, width, _ = rgb_uint8.shape
        image = QtGui.QImage(
            rgb_uint8.data,
            width,
            height,
            3 * width,
            QtGui.QImage.Format_RGB888,
        )
        self.image_view.set_base_pixmap(
            QtGui.QPixmap.fromImage(image),
            reset_fit=self._reset_zoom_on_next_image,
        )
        self._reset_zoom_on_next_image = False
        self._update_histogram_image(scaled, max_val)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._refresh_histogram_pixmap()

    def _samples_from_slider(self) -> int:
        min_samples = 1
        max_samples = 5_000_000
        t = self.samples_slider.value() / 1000.0
        log_min = math.log10(min_samples)
        log_max = math.log10(max_samples)
        value = 10 ** (log_min + t * (log_max - log_min))
        return int(max(min_samples, min(max_samples, round(value))))

    def _set_samples_slider(self, samples: int) -> None:
        min_samples = 1
        max_samples = 5_000_000
        samples = int(max(min_samples, min(max_samples, samples)))
        log_min = math.log10(min_samples)
        log_max = math.log10(max_samples)
        t = (math.log10(samples) - log_min) / (log_max - log_min)
        self.samples_slider.setValue(int(round(t * 1000)))
        self.samples_value.setText(f"{samples:,}")

    def _on_samples_changed(self) -> None:
        samples = self._samples_from_slider()
        self.samples_value.setText(f"{samples:,}")

    def _nearby_points_from_slider(self) -> int:
        raw = self.nearby_points_slider.value()
        if raw <= 0:
            return 0
        t = raw / 1000.0
        log_min = math.log10(1)
        log_max = math.log10(1000)
        value = 10 ** (log_min + t * (log_max - log_min))
        return int(max(1, min(1000, round(value))))

    def _set_nearby_points_slider(self, points: int) -> None:
        points = int(max(0, min(1000, points)))
        if points == 0:
            self.nearby_points_slider.setValue(0)
            self.nearby_points_value.setText("0")
            return
        log_min = math.log10(1)
        log_max = math.log10(1000)
        t = (math.log10(points) - log_min) / (log_max - log_min)
        self.nearby_points_slider.setValue(int(round(t * 1000)))
        self.nearby_points_value.setText(f"{points}")

    def _on_nearby_points_changed(self) -> None:
        points = self._nearby_points_from_slider()
        self.nearby_points_value.setText(f"{points}")

    def _nearby_distance_from_slider(self) -> float:
        t = self.nearby_distance_slider.value() / 1000.0
        log_min = math.log10(10.0)
        log_max = math.log10(10000.0)
        value = 10 ** (log_min + t * (log_max - log_min))
        return float(max(10.0, min(10000.0, value)))

    def _set_nearby_distance_slider(self, value: float) -> None:
        value = float(max(10.0, min(10000.0, value)))
        log_min = math.log10(10.0)
        log_max = math.log10(10000.0)
        t = (math.log10(value) - log_min) / (log_max - log_min)
        self.nearby_distance_slider.setValue(int(round(t * 1000)))
        self.nearby_distance_value.setText(f"{int(round(value))}")

    def _on_nearby_distance_changed(self) -> None:
        value = self._nearby_distance_from_slider()
        self.nearby_distance_value.setText(f"{int(round(value))}")

    def _on_soft_points_toggled(self) -> None:
        self._sync_soft_visibility()
        self._update_display()

    def _on_soft_strength_changed(self) -> None:
        inv_std = self.soft_strength_slider.value() / 100.0
        self.soft_strength_value.setText(f"{inv_std:.2f}")
        self._update_display()

    def _sync_soft_visibility(self) -> None:
        is_soft = self.soft_points_check.isChecked()
        self.soft_strength_widget.setVisible(is_soft)

    def _on_random_seed(self) -> None:
        self.seed_spin.setValue(random.randrange(0, 2_147_483_647))

    def _on_scale_changed(self) -> None:
        self._sync_root_visibility()
        self._update_display()

    def _on_root_changed(self) -> None:
        self.root_value.setText(f"{self.root_slider.value() / 100:.2f}")
        self._update_display()

    def _sync_root_visibility(self) -> None:
        is_root = self.scale_combo.currentText() == "nth root"
        self.root_widget.setVisible(is_root)

    def _on_clip_changed(self) -> None:
        min_val = self.min_clip_slider.value()
        max_val = self.max_clip_slider.value()
        if min_val > max_val:
            if self.sender() is self.min_clip_slider:
                self.max_clip_slider.setValue(min_val)
            else:
                self.min_clip_slider.setValue(max_val)
            return

        self.min_clip_value.setText(f"{min_val / 1000:.3f}")
        self.max_clip_value.setText(f"{max_val / 1000:.3f}")
        self._update_display()

    def _clip_values(self) -> tuple[float, float]:
        min_clip = self.min_clip_slider.value() / 1000.0
        max_clip = self.max_clip_slider.value() / 1000.0
        return min_clip, max_clip

    def _update_histogram_image(self, scaled: np.ndarray, max_val: float) -> None:
        if self._raw_histogram is None:
            return

        if max_val <= 0:
            return

        normalized = (scaled / max_val).ravel()
        bins = 256
        counts, _ = np.histogram(normalized, bins=bins, range=(0.0, 1.0))
        if counts.max() == 0:
            return

        width = 256
        height = 120
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = 20

        counts = counts.astype(np.float32)
        counts = np.log1p(counts)
        counts /= counts.max()
        for x in range(width):
            h = int(counts[x] * (height - 1))
            if h <= 0:
                continue
            img[height - h : height, x] = (220, 220, 220)

        min_clip, max_clip = self._clip_values()
        min_x = int(min_clip * (width - 1))
        max_x = int(max_clip * (width - 1))
        img[:, min_x : min_x + 2] = (50, 200, 255)
        img[:, max_x : max_x + 2] = (255, 160, 80)

        self._display_hist_buffer = img
        hist_image = QtGui.QImage(
            img.data,
            width,
            height,
            3 * width,
            QtGui.QImage.Format_RGB888,
        )
        self._hist_pixmap = QtGui.QPixmap.fromImage(hist_image)
        self._refresh_histogram_pixmap()

    def _refresh_histogram_pixmap(self) -> None:
        if not self._hist_pixmap:
            return
        target = self.histogram_label.size()
        self.histogram_label.setPixmap(
            self._hist_pixmap.scaled(
                target,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        )


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.resize(1200, 700)
    window.show()
    sys.exit(app.exec())
