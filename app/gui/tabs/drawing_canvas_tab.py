"""Drawing Canvas tab: pen controls + freehand canvas + recognition.

The "Recognize" button converts the canvas to a grayscale numpy array and
runs it through `recognition.recognizer.Recognizer`. The recognizer is heavy
to construct (loads an ML model), so it is created lazily on first use
rather than at tab construction time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from app.config import CanvasConfig, PathsConfig
from app.gui.widgets.canvas_widget import CanvasWidget
from app.gui.widgets.correction_dialog import CorrectionDialog
from feedback.config import load_feedback_config
from feedback.store import FeedbackStore
from language_model.rescoring import RescoringRecognizer, wrap_if_enabled
from models.adapters.resolver import resolve_adapter_path
from recognition.config import load_recognition_config
from recognition.recognizer import RecognitionResult, Recognizer

_PLACEHOLDER_TEXT = "Draw something and click Recognize."
_EMPTY_CANVAS_TEXT = "Canvas is empty — draw something first."
_RECOGNIZING_TEXT = "Recognizing…"
_CORRECTION_SAVED_TEXT = "Correction saved."
_CORRECTION_FAILED_TEXT = "Could not save correction: {error}"


class DrawingCanvasTab(QWidget):
    def __init__(
        self, canvas_config: CanvasConfig, paths_config: PathsConfig, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._paths_config = paths_config
        self._canvas = CanvasWidget(canvas_config)
        self._recognizer: Recognizer | RescoringRecognizer | None = None
        self._feedback_store: FeedbackStore | None = None
        # Snapshot of the last recognition — what the model saw and what it
        # returned. Kept so "Correct…" writes the exact image that was
        # recognized, not whatever the user has drawn on top since.
        self._last_image: np.ndarray | None = None
        self._last_result: RecognitionResult | None = None

        layout = QVBoxLayout(self)
        layout.addLayout(self._build_toolbar(canvas_config))
        layout.addWidget(self._canvas, stretch=1)
        layout.addWidget(self._build_status_bar())

        self._canvas.strokeFinished.connect(self._on_stroke_finished)
        self._install_shortcuts()

    # -- UI construction ----------------------------------------------------

    def _build_toolbar(self, canvas_config: CanvasConfig) -> QHBoxLayout:
        toolbar = QHBoxLayout()

        toolbar.addWidget(QLabel("Pen width:"))
        self._width_slider = QSlider(Qt.Orientation.Horizontal)
        self._width_slider.setMinimum(canvas_config.min_pen_width)
        self._width_slider.setMaximum(canvas_config.max_pen_width)
        self._width_slider.setValue(canvas_config.default_pen_width)
        self._width_slider.setFixedWidth(150)
        self._width_slider.valueChanged.connect(self._on_width_changed)
        toolbar.addWidget(self._width_slider)
        self._width_value_label = QLabel(str(canvas_config.default_pen_width))
        toolbar.addWidget(self._width_value_label)

        color_button = QPushButton("Pen color")
        color_button.clicked.connect(self._on_pick_color)
        toolbar.addWidget(color_button)

        grid_checkbox = QCheckBox("Grid")
        grid_checkbox.setChecked(canvas_config.grid_enabled)
        grid_checkbox.toggled.connect(self._canvas.set_grid_enabled)
        toolbar.addWidget(grid_checkbox)

        toolbar.addStretch(1)

        self._undo_button = QPushButton("Undo")
        self._undo_button.clicked.connect(self._on_undo)
        self._undo_button.setEnabled(False)
        toolbar.addWidget(self._undo_button)

        self._redo_button = QPushButton("Redo")
        self._redo_button.clicked.connect(self._on_redo)
        self._redo_button.setEnabled(False)
        toolbar.addWidget(self._redo_button)

        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self._on_clear)
        toolbar.addWidget(clear_button)

        save_button = QPushButton("Save…")
        save_button.clicked.connect(self._on_save)
        toolbar.addWidget(save_button)

        self._recognize_button = QPushButton("Recognize")
        self._recognize_button.clicked.connect(self._on_recognize)
        toolbar.addWidget(self._recognize_button)

        self._correct_button = QPushButton("Correct…")
        self._correct_button.clicked.connect(self._on_correct)
        self._correct_button.setEnabled(False)
        self._correct_button.setToolTip(
            "Save a corrected transcript for the last recognition."
        )
        toolbar.addWidget(self._correct_button)

        return toolbar

    def _build_status_bar(self) -> QLabel:
        self._status_label = QLabel(_PLACEHOLDER_TEXT)
        self._status_label.setStyleSheet("color: gray; font-style: italic;")
        return self._status_label

    def _install_shortcuts(self) -> None:
        undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        undo_shortcut.activated.connect(self._on_undo)
        redo_shortcut = QShortcut(QKeySequence("Ctrl+Shift+Z"), self)
        redo_shortcut.activated.connect(self._on_redo)

    # -- Slots ----------------------------------------------------------

    def _on_width_changed(self, value: int) -> None:
        self._canvas.set_pen_width(value)
        self._width_value_label.setText(str(value))

    def _on_pick_color(self) -> None:
        color = QColorDialog.getColor(self._canvas.pen_color(), self, "Select pen color")
        if color.isValid():
            self._canvas.set_pen_color(color)

    def _on_clear(self) -> None:
        self._canvas.clear()
        self._refresh_undo_redo_buttons()
        self._invalidate_last_recognition()

    def _on_undo(self) -> None:
        self._canvas.undo()
        self._refresh_undo_redo_buttons()

    def _on_redo(self) -> None:
        self._canvas.redo()
        self._refresh_undo_redo_buttons()

    def _on_save(self) -> None:
        self._paths_config.outputs.mkdir(parents=True, exist_ok=True)
        default_path = str(self._paths_config.outputs / "canvas.png")
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save canvas", default_path, "PNG Image (*.png)"
        )
        if file_path:
            self._canvas.save_to_file(Path(file_path))

    def _on_recognize(self) -> None:
        if self._canvas.is_blank():
            self._status_label.setText(_EMPTY_CANVAS_TEXT)
            return

        self._recognize_button.setEnabled(False)
        self._status_label.setText(_RECOGNIZING_TEXT)
        # Force the "Recognizing…" state to actually paint before the
        # (synchronous) recognizer call blocks the event loop.
        QApplication.processEvents()
        try:
            if self._recognizer is None:
                recognition_config = load_recognition_config()
                # Resolve the (optional) personalization adapter here rather
                # than inside Recognizer so this module owns the coupling
                # between recognition.yaml and feedback.yaml — Recognizer
                # itself stays feedback-config-unaware.
                feedback_config = load_feedback_config()
                adapter_path = resolve_adapter_path(
                    recognition_config.adapter_path,
                    feedback_config.adapter_dir_path,
                )
                base = Recognizer(
                    recognition_config.model_name,
                    device=recognition_config.resolved_device(),
                    max_new_tokens=recognition_config.max_new_tokens,
                    repetition_penalty=recognition_config.repetition_penalty,
                    no_repeat_ngram_size=recognition_config.no_repeat_ngram_size,
                    adapter_path=adapter_path,
                )
                # No-op when configs/language_model.yaml disables
                # rescoring (the default) — flipping that flag is the
                # only change needed to activate LM-assisted decoding.
                self._recognizer = wrap_if_enabled(base)
            image = self._canvas_to_grayscale_array()
            result = self._recognizer.recognize(image)
            self._last_image = image
            self._last_result = result
            self._correct_button.setEnabled(True)
            self._status_label.setText(
                f"Recognized: {result.text} ({result.confidence:.0%} confidence)"
            )
        finally:
            self._recognize_button.setEnabled(True)

    def _on_correct(self) -> None:
        if self._last_image is None or self._last_result is None:
            return
        dialog = CorrectionDialog(
            prediction=self._last_result.text,
            confidence=self._last_result.confidence,
            parent=self,
        )
        if dialog.exec() != CorrectionDialog.DialogCode.Accepted:
            return
        corrected = dialog.corrected_text()
        if not corrected:
            return
        self._save_correction(corrected)

    def _save_correction(self, corrected: str) -> None:
        # Guarded by _on_correct; asserted here so mypy/readers see the
        # invariant that _last_image / _last_result are populated whenever
        # this method runs.
        assert self._last_image is not None and self._last_result is not None
        try:
            if self._feedback_store is None:
                self._feedback_store = self._build_feedback_store()
            self._feedback_store.add(
                image=self._last_image,
                prediction=self._last_result.text,
                confidence=self._last_result.confidence,
                corrected=corrected,
            )
        except OSError as exc:
            self._status_label.setText(_CORRECTION_FAILED_TEXT.format(error=exc))
            return
        # Correction is single-shot: after saving, the same recognition
        # can't be corrected twice (would produce duplicate records).
        self._invalidate_last_recognition()
        self._status_label.setText(_CORRECTION_SAVED_TEXT)

    def _build_feedback_store(self) -> FeedbackStore:
        cfg = load_feedback_config()
        return FeedbackStore(
            storage_dir=cfg.storage_dir_path,
            image_dir=cfg.image_dir_path,
        )

    def _invalidate_last_recognition(self) -> None:
        self._last_image = None
        self._last_result = None
        self._correct_button.setEnabled(False)

    def _on_stroke_finished(self) -> None:
        self._refresh_undo_redo_buttons()
        # A new stroke means the canvas no longer matches what was last
        # recognized — offering "Correct…" against a stale prediction
        # would tag the wrong image, so retire it.
        self._invalidate_last_recognition()

    def _refresh_undo_redo_buttons(self) -> None:
        self._undo_button.setEnabled(self._canvas.can_undo())
        self._redo_button.setEnabled(self._canvas.can_redo())

    # -- Recognition helpers -------------------------------------------------

    def _canvas_to_grayscale_array(self) -> np.ndarray:
        return qimage_to_grayscale_array(self._canvas.get_image())

    # -- Exposed for tests -------------------------------------------------

    @property
    def canvas(self) -> CanvasWidget:
        return self._canvas

    @property
    def status_label(self) -> QLabel:
        return self._status_label

    @property
    def recognize_button(self) -> QPushButton:
        return self._recognize_button

    @property
    def correct_button(self) -> QPushButton:
        return self._correct_button


def qimage_to_grayscale_array(image: QImage) -> np.ndarray:
    """Convert a QImage to a grayscale (H, W) uint8 numpy array, ink-dark-on-
    light-background — matching the convention used everywhere else in the
    project (see preprocessing/image_ops.py)."""
    gray_image = image.convertToFormat(QImage.Format.Format_Grayscale8)
    width = gray_image.width()
    height = gray_image.height()
    bytes_per_line = gray_image.bytesPerLine()

    buffer = gray_image.constBits()
    raw = np.frombuffer(buffer, dtype=np.uint8, count=bytes_per_line * height)
    padded = raw.reshape(height, bytes_per_line)
    return np.ascontiguousarray(padded[:, :width])
