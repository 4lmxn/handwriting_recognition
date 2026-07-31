"""Drawing Canvas tab: pen controls + freehand canvas.

Recognition is not wired up yet (see docs/ROADMAP.md, Phase 3 onward) — the
status label below the canvas says so explicitly rather than faking a result.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
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


class DrawingCanvasTab(QWidget):
    def __init__(
        self, canvas_config: CanvasConfig, paths_config: PathsConfig, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._paths_config = paths_config
        self._canvas = CanvasWidget(canvas_config)

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

        return toolbar

    def _build_status_bar(self) -> QLabel:
        self._status_label = QLabel(
            "Recognition is not implemented yet — this canvas currently only captures strokes. "
            "See docs/ROADMAP.md."
        )
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

    def _on_stroke_finished(self) -> None:
        self._refresh_undo_redo_buttons()

    def _refresh_undo_redo_buttons(self) -> None:
        self._undo_button.setEnabled(self._canvas.can_undo())
        self._redo_button.setEnabled(self._canvas.can_redo())

    # -- Exposed for tests -------------------------------------------------

    @property
    def canvas(self) -> CanvasWidget:
        return self._canvas
