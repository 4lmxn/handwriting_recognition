"""Freehand drawing canvas: mouse (and stylus, via the same QMouseEvent path)
input rendered to an offscreen QImage, with bounded undo/redo."""

from __future__ import annotations

from collections import deque
from pathlib import Path

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPaintEvent, QPen, QResizeEvent
from PySide6.QtWidgets import QWidget

from app.config import CanvasConfig


class CanvasWidget(QWidget):
    """A resizable freehand drawing surface with undo/redo and grid overlay."""

    strokeFinished = Signal()

    def __init__(self, config: CanvasConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._config = config
        self._pen_width = config.default_pen_width
        self._pen_color = QColor(config.pen_color)
        self._background_color = QColor(config.background_color)
        self._grid_enabled = config.grid_enabled

        self._undo_stack: deque[QImage] = deque(maxlen=config.undo_stack_depth)
        self._redo_stack: deque[QImage] = deque(maxlen=config.undo_stack_depth)

        self._image = QImage(self.size(), QImage.Format.Format_ARGB32_Premultiplied)
        self._image.fill(self._background_color)

        self._drawing = False
        self._last_point = QPoint()

        self.setMinimumSize(200, 200)
        self.setAttribute(Qt.WidgetAttribute.WA_StaticContents)

    # -- Public API -----------------------------------------------------

    def set_pen_width(self, width: int) -> None:
        self._pen_width = max(self._config.min_pen_width, min(width, self._config.max_pen_width))

    def set_pen_color(self, color: QColor) -> None:
        self._pen_color = color

    def set_grid_enabled(self, enabled: bool) -> None:
        self._grid_enabled = enabled
        self.update()

    def is_grid_enabled(self) -> bool:
        return self._grid_enabled

    def pen_width(self) -> int:
        return self._pen_width

    def pen_color(self) -> QColor:
        return self._pen_color

    def get_image(self) -> QImage:
        return self._image.copy()

    def is_blank(self) -> bool:
        return self._image == self._blank_image()

    def save_to_file(self, path: Path) -> bool:
        return self._image.save(str(path))

    def clear(self) -> None:
        if self.is_blank():
            return
        self._push_undo()
        self._image.fill(self._background_color)
        self.update()

    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    def undo(self) -> None:
        if not self._undo_stack:
            return
        self._redo_stack.append(self._image.copy())
        self._image = self._undo_stack.pop()
        self.update()

    def redo(self) -> None:
        if not self._redo_stack:
            return
        self._undo_stack.append(self._image.copy())
        self._image = self._redo_stack.pop()
        self.update()

    # -- Internals --------------------------------------------------------

    def _blank_image(self) -> QImage:
        blank = QImage(self._image.size(), QImage.Format.Format_ARGB32_Premultiplied)
        blank.fill(self._background_color)
        return blank

    def _push_undo(self) -> None:
        self._undo_stack.append(self._image.copy())
        self._redo_stack.clear()

    # -- Qt event handlers --------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._push_undo()
            self._drawing = True
            self._last_point = event.position().toPoint()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drawing and (event.buttons() & Qt.MouseButton.LeftButton):
            current_point = event.position().toPoint()
            painter = QPainter(self._image)
            pen = QPen(
                self._pen_color,
                self._pen_width,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
            painter.setPen(pen)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.drawLine(self._last_point, current_point)
            painter.end()
            self._last_point = current_point
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._drawing:
            self._drawing = False
            self.strokeFinished.emit()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.drawImage(0, 0, self._image)
        if self._grid_enabled:
            self._draw_grid(painter)
        painter.end()

    def _draw_grid(self, painter: QPainter) -> None:
        spacing = self._config.grid_spacing
        pen = QPen(QColor(200, 200, 200, 120))
        pen.setWidth(1)
        painter.setPen(pen)
        for x in range(0, self.width(), spacing):
            painter.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), spacing):
            painter.drawLine(0, y, self.width(), y)

    def resizeEvent(self, event: QResizeEvent) -> None:
        new_size = event.size()
        if new_size.width() <= 0 or new_size.height() <= 0:
            return
        new_image = QImage(new_size, QImage.Format.Format_ARGB32_Premultiplied)
        new_image.fill(self._background_color)
        painter = QPainter(new_image)
        painter.drawImage(0, 0, self._image)
        painter.end()
        self._image = new_image
        super().resizeEvent(event)
