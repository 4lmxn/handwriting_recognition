"""Read-only page preview widget with optional word-box overlay
(Phase 6, PR 5).

Owns a source grayscale ndarray (uploaded image or rendered PDF page),
converts it to a QPixmap once, and repaints it scaled-to-fit on every
paint/resize. When a PageLayout is set, word bounding boxes are drawn
on top in display-space coordinates — computed each paint from the
current scale factor, so a resize doesn't require re-rendering the
pixmap.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRect
from PySide6.QtGui import QColor, QImage, QPainter, QPaintEvent, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from documents.layout import PageLayout


class PagePreviewWidget(QWidget):
    """Displays a page ndarray with optional overlay of layout bboxes."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._layout: PageLayout | None = None
        self._show_overlay = True
        self.setMinimumSize(400, 300)
        # QPalette background: light gray "letterbox" around the page —
        # makes the page's white paper visible against the widget.
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor("#e8e8e8"))
        self.setPalette(palette)

    # -- Public API ------------------------------------------------------

    def set_page(self, page: np.ndarray) -> None:
        """Set the source grayscale page (uint8, HxW). Clears any layout."""
        if page.ndim != 2 or page.dtype != np.uint8:
            raise ValueError(
                "PagePreviewWidget expects (H, W) uint8 grayscale "
                f"(got shape={page.shape} dtype={page.dtype})"
            )
        # Force C-contiguous — QImage from a non-contiguous ndarray
        # (e.g. a slice with non-default strides) reads garbage rows.
        contiguous = np.ascontiguousarray(page)
        height, width = contiguous.shape
        image = QImage(
            contiguous.data,
            width,
            height,
            width,  # bytes per line = width for uint8 grayscale
            QImage.Format.Format_Grayscale8,
        )
        # copy() because the ndarray backing `image` will be garbage-
        # collected the moment this method returns — the QPixmap needs
        # its own buffer.
        self._pixmap = QPixmap.fromImage(image.copy())
        self._layout = None
        self.update()

    def set_layout(self, layout: PageLayout | None) -> None:
        self._layout = layout
        self.update()

    def set_show_overlay(self, on: bool) -> None:
        self._show_overlay = on
        self.update()

    def clear(self) -> None:
        self._pixmap = None
        self._layout = None
        self.update()

    def has_page(self) -> bool:
        return self._pixmap is not None

    # -- Painting --------------------------------------------------------

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 (Qt override)
        if self._pixmap is None:
            return
        painter = QPainter(self)
        try:
            target = self._fit_rect()
            painter.drawPixmap(target, self._pixmap)
            if self._show_overlay and self._layout is not None:
                self._draw_overlay(painter, target)
        finally:
            painter.end()

    def _fit_rect(self) -> QRect:
        assert self._pixmap is not None
        widget_w, widget_h = self.width(), self.height()
        pix_w, pix_h = self._pixmap.width(), self._pixmap.height()
        scale = min(widget_w / pix_w, widget_h / pix_h)
        target_w = max(1, int(pix_w * scale))
        target_h = max(1, int(pix_h * scale))
        x = (widget_w - target_w) // 2
        y = (widget_h - target_h) // 2
        return QRect(x, y, target_w, target_h)

    def _draw_overlay(self, painter: QPainter, target: QRect) -> None:
        assert self._pixmap is not None and self._layout is not None
        scale_x = target.width() / self._pixmap.width()
        scale_y = target.height() / self._pixmap.height()
        pen = QPen(QColor("#e6194b"))
        pen.setWidth(2)
        painter.setPen(pen)
        for line in self._layout.lines:
            for word in line.words:
                x = target.x() + int(word.left * scale_x)
                y = target.y() + int(word.top * scale_y)
                w = max(1, int((word.right - word.left) * scale_x))
                h = max(1, int((word.bottom - word.top) * scale_y))
                painter.drawRect(x, y, w, h)
