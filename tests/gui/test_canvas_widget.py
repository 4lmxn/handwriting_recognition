from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

from app.gui.widgets.canvas_widget import CanvasWidget


def _draw_stroke(widget: CanvasWidget, start: QPoint, end: QPoint) -> None:
    QTest.mousePress(widget, Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(widget, pos=end)
    QTest.mouseRelease(widget, Qt.MouseButton.LeftButton, pos=end)


def test_canvas_starts_blank(qtbot, app_config):
    widget = CanvasWidget(app_config.canvas)
    qtbot.addWidget(widget)
    widget.resize(300, 300)
    assert widget.is_blank()
    assert not widget.can_undo()
    assert not widget.can_redo()


def test_drawing_a_stroke_marks_canvas_dirty(qtbot, app_config):
    widget = CanvasWidget(app_config.canvas)
    qtbot.addWidget(widget)
    widget.resize(300, 300)

    _draw_stroke(widget, QPoint(20, 20), QPoint(100, 100))

    assert not widget.is_blank()
    assert widget.can_undo()
    assert not widget.can_redo()


def test_undo_reverts_last_stroke(qtbot, app_config):
    widget = CanvasWidget(app_config.canvas)
    qtbot.addWidget(widget)
    widget.resize(300, 300)

    _draw_stroke(widget, QPoint(20, 20), QPoint(100, 100))
    assert not widget.is_blank()

    widget.undo()
    assert widget.is_blank()
    assert widget.can_redo()


def test_redo_reapplies_undone_stroke(qtbot, app_config):
    widget = CanvasWidget(app_config.canvas)
    qtbot.addWidget(widget)
    widget.resize(300, 300)

    _draw_stroke(widget, QPoint(20, 20), QPoint(100, 100))
    widget.undo()
    assert widget.is_blank()

    widget.redo()
    assert not widget.is_blank()


def test_clear_blanks_canvas_and_is_undoable(qtbot, app_config):
    widget = CanvasWidget(app_config.canvas)
    qtbot.addWidget(widget)
    widget.resize(300, 300)

    _draw_stroke(widget, QPoint(20, 20), QPoint(100, 100))
    widget.clear()
    assert widget.is_blank()

    widget.undo()
    assert not widget.is_blank()


def test_pen_width_is_clamped_to_config_bounds(qtbot, app_config):
    widget = CanvasWidget(app_config.canvas)
    qtbot.addWidget(widget)

    widget.set_pen_width(app_config.canvas.max_pen_width + 1000)
    assert widget.pen_width() == app_config.canvas.max_pen_width

    widget.set_pen_width(app_config.canvas.min_pen_width - 1000)
    assert widget.pen_width() == app_config.canvas.min_pen_width


def test_resize_preserves_existing_drawing(qtbot, app_config):
    widget = CanvasWidget(app_config.canvas)
    qtbot.addWidget(widget)
    widget.resize(300, 300)

    _draw_stroke(widget, QPoint(20, 20), QPoint(100, 100))
    assert not widget.is_blank()

    widget.resize(500, 500)
    assert not widget.is_blank()
