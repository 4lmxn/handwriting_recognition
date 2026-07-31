from unittest.mock import MagicMock, patch

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

from app.gui.tabs.drawing_canvas_tab import DrawingCanvasTab
from recognition.recognizer import RecognitionResult


def _draw_stroke(tab: DrawingCanvasTab) -> None:
    canvas = tab.canvas
    canvas.resize(300, 300)
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=QPoint(20, 20))
    QTest.mouseMove(canvas, pos=QPoint(100, 100))
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=QPoint(100, 100))


def _make_tab(qtbot, app_config) -> DrawingCanvasTab:
    tab = DrawingCanvasTab(app_config.canvas, app_config.paths)
    qtbot.addWidget(tab)
    tab.resize(400, 400)
    return tab


@patch("app.gui.tabs.drawing_canvas_tab.Recognizer")
def test_recognizing_blank_canvas_shows_empty_message_and_skips_recognizer(
    mock_recognizer_cls, qtbot, app_config
):
    tab = _make_tab(qtbot, app_config)
    assert tab.canvas.is_blank()

    tab.recognize_button.click()

    assert "empty" in tab.status_label.text().lower()
    mock_recognizer_cls.assert_not_called()


@patch("app.gui.tabs.drawing_canvas_tab.Recognizer")
def test_recognizing_a_stroke_calls_recognizer_and_updates_label(
    mock_recognizer_cls, qtbot, app_config
):
    mock_instance = MagicMock()
    mock_instance.recognize.return_value = RecognitionResult(text="hello", confidence=0.92)
    mock_recognizer_cls.return_value = mock_instance

    tab = _make_tab(qtbot, app_config)
    _draw_stroke(tab)
    assert not tab.canvas.is_blank()

    tab.recognize_button.click()

    mock_instance.recognize.assert_called_once()
    (image_arg,) = mock_instance.recognize.call_args.args
    assert image_arg.ndim == 2
    assert image_arg.dtype.name == "uint8"

    label_text = tab.status_label.text()
    assert "hello" in label_text
    assert "92%" in label_text
    assert tab.recognize_button.isEnabled()


@patch("app.gui.tabs.drawing_canvas_tab.Recognizer")
def test_recognizer_is_constructed_lazily_and_only_once(mock_recognizer_cls, qtbot, app_config):
    mock_instance = MagicMock()
    mock_instance.recognize.return_value = RecognitionResult(text="hi", confidence=0.5)
    mock_recognizer_cls.return_value = mock_instance

    tab = _make_tab(qtbot, app_config)
    mock_recognizer_cls.assert_not_called()

    _draw_stroke(tab)
    tab.recognize_button.click()
    assert mock_recognizer_cls.call_count == 1

    _draw_stroke(tab)
    tab.recognize_button.click()
    assert mock_recognizer_cls.call_count == 1


@patch("app.gui.tabs.drawing_canvas_tab.Recognizer")
def test_placeholder_text_before_first_recognition(mock_recognizer_cls, qtbot, app_config):
    tab = _make_tab(qtbot, app_config)
    assert "Recognize" in tab.status_label.text()
    mock_recognizer_cls.assert_not_called()
