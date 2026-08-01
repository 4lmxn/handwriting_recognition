from unittest.mock import MagicMock, patch

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialog

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


def _recognize_stroke(tab: DrawingCanvasTab, text: str = "hello", confidence: float = 0.9) -> None:
    mock_instance = MagicMock()
    mock_instance.recognize.return_value = RecognitionResult(text=text, confidence=confidence)
    with patch(
        "app.gui.tabs.drawing_canvas_tab.Recognizer", return_value=mock_instance
    ):
        _draw_stroke(tab)
        tab.recognize_button.click()


def test_correct_button_disabled_by_default(qtbot, app_config):
    tab = _make_tab(qtbot, app_config)
    assert not tab.correct_button.isEnabled()


def test_correct_button_enables_after_successful_recognition(qtbot, app_config):
    tab = _make_tab(qtbot, app_config)
    _recognize_stroke(tab)
    assert tab.correct_button.isEnabled()


def test_correct_button_disables_when_new_stroke_starts(qtbot, app_config):
    tab = _make_tab(qtbot, app_config)
    _recognize_stroke(tab)
    assert tab.correct_button.isEnabled()

    # Another stroke means the recognized snapshot no longer reflects the
    # canvas; the button must retire until the user recognizes again.
    _draw_stroke(tab)
    assert not tab.correct_button.isEnabled()


def test_correct_button_disables_after_clear(qtbot, app_config):
    tab = _make_tab(qtbot, app_config)
    _recognize_stroke(tab)
    tab._on_clear()
    assert not tab.correct_button.isEnabled()


@patch("app.gui.tabs.drawing_canvas_tab.load_feedback_config")
@patch("app.gui.tabs.drawing_canvas_tab.FeedbackStore")
def test_correct_button_saves_corrected_transcript_to_feedback_store(
    mock_store_cls, mock_load_cfg, qtbot, app_config, tmp_path
):
    cfg = MagicMock()
    cfg.storage_dir_path = tmp_path / "feedback"
    cfg.image_dir_path = tmp_path / "datasets/processed/feedback"
    mock_load_cfg.return_value = cfg
    mock_store = MagicMock()
    mock_store_cls.return_value = mock_store

    tab = _make_tab(qtbot, app_config)
    _recognize_stroke(tab, text="helllo", confidence=0.42)

    # Skip actually showing the dialog — just simulate user typing "hello"
    # and hitting Save.
    with patch(
        "app.gui.tabs.drawing_canvas_tab.CorrectionDialog"
    ) as mock_dialog_cls:
        mock_dialog = MagicMock()
        mock_dialog.exec.return_value = QDialog.DialogCode.Accepted
        mock_dialog.corrected_text.return_value = "hello"
        mock_dialog.DialogCode.Accepted = QDialog.DialogCode.Accepted
        mock_dialog_cls.return_value = mock_dialog
        mock_dialog_cls.DialogCode = QDialog.DialogCode
        tab.correct_button.click()

    mock_store_cls.assert_called_once_with(
        storage_dir=cfg.storage_dir_path,
        image_dir=cfg.image_dir_path,
    )
    mock_store.add.assert_called_once()
    kwargs = mock_store.add.call_args.kwargs
    assert kwargs["prediction"] == "helllo"
    assert kwargs["confidence"] == 0.42
    assert kwargs["corrected"] == "hello"
    assert kwargs["image"].ndim == 2
    assert kwargs["image"].dtype.name == "uint8"

    # After save, the correction is one-shot: button retires until the
    # user recognizes something new.
    assert not tab.correct_button.isEnabled()
    assert "saved" in tab.status_label.text().lower()


@patch("app.gui.tabs.drawing_canvas_tab.load_feedback_config")
@patch("app.gui.tabs.drawing_canvas_tab.FeedbackStore")
def test_cancelling_correction_dialog_does_not_touch_store(
    mock_store_cls, mock_load_cfg, qtbot, app_config, tmp_path
):
    cfg = MagicMock()
    cfg.storage_dir_path = tmp_path / "feedback"
    cfg.image_dir_path = tmp_path / "images"
    mock_load_cfg.return_value = cfg

    tab = _make_tab(qtbot, app_config)
    _recognize_stroke(tab)

    with patch(
        "app.gui.tabs.drawing_canvas_tab.CorrectionDialog"
    ) as mock_dialog_cls:
        mock_dialog = MagicMock()
        mock_dialog.exec.return_value = QDialog.DialogCode.Rejected
        mock_dialog_cls.return_value = mock_dialog
        mock_dialog_cls.DialogCode = QDialog.DialogCode
        tab.correct_button.click()

    mock_store_cls.assert_not_called()
    # Correction wasn't saved, so the snapshot must still be available.
    assert tab.correct_button.isEnabled()


@patch("app.gui.tabs.drawing_canvas_tab.load_feedback_config")
@patch("app.gui.tabs.drawing_canvas_tab.FeedbackStore")
def test_empty_corrected_text_is_ignored(
    mock_store_cls, mock_load_cfg, qtbot, app_config, tmp_path
):
    cfg = MagicMock()
    cfg.storage_dir_path = tmp_path / "feedback"
    cfg.image_dir_path = tmp_path / "images"
    mock_load_cfg.return_value = cfg

    tab = _make_tab(qtbot, app_config)
    _recognize_stroke(tab)

    with patch(
        "app.gui.tabs.drawing_canvas_tab.CorrectionDialog"
    ) as mock_dialog_cls:
        mock_dialog = MagicMock()
        mock_dialog.exec.return_value = QDialog.DialogCode.Accepted
        mock_dialog.corrected_text.return_value = ""
        mock_dialog_cls.return_value = mock_dialog
        mock_dialog_cls.DialogCode = QDialog.DialogCode
        tab.correct_button.click()

    mock_store_cls.assert_not_called()


@patch("app.gui.tabs.drawing_canvas_tab.load_feedback_config")
@patch("app.gui.tabs.drawing_canvas_tab.FeedbackStore")
def test_feedback_store_error_surfaces_in_status_label(
    mock_store_cls, mock_load_cfg, qtbot, app_config, tmp_path
):
    cfg = MagicMock()
    cfg.storage_dir_path = tmp_path / "feedback"
    cfg.image_dir_path = tmp_path / "images"
    mock_load_cfg.return_value = cfg
    mock_store = MagicMock()
    mock_store.add.side_effect = OSError("disk full")
    mock_store_cls.return_value = mock_store

    tab = _make_tab(qtbot, app_config)
    _recognize_stroke(tab)

    with patch(
        "app.gui.tabs.drawing_canvas_tab.CorrectionDialog"
    ) as mock_dialog_cls:
        mock_dialog = MagicMock()
        mock_dialog.exec.return_value = QDialog.DialogCode.Accepted
        mock_dialog.corrected_text.return_value = "hello"
        mock_dialog_cls.return_value = mock_dialog
        mock_dialog_cls.DialogCode = QDialog.DialogCode
        tab.correct_button.click()

    assert "disk full" in tab.status_label.text()
