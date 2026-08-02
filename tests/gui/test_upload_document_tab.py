"""GUI tests for UploadDocumentTab (Phase 6, PR 5).

The heavy dependencies (Recognizer + PageLayout + PageResult) are all
mocked so the tab's wiring can be verified without loading models or
touching a real PDF.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from app.gui.tabs.upload_document_tab import UploadDocumentTab
from documents.inference import PageResult, RecognizedLine, RecognizedWord
from documents.layout import LineLayout, PageLayout, WordBox


def _make_tab(qtbot) -> UploadDocumentTab:
    tab = UploadDocumentTab()
    qtbot.addWidget(tab)
    tab.resize(800, 600)
    return tab


def _fake_page(h: int = 200, w: int = 300) -> np.ndarray:
    return np.full((h, w), 255, dtype=np.uint8)


def _fake_layout() -> PageLayout:
    return PageLayout(
        lines=(
            LineLayout(
                top=10,
                bottom=40,
                words=(
                    WordBox(top=10, bottom=40, left=5, right=30),
                    WordBox(top=10, bottom=40, left=40, right=90),
                ),
            ),
            LineLayout(
                top=60,
                bottom=90,
                words=(WordBox(top=60, bottom=90, left=5, right=50),),
            ),
        ),
        skew_angle=0.0,
    )


def _fake_result() -> PageResult:
    def _word(text: str, box: WordBox) -> RecognizedWord:
        return RecognizedWord(text=text, confidence=0.9, box=box)

    return PageResult(
        lines=(
            RecognizedLine(
                top=10,
                bottom=40,
                words=(
                    _word("hello", WordBox(10, 40, 5, 30)),
                    _word("world", WordBox(10, 40, 40, 90)),
                ),
            ),
            RecognizedLine(
                top=60,
                bottom=90,
                words=(_word("foo", WordBox(60, 90, 5, 50)),),
            ),
        ),
        skew_angle=0.0,
    )


def test_tab_initial_state(qtbot):
    tab = _make_tab(qtbot)
    assert not tab.recognize_button.isEnabled()
    assert not tab.prev_button.isEnabled()
    assert not tab.next_button.isEnabled()
    assert tab.page_count == 0
    assert "Open" in tab.status_label.text()


@patch("app.gui.tabs.upload_document_tab.QFileDialog")
@patch("app.gui.tabs.upload_document_tab.load_image")
def test_open_image_loads_single_page(mock_load_image, mock_dialog, qtbot, tmp_path):
    fake_path = tmp_path / "sample.png"
    fake_path.write_bytes(b"")
    mock_dialog.getOpenFileName.return_value = (str(fake_path), "")
    mock_load_image.return_value = _fake_page()

    tab = _make_tab(qtbot)
    tab.open_button.click()

    assert tab.page_count == 1
    assert tab.recognize_button.isEnabled()
    assert not tab.prev_button.isEnabled()
    assert not tab.next_button.isEnabled()
    assert "Loaded sample.png" in tab.status_label.text()
    assert tab.preview.has_page()


@patch("app.gui.tabs.upload_document_tab.QFileDialog")
@patch("app.gui.tabs.upload_document_tab.load_pdf_pages")
def test_open_pdf_loads_multiple_pages(mock_load_pdf, mock_dialog, qtbot, tmp_path):
    fake_path = tmp_path / "doc.pdf"
    fake_path.write_bytes(b"")
    mock_dialog.getOpenFileName.return_value = (str(fake_path), "")
    mock_load_pdf.return_value = [_fake_page(), _fake_page(), _fake_page()]

    tab = _make_tab(qtbot)
    tab.open_button.click()

    assert tab.page_count == 3
    assert tab.page_index == 0
    assert tab.next_button.isEnabled()
    assert not tab.prev_button.isEnabled()
    assert "3 pages" in tab.status_label.text()


@patch("app.gui.tabs.upload_document_tab.QFileDialog")
@patch("app.gui.tabs.upload_document_tab.load_pdf_pages")
def test_page_navigation(mock_load_pdf, mock_dialog, qtbot, tmp_path):
    fake_path = tmp_path / "doc.pdf"
    fake_path.write_bytes(b"")
    mock_dialog.getOpenFileName.return_value = (str(fake_path), "")
    mock_load_pdf.return_value = [_fake_page(), _fake_page(), _fake_page()]

    tab = _make_tab(qtbot)
    tab.open_button.click()

    tab.next_button.click()
    assert tab.page_index == 1
    assert tab.prev_button.isEnabled()
    assert tab.next_button.isEnabled()

    tab.next_button.click()
    assert tab.page_index == 2
    assert not tab.next_button.isEnabled()

    tab.prev_button.click()
    assert tab.page_index == 1
    tab.prev_button.click()
    assert tab.page_index == 0
    assert not tab.prev_button.isEnabled()


@patch("app.gui.tabs.upload_document_tab.QFileDialog")
@patch("app.gui.tabs.upload_document_tab.QMessageBox")
@patch("app.gui.tabs.upload_document_tab.load_image")
def test_open_failure_surfaces_in_status(
    mock_load_image, mock_msgbox, mock_dialog, qtbot, tmp_path
):
    from documents.loader import InvalidImageError

    fake_path = tmp_path / "corrupt.png"
    fake_path.write_bytes(b"")
    mock_dialog.getOpenFileName.return_value = (str(fake_path), "")
    mock_load_image.side_effect = InvalidImageError("garbage bytes")

    tab = _make_tab(qtbot)
    tab.open_button.click()

    assert not tab.recognize_button.isEnabled()
    assert "corrupt.png" in tab.status_label.text()
    mock_msgbox.warning.assert_called_once()


@patch("app.gui.tabs.upload_document_tab.QFileDialog")
def test_cancelling_open_dialog_leaves_state_untouched(mock_dialog, qtbot):
    mock_dialog.getOpenFileName.return_value = ("", "")
    tab = _make_tab(qtbot)
    tab.open_button.click()
    assert tab.page_count == 0
    assert not tab.recognize_button.isEnabled()


@patch("app.gui.tabs.upload_document_tab.QFileDialog")
@patch("app.gui.tabs.upload_document_tab.load_image")
@patch("app.gui.tabs.upload_document_tab.recognize_page")
@patch("app.gui.tabs.upload_document_tab.analyze_page")
@patch("app.gui.tabs.upload_document_tab.load_feedback_config")
@patch("app.gui.tabs.upload_document_tab.load_recognition_config")
@patch("app.gui.tabs.upload_document_tab.Recognizer")
def test_recognize_runs_pipeline_and_populates_transcript(
    mock_recognizer_cls,
    mock_load_recog,
    mock_load_feedback,
    mock_analyze,
    mock_recognize_page,
    mock_load_image,
    mock_dialog,
    qtbot,
    tmp_path,
):
    fake_path = tmp_path / "sample.png"
    fake_path.write_bytes(b"")
    page = _fake_page()
    mock_dialog.getOpenFileName.return_value = (str(fake_path), "")
    mock_load_image.return_value = page
    mock_analyze.return_value = (page, _fake_layout())
    mock_recognize_page.return_value = _fake_result()

    rec_cfg = MagicMock()
    rec_cfg.model_name = "fake"
    rec_cfg.resolved_device.return_value = "cpu"
    rec_cfg.max_new_tokens = 32
    rec_cfg.repetition_penalty = 1.0
    rec_cfg.no_repeat_ngram_size = 3
    rec_cfg.adapter_path = None
    mock_load_recog.return_value = rec_cfg

    fb_cfg = MagicMock()
    fb_cfg.adapter_dir_path = tmp_path / "adapters"
    mock_load_feedback.return_value = fb_cfg

    mock_recognizer_cls.return_value = MagicMock()

    tab = _make_tab(qtbot)
    tab.open_button.click()
    tab.recognize_button.click()

    mock_analyze.assert_called_once()
    mock_recognize_page.assert_called_once()
    assert tab.transcript.toPlainText() == "hello world\nfoo"
    assert "3 word" in tab.status_label.text()


@patch("app.gui.tabs.upload_document_tab.QFileDialog")
@patch("app.gui.tabs.upload_document_tab.load_pdf_pages")
@patch("app.gui.tabs.upload_document_tab.recognize_page")
@patch("app.gui.tabs.upload_document_tab.analyze_page")
@patch("app.gui.tabs.upload_document_tab.load_feedback_config")
@patch("app.gui.tabs.upload_document_tab.load_recognition_config")
@patch("app.gui.tabs.upload_document_tab.Recognizer")
def test_per_page_results_are_cached_across_navigation(
    mock_recognizer_cls,
    mock_load_recog,
    mock_load_feedback,
    mock_analyze,
    mock_recognize_page,
    mock_load_pdf,
    mock_dialog,
    qtbot,
    tmp_path,
):
    fake_path = tmp_path / "doc.pdf"
    fake_path.write_bytes(b"")
    mock_dialog.getOpenFileName.return_value = (str(fake_path), "")
    mock_load_pdf.return_value = [_fake_page(), _fake_page()]
    mock_analyze.side_effect = lambda page, _cfg: (page, _fake_layout())
    mock_recognize_page.return_value = _fake_result()

    rec_cfg = MagicMock()
    rec_cfg.model_name = "fake"
    rec_cfg.resolved_device.return_value = "cpu"
    rec_cfg.max_new_tokens = 32
    rec_cfg.repetition_penalty = 1.0
    rec_cfg.no_repeat_ngram_size = 3
    rec_cfg.adapter_path = None
    mock_load_recog.return_value = rec_cfg

    fb_cfg = MagicMock()
    fb_cfg.adapter_dir_path = tmp_path / "adapters"
    mock_load_feedback.return_value = fb_cfg

    mock_recognizer_cls.return_value = MagicMock()

    tab = _make_tab(qtbot)
    tab.open_button.click()
    tab.recognize_button.click()  # recognize page 1
    assert mock_recognize_page.call_count == 1

    tab.next_button.click()
    # Navigating to page 2 must NOT re-run recognition on page 1.
    assert mock_recognize_page.call_count == 1
    # But it does clear the transcript since page 2 hasn't been done yet.
    assert tab.transcript.toPlainText() == ""

    tab.prev_button.click()
    # Back to page 1, the cached transcript reappears without running
    # the pipeline again.
    assert mock_recognize_page.call_count == 1
    assert tab.transcript.toPlainText() == "hello world\nfoo"


@patch("app.gui.tabs.upload_document_tab.QFileDialog")
@patch("app.gui.tabs.upload_document_tab.load_image")
@patch("app.gui.tabs.upload_document_tab.recognize_page")
@patch("app.gui.tabs.upload_document_tab.analyze_page")
@patch("app.gui.tabs.upload_document_tab.load_feedback_config")
@patch("app.gui.tabs.upload_document_tab.load_recognition_config")
@patch("app.gui.tabs.upload_document_tab.Recognizer")
def test_recognizer_constructed_lazily_and_once(
    mock_recognizer_cls,
    mock_load_recog,
    mock_load_feedback,
    mock_analyze,
    mock_recognize_page,
    mock_load_image,
    mock_dialog,
    qtbot,
    tmp_path,
):
    fake_path = tmp_path / "sample.png"
    fake_path.write_bytes(b"")
    mock_dialog.getOpenFileName.return_value = (str(fake_path), "")
    page = _fake_page()
    mock_load_image.return_value = page
    mock_analyze.return_value = (page, _fake_layout())
    mock_recognize_page.return_value = _fake_result()

    rec_cfg = MagicMock()
    rec_cfg.model_name = "fake"
    rec_cfg.resolved_device.return_value = "cpu"
    rec_cfg.max_new_tokens = 32
    rec_cfg.repetition_penalty = 1.0
    rec_cfg.no_repeat_ngram_size = 3
    rec_cfg.adapter_path = None
    mock_load_recog.return_value = rec_cfg

    fb_cfg = MagicMock()
    fb_cfg.adapter_dir_path = tmp_path / "adapters"
    mock_load_feedback.return_value = fb_cfg

    mock_recognizer_cls.return_value = MagicMock()

    tab = _make_tab(qtbot)
    mock_recognizer_cls.assert_not_called()

    tab.open_button.click()
    mock_recognizer_cls.assert_not_called()

    tab.recognize_button.click()
    assert mock_recognizer_cls.call_count == 1

    tab.recognize_button.click()
    assert mock_recognizer_cls.call_count == 1


def test_overlay_toggle_propagates_to_preview(qtbot):
    tab = _make_tab(qtbot)
    assert tab.overlay_checkbox.isChecked()
    tab.overlay_checkbox.setChecked(False)
    assert tab.preview._show_overlay is False
    tab.overlay_checkbox.setChecked(True)
    assert tab.preview._show_overlay is True
