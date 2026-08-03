"""Upload Document tab: load an image or PDF, recognize a full page.

Wires PR 1-4 into a single interactive surface:

    file open  →  documents.loader.load_image / pdf_loader.load_pdf_pages
    Recognize  →  documents.layout.analyze_page
                  documents.inference.recognize_page

The tab keeps the same lazy-Recognizer + config-loading pattern as the
drawing tab; each tab currently has its own Recognizer instance
(TrOCR-small loads twice if both tabs recognize this session). Sharing
one Recognizer across tabs is a legitimate cleanup but touches
MainWindow and is intentionally out of scope for this PR.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.gui.widgets.page_preview_widget import PagePreviewWidget
from documents.config import DocumentsConfig, load_documents_config
from documents.inference import PageResult, recognize_page
from documents.layout import PageLayout, analyze_page
from documents.loader import (
    ImageTooLargeError,
    InvalidImageError,
    UnsupportedImageFormatError,
    load_image,
)
from documents.pdf_loader import (
    InvalidPdfError,
    PdfTooLongError,
    load_pdf_pages,
)
from feedback.config import load_feedback_config
from language_model.rescoring import RescoringRecognizer, wrap_if_enabled
from models.adapters.resolver import resolve_adapter_path
from recognition.config import load_recognition_config
from recognition.recognizer import Recognizer

_PLACEHOLDER_TEXT = "Open an image or PDF to begin."
_LOADING_TEXT = "Loading {name}…"
_RECOGNIZING_TEXT = "Recognizing page {index} of {total}…"
_TRANSCRIPT_PLACEHOLDER = "Recognized transcript will appear here."

_IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.tif *.tiff)"
_PDF_FILTER = "PDF (*.pdf)"
_ALL_FILTER = "All supported (*.png *.jpg *.jpeg *.tif *.tiff *.pdf)"


class UploadDocumentTab(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._documents_config: DocumentsConfig = load_documents_config()
        self._recognizer: Recognizer | RescoringRecognizer | None = None

        self._pages: list[np.ndarray] = []
        self._page_index = 0
        # Per-page analysis / recognition cache. Same length as _pages
        # once populated; None entries mean "not yet analyzed for this
        # page". Cached so re-navigating to a page shows its previous
        # overlay/transcript without re-running the model.
        self._layouts: list[PageLayout | None] = []
        self._results: list[PageResult | None] = []
        self._working_pages: list[np.ndarray | None] = []

        self._build_ui()

    # -- UI construction --------------------------------------------------

    def _build_ui(self) -> None:
        # Preview constructed first so _build_toolbar's overlay checkbox
        # can wire directly into it.
        self._preview = PagePreviewWidget()
        self._transcript = QPlainTextEdit()
        self._transcript.setReadOnly(True)
        self._transcript.setPlaceholderText(_TRANSCRIPT_PLACEHOLDER)

        layout = QVBoxLayout(self)
        layout.addLayout(self._build_toolbar())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._preview)
        splitter.addWidget(self._transcript)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, stretch=1)

        layout.addWidget(self._build_status_bar())

    def _build_toolbar(self) -> QHBoxLayout:
        toolbar = QHBoxLayout()

        self._open_button = QPushButton("Open…")
        self._open_button.clicked.connect(self._on_open)
        toolbar.addWidget(self._open_button)

        self._prev_button = QPushButton("◀ Prev")
        self._prev_button.clicked.connect(self._on_prev_page)
        self._prev_button.setEnabled(False)
        toolbar.addWidget(self._prev_button)

        self._page_label = QLabel("—")
        self._page_label.setMinimumWidth(80)
        self._page_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        toolbar.addWidget(self._page_label)

        self._next_button = QPushButton("Next ▶")
        self._next_button.clicked.connect(self._on_next_page)
        self._next_button.setEnabled(False)
        toolbar.addWidget(self._next_button)

        toolbar.addStretch(1)

        self._overlay_checkbox = QCheckBox("Show word boxes")
        self._overlay_checkbox.setChecked(True)
        self._overlay_checkbox.toggled.connect(self._preview.set_show_overlay)
        toolbar.addWidget(self._overlay_checkbox)

        self._recognize_button = QPushButton("Recognize page")
        self._recognize_button.clicked.connect(self._on_recognize)
        self._recognize_button.setEnabled(False)
        toolbar.addWidget(self._recognize_button)

        return toolbar

    def _build_status_bar(self) -> QLabel:
        self._status_label = QLabel(_PLACEHOLDER_TEXT)
        self._status_label.setStyleSheet("color: gray; font-style: italic;")
        return self._status_label

    # -- Slots ------------------------------------------------------------

    def _on_open(self) -> None:
        file_path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Open document",
            "",
            f"{_ALL_FILTER};;{_IMAGE_FILTER};;{_PDF_FILTER}",
        )
        if not file_path_str:
            return
        path = Path(file_path_str)
        self._status_label.setText(_LOADING_TEXT.format(name=path.name))
        QApplication.processEvents()
        try:
            pages = self._load_document(path)
        except (
            UnsupportedImageFormatError,
            ImageTooLargeError,
            InvalidImageError,
            PdfTooLongError,
            InvalidPdfError,
            FileNotFoundError,
        ) as exc:
            self._status_label.setText(f"Could not open {path.name}: {exc}")
            QMessageBox.warning(self, "Open failed", str(exc))
            return
        if not pages:
            self._status_label.setText(f"{path.name} contains no pages.")
            return
        self._set_pages(pages, source_name=path.name)

    def _load_document(self, path: Path) -> list[np.ndarray]:
        if path.suffix.lower() == ".pdf":
            return load_pdf_pages(path, self._documents_config)
        return [load_image(path, self._documents_config)]

    def _set_pages(self, pages: list[np.ndarray], source_name: str) -> None:
        self._pages = pages
        self._layouts = [None] * len(pages)
        self._results = [None] * len(pages)
        self._working_pages = [None] * len(pages)
        self._page_index = 0
        self._recognize_button.setEnabled(True)
        self._refresh_page_display()
        self._transcript.clear()
        self._status_label.setText(
            f"Loaded {source_name} ({len(pages)} page{'s' if len(pages) != 1 else ''})."
        )

    def _on_prev_page(self) -> None:
        if self._page_index > 0:
            self._page_index -= 1
            self._refresh_page_display()

    def _on_next_page(self) -> None:
        if self._page_index < len(self._pages) - 1:
            self._page_index += 1
            self._refresh_page_display()

    def _on_recognize(self) -> None:
        if not self._pages:
            return
        idx = self._page_index
        self._recognize_button.setEnabled(False)
        self._status_label.setText(
            _RECOGNIZING_TEXT.format(index=idx + 1, total=len(self._pages))
        )
        QApplication.processEvents()
        try:
            working_page, layout = analyze_page(
                self._pages[idx], self._documents_config.layout
            )
            if self._recognizer is None:
                self._recognizer = self._build_recognizer()
            result = recognize_page(working_page, layout, self._recognizer)
            self._working_pages[idx] = working_page
            self._layouts[idx] = layout
            self._results[idx] = result
            self._refresh_page_display()
            self._status_label.setText(
                f"Page {idx + 1}: {sum(len(line.words) for line in result.lines)} "
                f"word(s) recognized."
            )
        finally:
            self._recognize_button.setEnabled(True)

    def _build_recognizer(self) -> Recognizer | RescoringRecognizer:
        recognition_config = load_recognition_config()
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
        # No-op when configs/language_model.yaml disables rescoring
        # (the default) — flipping that flag is the only change needed
        # to activate LM-assisted decoding.
        return wrap_if_enabled(base)

    # -- Display ----------------------------------------------------------

    def _refresh_page_display(self) -> None:
        if not self._pages:
            self._preview.clear()
            self._page_label.setText("—")
            self._prev_button.setEnabled(False)
            self._next_button.setEnabled(False)
            return
        idx = self._page_index
        # Prefer the working (deskewed) page for display if we've already
        # analyzed this page — the overlay's coordinates reference it.
        # `or` on ndarrays raises ValueError, so `is None` explicitly.
        working = self._working_pages[idx]
        display_page = working if working is not None else self._pages[idx]
        self._preview.set_page(display_page)
        self._preview.set_layout(self._layouts[idx])
        self._page_label.setText(f"Page {idx + 1} / {len(self._pages)}")
        self._prev_button.setEnabled(idx > 0)
        self._next_button.setEnabled(idx < len(self._pages) - 1)
        result = self._results[idx]
        self._transcript.setPlainText(result.text if result is not None else "")

    # -- Exposed for tests -------------------------------------------------

    @property
    def preview(self) -> PagePreviewWidget:
        return self._preview

    @property
    def transcript(self) -> QPlainTextEdit:
        return self._transcript

    @property
    def status_label(self) -> QLabel:
        return self._status_label

    @property
    def open_button(self) -> QPushButton:
        return self._open_button

    @property
    def prev_button(self) -> QPushButton:
        return self._prev_button

    @property
    def next_button(self) -> QPushButton:
        return self._next_button

    @property
    def recognize_button(self) -> QPushButton:
        return self._recognize_button

    @property
    def overlay_checkbox(self) -> QCheckBox:
        return self._overlay_checkbox

    @property
    def page_label(self) -> QLabel:
        return self._page_label

    @property
    def page_count(self) -> int:
        return len(self._pages)

    @property
    def page_index(self) -> int:
        return self._page_index
