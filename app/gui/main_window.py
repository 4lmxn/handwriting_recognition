from __future__ import annotations

from PySide6.QtWidgets import QMainWindow, QTabWidget

from app.config import AppConfig
from app.gui.tabs.drawing_canvas_tab import DrawingCanvasTab
from app.gui.tabs.upload_document_tab import UploadDocumentTab


class MainWindow(QMainWindow):
    """Top-level window. Additional tabs (Training Dashboard, ...) are
    added in later phases as their corresponding backends land."""

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self._config = config

        self.setWindowTitle(config.window.title)
        self.resize(config.window.width, config.window.height)

        tabs = QTabWidget()
        tabs.addTab(DrawingCanvasTab(config.canvas, config.paths), "Drawing Canvas")
        tabs.addTab(UploadDocumentTab(), "Upload Document")
        self.setCentralWidget(tabs)

        self.statusBar().showMessage(f"Device: {config.resolved_device()}")
