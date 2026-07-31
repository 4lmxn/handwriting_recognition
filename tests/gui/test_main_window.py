from PySide6.QtWidgets import QTabWidget

from app.gui.main_window import MainWindow


def test_main_window_constructs_with_drawing_canvas_tab(qtbot, app_config):
    window = MainWindow(app_config)
    qtbot.addWidget(window)

    assert window.windowTitle() == app_config.window.title

    tabs = window.centralWidget()
    assert isinstance(tabs, QTabWidget)
    assert tabs.count() >= 1
    assert tabs.tabText(0) == "Drawing Canvas"


def test_status_bar_reports_device(qtbot, app_config):
    window = MainWindow(app_config)
    qtbot.addWidget(window)

    assert "Device:" in window.statusBar().currentMessage()
