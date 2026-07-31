from __future__ import annotations

import logging
import sys

from PySide6.QtWidgets import QApplication

from app.config import load_config
from app.gui.main_window import MainWindow
from app.logging_config import setup_logging


def main() -> int:
    config = load_config()
    config.paths.ensure_exist()
    setup_logging(config.paths.logs, config.log_level)

    logger = logging.getLogger(__name__)
    logger.info(
        "Starting %s v%s (device=%s)", config.name, config.version, config.resolved_device()
    )

    app = QApplication(sys.argv)
    window = MainWindow(config)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
