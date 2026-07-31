"""Central logging setup, used by both the GUI app and (in later phases) training scripts."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(logs_dir: Path, level: str = "INFO", filename: str = "app.log") -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / filename

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
