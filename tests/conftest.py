import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from app.config import AppConfig, load_config

# `qapp` / `qtbot` fixtures are provided by pytest-qt (qt_api = "pyside6" in pyproject.toml).


@pytest.fixture()
def app_config() -> AppConfig:
    return load_config()
