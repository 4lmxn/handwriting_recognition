from pathlib import Path

from models.adapters.resolver import find_latest_adapter, resolve_adapter_path


def test_find_latest_adapter_returns_none_for_missing_dir(tmp_path):
    assert find_latest_adapter(tmp_path / "does-not-exist") is None


def test_find_latest_adapter_returns_none_for_empty_dir(tmp_path):
    assert find_latest_adapter(tmp_path) is None


def test_find_latest_adapter_picks_newest_by_name(tmp_path):
    (tmp_path / "v-1000-a").mkdir()
    (tmp_path / "v-3000-c").mkdir()
    (tmp_path / "v-2000-b").mkdir()
    (tmp_path / "unrelated").mkdir()
    latest = find_latest_adapter(tmp_path)
    assert latest is not None
    assert latest.name == "v-3000-c"


def test_find_latest_adapter_skips_rejected_versions(tmp_path):
    (tmp_path / "v-1000-a").mkdir()
    (tmp_path / "v-9999-BAD-REJECTED").mkdir()
    latest = find_latest_adapter(tmp_path)
    assert latest is not None
    # v-9999 is chronologically later but REJECTED — skipped, so v-1000 wins.
    # A rejected adapter must never seed the next update or a single bad
    # increment could poison every subsequent one.
    assert latest.name == "v-1000-a"


def test_resolve_adapter_path_returns_none_for_none(tmp_path):
    assert resolve_adapter_path(None, tmp_path) is None


def test_resolve_adapter_path_returns_none_for_empty_string(tmp_path):
    assert resolve_adapter_path("", tmp_path) is None


def test_resolve_adapter_path_resolves_latest_when_adapters_exist(tmp_path):
    (tmp_path / "v-1000-a").mkdir()
    (tmp_path / "v-2000-b").mkdir()
    resolved = resolve_adapter_path("latest", tmp_path)
    assert resolved is not None
    assert resolved.name == "v-2000-b"


def test_resolve_adapter_path_returns_none_when_latest_but_no_adapters(tmp_path):
    # A fresh install has no corrections yet, so "latest" should silently
    # fall back to None — recognition still needs to work out of the box.
    assert resolve_adapter_path("latest", tmp_path) is None


def test_resolve_adapter_path_returns_literal_path(tmp_path):
    literal = str(tmp_path / "my-adapter")
    resolved = resolve_adapter_path(literal, tmp_path)
    assert resolved == Path(literal)
