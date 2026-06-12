"""T0.4 tests: session layout creation, collision suffixing, latest discovery."""

from datetime import date
from pathlib import Path

import pytest

from acagent.session.paths import SessionPaths, create_session, find_latest_session, open_session

pytestmark = [pytest.mark.unit, pytest.mark.phase0]

DAY = date(2026, 6, 12)


class TestCreate:
    def test_creates_named_root_and_all_subdirs(self, tmp_path: Path) -> None:
        paths = create_session(tmp_path, today=DAY)
        assert paths.root == tmp_path / "session_20260612"
        for sub in ("checkpoints", "buffer", "setups", "optuna", "logs"):
            assert (paths.root / sub).is_dir()
        assert paths.metadata_file == paths.root / "run_metadata.json"

    def test_same_day_collision_gets_suffix(self, tmp_path: Path) -> None:
        first = create_session(tmp_path, today=DAY)
        second = create_session(tmp_path, today=DAY)
        third = create_session(tmp_path, today=DAY)
        assert first.root.name == "session_20260612"
        assert second.root.name == "session_20260612_02"
        assert third.root.name == "session_20260612_03"

    def test_creates_missing_base_dir(self, tmp_path: Path) -> None:
        base = tmp_path / "nested" / "sessions"
        paths = create_session(base, today=DAY)
        assert paths.root.is_dir()


class TestOpen:
    def test_open_existing(self, tmp_path: Path) -> None:
        created = create_session(tmp_path, today=DAY)
        opened = open_session(created.root)
        assert opened == created

    def test_open_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            open_session(tmp_path / "session_19990101")

    def test_open_restores_missing_subdirs(self, tmp_path: Path) -> None:
        created = create_session(tmp_path, today=DAY)
        (created.root / "logs").rmdir()
        opened = open_session(created.root)
        assert opened.logs.is_dir()


class TestFindLatest:
    def test_none_when_base_missing_or_empty(self, tmp_path: Path) -> None:
        assert find_latest_session(tmp_path / "nope") is None
        assert find_latest_session(tmp_path) is None

    def test_latest_across_dates(self, tmp_path: Path) -> None:
        create_session(tmp_path, today=date(2026, 6, 10))
        create_session(tmp_path, today=date(2026, 6, 12))
        create_session(tmp_path, today=date(2026, 6, 11))
        latest = find_latest_session(tmp_path)
        assert latest is not None
        assert latest.root.name == "session_20260612"

    def test_suffixed_beats_unsuffixed_same_day(self, tmp_path: Path) -> None:
        create_session(tmp_path, today=DAY)
        newest = create_session(tmp_path, today=DAY)
        latest = find_latest_session(tmp_path)
        assert latest is not None
        assert latest.root == newest.root

    def test_ignores_non_session_entries(self, tmp_path: Path) -> None:
        (tmp_path / "session_garbage").mkdir()
        (tmp_path / "zzz_not_a_session").mkdir()
        (tmp_path / "session_20991231.txt").write_text("file, not dir", encoding="utf-8")
        created = create_session(tmp_path, today=DAY)
        latest = find_latest_session(tmp_path)
        assert latest is not None
        assert latest.root == created.root

    def test_returns_sessionpaths(self, tmp_path: Path) -> None:
        create_session(tmp_path, today=DAY)
        assert isinstance(find_latest_session(tmp_path), SessionPaths)
