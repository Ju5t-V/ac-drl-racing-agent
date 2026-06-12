"""T0.5 tests: run_metadata.json content, git hash, atomic update."""

import json
import subprocess
from datetime import date
from pathlib import Path

import numpy
import pytest
import torch

from acagent.config import Config, config_to_dict
from acagent.session.metadata import (
    collect_versions,
    get_git_hash,
    read_metadata,
    update_metadata,
    write_metadata,
)
from acagent.session.paths import create_session

pytestmark = [pytest.mark.unit, pytest.mark.phase0]


@pytest.fixture()
def session(tmp_path: Path):
    return create_session(tmp_path, today=date(2026, 6, 12))


class TestWrite:
    def test_writes_all_required_sections(self, session) -> None:
        path = write_metadata(session, Config())
        data = json.loads(path.read_text(encoding="utf-8"))
        for key in ("created_utc", "config", "normalization_constants", "versions", "git_hash"):
            assert key in data

    def test_config_snapshot_matches(self, session) -> None:
        config = Config()
        write_metadata(session, config)
        data = read_metadata(session)
        expected = json.loads(json.dumps(config_to_dict(config)))  # tuples -> lists
        assert data["config"] == expected
        assert data["config"]["sac"]["gamma"] == config.sac.gamma

    def test_versions_match_imported_modules(self, session) -> None:
        write_metadata(session, Config())
        versions = read_metadata(session)["versions"]
        assert versions["torch"] == torch.__version__
        assert versions["numpy"] == numpy.__version__
        assert versions["python"].count(".") == 2

    def test_normalization_placeholder_defaults_empty(self, session) -> None:
        write_metadata(session, Config())
        assert read_metadata(session)["normalization_constants"] == {}

    def test_normalization_constants_recorded_when_given(self, session) -> None:
        constants = {"speed": {"scale": 90.0}, "rpm": {"scale": 9000.0}}
        write_metadata(session, Config(), normalization_constants=constants)
        assert read_metadata(session)["normalization_constants"] == constants


class TestGitHash:
    def test_none_outside_a_repo(self, tmp_path: Path) -> None:
        assert get_git_hash(tmp_path) is None

    def test_hash_inside_a_repo(self, tmp_path: Path) -> None:
        env_overrides = {
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
        }
        import os

        env = {**os.environ, **env_overrides}
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True, env=env)
        (tmp_path / "f.txt").write_text("x", encoding="utf-8")
        subprocess.run(["git", "add", "f.txt"], cwd=tmp_path, check=True, env=env)
        subprocess.run(
            ["git", "commit", "-q", "-m", "c", "--no-gpg-sign"], cwd=tmp_path, check=True, env=env
        )
        head = get_git_hash(tmp_path)
        assert head is not None
        assert len(head) == 40
        assert all(c in "0123456789abcdef" for c in head)


class TestUpdate:
    def test_update_merges_and_preserves(self, session) -> None:
        write_metadata(session, Config())
        before = read_metadata(session)
        after = update_metadata(session, {"compile_decision": "eager"})
        assert after["compile_decision"] == "eager"
        assert after["config"] == before["config"]
        assert read_metadata(session)["compile_decision"] == "eager"

    def test_collect_versions_keys(self) -> None:
        versions = collect_versions()
        assert {"acagent", "python", "torch", "numpy", "scipy", "platform"} <= set(versions)
