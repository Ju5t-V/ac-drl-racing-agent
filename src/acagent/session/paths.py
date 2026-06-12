"""Session directory layout: creation and discovery (SPEC §6, T0.4).

A session lives in ``<base>/session_YYYYMMDD/`` (same-day collisions get a
``_NN`` suffix: ``session_YYYYMMDD_02``, ``_03``, ...). Each session holds
SAC checkpoints, replay-buffer snapshots, per-stint setup files, Optuna
study DBs, logs, and ``run_metadata.json``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

_SESSION_NAME_RE = re.compile(r"^session_(\d{8})(?:_(\d{2,}))?$")


@dataclass(frozen=True)
class SessionPaths:
    """Resolved paths inside one session directory."""

    root: Path

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def buffer(self) -> Path:
        return self.root / "buffer"

    @property
    def setups(self) -> Path:
        return self.root / "setups"

    @property
    def optuna(self) -> Path:
        return self.root / "optuna"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def metadata_file(self) -> Path:
        return self.root / "run_metadata.json"

    def subdirs(self) -> tuple[Path, ...]:
        return (self.checkpoints, self.buffer, self.setups, self.optuna, self.logs)


def _sort_key(name: str) -> tuple[str, int]:
    match = _SESSION_NAME_RE.match(name)
    assert match is not None
    return (match.group(1), int(match.group(2) or 1))


def create_session(base_dir: str | Path, today: date | None = None) -> SessionPaths:
    """Create a new session directory (collision-suffixed) with all subdirs."""
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    stamp = (today or datetime.now().date()).strftime("%Y%m%d")
    suffix = 1
    while True:
        name = f"session_{stamp}" if suffix == 1 else f"session_{stamp}_{suffix:02d}"
        try:
            root = base / name
            root.mkdir(exist_ok=False)
            break
        except FileExistsError:
            suffix += 1
    paths = SessionPaths(root=root)
    for sub in paths.subdirs():
        sub.mkdir(exist_ok=True)
    return paths


def open_session(root: str | Path) -> SessionPaths:
    """Wrap an existing session directory, creating any missing subdirs."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"session directory does not exist: {root}")
    paths = SessionPaths(root=root)
    for sub in paths.subdirs():
        sub.mkdir(exist_ok=True)
    return paths


def find_latest_session(base_dir: str | Path) -> SessionPaths | None:
    """Return the most recent session under ``base_dir``, or None.

    Ordering is by date stamp, then collision suffix. Non-session entries
    are ignored. Used by auto-resume (SPEC §6 watchdogs).
    """
    base = Path(base_dir)
    if not base.is_dir():
        return None
    candidates = [
        entry for entry in base.iterdir() if entry.is_dir() and _SESSION_NAME_RE.match(entry.name)
    ]
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: _sort_key(p.name))
    return SessionPaths(root=latest)
