"""``run_metadata.json`` writer (SPEC §6, T0.5).

Records everything needed to interpret a session after the fact: the full
config snapshot (γ, all ``w_*`` weights, λ values once Phase 5 adds them),
normalization constants (placeholder until Phase 2 fills them), software
versions, and the git hash.
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy
import scipy
import torch

import acagent
from acagent.config import Config, config_to_dict
from acagent.session.atomic import atomic_write_text
from acagent.session.paths import SessionPaths

_REPO_ROOT = Path(__file__).resolve().parents[3]


def collect_versions() -> dict[str, str]:
    return {
        "acagent": acagent.__version__,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
    }


def get_git_hash(repo_dir: str | Path | None = None) -> str | None:
    """Current HEAD commit hash, or None when git/repo is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(repo_dir) if repo_dir is not None else _REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def write_metadata(
    paths: SessionPaths,
    config: Config,
    normalization_constants: dict[str, Any] | None = None,
) -> Path:
    """Write ``run_metadata.json`` atomically; returns its path."""
    metadata = {
        "created_utc": datetime.now(UTC).isoformat(),
        "config": config_to_dict(config),
        # Placeholder until Phase 2 (T2.4) records the fixed per-channel
        # normalization constants (SPEC §4 Module B).
        "normalization_constants": normalization_constants or {},
        "versions": collect_versions(),
        "git_hash": get_git_hash(),
    }
    atomic_write_text(paths.metadata_file, json.dumps(metadata, indent=2))
    return paths.metadata_file


def read_metadata(paths: SessionPaths) -> dict[str, Any]:
    return json.loads(paths.metadata_file.read_text(encoding="utf-8"))


def update_metadata(paths: SessionPaths, updates: dict[str, Any]) -> dict[str, Any]:
    """Read-merge-write (atomic). Later phases record e.g. normalization
    constants (T2.4) and the compile-vs-eager decision (T2.5) this way."""
    metadata = read_metadata(paths)
    metadata.update(updates)
    atomic_write_text(paths.metadata_file, json.dumps(metadata, indent=2))
    return metadata
