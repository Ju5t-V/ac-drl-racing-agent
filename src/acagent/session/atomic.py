"""Atomic file writes: temp file + ``os.replace`` in the destination directory.

Shared by metadata, checkpoint, and buffer-store writers (T0.5-T0.7) so a
crash mid-write can never leave a torn file where a consistent one stood
(SPEC §6 crash resilience).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    """Write ``data`` to ``path`` such that readers see old or new, never partial."""
    path = Path(path)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, text.encode(encoding))
