"""Replay-buffer snapshots: cadenced, memory-mappable, crash-consistent (T0.7).

SPEC §6: the replay buffer is snapshotted to disk on a fixed cadence
(default 15 min) in a memory-mappable format; on resume, the latest
*consistent* snapshot loads with its priorities.

Format: one directory per snapshot, ``snap_<step:012d>/``, holding one
``.npy`` file per named array (always including ``priorities`` once Phase 3
wires the PER buffer in) plus ``manifest.json``. Consistency protocol:
arrays are written into a hidden temp directory, the manifest is written
last, and the directory is renamed into place — a snapshot directory
without a valid manifest, or whose array files do not match the manifest's
recorded shapes/sizes, is treated as torn and skipped on load.

This module is deliberately generic over ``dict[str, np.ndarray]`` so the
Phase 3 PER buffer plugs into it without this module importing Phase 3 code
(BUILD_PLAN: no forward references).

Locking: instances are not thread-safe; the owning trainer serializes calls
(the Phase 3 learner thread snapshots from its own thread only).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_SNAP_PREFIX = "snap_"
_TMP_PREFIX = ".tmp_snap_"
_MANIFEST = "manifest.json"


@dataclass
class LoadedSnapshot:
    step: int
    arrays: dict[str, np.ndarray]
    meta: dict[str, Any]
    path: Path


class BufferStore:
    """Writes and discovers buffer snapshots under one directory."""

    def __init__(
        self,
        directory: str | Path,
        interval_s: float = 900.0,
        keep_last: int = 2,
        now_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be > 0")
        if keep_last < 1:
            raise ValueError("keep_last must be >= 1")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.interval_s = interval_s
        self.keep_last = keep_last
        self._now_fn = now_fn
        self._last_snapshot_at: float | None = None

    # -- writing -----------------------------------------------------------

    def maybe_snapshot(
        self, arrays: dict[str, np.ndarray], meta: dict[str, Any], step: int
    ) -> Path | None:
        """Snapshot if the cadence interval has elapsed (or never snapshotted)."""
        now = self._now_fn()
        if self._last_snapshot_at is not None and now - self._last_snapshot_at < self.interval_s:
            return None
        return self.snapshot(arrays, meta, step)

    def snapshot(self, arrays: dict[str, np.ndarray], meta: dict[str, Any], step: int) -> Path:
        """Write a consistent snapshot for ``step``. Returns its directory."""
        if not arrays:
            raise ValueError("snapshot requires at least one named array")
        self._clean_temp_dirs()
        tmp_dir = self.directory / f"{_TMP_PREFIX}{uuid.uuid4().hex}"
        tmp_dir.mkdir()
        manifest: dict[str, Any] = {"step": step, "meta": meta, "arrays": {}}
        for name, array in arrays.items():
            file_path = tmp_dir / f"{name}.npy"
            np.save(file_path, np.ascontiguousarray(array))
            manifest["arrays"][name] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "nbytes_on_disk": file_path.stat().st_size,
            }
        # Manifest last: its presence marks the snapshot as complete.
        (tmp_dir / _MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        final = self.directory / f"{_SNAP_PREFIX}{step:012d}"
        if final.exists():
            shutil.rmtree(final)
        os.rename(tmp_dir, final)
        self._last_snapshot_at = self._now_fn()
        logger.info("buffer snapshot written: %s", final.name)
        self._prune()
        return final

    def _prune(self) -> None:
        snaps = self._snapshot_dirs()
        for old in snaps[: -self.keep_last]:
            try:
                shutil.rmtree(old)
                logger.info("buffer snapshot pruned: %s", old.name)
            except OSError:
                logger.warning("could not prune snapshot %s", old.name, exc_info=True)

    def _clean_temp_dirs(self) -> None:
        for leftover in self.directory.glob(f"{_TMP_PREFIX}*"):
            try:
                shutil.rmtree(leftover)
            except OSError:
                pass

    # -- reading -----------------------------------------------------------

    def _snapshot_dirs(self) -> list[Path]:
        found = [
            p
            for p in self.directory.iterdir()
            if p.is_dir()
            and p.name.startswith(_SNAP_PREFIX)
            and p.name[len(_SNAP_PREFIX) :].isdigit()
        ]
        return sorted(found, key=lambda p: int(p.name[len(_SNAP_PREFIX) :]))

    def load_latest(self, mmap: bool = False) -> LoadedSnapshot | None:
        """Load the newest consistent snapshot, skipping torn ones.

        ``mmap=True`` returns ``np.memmap`` views (read-only) instead of RAM
        copies; note Windows holds the files locked while views are alive.
        """
        for snap_dir in reversed(self._snapshot_dirs()):
            loaded = self._try_load(snap_dir, mmap)
            if loaded is not None:
                return loaded
            logger.warning("inconsistent buffer snapshot skipped: %s", snap_dir.name)
        return None

    def _try_load(self, snap_dir: Path, mmap: bool) -> LoadedSnapshot | None:
        manifest_path = snap_dir / _MANIFEST
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            arrays: dict[str, np.ndarray] = {}
            for name, spec in manifest["arrays"].items():
                file_path = snap_dir / f"{name}.npy"
                if file_path.stat().st_size != spec["nbytes_on_disk"]:
                    return None  # torn array file
                array = np.load(file_path, mmap_mode="r" if mmap else None)
                if list(array.shape) != spec["shape"] or str(array.dtype) != spec["dtype"]:
                    return None
                arrays[name] = array
            return LoadedSnapshot(
                step=int(manifest["step"]),
                arrays=arrays,
                meta=manifest["meta"],
                path=snap_dir,
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return None
