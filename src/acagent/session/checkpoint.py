"""Full training-state checkpointing (SPEC §6, T0.6; CLAUDE.md Invariant 10).

A checkpoint must fully reconstruct training: network weights, optimizer
state, entropy coefficient, PER priority structure, step counters, and RNG
state. (The replay buffer's bulk arrays are snapshotted separately by
``buffer_store`` (T0.7); together the two artifacts reconstruct everything.)

Writes are atomic (temp + ``os.replace``): a crash mid-save can never
corrupt the previous checkpoint.
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from acagent.session.atomic import atomic_write_bytes
from acagent.session.paths import SessionPaths

logger = logging.getLogger(__name__)

_CKPT_RE = re.compile(r"^ckpt_(\d{12})\.pt$")


@dataclass
class TrainingState:
    """Everything Invariant 10 demands, keyed by stable component names.

    Phase 3's trainer populates this from live objects; Phase 0 proves the
    round-trip with dummy networks/optimizers of the same shapes.
    """

    networks: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    optimizers: dict[str, dict[str, Any]] = field(default_factory=dict)
    entropy_coef: float = 1.0
    per_state: dict[str, Any] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)
    rng_state: dict[str, Any] = field(default_factory=dict)


def checkpoint_path(paths: SessionPaths, step: int) -> Path:
    return paths.checkpoints / f"ckpt_{step:012d}.pt"


def save_checkpoint(
    paths: SessionPaths,
    state: TrainingState,
    step: int,
    keep_last: int | None = None,
) -> Path:
    """Atomically write the checkpoint for ``step``; optionally prune old ones."""
    if step < 0:
        raise ValueError("step must be >= 0")
    buffer = io.BytesIO()
    torch.save(state, buffer)
    target = checkpoint_path(paths, step)
    atomic_write_bytes(target, buffer.getvalue())
    logger.info("checkpoint saved: %s", target.name)
    if keep_last is not None and keep_last > 0:
        for old in list_checkpoints(paths)[:-keep_last]:
            old.unlink(missing_ok=True)
            logger.info("checkpoint pruned: %s", old.name)
    return target


def list_checkpoints(paths: SessionPaths) -> list[Path]:
    """All checkpoint files in the session, oldest first."""
    if not paths.checkpoints.is_dir():
        return []
    found = [p for p in paths.checkpoints.iterdir() if _CKPT_RE.match(p.name)]
    return sorted(found, key=lambda p: int(_CKPT_RE.match(p.name).group(1)))  # type: ignore[union-attr]


def load_checkpoint(path: str | Path) -> TrainingState:
    # weights_only=False: TrainingState carries numpy bit-generator dicts and
    # optimizer state, not just tensors. Checkpoints are our own artifacts.
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(state, TrainingState):
        raise TypeError(f"{path} did not contain a TrainingState")
    return state


def load_latest_checkpoint(paths: SessionPaths) -> tuple[TrainingState, int] | None:
    """Load the newest readable checkpoint, falling back past corrupt files.

    Returns ``(state, step)`` or None when no readable checkpoint exists.
    Atomic writes make corruption unlikely, but auto-resume (SPEC §6
    watchdogs) must survive it anyway.
    """
    for path in reversed(list_checkpoints(paths)):
        try:
            state = load_checkpoint(path)
        except Exception:
            logger.warning("unreadable checkpoint skipped: %s", path.name, exc_info=True)
            continue
        match = _CKPT_RE.match(path.name)
        assert match is not None
        return state, int(match.group(1))
    return None
