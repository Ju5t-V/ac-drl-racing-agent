"""Single seeded RNG manager (CLAUDE.md Invariants 7 and 10).

All randomness in the project flows through one :class:`RngManager`.
Components request *named* child generators; a child's stream is a pure
function of ``(seed, name)`` — independent of creation order — so any
component is reproducible in isolation. Checkpoints capture and restore
the full RNG state via :meth:`RngManager.get_state` / :meth:`set_state`.

The manager also owns the process-global generators (``torch``'s default
generator and Python's ``random`` module): they are seeded at construction
and included in state capture, because ``torch.nn`` module initialization
and dropout draw from the torch global generator rather than an explicit
``torch.Generator``.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

import numpy as np
import torch


def _name_words(name: str) -> list[int]:
    """Stable 4x32-bit entropy words derived from a child name."""
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return [int.from_bytes(digest[i : i + 4], "little") for i in range(0, 16, 4)]


class RngManager:
    """Hands out named, deterministically-seeded numpy and torch generators."""

    def __init__(self, seed: int) -> None:
        if seed < 0:
            raise ValueError("seed must be >= 0")
        self._seed = seed
        self._numpy: dict[str, np.random.Generator] = {}
        self._torch: dict[str, torch.Generator] = {}
        self._seed_globals()

    @property
    def seed(self) -> int:
        return self._seed

    def _child_seed_sequence(self, name: str) -> np.random.SeedSequence:
        return np.random.SeedSequence([self._seed, *_name_words(name)])

    def _seed_globals(self) -> None:
        seq = self._child_seed_sequence("__globals__")
        torch_seed, py_seed = (int(s) for s in seq.generate_state(2, dtype=np.uint64))
        torch.manual_seed(torch_seed)
        random.seed(py_seed)

    def numpy_generator(self, name: str) -> np.random.Generator:
        """Return the named numpy child generator, creating it on first use."""
        if name not in self._numpy:
            self._numpy[name] = np.random.Generator(
                np.random.PCG64(self._child_seed_sequence(name))
            )
        return self._numpy[name]

    def torch_generator(self, name: str) -> torch.Generator:
        """Return the named torch child generator (CPU), creating it on first use."""
        if name not in self._torch:
            seed = int(self._child_seed_sequence(name).generate_state(1, dtype=np.uint64)[0])
            gen = torch.Generator()
            gen.manual_seed(seed)
            self._torch[name] = gen
        return self._torch[name]

    def get_state(self) -> dict[str, Any]:
        """Full snapshot of seed, all child streams, and the global generators."""
        return {
            "seed": self._seed,
            "numpy": {name: gen.bit_generator.state for name, gen in self._numpy.items()},
            "torch": {name: gen.get_state().clone() for name, gen in self._torch.items()},
            "torch_global": torch.get_rng_state().clone(),
            "python_global": random.getstate(),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore a snapshot. Children absent from the manager are recreated.

        After restore, every named stream (and the globals) continues exactly
        where the snapshot left off; names not in the snapshot but requested
        later still derive deterministically from the restored seed.
        """
        self._seed = int(state["seed"])
        self._numpy = {}
        self._torch = {}
        for name, np_state in state["numpy"].items():
            gen = self.numpy_generator(name)
            gen.bit_generator.state = np_state
        for name, torch_state in state["torch"].items():
            self.torch_generator(name).set_state(torch_state.clone())
        torch.set_rng_state(state["torch_global"].clone())
        random.setstate(state["python_global"])
