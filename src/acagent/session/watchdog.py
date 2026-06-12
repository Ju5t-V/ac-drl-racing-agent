"""Process supervisor (SPEC §6 watchdogs, T0.8).

Runs a child command and restarts it whenever it exits abnormally, with a
configurable backoff and restart cap. Restarted children receive
``ACAGENT_RESUMED=1`` (first launch gets ``0``) and ``ACAGENT_RESTART_COUNT``
so the trainer entry point knows to auto-resume from the latest checkpoint
(SPEC §6: a GPU driver reset is a routine watchdog event, not a catastrophe).

This is the generic supervision core; Phase 1+ composes it into the trainer
watchdog and the AC-process watchdog. Thread-safety: ``run()`` is called
from one thread; ``stop()`` may be called from any thread.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

RESUMED_ENV = "ACAGENT_RESUMED"
RESTART_COUNT_ENV = "ACAGENT_RESTART_COUNT"


@dataclass(frozen=True)
class WatchdogResult:
    """Outcome of a supervision run."""

    spawns: int
    restarts: int
    final_exit_code: int | None
    gave_up: bool
    stopped: bool


class Watchdog:
    """Restart a child command on abnormal exit until it succeeds or the cap hits."""

    def __init__(
        self,
        command: list[str],
        max_restarts: int = 10,
        backoff_s: float = 1.0,
        extra_env: dict[str, str] | None = None,
        poll_interval_s: float = 0.05,
    ) -> None:
        if not command:
            raise ValueError("command must be non-empty")
        if max_restarts < 0:
            raise ValueError("max_restarts must be >= 0")
        if backoff_s < 0:
            raise ValueError("backoff_s must be >= 0")
        self.command = list(command)
        self.max_restarts = max_restarts
        self.backoff_s = backoff_s
        self.extra_env = dict(extra_env or {})
        self.poll_interval_s = poll_interval_s
        self._stop_event = threading.Event()
        self._child: subprocess.Popen[bytes] | None = None

    def stop(self) -> None:
        """Request shutdown: terminates the current child and ends supervision."""
        self._stop_event.set()

    def _spawn(self, restart_count: int) -> subprocess.Popen[bytes]:
        env = {
            **os.environ,
            **self.extra_env,
            RESUMED_ENV: "1" if restart_count > 0 else "0",
            RESTART_COUNT_ENV: str(restart_count),
        }
        logger.info(
            "watchdog spawning child (restart %d): %s", restart_count, " ".join(self.command)
        )
        return subprocess.Popen(self.command, env=env)

    def _wait_for_exit(self, child: subprocess.Popen[bytes]) -> int | None:
        """Wait until the child exits or stop() is requested. None means stopped."""
        while True:
            code = child.poll()
            if code is not None:
                return code
            if self._stop_event.is_set():
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                return None
            time.sleep(self.poll_interval_s)

    def _interruptible_backoff(self) -> None:
        self._stop_event.wait(self.backoff_s)

    def run(self) -> WatchdogResult:
        """Supervise until clean exit, stop(), or the restart cap is reached."""
        restart_count = 0
        spawns = 0
        while True:
            if self._stop_event.is_set():
                return WatchdogResult(spawns, restart_count, None, gave_up=False, stopped=True)
            self._child = self._spawn(restart_count)
            spawns += 1
            exit_code = self._wait_for_exit(self._child)
            if exit_code is None:
                return WatchdogResult(spawns, restart_count, None, gave_up=False, stopped=True)
            if exit_code == 0:
                logger.info("watchdog: child exited cleanly")
                return WatchdogResult(spawns, restart_count, 0, gave_up=False, stopped=False)
            logger.warning("watchdog: child exited with code %d", exit_code)
            if restart_count >= self.max_restarts:
                logger.error("watchdog: restart cap (%d) reached, giving up", self.max_restarts)
                return WatchdogResult(spawns, restart_count, exit_code, gave_up=True, stopped=False)
            restart_count += 1
            self._interruptible_backoff()
