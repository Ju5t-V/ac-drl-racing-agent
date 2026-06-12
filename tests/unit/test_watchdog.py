"""T0.8 tests: watchdog restarts a crashing dummy child; child reports resume."""

import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from acagent.session.watchdog import RESTART_COUNT_ENV, RESUMED_ENV, Watchdog

pytestmark = [pytest.mark.unit, pytest.mark.phase0]

# Dummy child: crashes until `crash_until` prior runs are recorded, then
# writes a marker with its RESUMED/RESTART_COUNT env values and exits 0.
CHILD_SCRIPT = """
import os
import sys
from pathlib import Path

state_dir = Path(sys.argv[1])
crash_until = int(sys.argv[2])

runs_file = state_dir / "runs.txt"
runs = int(runs_file.read_text()) if runs_file.exists() else 0
runs_file.write_text(str(runs + 1))

(state_dir / f"env_{runs}.txt").write_text(
    os.environ["__RESUMED__"] + "," + os.environ["__COUNT__"]
)

if runs < crash_until:
    sys.exit(1)
(state_dir / "success.txt").write_text("ok")
sys.exit(0)
""".replace("__RESUMED__", RESUMED_ENV).replace("__COUNT__", RESTART_COUNT_ENV)


@pytest.fixture()
def child(tmp_path: Path):
    script = tmp_path / "child.py"
    script.write_text(CHILD_SCRIPT, encoding="utf-8")

    def command(crash_until: int) -> list[str]:
        return [sys.executable, str(script), str(tmp_path), str(crash_until)]

    return command


class TestRestart:
    def test_restarts_crashing_child_until_success(self, child, tmp_path: Path) -> None:
        result = Watchdog(child(2), max_restarts=5, backoff_s=0.0).run()
        assert result.final_exit_code == 0
        assert result.restarts == 2
        assert result.spawns == 3
        assert not result.gave_up
        assert not result.stopped
        assert (tmp_path / "success.txt").exists()

    def test_resume_signal(self, child, tmp_path: Path) -> None:
        """First launch sees RESUMED=0; every restart sees RESUMED=1 + its count."""
        Watchdog(child(2), max_restarts=5, backoff_s=0.0).run()
        assert (tmp_path / "env_0.txt").read_text() == "0,0"
        assert (tmp_path / "env_1.txt").read_text() == "1,1"
        assert (tmp_path / "env_2.txt").read_text() == "1,2"

    def test_clean_exit_not_restarted(self, child, tmp_path: Path) -> None:
        result = Watchdog(child(0), max_restarts=5, backoff_s=0.0).run()
        assert result.spawns == 1
        assert result.restarts == 0
        assert result.final_exit_code == 0
        assert (tmp_path / "runs.txt").read_text() == "1"

    def test_gives_up_at_restart_cap(self, child, tmp_path: Path) -> None:
        result = Watchdog(child(99), max_restarts=3, backoff_s=0.0).run()
        assert result.gave_up
        assert result.restarts == 3
        assert result.spawns == 4
        assert result.final_exit_code == 1
        assert not (tmp_path / "success.txt").exists()


class TestStop:
    def test_stop_terminates_long_running_child(self, tmp_path: Path) -> None:
        script = tmp_path / "sleeper.py"
        script.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
        dog = Watchdog([sys.executable, str(script)], max_restarts=0, backoff_s=0.0)

        started = time.monotonic()
        stopper = threading.Timer(0.5, dog.stop)
        stopper.start()
        try:
            result = dog.run()
        finally:
            stopper.cancel()
        elapsed = time.monotonic() - started

        assert result.stopped
        assert not result.gave_up
        assert elapsed < 30  # nowhere near the child's 60 s sleep


class TestValidation:
    def test_bad_args_rejected(self) -> None:
        with pytest.raises(ValueError):
            Watchdog([])
        with pytest.raises(ValueError):
            Watchdog(["x"], max_restarts=-1)
        with pytest.raises(ValueError):
            Watchdog(["x"], backoff_s=-1.0)

    def test_child_env_passthrough(self, tmp_path: Path, monkeypatch) -> None:
        """extra_env reaches the child alongside the inherited environment."""
        script = tmp_path / "echo_env.py"
        script.write_text(
            "import os\nfrom pathlib import Path\n"
            f"Path(r'{tmp_path}', 'env.txt').write_text(os.environ['EXTRA_FLAG'])\n",
            encoding="utf-8",
        )
        result = Watchdog(
            [sys.executable, str(script)], max_restarts=0, extra_env={"EXTRA_FLAG": "yes"}
        ).run()
        assert result.final_exit_code == 0
        assert (tmp_path / "env.txt").read_text() == "yes"


def test_child_script_is_valid_python(tmp_path: Path) -> None:
    """Guard the test harness itself: the dummy child compiles."""
    script = tmp_path / "child.py"
    script.write_text(CHILD_SCRIPT, encoding="utf-8")
    subprocess.run([sys.executable, "-m", "py_compile", str(script)], check=True)
