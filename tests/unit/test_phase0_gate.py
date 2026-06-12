"""G0.A gate tests (BUILD_PLAN Phase 0).

1. Checkpoint round-trip: dummy actor/critic/optimizer + populated buffer
   with priorities; save; reload into fresh objects; exact equality of
   weights, optimizer tensors, entropy coefficient, priorities, counters,
   and RNG draws after restore.
2. Kill-resume simulation: N steps -> snapshot -> simulated kill -> resume
   -> M steps reproduces an uninterrupted N+M run seed-for-seed.
3. Watchdog end-to-end: a crashing dummy trainer child checkpoints through
   the real session machinery, is restarted by the watchdog, reports the
   resume signal, and its loss trajectory matches an uninterrupted run of
   the same script bit for bit.
"""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import torch
from dummy_trainer import DummyTrainer

from acagent.session.buffer_store import BufferStore
from acagent.session.checkpoint import load_latest_checkpoint, save_checkpoint
from acagent.session.paths import create_session
from acagent.session.watchdog import Watchdog

pytestmark = [pytest.mark.unit, pytest.mark.phase0]


def test_gate_full_checkpoint_round_trip(tmp_path: Path) -> None:
    """Scenario 1: everything Invariant 10 lists survives save/load exactly."""
    session = create_session(tmp_path, today=date(2026, 6, 13))
    store = BufferStore(session.buffer)

    trainer = DummyTrainer(seed=21)
    trainer.run(40)
    saved_state = trainer.training_state()
    # Peek at the next draws, then rewind to the saved state: the restored
    # trainer must reproduce exactly these draws after loading.
    expected_np = trainer.rng.numpy_generator("buffer").random(10)
    expected_torch = torch.rand(10, generator=trainer.rng.torch_generator("data"))
    trainer.rng.set_state(saved_state.rng_state)
    assert np.array_equal(trainer.rng.numpy_generator("buffer").random(10), expected_np)
    assert torch.equal(
        torch.rand(10, generator=trainer.rng.torch_generator("data")), expected_torch
    )
    trainer.rng.set_state(saved_state.rng_state)

    save_checkpoint(session, saved_state, step=trainer.step_count)
    store.snapshot(trainer.buffer_arrays(), {"size": trainer.step_count}, step=trainer.step_count)

    loaded = load_latest_checkpoint(session)
    snapshot = store.load_latest()
    assert loaded is not None and snapshot is not None
    state, step = loaded
    assert step == 40

    fresh = DummyTrainer(seed=909)  # deliberately different seed/init
    fresh.restore(state, snapshot.arrays)

    original = trainer.training_state()
    for key, tensor in original.networks["net"].items():
        assert torch.equal(fresh.net.state_dict()[key], tensor), key
    orig_opt = original.optimizers["opt"]["state"]
    fresh_opt = fresh.opt.state_dict()["state"]
    for idx in orig_opt:
        for key, value in orig_opt[idx].items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(fresh_opt[idx][key], value), f"opt[{idx}].{key}"
            else:
                assert fresh_opt[idx][key] == value
    assert fresh.entropy_coef == trainer.entropy_coef
    assert np.array_equal(fresh.priorities, trainer.priorities)
    assert np.array_equal(fresh.buffer_obs, trainer.buffer_obs)
    assert fresh.step_count == trainer.step_count
    assert np.array_equal(fresh.rng.numpy_generator("buffer").random(10), expected_np)
    assert torch.equal(torch.rand(10, generator=fresh.rng.torch_generator("data")), expected_torch)


def test_gate_kill_resume_matches_uninterrupted(tmp_path: Path) -> None:
    """Scenario 2: N + (kill) + M == N+M, seed for seed."""
    n_steps, m_steps = 25, 17

    reference = DummyTrainer(seed=33)
    reference.run(n_steps + m_steps)

    session = create_session(tmp_path, today=date(2026, 6, 13))
    store = BufferStore(session.buffer)
    first = DummyTrainer(seed=33)
    first.run(n_steps)
    save_checkpoint(session, first.training_state(), step=first.step_count)
    store.snapshot(first.buffer_arrays(), {"size": first.step_count}, step=first.step_count)
    pre_kill_losses = list(first.losses)
    del first  # simulated kill: all live objects gone

    resumed = DummyTrainer(seed=777)  # wrong seed; restore must fully rebuild
    loaded = load_latest_checkpoint(session)
    snapshot = store.load_latest()
    assert loaded is not None and snapshot is not None
    resumed.restore(loaded[0], snapshot.arrays)
    assert resumed.step_count == n_steps  # the counter continues, not restarts
    resumed.run(m_steps)

    assert pre_kill_losses + resumed.losses == reference.losses
    assert resumed.step_count == reference.step_count
    for key, tensor in reference.net.state_dict().items():
        assert torch.equal(resumed.net.state_dict()[key], tensor), key
    assert np.array_equal(resumed.buffer_obs, reference.buffer_obs)
    assert np.array_equal(resumed.priorities, reference.priorities)


# Child script for scenario 3: a crashing trainer that uses the *real*
# session machinery (paths/checkpoint/buffer_store) and the watchdog's
# resume signal. Crashes after each work slice until the target is reached.
CHILD_TRAINER = """
import os
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[4])  # test dir, for dummy_trainer import
from dummy_trainer import DummyTrainer

from acagent.session.buffer_store import BufferStore
from acagent.session.checkpoint import load_latest_checkpoint, save_checkpoint
from acagent.session.paths import create_session, find_latest_session

base = Path(sys.argv[1])
target_steps = int(sys.argv[2])
slice_steps = int(sys.argv[3])

session = find_latest_session(base) or create_session(base)
store = BufferStore(session.buffer)
trainer = DummyTrainer(seed=55)

loaded = load_latest_checkpoint(session)
snapshot = store.load_latest()
if loaded is not None:
    assert os.environ["ACAGENT_RESUMED"] == "1", "resume signal missing on restart"
    trainer.restore(loaded[0], snapshot.arrays)

trainer.run(slice_steps)

save_checkpoint(session, trainer.training_state(), step=trainer.step_count)
store.snapshot(trainer.buffer_arrays(), {"size": trainer.step_count}, step=trainer.step_count)

with open(base / "losses.log", "a") as f:
    for loss in trainer.losses:
        f.write(f"{loss!r}\\n")
with open(base / "progress.log", "a") as f:
    f.write(f"{trainer.step_count}\\n")

if trainer.step_count < target_steps:
    sys.exit(1)  # simulated crash mid-training
sys.exit(0)
"""


def test_gate_watchdog_restarts_and_child_resumes(tmp_path: Path) -> None:
    """Scenario 3: crash -> watchdog restart -> autonomous resume, no human input."""
    script = tmp_path / "child_trainer.py"
    script.write_text(CHILD_TRAINER, encoding="utf-8")
    test_dir = str(Path(__file__).parent)

    def run_session(base: Path, target: int, slice_steps: int) -> None:
        base.mkdir()
        result = Watchdog(
            [sys.executable, str(script), str(base), str(target), str(slice_steps), test_dir],
            max_restarts=10,
            backoff_s=0.0,
        ).run()
        assert result.final_exit_code == 0, "child never reached a clean exit"

    # Interrupted: 12 steps in slices of 4 -> two crashes, two resumes.
    crashy = tmp_path / "crashy"
    run_session(crashy, target=12, slice_steps=4)
    progress = [int(line) for line in (crashy / "progress.log").read_text().splitlines()]
    assert progress == [4, 8, 12]  # the step counter continues across restarts

    # Uninterrupted: same script, one slice of 12, separate session base.
    smooth = tmp_path / "smooth"
    run_session(smooth, target=12, slice_steps=12)

    interrupted = (crashy / "losses.log").read_text().splitlines()
    uninterrupted = (smooth / "losses.log").read_text().splitlines()
    assert len(interrupted) == len(uninterrupted) == 12
    assert interrupted == uninterrupted  # bit-for-bit identical trajectory
