"""T0.7 tests: snapshot round-trip with priorities, torn-snapshot skip, cadence."""

import json
from pathlib import Path

import numpy as np
import pytest

from acagent.session.buffer_store import BufferStore, LoadedSnapshot

pytestmark = [pytest.mark.unit, pytest.mark.phase0]


def make_arrays(seed: int = 0, n: int = 256) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "observations": rng.normal(size=(n, 24)).astype(np.float32),
        "actions": rng.normal(size=(n, 3)).astype(np.float32),
        "rewards": rng.normal(size=n).astype(np.float32),
        "priorities": np.abs(rng.normal(size=n)).astype(np.float64),
    }


META = {"size": 256, "insert_index": 256, "max_priority": 2.5}


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture()
def store(tmp_path: Path) -> BufferStore:
    return BufferStore(tmp_path / "buffer", interval_s=900.0, keep_last=2)


class TestRoundTrip:
    def test_snapshot_and_load_latest(self, store: BufferStore) -> None:
        arrays = make_arrays()
        store.snapshot(arrays, META, step=500)
        loaded = store.load_latest()
        assert isinstance(loaded, LoadedSnapshot)
        assert loaded.step == 500
        assert loaded.meta == META
        assert set(loaded.arrays) == set(arrays)
        for name, array in arrays.items():
            assert np.array_equal(loaded.arrays[name], array), name
            assert loaded.arrays[name].dtype == array.dtype, name

    def test_memory_mappable(self, store: BufferStore) -> None:
        store.snapshot(make_arrays(), META, step=1)
        loaded = store.load_latest(mmap=True)
        assert loaded is not None
        assert all(isinstance(a, np.memmap) for a in loaded.arrays.values())
        del loaded  # release Windows file locks before tmp cleanup

    def test_latest_of_several(self, store: BufferStore) -> None:
        for step in (10, 30, 20):
            store.snapshot(make_arrays(step), META, step=step)
        loaded = store.load_latest()
        assert loaded is not None
        assert loaded.step == 30

    def test_empty_store_returns_none(self, store: BufferStore) -> None:
        assert store.load_latest() is None


class TestConsistency:
    def test_snapshot_without_manifest_skipped(self, store: BufferStore) -> None:
        good = make_arrays(1)
        store.snapshot(good, META, step=100)
        torn = store.directory / "snap_000000000200"
        torn.mkdir()
        np.save(torn / "observations.npy", make_arrays(2)["observations"])
        loaded = store.load_latest()
        assert loaded is not None
        assert loaded.step == 100

    def test_truncated_array_file_skipped(self, store: BufferStore) -> None:
        store.snapshot(make_arrays(1), META, step=100)
        newer = store.snapshot(make_arrays(2), META, step=200)
        target = newer / "observations.npy"
        data = target.read_bytes()
        target.write_bytes(data[: len(data) // 2])  # simulate torn write
        loaded = store.load_latest()
        assert loaded is not None
        assert loaded.step == 100

    def test_garbage_manifest_skipped(self, store: BufferStore) -> None:
        store.snapshot(make_arrays(1), META, step=100)
        newer = store.snapshot(make_arrays(2), META, step=200)
        (newer / "manifest.json").write_text("{not json", encoding="utf-8")
        loaded = store.load_latest()
        assert loaded is not None
        assert loaded.step == 100

    def test_leftover_temp_dirs_ignored_and_cleaned(self, store: BufferStore) -> None:
        leftover = store.directory / ".tmp_snap_deadbeef"
        leftover.mkdir()
        (leftover / "junk.npy").write_bytes(b"junk")
        assert store.load_latest() is None
        store.snapshot(make_arrays(), META, step=1)
        assert not leftover.exists()

    def test_manifest_records_shapes_and_dtypes(self, store: BufferStore) -> None:
        snap = store.snapshot(make_arrays(), META, step=7)
        manifest = json.loads((snap / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["step"] == 7
        assert manifest["arrays"]["observations"]["shape"] == [256, 24]
        assert manifest["arrays"]["priorities"]["dtype"] == "float64"


class TestCadenceAndRetention:
    def test_first_call_snapshots_then_waits_interval(self, tmp_path: Path) -> None:
        clock = FakeClock()
        store = BufferStore(tmp_path / "b", interval_s=900.0, now_fn=clock)
        assert store.maybe_snapshot(make_arrays(), META, step=1) is not None
        clock.t += 100.0
        assert store.maybe_snapshot(make_arrays(), META, step=2) is None
        clock.t += 900.0
        assert store.maybe_snapshot(make_arrays(), META, step=3) is not None
        loaded = store.load_latest()
        assert loaded is not None
        assert loaded.step == 3

    def test_keep_last_prunes_oldest(self, tmp_path: Path) -> None:
        store = BufferStore(tmp_path / "b", keep_last=2)
        for step in (1, 2, 3, 4):
            store.snapshot(make_arrays(step), META, step=step)
        names = sorted(p.name for p in store.directory.iterdir())
        assert names == ["snap_000000000003", "snap_000000000004"]

    def test_invalid_construction_args_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            BufferStore(tmp_path, interval_s=0)
        with pytest.raises(ValueError):
            BufferStore(tmp_path, keep_last=0)

    def test_empty_arrays_rejected(self, store: BufferStore) -> None:
        with pytest.raises(ValueError):
            store.snapshot({}, META, step=1)
