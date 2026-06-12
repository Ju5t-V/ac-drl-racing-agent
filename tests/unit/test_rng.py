"""T0.3 tests: deterministic named streams, independence, state round-trip."""

import random

import numpy as np
import pytest
import torch

from acagent.rng import RngManager

pytestmark = [pytest.mark.unit, pytest.mark.phase0]


class TestDeterminism:
    def test_same_seed_same_name_identical_numpy_streams(self) -> None:
        a = RngManager(seed=7).numpy_generator("actor")
        b = RngManager(seed=7).numpy_generator("actor")
        assert np.array_equal(a.random(100), b.random(100))

    def test_same_seed_same_name_identical_torch_streams(self) -> None:
        a = RngManager(seed=7).torch_generator("learner")
        b = RngManager(seed=7).torch_generator("learner")
        assert torch.equal(torch.rand(100, generator=a), torch.rand(100, generator=b))

    def test_different_names_independent_streams(self) -> None:
        mgr = RngManager(seed=7)
        a = mgr.numpy_generator("actor").random(100)
        b = mgr.numpy_generator("buffer").random(100)
        assert not np.array_equal(a, b)

    def test_different_seeds_different_streams(self) -> None:
        a = RngManager(seed=7).numpy_generator("actor").random(100)
        b = RngManager(seed=8).numpy_generator("actor").random(100)
        assert not np.array_equal(a, b)

    def test_creation_order_does_not_matter(self) -> None:
        """A child's stream is a pure function of (seed, name)."""
        mgr_a = RngManager(seed=7)
        mgr_a.numpy_generator("first")
        a = mgr_a.numpy_generator("second").random(50)
        b = RngManager(seed=7).numpy_generator("second").random(50)
        assert np.array_equal(a, b)

    def test_same_name_returns_same_generator_object(self) -> None:
        mgr = RngManager(seed=7)
        assert mgr.numpy_generator("x") is mgr.numpy_generator("x")
        assert mgr.torch_generator("x") is mgr.torch_generator("x")

    def test_globals_seeded_deterministically(self) -> None:
        RngManager(seed=7)
        a = (torch.rand(10), random.random())
        RngManager(seed=7)
        b = (torch.rand(10), random.random())
        assert torch.equal(a[0], b[0])
        assert a[1] == b[1]

    def test_negative_seed_rejected(self) -> None:
        with pytest.raises(ValueError):
            RngManager(seed=-1)


class TestStateRoundTrip:
    def test_numpy_stream_continues_after_restore(self) -> None:
        mgr = RngManager(seed=3)
        mgr.numpy_generator("actor").random(17)  # advance
        state = mgr.get_state()
        expected = mgr.numpy_generator("actor").random(50)

        fresh = RngManager(seed=0)  # wrong seed on purpose; restore must win
        fresh.set_state(state)
        assert np.array_equal(fresh.numpy_generator("actor").random(50), expected)

    def test_torch_stream_continues_after_restore(self) -> None:
        mgr = RngManager(seed=3)
        torch.rand(17, generator=mgr.torch_generator("learner"))
        state = mgr.get_state()
        expected = torch.rand(50, generator=mgr.torch_generator("learner"))

        fresh = RngManager(seed=0)
        fresh.set_state(state)
        restored = torch.rand(50, generator=fresh.torch_generator("learner"))
        assert torch.equal(restored, expected)

    def test_global_streams_continue_after_restore(self) -> None:
        mgr = RngManager(seed=3)
        torch.rand(5)
        random.random()
        state = mgr.get_state()
        expected_torch = torch.rand(20)
        expected_py = [random.random() for _ in range(20)]

        fresh = RngManager(seed=99)
        fresh.set_state(state)
        assert torch.equal(torch.rand(20), expected_torch)
        assert [random.random() for _ in range(20)] == expected_py

    def test_new_child_after_restore_matches_original_run(self) -> None:
        """Names absent from the snapshot derive from the restored seed."""
        mgr = RngManager(seed=3)
        mgr.numpy_generator("existing")
        state = mgr.get_state()
        expected = mgr.numpy_generator("brand_new").random(50)

        fresh = RngManager(seed=0)
        fresh.set_state(state)
        assert np.array_equal(fresh.numpy_generator("brand_new").random(50), expected)
        assert fresh.seed == 3

    def test_snapshot_is_isolated_from_later_draws(self) -> None:
        """get_state returns copies, not views that keep advancing."""
        mgr = RngManager(seed=3)
        gen = mgr.torch_generator("learner")
        state = mgr.get_state()
        before = state["torch"]["learner"].clone()
        torch.rand(100, generator=gen)  # advance after snapshot
        assert torch.equal(state["torch"]["learner"], before)
