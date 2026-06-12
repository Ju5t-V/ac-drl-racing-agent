"""T0.6 tests: exact-equality checkpoint round-trip, atomicity, corrupt fallback.

The round-trip test follows G0.A: dummy actor/critic/optimizer + entropy
coefficient + PER priorities + counters + RNG; reload into fresh objects and
assert exact equality of every tensor and identical RNG draws after restore.
"""

from datetime import date
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from acagent.rng import RngManager
from acagent.session import checkpoint as ckpt
from acagent.session.checkpoint import (
    TrainingState,
    list_checkpoints,
    load_checkpoint,
    load_latest_checkpoint,
    save_checkpoint,
)
from acagent.session.paths import create_session

pytestmark = [pytest.mark.unit, pytest.mark.phase0]


def make_mlp(seed: int) -> nn.Sequential:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 2))


def build_training_state(seed: int = 11) -> tuple[TrainingState, RngManager]:
    """Dummy actor/critic with *non-trivial* optimizer state (a few Adam steps)."""
    actor = make_mlp(seed)
    critic = make_mlp(seed + 1)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-3)
    for _ in range(3):
        loss = actor(torch.randn(16, 8)).pow(2).mean() + critic(torch.randn(16, 8)).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    rng = RngManager(seed=seed)
    rng.numpy_generator("sampler").random(100)  # advance so state is mid-stream
    torch.rand(7, generator=rng.torch_generator("learner"))

    state = TrainingState(
        networks={"actor": actor.state_dict(), "critic": critic.state_dict()},
        optimizers={"adam": opt.state_dict()},
        entropy_coef=0.137,
        per_state={
            "priorities": np.abs(np.random.default_rng(seed).normal(size=512)),
            "max_priority": 3.7,
        },
        counters={"env_steps": 12345, "grad_steps": 6789, "episodes": 42},
        rng_state=rng.get_state(),
    )
    return state, rng


@pytest.fixture()
def session(tmp_path: Path):
    return create_session(tmp_path, today=date(2026, 6, 12))


class TestRoundTrip:
    def test_full_exact_equality(self, session) -> None:
        state, rng = build_training_state()
        expected_np = rng.numpy_generator("sampler").random(20)
        expected_torch = torch.rand(20, generator=rng.torch_generator("learner"))

        save_checkpoint(session, state, step=12345)
        loaded = load_checkpoint(ckpt.checkpoint_path(session, 12345))

        for net in ("actor", "critic"):
            assert state.networks[net].keys() == loaded.networks[net].keys()
            for key, tensor in state.networks[net].items():
                assert torch.equal(tensor, loaded.networks[net][key]), f"{net}.{key}"

        original_opt = state.optimizers["adam"]["state"]
        loaded_opt = loaded.optimizers["adam"]["state"]
        assert original_opt.keys() == loaded_opt.keys()
        for idx in original_opt:
            for key, value in original_opt[idx].items():
                if isinstance(value, torch.Tensor):
                    assert torch.equal(value, loaded_opt[idx][key]), f"opt[{idx}].{key}"
                else:
                    assert value == loaded_opt[idx][key]
        assert state.optimizers["adam"]["param_groups"] == loaded.optimizers["adam"]["param_groups"]

        assert loaded.entropy_coef == state.entropy_coef
        assert np.array_equal(loaded.per_state["priorities"], state.per_state["priorities"])
        assert loaded.per_state["max_priority"] == state.per_state["max_priority"]
        assert loaded.counters == state.counters

        # RNG draws after restore must match the uninterrupted stream.
        restored = RngManager(seed=0)
        restored.set_state(loaded.rng_state)
        assert np.array_equal(restored.numpy_generator("sampler").random(20), expected_np)
        assert torch.equal(
            torch.rand(20, generator=restored.torch_generator("learner")), expected_torch
        )

    def test_fresh_networks_load_state_dicts(self, session) -> None:
        """Reload into *fresh* module/optimizer objects, per G0.A wording."""
        state, _ = build_training_state()
        save_checkpoint(session, state, step=1)
        loaded = load_checkpoint(ckpt.checkpoint_path(session, 1))

        fresh_actor = make_mlp(seed=999)  # different init on purpose
        fresh_actor.load_state_dict(loaded.networks["actor"])
        for key, tensor in state.networks["actor"].items():
            assert torch.equal(fresh_actor.state_dict()[key], tensor)

        fresh_opt = torch.optim.Adam(fresh_actor.parameters())
        fresh_opt.load_state_dict(
            {
                "state": {
                    k: v
                    for k, v in loaded.optimizers["adam"]["state"].items()
                    if k < len(list(fresh_actor.parameters()))
                },
                "param_groups": [
                    {
                        **loaded.optimizers["adam"]["param_groups"][0],
                        "params": list(range(len(list(fresh_actor.parameters())))),
                    }
                ],
            }
        )
        assert fresh_opt.state_dict()["param_groups"][0]["lr"] == 1e-3


class TestAtomicityAndDiscovery:
    def test_failed_save_preserves_previous_checkpoint(self, session, monkeypatch) -> None:
        state, _ = build_training_state()
        save_checkpoint(session, state, step=1)

        def explode(path, data):
            raise OSError("disk on fire")

        monkeypatch.setattr(ckpt, "atomic_write_bytes", explode)
        with pytest.raises(OSError):
            save_checkpoint(session, state, step=1)
        monkeypatch.undo()

        loaded = load_checkpoint(ckpt.checkpoint_path(session, 1))
        assert loaded.counters == state.counters

    def test_no_temp_files_left_after_interrupted_write(self, session, monkeypatch) -> None:
        import os as os_module

        from acagent.session import atomic

        def explode(src, dst):
            raise OSError("replace failed")

        monkeypatch.setattr(atomic.os, "replace", explode)
        with pytest.raises(OSError):
            atomic.atomic_write_bytes(session.checkpoints / "x.bin", b"data")
        monkeypatch.undo()
        leftovers = [p for p in session.checkpoints.iterdir() if "tmp" in p.name]
        assert leftovers == []
        assert not (session.checkpoints / "x.bin").exists()
        assert os_module.path.isdir(session.checkpoints)

    def test_list_and_latest(self, session) -> None:
        state, _ = build_training_state()
        for step in (5, 100, 20):
            save_checkpoint(session, state, step=step)
        assert [int(p.stem.split("_")[1]) for p in list_checkpoints(session)] == [5, 20, 100]
        result = load_latest_checkpoint(session)
        assert result is not None
        _, step = result
        assert step == 100

    def test_latest_skips_corrupt_file(self, session) -> None:
        state, _ = build_training_state()
        save_checkpoint(session, state, step=10)
        bad = session.checkpoints / "ckpt_000000000099.pt"
        bad.write_bytes(b"not a checkpoint")
        result = load_latest_checkpoint(session)
        assert result is not None
        loaded, step = result
        assert step == 10
        assert loaded.counters == state.counters

    def test_latest_none_when_empty(self, session) -> None:
        assert load_latest_checkpoint(session) is None

    def test_keep_last_prunes(self, session) -> None:
        state, _ = build_training_state()
        for step in range(1, 6):
            save_checkpoint(session, state, step=step, keep_last=3)
        assert [int(p.stem.split("_")[1]) for p in list_checkpoints(session)] == [3, 4, 5]

    def test_negative_step_rejected(self, session) -> None:
        state, _ = build_training_state()
        with pytest.raises(ValueError):
            save_checkpoint(session, state, step=-1)
