"""Dummy trainer used by the G0.A gate tests (in-process and as a subprocess).

Minimal stand-in for the Phase 3 trainer: net + Adam + ring buffer with
priorities + entropy coefficient + step counter. Every source of randomness
flows through RngManager (Invariant 7), so a restored trainer continues the
exact trajectory of the original.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from acagent.rng import RngManager
from acagent.session.checkpoint import TrainingState

BUFFER_CAPACITY = 64
OBS_DIM = 4


class DummyTrainer:
    def __init__(self, seed: int) -> None:
        self.rng = RngManager(seed)
        self.net = nn.Sequential(nn.Linear(OBS_DIM, 16), nn.Tanh(), nn.Linear(16, 1))
        self.opt = torch.optim.Adam(self.net.parameters(), lr=1e-2)
        self.buffer_obs = np.zeros((BUFFER_CAPACITY, OBS_DIM), dtype=np.float32)
        self.priorities = np.zeros(BUFFER_CAPACITY, dtype=np.float64)
        self.entropy_coef = 1.0
        self.step_count = 0
        self.losses: list[float] = []

    def step(self) -> None:
        x = torch.randn(8, OBS_DIM, generator=self.rng.torch_generator("data"))
        loss = self.net(x).pow(2).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        row = self.rng.numpy_generator("buffer").normal(size=OBS_DIM).astype(np.float32)
        idx = self.step_count % BUFFER_CAPACITY
        self.buffer_obs[idx] = row
        self.priorities[idx] = abs(float(loss.item())) + 1e-3
        self.entropy_coef *= 0.999
        self.step_count += 1
        self.losses.append(float(loss.item()))

    def run(self, steps: int) -> None:
        for _ in range(steps):
            self.step()

    def training_state(self) -> TrainingState:
        return TrainingState(
            networks={"net": self.net.state_dict()},
            optimizers={"opt": self.opt.state_dict()},
            entropy_coef=self.entropy_coef,
            per_state={"max_priority": float(self.priorities.max())},
            counters={"env_steps": self.step_count},
            rng_state=self.rng.get_state(),
        )

    def buffer_arrays(self) -> dict[str, np.ndarray]:
        return {"obs": self.buffer_obs, "priorities": self.priorities}

    def restore(self, state: TrainingState, arrays: dict[str, np.ndarray]) -> None:
        self.net.load_state_dict(state.networks["net"])
        self.opt.load_state_dict(state.optimizers["opt"])
        self.entropy_coef = state.entropy_coef
        self.step_count = state.counters["env_steps"]
        self.rng.set_state(state.rng_state)
        self.buffer_obs = np.array(arrays["obs"], dtype=np.float32, copy=True)
        self.priorities = np.array(arrays["priorities"], dtype=np.float64, copy=True)
