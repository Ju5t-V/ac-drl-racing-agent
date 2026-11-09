# AC DRL Racing Agent

An autonomous **Soft Actor-Critic (SAC)** deep reinforcement learning agent that learns to drive in *Assetto Corsa*. It runs natively on Windows on a single machine with an AMD GPU (ROCm).

**Planned architecture:** a C# telemetry bridge reads the sim's shared memory and drives vJoy for control. A Python trainer runs SAC with prioritized experience replay, spline-based track normalization, a plateau scheduler, Optuna hyperparameter sweeps, and a car-setup optimizer.

The build is split into gated phases, and every component is testable without the game or GPU (see `docs/SPEC.md` and `docs/BUILD_PLAN.md`).

| Phase | Scope | Status |
|---|---|---|
| 0 | Scaffold, checkpointing, crash resilience (crash-consistent replay-buffer snapshots, watchdog with resume) | ✅ complete (92 tests) |
| 1 to 5 | Telemetry bridge, spline pipeline, SAC trainer, Optuna sweeps, setup optimizer | ☐ planned |

```bash
pip install -e .[dev]
pytest -m phase0
```

## Status
🟠 Work in progress. The foundation (Phase 0) is done and tested.
