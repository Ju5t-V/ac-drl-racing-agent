# CLAUDE.md — AC DRL Racing Agent

Autonomous Contextual SAC racing agent for Assetto Corsa. Native Windows 11, single machine.
Design authority: `docs/SPEC.md` (v2.1). Build authority: `docs/BUILD_PLAN.md`. State: `PROGRESS.md`.
If these documents ever conflict, STOP and ask the user. Do not improvise a resolution.

## Environment

- **Target machine:** Windows 11, AMD RX 9070 XT (RDNA 4, gfx1201), Assetto Corsa + vJoy installed.
- **Python:** 3.12, single venv at `.venv`. PyTorch 2.8+ ROCm wheel on the target machine only; plain CPU `torch` from PyPI is fine for development and all default tests.
- **C#:** .NET 8 SDK for `bridge/` (telemetry bridge + FakeAc test harness).
- **You (Claude Code) may not have AC, vJoy, or the GPU available.** Everything you build must be testable without them. Hardware-in-the-loop (HIL) gates are run by the user.

## Hard invariants — never violate, never "refactor away"

1. The replay buffer is NEVER flushed, cleared, or truncated by any code path. Off-policy data stays valid across all hyperparameter changes.
2. Discount factor γ is read from config once per run, is immutable mid-run, and appears in NO sweep space.
3. PER priorities come from |TD error| only. No reconstruction error, no novelty score, in the priority path.
4. The ego-centric spline transform executes every control step in the live path. No frame-skipping, no lazy evaluation in inference.
5. Inference and training default to eager mode. `torch.compile` lives behind a config flag inside try/except with eager fallback.
6. Vehicle setup changes go through AC setup files applied at stint boundaries. Never attempt in-session physics `.ini` hot-reload.
7. All randomness flows through one seeded RNG manager. Checkpoints capture and restore RNG state.
8. Every hardware-touching component (telemetry, actuator, reset, setup apply, GPU) has a mock or synthetic implementation behind the same interface.
9. Reward weights are named `w_*`. The symbols γ and ω are reserved for the discount factor and the setup vector respectively.
10. Checkpoints must fully reconstruct training: network weights, optimizer state, entropy coefficient, PER priorities, buffer, step counters, RNG.

## Repository conventions

- Layout per `docs/BUILD_PLAN.md` §"Repository layout". Source under `src/acagent/`, tests under `tests/`, C# under `bridge/`, user-run harnesses under `scripts/`.
- Type hints on every public function. Config via dataclasses loaded from `configs/*.yaml` — no magic constants in logic code.
- Logging via `logging`, never `print`, in library code.
- Thread-shared structures (replay buffer, telemetry queue) document their locking strategy in the module docstring.
- External formats (AC shared memory pages, `fast_lane.ai`, AC setup files, vJoy API) are NEVER invented from memory. Find official/community documentation first, implement against it, and record source URLs in code comments and `docs/REFERENCES.md`.

## Testing

- Framework: `pytest`. Markers: `unit` (default), `integration` (needs dotnet/loopback), `gpu` (needs ROCm GPU), `hil` (needs AC + user).
- Default run must pass on a clean CPU-only machine: `pytest -m "not gpu and not hil"`.
- Each BUILD_PLAN phase ends with an automated gate command. The gate must pass before the next phase begins. HIL gates: you prepare the harness + instructions, then STOP and wait for the user's result.

## Commands

```
.venv\Scripts\activate                      # Windows venv
pytest -m "not gpu and not hil"            # default test suite
pytest -m phase0                            # per-phase gate (phase0..phase5)
ruff check src tests                        # lint
ruff format src tests                       # format
dotnet build bridge/AcBridge                # C# bridge
dotnet run --project bridge/FakeAc          # synthetic AC for headless integration tests
```

## Workflow rules

1. Work strictly one phase at a time, in BUILD_PLAN order. Find the first incomplete phase in `PROGRESS.md`.
2. At phase start, present a short task plan and wait for approval.
3. Implement each task with its tests in the same task. Run them before moving on.
4. At phase end, run the phase gate command and show the full output.
5. Update `PROGRESS.md` after every completed task and every gate result.
6. Commit per task: `phaseN: <imperative summary>`. Never commit with failing default tests.
7. Never start live training, install drivers, or modify system settings. Anything requiring AC, vJoy, the GPU, or admin rights is a user action — prepare it, document it, stop.
