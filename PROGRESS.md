# PROGRESS.md — Build State

Claude Code: update this file after every completed task and every gate result. The first unchecked phase below is where work resumes.

## Phase gates

| Phase | Scope | Automated gate | HIL gate | Status |
|-------|-------|----------------|----------|--------|
| 0  | Scaffold, checkpointing, crash resilience | `pytest -m phase0` | — (none) | ◐ in progress |
| 1  | Telemetry bridge, actuator, failsafe, reset | `pytest -m phase1` | `scripts/hil_phase1.py` (2 h hold) | ☐ not started |
| 1b | Setup-apply spike | `pytest -m phase1b` | `scripts/hil_phase1b.py` (pressure change verified) | ☐ not started |
| 2  | Spline pipeline, normalization, GPU proofing | `pytest -m phase2` | `scripts/hil_phase2_endurance.py` (1 h, zero TDR) | ☐ not started |
| 3  | SAC + PER + reward + async trainer | `pytest -m phase3` | Simple-track clean laps | ☐ not started |
| 4  | Plateau scheduler + Optuna offline sweeps | `pytest -m phase4` | Sabotage recovery + sweep win | ☐ not started |
| 5  | Setup optimizer (Module E) | `pytest -m phase5` | Unattended setup convergence | ☐ not started |

Status values: `☐ not started` → `◐ in progress` → `✅ A-gate passed` → `✅✅ complete (A + HIL recorded)`

## Task log

<!-- Append entries newest-first: date, phase, task ID, one-line result, commit hash. -->

- 2026-06-13 · P0 · T0.7 · session/buffer_store.py: cadenced .npy snapshots (manifest-last consistency protocol, temp-dir rename), latest-consistent load skipping torn snapshots, mmap option, keep_last retention. 13 tests pass. · (this commit)
- 2026-06-13 · P0 · T0.6 · session/checkpoint.py: TrainingState (networks/optimizers/entropy coef/PER state/counters/RNG), atomic save, list/load-latest with corrupt-file fallback, optional keep_last pruning (Invariant 10). 9 tests pass incl. exact-equality round-trip into fresh objects. · (this commit)
- 2026-06-13 · P0 · T0.5 · session/metadata.py: run_metadata.json (config snapshot, normalization placeholder, versions, git hash) + atomic.py shared atomic-write helper; update_metadata for later phases. 9 tests pass. · (this commit)
- 2026-06-12 · P0 · T0.4 · session/paths.py: session_YYYYMMDD layout (+_NN collision suffix), open/create, latest-session discovery ignoring non-session entries. 11 tests pass. · (this commit)
- 2026-06-12 · P0 · T0.3 · rng.py RngManager: named numpy/torch child generators derived from (seed, name), globals seeded, full state export/import (Invariants 7, 10). 13 tests pass. · (this commit)
- 2026-06-12 · P0 · T0.2 · config.py dataclass tree + configs/default.yaml; loader rejects unknown keys with path, strict types, range validation; γ and all w_* present; frozen (Invariant 2). 24 tests pass. · (this commit)
- 2026-06-12 · P0 · T0.1 · Repo scaffold: git init + docs commit (5a7d5f2), pyproject (deps, pytest markers, ruff), package skeleton, conftest, REFERENCES.md stub, .venv on Python 3.12.10, smoke tests pass, ruff clean. · (this commit)

## Gate results

<!-- Append full gate outcomes here: command, pass/fail, notes, and for HIL gates the user's reported observations. -->

## Open issues / deviations from plan

<!-- Anything that required deviating from BUILD_PLAN.md or SPEC.md goes here WITH the user's explicit approval noted. -->
