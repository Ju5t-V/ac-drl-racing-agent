# PROGRESS.md — Build State

Claude Code: update this file after every completed task and every gate result. The first unchecked phase below is where work resumes.

## Phase gates

| Phase | Scope | Automated gate | HIL gate | Status |
|-------|-------|----------------|----------|--------|
| 0  | Scaffold, checkpointing, crash resilience | `pytest -m phase0` | — (none) | ☐ not started |
| 1  | Telemetry bridge, actuator, failsafe, reset | `pytest -m phase1` | `scripts/hil_phase1.py` (2 h hold) | ☐ not started |
| 1b | Setup-apply spike | `pytest -m phase1b` | `scripts/hil_phase1b.py` (pressure change verified) | ☐ not started |
| 2  | Spline pipeline, normalization, GPU proofing | `pytest -m phase2` | `scripts/hil_phase2_endurance.py` (1 h, zero TDR) | ☐ not started |
| 3  | SAC + PER + reward + async trainer | `pytest -m phase3` | Simple-track clean laps | ☐ not started |
| 4  | Plateau scheduler + Optuna offline sweeps | `pytest -m phase4` | Sabotage recovery + sweep win | ☐ not started |
| 5  | Setup optimizer (Module E) | `pytest -m phase5` | Unattended setup convergence | ☐ not started |

Status values: `☐ not started` → `◐ in progress` → `✅ A-gate passed` → `✅✅ complete (A + HIL recorded)`

## Task log

<!-- Append entries newest-first: date, phase, task ID, one-line result, commit hash. -->

## Gate results

<!-- Append full gate outcomes here: command, pass/fail, notes, and for HIL gates the user's reported observations. -->

## Open issues / deviations from plan

<!-- Anything that required deviating from BUILD_PLAN.md or SPEC.md goes here WITH the user's explicit approval noted. -->
