# Claude Code Kickoff — AC DRL Racing Agent

## One-time setup (do this before launching Claude Code)

1. Create the repo and place the documents:

```
mkdir ac-drl-agent
cd ac-drl-agent
git init
mkdir docs
# place files:
#   CLAUDE.md        → repo root
#   PROGRESS.md      → repo root
#   SPEC.md          → docs/SPEC.md
#   BUILD_PLAN.md    → docs/BUILD_PLAN.md
git add -A && git commit -m "phase0: project documents"
```

2. Install prerequisites on the dev machine: Python 3.12, .NET 8 SDK, git. (AC, vJoy, and the ROCm GPU are only needed for HIL gates.)
3. Launch `claude` inside `ac-drl-agent/` and paste the prompt below.

---

## The prompt (paste into Claude Code)

```
You are building the AC DRL Racing Agent in this repository.

Before writing any code, read these four documents in full, in this order:
1. CLAUDE.md          — hard invariants and working rules. These are non-negotiable.
2. docs/SPEC.md       — the v2.1 design authority. Module behavior, math, and gates come from here.
3. docs/BUILD_PLAN.md — the build authority. Phases, tasks, repository layout, and gate commands.
4. PROGRESS.md        — current build state. Resume at the first incomplete phase (initially Phase 0).

If any two documents conflict, stop and ask me. Do not improvise a resolution.

Working rules for this entire project:
- Work strictly one phase at a time in BUILD_PLAN order. Within a phase, follow the task list (T·) in order unless you propose and I approve a reordering.
- At the start of each phase: present a concise task plan (files you will create, tests you will write, the gate you will run) and WAIT for my approval before implementing.
- Implement each task together with its tests. Run the tests before moving to the next task.
- At phase end: run the phase's automated gate command (e.g. `pytest -m phase0`) and show me the full output. The default suite `pytest -m "not gpu and not hil"` must also pass.
- HIL gates: deliver the harness script plus a numbered instruction block for me, then STOP and wait for my reported result. Never proceed past an HIL gate on assumptions.
- Update PROGRESS.md after every completed task and every gate result. Commit per task with message format `phaseN: <imperative summary>`. Never commit with failing default tests.
- Never invent external formats. AC shared memory layouts, the AC setup-file format, fast_lane.ai, and the vJoy API must be implemented from official or community documentation that you locate first. Record every source URL in docs/REFERENCES.md and cite it in code comments at the point of use. If documentation cannot be found or is ambiguous, ask me — I can extract sample files from my AC installation as fixtures.
- Everything you build must be testable on this machine with CPU-only torch, no Assetto Corsa, no vJoy, and no GPU. Hardware-touching code goes behind the interfaces defined in BUILD_PLAN (TelemetrySource, Actuator, FakeAc, the synthetic track env) so the automated gates run headless.
- Never start live training, install drivers, or change system settings. Anything needing AC, vJoy, the GPU, or admin rights is mine to run — prepare it, document it, stop.

Begin now: confirm you have read all four documents by listing the ten hard invariants from CLAUDE.md in your own words, then propose the Phase 0 task plan.
```

---

## Notes

- The final line of the prompt (restating the invariants) is deliberate: it verifies the documents were actually ingested before any code is written.
- When you finish a HIL gate on the real machine, report the result in chat and tell Claude Code to record it in PROGRESS.md; it will then propose the next phase's plan.
- If a Claude Code session ends mid-phase, just relaunch and paste the same prompt — PROGRESS.md carries the state.
