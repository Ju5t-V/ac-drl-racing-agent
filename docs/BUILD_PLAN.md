# BUILD_PLAN.md — Phased Implementation Plan

Build authority for the AC DRL Racing Agent. Design rationale lives in `docs/SPEC.md` (v2.1); this document translates it into buildable, gated phases. Rules of engagement are in `CLAUDE.md`.

## How gating works

- **Automated gate (G·A):** a pytest command Claude Code runs itself. Must pass on CPU, with no AC, vJoy, or GPU present.
- **HIL gate (G·H):** requires the real machine, AC, and the user. Claude Code delivers a `scripts/hil_*.py` harness plus a numbered instruction block, then STOPS. The user runs it and reports results; the gate result is recorded in `PROGRESS.md` before the next phase begins.
- A phase is complete only when both its gates are recorded. No forward references: code for phase N must not import from phase N+1 modules.

## Key build decisions (fixed — do not relitigate)

1. **Self-contained SAC, not Stable-Baselines3.** SPEC Risk 3 anticipated SB3's abstractions fighting TD-error PER + contextual states + the async learner. A custom SAC (~600 lines) is the committed path. SB3 is not a dependency.
2. **Headless testability via a synthetic environment.** `tests/synthetic/track_env.py` implements a 2D point-mass/bicycle car on parametric tracks (straight, circle, oval) with the same observation/action contract as the real env. It is the workhorse for Phase 3–5 automated gates.
3. **FakeAc makes the C# bridge testable without the game.** `bridge/FakeAc` creates the same named memory-mapped files AC creates (`Local\acpmf_physics`, `Local\acpmf_graphics`, `Local\acpmf_static`) and writes synthetic struct data, so bridge→UDP→Python is integration-tested end to end on any Windows dev box.
4. **Dependencies:** `torch`, `numpy`, `scipy` (cKDTree), `gymnasium` (env API + spaces), `pyyaml`, `optuna`, `pytest`, `ruff`. Windows-only, import-guarded: `pyvjoy`. Reset keystrokes via `ctypes` SendInput (no extra dependency).

## Repository layout

```
ac-drl-agent/
├── CLAUDE.md
├── PROGRESS.md
├── pyproject.toml
├── configs/
│   ├── default.yaml                # γ, w_* reward weights, UTD, buffer size, ports, paths
│   └── tracks/
├── docs/
│   ├── SPEC.md                     # v2.1 design authority
│   ├── BUILD_PLAN.md               # this file
│   └── REFERENCES.md               # URLs for every external format implemented
├── bridge/
│   ├── AcBridge/                   # .NET 8: MMF reader → UDP sender @60Hz
│   └── FakeAc/                     # .NET 8: synthetic MMF writer (test harness)
├── src/acagent/
│   ├── config.py                   # dataclasses + yaml loader + validation
│   ├── rng.py                      # single seeded RNG manager (Invariant 7)
│   ├── session/                    # Phase 0: paths.py, checkpoint.py, buffer_store.py,
│   │                               #          metadata.py, watchdog.py
│   ├── io/                         # Phase 1: packets.py, telemetry.py, actuator.py,
│   │                               #          failsafe.py, reset.py
│   ├── setupio/                    # Phase 1b: setup_file.py, verify.py
│   ├── track/                      # Phase 2: spline.py, acquire.py, normalize.py
│   ├── rl/                         # Phase 3: networks.py, per_buffer.py, sac.py,
│   │                               #          reward.py, env.py, trainer.py
│   ├── meta/                       # Phase 4: monitor.py, scheduler.py, sweep.py
│   ├── setupopt/                   # Phase 5: objective.py, omega.py, manager.py
│   └── eval/protocol.py            # SPEC §7 standardized evaluation
├── tests/
│   ├── conftest.py                 # markers, fixtures, synthetic data builders
│   ├── synthetic/track_env.py      # 2D car env (Decision 2)
│   ├── unit/
│   └── integration/
└── scripts/
    ├── run_training.py
    ├── hil_phase1.py  hil_phase1b.py  hil_phase2_endurance.py
    ├── bench_spline.py  bench_compile.py
    └── record_spline.py            # manual-lap spline recording fallback
```

---

## Phase 0 — Scaffold, Logging, Checkpointing, Crash Resilience

**Objective:** A project skeleton where any training process can be killed at any moment and resume identically. (SPEC §6, §8 Phase 0.)

**Tasks**
- T0.1 Repo scaffold: `pyproject.toml`, ruff config, pytest config with the four markers, package skeleton, `docs/REFERENCES.md` stub.
- T0.2 `config.py`: dataclass tree mirroring `configs/default.yaml`; validation rejects unknown keys; γ and all `w_*` live here.
- T0.3 `rng.py`: seeded manager handing out child generators (numpy + torch); state export/import.
- T0.4 `session/paths.py`: `session_YYYYMMDD/` layout creation and discovery of the latest session.
- T0.5 `session/metadata.py`: `run_metadata.json` writer — config snapshot, normalization constants placeholder, software versions, git hash.
- T0.6 `session/checkpoint.py`: save/load of the full training state per CLAUDE.md Invariant 10. Atomic write (temp + rename).
- T0.7 `session/buffer_store.py`: replay-buffer snapshot to a memory-mappable format on a cadence; load latest consistent snapshot with priorities.
- T0.8 `session/watchdog.py`: process supervisor that restarts a child command on exit and signals "resumed" state; unit-tested with a dummy child script.

**G0.A (automated):** `pytest -m phase0`
- Checkpoint round-trip: build a dummy actor/critic/optimizer + populated PER buffer, save, reload into fresh objects, assert exact equality of weights, optimizer state tensors, entropy coefficient, priorities, step counter, and RNG draws after restore.
- Kill–resume simulation: run a dummy training loop N steps, snapshot, simulate kill, resume, run M steps; assert the resumed trajectory matches an uninterrupted N+M run seed-for-seed.
- Watchdog restarts a crashing dummy child and the child reports resume.

**G0.H:** none (fully headless phase).

---

## Phase 1 — Mechanical Boundary

**Objective:** Telemetry in, actuation out, with failsafe and automated episode reset. (SPEC §4 Module A.)

**Tasks**
- T1.1 Research task first: locate the official/community AC shared-memory documentation (`SPageFilePhysics`, `SPageFileGraphics`, `SPageFileStatic` field layouts) and the vJoy/pyvjoy axis API. Record URLs in `docs/REFERENCES.md`. Do not write struct code before this.
- T1.2 `bridge/AcBridge`: .NET 8 console app — open the three MMFs, poll at 60 Hz, marshal the documented structs, send one fixed-layout versioned UDP packet per tick to localhost with a monotonic timestamp and sequence number. Config: port, rate.
- T1.3 `bridge/FakeAc`: creates the same MMFs and writes scripted synthetic data (constant speed, scripted slip events, scripted stall) for tests.
- T1.4 `io/packets.py`: Python-side packet schema (struct), versioned; encode/decode; sequence-gap drop detection.
- T1.5 `io/telemetry.py`: `TelemetrySource` interface with `UdpTelemetrySource`, `ReplayTelemetrySource` (recorded file), `SyntheticTelemetrySource` (test patterns).
- T1.6 `io/actuator.py`: `Actuator` interface; `VJoyActuator` (import-guarded, maps steering [−1,1] and throttle/brake [0,1] to vJoy axis range, clamps), `MockActuator` (records calls); optional EMA smoothing per config, default off.
- T1.7 `io/failsafe.py`: state machine — >250 ms without a packet → neutral steering, zero throttle, full brake, session fault raised; recovery path defined.
- T1.8 `io/reset.py`: episode reset via configurable keybind emulation (ctypes SendInput) for AC's "Restart Session" binding; confirmation loop polls shared-memory-derived state (stationary, valid position) with timeout.

**G1.A (automated):** `pytest -m phase1`
- Packet encode/decode round-trip incl. version mismatch rejection; drop detection flags injected sequence gaps.
- Failsafe FSM unit tests on `SyntheticTelemetrySource` stall patterns; actuator mapping/clamping/EMA tests against `MockActuator` recordings.
- Integration (marker `integration`, requires dotnet): FakeAc → AcBridge → `UdpTelemetrySource` end-to-end; assert field fidelity, 60 Hz cadence tolerance, and induced-stall failsafe trigger.

**G1.H (user, `scripts/hil_phase1.py`):** SPEC Phase 1 gate — 2 h straight-line hold at 100 km/h via a scripted constant-throttle controller, memory growth monitored in both processes, packet loss < 0.1%, failsafe verified by killing AcBridge mid-run, automated reset verified by driving over the track limit. STOP after delivering the harness.

---

## Phase 1b — Setup-Apply Spike

**Objective:** Prove the setup-file → pit-apply → shared-memory-verify path before Module E exists. (SPEC §4 Module E mechanism, §8 Phase 1b.)

**Tasks**
- T1b.1 Research task: AC setup-file format (the files the garage screen reads/writes, per-car under the user's Documents AC folder). Obtain 2–3 real sample files from the user as fixtures. Record format references.
- T1b.2 `setupio/setup_file.py`: parse + serialize round-trip preserving unknown keys; typed accessors for pressures, wings, brake bias, ARBs, suspension fields present in fixtures.
- T1b.3 `setupio/verify.py`: post-apply verification — poll telemetry-derived values (e.g., cold pressures) and block until `np.allclose(expected, observed, tol)` or timeout → stint flagged invalid.

**G1b.A:** `pytest -m phase1b` — byte-faithful round-trip on fixtures (modulo edited fields); verifier passes/fails correctly against synthetic telemetry streams.

**G1b.H (user, `scripts/hil_phase1b.py`):** scripted tyre-pressure change written to the setup file, applied at a pit/session boundary without manual menu interaction beyond the agreed automation, confirmed by the verifier. **If this cannot be fully automated, STOP — Module E's mechanism gets redesigned now (semi-automated operator-confirmed fallback per SPEC Risk 2), not in Phase 5.**

---

## Phase 2 — Data Pipeline & Hardware Proofing

**Objective:** Live observation construction and GPU/toolchain validation. (SPEC §4 Module B, §3, §8 Phase 2.)

**Tasks**
- T2.1 Research task: `fast_lane.ai` AI-line format (centerline points + side/border data). Record references. Fallback path: `scripts/record_spline.py` records car world positions + track-edge passes from a manual lap and resamples.
- T2.2 `track/acquire.py`: parse AI line → centerline polyline + left/right track-edge distances; or load a recorded spline. Output a canonical track file in `configs/tracks/`.
- T2.3 `track/spline.py`: cKDTree index over the centerline; per-step ego-centric transform of the upcoming 150 m into a 10-point array normalized to [−1,1]; also returns spline tangent (for `V_prog`) and signed distance to the nearer track edge (`D_bnd`). Runs unconditionally every step (Invariant 4).
- T2.4 `track/normalize.py`: fixed per-channel normalization constants, written into `run_metadata.json`, applied identically in training and eval.
- T2.5 Device manager in `config.py`/`rl`: device selection, eager default, `torch.compile` behind flag with try/except fallback (Invariant 5). `scripts/bench_compile.py` compares eager vs compiled on a SAC-sized network and writes the decision to metadata. `scripts/bench_spline.py` measures transform latency.

**G2.A:** `pytest -m phase2`
- Analytic correctness: on a straight track the ego array is collinear ahead; on a circle of radius R the points match the closed-form arc; tangent and edge-distance match analytic values.
- Performance: median transform < 1 ms on CPU over 10k random poses.
- Normalization round-trip; acquire parser on a fixture AI-line file; recorded-spline path on synthetic lap data.

**G2.H (user, `scripts/hil_phase2_endurance.py`):** SPEC Phase 2 gate — dummy SAC-sized training loop at default UTD for 1 h while AC renders at capped settings; zero TDR/driver-reset events; `bench_compile` result recorded. Prerequisite: the GPU-driver remediation plan from SPEC §3 is complete. STOP after delivering the harness.

---

## Phase 3 — Core DRL Baseline

**Objective:** Contextual SAC + TD-error PER learning end to end, proven on the synthetic env before AC. (SPEC §4 Module C, §5, §8 Phase 3.)

**Tasks**
- T3.1 `tests/synthetic/track_env.py` (Decision 2): 2D kinematic car (point-mass/bicycle), parametric tracks, gymnasium API, same observation contract (kinematics + 10-point spline + ω placeholder) and action contract as the real env; deterministic under seed.
- T3.2 `rl/networks.py`: actor (squashed Gaussian, reparameterized), twin critics, targets; widths from config.
- T3.3 `rl/per_buffer.py`: sum-tree PER — |TD error| priorities (Invariant 3), priority exponent α, IS-weight annealing β, capacity ~1e6, thread-safe add/sample, snapshot hooks into `buffer_store`. No flush method exists (Invariant 1).
- T3.4 `rl/reward.py`: SPEC §5 exactly — `w_v·V_prog − w_b·exp(−k·D_bnd) − w_T·1[violation] − w_s·P_slip − w_a·E_action`; optional potential-based variant for the boundary term behind a config flag; exposes per-term breakdown for logging and the anti-crawl check (mean per-step reward at racing speed must be positive).
- T3.5 `rl/sac.py`: SAC update step — twin-critic targets, entropy auto-tune, fixed γ from config (Invariant 2), gradient clipping; pure function of (batch, state) for testability.
- T3.6 `rl/env.py`: gymnasium wrapper composing `TelemetrySource` + spline transform + reward + `Actuator` + reset protocol; terminal on violation; identical interface to the synthetic env.
- T3.7 `rl/trainer.py`: async actor (60 Hz loop) / learner (UTD-controlled) threads sharing the PER buffer via a lock-minimal queue; burst-training mode flag; full checkpoint integration; TD errors fed back to priorities and to the Phase 4 monitor hook.

**G3.A:** `pytest -m phase3`
- PER statistics: sampling frequencies match priority distribution within tolerance (chi-square), IS weights correct, priorities update after TD feedback; snapshot/restore preserves the tree.
- Reward unit tests per term, terminal handling, dominance check on synthetic racing-speed data; potential-based variant telescopes correctly.
- SAC sanity: seeded run on `track_env` (oval) reaches a configured mean-episode-reward threshold within a CPU budget ≤ 5 min; zero boundary terminals over the final 5 evaluation laps; rolling-median lap time improves monotonically over checkpoints.
- Trainer: actor/learner run concurrently on the synthetic env without deadlock; kill–resume mid-training reproduces the uninterrupted run (re-uses G0 machinery).

**G3.H (user):** SPEC Phase 3 gate on a simple real track — N consecutive clean laps, improving rolling-median lap time, positive mean per-step reward at speed. Claude Code provides the run config + monitoring instructions, then STOPS.

---

## Phase 4 — Meta-Adaptation Controller

**Objective:** Tier 1 online scheduler + Tier 2 offline Optuna harness. (SPEC §4 Module D, §7, §8 Phase 4.)

**Tasks**
- T4.1 `meta/monitor.py`: tracks mean |TD error| over the top decile of PER priorities; plateau detection (configurable window/slope).
- T4.2 `meta/scheduler.py`: Tier 1 — on plateau, reduce LR by config factor within bounds; on sustained regression, partial restore; every change logged to metadata. Touches nothing else (γ untouchable, Invariant 2; buffer untouched, Invariant 1).
- T4.3 `eval/protocol.py`: SPEC §7 — deterministic policy, fixed conditions config, one out-lap + N flying laps, trimmed mean + best lap.
- T4.4 `meta/sweep.py`: Tier 2 — Optuna TPE over {initial LR, batch size, τ, network width, UTD}; γ excluded by construction (the sweep-space builder must not accept it); `MedianPruner`; warm-start from intact buffer; architecture-change path = reinit networks + offline warm-start from buffer; scored via `eval/protocol`.

**G4.A:** `pytest -m phase4`
- Plateau detector: correct fire/no-fire on synthetic TD-error curves (plateau, noisy decline, regression).
- Sabotage recovery on synthetic env: mid-training LR ×100 → scheduler restores a healthy critic-loss trend within the configured window; buffer object identity unchanged throughout (Invariant 1 assertion).
- Sweep harness: ≥3 trials on the synthetic env with pruning observed; attempting to add γ to the sweep space raises.

**G4.H (user):** SPEC Phase 4 gates on the real rig — Tier 1 sabotage recovery during a real session; one offline sweep producing a config that beats the incumbent on standardized eval stints. STOP after delivering run instructions.

---

## Phase 5 — Vehicle Setup Optimizer

**Objective:** Module E outer loop. (SPEC §4 Module E, §8 Phase 5.)

**Tasks**
- T5.1 `setupopt/omega.py`: derive ω (aero balance ratio, front/rear combined suspension frequency, brake bias, compound index) from a parsed setup file + car constants; normalized; unit-tested against hand-computed fixtures.
- T5.2 `setupopt/objective.py`: SPEC §4 J_setup exactly — trimmed-mean valid flying laps (out/in laps excluded) + λ-weighted deviations (|ΔT_io − ΔT_io*|, |ΔT_lat|, core-temp window deviation); per-term breakdown logged.
- T5.3 `setupopt/manager.py`: master session state machine — propose setup (Optuna ask), write file, apply at stint boundary, verify (Phase 1b machinery), run ≥20-lap stint, score, tell Optuna, recompute ω and propagate to the env context; invalid-stint and timeout paths; condition-control config enforced.

**G5.A:** `pytest -m phase5`
- Objective signs: constructed imbalanced-temperature stints strictly increase J vs balanced ones; lap-time dominance holds at configured λ; out/in laps excluded from the trimmed mean.
- ω derivation matches fixtures; changing a setup field changes exactly the expected ω components.
- Manager state machine on fully mocked stints: happy path, failed verification → stint invalid, Optuna receives correct scores, ω propagates into subsequent observations.

**G5.H (user):** SPEC Phase 5 gate — start from a deliberately imbalanced setup; unattended convergence over multiple stints to temps in window, |ΔT_io − ΔT_io*| within tolerance, improved trimmed-mean lap time. STOP after delivering the session config.

---

## Appendix — External references to resolve before coding (record in docs/REFERENCES.md)

1. AC shared memory page layouts (`SPageFilePhysics` / `SPageFileGraphics` / `SPageFileStatic`) — official AC SDK / community documentation.
2. AC setup-file format and on-disk location per car.
3. `fast_lane.ai` AI-line binary format (centerline + side data).
4. vJoy driver + `pyvjoy` axis API and value ranges; ViGEmBus/`vgamepad` as fallback.
5. PyTorch-on-Windows ROCm wheel install index for gfx1201 (AMD docs) — target machine only.
6. Win32 `SendInput` scan-code reference for the reset keybind.
