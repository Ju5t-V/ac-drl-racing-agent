# Technical Specification v2.1: Autonomous Contextual DRL Agent for Assetto Corsa

**Status:** Production-ready specification. Supersedes v2.0.
**Deployment model:** Single Windows machine, native execution. No containers.

---

## Changelog from v2.0

| # | Change | Reason |
|---|--------|--------|
| 1 | Docker removed; native Windows 11 single-box deployment | AC, vJoy, and the trainer must share one machine; the Linux/Docker backend contradicted the Windows-only actuator and game |
| 2 | GPU corrected to RDNA 4 (gfx1201) | RX 9070 XT is Navi 48 / RDNA 4, not RDNA 3 |
| 3 | Apache TVM removed | SAC policy inference is sub-millisecond uncompiled; latency lives in IPC, not matmul |
| 4 | `torch.compile` demoted to optional, eager mode is the default | Inductor/Triton on Windows ROCm is not production-stable for gfx1201 |
| 5 | PER priority source changed from autoencoder reconstruction error to TD error | Reconstruction error measures input rarity, not learning value; canonical PER (Schaul et al., 2015) uses TD error |
| 6 | Spline transform computed every control step; lazy evaluation removed from the inference path | The spline array is part of the live observation; the policy cannot act without it |
| 7 | Module D rewritten as a three-tier Meta-Adaptation Controller; buffer flush removed entirely | SAC is off-policy; the buffer is valid across hyperparameter changes and is never discarded |
| 8 | Discount factor γ removed from the sweep space; fixed per run | γ defines the objective; changing it mid-run invalidates every learned Q-value |
| 9 | Module E objective rewritten as a minimization with deviation penalties | v2.0 signs rewarded larger temperature imbalance |
| 10 | Setup changes applied through AC's setup-file mechanism at stint boundaries, not physics `.ini` hot-reload | In-session physics `.ini` reload via console is not a documented AC capability; setup parameters are the supported path |
| 11 | Reward symbols renamed (`w_v, w_b, w_T, w_s, w_a`); γ and ω reserved for discount factor and setup vector | v2.0 reused γ and ω with two meanings each |
| 12 | Reward rationale corrected from "differentiability" to dense shaping | Model-free RL does not backpropagate through the reward; the exponential term is shaping, which is the correct mechanism |
| 13 | Asynchronous actor/learner split with configurable UTD ratio added | AC runs at 1× wall clock; gradient throughput must decouple from the 60 Hz collection rate |
| 14 | GPU contention management section added | One GPU serves both AC rendering and training; uncontrolled contention risks TDR resets |
| 15 | Crash resilience promoted to a Phase 0 gate | Known driver instability on the target machine makes mid-stint recovery a hard requirement |

---

## 1. Project Overview

A self-improving autonomous driving system for Assetto Corsa. A Contextual Soft Actor-Critic (SAC) agent learns vehicle control from live telemetry. A Meta-Adaptation Controller manages training hyperparameters across three tiers of intervention. An asynchronous Bayesian optimizer searches the vehicle setup space between stints. All components run natively on one Windows machine.

Three optimization loops operate at different timescales and are integrated in strict order (Phases 3 → 4 → 5) so that no loop destabilizes the one beneath it:

1. **Inner loop (every step):** SAC gradient updates.
2. **Middle loop (plateau-triggered / between sessions):** hyperparameter adaptation.
3. **Outer loop (every stint):** vehicle setup search.

---

## 2. Deployment Model & Core Stack

| Component | Selection | Notes |
|---|---|---|
| OS | Windows 11 | Required by AC and vJoy; PyTorch-on-Windows ROCm wheels target Windows 11 |
| Python | 3.12, single venv | Pinned to the official PyTorch-on-Windows ROCm wheel requirement |
| Deep learning | PyTorch 2.8+ with ROCm 6.4.4 components (gfx1201) | Official production support for RX 9070 XT on Windows |
| RL implementation | Stable-Baselines3 SAC with a custom `ReplayBuffer` subclass (PER + contextual states), or a self-contained SAC implementation if SB3 internals resist the customization | The PER sum-tree, contextual observation handling, and async learner are custom components either way; budget for this |
| Game interface | C# telemetry bridge reading AC shared memory, forwarding over localhost UDP | 60 Hz poll rate; packets timestamped |
| Actuation | `vJoy` + `pyvjoy`, native | Fallback: `ViGEmBus` + `vgamepad` (Xbox pad emulation) if the vJoy driver misbehaves on current Windows 11 builds |
| Optimization | Optuna (TPE sampler, `MedianPruner`) | Used offline only: between-session hyperparameter sweeps and Module E setup search |
| Inference compilation | None by default (eager mode) | `torch.compile(backend="inductor")` may be benchmarked per Phase 2; any failure or instability reverts to eager. TVM is removed from the project |

**Latency budget.** The control loop runs at 60 Hz (16.7 ms per step). Eager-mode inference for a SAC actor MLP is well under 1 ms on this GPU. The optimization target is the IPC path (shared memory → UDP → Python → vJoy write), not network execution.

---

## 3. Hardware Target & GPU Contention Management

* **GPU:** AMD Radeon RX 9070 XT — RDNA 4, LLVM target `gfx1201`. Officially listed as a supported compute GPU in current ROCm documentation.
* **Constraint:** the same GPU renders Assetto Corsa and executes training. Both compete for compute and VRAM, and the machine has a known history of driver timeout events (LiveKernelEvent 117/193). Sustained contention is the most likely cause of a mid-stint TDR reset.

Mitigations, in order of importance:

1. **Driver stability is a prerequisite, not a parallel task.** The existing remediation plan (DDU clean reinstall, HAGS off, overlays off, Fast Startup off, BIOS/chipset updates, PSU cabling verification) completes before Phase 2's endurance gate is attempted.
2. **Minimize AC's render load.** Lowest graphics preset, small fixed-size window, frame cap at 60 fps (matching the control rate; rendering faster buys nothing).
3. **Bounded learner throughput.** The learner thread exposes a configurable update-to-data (UTD) ratio. Default 1 gradient step per environment step; raise only while the Phase 2 endurance gate continues to pass.
4. **Burst-training mode.** As a fallback for contention problems, gradient updates can be deferred and executed in bursts while the car is in the pits between stints. Off-policy correctness is unaffected.
5. **Crash tolerance regardless.** Even with all of the above, Phase 0's checkpoint/resume machinery treats a driver reset as an expected event, not a catastrophe.

---

## 4. Module Specifications

### Module A: Mechanical Boundary (Sensors & Actuators)

* **Telemetry bridge (C#):** Polls AC shared memory at 60 Hz. Extracts kinematics (velocity vector, G-forces, slip angles per tyre, RPM, gear), tyre temperatures (inner/middle/outer per corner), and a monotonic timestamp. Forwards as a fixed-layout UDP packet to localhost.
* **Actuator (Python):** Maps policy outputs to vJoy axes. Steering ∈ [−1.0, 1.0]; throttle, brake ∈ [0.0, 1.0]. An optional exponential moving average on the steering channel (configurable, default off) is available if the action-jitter penalty alone proves insufficient.
* **Failsafe:** If no telemetry packet arrives for 250 ms, the actuator writes neutral steering, zero throttle, full brake, and raises a session fault.
* **Episode reset:** On a terminal state (track-limit violation), the session manager issues an automated restart (AC session restart or teleport-to-pits via the console/app interface) and confirms the car is stationary in a valid grid/pit position via shared memory before releasing control to the policy. Unattended operation depends on this path being fully automatic.

### Module B: Data Pipeline

* **Ego-centric spline transformer:** A KD-tree spatial index over the track spline locates the nearest spline node each step. The upcoming 150 m of track centerline is transformed into a 10-point ego-centric (car-relative) array, normalized to [−1, 1]. **This transform executes every control step, unconditionally** — it is part of the live observation. At ~10 points per frame the cost is microseconds; there is nothing to gain by skipping frames and the policy is blind without it.
* **Normalization:** All kinematic channels are normalized by fixed, per-channel constants recorded in `run_metadata.json` (running normalization is avoided so that checkpoints remain comparable across sessions).
* **Removed from v2.0:** The pre-trained autoencoder is no longer in the replay-priority path (see Module C). If a novelty signal is wanted later, the correct mechanism is an intrinsic-reward curiosity module (RND-style) added to the reward, as a separately gated, optional work item. It must not be conflated with replay prioritization.

### Module C: Core DRL Agent (Contextual SAC)

* **Algorithm:** Soft Actor-Critic with automatic entropy-coefficient tuning (standard SAC α auto-tune) and twin critics.
* **Replay:** Prioritized Experience Replay keyed to **absolute TD error** (Schaul et al., 2015), with standard priority exponent and importance-sampling annealing. High-TD transitions (typically corner entries, slides, saves) are naturally over-sampled while they are still poorly valued, and decay in priority as the critic masters them. Buffer capacity ~1×10⁶ transitions, held in RAM, snapshotted to disk on the Phase 0 schedule.
* **Contextual state space:**
  1. Normalized kinematic telemetry.
  2. Normalized 10-point ego-centric spline array.
  3. **Setup vector ω:** derived physical handling metrics (aero balance ratio, front/rear combined suspension frequency, brake bias, tyre compound index), normalized. Conditioning the actor and critics on ω keeps Q-values valid as Module E changes the car. Note the sample-complexity cost: the agent is learning a policy *over* the setup space, not for one car. ω is frozen for all of Phase 3.
* **Discount factor:** γ is fixed for the lifetime of a training run. It is excluded from every sweep space (see Module D).
* **Asynchronous actor/learner split:** The actor thread runs the 60 Hz sense→infer→act loop. The learner thread runs gradient updates continuously on the GPU at the configured UTD ratio. They share the replay buffer through a lock-minimal queue. AC runs at 1× wall clock (max ~216,000 environment steps per hour), so sample efficiency is the binding constraint; the async split lets gradient throughput exceed the collection rate when GPU headroom allows.

### Module D: Meta-Adaptation Controller

Replaces v2.0's mid-training Optuna design. Rationale: SAC is off-policy, so the replay buffer remains valid across hyperparameter changes and is **never flushed** — a transition (s, a, r, s′) does not depend on the policy that produced it, and collected experience is the most expensive asset in a real-time simulator. v2.0's "stale on-policy data" concern does not apply to SAC, and TPE's independent-trial model does not fit mid-run adaptation.

* **Trigger (retained from v2.0):** Monitor mean absolute TD error over the top decile of the PER buffer. A plateau there means the hardest physics states have stopped improving.

* **Tier 1 — Online, always active (single-run safe):**
  * SAC automatic entropy tuning (built into the algorithm).
  * Learning-rate plateau scheduler keyed to the top-decile TD-error signal: on plateau, reduce LR by a configured factor; on sustained regression after a reduction, partially restore. Bounded LR range logged per change.
  * These act on a single agent, online, with no trial structure required.

* **Tier 2 — Offline, between sessions (Optuna's correct role):**
  * Optuna TPE sweeps {initial learning rate, batch size, target-smoothing τ, network width, UTD ratio} across separate runs or warm-started restarts. **γ is excluded.**
  * Each trial is scored on standardized evaluation stints (Section 7). `MedianPruner` terminates losing trials early.
  * Trials inherit the intact replay buffer where applicable (warm start); on **architecture changes**, networks are reinitialized and warm-started by offline gradient updates from the preserved buffer. Data survives; only the learner is replaced.

* **Tier 3 — Optional, explicitly deferred: sequential population-based training.**
  * Canonical PBT requires a parallel population, and a single real-time AC instance cannot host one. If population methods are wanted, the only honest variant here is sequential PBT at stint granularity (population 3–4, time-sliced through the session manager, exploit/explore at stint boundaries). Wall-clock cost is roughly proportional to population size. Not part of the v2.1 critical path; revisit only if Tiers 1–2 prove insufficient.

* **Buffer protocol:** Never flush, under any tier. PER importance-sampling weights handle off-policy correction as designed.

### Module E: Vehicle Setup Optimizer

* **Role:** Master session manager. Owns stint boundaries, setup changes, evaluation, and the Optuna study over the setup space.
* **Stint floor:** Minimum 20 laps per setup evaluation. Out-lap and in-lap are excluded from scoring.
* **Setup application (corrected mechanism):** Setup parameters (tyre pressures, wing angles, brake bias, suspension geometry, ARBs) are written through AC's setup-file mechanism — the same files the garage screen reads — and applied at a stint boundary via an automated pit stop / session restart. **In-session hot-reload of physics `.ini` data is not assumed to exist.** This mechanism is proven by the Phase 1b spike before any other Module E work begins.
* **Verification handshake (retained):** After an apply, Python polls shared memory and blocks until `np.allclose()` confirms the expected setup-derived values (e.g., target cold pressures) within tolerance, or times out and flags the stint invalid.
* **Objective (corrected):** Minimize

$$
J_{setup} \;=\; \bar{t}_{lap} \;+\; \lambda_1\,\bigl|\Delta T_{io} - \Delta T_{io}^{*}\bigr| \;+\; \lambda_2\,\bigl|\Delta T_{lat}\bigr| \;+\; \lambda_3\,\mathrm{dev}\!\left(T_{core},\,W_{compound}\right)
$$

  * $\bar{t}_{lap}$: trimmed mean of valid flying-lap times over the stint (robust to single-lap noise).
  * $\Delta T_{io}$: mean inner/outer tyre temperature delta per axle; $\Delta T_{io}^{*}$ is the small inner-biased target appropriate to the camber, not zero.
  * $\Delta T_{lat}$: left/right temperature asymmetry beyond what the track layout explains.
  * $\mathrm{dev}(T_{core}, W_{compound})$: distance of core temperature from the compound's optimal operating window (zero inside the window).
  * Lap time dominates; temperature terms are health regularizers. All λ weights are logged in `run_metadata.json`. Under minimization, every term now points the right way — v2.0's signs rewarded *larger* imbalance.
* **Context propagation:** After each accepted setup change, the ω vector is recomputed and the new value accompanies all subsequent transitions.
* **Condition control:** Evaluation stints run under fixed, comparable conditions — fixed time of day, static/optimal track grip, identical fuel load and tyre compound — so that $J_{setup}$ differences are attributable to the setup.

---

## 5. Reward Function

$$
R_t \;=\; w_v\,V_{prog} \;-\; w_b\,e^{-k\,D_{bnd}} \;-\; w_T\,\mathbb{1}_{violation} \;-\; w_s\,P_{slip} \;-\; w_a\,E_{action}
$$

| Term | Meaning |
|---|---|
| $V_{prog}$ | Velocity projected onto the local track-spline tangent (progress rate) |
| $D_{bnd}$ | Distance to the absolute track limit; $k$ sets the decay of the warning band |
| $\mathbb{1}_{violation}$ | Terminal indicator (0/1) for a track-limit violation |
| $P_{slip}$ | Tyre-slip excess penalty (beyond the compound's useful slip range) |
| $E_{action}$ | High-frequency input-jitter penalty (squared action delta) |

Symbol hygiene: `w_*` are reward weights only. γ is reserved for the RL discount factor, ω for the setup vector. Neither appears in this equation.

**Design rationale (corrected from v2.0).** The exponential boundary term is **dense reward shaping**: it converts a sparse cliff at the wall into a smooth approach gradient the critic can learn from. It is *not* about differentiability — in model-free SAC, gradients flow through the critic loss and the reparameterized actor, never through the reward function, so the reward needs no derivative. The shaping benefit is what justifies keeping the term.

**Reward-hacking guards:**
* $w_v V_{prog}$ must dominate the summed penalties in normal driving, or the optimal policy degenerates to crawling to avoid slip/jitter/boundary costs. Verify the sign of the average per-step reward at racing speed during Phase 3.
* Optionally express the boundary term as potential-based shaping, $F = \gamma\,\Phi(s') - \Phi(s)$ with $\Phi(s) = -c\,e^{-k\,D_{bnd}}$, which provably leaves the optimal policy unchanged (Ng et al., 1999). Recommended if boundary-shy behavior (refusing to use full track width) appears.

---

## 6. Logging, Checkpointing & Crash Resilience

* **Directory layout:** `session_YYYYMMDD/` containing setup files per stint, SAC checkpoints, buffer snapshots, Optuna study DBs, and `run_metadata.json` (all weights, normalization constants, λ values, γ, software versions, git hash).
* **Checkpoint contents:** actor/critic/target weights, optimizer states, entropy coefficient, PER priority structure, environment step count, RNG states. A checkpoint that cannot fully reconstruct training is not a checkpoint.
* **Buffer persistence:** replay buffer snapshotted to disk on a fixed cadence (default 15 min) in a memory-mappable format; on resume, the latest consistent snapshot loads with its priorities.
* **Watchdogs:** (a) trainer watchdog restarts the Python process on crash and auto-resumes from the last checkpoint; (b) AC watchdog detects process hang/exit, relaunches, restores the session, and re-runs the Module A reset handshake. A GPU driver reset is treated as a routine watchdog event.

---

## 7. Standardized Evaluation Protocol

Used by Module D Tier 2 and Module E scoring.

* Fixed track, fixed time of day, static optimal grip, fixed fuel, fixed compound.
* Deterministic policy (mean action, no exploration noise).
* N flying laps (default 5) after one out-lap; report trimmed mean and best lap.
* Identical normalization constants as training (from `run_metadata.json`).

---

## 8. Implementation Roadmap & Validation Gates

* **Phase 0 — Logging & Crash Resilience**
  * Build the session directory structure, full checkpointing, buffer snapshots, both watchdogs, auto-resume.
  * **Gate:** kill the trainer process mid-run (and separately, kill AC) → system resumes autonomously with weights, optimizer state, buffer, and PER priorities intact, and the lap counter continues. No human input.

* **Phase 1 — Mechanical Boundary**
  * C# shared-memory bridge, UDP link, pyvjoy actuation, failsafe, automated episode reset.
  * **Gate:** hold 100 km/h in a straight line for 2 hours; no memory growth in either process; telemetry packet loss < 0.1%; failsafe verified by an induced telemetry stall; automated reset verified by an induced violation.

* **Phase 1b — Setup-Apply Spike (de-risk before Module E exists)**
  * Prove the setup-file → pit-apply → shared-memory-verify path end to end.
  * **Gate:** a scripted tyre-pressure change is written, applied at a stint boundary without human input, and confirmed by the `np.allclose()` handshake. If this gate cannot pass, Module E's mechanism is redesigned *now*, not in Phase 5.

* **Phase 2 — Data & Hardware Proofing**
  * KD-tree spline transform, normalization pipeline, PyTorch-on-gfx1201 validation, `torch.compile` benchmark with eager fallback.
  * **Gate:** spline transform < 1 ms per step; a dummy SAC-sized network trains for 1 hour at the default UTD *while AC renders at the capped settings* with zero TDR/driver-reset events; compile-vs-eager decision recorded in `run_metadata.json`.

* **Phase 3 — Core DRL Baseline**
  * SAC + TD-error PER + contextual state (ω frozen) + reward function + async actor/learner.
  * **Gate:** agent completes N consecutive laps of a simple track with zero boundary terminals and a monotonically improving rolling-median lap time; average per-step reward at racing speed is positive (anti-crawl check).

* **Phase 4 — Meta-Adaptation Integration**
  * Tier 1 plateau scheduler wired to the top-decile TD-error trigger; Tier 2 Optuna offline sweep harness on the Section 7 protocol.
  * **Gate (Tier 1):** sabotage the learning rate by ×100 mid-run → the scheduler restores a healthy critic-loss trend within a configured window, with the buffer untouched. **Gate (Tier 2):** an offline sweep produces a configuration that beats the incumbent on standardized evaluation stints.

* **Phase 5 — Setup Optimizer Integration**
  * Module E session management, Optuna setup study, ω propagation, condition control.
  * **Gate:** starting from a deliberately imbalanced setup, the system converges unattended over multiple stints to a mechanically sound baseline: tyre temperatures inside the compound window, $|\Delta T_{io} - \Delta T_{io}^{*}|$ within tolerance, and $\bar{t}_{lap}$ improved versus the starting setup.

---

## 9. Open Risks

1. **GPU driver stability.** The known LiveKernelEvent 117/193 history on this machine is the single largest threat to unattended multi-hour training. The remediation plan gates Phase 2; the watchdogs assume it will still occasionally fail.
2. **AC setup-apply automation.** Phase 1b exists because the entire Module E design rests on this mechanism. If AC's automation surface is narrower than expected, the fallback is operator-confirmed setup changes at stint boundaries (semi-automated outer loop).
3. **SB3 customization depth.** TD-error PER, contextual inputs, and the async learner push against SB3's abstractions. If the subclassing fights back, a self-contained SAC (~600 lines) is cheaper than bending the framework.
4. **Real-time sample budget.** ~216k steps/hour hard ceiling. SAC's sample efficiency plus a UTD > 1 is the lever; there is no vectorized-environment escape hatch with a single AC instance.
5. **Windows ROCm toolchain drift.** Eager mode is the contract; `torch.compile` is a bonus that may break across driver/wheel updates. Re-run the Phase 2 compile benchmark after any stack upgrade.
