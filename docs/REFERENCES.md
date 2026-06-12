# REFERENCES.md — External format & API documentation sources

Rule (CLAUDE.md, Repository conventions): external formats are NEVER invented
from memory. Every external format or API implemented in this repository must
cite its source URL both here and in a code comment at the point of use. If
documentation cannot be found or is ambiguous, ask the user — sample files can
be extracted from the user's AC installation as fixtures.

## 1. AC shared memory page layouts (SPageFilePhysics / SPageFileGraphics / SPageFileStatic)

_To be resolved in Phase 1 (T1.1) before any struct code is written._

## 2. AC setup-file format and on-disk location per car

_To be resolved in Phase 1b (T1b.1)._

## 3. `fast_lane.ai` AI-line format (centerline + side data)

_To be resolved in Phase 2 (T2.1)._

## 4. vJoy driver + pyvjoy axis API and value ranges (fallback: ViGEmBus/vgamepad)

_To be resolved in Phase 1 (T1.1)._

## 5. PyTorch-on-Windows ROCm wheel index for gfx1201

_Target machine only; to be resolved before Phase 2 GPU work._

## 6. Win32 SendInput scan-code reference (episode-reset keybind)

_To be resolved in Phase 1 (T1.8)._
