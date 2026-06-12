"""Phase 0 session machinery: paths, metadata, checkpointing, buffer snapshots, watchdog.

Everything here exists so that any training process can be killed at any moment
and resume identically (SPEC §6, §8 Phase 0; CLAUDE.md Invariant 10).
"""
