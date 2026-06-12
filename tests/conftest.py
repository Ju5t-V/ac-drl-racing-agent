"""Shared pytest fixtures and synthetic data builders.

Markers are declared in pyproject.toml ([tool.pytest.ini_options]).
The default suite — ``pytest -m "not gpu and not hil"`` — must pass on a
CPU-only machine with no Assetto Corsa, vJoy, or GPU present (CLAUDE.md,
Invariant 8). Hardware-touching code is only ever exercised here through
its mock/synthetic twin.
"""
