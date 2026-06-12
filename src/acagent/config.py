"""Typed configuration tree loaded from ``configs/*.yaml``.

Mirrors ``configs/default.yaml`` exactly. The loader rejects unknown keys
anywhere in the tree so a typo can never silently fall back to a default.
Missing keys are allowed and take the dataclass default.

Invariant 2 (CLAUDE.md): the discount factor ``sac.gamma`` is read from here
once per run, is immutable mid-run (all config dataclasses are frozen), and
must never appear in any sweep space.

Symbol hygiene (Invariant 9): reward weights are named ``w_*``; γ is the
discount factor only; ω is the setup vector only.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml


class ConfigError(ValueError):
    """Raised for unknown keys, type mismatches, or out-of-range values."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


@dataclass(frozen=True)
class SessionConfig:
    """Session directory layout and persistence cadences (SPEC §6)."""

    root_dir: str = "sessions"
    checkpoint_interval_s: float = 900.0
    # SPEC §6: replay buffer snapshotted on a fixed cadence, default 15 min.
    buffer_snapshot_interval_s: float = 900.0

    def __post_init__(self) -> None:
        _require(bool(self.root_dir), "session.root_dir must be non-empty")
        _require(self.checkpoint_interval_s > 0, "session.checkpoint_interval_s must be > 0")
        _require(
            self.buffer_snapshot_interval_s > 0, "session.buffer_snapshot_interval_s must be > 0"
        )


@dataclass(frozen=True)
class IoConfig:
    """Telemetry link and failsafe parameters (SPEC §4 Module A)."""

    telemetry_port: int = 47800
    telemetry_rate_hz: float = 60.0
    # SPEC Module A: no packet for 250 ms -> neutral steer, zero throttle,
    # full brake, session fault.
    failsafe_timeout_ms: float = 250.0
    # Optional EMA smoothing on the steering channel; SPEC Module A, default off.
    steering_ema_enabled: bool = False
    steering_ema_coef: float = 0.5

    def __post_init__(self) -> None:
        _require(
            1 <= self.telemetry_port <= 65535,
            "io.telemetry_port must be a valid UDP port (1-65535)",
        )
        _require(self.telemetry_rate_hz > 0, "io.telemetry_rate_hz must be > 0")
        _require(self.failsafe_timeout_ms > 0, "io.failsafe_timeout_ms must be > 0")
        _require(0.0 < self.steering_ema_coef <= 1.0, "io.steering_ema_coef must be in (0, 1]")


@dataclass(frozen=True)
class SacConfig:
    """Core SAC hyperparameters (SPEC §4 Module C)."""

    # Invariant 2: fixed for the lifetime of a run; excluded from every sweep.
    gamma: float = 0.99
    lr: float = 3.0e-4
    batch_size: int = 256
    tau: float = 0.005
    hidden_sizes: tuple[int, ...] = (256, 256)
    entropy_coef_init: float = 1.0
    # None -> automatic target entropy (-dim(action space)), standard SAC.
    target_entropy: float | None = None
    grad_clip: float | None = 10.0

    def __post_init__(self) -> None:
        _require(0.0 < self.gamma < 1.0, "sac.gamma must be in (0, 1)")
        _require(self.lr > 0, "sac.lr must be > 0")
        _require(self.batch_size > 0, "sac.batch_size must be > 0")
        _require(0.0 < self.tau <= 1.0, "sac.tau must be in (0, 1]")
        _require(
            len(self.hidden_sizes) > 0 and all(h > 0 for h in self.hidden_sizes),
            "sac.hidden_sizes must be a non-empty list of positive ints",
        )
        _require(self.entropy_coef_init > 0, "sac.entropy_coef_init must be > 0")
        _require(self.grad_clip is None or self.grad_clip > 0, "sac.grad_clip must be > 0 or null")


@dataclass(frozen=True)
class PerConfig:
    """Prioritized Experience Replay (SPEC §4 Module C; Schaul et al., 2015).

    Invariant 3: priorities come from |TD error| only — there is deliberately
    no field here for any alternative priority source.
    """

    capacity: int = 1_000_000
    alpha: float = 0.6
    beta_start: float = 0.4
    beta_end: float = 1.0
    beta_anneal_steps: int = 1_000_000
    priority_epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        _require(self.capacity > 0, "per.capacity must be > 0")
        _require(0.0 <= self.alpha <= 1.0, "per.alpha must be in [0, 1]")
        _require(0.0 <= self.beta_start <= 1.0, "per.beta_start must be in [0, 1]")
        _require(
            self.beta_start <= self.beta_end <= 1.0,
            "per.beta_end must be in [beta_start, 1]",
        )
        _require(self.beta_anneal_steps > 0, "per.beta_anneal_steps must be > 0")
        _require(self.priority_epsilon > 0, "per.priority_epsilon must be > 0")


@dataclass(frozen=True)
class RewardConfig:
    """Reward weights and shaping constants (SPEC §5).

    R_t = w_v*V_prog - w_b*exp(-k*D_bnd) - w_T*1[violation] - w_s*P_slip - w_a*E_action
    """

    w_v: float = 1.0
    w_b: float = 1.0
    w_T: float = 10.0  # noqa: N815 — symbol fixed by SPEC §5
    w_s: float = 0.1
    w_a: float = 0.1
    k: float = 0.5
    # SPEC §5 reward-hacking guard: optional potential-based variant of the
    # boundary term, F = gamma*Phi(s') - Phi(s) with Phi(s) = -c*exp(-k*D_bnd).
    potential_based_boundary: bool = False

    def __post_init__(self) -> None:
        for name in ("w_v", "w_b", "w_T", "w_s", "w_a"):
            _require(getattr(self, name) >= 0, f"reward.{name} must be >= 0")
        _require(self.k > 0, "reward.k must be > 0")


@dataclass(frozen=True)
class TrainerConfig:
    """Async actor/learner loop parameters (SPEC §3, §4 Module C)."""

    control_rate_hz: float = 60.0
    # SPEC §3: default 1 gradient step per environment step; raise only while
    # the Phase 2 endurance gate continues to pass.
    utd_ratio: float = 1.0
    # SPEC §3 mitigation 4: defer gradient updates and run them in bursts
    # while the car is in the pits.
    burst_mode: bool = False

    def __post_init__(self) -> None:
        _require(self.control_rate_hz > 0, "trainer.control_rate_hz must be > 0")
        _require(self.utd_ratio > 0, "trainer.utd_ratio must be > 0")


@dataclass(frozen=True)
class Config:
    """Root of the configuration tree. Loaded once per run via load_config()."""

    seed: int = 42
    session: SessionConfig = field(default_factory=SessionConfig)
    io: IoConfig = field(default_factory=IoConfig)
    sac: SacConfig = field(default_factory=SacConfig)
    per: PerConfig = field(default_factory=PerConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    def __post_init__(self) -> None:
        _require(self.seed >= 0, "seed must be >= 0")


def _is_optional_float(hint: Any) -> bool:
    if get_origin(hint) in (Union, UnionType):
        args = set(get_args(hint))
        return args == {float, type(None)}
    return False


def _coerce(value: Any, hint: Any, path: str) -> Any:
    """Coerce a YAML scalar/list to the annotated type, strictly."""
    if _is_optional_float(hint):
        if value is None:
            return None
        hint = float
    if hint is float:
        # YAML often parses whole numbers as int; accept them for float fields.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected a number, got {value!r}")
        return float(value)
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected an integer, got {value!r}")
        return value
    if hint is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected a boolean, got {value!r}")
        return value
    if hint is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected a string, got {value!r}")
        return value
    if get_origin(hint) is tuple:
        if not isinstance(value, list):
            raise ConfigError(f"{path}: expected a list, got {value!r}")
        (item_type, _ellipsis) = get_args(hint)
        return tuple(_coerce(item, item_type, f"{path}[{i}]") for i, item in enumerate(value))
    raise ConfigError(f"{path}: unsupported config field type {hint!r}")  # pragma: no cover


def _build(cls: type, data: Any, path: str) -> Any:
    """Recursively construct a dataclass from a nested dict, rejecting unknown keys."""
    if not isinstance(data, dict):
        raise ConfigError(f"{path or 'config root'}: expected a mapping, got {data!r}")
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        where = path or "config root"
        raise ConfigError(f"{where}: unknown key(s) {unknown}")
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue  # dataclass default applies
        hint = hints[f.name]
        child_path = f"{path}.{f.name}" if path else f.name
        if dataclasses.is_dataclass(hint):
            kwargs[f.name] = _build(hint, data[f.name], child_path)
        else:
            kwargs[f.name] = _coerce(data[f.name], hint, child_path)
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config file into the frozen Config tree."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    return _build(Config, raw, "")


def config_to_dict(config: Config) -> dict[str, Any]:
    """Plain-dict snapshot of the config (for run_metadata.json, T0.5)."""
    return dataclasses.asdict(config)
