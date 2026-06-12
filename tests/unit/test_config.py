"""T0.2 tests: config loading, unknown-key rejection, type and range validation."""

import dataclasses
from pathlib import Path

import pytest
import yaml

from acagent.config import Config, ConfigError, config_to_dict, load_config

pytestmark = [pytest.mark.unit, pytest.mark.phase0]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_YAML = REPO_ROOT / "configs" / "default.yaml"


def write_yaml(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def default_dict() -> dict:
    return yaml.safe_load(DEFAULT_YAML.read_text(encoding="utf-8"))


class TestLoadDefault:
    def test_default_yaml_loads(self) -> None:
        config = load_config(DEFAULT_YAML)
        assert isinstance(config, Config)

    def test_values_match_yaml(self) -> None:
        config = load_config(DEFAULT_YAML)
        raw = default_dict()
        assert config.seed == raw["seed"]
        assert config.sac.gamma == raw["sac"]["gamma"]
        assert config.reward.w_b == raw["reward"]["w_b"]
        assert config.io.telemetry_port == raw["io"]["telemetry_port"]
        assert config.per.capacity == raw["per"]["capacity"]
        assert config.trainer.utd_ratio == raw["trainer"]["utd_ratio"]

    def test_gamma_and_all_reward_weights_present(self) -> None:
        """CLAUDE.md: gamma and all w_* live in the config (T0.2)."""
        config = load_config(DEFAULT_YAML)
        assert 0.0 < config.sac.gamma < 1.0
        for name in ("w_v", "w_b", "w_T", "w_s", "w_a"):
            assert getattr(config.reward, name) >= 0.0

    def test_hidden_sizes_is_tuple(self) -> None:
        config = load_config(DEFAULT_YAML)
        assert isinstance(config.sac.hidden_sizes, tuple)
        assert all(isinstance(h, int) for h in config.sac.hidden_sizes)

    def test_empty_yaml_gives_all_defaults(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")
        config = load_config(path)
        assert config == Config()


class TestUnknownKeys:
    def test_unknown_top_level_key_rejected(self, tmp_path: Path) -> None:
        data = default_dict()
        data["dicount"] = 0.99  # typo'd key must not silently vanish
        with pytest.raises(ConfigError, match="dicount"):
            load_config(write_yaml(tmp_path, data))

    def test_unknown_nested_key_rejected_with_path(self, tmp_path: Path) -> None:
        data = default_dict()
        data["reward"]["w_x"] = 1.0
        with pytest.raises(ConfigError, match=r"reward.*w_x"):
            load_config(write_yaml(tmp_path, data))


class TestTypes:
    def test_string_for_float_rejected(self, tmp_path: Path) -> None:
        data = default_dict()
        data["sac"]["lr"] = "fast"
        with pytest.raises(ConfigError, match=r"sac\.lr"):
            load_config(write_yaml(tmp_path, data))

    def test_float_for_int_rejected(self, tmp_path: Path) -> None:
        data = default_dict()
        data["sac"]["batch_size"] = 256.5
        with pytest.raises(ConfigError, match=r"sac\.batch_size"):
            load_config(write_yaml(tmp_path, data))

    def test_int_accepted_for_float_field(self, tmp_path: Path) -> None:
        data = default_dict()
        data["sac"]["lr"] = 1  # YAML whole number
        config = load_config(write_yaml(tmp_path, data))
        assert config.sac.lr == 1.0
        assert isinstance(config.sac.lr, float)

    def test_bool_for_float_rejected(self, tmp_path: Path) -> None:
        data = default_dict()
        data["reward"]["w_v"] = True
        with pytest.raises(ConfigError, match=r"reward\.w_v"):
            load_config(write_yaml(tmp_path, data))

    def test_null_grad_clip_accepted(self, tmp_path: Path) -> None:
        data = default_dict()
        data["sac"]["grad_clip"] = None
        config = load_config(write_yaml(tmp_path, data))
        assert config.sac.grad_clip is None

    def test_scalar_where_mapping_expected_rejected(self, tmp_path: Path) -> None:
        data = default_dict()
        data["reward"] = 3
        with pytest.raises(ConfigError, match="reward"):
            load_config(write_yaml(tmp_path, data))


class TestRanges:
    @pytest.mark.parametrize("gamma", [0.0, 1.0, 1.5, -0.1])
    def test_gamma_out_of_range_rejected(self, tmp_path: Path, gamma: float) -> None:
        data = default_dict()
        data["sac"]["gamma"] = gamma
        with pytest.raises(ConfigError, match="gamma"):
            load_config(write_yaml(tmp_path, data))

    def test_negative_reward_weight_rejected(self, tmp_path: Path) -> None:
        data = default_dict()
        data["reward"]["w_s"] = -0.1
        with pytest.raises(ConfigError, match="w_s"):
            load_config(write_yaml(tmp_path, data))

    def test_invalid_port_rejected(self, tmp_path: Path) -> None:
        data = default_dict()
        data["io"]["telemetry_port"] = 70000
        with pytest.raises(ConfigError, match="telemetry_port"):
            load_config(write_yaml(tmp_path, data))

    def test_zero_tau_rejected(self, tmp_path: Path) -> None:
        data = default_dict()
        data["sac"]["tau"] = 0.0
        with pytest.raises(ConfigError, match="tau"):
            load_config(write_yaml(tmp_path, data))

    def test_zero_utd_rejected(self, tmp_path: Path) -> None:
        data = default_dict()
        data["trainer"]["utd_ratio"] = 0
        with pytest.raises(ConfigError, match="utd_ratio"):
            load_config(write_yaml(tmp_path, data))

    def test_beta_end_below_beta_start_rejected(self, tmp_path: Path) -> None:
        data = default_dict()
        data["per"]["beta_start"] = 0.8
        data["per"]["beta_end"] = 0.5
        with pytest.raises(ConfigError, match="beta_end"):
            load_config(write_yaml(tmp_path, data))


class TestImmutabilityAndSnapshot:
    def test_config_is_frozen(self) -> None:
        """Invariant 2: gamma cannot be mutated mid-run."""
        config = load_config(DEFAULT_YAML)
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.sac.gamma = 0.5  # type: ignore[misc]

    def test_config_to_dict_round_trip(self, tmp_path: Path) -> None:
        """The snapshot dict re-loads to an identical config (T0.5 dependency)."""
        config = load_config(DEFAULT_YAML)
        snapshot = config_to_dict(config)
        snapshot["sac"]["hidden_sizes"] = list(snapshot["sac"]["hidden_sizes"])
        reloaded = load_config(write_yaml(tmp_path, snapshot))
        assert reloaded == config
